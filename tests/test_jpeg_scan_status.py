"""Equivalence tests for _jpeg_scan_status's vectorised fast path.

_jpeg_scan_status resolves the common case with numpy and falls back to
_jpeg_scan_walk — the reference byte walk — for anything position-dependent.
These tests pin the two to identical verdicts, since the fast path is only
worth having if it can never disagree.
"""

import io

import numpy as np
import pytest
from PIL import Image

from orient_express.utils.image_processor import (
    _jpeg_scan_start,
    _jpeg_scan_status,
    _jpeg_scan_walk,
)


def reference(data: bytes):
    """The verdict the original implementation produced: walk, no fast path."""
    pos = _jpeg_scan_start(data)
    if pos is None:
        return None
    return _jpeg_scan_walk(
        data, pos + 2 + int.from_bytes(data[pos + 2 : pos + 4], "big")
    )


def jpeg_bytes(width=320, height=240, quality=90, progressive=False, seed=0):
    rng = np.random.default_rng(seed)
    # noise, not flat colour: a flat image compresses to an entropy stream with
    # almost no FF bytes, which would not exercise the scan
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(
        buf, format="JPEG", quality=quality, progressive=progressive
    )
    return buf.getvalue()


BASELINE = jpeg_bytes()
PROGRESSIVE = jpeg_bytes(progressive=True)


def splice(data: bytes, marker: bytes, at: float = 0.5) -> bytes:
    out = bytearray(data)
    i = int(len(out) * at)
    out[i : i + 2] = marker
    return bytes(out)


CASES = {
    "baseline": BASELINE,
    "progressive": PROGRESSIVE,
    "halved": BASELINE[: len(BASELINE) // 2],
    "decimated": BASELINE[: len(BASELINE) // 10],
    "eoi_stripped": BASELINE[:-2],
    "zero_padded": BASELINE + b"\x00" * 32,
    "trailing_junk": BASELINE + b"appended video bytes",
    "spliced_invalid": splice(BASELINE, b"\xff\x3f"),
    "spliced_segment": splice(BASELINE, b"\xff\xc4"),
    "spliced_fill": splice(BASELINE, b"\xff\xff"),
    "spliced_restart": splice(BASELINE, b"\xff\xd0"),
    "progressive_spliced": splice(PROGRESSIVE, b"\xff\x3f"),
    "empty": b"",
    "soi_only": b"\xff\xd8",
    "not_jpeg": b"this is not a jpeg at all",
    "sos_header_only": b"\xff\xd8\xff\xda\x00\x02",
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_reference_walk(name):
    data = CASES[name]
    assert _jpeg_scan_status(data) == reference(data)


def test_random_bytes_match_reference():
    """Fuzz: arbitrary bytes must never make the two implementations diverge."""
    rng = np.random.default_rng(1234)
    for _ in range(300):
        size = int(rng.integers(0, 4000))
        data = rng.integers(0, 256, size, dtype=np.uint8).tobytes()
        assert _jpeg_scan_status(data) == reference(data)
        # and prefixed with a real SOI/SOS header, so the walk actually starts
        hybrid = BASELINE[:200] + data
        assert _jpeg_scan_status(hybrid) == reference(hybrid)


def test_verdicts_are_actually_exercised():
    """Guard the tests above: they prove nothing if every case returns None."""
    verdicts = {reference(d) for d in CASES.values()}
    assert {"complete", "truncated", "corrupt"} <= verdicts
    assert reference(BASELINE) == "complete"
    assert reference(BASELINE[: len(BASELINE) // 2]) == "truncated"
    assert reference(splice(BASELINE, b"\xff\x3f")) == "corrupt"


def test_corrupt_stream_can_still_end_in_eoi():
    r"""Why the cheap tail check is not a substitute for the scan.

    Damage mid-stream leaves the tail intact, so `data.endswith(b'\\xff\\xd9')`
    reports a corrupt file as healthy — and cv2.imdecode returns pixels for it
    without complaint, which is the whole reason this scan exists. The tail
    check is only valid for "is the *header* intact" (e.g. trusting an EXIF
    read), never for "is the *stream* sound".
    """
    corrupt = splice(BASELINE, b"\xff\x3f")
    assert _jpeg_scan_status(corrupt) == "corrupt"
    assert corrupt.endswith(b"\xff\xd9")


def test_fast_path_resolves_healthy_images_without_the_walk(monkeypatch):
    """The speedup only exists if healthy files never reach the walk."""
    calls = []
    original = _jpeg_scan_walk

    def spy(data, pos):
        calls.append(pos)
        return original(data, pos)

    monkeypatch.setattr("orient_express.utils.image_processor._jpeg_scan_walk", spy)
    for seed in range(20):
        assert _jpeg_scan_status(jpeg_bytes(seed=seed)) == "complete"
    assert calls == []

    # a segment marker inside the scan IS position-dependent (its length field
    # makes the walk skip bytes), so it must fall back
    assert _jpeg_scan_status(splice(BASELINE, b"\xff\xc4")) == reference(
        splice(BASELINE, b"\xff\xc4")
    )
    assert calls
