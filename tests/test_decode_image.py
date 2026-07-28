"""Tests for decode_image: cv2 fast path, PIL fallback, fault semantics."""

import io
import logging

import numpy as np
import pytest
from PIL import Image

from orient_express.utils.image_processor import decode_image


def jpeg_bytes(width=800, height=600, mode="RGB", quality=90, orientation=None):
    rng = np.random.default_rng(0)
    if mode == "L":
        arr = rng.integers(0, 255, (height, width), dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L" if mode == "L" else "RGB")
    if mode == "CMYK":
        img = img.convert("CMYK")
    buf = io.BytesIO()
    save_kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation  # EXIF Orientation tag
        save_kwargs["exif"] = exif
    img.save(buf, format="JPEG", quality=quality, **save_kwargs)
    return buf.getvalue()


def png_bytes(width=320, height=240):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_jpeg_pixels_match_pil():
    # both decoders wrap libjpeg-turbo: the fast path must be
    # pixel-identical to PIL for baseline JPEGs (golden-safety)
    data = jpeg_bytes()
    via_pil = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    via_fast = np.asarray(decode_image(data))
    np.testing.assert_array_equal(via_pil, via_fast)


def test_png_pixels_match_pil():
    data = png_bytes()
    via_pil = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    via_fast = np.asarray(decode_image(data))
    np.testing.assert_array_equal(via_pil, via_fast)


def test_grayscale_jpeg_decodes_to_rgb():
    image = decode_image(jpeg_bytes(mode="L"))
    assert image.mode == "RGB"
    assert image.size == (800, 600)


def test_cmyk_jpeg_falls_back_to_pil():
    # cv2.imdecode returns None for CMYK JPEGs; the PIL fallback must
    # still produce an RGB image
    image = decode_image(jpeg_bytes(mode="CMYK"))
    assert image.mode == "RGB"
    assert image.size == (800, 600)


def test_corrupt_data_raises():
    # via the PIL fallback (cv2 returns None), which raises on garbage
    with pytest.raises((OSError, ValueError)):
        decode_image(b"not an image at all")


def test_truncated_jpeg_decodes_partially_with_warning(caplog):
    # an interrupted upload is processed (readable rows + gray fill), not
    # lost — but the truncation is loudly logged
    data = jpeg_bytes()
    with caplog.at_level(logging.WARNING):
        image = decode_image(data[: len(data) // 2])
    assert image.size == (800, 600)  # full canvas
    assert any("truncated" in r.message for r in caplog.records)
    # the readable top of the image matches the full decode
    full = np.asarray(decode_image(data))
    np.testing.assert_array_equal(np.asarray(image)[:100], full[:100])


def test_jpeg_with_trailing_data_decodes_cleanly(caplog):
    # motion photos are a complete JPEG with a video appended after the
    # EOI marker; they must decode identically and without a warning
    data = jpeg_bytes()
    tail = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 8
    with caplog.at_level(logging.WARNING):
        image = decode_image(data + tail)
    assert not caplog.records  # neither truncated nor corrupt
    np.testing.assert_array_equal(np.asarray(image), np.asarray(decode_image(data)))


def test_nul_padded_jpeg_decodes_cleanly(caplog):
    data = jpeg_bytes()
    with caplog.at_level(logging.WARNING):
        image = decode_image(data + b"\x00" * 1024)
    assert not caplog.records
    assert image.size == (800, 600)


def test_progressive_jpeg_decodes_cleanly(caplog):
    # multi-scan files exercise the marker walk's segment skipping
    img = Image.fromarray(
        np.random.default_rng(3).integers(0, 255, (600, 800, 3), dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90, progressive=True)
    with caplog.at_level(logging.WARNING):
        image = decode_image(buf.getvalue())
    assert not caplog.records
    assert image.size == (800, 600)


def test_exif_orientation_applied():
    # orientation 6 = rotate 90 CW to display: a 200x100 file is a 100x200
    # upright image; cv2 applies this on decode
    image = decode_image(jpeg_bytes(width=200, height=100, orientation=6))
    assert image.size == (100, 200)


def test_exif_orientation_applied_in_pil_fallback():
    # CMYK forces the PIL fallback path, which must transpose the same way
    image = decode_image(jpeg_bytes(width=200, height=100, mode="CMYK", orientation=6))
    assert image.size == (100, 200)


def test_return_original_size_exact():
    image, original = decode_image(jpeg_bytes(), return_original_size=True)
    assert original == (800, 600)
    assert image.size == (800, 600)


def test_return_original_size_under_fast_reduction():
    data = jpeg_bytes(width=2400, height=1800)
    image, original = decode_image(
        data, fast_target=(576, 576), return_original_size=True
    )
    assert image.size == (1200, 900)  # factor-2 reduction
    assert original == (2400, 1800)  # reported in original-photo pixels


def test_return_original_size_is_upright_under_fast_reduction():
    # orientation 6 swaps the header dimensions; the reported original size
    # must be the upright (post-rotation) one, matching the decoded image
    data = jpeg_bytes(width=2400, height=1800, orientation=6)
    image, original = decode_image(
        data, fast_target=(576, 576), return_original_size=True
    )
    assert image.size == (900, 1200)  # upright, factor-2 reduced
    assert original == (1800, 2400)  # upright, full resolution


def test_fast_target_reduces_large_jpegs():
    data = jpeg_bytes(width=2400, height=1800)
    image = decode_image(data, fast_target=(576, 576))
    # largest power-of-two reduction still covering 576: factor 2 gives
    # 1200x900 (factor 4 would give 600x450 — height under 576)
    assert image.size == (1200, 900)


def test_fast_target_skips_reduction_when_too_small():
    data = jpeg_bytes(width=800, height=600)
    image = decode_image(data, fast_target=(576, 576))
    assert image.size == (800, 600)  # any reduction would undershoot


def test_fast_target_ignored_for_png():
    data = png_bytes()
    image = decode_image(data, fast_target=(100, 100))
    assert image.size == (320, 240)  # DCT reduction is JPEG-only


def test_empty_bytes_raise_cleanly():
    with pytest.raises(OSError, match="empty image data"):
        decode_image(b"")


def test_corrupt_midstream_jpeg_warns_and_decodes(caplog):
    # corrupt entropy data mid-stream but EOI-terminated (the kind strict
    # PIL refuses as "broken data stream"): the marker scan must flag it
    # (invalid FF pairs in the garbage) and cv2's recovery must still
    # produce an image rather than lose the item
    data = jpeg_bytes()
    garbage = np.random.default_rng(5).integers(0, 256, 2000, dtype=np.uint8)
    mid = len(data) // 2
    with caplog.at_level(logging.WARNING):
        image = decode_image(data[:mid] + garbage.tobytes() + b"\xff\xd9")
    assert image.size == (800, 600)
    assert any("corrupt" in r.message for r in caplog.records)


def test_corrupt_jpeg_without_eoi_still_decodes(caplog):
    # corruption AND no EOI at all: flagged, and the synthetic EOI keeps
    # the file decodable
    data = jpeg_bytes()
    garbage = np.random.default_rng(6).integers(0, 256, 2000, dtype=np.uint8)
    with caplog.at_level(logging.WARNING):
        image = decode_image(data[: len(data) // 2] + garbage.tobytes())
    assert image.size == (800, 600)
    assert any(
        "corrupt" in r.message or "truncated" in r.message for r in caplog.records
    )
