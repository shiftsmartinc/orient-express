import base64
import ipaddress
import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from PIL import ExifTags, Image, ImageOps

from .gs import get_gcs_from_http_url, read_file_bytes

DOWNLOAD_TIMEOUT_SECONDS = 30


class UnsafeUrlError(ValueError):
    pass


def validate_url(http_url):
    # Reject URLs that could be used for SSRF: the request runs server-side
    # with the service account's network access, so only allow http(s) URLs
    # that resolve to public IPs (blocks the GCE metadata server, localhost,
    # and private/internal ranges).
    parsed = urlparse(http_url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"unsupported URL scheme: {parsed.scheme}")
    if not parsed.hostname:
        raise UnsafeUrlError("URL has no hostname")
    try:
        addr_infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as e:
        raise UnsafeUrlError(f"could not resolve host: {parsed.hostname}") from e
    for addr_info in addr_infos:
        ip = ipaddress.ip_address(addr_info[4][0])
        if not ip.is_global:
            raise UnsafeUrlError(
                f"URL host {parsed.hostname} resolves to non-public address {ip}"
            )


def read_image_from_url(http_url, http_as_gcs=False, session=None) -> Image.Image:
    # Extract GSC URI from http link and download the file directly.
    # It will increase reliability, as it will use GCP driver to fetch data
    # If URL is not GCS HTTP URL, download it through HTTP
    if http_as_gcs:
        gs_uri = get_gcs_from_http_url(http_url)
        if gs_uri:
            return read_image_from_gs(gs_uri)

    validate_url(http_url)
    client = session if session is not None else requests
    response = client.get(http_url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image


def read_image_from_gs(gs_url) -> Image.Image:
    bytes_content = read_file_bytes(gs_url)
    image = Image.open(BytesIO(bytes_content))
    return image


def clean_exif(image):
    if hasattr(image, "_getexif"):
        image.info.pop("exif", None)
        if hasattr(image, "_exif"):
            image._exif = None
    return image


def fix_rotation(image):
    """Return the image EXIF-upright, via the full 8-orientation transform.

    Matches decode_image's cv2 path (cv2 auto-applies EXIF orientation on
    decode), so the serving path and the streaming loaders agree on phone
    photos — including the mirrored orientations 2/4/5/7 that the old
    3/6/8-only rotation passed through untouched.
    """
    image = ImageOps.exif_transpose(image)
    return clean_exif(image)


def image_to_base64(image):
    bytes_content = image_to_bytes(image)
    return base64.b64encode(bytes_content).decode("utf-8")


def mask_to_base64(mask: np.ndarray) -> str:
    buffered = BytesIO()
    Image.fromarray(mask).save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def array_to_base64_npy(array: np.ndarray) -> str:
    """Encode an arbitrary-dtype array as base64 .npy bytes (lossless).

    Decode with np.load(BytesIO(base64.b64decode(data))).
    """
    buffered = BytesIO()
    np.save(buffered, array)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def image_to_bytes(image):
    buffered = BytesIO()
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return buffered.getvalue()


def base64_to_image(base64_data: str):
    return Image.open(BytesIO(base64.b64decode(base64_data)))


# JPEG-only DCT-domain reductions for decode_image's fast path, largest
# first so we pick the most aggressive factor that still covers the target
_JPEG_REDUCTIONS = (
    (8, cv2.IMREAD_REDUCED_COLOR_8),
    (4, cv2.IMREAD_REDUCED_COLOR_4),
    (2, cv2.IMREAD_REDUCED_COLOR_2),
)


def _jpeg_scan_start(data: bytes):
    """Offset of the first SOS marker, walking header segments from SOI.

    Returns None when data is not a JPEG or its header is malformed or cut
    off before the entropy-coded scan begins.
    """
    if data[:2] != b"\xff\xd8":
        return None
    pos = 2
    end = len(data)
    while pos + 4 <= end:
        if data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        if marker == 0xDA:  # SOS: scan data follows
            return pos
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2  # standalone markers carry no length field
            continue
        segment_length = int.from_bytes(data[pos + 2 : pos + 4], "big")
        if segment_length < 2:
            return None
        pos += 2 + segment_length
    return None


# markers that may legally appear inside/between scans (tables, next scan,
# comments, app segments); every one carries a two-byte length to skip
_SCAN_SEGMENT_MARKERS = frozenset(
    [0xC4, 0xCC, 0xDB, 0xDD, 0xDA, 0xDC, 0xFE, *range(0xE0, 0xF0)]
)


def _jpeg_scan_walk(data: bytes, pos: int):
    """The exact byte walk, from `pos` inside the entropy stream.

    Kept as the reference implementation and the fallback for anything the
    vectorised pass in _jpeg_scan_status cannot decide.
    """
    end = len(data)
    while True:
        idx = data.find(b"\xff", pos)
        if idx == -1 or idx + 1 >= end:
            return "truncated"  # stream ran out with no EOI
        nxt = data[idx + 1]
        if nxt == 0x00 or 0xD0 <= nxt <= 0xD7:
            pos = idx + 2
        elif nxt == 0xFF:  # fill byte before a marker
            pos = idx + 1
        elif nxt == 0xD9:
            return "complete"
        elif nxt in _SCAN_SEGMENT_MARKERS:
            if idx + 4 > end:
                return "truncated"
            segment_length = int.from_bytes(data[idx + 2 : idx + 4], "big")
            if segment_length < 2:
                return "corrupt"
            pos = idx + 2 + segment_length
        else:
            return "corrupt"


def _jpeg_scan_status(data: bytes):
    """Classify a JPEG's entropy stream: 'complete', 'truncated' or 'corrupt'.

    Returns None when the data is not a walkable JPEG (the decoders judge
    those).

    A structural byte walk, not a decode, so it costs a tiny fraction of
    one. Inside a scan, FF may only be followed by 00 (stuffed data byte),
    D0-D7 (restart), FF (fill), D9 (EOI), or a segment marker — anything
    else means the entropy data was damaged (e.g. an interrupted upload
    that spliced garbage into the stream). Walking from the first SOS
    makes the truncation verdict immune to EXIF thumbnails (their EOI
    sits before the scan) and to trailing non-JPEG bytes such as motion
    photos' appended video (they sit after the real EOI). Damage with no
    FF violations (e.g. zeroed spans) is invisible here — but libjpeg
    decodes that without complaint too, so no stream-level check catches
    it.

    The walk itself is one Python loop iteration per FF byte — roughly two
    thousand of them in a 500KB photo — and it holds the GIL throughout, so
    on a decode thread pool it serialises against every other thread
    (measured: 1.11x across 8 threads, i.e. no scaling at all). The common
    case never needs the loop: a healthy scan contains only FF00 (stuffed
    data) and FFD0-D7 (restart markers) before its FFD9, and the walk
    advances pos=idx+2 through every one of them, visiting exactly the FF
    positions numpy finds in a single C-level pass. Only a segment marker
    (whose length field makes the walk *skip* bytes, so later FFs may not be
    real scan positions) or an FF fill makes position history matter, and
    those fall back to _jpeg_scan_walk. Verdicts are identical by
    construction, and test_jpeg_scan_status.py pins that against the walk
    over mutated and fuzzed inputs.

    The scan still costs real time on a decode pool (it allocates a mask the
    size of the scan), so it is tempting to skip it for files whose last two
    bytes are FFD9. Do not: damage mid-stream leaves the tail intact, so 96
    of 100 deliberately corrupted files still end in EOI, and cv2.imdecode
    returns pixels for them without complaint — which is precisely what this
    check exists to catch. A tail check can only answer "is the *header*
    intact" (e.g. whether an EXIF read can be trusted), never "is the
    *stream* sound".
    """
    pos = _jpeg_scan_start(data)
    if pos is None:
        return None
    pos = pos + 2 + int.from_bytes(data[pos + 2 : pos + 4], "big")  # skip SOS hdr
    array = np.frombuffer(data, np.uint8)
    if pos >= len(array) - 1:
        return _jpeg_scan_walk(data, pos)
    # every FF that has a following byte; the walk can never look past these
    marks = np.flatnonzero(array[pos:-1] == 0xFF) + pos
    if marks.size == 0:
        return "truncated"  # no FF at all: no EOI either
    following = array[marks + 1]
    stuffed = (following == 0x00) | ((following >= 0xD0) & (following <= 0xD7))
    decisive = np.flatnonzero(~stuffed)
    if decisive.size == 0:
        return "truncated"  # only stuffed/restart bytes, stream ran out
    if following[decisive[0]] == 0xD9:
        return "complete"
    return _jpeg_scan_walk(data, pos)


def decode_image(
    data: bytes,
    *,
    fast_target: tuple[int, int] | None = None,
    return_original_size: bool = False,
):
    """Decode an encoded image (JPEG/PNG/WebP...) to a PIL RGB image, fast.

    cv2's decoder releases the GIL where PIL's holds it, so loader worker
    threads decode in parallel. EXIF orientation is
    applied — the image comes back upright, matching serving's
    fix_rotation. Pixels otherwise match PIL's for baseline JPEGs (both
    wrap libjpeg-turbo). Use inside an ImageLoader `load` callable when you
    fetch encoded bytes yourself:

        ImageLoader(rows, load=lambda r: decode_image(my_bytes(r)))

    fast_target=(width, height): opt-in reduced decode — JPEGs decode at
    the largest power-of-two reduction that still covers the target size.
    Faster, but pixels differ from a full decode (DCT-domain downscale);
    validate accuracy before enabling on a production model.

    return_original_size=True returns (image, (width, height)) instead,
    where the size is the full image's upright dimensions — equal to
    image.size except under a fast_target reduction.

    Incomplete JPEGs are processed, not lost, and warned about: a file
    cut off before its EOI marker decodes to the readable rows (missing
    region gray), and a file with corrupt entropy data mid-stream (the
    kind strict PIL refuses as "broken data stream") decodes via
    libjpeg's recovery, damaged areas coming out gray or flat. Detection
    is a structural marker scan (see _jpeg_scan_status), never a second
    decode. Data that yields no pixels at all still raises — the
    contract the loaders' per-item skip depends on.
    """
    if not data:
        raise OSError("empty image data (0 bytes)")
    flag = cv2.IMREAD_COLOR
    original_size = None
    if fast_target is not None or return_original_size:
        # PIL reads only the header here (no pixel decode); cv2 cannot
        # report dimensions without decoding, and blindly reducing an
        # already-small image would undershoot the target and upscale
        try:
            with Image.open(BytesIO(data)) as probe:
                width, height = probe.size
                probe_format = probe.format
                # cv2 applies EXIF orientation but the header size is
                # pre-rotation; swap for the transposed orientations
                if probe.getexif().get(ExifTags.Base.Orientation, 1) in (5, 6, 7, 8):
                    width, height = height, width
        except Exception:  # noqa: BLE001 - not PIL-readable; decoders below judge
            pass
        else:
            original_size = (width, height)
            if fast_target is not None and probe_format == "JPEG":
                for factor, reduced_flag in _JPEG_REDUCTIONS:
                    if (
                        width // factor >= fast_target[0]
                        and height // factor >= fast_target[1]
                    ):
                        flag = reduced_flag
                        break

    status = _jpeg_scan_status(data)
    if status == "truncated":
        logging.warning(
            f"truncated JPEG ({len(data)} bytes, no EOI marker in scan): "
            "decoding the readable part; the missing region is gray"
        )
    elif status == "corrupt":
        logging.warning(
            f"corrupt JPEG ({len(data)} bytes, invalid data mid-scan): "
            "decoding with recovered pixels; damaged regions may appear "
            "gray or flat"
        )
    if status in ("truncated", "corrupt") and not data.rstrip(b"\x00").endswith(
        b"\xff\xd9"
    ):
        # a synthetic EOI lets libjpeg-turbo decode the readable part onto
        # the full canvas — without it cv2 rejects the whole file
        data = data + b"\xff\xd9"

    array = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    if array is None:
        # cv2 returns None (never raises) both for corrupt data and for
        # valid formats it can't handle (e.g. CMYK JPEG). PIL is the
        # robust fallback: it decodes the exotic formats and RAISES on
        # true corruption.
        image = Image.open(BytesIO(data))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        return (image, image.size) if return_original_size else image

    image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    if return_original_size:
        return image, (original_size or image.size)
    return image


def image_to_array(image: Image.Image):
    """PIL -> HxWx3 uint8 RGB array with a single full-resolution copy.

    The obvious np.array(image.convert("RGB")) materializes three
    full-resolution temporaries (convert copy + tobytes + array copy), which
    is page-fault-bound on large photos (~40 ms at 4K vs ~4 ms here). The
    returned array is read-only; callers that need to write should copy.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.frombuffer(image.tobytes(), dtype=np.uint8).reshape(
        image.height, image.width, 3
    )


# Threading pays off only when each per-mask resize is heavy enough to
# amortize task dispatch AND the whole job is heavy enough to amortize pool
# setup; small outputs are always faster serial, even for hundreds of masks
# (both thresholds calibrated empirically).
THREADED_RESIZE_MIN_PIXELS_PER_MASK = 100_000
THREADED_RESIZE_MIN_TOTAL_PIXELS = 8_000_000


def resize_masks(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinearly resize a stack of (N, h, w) float masks to (N, height, width).

    Matches torch's F.interpolate(mode="bilinear", align_corners=False) /
    OpenCV's half-pixel-center convention (verified equal to within float32
    precision). cv2.resize releases the GIL, so heavy jobs resize on a
    thread pool; output is identical either way.
    """
    n = masks.shape[0]
    resized = np.empty((n, height, width), dtype=np.float32)

    def resize_one(i):
        resized[i] = cv2.resize(
            masks[i].astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    pixels_per_mask = height * width
    use_threads = (
        pixels_per_mask >= THREADED_RESIZE_MIN_PIXELS_PER_MASK
        and n * pixels_per_mask >= THREADED_RESIZE_MIN_TOTAL_PIXELS
    )
    if use_threads:
        workers = min(8, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(resize_one, range(n)))
    else:
        for i in range(n):
            resize_one(i)
    return resized


def pil_to_opencv(image: Image.Image):
    return cv2.cvtColor(image_to_array(image), cv2.COLOR_RGB2BGR)


def opencv_to_pil(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image)
