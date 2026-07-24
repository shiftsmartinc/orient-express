"""Tests for decode_image: cv2 fast path, PIL fallback, fault semantics."""

import io

import numpy as np
import pytest
from PIL import Image

from orient_express.utils.image_processor import decode_image


def jpeg_bytes(width=800, height=600, mode="RGB", quality=90):
    rng = np.random.default_rng(0)
    if mode == "L":
        arr = rng.integers(0, 255, (height, width), dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L" if mode == "L" else "RGB")
    if mode == "CMYK":
        img = img.convert("CMYK")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
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


def test_truncated_jpeg_raises():
    # cv2 silently returns a half-gray partial image for truncated JPEGs;
    # the loaders' fault tolerance depends on a raise instead
    data = jpeg_bytes()
    with pytest.raises(OSError, match="truncated"):
        decode_image(data[: len(data) // 2])


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
