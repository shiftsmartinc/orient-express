"""Pins resize_masks bilinear behavior (verified float32-equal to torch's F.interpolate(mode="bilinear", align_corners=False) during the torch removal; see that PR for the comparison harness)."""

import numpy as np

from orient_express.utils.image_processor import resize_masks


def test_shape_and_dtype():
    masks = np.random.default_rng(0).uniform(-5, 5, (7, 64, 48)).astype(np.float32)
    out = resize_masks(masks, 480, 640)
    assert out.shape == (7, 480, 640)
    assert out.dtype == np.float32


def test_constant_field_is_exact():
    masks = np.full((5, 32, 32), 3.5, dtype=np.float32)
    out = resize_masks(masks, 100, 200)
    np.testing.assert_allclose(out, 3.5, rtol=0, atol=1e-6)


def test_same_size_is_identity():
    masks = np.random.default_rng(1).uniform(-5, 5, (3, 50, 50)).astype(np.float32)
    np.testing.assert_array_equal(resize_masks(masks, 50, 50), masks)


def test_ramp_preserves_range_and_corners():
    ramp = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    out = resize_masks(ramp, 8, 8)
    assert out.min() >= 0.0 and out.max() <= 3.0
    # half-pixel-center convention: exact source values at the corners
    assert abs(out[0, 0, 0] - 0.0) < 1e-6
    assert abs(out[0, -1, -1] - 3.0) < 1e-6


def test_threaded_and_serial_paths_agree():
    from orient_express.utils.image_processor import (
        THREADED_RESIZE_MIN_PIXELS_PER_MASK,
        THREADED_RESIZE_MIN_TOTAL_PIXELS,
    )

    rng = np.random.default_rng(2)
    # 10 masks -> 1000x1000 crosses both thresholds, so the full call threads
    h = w = 1000
    n = 10
    assert h * w >= THREADED_RESIZE_MIN_PIXELS_PER_MASK
    assert n * h * w >= THREADED_RESIZE_MIN_TOTAL_PIXELS
    masks = rng.uniform(-5, 5, (n, 40, 40)).astype(np.float32)
    full = resize_masks(masks, h, w)  # threaded path
    per_mask = np.stack(
        [resize_masks(masks[i : i + 1], h, w)[0] for i in range(n)]
    )  # serial path (single mask is below the total-work threshold)
    np.testing.assert_array_equal(full, per_mask)
