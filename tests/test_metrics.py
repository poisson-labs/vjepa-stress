"""Sanity tests for the M1 / M2 / M3 metric contracts.

- M1(features, features) == 0
- M2(features, features) == 0
- M2(features, reverse(features)) > 0
- M3 with full-image masks tracks perfectly (= 1)
- M3 with no instances returns NaN
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from vjepa_stress.metrics import compute_m1, compute_m2, compute_m3
from vjepa_stress.metrics.m3_correspondence import (
    _tubelet_pool_masks,
    compute_m3_for_clip,
)


def _features(t: int = 8, p: int = 16, d: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    f = torch.randn(t, p, d, generator=g)
    return torch.nn.functional.normalize(f, dim=-1)


def test_m1_zero_on_identical_features():
    f = _features()
    out = compute_m1(f, f)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_m1_positive_on_orthogonal_features():
    f = _features()
    g = torch.Generator().manual_seed(99)
    f_other = torch.nn.functional.normalize(
        torch.randn(*f.shape, generator=g), dim=-1
    )
    out = compute_m1(f, f_other)
    assert out.item() > 0.0


def test_m1_shape_mismatch_raises():
    f = _features()
    g = _features(t=8, p=8, d=32)
    try:
        compute_m1(f, g)
    except ValueError:
        return
    raise AssertionError("expected ValueError on shape mismatch")


def test_m2_zero_on_identical_features():
    f = _features()
    out = compute_m2(f, f)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_m2_positive_on_reversed_features():
    """Algorithmic-correctness check on synthetic random features.

    This asserts only that ``M2(features, reversed_features) > 0`` — i.e.
    the gradient-based form is *not* invariant under temporal reversal,
    which a naive ``mean cos(F_t, F_{t+1})`` formulation would be (see
    ``m2_temporal.py`` docstring).

    It does **not** validate the numerical calibration of M2 on real
    V-JEPA features. Random Gaussian features will produce a different
    M2 magnitude (~1.07) than real V-JEPA features (~1.14 per
    RESEARCH.md §3.2). The real-feature calibration check lives in
    :func:`test_vjepa_loader.test_m2_reverse_calibration_real_features`
    and is gated behind ``@pytest.mark.slow``.
    """
    f = _features()
    f_rev = torch.flip(f, dims=[0])
    out = compute_m2(f, f_rev)
    assert out.item() > 0.0


def test_m2_requires_two_frames():
    f = _features(t=1)
    try:
        compute_m2(f, f)
    except ValueError:
        return
    raise AssertionError("expected ValueError on T<2")


def test_m3_perfect_tracking_with_full_mask():
    t, h_t, w_t, d = 4, 14, 14, 16
    h_img, w_img = 224, 224
    g = torch.Generator().manual_seed(0)
    f = torch.nn.functional.normalize(
        torch.randn(t, h_t, w_t, d, generator=g), dim=-1
    )
    masks = torch.ones(t, h_img, w_img, dtype=torch.long)
    out = compute_m3(f, masks, keypoints_per_mask=4, search_window=7, seed=0)
    assert out.item() == 1.0


def test_m3_returns_nan_when_no_instances():
    t, h_t, w_t, d = 4, 14, 14, 16
    h_img, w_img = 224, 224
    g = torch.Generator().manual_seed(0)
    f = torch.nn.functional.normalize(
        torch.randn(t, h_t, w_t, d, generator=g), dim=-1
    )
    masks = torch.zeros(t, h_img, w_img, dtype=torch.long)
    out = compute_m3(f, masks, keypoints_per_mask=4)
    assert torch.isnan(out)


# ---------------------------------------------------------------------------
# M3 adapter — compute_m3_for_clip
# ---------------------------------------------------------------------------


def test_tubelet_pool_masks_size_one_is_identity():
    masks = np.random.default_rng(0).integers(0, 3, size=(8, 16, 16), dtype=np.int32)
    pooled = _tubelet_pool_masks(masks, 1)
    assert pooled.shape == (8, 16, 16)
    assert pooled.dtype == torch.int32
    assert torch.equal(pooled, torch.from_numpy(masks))


def test_tubelet_pool_masks_takes_per_pixel_max():
    """Per-pixel max means any-frame foreground stays foreground in the pool."""
    masks = np.zeros((4, 4, 4), dtype=np.int32)
    masks[0, 0, 0] = 5
    masks[1, 0, 0] = 0
    masks[2, 1, 1] = 7
    masks[3, 1, 1] = 0
    pooled = _tubelet_pool_masks(masks, 2)
    assert pooled.shape == (2, 4, 4)
    assert pooled[0, 0, 0].item() == 5
    assert pooled[1, 1, 1].item() == 7


def test_tubelet_pool_masks_rejects_wrong_T():
    masks = np.zeros((5, 4, 4), dtype=np.int32)
    with pytest.raises(ValueError):
        _tubelet_pool_masks(masks, 2)


def test_compute_m3_for_clip_reshapes_token_grid_correctly():
    """Adapter feeds (T_tubelet, P, D) features and pools masks to T_tubelet."""
    t_tubelet, side, d = 4, 6, 16
    p = side * side
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(t_tubelet, p, d, generator=g)  # not pre-normalized
    # 8 source frames -> tubelet_size=2 -> 4 pooled time steps
    h_img = w_img = 96
    masks = np.zeros((8, h_img, w_img), dtype=np.int32)
    masks[:, 30:60, 30:60] = 1  # full instance present in every frame
    out = compute_m3_for_clip(
        features=feats,
        masks=masks,
        tubelet_size=2,
        keypoints_per_mask=4,
        seed=0,
    )
    assert torch.is_tensor(out)
    assert math.isfinite(out.item())
    assert 0.0 <= out.item() <= 1.0


def test_compute_m3_for_clip_keypoint_seed_pins_clean_perturbed_match():
    """Same seed + same features → identical M3 (the H5 invariant)."""
    t_tubelet, side, d = 4, 6, 16
    p = side * side
    g = torch.Generator().manual_seed(7)
    feats = torch.randn(t_tubelet, p, d, generator=g)
    masks = np.zeros((8, 96, 96), dtype=np.int32)
    masks[:, 20:80, 20:80] = 1
    out_a = compute_m3_for_clip(
        features=feats, masks=masks, tubelet_size=2, seed=42
    )
    out_b = compute_m3_for_clip(
        features=feats, masks=masks, tubelet_size=2, seed=42
    )
    assert out_a.item() == out_b.item()


def test_compute_m3_for_clip_rejects_non_square_grid():
    feats = torch.randn(4, 17, 16)  # 17 patches: not a perfect square
    masks = np.zeros((4, 32, 32), dtype=np.int32)
    masks[:, 5:25, 5:25] = 1
    with pytest.raises(ValueError):
        compute_m3_for_clip(features=feats, masks=masks, tubelet_size=1)
