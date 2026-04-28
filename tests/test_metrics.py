"""Sanity tests for the M1 / M2 / M3 metric contracts.

- M1(features, features) == 0
- M2(features, features) == 0
- M2(features, reverse(features)) > 0
- M3 with full-image masks tracks perfectly (= 1)
- M3 with no instances returns NaN
"""

from __future__ import annotations

import torch

from vjepa_stress.metrics import compute_m1, compute_m2, compute_m3


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
