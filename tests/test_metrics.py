"""Sanity tests for the M1 / M2 metric contracts.

These tests are *expected to fail* until Stage 0 implements the metric
bodies; the function signatures and the calibration anchors documented in
RESEARCH.md \u00a73 are the part that's load-bearing today.

- M1(features, features) == 0
- M2(features, features) == 0
- M2(features, reverse(features)) > 0
"""

from __future__ import annotations

import torch

from src.metrics import compute_m1, compute_m2


def _features(t: int = 8, p: int = 16, d: int = 32) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    f = torch.randn(t, p, d, generator=g)
    return torch.nn.functional.normalize(f, dim=-1)


def test_m1_zero_on_identical_features():
    f = _features()
    out = compute_m1(f, f)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_m2_zero_on_identical_features():
    f = _features()
    out = compute_m2(f, f)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_m2_positive_on_reversed_features():
    f = _features()
    f_rev = torch.flip(f, dims=[0])
    out = compute_m2(f, f_rev)
    assert out.item() > 0.0
