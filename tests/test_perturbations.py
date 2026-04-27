"""Contract tests for the perturbation API.

These tests run with no model weights and no GPU. They verify the contract
documented in ``src/perturbations/base.py``: ``strength=0`` is identity,
shape and dtype are preserved, and the calibration probes (Identity,
Reverse) behave as advertised at ``strength=1``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.perturbations import Identity, Reverse, all_perturbations


def _synthetic_clip(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("perturbation", all_perturbations(), ids=lambda p: p.name)
def test_strength_zero_is_identity(perturbation):
    clip = _synthetic_clip()
    out = perturbation.apply(clip, strength=0.0, seed=42)
    np.testing.assert_array_equal(out, clip)


@pytest.mark.parametrize("perturbation", all_perturbations(), ids=lambda p: p.name)
def test_shape_and_dtype_preserved(perturbation):
    clip = _synthetic_clip()
    out = perturbation.apply(clip, strength=0.0, seed=42)
    assert out.shape == clip.shape
    assert out.dtype == clip.dtype


def test_identity_is_identity_at_full_strength():
    clip = _synthetic_clip()
    out = Identity().apply(clip, strength=1.0, seed=42)
    np.testing.assert_array_equal(out, clip)


def test_reverse_at_full_strength_reverses_frames():
    clip = _synthetic_clip()
    out = Reverse().apply(clip, strength=1.0, seed=42)
    np.testing.assert_array_equal(out, clip[::-1])
    assert out.shape == clip.shape
    assert out.dtype == clip.dtype


@pytest.mark.parametrize("perturbation", all_perturbations(), ids=lambda p: p.name)
def test_strength_out_of_range_raises(perturbation):
    clip = _synthetic_clip()
    with pytest.raises(ValueError):
        perturbation.apply(clip, strength=-0.1, seed=0)
    with pytest.raises(ValueError):
        perturbation.apply(clip, strength=1.5, seed=0)
