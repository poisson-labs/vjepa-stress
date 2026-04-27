"""Contract tests for the perturbation API.

These tests run with no model weights and no GPU. They verify the contract
documented in ``src/perturbations/base.py``: ``strength=0`` is identity,
shape and dtype are preserved at every strength, and the calibration
probes (Identity, Reverse, HorizontalFlip) behave as advertised.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.perturbations import (
    FrameDrops,
    GaussianNoise,
    HorizontalFlip,
    Identity,
    JPEGCompression,
    LowLight,
    MotionBlur,
    Occlusion,
    Reverse,
    all_perturbations,
)


def _synthetic_clip(num_frames: int = 8, size: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(num_frames, size, size, 3), dtype=np.uint8)


# Perturbations whose ``apply`` is defined at strength=0.5. Reverse and
# HorizontalFlip raise on intermediate strengths by design.
_CONTINUOUS = [
    GaussianNoise(),
    MotionBlur(),
    FrameDrops(),
    Occlusion(),
    LowLight(),
    Identity(),
    JPEGCompression(),
]


@pytest.mark.parametrize("perturbation", all_perturbations(), ids=lambda p: p.name)
def test_strength_zero_is_identity(perturbation):
    clip = _synthetic_clip()
    out = perturbation.apply(clip, strength=0.0, seed=42)
    np.testing.assert_array_equal(out, clip)


@pytest.mark.parametrize("perturbation", all_perturbations(), ids=lambda p: p.name)
def test_shape_and_dtype_preserved_at_zero(perturbation):
    clip = _synthetic_clip()
    out = perturbation.apply(clip, strength=0.0, seed=42)
    assert out.shape == clip.shape
    assert out.dtype == clip.dtype


@pytest.mark.parametrize("perturbation", all_perturbations(), ids=lambda p: p.name)
def test_shape_and_dtype_preserved_at_full_strength(perturbation):
    clip = _synthetic_clip()
    out = perturbation.apply(clip, strength=1.0, seed=42)
    assert out.shape == clip.shape
    assert out.dtype == clip.dtype


@pytest.mark.parametrize("perturbation", _CONTINUOUS, ids=lambda p: p.name)
def test_shape_and_dtype_preserved_at_mid_strength(perturbation):
    clip = _synthetic_clip()
    out = perturbation.apply(clip, strength=0.5, seed=42)
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


def test_horizontal_flip_at_full_strength_flips_width():
    clip = _synthetic_clip()
    out = HorizontalFlip().apply(clip, strength=1.0, seed=42)
    np.testing.assert_array_equal(out, clip[:, :, ::-1, :])


@pytest.mark.parametrize("perturbation", all_perturbations(), ids=lambda p: p.name)
def test_strength_out_of_range_raises(perturbation):
    clip = _synthetic_clip()
    with pytest.raises(ValueError):
        perturbation.apply(clip, strength=-0.1, seed=0)
    with pytest.raises(ValueError):
        perturbation.apply(clip, strength=1.5, seed=0)


# Perturbations that should produce different pixels at strength=1.
# Identity is excluded by design. FrameDrops uses a longer clip so the
# drop-probability schedule reliably fires.
@pytest.mark.parametrize(
    "perturbation",
    [
        GaussianNoise(),
        MotionBlur(),
        Occlusion(),
        LowLight(),
        Reverse(),
        HorizontalFlip(),
        JPEGCompression(),
    ],
    ids=lambda p: p.name,
)
def test_full_strength_changes_frames(perturbation):
    clip = _synthetic_clip()
    out = perturbation.apply(clip, strength=1.0, seed=42)
    assert not np.array_equal(out, clip), (
        f"{perturbation.name} at strength=1 left frames unchanged"
    )


def test_frame_drops_full_strength_changes_frames():
    clip = _synthetic_clip(num_frames=16)
    out = FrameDrops().apply(clip, strength=1.0, seed=42)
    assert not np.array_equal(out, clip)
    assert out.shape == clip.shape


def test_reverse_intermediate_strength_raises():
    clip = _synthetic_clip()
    with pytest.raises(ValueError):
        Reverse().apply(clip, strength=0.5, seed=0)


def test_horizontal_flip_intermediate_strength_raises():
    clip = _synthetic_clip()
    with pytest.raises(ValueError):
        HorizontalFlip().apply(clip, strength=0.5, seed=0)
