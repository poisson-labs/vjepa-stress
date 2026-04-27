"""Spatial perturbations: P4 Occlusion, P8 Horizontal flip."""

from __future__ import annotations

import numpy as np

from src.perturbations.base import Perturbation


class Occlusion(Perturbation):
    """P4 — Random rectangular occlusion (cutout) in pixel space.

    Strength maps to the side length of the occluding rectangle as a fraction
    of the short side. Position is drawn per video from ``seed`` and held
    constant across frames so the occlusion is static.
    """

    name = "occlusion"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        raise NotImplementedError("Occlusion: implement in Stage 0.")


class HorizontalFlip(Perturbation):
    """P8 — Horizontal flip of every frame (calibration probe).

    Only ``strength=1`` is meaningful (full flip). ``strength=0`` is identity.
    """

    name = "horizontal_flip"
    tier = 2

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        if strength == 1.0:
            return np.ascontiguousarray(frames[:, :, ::-1, :])
        raise ValueError(
            "HorizontalFlip only accepts strength in {0.0, 1.0}; got "
            f"{strength}."
        )
