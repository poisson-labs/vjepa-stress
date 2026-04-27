"""Sensor-domain perturbations: P1 Gaussian noise, P2 Motion blur, P5 Low-light.

Strength parameterizations are documented per class. Stage 0 will fill in the
forward implementations; the strength=0 contract is already enforced.
"""

from __future__ import annotations

import numpy as np

from src.perturbations.base import Perturbation


class GaussianNoise(Perturbation):
    """P1 — Additive Gaussian noise in pixel space.

    Strength maps linearly to noise standard deviation in 8-bit units; the
    exact mapping is fixed in Stage 0. Noise is drawn per (video, frame) via
    a deterministic ``numpy.random.Generator`` seeded from ``seed``.
    """

    name = "gaussian_noise"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        raise NotImplementedError("GaussianNoise: implement in Stage 0.")


class MotionBlur(Perturbation):
    """P2 — Linear motion blur with a directional kernel.

    Strength maps to kernel length in pixels (capped to a fraction of the
    short side). Kernel angle is drawn per video from ``seed``.
    """

    name = "motion_blur"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        raise NotImplementedError("MotionBlur: implement in Stage 0.")


class LowLight(Perturbation):
    """P5 — Low-light simulation: gamma + multiplicative gain + Poisson noise.

    Strength interpolates from clean (gain=1, gamma=1, no noise) to a fixed
    "dark frame" target. Mapping is fixed in Stage 0.
    """

    name = "low_light"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        raise NotImplementedError("LowLight: implement in Stage 0.")
