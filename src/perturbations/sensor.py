"""Sensor-domain perturbations: P1 Gaussian noise, P2 Motion blur, P5 Low-light.

Strength parameterizations are documented per class. ``strength=0`` is identity
for all three; ``strength=1`` saturates the most extreme variant we sweep.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.perturbations.base import Perturbation


# Strength=1 endpoints. Hand-picked to span "still recoverable" (s ~ 0.3)
# through "obviously degraded" (s = 1.0); refined empirically in Stage 1.
_GAUSSIAN_SIGMA_MAX: float = 50.0       # 8-bit units of additive noise
_MOTION_BLUR_LEN_MAX: int = 25          # kernel side length, in pixels
_LOW_LIGHT_GAIN_MIN: float = 0.15       # multiplicative gain at s=1
_LOW_LIGHT_GAMMA_MAX: float = 2.5       # gamma exponent at s=1
_LOW_LIGHT_NOISE_SIGMA_MAX: float = 0.02  # additive noise on [0,1]-scaled image


class GaussianNoise(Perturbation):
    """P1 — Additive Gaussian noise in pixel space.

    ``sigma = strength * 50`` in 8-bit units. Per-frame noise is drawn from a
    deterministic ``numpy.random.Generator`` seeded from ``seed``, so the same
    video gets the same realization at the same strength across models.
    """

    name = "gaussian_noise"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        rng = np.random.default_rng(seed)
        sigma = float(strength) * _GAUSSIAN_SIGMA_MAX
        noise = rng.normal(0.0, sigma, frames.shape).astype(np.float32)
        out = frames.astype(np.float32) + noise
        return np.clip(out, 0.0, 255.0).astype(np.uint8)


class MotionBlur(Perturbation):
    """P2 — Linear motion blur with a directional kernel.

    Strength maps to kernel length in pixels (capped at ``_MOTION_BLUR_LEN_MAX``).
    Kernel angle is drawn once per video from ``seed`` and held constant across
    frames so the blur direction is coherent through the clip.
    """

    name = "motion_blur"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        rng = np.random.default_rng(seed)
        size = max(3, int(round(float(strength) * _MOTION_BLUR_LEN_MAX)))
        if size % 2 == 0:
            size += 1
        angle_rad = float(rng.uniform(0.0, np.pi))
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        dx, dy = np.cos(angle_rad), np.sin(angle_rad)
        for i in range(size):
            offset = i - center
            x = int(round(center + offset * dx))
            y = int(round(center + offset * dy))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        kernel /= kernel.sum()
        out = np.empty_like(frames)
        for t in range(frames.shape[0]):
            out[t] = cv2.filter2D(frames[t], -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        return out


class LowLight(Perturbation):
    """P5 — Low-light simulation: gamma + multiplicative gain + Gaussian noise.

    At ``strength=1`` we apply ``gamma=2.5`` darkening, multiply by gain
    ``0.15``, and add Gaussian noise with ``sigma=0.02`` (on the ``[0, 1]``
    image scale). Mapping is linear in ``strength`` between identity and
    these endpoints.
    """

    name = "low_light"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        rng = np.random.default_rng(seed)
        s = float(strength)
        gain = 1.0 + s * (_LOW_LIGHT_GAIN_MIN - 1.0)
        gamma = 1.0 + s * (_LOW_LIGHT_GAMMA_MAX - 1.0)
        noise_sigma = s * _LOW_LIGHT_NOISE_SIGMA_MAX
        f01 = frames.astype(np.float32) / 255.0
        dark = np.power(f01, gamma) * gain
        if noise_sigma > 0.0:
            dark = dark + rng.normal(0.0, noise_sigma, dark.shape).astype(np.float32)
        return np.clip(dark * 255.0, 0.0, 255.0).astype(np.uint8)
