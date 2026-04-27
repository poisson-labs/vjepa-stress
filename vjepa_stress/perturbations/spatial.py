"""Spatial perturbations: P4 Occlusion, P8 Horizontal flip."""

from __future__ import annotations

import numpy as np

from vjepa_stress.perturbations.base import Perturbation


# Side length of the occluding square at strength=1, as a fraction of the
# short side. 0.5 covers a quarter of the frame area at full strength.
_OCCLUSION_MAX_FRAC: float = 0.5
_OCCLUSION_FILL: int = 128  # neutral gray


class Occlusion(Perturbation):
    """P4 — Random rectangular occlusion (cutout) in pixel space.

    Square side length is ``strength * 0.5 * min(H, W)``. Position is drawn
    once per video from ``seed`` and held constant across frames so the
    occlusion is static — this isolates the spatial-occluder failure mode
    from any motion-handling failure.
    """

    name = "occlusion"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        rng = np.random.default_rng(seed)
        _, h, w, _ = frames.shape
        side = max(1, int(round(float(strength) * _OCCLUSION_MAX_FRAC * min(h, w))))
        side = min(side, h, w)
        y0 = int(rng.integers(0, h - side + 1))
        x0 = int(rng.integers(0, w - side + 1))
        out = frames.copy()
        out[:, y0 : y0 + side, x0 : x0 + side, :] = _OCCLUSION_FILL
        return out


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
