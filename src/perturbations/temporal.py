"""Temporal perturbations: P3 Frame drops, P7 Reverse.

Frame drops perturb the time axis by removing frames and re-padding. Reverse
inverts the time axis as a calibration probe: a model that was sensitive to
direction should show a measurable M2 jump.
"""

from __future__ import annotations

import numpy as np

from src.perturbations.base import Perturbation


class FrameDrops(Perturbation):
    """P3 — Drop a fraction of frames and re-pad with the previous frame.

    Strength maps to drop probability in ``[0, drop_max]``; the exact bound
    is fixed in Stage 0. Drops are sampled per video from ``seed``.
    """

    name = "frame_drops"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        raise NotImplementedError("FrameDrops: implement in Stage 0.")


class Reverse(Perturbation):
    """P7 — Reverse the temporal axis.

    Calibration probe: only ``strength=1`` is meaningful (full reversal).
    ``strength=0`` is identity. Intermediate strengths are not defined for
    this perturbation, so the runner should only schedule it at the
    endpoints.
    """

    name = "reverse"
    tier = 2

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        if strength == 1.0:
            return frames[::-1].copy()
        raise ValueError(
            "Reverse only accepts strength in {0.0, 1.0}; got "
            f"{strength}. Use Identity (P6) at strength=0 if you want a "
            "no-op anchor."
        )
