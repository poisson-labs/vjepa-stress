"""Codec-domain perturbations: P9 JPEG compression.

JPEG is a tier-3 sanity sweep — we only run it on ViT-L. The point is to
confirm that codec artifacts behave like the other low-frequency stressors
and don't surface a separate failure mode.
"""

from __future__ import annotations

import numpy as np

from src.perturbations.base import Perturbation


class JPEGCompression(Perturbation):
    """P9 — Per-frame JPEG re-encode at a strength-controlled quality factor.

    Strength maps inversely to JPEG quality: ``strength=0`` keeps the input
    untouched, ``strength=1`` corresponds to the most aggressive quality
    factor we sweep (fixed in Stage 0). Encoding is deterministic, so
    ``seed`` is unused in practice but must be accepted for API uniformity.
    """

    name = "jpeg"
    tier = 3

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        raise NotImplementedError("JPEGCompression: implement in Stage 0.")
