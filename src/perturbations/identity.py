"""P6 — Identity perturbation (calibration probe).

Returns the input unchanged at every strength. Acts as a calibration anchor:
M1 should be exactly zero, M2 should be exactly zero, M3 should match the
"clean" baseline. Any drift on Identity is a bug in the harness, not in the
encoder.
"""

from __future__ import annotations

import numpy as np

from src.perturbations.base import Perturbation


class Identity(Perturbation):
    name = "identity"
    tier = 2

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        return frames.copy()
