"""Abstract base class for all perturbations.

The contract is intentionally narrow:

- Inputs and outputs are RGB ``uint8`` arrays of shape ``(T, H, W, 3)`` in
  ``[0, 255]``. Stay in image space; no model-side normalization here.
- ``strength`` is a float in ``[0, 1]``, where ``0`` is identity and ``1`` is
  the most extreme variant we sweep. The mapping from this scalar to the
  concrete physical parameter (sigma, kernel size, alpha, ...) is the
  perturbation's own responsibility and is documented in its docstring.
- ``seed`` is per-video so that the same video gets the same realization at
  the same strength across models.

The strength=0 invariant is load-bearing: it lets us treat "any perturbation
at strength 0" and "Identity at any strength" as exchangeable, which is what
makes the calibration probes interpretable. ``tests/test_perturbations.py``
enforces it for every concrete subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Perturbation(ABC):
    """Abstract base for all perturbations."""

    #: Stable identifier, snake_case (e.g. ``"gaussian_noise"``).
    name: str

    #: Experiment tier this perturbation belongs to (1, 2, or 3 — see
    #: ``vjepa_stress.sweep.matrix``).
    tier: int

    @abstractmethod
    def apply(
        self,
        frames: np.ndarray,
        strength: float,
        seed: int,
    ) -> np.ndarray:
        """Apply the perturbation to a clip.

        Args:
            frames: Array of shape ``(T, H, W, 3)``, dtype ``uint8``, RGB in
                ``[0, 255]``.
            strength: Float in ``[0, 1]``. ``0`` MUST return the input
                unchanged (a copy is acceptable). ``1`` is the most extreme
                variant in the sweep.
            seed: Per-video integer used for any stochastic component, so
                that the realization is identical across models.

        Returns:
            Perturbed frames with the same shape and dtype as ``frames``.
        """

    @staticmethod
    def _validate(frames: np.ndarray, strength: float) -> None:
        """Shared shape/dtype/strength validation."""
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"frames must have shape (T, H, W, 3); got {frames.shape}"
            )
        if frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8; got {frames.dtype}")
        if not 0.0 <= float(strength) <= 1.0:
            raise ValueError(f"strength must be in [0, 1]; got {strength}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, tier={self.tier})"
