"""Perturbation modules for vjepa-stress.

The perturbation taxonomy follows RESEARCH.md §2:

- P1 Gaussian noise        (sensor)
- P2 Motion blur           (sensor)
- P3 Frame drops           (temporal)
- P4 Occlusion             (spatial)
- P5 Low-light             (sensor)
- P6 Identity              (calibration probe)
- P7 Reverse               (temporal calibration probe)
- P8 Horizontal flip       (spatial calibration probe)
- P9 JPEG compression      (codec)

Every perturbation is a subclass of ``Perturbation`` and obeys the contract
that ``strength=0`` returns the input unchanged.
"""

from __future__ import annotations

from src.perturbations.base import Perturbation
from src.perturbations.codec import JPEGCompression
from src.perturbations.identity import Identity
from src.perturbations.sensor import GaussianNoise, LowLight, MotionBlur
from src.perturbations.spatial import HorizontalFlip, Occlusion
from src.perturbations.temporal import FrameDrops, Reverse


def all_perturbations() -> list[Perturbation]:
    """Return one instance of every concrete perturbation, in canonical order."""
    return [
        GaussianNoise(),
        MotionBlur(),
        FrameDrops(),
        Occlusion(),
        LowLight(),
        Identity(),
        Reverse(),
        HorizontalFlip(),
        JPEGCompression(),
    ]


__all__ = [
    "FrameDrops",
    "GaussianNoise",
    "HorizontalFlip",
    "Identity",
    "JPEGCompression",
    "LowLight",
    "MotionBlur",
    "Occlusion",
    "Perturbation",
    "Reverse",
    "all_perturbations",
]
