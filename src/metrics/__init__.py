"""Feature-stability metrics for vjepa-stress.

See RESEARCH.md §3 for definitions:

- M1 — Per-frame structural drift: average cosine distance between paired
  tokens of the clean and perturbed clip.
- M2 — Temporal consistency degradation: change in clip-internal temporal
  smoothness when the clip is perturbed.
- M3 — DAVIS patch correspondence: ability to track mask keypoints across
  frames in token space, evaluated on the perturbed clip.
"""

from __future__ import annotations

from src.metrics.m1_drift import compute_m1
from src.metrics.m2_temporal import compute_m2
from src.metrics.m3_correspondence import compute_m3

__all__ = ["compute_m1", "compute_m2", "compute_m3"]
