"""DAVIS 2017 loader for M3 patch correspondence.

The sweep uses the DAVIS 2017 validation split. We sample 30 sequences pinned
in the manifest, decode them at native resolution, and return both the frames
and the per-frame instance masks. M3 is computed by lifting a sparse mask
keypoint set into V-JEPA token space and tracking each keypoint via cosine
nearest-neighbor across frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DAVISSample:
    """One DAVIS sequence: frames + per-frame instance masks."""

    sequence_id: str
    frames: np.ndarray  # (T, H, W, 3), uint8, RGB
    masks: np.ndarray   # (T, H, W),    uint8, instance ids; 0 = background


class DAVISDataset:
    """Stage-0 placeholder.

    The real implementation needs to:

    1. Read the pinned sequence list from ``sequence_ids_path``.
    2. Load JPEGs from ``JPEGImages/480p/<seq>/`` and PNG masks from
       ``Annotations/480p/<seq>/``.
    3. Resample to ``num_frames`` if requested (DAVIS sequences are short, so
       in practice we may keep all frames).
    4. Return ``DAVISSample``.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        sequence_ids_path: Path | None,
        num_frames: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sequence_ids_path = Path(sequence_ids_path) if sequence_ids_path else None
        self.num_frames = num_frames

    def __len__(self) -> int:  # pragma: no cover - placeholder
        raise NotImplementedError

    def __getitem__(self, idx: int) -> DAVISSample:  # pragma: no cover - placeholder
        raise NotImplementedError
