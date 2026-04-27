"""Something-Something v2 loader.

The sweep evaluates on a deterministic 200-video subsample of the SSv2
validation split. The exact IDs live in `data.ssv2_video_ids` and are pinned
in the manifest.

Frames are returned as ``np.ndarray`` of shape ``(T, H, W, 3)``, dtype
``uint8``, RGB, in ``[0, 255]`` — matching the ``Perturbation.apply``
contract so the perturbation pipeline can run on raw frames before any
model-specific normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SSv2Sample:
    """One decoded clip plus the metadata we need for tracing."""

    video_id: str
    frames: np.ndarray  # (T, H, W, 3), uint8, RGB
    fps: float
    duration_s: float


class SSv2Dataset:
    """Stage-0 placeholder.

    The real implementation needs to:

    1. Read the pinned video-id list from ``video_ids_path``.
    2. Decode each video to a fixed ``num_frames`` using uniform sampling.
       Decord on Linux, OpenCV on macOS.
    3. Resize the short side to ``image_size`` and center-crop to a square.
    4. Return ``SSv2Sample`` with frames ready for ``Perturbation.apply``.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        video_ids_path: Path | None,
        num_frames: int = 16,
        image_size: int = 224,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.video_ids_path = Path(video_ids_path) if video_ids_path else None
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self) -> int:  # pragma: no cover - placeholder
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SSv2Sample:  # pragma: no cover - placeholder
        raise NotImplementedError
