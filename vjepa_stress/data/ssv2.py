"""Something-Something v2 (SSv2) loader.

The sweep evaluates on a deterministic subsample of the SSv2 validation
split. Per Stage 0.6 Task 4, the prior is **manual download only**: SSv2
is gated (~50 GB), so the loader expects videos to already exist under
``~/data/ssv2/`` (or wherever ``$VJEPA_STRESS_SSV2_DIR`` points) and
fails fast with installation instructions when they don't.

Frames are returned as ``np.ndarray`` of shape ``(T, H, W, 3)``, dtype
``uint8``, RGB, in ``[0, 255]`` — matching the ``Perturbation.apply``
contract and the :class:`vjepa_stress.testing.fakes.FakeVideoDataset`
interface, so the runner stays agnostic to whether it's iterating fakes
or real clips.

Decode uses OpenCV (``cv2.VideoCapture``). decord is the fast option on
Linux and is wired in via ``pyproject.toml``'s ``sys_platform != 'darwin'``
guard, but this loader sticks to OpenCV so the same code path runs on
Mac dev boxes and on RunPod.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Where to point users when the cache dir is missing or empty.
SSV2_INSTALL_URL: str = (
    "https://developer.qualcomm.com/software/ai-datasets/something-something"
)

#: Default cache dir. Overridable via ``$VJEPA_STRESS_SSV2_DIR`` or the
#: ``cache_dir`` argument to :func:`load_ssv2_dataset`.
DEFAULT_CACHE_DIR: Path = Path.home() / "data" / "ssv2"

#: SSv2 ships in WebM (VP9) by default; some redistribution paths use
#: mp4. We accept both so the loader works against either tree.
DEFAULT_FILE_PATTERNS: tuple[str, ...] = ("*.webm", "*.mp4")


@dataclass
class SSv2Sample:
    """One decoded clip plus the metadata the runner needs.

    Mirrors :class:`vjepa_stress.testing.fakes.FakeVideo` so both can flow
    through the runner's iteration loop without a type switch.
    """

    video_id: str
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    """Resolve the SSv2 cache root in priority order: arg → env → default."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get("VJEPA_STRESS_SSV2_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CACHE_DIR


def _glob_video_paths(
    split_dir: Path, file_patterns: tuple[str, ...]
) -> list[Path]:
    """Return a *sorted* list of video paths under ``split_dir``."""
    paths: list[Path] = []
    for pattern in file_patterns:
        paths.extend(split_dir.glob(pattern))
    return sorted(paths)


def _load_frames_with_opencv(
    path: Path, num_frames: int, image_size: int
) -> np.ndarray:
    """Decode one video to ``(num_frames, image_size, image_size, 3)`` uint8 RGB.

    Frame indices are chosen by ``np.linspace(0, total-1, num_frames)``
    rounded to int (the same convention the FakeVideoDataset would use
    if we ever needed to re-sample). Short-side resize + center-crop
    keeps aspect ratio without breaking object scale.

    **Decode strategy.** We sequentially decode every frame into memory
    and then downsample by index, rather than seeking with
    ``CAP_PROP_POS_FRAMES``. Two reasons:

    1. ``CAP_PROP_FRAME_COUNT`` lies on a non-trivial fraction of SSv2
       webms (e.g. ``10220.webm`` claims 43 frames but only 42 are
       decodable — frame index 42 returns ``ok=False`` and the seek
       path raises mid-sweep). Sequential decode gives us the truthful
       count.
    2. Even when the count is honest, OpenCV seek on VFR WebM is not
       frame-accurate; it can return a neighboring keyframe. Sequential
       decode is bit-deterministic.

    SSv2 clips average ~50 frames so decoding all of them is
    negligible vs. the encode pass that follows.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV failed to open {path}")

    try:
        # Sequential decode of the entire clip. Stop on the first
        # ``ok=False`` from cap.read(), which is the truthful EOF
        # marker regardless of what the header claims.
        frames_bgr: list[np.ndarray] = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            frames_bgr.append(frame_bgr)

        total = len(frames_bgr)
        if total == 0:
            raise RuntimeError(f"video has 0 decodable frames: {path}")

        last_valid = total - 1
        indices = np.linspace(0, last_valid, num_frames).round().astype(np.int64)
        indices = np.clip(indices, 0, last_valid)

        out = np.empty(
            (num_frames, image_size, image_size, 3), dtype=np.uint8
        )
        for i, idx in enumerate(indices):
            frame_rgb = cv2.cvtColor(frames_bgr[int(idx)], cv2.COLOR_BGR2RGB)
            out[i] = _resize_and_center_crop(frame_rgb, image_size)
        return out
    finally:
        cap.release()


def _resize_and_center_crop(frame: np.ndarray, image_size: int) -> np.ndarray:
    """Resize short side to ``image_size``, then center-crop to a square."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"degenerate frame size {(h, w)}")
    if h < w:
        new_h = image_size
        new_w = max(image_size, int(round(w * image_size / h)))
    else:
        new_w = image_size
        new_h = max(image_size, int(round(h * image_size / w)))
    # INTER_AREA is the right choice when downsampling; INTER_LINEAR for up.
    interp = cv2.INTER_AREA if (new_h * new_w) < (h * w) else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    top = (new_h - image_size) // 2
    left = (new_w - image_size) // 2
    return resized[top : top + image_size, left : left + image_size, :]


class SSv2Dataset:
    """Iterable dataset of decoded SSv2 clips.

    The constructor is intentionally low-level (it takes a pre-sampled
    list of paths). Callers should usually go through
    :func:`load_ssv2_dataset` instead, which handles cache resolution,
    sampling, and the missing-data error path.
    """

    def __init__(
        self,
        *,
        video_paths: list[Path],
        num_frames: int = 16,
        image_size: int = 384,
        dataset_name: str = "ssv2",
    ) -> None:
        if not video_paths:
            raise ValueError("video_paths must be non-empty")
        # Defensive copy so callers can't mutate iteration order from outside.
        self._paths: list[Path] = list(video_paths)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.dataset_name = dataset_name

    @staticmethod
    def _video_id_for(path: Path) -> str:
        # SSv2 IDs are the bare numeric stem (``"12345.webm"`` → ``"12345"``);
        # using ``.stem`` keeps the manifest-friendly identifier without us
        # having to parse anything.
        return path.stem

    def video_ids(self) -> list[str]:
        return [self._video_id_for(p) for p in self._paths]

    def __len__(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[SSv2Sample]:
        for i in range(len(self._paths)):
            yield self[i]

    def __getitem__(self, index: int) -> SSv2Sample:
        if index < 0 or index >= len(self._paths):
            raise IndexError(index)
        path = self._paths[index]
        frames = _load_frames_with_opencv(
            path, num_frames=self.num_frames, image_size=self.image_size
        )
        return SSv2Sample(
            video_id=self._video_id_for(path),
            frames=frames,
        )


def load_ssv2_dataset(
    *,
    n_videos: int = 200,
    seed: int = 0,
    split: str = "validation",
    cache_dir: Path | str | None = None,
    num_frames: int = 16,
    image_size: int = 384,
    file_patterns: tuple[str, ...] = DEFAULT_FILE_PATTERNS,
) -> SSv2Dataset:
    """Build an :class:`SSv2Dataset` from a manually-downloaded SSv2 tree.

    Args:
        n_videos: number of clips to subsample (default 200, matching
            the sweep design). Capped at the available pool with a
            warning if the on-disk tree is smaller.
        seed: subsample seed. Same seed → same set of video IDs (after
            the deterministic sort).
        split: subdirectory under the cache root, e.g. ``"validation"``.
        cache_dir: explicit path; overrides ``$VJEPA_STRESS_SSV2_DIR``
            and the default ``~/data/ssv2/``.
        num_frames, image_size: forwarded to :class:`SSv2Dataset`.
        file_patterns: globs to match. Default catches both the canonical
            ``.webm`` distribution and ``.mp4`` re-encodes.

    Raises:
        FileNotFoundError: if the resolved cache dir doesn't exist or
            contains no matching videos. The error message includes the
            install URL so a fresh user knows where to go.
    """
    root = _resolve_cache_dir(cache_dir)
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"SSv2 split dir not found at {split_dir}. "
            f"SSv2 is gated and must be downloaded manually; see "
            f"{SSV2_INSTALL_URL}. You can also point the loader elsewhere "
            f"by setting cache_dir=... or $VJEPA_STRESS_SSV2_DIR."
        )

    all_paths = _glob_video_paths(split_dir, file_patterns)
    if not all_paths:
        raise FileNotFoundError(
            f"No SSv2 videos matching {file_patterns!r} under {split_dir}. "
            f"SSv2 is gated and must be downloaded manually; see "
            f"{SSV2_INSTALL_URL}."
        )

    if n_videos > len(all_paths):
        logger.warning(
            "SSv2 pool has %d videos under %s but n_videos=%d was requested; "
            "using all %d.",
            len(all_paths),
            split_dir,
            n_videos,
            len(all_paths),
        )
        chosen = all_paths
    else:
        rng = random.Random(seed)
        chosen = rng.sample(all_paths, n_videos)

    # Sort *after* sampling so iteration order is deterministic and easy
    # to audit, while still being a function of (seed, pool) only.
    chosen = sorted(chosen)

    return SSv2Dataset(
        video_paths=chosen,
        num_frames=num_frames,
        image_size=image_size,
    )
