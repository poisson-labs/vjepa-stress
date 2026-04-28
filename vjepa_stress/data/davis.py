"""DAVIS 2017 loader for M3 patch correspondence.

The sweep evaluates M3 against the 30 sequences in the DAVIS 2017
validation split. DAVIS is a public, ungated download; the loader
expects the standard layout::

    <root>/
        ImageSets/2017/<split>.txt
        JPEGImages/480p/<sequence>/<NNNNN>.jpg
        Annotations/480p/<sequence>/<NNNNN>.png

Frames are returned as ``(T, R, R, 3)`` uint8 RGB. Masks are returned
as ``(T, R, R)`` int32 with palette indices preserved — ``0`` is
background, ``1+`` are instance IDs. Mask resize must use nearest-
neighbour to keep IDs as integers; OpenCV's bilinear resize would
produce non-integer values that don't index cleanly.

Frame and mask indices share the same ``np.linspace`` sample (per
Stage 0.6 Task 5 spec) so M3 keypoint tracking lines up between the
encoder's input and the per-pixel ground truth.

DAVIS PNG masks are *paletted* PNGs: each pixel byte is a palette
index, and the palette table maps that index to an RGB colour for
visualisation. ``cv2.imread`` decodes the palette and returns RGB,
which silently destroys the instance IDs. PIL is used to read masks
because ``np.array(Image.open(p))`` returns the indices unchanged
(provided the image is in ``"P"`` mode, which DAVIS's are).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#: Where to point users when the cache dir is missing.
DAVIS_INSTALL_URL: str = "https://davischallenge.org/davis2017/code.html"

#: Default cache dir. Overridable via ``$VJEPA_STRESS_DAVIS_DIR`` or the
#: ``cache_dir`` argument to :func:`load_davis_dataset`.
DEFAULT_CACHE_DIR: Path = Path.home() / "data" / "davis"

#: DAVIS 2017 val has exactly 30 sequences. Pinned here so the test
#: suite can assert it without re-reading the ImageSets file.
DAVIS_2017_VAL_COUNT: int = 30


@dataclass
class DAVISSample:
    """One DAVIS sequence: frames + per-frame instance masks.

    ``video_id`` is duplicated as ``sequence_id`` for symmetry with
    DAVIS's own naming, but the runner only ever reads ``video_id``
    (matching :class:`vjepa_stress.testing.fakes.FakeVideo`).
    """

    sequence_id: str
    frames: np.ndarray  # (T, H, W, 3) uint8 RGB
    masks: np.ndarray  # (T, H, W) int32; 0 = bg, 1+ = instance IDs

    @property
    def video_id(self) -> str:
        return self.sequence_id


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    """Resolve the DAVIS cache root: arg → env → default."""
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    env = os.environ.get("VJEPA_STRESS_DAVIS_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_CACHE_DIR


def _unwrap_inner_davis(root: Path) -> Path:
    """Some downloads keep an inner ``DAVIS/`` directory; unwrap it.

    Accepts either::

        <root>/JPEGImages/...                  # already unwrapped
        <root>/DAVIS/JPEGImages/...            # inner DAVIS/ dir present
    """
    if (root / "JPEGImages").is_dir() and (root / "Annotations").is_dir():
        return root
    inner = root / "DAVIS"
    if (inner / "JPEGImages").is_dir() and (inner / "Annotations").is_dir():
        return inner
    return root  # let the missing-data error path catch this below


def _read_sequence_ids(root: Path, split: str) -> list[str]:
    """Read the sequence list from ``ImageSets/2017/<split>.txt``.

    Falls back to listing ``JPEGImages/480p/`` if the ImageSets file
    is missing — some DAVIS distributions ship without the val.txt
    list, and we'd rather succeed loudly than fail mysteriously.
    """
    txt = root / "ImageSets" / "2017" / f"{split}.txt"
    if txt.is_file():
        seqs = [
            line.strip()
            for line in txt.read_text().splitlines()
            if line.strip()
        ]
        return seqs
    fallback = root / "JPEGImages" / "480p"
    if fallback.is_dir():
        return sorted(p.name for p in fallback.iterdir() if p.is_dir())
    raise FileNotFoundError(
        f"Cannot read sequence list: neither {txt} nor {fallback} exists. "
        f"DAVIS 2017 must be downloaded from {DAVIS_INSTALL_URL}."
    )


def _resize_and_center_crop_frame(
    frame: np.ndarray, image_size: int
) -> np.ndarray:
    """Same short-side-resize + center-crop as SSv2, but kept local so
    the two loaders stay independently auditable."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"degenerate frame size {(h, w)}")
    if h < w:
        new_h = image_size
        new_w = max(image_size, int(round(w * image_size / h)))
    else:
        new_w = image_size
        new_h = max(image_size, int(round(h * image_size / w)))
    interp = cv2.INTER_AREA if (new_h * new_w) < (h * w) else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    top = (new_h - image_size) // 2
    left = (new_w - image_size) // 2
    return resized[top : top + image_size, left : left + image_size, :]


def _resize_and_center_crop_mask(
    mask: np.ndarray, image_size: int
) -> np.ndarray:
    """Same crop-resize geometry as the frame helper, but with NEAREST
    interpolation so palette indices remain integers."""
    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"degenerate mask size {(h, w)}")
    if h < w:
        new_h = image_size
        new_w = max(image_size, int(round(w * image_size / h)))
    else:
        new_w = image_size
        new_h = max(image_size, int(round(h * image_size / w)))
    resized = cv2.resize(
        mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    )
    top = (new_h - image_size) // 2
    left = (new_w - image_size) // 2
    return resized[top : top + image_size, left : left + image_size]


def _load_jpeg(path: Path) -> np.ndarray:
    """Decode one DAVIS JPEG to ``(H, W, 3)`` uint8 RGB."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"OpenCV failed to read JPEG: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_palette_png(path: Path) -> np.ndarray:
    """Decode one DAVIS paletted PNG to ``(H, W)`` int32 instance IDs.

    DAVIS masks ship in PIL ``"P"`` mode where each pixel is a palette
    index from 0..255. ``np.array(Image.open(p))`` returns those indices
    directly. We cast to int32 so downstream comparisons (``mask == k``)
    don't sign-coerce uint8 unexpectedly.
    """
    with Image.open(path) as im:
        # Convert defensively if the PNG was saved as RGB (rare, but a
        # well-meaning preprocessor might have done it).
        if im.mode == "RGB" or im.mode == "RGBA":
            arr = np.array(im.convert("L"))
        else:
            arr = np.array(im)
    if arr.ndim != 2:
        raise RuntimeError(
            f"DAVIS mask {path} is not a 2-D paletted PNG; got shape {arr.shape}"
        )
    return arr.astype(np.int32, copy=False)


def _linspace_indices(total: int, num_frames: int) -> np.ndarray:
    if total <= 0:
        raise ValueError(f"total must be positive; got {total}")
    return np.linspace(0, total - 1, num_frames).round().astype(np.int64)


def _load_sequence(
    root: Path,
    sequence_id: str,
    *,
    num_frames: int,
    image_size: int,
) -> DAVISSample:
    """Load and decode one DAVIS sequence into a :class:`DAVISSample`."""
    jpeg_dir = root / "JPEGImages" / "480p" / sequence_id
    mask_dir = root / "Annotations" / "480p" / sequence_id
    if not jpeg_dir.is_dir():
        raise FileNotFoundError(
            f"Missing JPEG dir for sequence {sequence_id!r}: {jpeg_dir}"
        )
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Missing Annotations dir for sequence {sequence_id!r}: {mask_dir}"
        )

    jpeg_paths = sorted(jpeg_dir.glob("*.jpg"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not jpeg_paths or not mask_paths:
        raise FileNotFoundError(
            f"sequence {sequence_id!r} has no frames or no masks "
            f"(jpeg={len(jpeg_paths)}, png={len(mask_paths)})"
        )
    if len(jpeg_paths) != len(mask_paths):
        raise RuntimeError(
            f"sequence {sequence_id!r}: frame/mask count mismatch "
            f"({len(jpeg_paths)} vs {len(mask_paths)}). DAVIS distributions "
            f"are 1:1 — check the cache integrity."
        )

    indices = _linspace_indices(len(jpeg_paths), num_frames)

    frames = np.empty((num_frames, image_size, image_size, 3), dtype=np.uint8)
    masks = np.empty((num_frames, image_size, image_size), dtype=np.int32)
    for i, idx in enumerate(indices):
        frames[i] = _resize_and_center_crop_frame(
            _load_jpeg(jpeg_paths[int(idx)]), image_size
        )
        masks[i] = _resize_and_center_crop_mask(
            _load_palette_png(mask_paths[int(idx)]), image_size
        )

    return DAVISSample(sequence_id=sequence_id, frames=frames, masks=masks)


class DAVISDataset:
    """Iterable over DAVIS sequences as :class:`DAVISSample` records.

    Implements the same iteration surface as
    :class:`vjepa_stress.testing.fakes.FakeVideoDataset` —
    ``video_ids()``, ``__len__``, ``__iter__``, ``__getitem__`` —
    plus per-sample masks. The runner sees only ``video_id`` and
    ``frames`` today; M3 wiring will pull ``masks`` later.
    """

    def __init__(
        self,
        *,
        root: Path,
        sequence_ids: list[str],
        num_frames: int = 16,
        image_size: int = 384,
    ) -> None:
        if not sequence_ids:
            raise ValueError("sequence_ids must be non-empty")
        self.root: Path = Path(root)
        self._sequence_ids: list[str] = list(sequence_ids)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    def video_ids(self) -> list[str]:
        return list(self._sequence_ids)

    def sequence_ids(self) -> list[str]:
        return list(self._sequence_ids)

    def __len__(self) -> int:
        return len(self._sequence_ids)

    def __iter__(self) -> Iterator[DAVISSample]:
        for i in range(len(self._sequence_ids)):
            yield self[i]

    def __getitem__(self, index: int) -> DAVISSample:
        if index < 0 or index >= len(self._sequence_ids):
            raise IndexError(index)
        return _load_sequence(
            self.root,
            self._sequence_ids[index],
            num_frames=self.num_frames,
            image_size=self.image_size,
        )


def load_davis_dataset(
    *,
    split: str = "val",
    cache_dir: Path | str | None = None,
    num_frames: int = 16,
    image_size: int = 384,
    sequence_ids: list[str] | None = None,
) -> DAVISDataset:
    """Build a :class:`DAVISDataset` from a manually-downloaded DAVIS tree.

    Args:
        split: subset name; ``"val"`` uses ``ImageSets/2017/val.txt``.
        cache_dir: explicit path; overrides ``$VJEPA_STRESS_DAVIS_DIR``
            and the default ``~/data/davis/``. Either the unwrapped root
            (containing ``JPEGImages/`` etc.) or a parent containing an
            inner ``DAVIS/`` directory is accepted.
        num_frames, image_size: forwarded to :class:`DAVISDataset`.
        sequence_ids: explicit override; if provided, skips the
            ImageSets file entirely. Mostly for tests.

    Raises:
        FileNotFoundError: if the cache root, the
            ``JPEGImages``/``Annotations`` subdirs, or the ImageSets
            file are missing. The error message includes the install
            URL and the env-var override hint.
    """
    raw_root = _resolve_cache_dir(cache_dir)
    if not raw_root.is_dir():
        raise FileNotFoundError(
            f"DAVIS cache dir not found: {raw_root}. "
            f"Download DAVIS 2017 from {DAVIS_INSTALL_URL} and unpack "
            f"into {raw_root}, or set $VJEPA_STRESS_DAVIS_DIR / pass "
            f"cache_dir=... to point elsewhere."
        )
    root = _unwrap_inner_davis(raw_root)

    if not (root / "JPEGImages" / "480p").is_dir() or not (
        root / "Annotations" / "480p"
    ).is_dir():
        raise FileNotFoundError(
            f"DAVIS layout not found under {root} — expected "
            f"JPEGImages/480p/ and Annotations/480p/. "
            f"See {DAVIS_INSTALL_URL} for the canonical archive."
        )

    if sequence_ids is None:
        sequence_ids = _read_sequence_ids(root, split)
    if not sequence_ids:
        raise FileNotFoundError(
            f"DAVIS sequence list for split={split!r} is empty under {root}. "
            f"See {DAVIS_INSTALL_URL}."
        )

    return DAVISDataset(
        root=root,
        sequence_ids=sequence_ids,
        num_frames=num_frames,
        image_size=image_size,
    )
