"""Deterministic fakes for the runner.

These stand in for real V-JEPA models and real video datasets while we
build the sweep infrastructure. They have two non-negotiable properties:

1. **Determinism.** Same inputs always yield the same outputs. The fake
   encoder hashes raw frame bytes + a per-model salt; the fake dataset
   hashes ``(dataset_name, index, seed)``. This is what lets us assert
   ``m1 == m2 == 0`` under the Identity probe and ``m1 > 0`` under any
   non-trivial perturbation without flakiness.
2. **Cheap.** No torch model load, no video decode. The full 222-cell
   sweep with fakes finishes in single-digit seconds on a laptop, which
   is what makes the runner's restart-semantics tests practical.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

#: Token grid is 4x4 = 16 patches. Independent of image size; the real
#: encoder also returns a fixed-grid token map regardless of the input
#: resolution it was trained on.
FAKE_PATCHES: int = 16

#: Embedding dimension per fake model. Smaller than the real V-JEPA dims
#: (768/1024/1408/1792) but distinct across variants so a misrouted load
#: would surface as a shape mismatch.
FAKE_EMBED_DIMS: dict[str, int] = {
    "vjepa-2.1-vit-b": 16,
    "vjepa-2.1-vit-l": 24,
    "vjepa-2.1-vit-g": 32,
    "vjepa-2.1-vit-giant": 40,
}


def _embed_dim_for(model_name: str) -> int:
    if model_name in FAKE_EMBED_DIMS:
        return FAKE_EMBED_DIMS[model_name]
    # Fallback: stable hash so unknown names still get a reproducible dim.
    h = hashlib.blake2b(model_name.encode("utf-8"), digest_size=2)
    return 8 + (int.from_bytes(h.digest(), "little") % 56)


def _hash_to_seed(*parts: bytes) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p)
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


class FakeEncoder:
    """Deterministic stand-in for the real V-JEPA frozen encoder.

    Each frame's per-patch features are drawn from a Gaussian seeded by
    ``blake2b(frame_bytes + model_salt)``. Same frame in -> same features
    out, regardless of which call produced it. L2-normalize per token so
    the cosine-based metrics behave the same as on the real encoder.

    The encoder counts ``forward_calls`` on every call so tests can assert
    that the runner reused a single load across all cells of one model.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.embed_dim = _embed_dim_for(model_name)
        self.num_patches = FAKE_PATCHES
        self._salt = hashlib.blake2b(
            model_name.encode("utf-8"), digest_size=8
        ).digest()
        self.forward_calls: int = 0

    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """Encode a clip of shape ``(T, H, W, 3)`` uint8 to ``(T, P, D)`` float32."""
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(
                f"FakeEncoder.encode expects (T,H,W,3) uint8; got "
                f"shape={frames.shape} dtype={frames.dtype}"
            )
        self.forward_calls += 1
        t = frames.shape[0]
        out = torch.empty((t, self.num_patches, self.embed_dim), dtype=torch.float32)
        for i in range(t):
            seed = _hash_to_seed(self._salt, frames[i].tobytes())
            g = torch.Generator().manual_seed(seed % (2**31))
            f = torch.randn(
                self.num_patches, self.embed_dim, generator=g, dtype=torch.float32
            )
            out[i] = torch.nn.functional.normalize(f, dim=-1)
        return out

    # Convenience: callable form mirrors a torch ``nn.Module``.
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        return self.encode(frames)


def fake_encoder_factory(model_name: str) -> FakeEncoder:
    """Top-level factory the runner can use as ``encoder_factory``."""
    return FakeEncoder(model_name)


@dataclass(frozen=True)
class FakeVideo:
    """One synthetic clip plus its identifier."""

    video_id: str
    frames: np.ndarray  # (T, H, W, 3) uint8


class FakeVideoDataset:
    """Deterministic synthetic video dataset.

    Produces clips of shape ``(num_frames, image_size, image_size, 3)``
    uint8. The pixel content is a function of ``(dataset_name, index,
    seed)`` only, so the same dataset configured the same way always
    yields the same clips.
    """

    def __init__(
        self,
        *,
        dataset_name: str = "fake",
        n_videos: int = 4,
        num_frames: int = 4,
        image_size: int = 32,
        seed: int = 0,
    ) -> None:
        if n_videos <= 0:
            raise ValueError(f"n_videos must be positive; got {n_videos}")
        self.dataset_name = dataset_name
        self.n_videos = n_videos
        self.num_frames = num_frames
        self.image_size = image_size
        self.seed = seed

    def video_ids(self) -> list[str]:
        return [f"{self.dataset_name}_{i:04d}" for i in range(self.n_videos)]

    def __len__(self) -> int:
        return self.n_videos

    def __iter__(self) -> Iterator[FakeVideo]:
        for i in range(self.n_videos):
            yield self[i]

    def __getitem__(self, index: int) -> FakeVideo:
        if index < 0 or index >= self.n_videos:
            raise IndexError(index)
        ds_seed = _hash_to_seed(
            self.dataset_name.encode("utf-8"),
            index.to_bytes(8, "little"),
            self.seed.to_bytes(8, "little", signed=False),
        )
        rng = np.random.default_rng(ds_seed)
        frames = rng.integers(
            0,
            256,
            size=(self.num_frames, self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        return FakeVideo(
            video_id=f"{self.dataset_name}_{index:04d}",
            frames=frames,
        )


def fake_dataset_factory(
    *,
    dataset_name: str = "fake",
    n_videos: int = 4,
    num_frames: int = 4,
    image_size: int = 32,
    seed: int = 0,
) -> FakeVideoDataset:
    """Top-level factory the CLI can use to build the dataset under ``--fake``."""
    return FakeVideoDataset(
        dataset_name=dataset_name,
        n_videos=n_videos,
        num_frames=num_frames,
        image_size=image_size,
        seed=seed,
    )
