"""Run-time configuration for vjepa-stress.

Configs are dataclasses so they round-trip cleanly through YAML and through
the manifest. The runner reads a `SweepConfig`; everything downstream takes
its sub-configs by reference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Where datasets live on disk and which splits to use."""

    ssv2_root: Path = Path("data/ssv2")
    ssv2_split: str = "validation"
    ssv2_video_ids: Path | None = None  # JSON file with the 200 IDs we sample

    davis_root: Path = Path("data/davis")
    davis_split: str = "val"
    davis_video_ids: Path | None = None  # JSON file with the 30 IDs we sample

    num_frames: int = 16
    frame_sampling: str = "uniform"  # uniform | stride | dense
    image_size: int = 224


@dataclass
class ModelConfig:
    """V-JEPA 2.1 variant selection and load-time options."""

    # HF repo IDs are filled in by the model loader; left as identifiers here.
    variant: str = "vjepa-2.1-vit-large"
    revision: str | None = None  # commit hash, pinned via the manifest
    dtype: str = "float16"  # float16 on GPU, float32 on MPS fallback
    device: str = "auto"  # auto | cuda | mps | cpu


@dataclass
class SweepConfig:
    """Top-level sweep configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    output_dir: Path = Path("checkpoints")
    manifest_path: Path = Path("manifests/manifest.json")

    seed_video_selection: int = 20260427
    seed_perturbation: int = 1
    seed_frame_sampling: int = 2

    batch_size: int = 1
    log_every: int = 5

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> SweepConfig:
        """Load a `SweepConfig` from a YAML file."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            **{k: v for k, v in raw.items() if k not in {"data", "model"}},
        )
