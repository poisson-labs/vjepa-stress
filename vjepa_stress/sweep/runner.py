"""Sweep runner: iterate the experiment matrix and write per-cell shards.

The runner is the only component that knows about all of (perturbation,
metric, model, dataset). It is intentionally thin:

1. Resolve (or create) the manifest in ``output_dir``. Seeds are
   precomputed there so a resumed run is bit-identical to a fresh run.
2. Group cells by ``model`` so each encoder is loaded **once**, no matter
   how many cells reference it.
3. For each cell, iterate the video dataset, apply the perturbation,
   encode clean+perturbed, compute M1/M2 (M3 is left NaN until DAVIS
   masks land in Stage 0.6).
4. Write one Parquet shard per cell, atomically.

The runner takes its dependencies through ``RunnerConfig`` factories so
the same code path drives both the test suite (with fakes) and a real
sweep (with V-JEPA + SSv2/DAVIS).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import torch
from tqdm import tqdm

from vjepa_stress.metrics import compute_m1, compute_m2
from vjepa_stress.perturbations import Perturbation, all_perturbations
from vjepa_stress.sweep.checkpoint import (
    cell_completed,
    cleanup_stale_tmp,
    write_shard,
)
from vjepa_stress.sweep.manifest import (
    Manifest,
    build_manifest,
    lookup_seed,
    manifest_exists,
    read_manifest,
    write_manifest,
)
from vjepa_stress.sweep.matrix import Cell
from vjepa_stress.sweep.schema import SCHEMA_VERSION, Row

logger = logging.getLogger(__name__)


class CrashInjected(RuntimeError):
    """Raised by the test-only crash-injection hook to simulate failure mid-sweep."""


class _EncoderProtocol(Protocol):
    forward_calls: int

    def encode(self, frames: np.ndarray) -> torch.Tensor: ...


class _DatasetProtocol(Protocol):
    def video_ids(self) -> list[str]: ...

    def __iter__(self): ...


EncoderFactory = Callable[[str], _EncoderProtocol]
DatasetFactory = Callable[[], _DatasetProtocol]


@dataclass
class RunnerConfig:
    """Everything the runner needs to execute a (subset of the) matrix."""

    output_dir: Path
    encoder_factory: EncoderFactory
    dataset_factory: DatasetFactory
    master_seed: int = 20260427
    resume: bool = True
    inject_crash_after_cell: str | None = None
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    extra_manifest: dict = field(default_factory=dict)
    perturbations: dict[str, Perturbation] = field(
        default_factory=lambda: {p.name: p for p in all_perturbations()}
    )


@dataclass
class RunSummary:
    """What the runner returns to its caller."""

    manifest: Manifest
    cells_total: int
    cells_skipped: int
    cells_written: int
    rows_written: int
    output_dir: Path


def _group_cells_by_model(cells: list[Cell]) -> list[tuple[str, list[Cell]]]:
    """Preserve canonical order while clustering cells under the same model."""
    seen: dict[str, list[Cell]] = {}
    order: list[str] = []
    for cell in cells:
        if cell.model not in seen:
            seen[cell.model] = []
            order.append(cell.model)
        seen[cell.model].append(cell)
    return [(m, seen[m]) for m in order]


def _all_perturbation_names(cells: Iterable[Cell]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in cells:
        if c.perturbation not in seen:
            seen.add(c.perturbation)
            out.append(c.perturbation)
    return out


def _ensure_manifest(
    config: RunnerConfig,
    cells: list[Cell],
    video_ids: list[str],
) -> Manifest:
    """Build a fresh manifest, or reuse the existing one when resuming.

    Refuses to resume against a manifest with a mismatched
    ``schema_version`` — that would mean the on-disk shards were written
    by code that disagrees with this code's idea of the row schema.
    """
    output_dir = Path(config.output_dir)
    perturbation_names = _all_perturbation_names(cells)

    if config.resume and manifest_exists(output_dir):
        existing = read_manifest(output_dir)
        if existing.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"manifest at {output_dir} has schema_version "
                f"{existing.schema_version!r}; this code expects "
                f"{SCHEMA_VERSION!r}. Refusing to resume."
            )
        return existing

    manifest = build_manifest(
        video_ids=video_ids,
        perturbation_names=perturbation_names,
        master_seed=config.master_seed,
        repo_root=config.repo_root,
        extra=config.extra_manifest,
    )
    write_manifest(output_dir, manifest)
    return manifest


def _features_to_token_grid(
    features: torch.Tensor, expected_t: int
) -> torch.Tensor:
    """Sanity-check the encoder output shape ``(T, P, D)``."""
    if features.ndim != 3:
        raise ValueError(
            f"encoder must return (T,P,D); got shape {tuple(features.shape)}"
        )
    if features.shape[0] != expected_t:
        raise ValueError(
            f"encoder returned T={features.shape[0]}; expected {expected_t}"
        )
    return features


def _compute_metrics_for_cell(
    cell: Cell,
    frames_clean: np.ndarray,
    frames_perturbed: np.ndarray,
    encoder: _EncoderProtocol,
) -> tuple[float, float, float]:
    """Encode clean+perturbed and return ``(m1, m2, m3)``.

    M3 is NaN until DAVIS masks are wired up (Stage 0.6). The Parquet
    schema stores it as a float, so ``math.nan`` is the right placeholder
    — readers that aggregate M3 will simply ignore those rows.
    """
    feats_clean = _features_to_token_grid(
        encoder.encode(frames_clean), expected_t=frames_clean.shape[0]
    )
    feats_perturbed = _features_to_token_grid(
        encoder.encode(frames_perturbed), expected_t=frames_perturbed.shape[0]
    )

    m1 = float(compute_m1(feats_clean, feats_perturbed).item())
    if feats_clean.shape[0] >= 2:
        m2 = float(compute_m2(feats_clean, feats_perturbed).item())
    else:
        m2 = math.nan
    m3 = math.nan
    return m1, m2, m3


def run_sweep(config: RunnerConfig, cells: list[Cell]) -> RunSummary:
    """Execute the runner against a list of cells. Returns a summary."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_tmp(output_dir)

    dataset = config.dataset_factory()
    video_ids = list(dataset.video_ids())

    manifest = _ensure_manifest(config, cells, video_ids)

    cells_total = len(cells)
    cells_skipped = 0
    cells_written = 0
    rows_written = 0

    grouped = _group_cells_by_model(cells)
    progress = tqdm(total=cells_total, desc="cells", unit="cell")

    for model_name, model_cells in grouped:
        encoder: _EncoderProtocol | None = None

        for cell in model_cells:
            if config.resume and cell_completed(output_dir, cell):
                cells_skipped += 1
                progress.update(1)
                continue

            if cell.perturbation not in config.perturbations:
                raise KeyError(
                    f"unknown perturbation {cell.perturbation!r} on cell {cell.cell_id}"
                )
            perturbation = config.perturbations[cell.perturbation]

            if encoder is None:
                logger.info("loading encoder for model=%s", model_name)
                encoder = config.encoder_factory(model_name)

            # Re-iterate the dataset for each cell so we don't hold T*P*D
            # tensors across all videos in memory at once.
            iterator = config.dataset_factory()
            rows: list[Row] = []
            for video in iterator:
                seed = lookup_seed(manifest, video.video_id, cell.perturbation)
                frames_clean = np.ascontiguousarray(video.frames)
                frames_perturbed = perturbation.apply(
                    frames_clean, strength=cell.strength, seed=seed
                )
                m1, m2, m3 = _compute_metrics_for_cell(
                    cell, frames_clean, frames_perturbed, encoder
                )
                rows.append(
                    Row(
                        cell_id=cell.cell_id,
                        tier=cell.tier,
                        model=cell.model,
                        perturbation=cell.perturbation,
                        strength=float(cell.strength),
                        video_id=video.video_id,
                        seed=seed,
                        m1=m1,
                        m2=m2,
                        m3=m3,
                    )
                )

            write_shard(output_dir, cell, rows)
            cells_written += 1
            rows_written += len(rows)
            progress.update(1)

            if (
                config.inject_crash_after_cell is not None
                and cell.cell_id == config.inject_crash_after_cell
            ):
                progress.close()
                raise CrashInjected(
                    f"inject_crash_after_cell={cell.cell_id!r}"
                )

    progress.close()
    return RunSummary(
        manifest=manifest,
        cells_total=cells_total,
        cells_skipped=cells_skipped,
        cells_written=cells_written,
        rows_written=rows_written,
        output_dir=output_dir,
    )
