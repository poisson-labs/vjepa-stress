"""Sweep runner: iterate the experiment matrix and write per-cell shards.

The runner is the only component that knows about all of (perturbation,
metric, model, dataset). It is intentionally thin:

1. Resolve (or create) the manifest in ``output_dir``. Seeds are
   precomputed there so a resumed run is bit-identical to a fresh run.
2. Group cells by ``model`` so each encoder is loaded **once**, no matter
   how many cells reference it.
3. For each cell, iterate the video dataset, apply the perturbation,
   encode clean+perturbed, compute M1/M2 (and, for DAVIS cells, m3_clean
   + m3_perturbed via :func:`compute_m3_for_clip`).
4. Write one Parquet shard per cell, atomically.

The runner takes its dependencies through ``RunnerConfig`` factories so
the same code path drives both the test suite (with fakes) and a real
sweep (with V-JEPA + SSv2/DAVIS).

**Dataset dispatch.** A cell carries a ``dataset`` axis (``"ssv2"`` or
``"davis"``); the runner resolves which factory to call per cell. The
legacy ``dataset_factory`` field stays as the default for SSv2 + tests,
and ``ssv2_factory`` / ``davis_factory`` override per-axis when present.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import torch
from tqdm import tqdm

from vjepa_stress.metrics import compute_m1, compute_m2
from vjepa_stress.metrics.m3_correspondence import compute_m3_for_clip
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
    embedding_dim: int
    tubelet_size: int
    patch_size: int
    input_resolution: int
    forward_calls: int

    def output_temporal_dim(self, n_frames: int) -> int: ...

    def output_spatial_patches(self) -> int: ...

    def encode(self, frames: np.ndarray) -> torch.Tensor: ...


class _DatasetProtocol(Protocol):
    def video_ids(self) -> list[str]: ...

    def __iter__(self): ...


EncoderFactory = Callable[[str], _EncoderProtocol]
DatasetFactory = Callable[[], _DatasetProtocol]


@dataclass
class RunnerConfig:
    """Everything the runner needs to execute a (subset of the) matrix.

    ``dataset_factory`` is the legacy single-dataset entry point and is
    used for any cell whose ``dataset`` axis lacks a specific factory.
    ``ssv2_factory`` / ``davis_factory`` override per-axis when set; that
    is what the DAVIS / H5 sweep wires up to point at the real DAVIS
    loader while the SSv2 axis stays on the SSv2 loader. At least one of
    ``dataset_factory`` or the per-axis factories must resolve a cell's
    dataset, or the runner raises a ``KeyError`` at iteration time.
    """

    output_dir: Path
    encoder_factory: EncoderFactory
    dataset_factory: DatasetFactory | None = None
    ssv2_factory: DatasetFactory | None = None
    davis_factory: DatasetFactory | None = None
    master_seed: int = 20260427
    resume: bool = True
    inject_crash_after_cell: str | None = None
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    extra_manifest: dict = field(default_factory=dict)
    perturbations: dict[str, Perturbation] = field(
        default_factory=lambda: {p.name: p for p in all_perturbations()}
    )

    def __post_init__(self) -> None:
        # The runner needs *some* way to materialize a dataset; an entirely
        # empty config is a programmer bug, not a runtime fallback.
        if (
            self.dataset_factory is None
            and self.ssv2_factory is None
            and self.davis_factory is None
        ):
            raise ValueError(
                "RunnerConfig requires at least one of dataset_factory, "
                "ssv2_factory, or davis_factory."
            )

    def factory_for_dataset(self, dataset: str) -> DatasetFactory:
        """Resolve the dataset factory for a cell's dataset axis.

        Per-axis factory wins; falls back to the legacy ``dataset_factory``
        when present. Raises if no factory covers the axis — the caller
        (the runner) should treat this as a hard configuration error.
        """
        if dataset == "ssv2" and self.ssv2_factory is not None:
            return self.ssv2_factory
        if dataset == "davis" and self.davis_factory is not None:
            return self.davis_factory
        if self.dataset_factory is not None:
            return self.dataset_factory
        raise KeyError(
            f"No factory configured for dataset={dataset!r}; set "
            f"RunnerConfig.{dataset}_factory or .dataset_factory."
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


def _all_dataset_axes(cells: Iterable[Cell]) -> list[str]:
    """Return the dataset axes referenced by ``cells`` in canonical order."""
    seen: set[str] = set()
    out: list[str] = []
    for c in cells:
        if c.dataset not in seen:
            seen.add(c.dataset)
            out.append(c.dataset)
    return out


def _collect_video_ids(
    config: RunnerConfig, cells: list[Cell]
) -> list[str]:
    """Resolve the union of video_ids across the dataset axes the cells touch.

    DAVIS cells get ``davis::<sequence_id>`` so they never collide with
    SSv2 cells in the manifest's seed dict — the perturbation seed is
    keyed on (video_id, perturbation_name) and the same sequence name
    could conceivably show up in both datasets.
    """
    out: list[str] = []
    for axis in _all_dataset_axes(cells):
        factory = config.factory_for_dataset(axis)
        ds = factory()
        for vid in ds.video_ids():
            tagged = vid if axis == "ssv2" else f"{axis}::{vid}"
            if tagged not in out:
                out.append(tagged)
    return out


def _manifest_video_id(dataset: str, video_id: str) -> str:
    """Map a per-dataset video id to its manifest-seed key."""
    return video_id if dataset == "ssv2" else f"{dataset}::{video_id}"


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
    """Sanity-check the encoder output shape ``(T_tubelet, P, D)``.

    ``expected_t`` is the encoder's *output* temporal dimension after any
    tubelet collapse, not the raw input frame count. Callers should pass
    ``encoder.output_temporal_dim(n_frames)``.
    """
    if features.ndim != 3:
        raise ValueError(
            f"encoder must return (T,P,D); got shape {tuple(features.shape)}"
        )
    if features.shape[0] != expected_t:
        raise ValueError(
            f"encoder returned T={features.shape[0]}; expected {expected_t}"
        )
    return features


def _m3_seed_for_video(video_id: str) -> int:
    """Per-video M3 keypoint seed, *independent* of the perturbation seed.

    H5 needs M3 to be evaluated against the same set of keypoints on the
    clean and perturbed clip — otherwise ΔM3 mixes "the encoder drifted"
    with "we tracked a different mask region this time". The seed is a
    deterministic function of ``video_id`` only.
    """
    h = hashlib.blake2b(video_id.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little") & ((1 << 31) - 1)


def _compute_metrics_for_cell(
    cell: Cell,
    frames_clean: np.ndarray,
    frames_perturbed: np.ndarray,
    encoder: _EncoderProtocol,
    *,
    masks: np.ndarray | None = None,
    video_id: str | None = None,
) -> tuple[float, float, float, float]:
    """Encode clean+perturbed and return ``(m1, m2, m3_clean, m3_perturbed)``.

    For SSv2 cells (``masks is None``), m3_clean and m3_perturbed are
    NaN — M3 requires per-frame instance masks, which SSv2 does not
    have. The Parquet schema stores both as float64, so ``math.nan`` is
    the right placeholder.

    For DAVIS cells (``masks is not None``), both M3 readings are
    computed against the *original* ground-truth masks. ``ΔM3 =
    m3_clean − m3_perturbed`` is then the LHS of the H5 Pearson r —
    the perturbed run's degradation against fixed ground truth.
    Keypoints share a seed (``video_id``-derived) so the same patches
    are tracked in both runs.
    """
    expected_t_clean = encoder.output_temporal_dim(frames_clean.shape[0])
    expected_t_perturb = encoder.output_temporal_dim(frames_perturbed.shape[0])
    feats_clean = _features_to_token_grid(
        encoder.encode(frames_clean), expected_t=expected_t_clean
    )
    feats_perturbed = _features_to_token_grid(
        encoder.encode(frames_perturbed), expected_t=expected_t_perturb
    )

    m1 = float(compute_m1(feats_clean, feats_perturbed).item())
    if feats_clean.shape[0] >= 2:
        m2 = float(compute_m2(feats_clean, feats_perturbed).item())
    else:
        m2 = math.nan

    if masks is None:
        return m1, m2, math.nan, math.nan

    if video_id is None:
        raise ValueError(
            "video_id is required when masks are provided (M3 keypoint seed)"
        )
    seed_m3 = _m3_seed_for_video(video_id)
    m3_clean = float(
        compute_m3_for_clip(
            features=feats_clean,
            masks=masks,
            tubelet_size=encoder.tubelet_size,
            seed=seed_m3,
        ).item()
    )
    m3_perturbed = float(
        compute_m3_for_clip(
            features=feats_perturbed,
            masks=masks,
            tubelet_size=encoder.tubelet_size,
            seed=seed_m3,
        ).item()
    )
    return m1, m2, m3_clean, m3_perturbed


def run_sweep(config: RunnerConfig, cells: list[Cell]) -> RunSummary:
    """Execute the runner against a list of cells. Returns a summary."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_tmp(output_dir)

    # Resolve per-axis video_ids for the manifest before any cell runs;
    # this ensures resume re-uses the same seed table even on a slice
    # that only covers some axes of the original sweep.
    video_ids = _collect_video_ids(config, cells)

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
            # tensors across all videos in memory at once. Dataset axis
            # is resolved per cell so a single config can serve mixed
            # SSv2 + DAVIS slices.
            factory = config.factory_for_dataset(cell.dataset)
            iterator = factory()
            rows: list[Row] = []
            cuda_available = torch.cuda.is_available()
            for video in iterator:
                manifest_vid = _manifest_video_id(cell.dataset, video.video_id)
                seed = lookup_seed(manifest, manifest_vid, cell.perturbation)
                frames_clean = np.ascontiguousarray(video.frames)
                frames_perturbed = perturbation.apply(
                    frames_clean, strength=cell.strength, seed=seed
                )
                masks = getattr(video, "masks", None) if cell.dataset == "davis" else None
                # Per-(cell, video) wallclock around the encode pair +
                # metrics. CUDA work is async, so synchronize on both ends
                # when a GPU is present — otherwise we'd be timing kernel
                # launches, not kernel execution.
                if cuda_available:
                    torch.cuda.synchronize()
                t_start = time.perf_counter()
                m1, m2, m3_clean, m3_perturbed = _compute_metrics_for_cell(
                    cell,
                    frames_clean,
                    frames_perturbed,
                    encoder,
                    masks=masks,
                    video_id=video.video_id,
                )
                if cuda_available:
                    torch.cuda.synchronize()
                wallclock_sec = time.perf_counter() - t_start
                rows.append(
                    Row(
                        cell_id=cell.cell_id,
                        tier=cell.tier,
                        dataset=cell.dataset,
                        model=cell.model,
                        perturbation=cell.perturbation,
                        strength=float(cell.strength),
                        video_id=video.video_id,
                        seed=seed,
                        m1=m1,
                        m2=m2,
                        m3_clean=m3_clean,
                        m3_perturbed=m3_perturbed,
                        wallclock_sec=wallclock_sec,
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
