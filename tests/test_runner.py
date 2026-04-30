"""Runner glue tests using the fake encoder and fake dataset.

These tests exercise the runner end-to-end without touching V-JEPA or
real datasets. The non-negotiable assertions are:

- Per-cell shards are atomic (mocked rename failure leaves no shard).
- The encoder is loaded **once** per unique model across all cells of
  that model (``forward_calls`` math).
- A crashed run resumes idempotently — re-running the same config
  re-writes nothing.
- Dataset axis dispatch: SSv2 cells use ``ssv2_factory``, DAVIS cells
  use ``davis_factory``, and DAVIS rows carry m3_clean / m3_perturbed
  in [0, 1].
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from unittest import mock

import numpy as np
import pytest

from vjepa_stress.sweep.checkpoint import cell_completed, load_shards, shard_path
from vjepa_stress.sweep.manifest import manifest_exists, read_manifest
from vjepa_stress.sweep.matrix import Cell
from vjepa_stress.sweep.runner import (
    CrashInjected,
    RunnerConfig,
    run_sweep,
)
from vjepa_stress.sweep.schema import ARROW_SCHEMA, SCHEMA_VERSION
from vjepa_stress.testing.fakes import FakeEncoder, FakeVideo, FakeVideoDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(n_videos: int = 3) -> FakeVideoDataset:
    return FakeVideoDataset(
        dataset_name="fake",
        n_videos=n_videos,
        num_frames=4,
        image_size=32,
        seed=0,
    )


def _make_config(
    output_dir: Path,
    *,
    encoder_log: list[str] | None = None,
    n_videos: int = 3,
    resume: bool = True,
    inject_crash_after_cell: str | None = None,
) -> RunnerConfig:
    def encoder_factory(model_name: str) -> FakeEncoder:
        if encoder_log is not None:
            encoder_log.append(model_name)
        return FakeEncoder(model_name)

    def dataset_factory() -> FakeVideoDataset:
        return _make_dataset(n_videos)

    return RunnerConfig(
        output_dir=output_dir,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        master_seed=20260427,
        resume=resume,
        inject_crash_after_cell=inject_crash_after_cell,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_matrix_writes_manifest_and_no_shards(tmp_path: Path):
    config = _make_config(tmp_path)
    summary = run_sweep(config, cells=[])

    assert manifest_exists(tmp_path)
    assert summary.cells_total == 0
    assert summary.cells_written == 0
    assert summary.rows_written == 0
    assert not list(tmp_path.glob("*.parquet"))


def test_single_cell_writes_one_shard_with_n_video_rows(tmp_path: Path):
    cell = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="gaussian_noise",
        strength=0.5,
        tier=1,
    )
    config = _make_config(tmp_path, n_videos=3)
    summary = run_sweep(config, cells=[cell])

    assert summary.cells_written == 1
    assert summary.rows_written == 3

    shard = shard_path(tmp_path, cell)
    assert shard.exists()

    rel = load_shards(tmp_path)
    df = rel.df()
    assert len(df) == 3
    assert df["cell_id"].unique().tolist() == [cell.cell_id]
    assert (df["m1"] > 0).all(), "gaussian_noise should produce non-zero M1"


def test_identity_probe_yields_zero_metrics(tmp_path: Path):
    cell = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="identity",
        strength=1.0,
        tier=2,
    )
    config = _make_config(tmp_path, n_videos=4)
    run_sweep(config, cells=[cell])

    df = load_shards(tmp_path).df()
    assert len(df) == 4
    assert (df["m1"].abs() < 1e-6).all()
    assert (df["m2"].abs() < 1e-6).all()
    # SSv2 cells: M3 is masks-required-only and stays NaN on both columns.
    assert df["m3_clean"].apply(math.isnan).all()
    assert df["m3_perturbed"].apply(math.isnan).all()
    # Default dataset axis is ssv2.
    assert (df["dataset"] == "ssv2").all()


def test_skips_completed_cells_on_resume(tmp_path: Path):
    cells = [
        Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.1, tier=1),
        Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.2, tier=1),
    ]

    log_first: list[str] = []
    config_first = _make_config(tmp_path, encoder_log=log_first, n_videos=2)
    s1 = run_sweep(config_first, cells=cells)
    assert s1.cells_written == 2
    assert s1.cells_skipped == 0

    log_second: list[str] = []
    config_second = _make_config(tmp_path, encoder_log=log_second, n_videos=2)
    s2 = run_sweep(config_second, cells=cells)
    assert s2.cells_written == 0
    assert s2.cells_skipped == 2
    # When every cell is already done, the runner does not need to load the
    # encoder at all.
    assert log_second == []

    # Manifest UUID is stable across resumes.
    assert read_manifest(tmp_path).manifest_id == s1.manifest.manifest_id


def test_shard_schema_is_locked(tmp_path: Path):
    import pyarrow.parquet as pq

    cell = Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.5, tier=1)
    config = _make_config(tmp_path, n_videos=2)
    run_sweep(config, cells=[cell])

    schema = pq.read_schema(shard_path(tmp_path, cell))
    assert schema.names == ARROW_SCHEMA.names
    for f in schema:
        assert f.type == ARROW_SCHEMA.field(f.name).type, f.name


def test_runner_populates_wallclock_sec(tmp_path: Path):
    """v1.1: the runner records a finite, non-negative per-row wallclock."""
    cell = Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.5, tier=1)
    config = _make_config(tmp_path, n_videos=3)
    run_sweep(config, cells=[cell])

    df = load_shards(tmp_path).df()
    assert len(df) == 3
    assert "wallclock_sec" in df.columns
    # Even on the FakeEncoder these aren't zero — perf_counter has sub-µs
    # resolution but encode + metric work is the dominant cost.
    assert (df["wallclock_sec"] >= 0.0).all()
    assert df["wallclock_sec"].apply(math.isfinite).all()


def test_restart_after_injected_crash(tmp_path: Path):
    cells = [
        Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.1, tier=1),
        Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.2, tier=1),
        Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.3, tier=1),
    ]

    crashy = _make_config(
        tmp_path,
        n_videos=2,
        inject_crash_after_cell=cells[0].cell_id,
    )
    with pytest.raises(CrashInjected):
        run_sweep(crashy, cells=cells)

    # Only the first cell got written.
    assert cell_completed(tmp_path, cells[0])
    assert not cell_completed(tmp_path, cells[1])
    assert not cell_completed(tmp_path, cells[2])

    # Manifest already exists; the resume must reuse it.
    pre_id = read_manifest(tmp_path).manifest_id

    resume_cfg = _make_config(tmp_path, n_videos=2)
    summary = run_sweep(resume_cfg, cells=cells)
    assert summary.cells_skipped == 1
    assert summary.cells_written == 2
    assert read_manifest(tmp_path).manifest_id == pre_id

    # All shards present.
    for c in cells:
        assert cell_completed(tmp_path, c)


def test_encoder_loaded_once_per_unique_model(tmp_path: Path):
    cells = []
    for model in ("vjepa-2.1-vit-base", "vjepa-2.1-vit-large"):
        for strength in (0.1, 0.2, 0.3):
            cells.append(Cell(model, "gaussian_noise", strength, tier=1))

    log: list[str] = []
    config = _make_config(tmp_path, encoder_log=log, n_videos=1)
    run_sweep(config, cells=cells)

    # 6 cells across 2 models — the factory should fire exactly twice.
    assert log == ["vjepa-2.1-vit-base", "vjepa-2.1-vit-large"]


def test_atomic_write_failure_leaves_no_partial_shard(tmp_path: Path):
    cell = Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.5, tier=1)
    config = _make_config(tmp_path, n_videos=2)

    real_rename = os.rename

    def boom(src, dst, *args, **kwargs):
        # Fail only on a Parquet shard rename inside the writer; let other
        # renames (the manifest, etc.) pass through.
        if str(dst).endswith(".parquet"):
            raise OSError("simulated rename failure")
        return real_rename(src, dst, *args, **kwargs)

    with mock.patch("vjepa_stress.sweep.checkpoint.os.rename", side_effect=boom):
        with pytest.raises(OSError, match="simulated rename failure"):
            run_sweep(config, cells=[cell])

    # No final shard, no orphan tmp.
    assert not shard_path(tmp_path, cell).exists()
    assert not list(tmp_path.glob("*.parquet"))
    # The runner removes orphan tmp files at the start of any run, but a
    # failed rename must leave the writer in a state that the next run
    # can clean up — verify the next run succeeds end-to-end.
    config2 = _make_config(tmp_path, n_videos=2)
    summary = run_sweep(config2, cells=[cell])
    assert summary.cells_written == 1
    assert cell_completed(tmp_path, cell)


def test_manifest_records_schema_version_and_seeds(tmp_path: Path):
    cell = Cell("vjepa-2.1-vit-base", "gaussian_noise", 0.1, tier=1)
    config = _make_config(tmp_path, n_videos=2)
    run_sweep(config, cells=[cell])

    manifest = read_manifest(tmp_path)
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.n_videos == 2
    assert "fake_0000" in manifest.seeds
    assert "gaussian_noise" in manifest.seeds["fake_0000"]
    assert isinstance(manifest.seeds["fake_0000"]["gaussian_noise"], int)


# ---------------------------------------------------------------------------
# Dataset axis dispatch (Schema v1.2 / DAVIS / H5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeDavisVideo:
    """Mimics ``DAVISSample``: frames + per-frame masks + video_id."""

    video_id: str
    frames: np.ndarray  # (T, H, W, 3) uint8
    masks: np.ndarray  # (T, H, W) int32


class _FakeDavisDataset:
    """Fake DAVIS-shaped iterable: yields a foreground mask covering the
    middle of every frame so M3 has something to track."""

    def __init__(self, n_videos: int = 2, num_frames: int = 4, image_size: int = 32):
        self.n = n_videos
        self.num_frames = num_frames
        self.image_size = image_size

    def video_ids(self) -> list[str]:
        return [f"davis_seq_{i:02d}" for i in range(self.n)]

    def __iter__(self) -> Iterator[_FakeDavisVideo]:
        for i in range(self.n):
            rng = np.random.default_rng(i + 1)
            frames = rng.integers(
                0,
                256,
                size=(self.num_frames, self.image_size, self.image_size, 3),
                dtype=np.uint8,
            )
            masks = np.zeros(
                (self.num_frames, self.image_size, self.image_size), dtype=np.int32
            )
            half = self.image_size // 2
            quarter = self.image_size // 4
            masks[:, quarter : quarter + half, quarter : quarter + half] = 1
            yield _FakeDavisVideo(
                video_id=f"davis_seq_{i:02d}",
                frames=frames,
                masks=masks,
            )


def _make_dual_axis_config(
    output_dir: Path, *, n_videos: int = 2, encoder_log: list[str] | None = None
) -> RunnerConfig:
    """RunnerConfig wired to *both* SSv2 (FakeVideoDataset) and DAVIS (fake)
    factories. Tests that touch only one axis still work."""

    def encoder_factory(model_name: str) -> FakeEncoder:
        if encoder_log is not None:
            encoder_log.append(model_name)
        return FakeEncoder(model_name)

    def ssv2_factory() -> FakeVideoDataset:
        return FakeVideoDataset(
            dataset_name="fake",
            n_videos=n_videos,
            num_frames=4,
            image_size=32,
            seed=0,
        )

    def davis_factory() -> _FakeDavisDataset:
        return _FakeDavisDataset(
            n_videos=n_videos, num_frames=4, image_size=32
        )

    return RunnerConfig(
        output_dir=output_dir,
        encoder_factory=encoder_factory,
        ssv2_factory=ssv2_factory,
        davis_factory=davis_factory,
        master_seed=20260427,
    )


def test_runner_config_requires_some_factory():
    """Empty config = programmer bug, not a soft fallback."""
    with pytest.raises(ValueError):
        RunnerConfig(
            output_dir=Path("/tmp"),
            encoder_factory=lambda _name: FakeEncoder(_name),
        )


def test_runner_dispatches_ssv2_vs_davis_axis(tmp_path: Path):
    ssv2_cell = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="gaussian_noise",
        strength=0.5,
        tier=1,
        dataset="ssv2",
    )
    davis_cell = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="gaussian_noise",
        strength=0.5,
        tier=1,
        dataset="davis",
    )
    config = _make_dual_axis_config(tmp_path, n_videos=2)
    summary = run_sweep(config, cells=[ssv2_cell, davis_cell])

    assert summary.cells_written == 2
    assert summary.rows_written == 4

    df = load_shards(tmp_path).df()
    ssv2_rows = df[df["dataset"] == "ssv2"]
    davis_rows = df[df["dataset"] == "davis"]
    assert len(ssv2_rows) == 2
    assert len(davis_rows) == 2

    # SSv2 rows: M3 stays NaN (no masks).
    assert ssv2_rows["m3_clean"].apply(math.isnan).all()
    assert ssv2_rows["m3_perturbed"].apply(math.isnan).all()
    # DAVIS rows: M3 in [0, 1].
    assert davis_rows["m3_clean"].apply(math.isfinite).all()
    assert davis_rows["m3_perturbed"].apply(math.isfinite).all()
    assert (davis_rows["m3_clean"] >= 0.0).all()
    assert (davis_rows["m3_clean"] <= 1.0).all()
    assert (davis_rows["m3_perturbed"] >= 0.0).all()
    assert (davis_rows["m3_perturbed"] <= 1.0).all()

    # Video_id space is dataset-scoped; DAVIS rows should carry the
    # davis_seq_* ids while SSv2 rows carry fake_*.
    assert all(v.startswith("fake_") for v in ssv2_rows["video_id"])
    assert all(v.startswith("davis_seq_") for v in davis_rows["video_id"])


def test_runner_davis_identity_perturbation_clean_equals_perturbed(tmp_path: Path):
    """The H5 invariant: same encoder + same masks + same keypoint seed
    means an identity perturbation produces m3_clean == m3_perturbed."""
    cell = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="identity",
        strength=1.0,
        tier=2,
        dataset="davis",
    )
    config = _make_dual_axis_config(tmp_path, n_videos=2)
    run_sweep(config, cells=[cell])
    df = load_shards(tmp_path).df()
    assert len(df) == 2
    for _, row in df.iterrows():
        assert math.isfinite(row["m3_clean"])
        assert math.isfinite(row["m3_perturbed"])
        assert row["m3_clean"] == row["m3_perturbed"]
