"""Slow integration smoke: real V-JEPA + real SSv2 + runner end-to-end.

This is the only test that wires every Stage 0.6 piece together:

    SSv2 loader  →  VJEPAEncoder  →  perturbation  →  M1/M2  →  Parquet shard

It runs **one** Cell (gaussian_noise / vit-base / strength=0.5) over 2
SSv2 clips through the real runner, then reads back the resulting shard
to assert the metric contract:

- ``m1`` is finite and strictly positive (gaussian_noise should perturb
  features detectably; an m1 of zero would mean the perturbation isn't
  reaching the encoder).
- ``m2`` is finite and strictly positive (T_tubelet=8 ≥ 2, so M2 is
  computed; reverse-of-clean would also be the wrong number here, hence
  ``> 0``).
- ``m3`` is NaN — DAVIS masks aren't wired into the runner yet, and the
  schema reserves ``math.nan`` for that placeholder (devlog 2026-04-27,
  Task 5). This stays NaN until M3 plumbing lands.

The test soft-skips when no SSv2 tree is reachable from the resolved
cache root, mirroring the SSv2 loader's manual-download contract. On a
RunPod box with weights + SSv2 cached, this is the canonical "does the
real sweep actually work" smoke.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from vjepa_stress.data.ssv2 import (
    SSV2_INSTALL_URL,
    _resolve_cache_dir,
    load_ssv2_dataset,
)
from vjepa_stress.models.vjepa import load_encoder
from vjepa_stress.sweep.checkpoint import load_shards, shard_path
from vjepa_stress.sweep.matrix import Cell
from vjepa_stress.sweep.runner import RunnerConfig, run_sweep


def _real_ssv2_available() -> bool:
    root = _resolve_cache_dir(None)
    return (root / "validation").is_dir()


@pytest.mark.slow
def test_smoke_real_pipeline(tmp_path: Path):
    if not _real_ssv2_available():
        pytest.skip(
            f"No SSv2 tree at {_resolve_cache_dir(None)}; download manually "
            f"from {SSV2_INSTALL_URL} to run the real-pipeline smoke."
        )

    cell = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="gaussian_noise",
        strength=0.5,
        tier=1,
    )

    # The runner calls ``encoder_factory(model_name)`` once per unique
    # model in the slice. We have exactly one model, so a one-shot cache
    # would be redundant — let the runner's own grouping handle it.
    def encoder_factory(model_name: str):
        return load_encoder(model_name, device="cpu")

    def dataset_factory():
        return load_ssv2_dataset(
            n_videos=2,
            seed=0,
            num_frames=16,
            image_size=384,
        )

    config = RunnerConfig(
        output_dir=tmp_path,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        master_seed=20260427,
    )

    summary = run_sweep(config, cells=[cell])

    assert summary.cells_written == 1
    assert summary.rows_written == 2
    assert shard_path(tmp_path, cell).exists()

    df = load_shards(tmp_path).df()
    assert len(df) == 2
    assert df["cell_id"].unique().tolist() == [cell.cell_id]

    # M1 / M2 contract on a real perturbation.
    m1_vals = df["m1"].tolist()
    m2_vals = df["m2"].tolist()
    assert all(math.isfinite(v) for v in m1_vals), f"m1 must be finite; got {m1_vals}"
    assert all(math.isfinite(v) for v in m2_vals), f"m2 must be finite; got {m2_vals}"
    assert all(v > 0 for v in m1_vals), (
        f"gaussian_noise should produce strictly positive m1; got {m1_vals}"
    )
    assert all(v > 0 for v in m2_vals), (
        f"gaussian_noise should produce strictly positive m2; got {m2_vals}"
    )

    # M3 stays NaN until DAVIS masks are wired into the runner.
    m3_vals = df["m3"].tolist()
    assert all(math.isnan(v) for v in m3_vals), (
        f"m3 should be NaN until DAVIS plumbing lands; got {m3_vals}"
    )
