"""Per-cell checkpointing to Parquet.

Each completed cell writes one Parquet shard under
``{output_dir}/{cell_id}.parquet`` containing one row per
(video_id, metric, value) tuple plus the cell's coordinates. The full sweep
is recovered by globbing the shard directory and concatenating with DuckDB.

This decouples cell completion from the global sweep state: a crashed run
can be resumed by skipping cells whose shard already exists on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.sweep.matrix import Cell


@dataclass
class CellResult:
    """The metric payload for one (cell, video_id) pair."""

    video_id: str
    m1: float | None = None
    m2: float | None = None
    m3: float | None = None
    notes: str = ""


def shard_path(output_dir: Path, cell: Cell) -> Path:
    """Resolve the Parquet path for one cell."""
    return Path(output_dir) / f"{cell.cell_id}.parquet"


def is_complete(output_dir: Path, cell: Cell) -> bool:
    """Return True if a shard already exists for this cell."""
    return shard_path(output_dir, cell).exists()


def write_shard(  # pragma: no cover - placeholder
    output_dir: Path,
    cell: Cell,
    results: list[CellResult],
) -> Path:
    """Write one cell's results to Parquet and return the path.

    The Stage 0 implementation should serialize ``results`` to a tidy
    DataFrame with columns
    ``[cell_id, tier, model, perturbation, strength, video_id, m1, m2, m3, notes]``
    and write it via ``pyarrow``. Atomic write (write to ``.tmp`` and rename)
    is required so that a crash mid-write doesn't leave a half-written shard
    that ``is_complete`` would treat as done.
    """
    raise NotImplementedError


def load_shards(output_dir: Path) -> pd.DataFrame:  # pragma: no cover - placeholder
    """Glob all Parquet shards and return one tidy DataFrame."""
    raise NotImplementedError
