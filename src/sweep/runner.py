"""Sweep runner: iterate the experiment matrix and accumulate results.

The runner is the only component that knows about all of (perturbation,
metric, model, dataset). It is intentionally thin: it loads one model,
streams the dataset, applies each cell's perturbation, encodes both clean
and perturbed clips, computes the metrics, and hands a list of
``CellResult`` to the checkpoint writer.

Important: encoders are loaded once per model variant and reused across
every cell that targets that variant. Loading is dominated by disk and
network I/O, so per-cell load would dominate the sweep wall clock.
"""

from __future__ import annotations

from src.config import SweepConfig
from src.sweep.matrix import Cell, build_experiment_matrix


def run_sweep(  # pragma: no cover - placeholder
    config: SweepConfig,
    cells: list[Cell] | None = None,
) -> None:
    """Execute the sweep.

    Args:
        config: Top-level sweep configuration.
        cells: Optional cell subset (for pilot runs). Defaults to the full
            matrix.

    Stage 0 implementation outline:

    1. Resolve ``cells`` (default: ``build_experiment_matrix()``).
    2. Skip cells that already have a shard on disk.
    3. Group remaining cells by ``model`` to amortize encoder loads.
    4. For each group: load the encoder, iterate cells, iterate videos,
       encode clean + perturbed, compute metrics, append to a buffer, write
       the shard, free intermediates.
    5. Always update ``manifests/manifest.json`` before exit.
    """
    raise NotImplementedError
