"""CLI entry point for the sweep runner.

Run with the fake encoder + fake dataset for development:

    python -m vjepa_stress.run_sweep --matrix tier2 --output /tmp/test_shards/ --fake

Or against a saved JSON list of cells (useful for pilot subsets):

    python -m vjepa_stress.run_sweep --matrix path/to/cells.json --output ... --fake

Real V-JEPA + SSv2/DAVIS factories arrive in Stage 0.6.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vjepa_stress.sweep.matrix import Cell, build_experiment_matrix
from vjepa_stress.sweep.runner import RunnerConfig, run_sweep
from vjepa_stress.testing.fakes import FakeEncoder, fake_dataset_factory


def _load_cells_from_json(path: Path) -> list[Cell]:
    with open(path) as f:
        raw = json.load(f)
    out: list[Cell] = []
    for d in raw:
        out.append(
            Cell(
                model=d["model"],
                perturbation=d["perturbation"],
                strength=float(d["strength"]),
                tier=int(d["tier"]),
            )
        )
    return out


def _select_cells(spec: str) -> list[Cell]:
    """Resolve ``--matrix`` to a concrete list of cells."""
    full = build_experiment_matrix()
    if spec == "all":
        return full
    if spec == "tier1":
        return [c for c in full if c.tier == 1]
    if spec == "tier2":
        return [c for c in full if c.tier == 2]
    if spec == "tier3":
        return [c for c in full if c.tier == 3]
    return _load_cells_from_json(Path(spec))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vjepa-stress sweep runner",
        description="Run a vjepa-stress sweep over a slice of the experiment matrix.",
    )
    p.add_argument(
        "--matrix",
        default="all",
        help="Which cells to run: 'all', 'tier1', 'tier2', 'tier3', or a path "
        "to a JSON list of cell dicts.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for per-cell Parquet shards and manifest.json.",
    )
    p.add_argument(
        "--fake",
        action="store_true",
        help="Use the fake encoder and fake dataset (Stage 0.5 default; "
        "real V-JEPA + SSv2 land in Stage 0.6).",
    )
    p.add_argument(
        "--n-videos",
        type=int,
        default=4,
        help="Number of synthetic videos when --fake is set.",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=20260427,
        help="Seed for the precomputed (video,perturbation) seed dict.",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-run cells whose shards already exist.",
    )
    p.add_argument(
        "--inject-crash-after-cell",
        default=None,
        help="Test hook: raise CrashInjected after writing the named cell's shard.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.set_defaults(resume=True)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.fake:
        parser.error(
            "Stage 0.5 only supports --fake. Real V-JEPA + dataset loaders "
            "arrive in Stage 0.6."
        )

    cells = _select_cells(args.matrix)
    if not cells:
        print(f"No cells selected for matrix={args.matrix!r}", file=sys.stderr)
        return 2

    def encoder_factory(model_name: str) -> FakeEncoder:
        return FakeEncoder(model_name)

    def dataset_factory():
        return fake_dataset_factory(
            dataset_name="fake",
            n_videos=args.n_videos,
            num_frames=4,
            image_size=32,
            seed=0,
        )

    config = RunnerConfig(
        output_dir=args.output,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        master_seed=args.master_seed,
        resume=args.resume,
        inject_crash_after_cell=args.inject_crash_after_cell,
        extra_manifest={"fake": True, "n_videos": args.n_videos},
    )

    summary = run_sweep(config, cells)
    print(
        f"cells_total={summary.cells_total} "
        f"cells_skipped={summary.cells_skipped} "
        f"cells_written={summary.cells_written} "
        f"rows_written={summary.rows_written} "
        f"output_dir={summary.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
