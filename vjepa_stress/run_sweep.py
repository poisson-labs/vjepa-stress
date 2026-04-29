"""CLI entry point for the sweep runner.

Run with the fake encoder + fake dataset for development:

    python -m vjepa_stress.run_sweep --matrix tier2 --output /tmp/test_shards/ --fake

Run against the real V-JEPA 2.1 encoder + a manually-downloaded SSv2 tree:

    python -m vjepa_stress.run_sweep --matrix tier2 --output /tmp/sweep/ \\
        --device auto --n-videos 200

Or against a saved JSON list of cells (useful for pilot subsets):

    python -m vjepa_stress.run_sweep --matrix path/to/cells.json --output ... --fake

The real-mode dataset path expects SSv2 to already be on disk; see
``vjepa_stress.data.ssv2`` for the manual-download contract.
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
        help="Use the fake encoder and fake dataset (Stage 0.5 dev default). "
        "Without this flag the runner loads real V-JEPA 2.1 weights and the "
        "SSv2 validation tree.",
    )
    p.add_argument(
        "--n-videos",
        type=int,
        default=4,
        help="Number of videos per dataset iteration. With --fake, synthetic "
        "videos are generated; in real mode this subsamples SSv2.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Device for the real V-JEPA encoder: 'auto' (default), 'cpu', "
        "'cuda', or 'mps'. Ignored when --fake is set.",
    )
    p.add_argument(
        "--ssv2-cache-dir",
        type=Path,
        default=None,
        help="Override SSv2 cache root. Falls back to $VJEPA_STRESS_SSV2_DIR "
        "and then ~/data/ssv2/. Ignored when --fake is set.",
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


def _build_fake_factories(args: argparse.Namespace):
    """Wire up the dev-only fake encoder + fake dataset."""

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

    extra_manifest = {"fake": True, "n_videos": args.n_videos}
    return encoder_factory, dataset_factory, extra_manifest


def _build_real_factories(args: argparse.Namespace):
    """Wire up real V-JEPA 2.1 + the manually-downloaded SSv2 tree.

    Imports happen here (not at module top) so ``--fake`` runs don't pay
    the torch.hub / cv2 / Pillow import cost just to print help text or
    spin synthetic data.
    """
    from vjepa_stress.data.ssv2 import load_ssv2_dataset
    from vjepa_stress.models.vjepa import load_encoder

    def encoder_factory(model_name: str):
        # The runner already ensures this is called once per unique model
        # in the slice, so we don't add an extra cache layer here.
        return load_encoder(model_name, device=args.device)

    def dataset_factory():
        # Seed the SSv2 subsample off ``master_seed`` so a sweep run is
        # uniquely identified by (master_seed, n_videos, on-disk pool).
        return load_ssv2_dataset(
            n_videos=args.n_videos,
            seed=args.master_seed,
            cache_dir=args.ssv2_cache_dir,
        )

    extra_manifest = {
        "fake": False,
        "n_videos": args.n_videos,
        "device": args.device,
        "ssv2_cache_dir": (
            str(args.ssv2_cache_dir) if args.ssv2_cache_dir is not None else None
        ),
    }
    return encoder_factory, dataset_factory, extra_manifest


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cells = _select_cells(args.matrix)
    if not cells:
        print(f"No cells selected for matrix={args.matrix!r}", file=sys.stderr)
        return 2

    if args.fake:
        encoder_factory, dataset_factory, extra_manifest = _build_fake_factories(args)
    else:
        encoder_factory, dataset_factory, extra_manifest = _build_real_factories(args)

    config = RunnerConfig(
        output_dir=args.output,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        master_seed=args.master_seed,
        resume=args.resume,
        inject_crash_after_cell=args.inject_crash_after_cell,
        extra_manifest=extra_manifest,
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
