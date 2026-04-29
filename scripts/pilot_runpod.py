"""Pilot smoke for the RunPod image: one model x one perturbation x 10 strengths.

This is the canonical "does the real sweep work end-to-end on this box"
script. It runs a single Tier-1 slice (default: ``vit-large`` x
``gaussian_noise``, all 10 strengths = 10 cells) against:

- the real V-JEPA 2.1 encoder (loaded once and reused across the slice);
- a small SSv2 subsample (default ``--n-videos 4``);
- the real runner, writing one Parquet shard per cell to the output dir.

If this finishes cleanly, the box is ready for ``run_full_sweep.py`` (the
222-cell Tier-1+2+3 sweep). Total wall time on a single H100 with
``--n-videos 4`` is on the order of minutes; on CPU it is closer to an
hour, mostly model load.

Usage on RunPod (with SSv2 already mounted at /workspace/data/ssv2)::

    python scripts/pilot_runpod.py \\
        --output /workspace/runs/pilot \\
        --device cuda

Usage locally (CPU, smaller subsample)::

    python scripts/pilot_runpod.py \\
        --output /tmp/pilot --device cpu --n-videos 2

The script is intentionally narrow: any extra knob (different matrix
slice, JSON cell list, fake mode, crash injection) goes through
``python -m vjepa_stress.run_sweep`` instead.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vjepa_stress.sweep.matrix import STRENGTHS, Cell
from vjepa_stress.sweep.runner import RunnerConfig, run_sweep


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vjepa-stress pilot",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        default="vjepa-2.1-vit-large",
        help="V-JEPA 2.1 variant short name. Default vit-large keeps the pilot "
        "modestly heavy (≈1.5 GB weights) without spending vit-gigantic time.",
    )
    p.add_argument(
        "--perturbation",
        default="gaussian_noise",
        help="Tier-1 perturbation to sweep across all 10 strengths.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for per-cell Parquet shards and manifest.json.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="Torch device for the encoder: 'auto', 'cuda', 'cpu', or 'mps'.",
    )
    p.add_argument(
        "--n-videos",
        type=int,
        default=4,
        help="SSv2 subsample size per dataset iteration. Each cell processes "
        "this many videos; the pilot's 10 cells × n_videos rows total.",
    )
    p.add_argument(
        "--ssv2-cache-dir",
        type=Path,
        default=None,
        help="Override SSv2 cache root (default: $VJEPA_STRESS_SSV2_DIR or "
        "/workspace/data/ssv2 inside the image).",
    )
    p.add_argument(
        "--master-seed",
        type=int,
        default=20260427,
        help="Seed for the precomputed (video, perturbation) seed dict and "
        "the SSv2 subsample selection.",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-run cells whose shards already exist.",
    )
    p.set_defaults(resume=True)
    return p.parse_args(argv)


def _build_cells(model: str, perturbation: str) -> list[Cell]:
    """Tier-1 slice: 10 strengths × 1 model × 1 perturbation = 10 cells."""
    return [
        Cell(model=model, perturbation=perturbation, strength=s, tier=1)
        for s in STRENGTHS
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("pilot")

    # Lazy imports: keep --help cheap and let users see argument errors
    # before paying the torch.hub / opencv import cost.
    from vjepa_stress.data.ssv2 import load_ssv2_dataset
    from vjepa_stress.models.vjepa import load_encoder

    cells = _build_cells(args.model, args.perturbation)
    log.info(
        "pilot slice: model=%s perturbation=%s strengths=%d n_videos=%d device=%s",
        args.model,
        args.perturbation,
        len(cells),
        args.n_videos,
        args.device,
    )

    def encoder_factory(model_name: str):
        # Runner caches encoders by name; with one model in the slice this
        # is loaded exactly once.
        return load_encoder(model_name, device=args.device)

    def dataset_factory():
        return load_ssv2_dataset(
            n_videos=args.n_videos,
            seed=args.master_seed,
            cache_dir=args.ssv2_cache_dir,
        )

    config = RunnerConfig(
        output_dir=args.output,
        encoder_factory=encoder_factory,
        dataset_factory=dataset_factory,
        master_seed=args.master_seed,
        resume=args.resume,
        extra_manifest={
            "fake": False,
            "n_videos": args.n_videos,
            "device": args.device,
            "ssv2_cache_dir": (
                str(args.ssv2_cache_dir) if args.ssv2_cache_dir is not None else None
            ),
            "pilot": True,
            "pilot_model": args.model,
            "pilot_perturbation": args.perturbation,
        },
    )

    summary = run_sweep(config, cells)
    print(
        f"cells_total={summary.cells_total} "
        f"cells_skipped={summary.cells_skipped} "
        f"cells_written={summary.cells_written} "
        f"rows_written={summary.rows_written} "
        f"output_dir={summary.output_dir}"
    )

    if summary.cells_written + summary.cells_skipped != summary.cells_total:
        log.error("pilot did not cover every cell — check logs above")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
