"""Stage-2 pilot: one model, one perturbation, full strength sweep.

This is the smoke test we run on RunPod before launching the 222-cell sweep.
It exercises model load, perturbation pipeline, encoding, metric compute,
and shard write end-to-end on a single (model, perturbation) slice.

Usage:
    python scripts/pilot_runpod.py \
        --model vjepa-2.1-vit-large \
        --perturbation gaussian_noise

If this finishes cleanly, ``run_full_sweep.py`` should be safe to launch.
"""

from __future__ import annotations

import argparse

from src.config import SweepConfig
from src.sweep.matrix import STRENGTHS, Cell
from src.sweep.runner import run_sweep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vjepa-2.1-vit-large")
    parser.add_argument("--perturbation", default="gaussian_noise")
    parser.add_argument("--config", default=None, help="optional YAML config path")
    args = parser.parse_args()

    config = SweepConfig.from_yaml(args.config) if args.config else SweepConfig()
    cells = [
        Cell(model=args.model, perturbation=args.perturbation, strength=s, tier=1)
        for s in STRENGTHS
    ]
    run_sweep(config, cells=cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
