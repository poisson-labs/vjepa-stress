"""Stage-4 entry point: launch the full 222-cell sweep.

Usage:
    python scripts/run_full_sweep.py --config configs/sweep.yaml

The runner is resumable: cells whose Parquet shard already exists in
``config.output_dir`` are skipped.
"""

from __future__ import annotations

import argparse

from vjepa_stress.config import SweepConfig

# TODO Stage 4: wire to ``vjepa_stress.sweep.runner.run_sweep`` once the runner and
# its dependencies (data loaders, model loaders, metrics) are implemented.
# Until then, this script is a stub so that the entry point exists in the
# repo from day one.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to sweep YAML config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the matrix and exit; do not load models or data",
    )
    args = parser.parse_args()

    _ = SweepConfig.from_yaml(args.config)
    if args.dry_run:
        from vjepa_stress.sweep.matrix import build_experiment_matrix

        cells = build_experiment_matrix()
        print(f"Would run {len(cells)} cells.")
        return 0

    raise NotImplementedError("Full sweep entry point — implement once runner is ready.")


if __name__ == "__main__":
    raise SystemExit(main())
