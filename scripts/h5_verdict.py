"""H5 verdict: Pearson r between ΔM3 and M2 across the DAVIS Tier-1 grid.

Reads the DAVIS sweep shards, validates the schema/finiteness contract,
prints distribution stats, and renders the H5 verdict per
``manifests/predictions.md`` thresholds:

- r > 0.5  → CONFIRMED
- 0.3 ≤ r ≤ 0.5 → AMBIGUOUS
- r < 0.3  → REFUTED

Usage:
    python scripts/h5_verdict.py runs/sweep-davis-9fb565d
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vjepa_stress.sweep.checkpoint import load_shards


def main(run_dir: Path) -> int:
    if not run_dir.is_dir():
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        return 2

    df: pd.DataFrame = load_shards(run_dir).df()
    n = len(df)
    print(f"== Run: {run_dir} ==")
    print(f"rows: {n}")
    print(f"cells (unique cell_id): {df['cell_id'].nunique()}")
    print(f"models: {sorted(df['model'].unique().tolist())}")
    print(f"perturbations: {sorted(df['perturbation'].unique().tolist())}")
    print(f"strengths: {sorted(df['strength'].unique().tolist())}")
    print(f"datasets: {sorted(df['dataset'].unique().tolist())}")
    print(f"clips per cell: {df.groupby('cell_id').size().describe().to_dict()}")

    # Contract checks ------------------------------------------------------
    expected_rows = 100 * 30  # 4 models × 5 perts × 5 strengths × 30 clips
    if n != expected_rows:
        print(
            f"WARN: row count {n} ≠ expected {expected_rows} "
            f"(continuing anyway)"
        )

    davis = df[df["dataset"] == "davis"].copy()
    if len(davis) == 0:
        print("ERROR: no DAVIS rows", file=sys.stderr)
        return 2

    finite_clean = davis["m3_clean"].apply(math.isfinite)
    finite_pert = davis["m3_perturbed"].apply(math.isfinite)
    n_bad_clean = (~finite_clean).sum()
    n_bad_pert = (~finite_pert).sum()
    if n_bad_clean or n_bad_pert:
        print(
            f"ERROR: non-finite m3 on DAVIS rows — clean={n_bad_clean} "
            f"perturbed={n_bad_pert}",
            file=sys.stderr,
        )
        return 2

    in_range = davis["m3_clean"].between(0, 1) & davis["m3_perturbed"].between(0, 1)
    n_oor = (~in_range).sum()
    if n_oor:
        print(f"ERROR: {n_oor} DAVIS rows have m3 outside [0,1]", file=sys.stderr)
        return 2
    print(f"all {len(davis)} DAVIS rows have finite m3_clean, m3_perturbed in [0,1]")

    # Distribution ---------------------------------------------------------
    print()
    print("== M3 distribution (DAVIS) ==")
    for col in ("m3_clean", "m3_perturbed", "m1", "m2"):
        s = davis[col]
        print(
            f"  {col:14s} n={len(s):5d}  "
            f"mean={s.mean():.4f}  std={s.std():.4f}  "
            f"min={s.min():.4f}  max={s.max():.4f}"
        )

    delta_m3 = davis["m3_clean"] - davis["m3_perturbed"]
    print(
        f"  ΔM3            n={len(delta_m3):5d}  "
        f"mean={delta_m3.mean():.4f}  std={delta_m3.std():.4f}  "
        f"min={delta_m3.min():.4f}  max={delta_m3.max():.4f}"
    )

    # H5 verdict -----------------------------------------------------------
    m2 = davis["m2"].to_numpy()
    dm3 = delta_m3.to_numpy()
    if len(m2) < 2 or np.std(m2) == 0 or np.std(dm3) == 0:
        print("ERROR: cannot compute r (zero variance)", file=sys.stderr)
        return 2
    r = float(np.corrcoef(dm3, m2)[0, 1])
    print()
    print("== H5 verdict ==")
    print(f"Pearson r(ΔM3, M2) = {r:+.4f}  (n = {len(m2)})")
    if r > 0.5:
        verdict = "CONFIRMED"
    elif r >= 0.3:
        verdict = "AMBIGUOUS"
    else:
        verdict = "REFUTED"
    print(f"H5: {verdict}")

    # Per-model breakdown for the post.
    print()
    print("== Per-model r(ΔM3, M2) ==")
    for model, sub in davis.groupby("model"):
        sub_dm3 = (sub["m3_clean"] - sub["m3_perturbed"]).to_numpy()
        sub_m2 = sub["m2"].to_numpy()
        if np.std(sub_m2) == 0 or np.std(sub_dm3) == 0:
            continue
        rr = float(np.corrcoef(sub_dm3, sub_m2)[0, 1])
        print(f"  {model:28s} r={rr:+.4f}  (n={len(sub)})")

    print()
    print("== Per-perturbation r(ΔM3, M2) ==")
    for pert, sub in davis.groupby("perturbation"):
        sub_dm3 = (sub["m3_clean"] - sub["m3_perturbed"]).to_numpy()
        sub_m2 = sub["m2"].to_numpy()
        if np.std(sub_m2) == 0 or np.std(sub_dm3) == 0:
            continue
        rr = float(np.corrcoef(sub_dm3, sub_m2)[0, 1])
        print(f"  {pert:18s} r={rr:+.4f}  (n={len(sub)})")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/h5_verdict.py <run_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
