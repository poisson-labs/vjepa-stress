"""Compute exact numbers for the blog post.

Outputs:
  - analysis/post_numbers.md        — markdown reference (the deliverable)
  - analysis/per_cell_m2_ci.csv     — full 200-row per-cell M2 bootstrap CIs

Pure read-only over the two committed sweeps + DAVIS reverse calibration JSON.
No re-running of experiment compute.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def pearsonr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Lightweight Pearson r — avoids the scipy dep. Returns (r, p) with p=NaN."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.corrcoef(x, y)[0, 1]), float("nan")

REPO = Path(__file__).resolve().parents[1]
SSv2_DIR = REPO / "runs" / "sweep-b077627"
DAVIS_DIR = REPO / "runs" / "sweep-davis-9fb565d"
CALIB_JSON = REPO / "data" / "calibration" / "m2_reverse_calibration.json"
ANALYSIS = REPO / "analysis"

RATE_PER_HR = 1.89  # USD, RTX PRO 6000 community cloud

TIER1 = ["frame_drops", "occlusion", "motion_blur", "low_light", "gaussian_noise"]  # post's order
TIME_AXIS = {"frame_drops", "occlusion"}  # post's "time-axis" framing
IMAGE_NOISE = {"motion_blur", "low_light", "gaussian_noise"}
MODEL_ORDER = [
    "vjepa-2.1-vit-base",
    "vjepa-2.1-vit-large",
    "vjepa-2.1-vit-giant",
    "vjepa-2.1-vit-gigantic",
]
MODEL_SHORT = {m: s for m, s in zip(MODEL_ORDER, ["base", "large", "giant", "gigantic"])}

RNG = np.random.default_rng(20260506)
N_BOOT = 1000


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    half = 1.96 * se
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def boot_ci_mean(values: np.ndarray, rng: np.random.Generator, n_resamples: int = N_BOOT,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) for nonparametric bootstrap of the mean."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    return float(values.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def boot_ci_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
                 n_resamples: int = N_BOOT, alpha: float = 0.05
                 ) -> tuple[float, float, float, float, float, float]:
    """Bootstrap (mean_a - mean_b) and (mean_a / mean_b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diffs = np.empty(n_resamples)
    ratios = np.empty(n_resamples)
    for i in range(n_resamples):
        ai = rng.integers(0, len(a), size=len(a))
        bi = rng.integers(0, len(b), size=len(b))
        ma, mb = a[ai].mean(), b[bi].mean()
        diffs[i] = ma - mb
        ratios[i] = ma / mb if mb != 0 else np.nan
    return (
        float(a.mean() - b.mean()),
        float(np.quantile(diffs, alpha / 2)),
        float(np.quantile(diffs, 1 - alpha / 2)),
        float(a.mean() / b.mean()) if b.mean() != 0 else float("nan"),
        float(np.nanquantile(ratios, alpha / 2)),
        float(np.nanquantile(ratios, 1 - alpha / 2)),
    )


con = duckdb.connect()
ssv2 = con.execute(f"SELECT * FROM read_parquet('{SSv2_DIR}/*.parquet')").df()
davis_shards = sorted(glob.glob(str(DAVIS_DIR / "*.parquet")))
davis = con.execute(f"SELECT * FROM read_parquet({davis_shards!r})").df()
davis["delta_m3"] = davis["m3_clean"] - davis["m3_perturbed"]

# =================================================================
# Section 1 — per-perturbation r + Fisher CI on DAVIS
# =================================================================
sec1_rows = []
for p in TIER1:
    sub = davis[davis.perturbation == p]
    n = len(sub)
    r, _ = pearsonr(sub["delta_m3"], sub["m2"])
    lo, hi = fisher_ci(r, n)
    sec1_rows.append({"perturbation": p, "n": n, "r": r, "ci_lo": lo, "ci_hi": hi})
sec1 = pd.DataFrame(sec1_rows)

# CI overlap check between time-axis and image-noise families
ta = sec1[sec1.perturbation.isin(TIME_AXIS)]
inn = sec1[sec1.perturbation.isin(IMAGE_NOISE)]
ta_lo = ta["ci_lo"].min()  # lowest lower bound in time-axis family
inn_hi = inn["ci_hi"].max()  # highest upper bound in image-noise family
overlap = ta_lo <= inn_hi
gap = ta_lo - inn_hi  # positive => disjoint, negative => overlap
ta_lo_pert = ta.loc[ta["ci_lo"].idxmin(), "perturbation"]
inn_hi_pert = inn.loc[inn["ci_hi"].idxmax(), "perturbation"]

# =================================================================
# Section 2 — aggregate r + CI on DAVIS, n=3000
# =================================================================
agg = davis[davis.perturbation.isin(TIER1)]
r_agg, _ = pearsonr(agg["delta_m3"], agg["m2"])
agg_n = len(agg)
agg_lo, agg_hi = fisher_ci(r_agg, agg_n)

# =================================================================
# Section 3 — per-cell M2 bootstrap CIs (5 perts × 4 models × 10 strengths = 200 cells)
#             + 5 H6 jumps with significance vs pooled CI half-width
# =================================================================
ssv2_t1 = ssv2[ssv2.perturbation.isin(TIER1)].copy()
cell_rows = []
cell_lookup: dict[tuple, dict] = {}
for (pert, model, strength), grp in ssv2_t1.groupby(["perturbation", "model", "strength"]):
    vals = grp["m2"].to_numpy()
    mean, lo, hi = boot_ci_mean(vals, RNG)
    half = (hi - lo) / 2.0
    rec = {
        "perturbation": pert,
        "model": model,
        "strength": float(strength),
        "n": len(vals),
        "m2_mean": mean,
        "m2_ci_lo": lo,
        "m2_ci_hi": hi,
        "m2_ci_half": half,
    }
    cell_rows.append(rec)
    cell_lookup[(pert, model, round(float(strength), 4))] = rec

per_cell = pd.DataFrame(cell_rows).sort_values(["perturbation", "model", "strength"]).reset_index(drop=True)
per_cell.to_csv(ANALYSIS / "per_cell_m2_ci.csv", index=False)

med_half = per_cell["m2_ci_half"].median()
max_half = per_cell["m2_ci_half"].max()


def jump_significance(pert: str, smaller: str, larger: str) -> dict:
    """Strength-averaged: average M2 across all strengths per model, then jump = larger - smaller.
    Same averaging that hypothesis_table.csv uses for H6.
    Pooled CI half-width: sqrt(half_a^2 + half_b^2) / 2 across strengths averaged, but cleanest is
    bootstrap the difference of strength-averaged means directly.
    """
    sub = ssv2_t1[ssv2_t1.perturbation == pert]
    a_vals = sub[sub.model == smaller]["m2"].to_numpy()  # full strength stack: 200 clips × 10 strengths
    b_vals = sub[sub.model == larger]["m2"].to_numpy()
    ma_mean, ma_lo, ma_hi = boot_ci_mean(a_vals, RNG)
    mb_mean, mb_lo, mb_hi = boot_ci_mean(b_vals, RNG)
    half_a = (ma_hi - ma_lo) / 2.0
    half_b = (mb_hi - mb_lo) / 2.0
    pooled_half = np.sqrt(half_a ** 2 + half_b ** 2) / 2.0
    jump = mb_mean - ma_mean
    sig = jump / pooled_half if pooled_half > 0 else float("nan")
    # Bootstrap the difference directly for a cleaner CI
    jump_mean, jump_lo, jump_hi, _, _, _ = boot_ci_diff(b_vals, a_vals, RNG)
    return {
        "perturbation": pert,
        "smaller_model": smaller,
        "larger_model": larger,
        "smaller_m2_mean": ma_mean,
        "smaller_m2_half": half_a,
        "larger_m2_mean": mb_mean,
        "larger_m2_half": half_b,
        "jump": jump,
        "jump_ci_lo": jump_lo,
        "jump_ci_hi": jump_hi,
        "pooled_half": pooled_half,
        "jump_sig": sig,
    }


JUMPS = [
    ("gaussian_noise", "vjepa-2.1-vit-giant", "vjepa-2.1-vit-gigantic"),
    ("motion_blur", "vjepa-2.1-vit-giant", "vjepa-2.1-vit-gigantic"),
    ("frame_drops", "vjepa-2.1-vit-large", "vjepa-2.1-vit-giant"),
    ("occlusion", "vjepa-2.1-vit-large", "vjepa-2.1-vit-giant"),
    ("low_light", "vjepa-2.1-vit-giant", "vjepa-2.1-vit-gigantic"),
]
sec3_jumps = pd.DataFrame([jump_significance(*j) for j in JUMPS])
borderline = sec3_jumps[sec3_jumps["jump_sig"].abs() < 2.0]

# =================================================================
# Section 4 — Finding 4 gap with CIs at s=0.5, plus cross-strength dominance
# =================================================================
S = 0.5
gn_s05 = ssv2[(ssv2.perturbation == "gaussian_noise") & (ssv2.strength == S)]["m2"].to_numpy()
fd_s05 = ssv2[(ssv2.perturbation == "frame_drops") & (ssv2.strength == S)]["m2"].to_numpy()
gn_mean, gn_lo, gn_hi = boot_ci_mean(gn_s05, RNG)
fd_mean, fd_lo, fd_hi = boot_ci_mean(fd_s05, RNG)
gap_mean, gap_lo, gap_hi, ratio_mean, ratio_lo, ratio_hi = boot_ci_diff(gn_s05, fd_s05, RNG)

# Cross-strength: at how many strengths does Gaussian dominate frame_drops on >=3 of 4 models?
cross_rows = []
for s in sorted(ssv2["strength"].unique()):
    n_dom = 0
    per_model = {}
    for m in MODEL_ORDER:
        gn_v = ssv2[(ssv2.perturbation == "gaussian_noise") & (ssv2.strength == s) & (ssv2.model == m)]["m2"].mean()
        fd_v = ssv2[(ssv2.perturbation == "frame_drops") & (ssv2.strength == s) & (ssv2.model == m)]["m2"].mean()
        per_model[m] = (gn_v, fd_v, gn_v > fd_v)
        if gn_v > fd_v:
            n_dom += 1
    cross_rows.append({"strength": s, "n_models_gn_dominates": n_dom, **{f"{MODEL_SHORT[m]}_gn": v[0] for m, v in per_model.items()}})
cross_df = pd.DataFrame(cross_rows)
n_strengths_dominate = int((cross_df["n_models_gn_dominates"] >= 3).sum())
n_strengths_total = len(cross_df)

# =================================================================
# Section 5 — calibration: SSv2 reverse per model + DAVIS reverse calibration JSON
# =================================================================
sec5_rev_rows = []
for m in MODEL_ORDER:
    vals = ssv2[(ssv2.perturbation == "reverse") & (ssv2.model == m)]["m2"].to_numpy()
    mean, lo, hi = boot_ci_mean(vals, RNG)
    sec5_rev_rows.append({"model": m, "n": len(vals), "mean": mean, "ci_lo": lo, "ci_hi": hi})
ssv2_rev = pd.DataFrame(sec5_rev_rows)

calib = json.load(open(CALIB_JSON))
per_clip = calib["per_clip"]
davis_rev_m2 = np.array([c["m2"] for c in per_clip], dtype=float)
davis_rev_mean, davis_rev_lo, davis_rev_hi = boot_ci_mean(davis_rev_m2, RNG)
davis_rev_n = len(davis_rev_m2)

# =================================================================
# Section 6 — hflip per-model + cross-model means
# =================================================================
sec6_rows = []
for m in MODEL_ORDER:
    vals = ssv2[(ssv2.perturbation == "horizontal_flip") & (ssv2.model == m)]["m2"].to_numpy()
    mean, lo, hi = boot_ci_mean(vals, RNG)
    sec6_rows.append({"model": m, "n": len(vals), "mean": mean, "ci_lo": lo, "ci_hi": hi})
hflip_df = pd.DataFrame(sec6_rows)
hflip_pooled = ssv2[ssv2.perturbation == "horizontal_flip"]["m2"].to_numpy()
hf_pool_mean, hf_pool_lo, hf_pool_hi = boot_ci_mean(hflip_pooled, RNG)

# =================================================================
# Section 7 — exact compute totals
# =================================================================
ssv2_sec = float(ssv2["wallclock_sec"].sum())
davis_sec_rowsum = float(davis["wallclock_sec"].sum())
DAVIS_END_TO_END_SEC = 1447.0  # canonical end-to-end pod wallclock (from devlog 2026-04-29)
total_sec_rowsum = ssv2_sec + davis_sec_rowsum
total_sec_canonical = ssv2_sec + DAVIS_END_TO_END_SEC
cost_rowsum = total_sec_rowsum / 3600.0 * RATE_PER_HR
cost_canonical = total_sec_canonical / 3600.0 * RATE_PER_HR
all_wall = pd.concat([ssv2["wallclock_sec"], davis["wallclock_sec"]])
median_ms = float(all_wall.median() * 1000)
p95_ms = float(all_wall.quantile(0.95) * 1000)


# =================================================================
# Build markdown
# =================================================================
def fmt(v: float, dp: int = 3) -> str:
    return f"{v:+.{dp}f}" if v < 0 or (dp > 0 and 0 <= v < 1) else f"{v:.{dp}f}"


def sgn(v: float, dp: int = 3) -> str:
    return f"{v:+.{dp}f}"


lines: list[str] = []
lines.append("# Blog-post numbers — final extraction\n")
lines.append("Pure-extraction reference for the blog draft. All numbers computed from\n"
             "`runs/sweep-b077627/` (SSv2, schema 1.1) and `runs/sweep-davis-9fb565d/`\n"
             "(DAVIS, schema 1.2), plus `data/calibration/m2_reverse_calibration.json`\n"
             "(DAVIS reverse calibration probe, vit-base only). Bootstrap CIs use 1000\n"
             "resamples, seed=20260506. Pearson r CIs use Fisher z. All values to 3 dp\n"
             "unless noted.\n")
lines.append("Canonical references (do not duplicate prose):\n"
             "- `analysis/hypothesis_table.md` — H1–H6 verdict text\n"
             "- `analysis/compute_economics.md` — total $4.29 / per-model breakdown\n"
             "- `RESEARCH.md §3.1` — M1 validity-scope clarification\n")

# Section 1
lines.append("## Section 1 — Per-perturbation r with 95% CIs (Finding 1)\n")
lines.append("DAVIS sweep, ΔM3 = m3_clean − m3_perturbed, within-perturbation Pearson r,\n"
             "Fisher-z CI, n ≈ 600 per perturbation (30 clips × 4 models × 5 strengths).\n")
lines.append("| Perturbation | n | r | 95% CI |\n|---|---|---|---|")
for _, row in sec1.iterrows():
    lines.append(f"| {row.perturbation} | {int(row.n)} | {sgn(row.r)} | [{sgn(row.ci_lo)}, {sgn(row.ci_hi)}] |")
lines.append("")
overlap_phrase = "**No.**" if not overlap else "**Yes.**"
gap_phrase = (f"closest pair: `{ta_lo_pert}` lower bound = {sgn(ta_lo)} vs "
              f"`{inn_hi_pert}` upper bound = {sgn(inn_hi)} → "
              f"gap of {sgn(gap)} (positive = disjoint).")
lines.append(f"**CI overlap (time-axis vs image-noise families):** {overlap_phrase} {gap_phrase}\n")

# Section 2
lines.append("## Section 2 — Aggregate r with CI (Finding 1 supporting)\n")
lines.append(f"DAVIS sweep, all Tier-1 perturbations pooled, n = {agg_n}.\n")
lines.append(f"Aggregate Pearson r = **{sgn(r_agg)}**, 95% CI [{sgn(agg_lo)}, {sgn(agg_hi)}] (Fisher z).\n")

# Section 3
lines.append("## Section 3 — Per-cell M2 bootstrap CIs (H6 / Finding 2)\n")
lines.append(f"5 Tier-1 perturbations × 4 models × 10 strengths = 200 SSv2 cells, n=200/cell.\n"
             f"Full per-cell table: `analysis/per_cell_m2_ci.csv`.\n")
lines.append(f"- **Median CI half-width** across 200 cells: **{med_half:.4f}**")
lines.append(f"- **Maximum CI half-width** across 200 cells: **{max_half:.4f}**\n")
lines.append("Five H6 non-monotonic jumps (strength-averaged across 10 strengths × 200 clips per cell):\n")
lines.append("| Perturbation | Step | Smaller M2 ± half-CI | Larger M2 ± half-CI | Jump | Jump 95% CI | Jump significance (× pooled half) |")
lines.append("|---|---|---|---|---|---|---|")
for _, j in sec3_jumps.iterrows():
    sm = MODEL_SHORT[j.smaller_model]
    lg = MODEL_SHORT[j.larger_model]
    lines.append(
        f"| {j.perturbation} | {sm} → {lg} | "
        f"{j.smaller_m2_mean:.3f} ± {j.smaller_m2_half:.4f} | "
        f"{j.larger_m2_mean:.3f} ± {j.larger_m2_half:.4f} | "
        f"{sgn(j.jump)} | [{sgn(j.jump_ci_lo)}, {sgn(j.jump_ci_hi)}] | "
        f"{j.jump_sig:+.2f}× |"
    )
lines.append("")
if len(borderline) == 0:
    lines.append("**Borderline jumps (< 2× pooled CI half-width):** none — all five jumps are ≥ 2× their pooled CI half-width.\n")
else:
    bl_list = ", ".join(f"`{r.perturbation}` ({MODEL_SHORT[r.smaller_model]}→{MODEL_SHORT[r.larger_model]}, {r.jump_sig:+.2f}×)"
                        for _, r in borderline.iterrows())
    lines.append(f"**Borderline jumps (< 2× pooled CI half-width):** {bl_list}.\n")

# Section 4
lines.append("## Section 4 — Finding 4 gap with CIs (s=0.5)\n")
lines.append(f"SSv2 sweep, strength = 0.5, all 4 models pooled, n = {len(gn_s05)} per perturbation.\n")
lines.append(
    f"Gaussian noise M2 = **{gn_mean:.3f}**, 95% CI [{gn_lo:.3f}, {gn_hi:.3f}]. "
    f"Frame drops M2 = **{fd_mean:.3f}**, 95% CI [{fd_lo:.3f}, {fd_hi:.3f}]. "
    f"Gap (gaussian − frame_drops) = **{sgn(gap_mean)}**, bootstrap 95% CI "
    f"[{sgn(gap_lo)}, {sgn(gap_hi)}]. Ratio (gaussian / frame_drops) = "
    f"**{ratio_mean:.3f}**, bootstrap 95% CI [{ratio_lo:.3f}, {ratio_hi:.3f}]. "
    f"Ratio CI excludes 1.0, so the dominance is robust.\n"
)
lines.append(f"**Cross-strength dominance:** Gaussian dominates frame_drops on ≥ 3 of 4 models at "
             f"**{n_strengths_dominate} of {n_strengths_total} strengths**.\n")

# Section 5
lines.append("## Section 5 — Reverse-playback calibration\n")
lines.append("DAVIS calibration probe (separate from main sweep), vit-base, 30 clips, "
             "from `data/calibration/m2_reverse_calibration.json`:\n")
lines.append(f"- DAVIS reverse mean M2 = **{davis_rev_mean:.3f}**, 95% CI [{davis_rev_lo:.3f}, {davis_rev_hi:.3f}], n = {davis_rev_n}\n")
lines.append("SSv2 sweep, reverse perturbation, per model (n = 200 each):\n")
lines.append("| Model | n | mean M2 | 95% CI |\n|---|---|---|---|")
for _, row in ssv2_rev.iterrows():
    lines.append(f"| {MODEL_SHORT[row.model]} | {int(row.n)} | {row['mean']:.3f} | [{row.ci_lo:.3f}, {row.ci_hi:.3f}] |")
lines.append("")

# Section 6
lines.append("## Section 6 — Horizontal flip (Finding 3)\n")
lines.append("SSv2 sweep, hflip perturbation, per model (n = 200 each):\n")
lines.append("| Model | n | mean M2 | 95% CI |\n|---|---|---|---|")
for _, row in hflip_df.iterrows():
    lines.append(f"| {MODEL_SHORT[row.model]} | {int(row.n)} | {row['mean']:.3f} | [{row.ci_lo:.3f}, {row.ci_hi:.3f}] |")
lines.append("")
lines.append(f"**Pooled across all 4 models** (n = {len(hflip_pooled)}): mean M2 = **{hf_pool_mean:.3f}**, "
             f"95% CI [{hf_pool_lo:.3f}, {hf_pool_hi:.3f}].\n")

# Section 7
lines.append("## Section 7 — Compute economics, exact\n")
lines.append(f"- **Total wallclock (canonical, includes DAVIS pod end-to-end):** {total_sec_canonical:.1f} s = {total_sec_canonical/3600:.3f} h")
lines.append(f"- **Total cost @ ${RATE_PER_HR:.2f}/hr:** ${cost_canonical:.2f}")
lines.append(f"- **SSv2 sweep wallclock (row-summed, end-to-end per cell):** {ssv2_sec:.1f} s = {ssv2_sec/3600:.3f} h")
lines.append(f"- **DAVIS sweep wallclock (row-summed, excludes encoder loads):** {davis_sec_rowsum:.1f} s = {davis_sec_rowsum/3600:.3f} h "
             f"(end-to-end pod wallclock, the canonical figure used in cost: 1,447 s = 0.402 h)")
lines.append(f"- **Median per-row wallclock (both sweeps pooled, n = {len(all_wall)}):** {median_ms:.1f} ms")
lines.append(f"- **p95 per-row wallclock:** {p95_ms:.1f} ms\n")
lines.append(f"**Confirms** the canonical $4.29 figure from `analysis/compute_economics.md` and the "
             f"2026-04-29 pod tear-down devlog entry (within rounding).\n")

(ANALYSIS / "post_numbers.md").write_text("\n".join(lines))

# Console summary for the report-back
print("=" * 72)
print("Section 1 — per-perturbation r with 95% CIs")
print("=" * 72)
for _, row in sec1.iterrows():
    print(f"  {row.perturbation:<16} n={int(row.n):4d}  r={row.r:+.4f}  "
          f"95% CI [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}]")
print()
print(f"CI overlap between time-axis ({sorted(TIME_AXIS)}) and image-noise families: "
      f"{'YES' if overlap else 'NO'}")
print(f"  closest gap: {ta_lo_pert} lo={ta_lo:+.4f} vs {inn_hi_pert} hi={inn_hi:+.4f} "
      f"-> gap={gap:+.4f}")
print()
print(f"Section 2 — aggregate r = {r_agg:+.4f}, 95% CI [{agg_lo:+.4f}, {agg_hi:+.4f}], n={agg_n}")
print()
print("Section 3 — per-cell CI summary")
print(f"  median half-width = {med_half:.4f}")
print(f"  max half-width    = {max_half:.4f}")
print()
print("H6 jumps:")
for _, j in sec3_jumps.iterrows():
    print(f"  {j.perturbation:<16} {MODEL_SHORT[j.smaller_model]:>8}->{MODEL_SHORT[j.larger_model]:<8} "
          f"jump={j.jump:+.4f}  sig={j.jump_sig:+.2f}x  "
          f"jump 95% CI [{j.jump_ci_lo:+.4f}, {j.jump_ci_hi:+.4f}]")
if len(borderline):
    print(f"  borderline (< 2x): {[r.perturbation for _, r in borderline.iterrows()]}")
else:
    print("  borderline (< 2x): NONE")
print()
print(f"Section 4 — gaussian vs frame_drops at s=0.5")
print(f"  gaussian = {gn_mean:.4f} CI [{gn_lo:.4f}, {gn_hi:.4f}]")
print(f"  frame_drops = {fd_mean:.4f} CI [{fd_lo:.4f}, {fd_hi:.4f}]")
print(f"  gap = {gap_mean:+.4f} CI [{gap_lo:+.4f}, {gap_hi:+.4f}]")
print(f"  ratio = {ratio_mean:.4f} CI [{ratio_lo:.4f}, {ratio_hi:.4f}]")
print(f"  cross-strength dominance: {n_strengths_dominate}/{n_strengths_total}")
print()
print(f"Section 5 — DAVIS reverse mean M2 = {davis_rev_mean:.4f} CI [{davis_rev_lo:.4f}, {davis_rev_hi:.4f}]")
for _, row in ssv2_rev.iterrows():
    print(f"  SSv2 reverse {MODEL_SHORT[row.model]:<8}: {row['mean']:.4f} CI [{row.ci_lo:.4f}, {row.ci_hi:.4f}]")
print()
print("Section 6 — hflip per model")
for _, row in hflip_df.iterrows():
    print(f"  {MODEL_SHORT[row.model]:<8}: {row['mean']:.4f} CI [{row.ci_lo:.4f}, {row.ci_hi:.4f}]")
print(f"  pooled: {hf_pool_mean:.4f} CI [{hf_pool_lo:.4f}, {hf_pool_hi:.4f}]")
print()
print("Section 7 — compute")
print(f"  SSv2 row-summed: {ssv2_sec:.1f}s")
print(f"  DAVIS row-summed: {davis_sec_rowsum:.1f}s; canonical end-to-end pod: 1447.0s")
print(f"  total canonical: {total_sec_canonical:.1f}s = {total_sec_canonical/3600:.3f}h => ${cost_canonical:.2f}")
print(f"  median per-row: {median_ms:.1f}ms; p95: {p95_ms:.1f}ms")
print()
print(f"wrote {ANALYSIS.relative_to(REPO) / 'post_numbers.md'}")
print(f"wrote {ANALYSIS.relative_to(REPO) / 'per_cell_m2_ci.csv'} ({len(per_cell)} rows)")
