"""Build figures/fig5_partitioning.{png,pdf} — per-perturbation Pearson r between
M2 and ΔM3 on the DAVIS sweep, sorted DESC. Reference lines at r=0.3 (ambiguous)
and r=0.5 (H5 confirmation). Aggregate r marker. Color-coded above/below 0.3.
"""
from __future__ import annotations

import glob
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DAVIS_RUN = REPO / "runs" / "sweep-davis-9fb565d"
FIG_DIR = REPO / "figures"
FIG_DIR.mkdir(exist_ok=True)

TIER1 = ["gaussian_noise", "motion_blur", "frame_drops", "occlusion", "low_light"]
PRETTY = {
    "gaussian_noise": "Gaussian noise",
    "motion_blur": "Motion blur",
    "frame_drops": "Frame drops",
    "occlusion": "Occlusion",
    "low_light": "Low light",
}

con = duckdb.connect()
shards = sorted(glob.glob(str(DAVIS_RUN / "*.parquet")))
df = con.execute(f"SELECT * FROM read_parquet({shards!r})").df()
df["delta_m3"] = df["m3_clean"] - df["m3_perturbed"]

per_pert: dict[str, float] = {}
for p in TIER1:
    sub = df[df["perturbation"] == p]
    r = float(np.corrcoef(sub["m2"], sub["delta_m3"])[0, 1])
    per_pert[p] = r
agg_sub = df[df["perturbation"].isin(TIER1)]
r_agg = float(np.corrcoef(agg_sub["m2"], agg_sub["delta_m3"])[0, 1])

ordered = sorted(per_pert.items(), key=lambda kv: kv[1], reverse=True)
labels = [PRETTY[p] for p, _ in ordered]
values = [r for _, r in ordered]
n_per = len(df[df["perturbation"] == TIER1[0]])

ABOVE = "#0f766e"  # deep teal
BELOW = "#9ca3af"  # warm gray
AMBIG_THRESH = 0.3
H5_THRESH = 0.5

bar_colors = [ABOVE if v >= AMBIG_THRESH else BELOW for v in values]

fig, ax = plt.subplots(figsize=(8.5, 4.5))
y_pos = np.arange(len(labels))[::-1]
bars = ax.barh(y_pos, values, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Within-perturbation Pearson r (M2 vs ΔM3, DAVIS, n=600/perturbation)", fontsize=10)
ax.set_xlim(-0.15, 0.65)
ax.axvline(0, color="black", linewidth=0.6, zorder=2)
ax.axvline(AMBIG_THRESH, color="#b45309", linestyle="--", linewidth=1.2, zorder=2)
ax.axvline(H5_THRESH, color="#15803d", linestyle="--", linewidth=1.2, zorder=2)

ymax = ax.get_ylim()[1]
ax.text(AMBIG_THRESH, ymax + 0.05, "ambiguous\nthreshold (0.30)",
        ha="center", va="bottom", fontsize=8, color="#b45309")
ax.text(H5_THRESH, ymax + 0.05, "pre-registered H5\nthreshold (0.50)",
        ha="center", va="bottom", fontsize=8, color="#15803d")

for bar, v in zip(bars, values):
    x = v + (0.012 if v >= 0 else -0.012)
    ha = "left" if v >= 0 else "right"
    ax.text(x, bar.get_y() + bar.get_height() / 2, f"{v:+.2f}",
            va="center", ha=ha, fontsize=10)

ax.axvline(r_agg, color="#1f2937", linestyle=":", linewidth=1.0, zorder=2)
ax.annotate(f"aggregate r = {r_agg:+.2f}\n(n=3000)",
            xy=(r_agg, -0.45),
            xytext=(r_agg + 0.05, -0.55),
            fontsize=8, color="#1f2937",
            arrowprops=dict(arrowstyle="-", color="#1f2937", lw=0.6))

ax.set_title("M2 predicts downstream task failure — for some perturbations",
             fontsize=12, pad=22)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle=":", color="#d1d5db", alpha=0.7, zorder=1)

plt.tight_layout()

out_png = FIG_DIR / "fig5_partitioning.png"
out_pdf = FIG_DIR / "fig5_partitioning.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"per-perturbation r (sorted DESC):")
for p, r in ordered:
    flag = "ABOVE" if r >= AMBIG_THRESH else "below"
    print(f"  {p:<16} r={r:+.4f}  [{flag} ambiguous threshold]")
print(f"aggregate r = {r_agg:+.4f} (n={len(agg_sub)})")
print(f"per-perturbation n = {n_per}")
print(f"wrote {out_png}")
print(f"wrote {out_pdf}")
