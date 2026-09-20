"""Final styling pass for the three blog-post figures.

Outputs (PNG @ 300 DPI, vector PDF, vector SVG):
  - figures/fig2_scaling.{png,pdf,svg}
  - figures/fig3_calibration.{png,pdf,svg}
  - figures/fig5_partitioning.{png,pdf,svg}

Pure styling pass over the existing data — numbers identical to the
prior renders. Other figures (fig1, fig4, h4_strength_comparison) are
not touched and stay as-is for the paper appendix.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import duckdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SSv2_DIR = REPO / "runs" / "sweep-b077627"
DAVIS_DIR = REPO / "runs" / "sweep-davis-9fb565d"
FIG_DIR = REPO / "figures"

# ---------------------------------------------------------------------
# Shared visual identity
# ---------------------------------------------------------------------
TEAL = "#1f6f6f"          # above-threshold / strong-signal
TEAL_LIGHT = "#5fa5a5"
GRAY = "#9ca3af"          # below-threshold / weak-signal
GRAY_LIGHT = "#d1d5db"
INK = "#111827"
RULE = "#374151"
RED_X = "#c0392b"
AMBER = "#b45309"
GREEN = "#15803d"
BAND_GREEN = "#9bc995"

# Viridis-derived blues for the model-size sequence (base → gigantic).
# Hand-picked to be perceptually monotonic and readable on white.
MODEL_BLUES = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]

MODEL_ORDER = [
    "vjepa-2.1-vit-base",
    "vjepa-2.1-vit-large",
    "vjepa-2.1-vit-giant",
    "vjepa-2.1-vit-gigantic",
]
MODEL_SHORT = {m: s for m, s in zip(MODEL_ORDER, ["base", "large", "giant", "gigantic"])}
MODEL_PARAMS_M = {
    "vjepa-2.1-vit-base": 80,
    "vjepa-2.1-vit-large": 300,
    "vjepa-2.1-vit-giant": 1000,
    "vjepa-2.1-vit-gigantic": 2000,
}
MODEL_BLUE = dict(zip(MODEL_ORDER, MODEL_BLUES))

TIER1 = ["frame_drops", "occlusion", "motion_blur", "low_light", "gaussian_noise"]
PERTURB_LABEL = {
    "frame_drops": "Frame drops",
    "occlusion": "Occlusion",
    "motion_blur": "Motion blur",
    "low_light": "Low light",
    "gaussian_noise": "Gaussian noise",
}
TIME_AXIS = {"frame_drops", "occlusion"}

# Sans-serif stack with reasonable cross-platform fallbacks.
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial",
                        "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def save_three(fig: plt.Figure, stem: str) -> None:
    """Save PNG (300 DPI), PDF (vector), and SVG (vector) under figures/."""
    base = FIG_DIR / stem
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def boot_ci_mean(values: np.ndarray, rng: np.random.Generator,
                 n_resamples: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = values[idx].mean()
    return float(values.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
con = duckdb.connect()
ssv2 = con.execute(f"SELECT * FROM read_parquet('{SSv2_DIR}/*.parquet')").df()
davis_shards = sorted(glob.glob(str(DAVIS_DIR / "*.parquet")))
davis = con.execute(f"SELECT * FROM read_parquet({davis_shards!r})").df()
davis["delta_m3"] = davis["m3_clean"] - davis["m3_perturbed"]

t1 = ssv2[ssv2.perturbation.isin(TIER1)].copy()


# =====================================================================
# Figure 5 — partitioning (style pass; current cleanest)
# =====================================================================
def render_fig5() -> None:
    per_pert: dict[str, float] = {}
    for p in TIER1:
        sub = davis[davis.perturbation == p]
        per_pert[p] = float(np.corrcoef(sub.m2, sub["delta_m3"])[0, 1])
    agg_sub = davis[davis.perturbation.isin(TIER1)]
    r_agg = float(np.corrcoef(agg_sub.m2, agg_sub["delta_m3"])[0, 1])
    ordered = sorted(per_pert.items(), key=lambda kv: kv[1], reverse=True)

    AMBIG, H5T = 0.30, 0.50
    labels = [PERTURB_LABEL[p] for p, _ in ordered]
    values = [r for _, r in ordered]
    colors = [TEAL if v >= AMBIG else GRAY for v in values]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    y_pos = np.arange(len(labels))[::-1]
    bars = ax.barh(y_pos, values, color=colors, edgecolor=INK, linewidth=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Within-perturbation Pearson r (M2 vs ΔM3, DAVIS, n=600/perturbation)")
    ax.set_xlim(-0.20, 0.65)

    # Reserve vertical space above the chart for threshold callouts so they
    # never overlap the title.
    ax.set_ylim(-1.2, len(labels) - 0.2)

    # Zero rule — neutral
    ax.axvline(0, color=INK, linewidth=0.6, zorder=2)

    # Faint x-axis gridlines (10% opacity)
    ax.grid(axis="x", linestyle="-", color=GRAY_LIGHT, alpha=0.10, zorder=1)

    # Threshold lines: dashed, different colors so they read distinctly
    # from the bars.
    ax.axvline(AMBIG, color=AMBER, linestyle="--", linewidth=1.4, zorder=2.5)
    ax.axvline(H5T, color=GREEN, linestyle="--", linewidth=1.4, zorder=2.5)

    # Threshold annotations sit ABOVE the chart in the reserved margin.
    callout_y = len(labels) - 0.35
    ax.text(AMBIG, callout_y, "ambiguous threshold (0.30)",
            ha="center", va="bottom", fontsize=9, color=AMBER,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text(H5T, callout_y, "pre-registered H5 threshold (0.50)",
            ha="center", va="bottom", fontsize=9, color=GREEN,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # Bar value annotations
    for bar, v in zip(bars, values):
        x = v + (0.012 if v >= 0 else -0.012)
        ha = "left" if v >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2, f"{v:+.2f}",
                va="center", ha=ha, fontsize=11, color=INK)

    # Aggregate r marker — dotted, sits below the bars in the reserved margin.
    ax.axvline(r_agg, color=RULE, linestyle=":", linewidth=1.0, zorder=2)
    ax.annotate(f"aggregate r = {r_agg:+.2f} (n=3000)",
                xy=(r_agg, -0.6), xytext=(r_agg + 0.05, -0.95),
                fontsize=9, color=RULE,
                arrowprops=dict(arrowstyle="-", color=RULE, lw=0.6))

    ax.set_title("M2 predicts downstream task failure — for some perturbations",
                 fontsize=13, pad=28, loc="left", color=INK)

    plt.tight_layout()
    save_three(fig, "fig5_partitioning")
    plt.close(fig)
    print(f"[fig5] r values: {[(p, round(v,3)) for p,v in ordered]}, agg={r_agg:+.3f}")


# =====================================================================
# Figure 3 — calibration (3-panel: identity, reverse, hflip)
# =====================================================================
def render_fig3() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    bar_kwargs = dict(edgecolor=INK, linewidth=0.6)

    x_pos = np.arange(len(MODEL_ORDER))
    bar_colors = [MODEL_BLUE[m] for m in MODEL_ORDER]
    short_labels = [MODEL_SHORT[m] for m in MODEL_ORDER]

    # ---- Panel 1: identity ----
    ax = axes[0]
    means = (ssv2[ssv2.perturbation == "identity"].groupby("model")["m2"].mean()
             .reindex(MODEL_ORDER).values)
    abs_max = float(np.max(np.abs(means)))
    # Empty panel — actual values at ~1e-8 are well below any meaningful
    # threshold. We don't render bars at all; we draw a flat baseline at
    # zero and explain the result with a single annotation.
    ax.axhline(0, color=INK, linewidth=1.0, zorder=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.set_title("Identity", loc="left", fontsize=12)
    ax.text(0.5, 0.6, f"All values |M2| < 1e-7\n(within numerical noise floor;\nmax |M2| = {abs_max:.1e})",
            transform=ax.transAxes, ha="center", va="center", fontsize=11,
            color=INK,
            bbox=dict(facecolor="white", edgecolor=GRAY_LIGHT, pad=8, linewidth=0.6))

    # ---- Panel 2: reverse ----
    ax = axes[1]
    means_r = (ssv2[ssv2.perturbation == "reverse"].groupby("model")["m2"].mean()
               .reindex(MODEL_ORDER).values)
    ax.axhspan(0.9, 1.4, color=BAND_GREEN, alpha=0.18, zorder=1)
    ax.axhline(1.14, color=RED_X, linewidth=1.2, linestyle="--", zorder=2)
    ax.bar(x_pos, means_r, color=bar_colors, zorder=3, **bar_kwargs)
    for i, v in enumerate(means_r):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    # In-panel annotations instead of a legend (no overlap with bars)
    ax.text(0.02, 0.96, "pre-registered band [0.9, 1.4]",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="#3b6f3b",
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.9))
    ax.text(0.02, 0.86, "pre-registered center (1.14)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=RED_X,
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.9))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.6)
    ax.set_ylabel("Mean M2", fontsize=11)
    ax.set_title("Reverse", loc="left", fontsize=12)
    ax.grid(axis="y", linestyle="-", color=GRAY_LIGHT, alpha=0.10, zorder=0)

    # ---- Panel 3: hflip ----
    ax = axes[2]
    means_h = (ssv2[ssv2.perturbation == "horizontal_flip"].groupby("model")["m2"].mean()
               .reindex(MODEL_ORDER).values)
    ax.axhline(0.30, color=RED_X, linewidth=1.2, linestyle="--", zorder=2)
    ax.bar(x_pos, means_h, color=bar_colors, zorder=3, **bar_kwargs)
    for i, v in enumerate(means_h):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    # In-panel annotation at the top of the panel — same vertical position
    # as the legend annotations on the reverse panel, so the two panels
    # read as visually parallel.
    ax.text(0.02, 0.96, "pre-registered upper bound (0.30)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9, color=RED_X,
            bbox=dict(facecolor="white", edgecolor="none", pad=2, alpha=0.9))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("H-flip", loc="left", fontsize=12)
    ax.grid(axis="y", linestyle="-", color=GRAY_LIGHT, alpha=0.10, zorder=0)

    fig.suptitle("Calibration probes — identity / reverse / h-flip vs pre-registration",
                 fontsize=13, y=1.02, x=0.02, ha="left")

    # H2 M1-withdrawn footnote (carried over)
    fig.text(0.02, -0.03,
             "Reverse: H2 M2 component confirmed (mean ≈ 1.035 across all four models). "
             "H2 M1 component withdrawn — M1 is ill-defined under time-axis perturbations "
             "(see RESEARCH.md §3.1 Validity scope).",
             fontsize=9, color=RULE, ha="left")

    plt.tight_layout()
    save_three(fig, "fig3_calibration")
    plt.close(fig)
    print(f"[fig3] identity max |M2| = {abs_max:.2e}; "
          f"reverse means = {[round(v,3) for v in means_r]}; "
          f"hflip means = {[round(v,3) for v in means_h]}")


# =====================================================================
# Figure 2 — scaling (zoomed y-axis, consistent ✗ offsets, tightened title)
# =====================================================================
def render_fig2() -> None:
    rng = np.random.default_rng(20260506)

    # Per-perturbation per-model mean M2 with bootstrap CIs, averaged
    # across all 10 strengths (n=2000 per cell).
    rows = []
    for pert in TIER1:
        for model in MODEL_ORDER:
            vals = t1.loc[(t1.perturbation == pert) & (t1.model == model), "m2"].to_numpy()
            m, lo, hi = boot_ci_mean(vals, rng)
            rows.append(dict(perturbation=pert, model=model,
                             params_M=MODEL_PARAMS_M[model],
                             mean_m2=m, ci_lo=lo, ci_hi=hi))
    sf = pd.DataFrame(rows)

    # Detect non-monotonic transitions: any step where M2 increases as model
    # size grows. Same logic as the H6 panel.
    nonmono: list[tuple[str, str, str]] = []
    for pert in TIER1:
        sub = sf[sf.perturbation == pert].sort_values("params_M").reset_index(drop=True)
        for i in range(len(sub) - 1):
            if sub.loc[i + 1, "mean_m2"] > sub.loc[i, "mean_m2"]:
                nonmono.append((pert, sub.loc[i, "model"], sub.loc[i + 1, "model"]))

    # ---- Plot ----
    # Tall aspect ratio so the y-axis has room to breathe; occlusion's
    # ~0.02 M2 spread reads as a real (small) variation rather than a flat
    # line at the data zero, while still keeping all 5 perturbations
    # visible (gaussian_noise tops out near 0.50).
    fig, ax = plt.subplots(figsize=(10.0, 6.8))

    # Color palette: deep teal for the time-axis pair (frame_drops, occlusion),
    # viridis-style for the others. Reading order top-to-bottom matches the
    # post's framing. Use perceptually distinct shades.
    PERT_COLORS = {
        "frame_drops":    "#1f6f6f",   # teal (time-axis)
        "occlusion":      "#5fa5a5",   # lighter teal (time-axis)
        "motion_blur":    "#7c3aed",   # violet
        "low_light":      "#ea580c",   # orange
        "gaussian_noise": "#0369a1",   # blue
    }

    label_offsets = {
        "frame_drops": 0,
        "occlusion": 0,
        "motion_blur": 0,
        "low_light": 0,
        "gaussian_noise": 0,
    }

    for pert in TIER1:
        sub = sf[sf.perturbation == pert].sort_values("params_M").reset_index(drop=True)
        xs = sub["params_M"].values
        ys = sub["mean_m2"].values
        los = sub["ci_lo"].values
        his = sub["ci_hi"].values
        yerr = np.array([ys - los, his - ys])
        color = PERT_COLORS[pert]
        ax.errorbar(xs, ys, yerr=yerr, color=color, marker="o", markersize=6,
                    linewidth=1.8, capsize=3, zorder=3, label=PERTURB_LABEL[pert])
        # Right-edge label
        ax.annotate(PERTURB_LABEL[pert], xy=(xs[-1], ys[-1] + label_offsets[pert]),
                    xytext=(10, 0), textcoords="offset points",
                    fontsize=11, color=color, va="center", fontweight="medium")

    # Non-monotonic markers — each ✗ sits at a fixed display-pixel offset
    # ABOVE the higher endpoint of its transition. Anchoring to the higher
    # endpoint (rather than the midpoint) produces visually consistent
    # marker heights regardless of how steeply each segment slopes.
    Y_OFFSET_DISP = 14  # pixels above the higher endpoint
    for pert, a_model, b_model in nonmono:
        sub = sf[sf.perturbation == pert].sort_values("params_M").reset_index(drop=True)
        ra = sub[sub.model == a_model].iloc[0]
        rb = sub[sub.model == b_model].iloc[0]
        # X position: geometric mean of the two model x-positions (visual
        # midpoint on a log axis).
        mid_x = (ra.params_M * rb.params_M) ** 0.5
        # Y anchor: the higher of the two endpoints (the post-regression
        # value, which is the "wrong" direction we're flagging).
        higher_y = max(ra.mean_m2, rb.mean_m2)
        x_disp, y_disp = ax.transData.transform((mid_x, higher_y))
        x_data, y_data = ax.transData.inverted().transform((x_disp, y_disp + Y_OFFSET_DISP))
        ax.scatter([x_data], [y_data], s=110, marker="x", color=RED_X,
                   zorder=5, linewidths=2.5)

    ax.set_xscale("log")
    ax.set_xticks([80, 300, 1000, 2000])
    ax.set_xticklabels(["base\n80M", "large\n300M", "giant\n1B", "gigantic\n2B"])
    ax.set_xlabel("Model size (parameters)")
    ax.set_ylabel("Mean M2 across all strengths (95% bootstrap CI)")

    # Y-range: keep all 5 perturbations visible (gaussian_noise tops out
    # at ~0.50). Occlusion is flat-ish near 0.12 — handled by inset zoom.
    ax.set_ylim(0.08, 0.55)
    ax.set_xlim(60, 6500)

    # Faint horizontal gridlines only
    ax.grid(axis="y", linestyle="-", color=GRAY_LIGHT, alpha=0.10, zorder=0)

    ax.set_title("Scaling: M2 vs model size — every Tier-1 perturbation is non-monotonic",
                 fontsize=13, pad=14, loc="left")

    # ✗ marker legend — placed in the lower-right empty region (below
    # occlusion's right endpoint, away from any line). Annotation, not a
    # boxed legend, so it reads as part of the figure rather than a UI
    # element.
    # Use a Line2D proxy in a frameless legend so the marker is rendered
    # by matplotlib (correct glyph for the chosen font) and the label
    # text doesn't have to embed unicode crosses.
    from matplotlib.lines import Line2D
    proxy = Line2D([0], [0], marker="x", color=RED_X, markersize=11,
                   markeredgewidth=2.5, linestyle="None",
                   label="non-monotonic transition\n(giant or gigantic regresses\nbelow the prior model)")
    leg = ax.legend(handles=[proxy], loc="upper left", frameon=True,
                    fontsize=10, handletextpad=0.6, borderpad=0.8,
                    bbox_to_anchor=(0.005, 0.995))
    leg.get_frame().set_edgecolor(GRAY_LIGHT)
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.95)

    plt.tight_layout()
    save_three(fig, "fig2_scaling")
    plt.close(fig)
    print(f"[fig2] non-monotonic transitions: {nonmono}")
    print(f"[fig2] y-range used: 0.08 – 0.55")


if __name__ == "__main__":
    render_fig5()
    render_fig3()
    render_fig2()
    print("done. wrote {png,pdf,svg} for fig2_scaling, fig3_calibration, fig5_partitioning.")
