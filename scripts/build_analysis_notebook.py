"""Build notebooks/analysis.ipynb for Stage 5 from the canonical sweep dir.

This script does NOT contain analysis logic itself; it stitches together the
cell sources and writes a .ipynb. Run it once, then execute the notebook via
nbclient (or `jupyter nbconvert --execute`) to populate outputs.
"""

from __future__ import annotations

import nbformat as nbf
from pathlib import Path


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


CELLS: list[nbf.NotebookNode] = []

CELLS.append(md(
    """# vjepa-stress — Stage 5 Analysis

Loads `runs/sweep-b077627/` (44,400 rows, 222 cells), sanity-checks against
the morning validation, then produces:

- the H1–H4, H6 hypothesis table,
- four primary figures (PNG + PDF),
- case-study candidate CSVs,
- a wallclock / compute summary.

Pre-registration is locked. We report; we do not edit predictions."""
))

CELLS.append(code(
    """import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path

REPO = Path('..').resolve() if Path('..').resolve().name == 'vjepa-stress' else Path('.').resolve()
RUN_DIR = REPO / 'runs' / 'sweep-b077627'
FIG_DIR = REPO / 'figures'
ANL_DIR = REPO / 'analysis'
FIG_DIR.mkdir(exist_ok=True)
ANL_DIR.mkdir(exist_ok=True)

# Canonical orderings used everywhere downstream.
MODEL_ORDER = [
    'vjepa-2.1-vit-base',
    'vjepa-2.1-vit-large',
    'vjepa-2.1-vit-giant',
    'vjepa-2.1-vit-gigantic',
]
MODEL_SHORT = {
    'vjepa-2.1-vit-base': 'base',
    'vjepa-2.1-vit-large': 'large',
    'vjepa-2.1-vit-giant': 'giant',
    'vjepa-2.1-vit-gigantic': 'gigantic',
}
MODEL_PARAMS_M = {  # millions of params; per Meta torch.hub naming convention
    'vjepa-2.1-vit-base': 80,
    'vjepa-2.1-vit-large': 300,
    'vjepa-2.1-vit-giant': 1000,
    'vjepa-2.1-vit-gigantic': 2000,
}
TIER1 = ['gaussian_noise', 'motion_blur', 'frame_drops', 'occlusion', 'low_light']

# Clean, paper-friendly defaults; no gray ggplot background.
sns.set_theme(context='paper', style='whitegrid', font_scale=1.0)
mpl.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
})

# Monochrome-friendly palette for the four model sizes (light → dark).
MODEL_COLORS = {
    'vjepa-2.1-vit-base':     '#9ecae1',
    'vjepa-2.1-vit-large':    '#4292c6',
    'vjepa-2.1-vit-giant':    '#2171b5',
    'vjepa-2.1-vit-gigantic': '#08306b',
}

print('repo:', REPO)
print('run dir:', RUN_DIR, '| exists:', RUN_DIR.exists())"""
))

CELLS.append(md("""## Phase 1 — Load and sanity-check"""))

CELLS.append(code(
    """con = duckdb.connect()
df = con.execute(f"SELECT * FROM '{RUN_DIR}/*.parquet'").df()

# Shape gates
assert len(df) == 44400, f'expected 44400 rows, got {len(df)}'
assert df['cell_id'].nunique() == 222, f'expected 222 cells, got {df.cell_id.nunique()}'
assert sorted(df['model'].unique()) == sorted(MODEL_ORDER), df['model'].unique()
assert df['perturbation'].nunique() == 9, df['perturbation'].nunique()

# Tier counts
tier_counts = df.groupby('tier').agg(n_cells=('cell_id', 'nunique'), n_rows=('cell_id', 'size'))
expected_tier = pd.DataFrame(
    {'n_cells': [200, 12, 10], 'n_rows': [40000, 2400, 2000]},
    index=pd.Index([1, 2, 3], name='tier'),
)
assert tier_counts.equals(expected_tier), f'tier mismatch:\\n{tier_counts}'

# Per-cell row count: must be exactly 200
per_cell = df.groupby('cell_id').size()
assert (per_cell == 200).all(), f'non-200 cells: {per_cell[per_cell != 200].head()}'

print('shape gates: PASS')
print(tier_counts)"""
))

CELLS.append(code(
    """# Identity gate: max |M1|, max |M2| within 1e-5 of zero, per model
identity_gate = (
    df[df['perturbation'] == 'identity']
    .groupby('model')
    .agg(max_abs_m1=('m1', lambda s: s.abs().max()),
         max_abs_m2=('m2', lambda s: s.abs().max()))
    .reindex(MODEL_ORDER)
)
assert (identity_gate['max_abs_m1'] < 1e-5).all(), identity_gate
assert (identity_gate['max_abs_m2'] < 1e-5).all(), identity_gate

# Reverse gate: mean M2 ∈ [0.9, 1.4] per model
reverse_gate = (
    df[df['perturbation'] == 'reverse']
    .groupby('model')['m2']
    .mean()
    .reindex(MODEL_ORDER)
    .rename('mean_m2')
)
assert ((reverse_gate >= 0.9) & (reverse_gate <= 1.4)).all(), reverse_gate

print('identity gate (max |M1|, max |M2|):')
print(identity_gate)
print()
print('reverse gate (mean M2):')
print(reverse_gate)
print()
print('CALIBRATION GATES: PASS')"""
))

CELLS.append(code(
    """# Per-model summary
summary = (
    df.groupby('model')
      .agg(n_cells=('cell_id', 'nunique'),
           n_rows=('cell_id', 'size'),
           mean_wallclock_sec=('wallclock_sec', 'mean'))
      .reindex(MODEL_ORDER)
      .round(4)
)
print(summary)"""
))

CELLS.append(md("""## Phase 2 — Hypothesis table

Each row reports the pre-registered prediction, the measured value across all
4 models, and the verdict per the decision rule in RESEARCH §7.
"""))

CELLS.append(code(
    """rng = np.random.default_rng(20260430)

def boot_mean_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


# H1: identity → max |M1|, max |M2| < 1e-5
h1_max_m1 = df.loc[df.perturbation == 'identity', 'm1'].abs().max()
h1_max_m2 = df.loc[df.perturbation == 'identity', 'm2'].abs().max()
h1_pass = (h1_max_m1 < 1e-5) and (h1_max_m2 < 1e-5)

# H2: reverse → mean M2 > 0.8 AND mean M1 < 0.10
# (Note: per-model means; the strict reading of the rule is satisfied if every
#  model is in the band. Calibration log records mean M1 ≈ 0.27, so we report
#  it honestly: the M2 condition holds, the M1 condition does not.)
h2 = df[df.perturbation == 'reverse']
h2_mean_m2 = h2['m2'].mean()
h2_mean_m1 = h2['m1'].mean()
h2_pass = (h2_mean_m2 > 0.8) and (h2_mean_m1 < 0.10)

# H3: hflip → mean M2 < 0.30 AND mean M1 > 0.05  → known refuted (M2 ≈ 0.91)
h3 = df[df.perturbation == 'horizontal_flip']
h3_mean_m2 = h3['m2'].mean()
h3_mean_m1 = h3['m1'].mean()
h3_m2_per_model = h3.groupby('model')['m2'].mean().reindex(MODEL_ORDER).round(4).to_dict()
h3_pass = (h3_mean_m2 < 0.30) and (h3_mean_m1 > 0.05)

# H4: frame_drops M2 at s=0.5 > gaussian_noise M2 at s=0.5
h4_fd = df[(df.perturbation == 'frame_drops') & (df.strength == 0.5)]['m2']
h4_gn = df[(df.perturbation == 'gaussian_noise') & (df.strength == 0.5)]['m2']
h4_fd_mean = h4_fd.mean()
h4_gn_mean = h4_gn.mean()
h4_pass = h4_fd_mean > h4_gn_mean

# H6: non-monotonic robustness on at least one Tier-1 perturbation.
# Per-perturbation, strength-averaged M2 by model. Monotonic = M2 is non-increasing
# along base → large → giant → gigantic (bigger model ⇒ same or lower drift).
# (Equivalently: bigger models more robust at every step.)
mono_rows = []
nonmono_perturbs = []
for pert in TIER1:
    means = (df[df.perturbation == pert]
             .groupby('model')['m2'].mean()
             .reindex(MODEL_ORDER))
    diffs = means.diff().iloc[1:]  # negative = better with size
    monotonic = (diffs <= 0).all()
    violations = []
    if not monotonic:
        for (a, b), d in zip(zip(MODEL_ORDER[:-1], MODEL_ORDER[1:]), diffs.values):
            if d > 0:
                violations.append(f'{MODEL_SHORT[a]}→{MODEL_SHORT[b]} (+{d:.4f})')
    if not monotonic:
        nonmono_perturbs.append((pert, violations))
    mono_rows.append({
        'perturbation': pert,
        'm2_base': round(means['vjepa-2.1-vit-base'], 4),
        'm2_large': round(means['vjepa-2.1-vit-large'], 4),
        'm2_giant': round(means['vjepa-2.1-vit-giant'], 4),
        'm2_gigantic': round(means['vjepa-2.1-vit-gigantic'], 4),
        'monotonic': monotonic,
        'violations': '; '.join(violations) if violations else '',
    })
h6_pass = len(nonmono_perturbs) > 0

mono_df = pd.DataFrame(mono_rows)
print('Strength-averaged M2 by model and perturbation:')
print(mono_df.to_string(index=False))
print()
print(f'Non-monotonic perturbations: {[p for p, _ in nonmono_perturbs]}')"""
))

CELLS.append(code(
    """hyp_rows = [
    {
        'hypothesis': 'H1',
        'prediction': 'identity → max |M1|, |M2| < 1e-5',
        'measured': f'max |M1|={h1_max_m1:.2e}, max |M2|={h1_max_m2:.2e}',
        'rule_met': h1_pass,
        'verdict': 'CONFIRMED' if h1_pass else 'REFUTED',
    },
    {
        'hypothesis': 'H2',
        'prediction': 'reverse → mean M2 > 0.8 and mean M1 < 0.10',
        'measured': f'mean M2={h2_mean_m2:.4f}, mean M1={h2_mean_m1:.4f}',
        'rule_met': h2_pass,
        'verdict': 'PARTIAL (M2 ✓, M1 ✗)' if (h2_mean_m2 > 0.8 and not h2_pass) else ('CONFIRMED' if h2_pass else 'REFUTED'),
    },
    {
        'hypothesis': 'H3',
        'prediction': 'hflip → mean M2 < 0.30 and mean M1 > 0.05',
        'measured': f'mean M2={h3_mean_m2:.4f}, mean M1={h3_mean_m1:.4f}; per-model M2={h3_m2_per_model}',
        'rule_met': h3_pass,
        'verdict': 'CONFIRMED' if h3_pass else 'REFUTED',
    },
    {
        'hypothesis': 'H4',
        'prediction': 'frame_drops M2(s=0.5) > gaussian_noise M2(s=0.5)',
        'measured': f'frame_drops={h4_fd_mean:.4f}, gaussian={h4_gn_mean:.4f}, Δ={h4_fd_mean - h4_gn_mean:+.4f}',
        'rule_met': h4_pass,
        'verdict': 'CONFIRMED' if h4_pass else 'REFUTED',
    },
    {
        'hypothesis': 'H6',
        'prediction': 'non-monotonic robustness on ≥1 Tier-1 perturbation',
        'measured': (
            f"{len(nonmono_perturbs)} of 5 perturbations non-monotonic: "
            + ('; '.join(f'{p} [' + ', '.join(v) + ']' for p, v in nonmono_perturbs) or 'none')
        ),
        'rule_met': h6_pass,
        'verdict': 'CONFIRMED' if h6_pass else 'REFUTED',
    },
]
hypothesis_table = pd.DataFrame(hyp_rows)
print(hypothesis_table.to_string(index=False))

hypothesis_table.to_csv(ANL_DIR / 'hypothesis_table.csv', index=False)
mono_df.to_csv(ANL_DIR / 'monotonicity_by_perturbation.csv', index=False)"""
))

CELLS.append(md("""## Phase 3 — Primary figures

Y-limits and palettes pinned across panels so visual comparison is honest.
Both PNG (300 DPI) and PDF saved per figure."""))

CELLS.append(code(
    """def bootstrap_ci_arr(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05):
    if len(values) == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def cell_stats(sub_df: pd.DataFrame, value_col: str = 'm2'):
    rows = []
    for (model, pert, s), g in sub_df.groupby(['model', 'perturbation', 'strength']):
        m, lo, hi = bootstrap_ci_arr(g[value_col].to_numpy())
        rows.append(dict(model=model, perturbation=pert, strength=s,
                         mean=m, ci_lo=lo, ci_hi=hi, n=len(g)))
    return pd.DataFrame(rows)

t1_df = df[df.tier == 1].copy()
t1_stats = cell_stats(t1_df, 'm2')
print(t1_stats.head())"""
))

CELLS.append(code(
    """# Figure 1: 4 (rows = perturbation) × 4 (cols = model) → wait, spec says
# "4 columns × 5 rows" = 4 models, 5 perturbations.  Use that orientation.

PERTURB_LABEL = {
    'gaussian_noise': 'Gaussian noise',
    'motion_blur':    'Motion blur',
    'frame_drops':    'Frame drops',
    'occlusion':      'Occlusion',
    'low_light':      'Low-light gamma',
}

fig, axes = plt.subplots(
    nrows=len(TIER1), ncols=len(MODEL_ORDER),
    figsize=(12, 12), sharex=True, sharey=True,
)

YMAX = 1.0  # tier-1 M2 max observed ≈ 0.98; pin at 1.0 for headroom
for r, pert in enumerate(TIER1):
    for c, model in enumerate(MODEL_ORDER):
        ax = axes[r, c]
        sub = t1_stats[(t1_stats.perturbation == pert) & (t1_stats.model == model)].sort_values('strength')
        color = MODEL_COLORS[model]
        ax.fill_between(sub.strength, sub.ci_lo, sub.ci_hi, color=color, alpha=0.25, linewidth=0)
        ax.plot(sub.strength, sub['mean'], color=color, marker='o', markersize=3, linewidth=1.2)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, YMAX)
        ax.tick_params(labelsize=8)
        if r == 0:
            ax.set_title(MODEL_SHORT[model], fontsize=10, fontweight='bold')
        if c == 0:
            ax.set_ylabel(f'{PERTURB_LABEL[pert]}\\nM2', fontsize=9)
        if r == len(TIER1) - 1:
            ax.set_xlabel('strength', fontsize=9)

fig.suptitle('Figure 1 — Tier-1 robustness curves (M2 vs strength, 95% bootstrap CI)',
             fontsize=11, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(FIG_DIR / 'fig1_robustness_grid.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig1_robustness_grid.pdf', bbox_inches='tight')
plt.show()"""
))

CELLS.append(code(
    """# Figure 2 — scaling: log-x param count, M2 averaged across strengths,
# one line per Tier-1 perturbation.

PERT_PALETTE = dict(zip(TIER1, sns.color_palette('viridis', n_colors=len(TIER1))))

fig, ax = plt.subplots(figsize=(7, 5))

scaling_rows = []
for pert in TIER1:
    xs, ys, los, his = [], [], [], []
    for model in MODEL_ORDER:
        vals = t1_df.loc[(t1_df.perturbation == pert) & (t1_df.model == model), 'm2'].to_numpy()
        m, lo, hi = bootstrap_ci_arr(vals)
        xs.append(MODEL_PARAMS_M[model])
        ys.append(m); los.append(lo); his.append(hi)
        scaling_rows.append(dict(perturbation=pert, model=model, params_M=MODEL_PARAMS_M[model],
                                 mean_m2=m, ci_lo=lo, ci_hi=hi))
    color = PERT_PALETTE[pert]
    yerr = np.array([[m - lo for m, lo in zip(ys, los)],
                     [hi - m for m, hi in zip(ys, his)]])
    ax.errorbar(xs, ys, yerr=yerr, color=color, marker='o', linewidth=1.5,
                capsize=3, label=PERTURB_LABEL[pert])
    # Annotate the perturbation directly at the rightmost point
    ax.annotate(PERTURB_LABEL[pert], xy=(xs[-1], ys[-1]),
                xytext=(8, 0), textcoords='offset points', fontsize=9,
                color=color, va='center')

# Mark non-monotonic violations explicitly
for pert, violations in nonmono_perturbs:
    sub_rows = [r for r in scaling_rows if r['perturbation'] == pert]
    sub_rows.sort(key=lambda r: r['params_M'])
    for v in violations:
        # parse "short_a→short_b (+delta)" → mark midpoint
        names = v.split(' (')[0]
        a_short, b_short = names.split('→')
        a_full = next(m for m in MODEL_ORDER if MODEL_SHORT[m] == a_short)
        b_full = next(m for m in MODEL_ORDER if MODEL_SHORT[m] == b_short)
        ra = next(r for r in sub_rows if r['model'] == a_full)
        rb = next(r for r in sub_rows if r['model'] == b_full)
        mid_x = (ra['params_M'] * rb['params_M']) ** 0.5
        mid_y = (ra['mean_m2'] + rb['mean_m2']) / 2
        ax.scatter([mid_x], [mid_y], s=80, marker='x', color='red', zorder=5, linewidths=2)

ax.set_xscale('log')
ax.set_xticks([80, 300, 1000, 2000])
ax.set_xticklabels(['base\\n80M', 'large\\n300M', 'giant\\n1B', 'gigantic\\n2B'], fontsize=9)
ax.set_xlabel('Model size (params)', fontsize=10)
ax.set_ylabel('Mean M2 across all strengths (95% bootstrap CI)', fontsize=10)
ax.set_title('Figure 2 — Scaling: M2 vs model size by perturbation\\n'
             '(red × marks non-monotonic transitions)', fontsize=11)
ax.set_xlim(60, 5000)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig2_scaling.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig2_scaling.pdf', bbox_inches='tight')
plt.show()

scaling_df = pd.DataFrame(scaling_rows)
scaling_df.to_csv(ANL_DIR / 'scaling_table.csv', index=False)"""
))

CELLS.append(code(
    """# Figure 3 — calibration validation panel (3 subplots)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

panel_specs = [
    ('identity',        'm2', 'Identity (M2)',          0.0,  None),
    ('reverse',         'm2', 'Reverse (M2)',           1.14, (0.9, 1.4)),
    ('horizontal_flip', 'm2', 'H-flip (M2)',            0.30, None),  # 0.30 = pre-registered upper bound
]
PANEL_TITLES = ['Identity', 'Reverse', 'H-flip']

x_pos = np.arange(len(MODEL_ORDER))
for ax, (pert, col, label, anchor, band), title in zip(axes, panel_specs, PANEL_TITLES):
    means = (df[df.perturbation == pert].groupby('model')[col].mean()
             .reindex(MODEL_ORDER))
    bars = ax.bar(x_pos, means.values,
                  color=[MODEL_COLORS[m] for m in MODEL_ORDER],
                  edgecolor='#333', linewidth=0.6)
    if band is not None:
        ax.axhspan(band[0], band[1], color='#9bc995', alpha=0.18, zorder=0,
                   label=f'pre-registered band [{band[0]}, {band[1]}]')
    if anchor is not None:
        line_label = (
            'analytical prediction' if pert == 'identity'
            else 'pre-registered upper bound' if pert == 'horizontal_flip'
            else 'pre-registered center (1.14)'
        )
        ax.axhline(anchor, color='#c0392b', linewidth=1.2, linestyle='--', label=line_label)
    for i, v in enumerate(means.values):
        ax.text(i, v, f'{v:.4f}' if pert == 'identity' else f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(label, fontsize=9)
    ax.legend(fontsize=8, loc='upper right' if pert != 'identity' else 'upper left')

# Identity bars are tiny; force a sensible y-axis on that panel
axes[0].set_ylim(-1e-7, 5e-7)
axes[0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[1].set_ylim(0, 1.6)
axes[2].set_ylim(0, 1.0)

fig.suptitle('Figure 3 — Calibration: identity / reverse / h-flip vs pre-registration',
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig3_calibration.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig3_calibration.pdf', bbox_inches='tight')
plt.show()"""
))

CELLS.append(code(
    """# Figure 4 — per-perturbation comparison at s=0.5
S = 0.5
sub = t1_df[t1_df.strength == S].copy()

means = (sub.groupby(['model', 'perturbation'])['m2'].mean()
         .unstack('perturbation').reindex(index=MODEL_ORDER, columns=TIER1))
ci_lo = pd.DataFrame(index=MODEL_ORDER, columns=TIER1, dtype=float)
ci_hi = pd.DataFrame(index=MODEL_ORDER, columns=TIER1, dtype=float)
for model in MODEL_ORDER:
    for pert in TIER1:
        vals = sub.loc[(sub.model == model) & (sub.perturbation == pert), 'm2'].to_numpy()
        m, lo, hi = bootstrap_ci_arr(vals)
        ci_lo.loc[model, pert] = lo
        ci_hi.loc[model, pert] = hi

fig, ax = plt.subplots(figsize=(11, 5))
n_models = len(MODEL_ORDER)
group_w = 0.8
bar_w = group_w / n_models
xs = np.arange(len(TIER1))

for i, model in enumerate(MODEL_ORDER):
    offsets = xs - group_w / 2 + (i + 0.5) * bar_w
    heights = means.loc[model].values
    yerr = np.array([heights - ci_lo.loc[model].values,
                     ci_hi.loc[model].values - heights])
    ax.bar(offsets, heights, bar_w, color=MODEL_COLORS[model],
           edgecolor='#333', linewidth=0.5,
           yerr=yerr, error_kw=dict(ecolor='#333', lw=0.8, capsize=2),
           label=MODEL_SHORT[model])

ax.set_xticks(xs)
ax.set_xticklabels([PERTURB_LABEL[p] for p in TIER1], fontsize=9)
ax.set_ylabel(f'M2 at s={S}', fontsize=10)
ax.set_title('Figure 4 — Per-perturbation M2 at s=0.5 (4 models, 95% bootstrap CI)', fontsize=11)
ax.set_ylim(0, max(0.7, ci_hi.values.max() * 1.1))
ax.legend(title='model', fontsize=9, ncol=4, loc='upper left')

# H4 annotation: frame_drops vs gaussian_noise at s=0.5
gn_idx = TIER1.index('gaussian_noise')
fd_idx = TIER1.index('frame_drops')
gn_h = means['gaussian_noise'].max()
fd_h = means['frame_drops'].max()
top = max(gn_h, fd_h) + 0.08
ax.annotate('', xy=(fd_idx, top), xytext=(gn_idx, top),
            arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.2))
ax.text((gn_idx + fd_idx) / 2, top + 0.01,
        f'H4: frame_drops > gaussian_noise  (Δ={h4_fd_mean - h4_gn_mean:+.3f})',
        ha='center', fontsize=9, color='#c0392b')

fig.tight_layout()
fig.savefig(FIG_DIR / 'fig4_perturbation_comparison.png', dpi=300, bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig4_perturbation_comparison.pdf', bbox_inches='tight')
plt.show()"""
))

CELLS.append(md("""## Phase 4 — Case-study candidates"""))

CELLS.append(code(
    """eps = 1e-6

# Texture-sticking: high M2 / low M1 under motion blur, s ≥ 0.7
mb = df[(df.perturbation == 'motion_blur') & (df.strength >= 0.7)].copy()
mb['ratio'] = mb['m2'] / (mb['m1'] + eps)
texture_sticking = mb.nlargest(5, 'ratio')[['video_id', 'model', 'strength', 'm1', 'm2', 'ratio']]
texture_sticking['failure_mode'] = 'texture_sticking'
texture_sticking['perturbation'] = 'motion_blur'

# Identity-drift: high raw M2 under frame_drops, s ≥ 0.5
fd = df[(df.perturbation == 'frame_drops') & (df.strength >= 0.5)].copy()
identity_drift = fd.nlargest(5, 'm2')[['video_id', 'model', 'strength', 'm1', 'm2']].copy()
identity_drift['ratio'] = float('nan')
identity_drift['failure_mode'] = 'identity_drift'
identity_drift['perturbation'] = 'frame_drops'

# Catastrophic: top-decile in BOTH M1 and M2 within a single Tier-1 perturbation
catas_rows = []
for pert in TIER1:
    pdf = df[df.perturbation == pert]
    m1_q90 = pdf['m1'].quantile(0.90)
    m2_q90 = pdf['m2'].quantile(0.90)
    hits = pdf[(pdf['m1'] >= m1_q90) & (pdf['m2'] >= m2_q90)].copy()
    hits['m1_plus_m2'] = hits['m1'] + hits['m2']
    catas_rows.append(hits.nlargest(5, 'm1_plus_m2')
                       .assign(failure_mode='catastrophic'))
catastrophic = pd.concat(catas_rows, ignore_index=True)
catastrophic = (catastrophic.nlargest(5, 'm1_plus_m2')
                            [['video_id', 'model', 'perturbation', 'strength', 'm1', 'm2', 'failure_mode']])
catastrophic['ratio'] = float('nan')

cols = ['failure_mode', 'video_id', 'model', 'perturbation', 'strength', 'm1', 'm2', 'ratio']
candidates = pd.concat([
    texture_sticking[cols],
    identity_drift[cols],
    catastrophic[cols],
], ignore_index=True)

candidates.to_csv(ANL_DIR / 'case_study_candidates.csv', index=False)
print(candidates.to_string(index=False))"""
))

CELLS.append(md("""## Phase 5 — Wallclock and compute summary"""))

CELLS.append(code(
    """RATE_PER_HR = 1.89  # USD, RTX PRO 6000 community cloud

total_sec = float(df['wallclock_sec'].sum())
total_hr = total_sec / 3600
total_cost = total_hr * RATE_PER_HR

per_model = (df.groupby('model')['wallclock_sec'].sum().reindex(MODEL_ORDER))
per_pert = df.groupby('perturbation')['wallclock_sec'].sum().sort_values(ascending=False)

dist = df['wallclock_sec'].agg(['mean', 'median',
                                 lambda s: s.quantile(0.95),
                                 'max']).rename({'<lambda_0>': 'p95'})
dist.index = ['mean', 'median', 'p95', 'max']

lines = []
lines.append('# Compute summary — sweep-b077627\\n')
lines.append(f'- **Total wallclock**: {total_sec:,.1f} s = {total_hr:.2f} h')
lines.append(f'- **GPU rate**: ${RATE_PER_HR}/hr (RTX PRO 6000, community cloud)')
lines.append(f'- **Estimated cost**: ${total_cost:.2f}\\n')
lines.append('## Per-row wallclock distribution (seconds)\\n')
lines.append('| stat | value |')
lines.append('|------|-------|')
for k, v in dist.items():
    lines.append(f'| {k} | {v:.4f} |')
lines.append('')
lines.append('## Per-model wallclock\\n')
lines.append('| model | total_sec | total_hr | est_cost_usd |')
lines.append('|-------|-----------|----------|--------------|')
for model, sec in per_model.items():
    lines.append(f'| {model} | {sec:,.1f} | {sec/3600:.3f} | ${sec/3600*RATE_PER_HR:.2f} |')
lines.append('')
lines.append('## Per-perturbation wallclock\\n')
lines.append('| perturbation | total_sec | total_hr |')
lines.append('|--------------|-----------|----------|')
for pert, sec in per_pert.items():
    lines.append(f'| {pert} | {sec:,.1f} | {sec/3600:.3f} |')
lines.append('')

(ANL_DIR / 'compute_summary.md').write_text('\\n'.join(lines))
print('\\n'.join(lines))"""
))

CELLS.append(md("""## Done

Verification: hypothesis_table.csv, compute_summary.md, case_study_candidates.csv
written under `analysis/`; four figure pairs under `figures/`. Run dir untouched."""))


nb = nbf.v4.new_notebook(cells=CELLS)
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}
nb.metadata['language_info'] = {'name': 'python', 'version': '3.12'}

OUT = Path(__file__).resolve().parents[1] / 'notebooks' / 'analysis.ipynb'
nbf.write(nb, OUT)
print('wrote', OUT)
