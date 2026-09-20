# Blog-post numbers — final extraction

Pure-extraction reference for the blog draft. All numbers computed from
`runs/sweep-b077627/` (SSv2, schema 1.1) and `runs/sweep-davis-9fb565d/`
(DAVIS, schema 1.2), plus `data/calibration/m2_reverse_calibration.json`
(DAVIS reverse calibration probe, vit-base only). Bootstrap CIs use 1000
resamples, seed=20260506. Pearson r CIs use Fisher z. All values to 3 dp
unless noted.

Canonical references (do not duplicate prose):
- `analysis/hypothesis_table.md` — H1–H6 verdict text
- `analysis/compute_economics.md` — total $4.29 / per-model breakdown
- `RESEARCH.md §3.1` — M1 validity-scope clarification

## Section 1 — Per-perturbation r with 95% CIs (Finding 1)

DAVIS sweep, ΔM3 = m3_clean − m3_perturbed, within-perturbation Pearson r,
Fisher-z CI, n ≈ 600 per perturbation (30 clips × 4 models × 5 strengths).

| Perturbation | n | r | 95% CI |
|---|---|---|---|
| frame_drops | 600 | +0.370 | [+0.299, +0.437] |
| occlusion | 600 | +0.350 | [+0.278, +0.418] |
| motion_blur | 600 | +0.093 | [+0.013, +0.171] |
| low_light | 600 | +0.049 | [-0.031, +0.128] |
| gaussian_noise | 600 | -0.055 | [-0.135, +0.025] |

**CI overlap (time-axis vs image-noise families):** **No.** closest pair: `occlusion` lower bound = +0.278 vs `motion_blur` upper bound = +0.171 → gap of +0.106 (positive = disjoint).

## Section 2 — Aggregate r with CI (Finding 1 supporting)

DAVIS sweep, all Tier-1 perturbations pooled, n = 3000.

Aggregate Pearson r = **+0.161**, 95% CI [+0.126, +0.195] (Fisher z).

## Section 3 — Per-cell M2 bootstrap CIs (H6 / Finding 2)

5 Tier-1 perturbations × 4 models × 10 strengths = 200 SSv2 cells, n=200/cell.
Full per-cell table: `analysis/per_cell_m2_ci.csv`.

- **Median CI half-width** across 200 cells: **0.0154**
- **Maximum CI half-width** across 200 cells: **0.0254**

Five H6 non-monotonic jumps (strength-averaged across 10 strengths × 200 clips per cell):

| Perturbation | Step | Smaller M2 ± half-CI | Larger M2 ± half-CI | Jump | Jump 95% CI | Jump significance (× pooled half) |
|---|---|---|---|---|---|---|
| gaussian_noise | giant → gigantic | 0.460 ± 0.0085 | 0.498 ± 0.0076 | +0.038 | [+0.027, +0.049] | +6.67× |
| motion_blur | giant → gigantic | 0.229 ± 0.0062 | 0.280 ± 0.0074 | +0.050 | [+0.040, +0.060] | +10.49× |
| frame_drops | large → giant | 0.333 ± 0.0099 | 0.373 ± 0.0095 | +0.041 | [+0.029, +0.053] | +5.94× |
| occlusion | large → giant | 0.117 ± 0.0039 | 0.134 ± 0.0047 | +0.017 | [+0.011, +0.024] | +5.66× |
| low_light | giant → gigantic | 0.299 ± 0.0091 | 0.334 ± 0.0103 | +0.036 | [+0.022, +0.049] | +5.21× |

**Borderline jumps (< 2× pooled CI half-width):** none — all five jumps are ≥ 2× their pooled CI half-width.

## Section 4 — Finding 4 gap with CIs (s=0.5)

SSv2 sweep, strength = 0.5, all 4 models pooled, n = 800 per perturbation.

Gaussian noise M2 = **0.499**, 95% CI [0.489, 0.509]. Frame drops M2 = **0.324**, 95% CI [0.314, 0.334]. Gap (gaussian − frame_drops) = **+0.175**, bootstrap 95% CI [+0.160, +0.189]. Ratio (gaussian / frame_drops) = **1.541**, bootstrap 95% CI [1.481, 1.598]. Ratio CI excludes 1.0, so the dominance is robust.

**Cross-strength dominance:** Gaussian dominates frame_drops on ≥ 3 of 4 models at **10 of 10 strengths**.

## Section 5 — Reverse-playback calibration

DAVIS calibration probe (separate from main sweep), vit-base, 30 clips, from `data/calibration/m2_reverse_calibration.json`:

- DAVIS reverse mean M2 = **1.020**, 95% CI [1.006, 1.032], n = 30

SSv2 sweep, reverse perturbation, per model (n = 200 each):

| Model | n | mean M2 | 95% CI |
|---|---|---|---|
| base | 200 | 1.035 | [1.027, 1.042] |
| large | 200 | 1.037 | [1.029, 1.045] |
| giant | 200 | 1.034 | [1.027, 1.041] |
| gigantic | 200 | 1.036 | [1.030, 1.043] |

## Section 6 — Horizontal flip (Finding 3)

SSv2 sweep, hflip perturbation, per model (n = 200 each):

| Model | n | mean M2 | 95% CI |
|---|---|---|---|
| base | 200 | 0.910 | [0.904, 0.916] |
| large | 200 | 0.909 | [0.903, 0.915] |
| giant | 200 | 0.920 | [0.913, 0.926] |
| gigantic | 200 | 0.917 | [0.911, 0.922] |

**Pooled across all 4 models** (n = 800): mean M2 = **0.914**, 95% CI [0.911, 0.917].

## Section 7 — Compute economics, exact

- **Total wallclock (canonical, includes DAVIS pod end-to-end):** 8176.7 s = 2.271 h
- **Total cost @ $1.89/hr:** $4.29
- **SSv2 sweep wallclock (row-summed, end-to-end per cell):** 6729.7 s = 1.869 h
- **DAVIS sweep wallclock (row-summed, excludes encoder loads):** 582.4 s = 0.162 h (end-to-end pod wallclock, the canonical figure used in cost: 1,447 s = 0.402 h)
- **Median per-row wallclock (both sweeps pooled, n = 47400):** 154.9 ms
- **p95 per-row wallclock:** 254.2 ms

**Confirms** the canonical $4.29 figure from `analysis/compute_economics.md` and the 2026-04-29 pod tear-down devlog entry (within rounding).
