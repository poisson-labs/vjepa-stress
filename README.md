# V-JEPA 2.1 Robustness Study (`vjepa-stress`)

[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![Poisson Labs Research](https://img.shields.io/badge/Poisson%20Labs-Research-teal.svg)](https://poissonlabs.ai/research/vjepa-2-1-robustness/)

Code, raw data, analysis scripts, and figures backing the Poisson Labs research post:  
**[Where V-JEPA 2.1's Dense Features Hold Up (and Where They Don't)](https://poissonlabs.ai/research/vjepa-2-1-robustness/)**

This study evaluates feature stability across all four released variants of Meta's **V-JEPA 2.1** foundation video model under controlled perceptual stress.

---

## What the Study Measured

We pre-registered a frozen-encoder robustness study on V-JEPA 2.1 (released March 2026) across all four released model sizes:
- **ViT-base** (80M parameters)
- **ViT-large** (300M parameters)
- **ViT-giant** (1B parameters)
- **ViT-gigantic** (2B parameters)

The experiment evaluated **322 distinct cells** across two sweep tiers tracked in `runs/`:
1. **Primary Sweep (`runs/sweep-b077627/`, 222 Parquet shards):** 200 validation clips from Something-Something V2 (SSv2) across 5 Tier-1 perturbations (`gaussian_noise`, `motion_blur`, `frame_drops`, `occlusion`, `low_light`) at 10 strength levels (s = 0.1 to 1.0), plus calibration probes (`identity`, `reverse`, `hflip`).
2. **Downstream Task Sweep (`runs/sweep-davis-9fb565d/`, 100 Parquet shards):** 30 sequences from the DAVIS 2017 video object segmentation benchmark across 4 model sizes and 5 perturbations at 5 strengths, measuring patch correspondence survival under corruption.

### Metrics
- **M1 (Feature Drift):** Mean cosine distance between clean and perturbed spatial-temporal token embeddings at identical indices.
- **M2 (Temporal Consistency Ratio):** Ratio of feature displacement over time under perturbation relative to clean baseline. An M2 near 0 indicates temporal stability; values > 0.30 indicate temporal breakdown.
- **M3 (Downstream Task Correspondence):** Nearest-neighbour patch correspondence tracking accuracy (IoU against ground-truth segmentation masks on DAVIS). $\Delta\text{M3} = \text{M3}_{\text{clean}} - \text{M3}_{\text{perturbed}}$.

---

## What It Found

Every figure and measurement reported below is cited directly from committed analysis artifacts:

1. **Dense Features Are Partitioned by Perturbation Family**
   - Under temporal disruptions, dense feature instability strongly predicts downstream task failure:
     - `frame_drops`: $r = +0.370$ (95% CI $[+0.299, +0.437]$, $n = 600$) [*source:* [`analysis/post_numbers.md §1`](analysis/post_numbers.md#section-1--per-perturbation-r-with-95-cis-finding-1)]
     - `occlusion`: $r = +0.350$ (95% CI $[+0.278, +0.418]$, $n = 600$) [*source:* [`analysis/post_numbers.md §1`](analysis/post_numbers.md#section-1--per-perturbation-r-with-95-cis-finding-1)]
   - Under appearance / image-noise corruptions, dense feature drift does not predict downstream failure:
     - `motion_blur`: $r = +0.093$ (95% CI $[+0.013, +0.171]$, $n = 600$) [*source:* [`analysis/post_numbers.md §1`](analysis/post_numbers.md#section-1--per-perturbation-r-with-95-cis-finding-1)]
     - `low_light`: $r = +0.049$ (95% CI $[-0.031, +0.128]$, $n = 600$) [*source:* [`analysis/post_numbers.md §1`](analysis/post_numbers.md#section-1--per-perturbation-r-with-95-cis-finding-1)]
     - `gaussian_noise`: $r = -0.055$ (95% CI $[-0.135, +0.025]$, $n = 600$) [*source:* [`analysis/post_numbers.md §1`](analysis/post_numbers.md#section-1--per-perturbation-r-with-95-cis-finding-1)]
   - The 95% confidence intervals between the time-axis family and the image-noise family are disjoint (gap of $+0.106$ between `occlusion` lower bound $+0.278$ and `motion_blur` upper bound $+0.171$) [*source:* [`analysis/post_numbers.md §1`](analysis/post_numbers.md#section-1--per-perturbation-r-with-95-cis-finding-1)].
   - Aggregate Pearson correlation across all pooled Tier-1 cells is $r = +0.161$ (95% CI $[+0.126, +0.195]$, $n = 3000$) [*source:* [`analysis/post_numbers.md §2`](analysis/post_numbers.md#section-2--aggregate-r-with-ci-finding-1-supporting)].
   - *Primary artifact:* [`figures/fig5_partitioning.svg`](figures/fig5_partitioning.svg) ([PDF](figures/fig5_partitioning.pdf)).

2. **Scaling Is Non-Monotonic Across All Five Perturbations**
   - Across 200 SSv2 cells (median bootstrap CI half-width of $0.0154$, max half-width $0.0254$), model size does not monotonically confer robustness [*source:* [`analysis/post_numbers.md §3`](analysis/post_numbers.md#section-3--per-cell-m2-bootstrap-cis-h6--finding-2), [`analysis/per_cell_m2_ci.csv`](analysis/per_cell_m2_ci.csv)]:
     - `gaussian_noise`: giant ($0.460 \pm 0.0085$) $\rightarrow$ gigantic ($0.498 \pm 0.0076$), jump $= +0.038$ (95% CI $[+0.027, +0.049]$, $+6.67\times$ pooled half-width)
     - `motion_blur`: giant ($0.229 \pm 0.0062$) $\rightarrow$ gigantic ($0.280 \pm 0.0074$), jump $= +0.050$ (95% CI $[+0.040, +0.060]$, $+10.49\times$ pooled half-width)
     - `low_light`: giant ($0.299 \pm 0.0091$) $\rightarrow$ gigantic ($0.334 \pm 0.0103$), jump $= +0.036$ (95% CI $[+0.022, +0.049]$, $+5.21\times$ pooled half-width)
     - `frame_drops`: large ($0.333 \pm 0.0099$) $\rightarrow$ giant ($0.373 \pm 0.0095$), jump $= +0.041$ (95% CI $[+0.029, +0.053]$, $+5.94\times$ pooled half-width)
     - `occlusion`: large ($0.117 \pm 0.0039$) $\rightarrow$ giant ($0.134 \pm 0.0047$), jump $= +0.017$ (95% CI $[+0.011, +0.024]$, $+5.66\times$ pooled half-width)
   - On three of five perturbations (`gaussian_noise`, `motion_blur`, `low_light`), the largest 2B model regresses and is less robust than the 1B model [*source:* [`analysis/post_numbers.md §3`](analysis/post_numbers.md#section-3--per-cell-m2-bootstrap-cis-h6--finding-2)].
   - *Primary artifact:* [`figures/fig2_scaling.svg`](figures/fig2_scaling.svg) ([PDF](figures/fig2_scaling.pdf)).

3. **Horizontal Flip Disturbs Features Almost as Much as Playing Backwards**
   - Playing clips backwards yields mean M2 $= 1.020$ on DAVIS (95% CI $[1.006, 1.032]$, $n = 30$) and $\approx 1.035$ on SSv2 [*source:* [`analysis/post_numbers.md §5`](analysis/post_numbers.md#section-5--reverse-playback-calibration)].
   - In contrast, horizontally flipping a clip (which preserves physics and temporal continuity) degrades representation geometry substantially: pooled mean M2 $= 0.914$ (95% CI $[0.911, 0.917]$, $n = 800$) [*source:* [`analysis/post_numbers.md §6`](analysis/post_numbers.md#section-6--horizontal-flip-finding-3)].
   - *Primary artifact:* [`figures/fig3_calibration.svg`](figures/fig3_calibration.svg) ([PDF](figures/fig3_calibration.pdf)).

4. **Image Noise Hits Latent Geometry Harder Than Frame Drops**
   - At strength $s = 0.5$ (pooled across models, $n = 800$), Gaussian noise produces M2 $= 0.499$ (95% CI $[0.489, 0.509]$) versus frame drops M2 $= 0.324$ (95% CI $[0.314, 0.334]$). Gap $= +0.175$ (95% CI $[+0.160, +0.189]$); ratio $= 1.541$ (95% CI $[1.481, 1.598]$) [*source:* [`analysis/post_numbers.md §4`](analysis/post_numbers.md#section-4--finding-4-gap-with-cis-s05)].
   - Gaussian noise dominates frame drops on $\ge 3$ of 4 models across $10 \text{ of } 10$ strength levels [*source:* [`analysis/post_numbers.md §4`](analysis/post_numbers.md#section-4--finding-4-gap-with-cis-s05)].

5. **Compute Economics**
   - Total canonical sweep wallclock time: $8,176.7\text{ s}$ ($2.271\text{ h}$), totaling $\$4.29$ compute cost at $\$1.89/\text{hr}$ on an NVIDIA A100 GPU [*source:* [`analysis/post_numbers.md §7`](analysis/post_numbers.md#section-7--compute-economics-exact), [`analysis/compute_economics.md`](analysis/compute_economics.md)].
   - Median per-row inference time: $154.9\text{ ms}$ (p95: $254.2\text{ ms}$, $n = 47,400$) [*source:* [`analysis/post_numbers.md §7`](analysis/post_numbers.md#section-7--compute-economics-exact)].

---

## What the Study Could Not Detect

To remain scientifically rigorous, the study design acknowledges three specific boundaries:

1. **M1 Metric Breakdown Under Temporal Reorganization:** M1 measures cosine distance at identical temporal indices. When time is reversed or frames are dropped, index-to-index correspondence is destroyed, making M1 uninformative on those axes. This limitation was identified during calibration and documented in [`RESEARCH.md §3.1`](RESEARCH.md#31-temporal-consistency-m2-ratio-of-consecutive-frame-differences).
2. **Latent vs. Task-Policy Generalization:** The study measured representation geometry on frozen backbones and nearest-neighbour patch correspondence on DAVIS. It did not evaluate whether fine-tuning an action decoder or policy head (e.g. V-JEPA-AC) can compensate for frozen encoder degradation in closed-loop robotics simulation.
3. **Severe Codec Bitrate Starvation:** While sensor noise, frame drops, motion blur, and occlusions were swept systematically, extreme compression artifacts (such as macroblocking from sub-50 kbps video streaming) were excluded from Tier-1 evaluations.

---

## How to Reproduce

### 1. Environment Setup

Clone the repository and set up a project-local virtual environment with Python 3.11+:

```bash
git clone https://github.com/poisson-labs/vjepa-stress.git
cd vjepa-stress

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Run test suite sanity checks:
```bash
pytest tests/ -m "not slow and not calibration"
```

### 2. Rebuild Numbers and Figures From Committed Shards

All 322 raw Parquet shards from the sweep are tracked in `runs/sweep-b077627/` and `runs/sweep-davis-9fb565d/`. You do not need GPU access to reproduce the statistical analysis:

```bash
# Extract post statistics, bootstrap CIs, and compute metrics
python scripts/build_post_numbers.py

# Re-render publication figures (outputs to figures/*.{png,pdf,svg})
python scripts/restyle_blog_figures.py
```

### 3. Re-run Analysis Notebook End-to-End

To execute the complete 26-cell analysis pipeline from the tracked data:
```bash
jupyter execute notebooks/analysis.ipynb
```

### 4. Running Sweeps From Scratch on GPU

To execute the full sweep from scratch with PyTorch Hub model downloads:
```bash
# Verify perturbation pipeline on a local sample
python scripts/verify_perturbations.py

# Run the sweep runner (requires GPU and SSv2 / DAVIS data paths configured)
python -m vjepa_stress.run_sweep --help
```
Detailed instructions for remote GPU environments (e.g. RunPod / Lambda) are provided in [`docs/runpod.md`](docs/runpod.md).

---

## Repository Index & Artifacts

- **Standard & Hygiene:** [`docs/repo-hygiene.md`](docs/repo-hygiene.md)
- **Methodology & Hypotheses:** [`RESEARCH.md`](RESEARCH.md), [`EXPERIMENT.md`](EXPERIMENT.md)
- **Canonical Findings & Data:**
  - Blog post numbers & CIs: [`analysis/post_numbers.md`](analysis/post_numbers.md)
  - Per-cell M2 bootstrap CIs (200 rows): [`analysis/per_cell_m2_ci.csv`](analysis/per_cell_m2_ci.csv)
  - Pre-registered hypothesis verdicts: [`analysis/hypothesis_table.md`](analysis/hypothesis_table.md)
  - Compute economics breakdown: [`analysis/compute_economics.md`](analysis/compute_economics.md)
- **Publication Figures:**
  - Figure 1 (Robustness Grid): [`figures/fig1_robustness_grid.pdf`](figures/fig1_robustness_grid.pdf)
  - Figure 2 (Scaling Law & Non-monotonicity): [`figures/fig2_scaling.svg`](figures/fig2_scaling.svg) ([PNG](figures/fig2_scaling.png), [PDF](figures/fig2_scaling.pdf))
  - Figure 3 (Calibration Probes): [`figures/fig3_calibration.svg`](figures/fig3_calibration.svg) ([PNG](figures/fig3_calibration.png), [PDF](figures/fig3_calibration.pdf))
  - Figure 4 (Perturbation Comparison): [`figures/fig4_perturbation_comparison.pdf`](figures/fig4_perturbation_comparison.pdf)
  - Figure 5 (H5 Partitioning): [`figures/fig5_partitioning.svg`](figures/fig5_partitioning.svg) ([PNG](figures/fig5_partitioning.png), [PDF](figures/fig5_partitioning.pdf))
- **Third-Party Licences:** [`NOTICE`](NOTICE), [`LICENSE`](LICENSE)
- **Execution Log:** [`devlog.md`](devlog.md)

---

## Citation

```bibtex
@misc{kolasinski2026vjepastress,
  author = {Kolasinski, Taylor},
  title  = {Where V-JEPA 2.1's Dense Features Hold Up (and Where They Don't)},
  year   = {2026},
  month  = {May},
  howpublished = {\url{https://poissonlabs.ai/research/vjepa-2-1-robustness/}},
  note   = {Poisson Labs research report; source code and data at \url{https://github.com/poisson-labs/vjepa-stress}}
}
```

## Licence

MIT. See [`LICENSE`](LICENSE). Third-party dependencies and model weight terms are documented in [`NOTICE`](NOTICE).
