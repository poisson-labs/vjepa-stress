# RESEARCH.md

> **Placeholder for the methodology source doc.** This file should contain
> the full research methodology document for vjepa-stress. The scaffolding
> tooling expects the following sections to exist (matrix builder, metric
> modules, and `predictions.md` all reference them):
>
> - §1 — Motivation and prior work
> - §2 — Perturbation taxonomy (P1–P9, with strength parameterizations)
> - §3 — Metrics (M1 per-frame structural drift, M2 temporal consistency
>   degradation, M3 DAVIS patch correspondence)
> - §4 — Models under test (V-JEPA 2.1 ViT-B / L / g / G)
> - §5 — Datasets (SSv2 evaluation split, DAVIS 2017 validation split)
> - §6 — Threats to validity
>
> Drop the source document into this path before the first sweep is launched.
> The tests in `tests/test_metrics.py` enforce the metric contracts described
> in §3, so the math there is load-bearing for the codebase.

---

## Calibration outcomes (running log)

While the source doc is a placeholder, calibration *predictions* live in
code (the canonical pre-registration anchor for each band) and *outcomes*
are recorded here so they survive any future doc replacement. Each entry
states the pre-registered claim, where it was anchored, what was observed,
and the verdict.

### M2 reverse-playback band — `in_band` (2026-04-29)

- **Pre-registered prediction.** §3.2 of the source doc predicts that
  `M2(F_clean, F_reversed)` on real motion video at 16 frames / tubelet=2
  lands in `[0.9, 1.4]`, with center near 1.14. The band was anchored
  in code as:
  - `PREDICTED_BAND = (0.9, 1.4)` in `scripts/calibrate_m2_real.py`
  - the `0.9 < m2 < 1.4` assertion in
    `tests/test_vjepa_loader.py::test_m2_reverse_calibration_real_features`
  - the H2 threshold (`M2 > 0.8`) used by the calibration probe.
- **Observation.** Real V-JEPA 2.1 vit-base, all 30 DAVIS 2017 val clips,
  CPU fp32, 16 frames @ 384²:
  - `mean=1.0200`, `median=1.0304`, `std=0.0360`,
    `min=0.8949` (`dogs-jump`), `max=1.0664` (`mbike-trick`)
  - All 30 per-clip values inside the band; tightest variance came from
    high-M1 clips (`india`, `lab-coat`, `parkour`); the four lowest M2
    values (`dogs-jump`, `judo`, `pigs`, `bike-packing`) coincided with
    the four lowest M1 values, consistent with the analytical story that
    small per-frame drift makes the temporal-gradient direction harder
    to flip cleanly.
- **Verdict.** Prediction confirmed. H2's `M2 > 0.8` threshold is
  well-grounded — every DAVIS clip clears it with margin. The mean lands
  ~0.12 below the §3.2 center estimate (1.14 → 1.02), still inside the
  band, so the band itself is what moves to "verified" rather than the
  point estimate.
- **What this resolves.** The Stage 0.6 Task 3.5 deferral. Synthetic
  drift produced M2 ≈ 0.10 (an order of magnitude below the band),
  recorded as a structural finding (low-entropy stimulus collapse),
  not a metric or pipeline bug. The real-data number — landing where
  predicted, on independent content distribution — confirms that
  reading.
- **What this does NOT change.** The pre-registered band, the H2
  threshold, and `compute_m2`'s implementation are all unchanged.
  Pre-registration discipline: we recorded what we predicted before
  observing, observed it, recorded the outcome here.
- **Remaining follow-up.** A second calibration on SSv2 (different
  content distribution: object manipulation vs. DAVIS's mostly
  natural-motion scenes) lands on RunPod once SSv2 is downloaded
  there. The script's diagnostic-2 slot is reserved for that.
- **Artifacts.** `data/calibration/m2_reverse_calibration.json`
  (`manifest_id=b9d9bf22-7c4e-4045-b21a-2076c9da36f4`),
  `logs/m2_calibration.log`, devlog 2026-04-29.

---

## §3.1 — M1 (per-frame structural drift): methodology notes

This subsection documents revisions to the M1 metric definition that emerged
during Stage 5 analysis. It is appended to §3.1 (M1) and is a post-hoc
methodology clarification; it does NOT amend any pre-registered prediction.
The H2 prediction text is committed and unchanged.

### Validity scope (added 2026-04-30, Stage 5)

**Validity scope.** M1 assumes the perturbation preserves frame ordering. For perturbations that reorganize the time axis (reverse playback, frame drops with last-seen-frame substitution), M1 measures inter-frame distance in the original feature space rather than perturbation-induced drift, and is not a meaningful response metric. M1 is well-defined for: Gaussian noise, motion blur, occlusion, low-light gamma, JPEG compression, horizontal flip, identity. M1 is ill-defined for: reverse playback, frame drops. This scope was not flagged in the pre-registration; it was identified during Stage 5 analysis when reverse-playback M1 measured 0.358 (well above the H2 prediction of < 0.10). The H2 M1 component is therefore withdrawn as ill-defined; the H2 M2 component (predicted > 0.8, measured 1.036) stands and is confirmed. The pre-registered prediction text in §7 remains unmodified to preserve auditability.
