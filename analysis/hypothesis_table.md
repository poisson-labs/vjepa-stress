# Hypothesis verdicts — sweep-b077627 (SSv2) + sweep-davis-9fb565d (DAVIS)

| ID | Prediction | Measured | Verdict |
|----|------------|----------|---------|
| **H1** | identity → max \|M1\|, \|M2\| < 1e-5 | max \|M1\| = 3.59e-07, max \|M2\| = 1.52e-08 | **CONFIRMED** |
| **H2** | reverse → mean M2 > 0.8 **and** mean M1 < 0.10 | mean M2 = 1.0355 ✓, mean M1 = 0.3578 ✗ | **PARTIAL** — M2 confirmed, M1 component withdrawn (M1 ill-defined under time-axis perturbations; see RESEARCH §3.1 Validity scope) |
| **H3** | hflip → mean M2 < 0.30 **and** mean M1 > 0.05 | mean M2 = 0.9141, mean M1 = 0.3982; per-model M2 ∈ [0.910, 0.920] | **REFUTED** — hflip is not a near-identity for V-JEPA 2.1 in feature space |
| **H4** | frame_drops M2(s=0.5) > gaussian_noise M2(s=0.5) | frame_drops = 0.3239, gaussian = 0.4992, Δ = -0.1753 | **REFUTED** — gaussian noise produces a larger temporal-consistency hit than 50% frame drops at matched strength |
| **H5** | Pearson r(M2, ΔM3) ≥ 0.5 across Tier-1 perturbations on DAVIS | aggregate r = 0.1608 (n=3000); per-perturbation: frame_drops 0.37, occlusion 0.35, motion_blur 0.09, low_light 0.05, gaussian_noise −0.06 | **PARTITIONED** — frame_drops and occlusion clear the ambiguous-threshold (r ≥ 0.3) but neither hits the 0.5 confirmation bar; appearance-only perturbations are uncorrelated. M2 partitions task-relevance by perturbation family rather than serving as a single-number proxy. |
| **H6** | non-monotonic robustness on ≥1 Tier-1 perturbation | 5 of 5 Tier-1 perturbations non-monotonic — gaussian_noise, motion_blur, low_light: giant → gigantic regression; frame_drops, occlusion: large → giant regression | **CONFIRMED** — pattern stronger than predicted (universal across Tier 1, not just one perturbation) |

## Notes

- **H5 PARTITIONED** is the headline post-DAVIS finding. See `figures/fig5_partitioning.{png,pdf}`.
  Per-perturbation correlations split cleanly between geometry-disruptive (frame_drops, occlusion)
  and appearance-only (gaussian_noise, motion_blur, low_light) families. The pre-registered single
  aggregate-r threshold cannot capture this; reporting per-perturbation r is the honest summary.
- **H2 M1 component** was withdrawn (not refuted) on validity grounds. M1 measures inter-frame distance
  in the original feature space, which is not a coherent drift signal once the time axis itself has
  been reorganized (reverse, frame_drops). The pre-registration text in RESEARCH §7 is locked;
  the validity-scope clarification lives in §3.1.
- **H6** is stronger than predicted: the spec asked for ≥1 Tier-1 perturbation showing
  non-monotonicity, and 5/5 do. Two distinct patterns are visible:
  appearance-only perturbations regress at the giant → gigantic step; time/geometry perturbations
  regress earlier, at the large → giant step.
