# Pre-Registered Predictions

> **Pre-registration is binding.** This file is the canonical pre-registration
> artifact for the vjepa-stress study. Predictions here cannot be revised in
> light of results; if a prediction is later found to be wrong, that is a
> *finding*, not a reason to edit the file.

## Status

- **Recorded on**: 2026-04-29 (retrospective consolidation of H1–H4, H6);
  **H5 registered 2026-04-30** ahead of the DAVIS / M3 follow-up sweep.
- **SSv2 sweep launched on**: 2026-04-29 evening; completed 2026-04-30
  morning. Canonical artifact: `runs/sweep-b077627/`
  (`code_commit=b077627fd5920cc3709a1a087f008cac324e6da2`,
  `manifest_id=3fa7d072-f34b-4e48-a8a3-713db55a292a`,
  `schema_version=1.1`).
- **DAVIS sweep launched on**: TBD (this commit registers H5 ahead of it;
  the sweep's `code_commit` will be appended to this file as a follow-up
  Status note when it lands, but no prediction text below will be edited).
- **Manifest hash at pre-registration**:
  `runs/sweep-b077627/manifest.json` → `e118195d68ab4c3af0facf7e72d91e72cae9f345f8114b293d2825b6de5a98b9`
  (sha256). This binds H1–H4 and H6 to the SSv2 sweep against which
  they were evaluated.

### Provenance footnote

H1–H4 and H6 were drafted in research-planning documents (`RESEARCH.md`
draft, working chats, Notion) during 2026-04-27 to 2026-04-28. The SSv2
sweep ran on 2026-04-29 → 2026-04-30 against `code_commit=b077627`,
which embedded those predictions in its calibration probes (Tier 2
identity / reverse / horizontal_flip cells) and in the strength sweeps
they reference. **The canonical `predictions.md` artifact is being
created retrospectively on 2026-04-29 to consolidate them as a single
pre-registration record.** The Stage 5 verdicts in `devlog.md`
(2026-04-29 entry, sub-heading "Hypothesis verdicts") were written
*after* this consolidation and reference these texts verbatim.

H5 has *not* been run against any compute. It is being registered
**before** the DAVIS / M3 follow-up sweep, which is the cleanest
available pre-registration moment in this project. No M3 numbers from
DAVIS exist yet; only the M2-on-DAVIS calibration outcome
(`mean=1.020`, `RESEARCH.md` §3.2) is on the record.

What's not defensible — and what this file is fixing — is claiming
`predictions.md` was committed before the SSv2 sweep when it wasn't.
This file commits both consolidations honestly, with their actual
recording dates.

---

## Predictions

### H1: Identity probe yields zero metric response

- **Metric**: M1, M2
- **Models**: All four (vit-base, vit-large, vit-giant, vit-gigantic)
- **Perturbation**: `identity` at strength 1.0 (Tier 2 calibration probe)
- **Direction / threshold**: `max(|M1|) < 1e-5` AND `max(|M2|) < 1e-5`
  across every video in every model. Float-epsilon clean — anything
  larger means the encoder pair is non-deterministic, which would
  invalidate every other row.
- **Mechanism**: Identity returns the input frame array unchanged. The
  encoder is run twice (clean and "perturbed") on byte-identical
  tensors; with `eval()` mode and no dropout, output should be
  bit-identical → metric = 0.
- **Disconfirmation**: Any model exhibits `|M1| ≥ 1e-5` or `|M2| ≥ 1e-5`.
  This would not be a robustness finding; it would be a pipeline bug.

### H2: Reverse playback produces large M2, small M1

- **Metric**: M1, M2
- **Models**: All four
- **Perturbation**: `reverse` at strength 1.0 (Tier 2 calibration probe)
- **Direction / threshold**: Mean `M2 > 0.8` AND mean `M1 < 0.10`,
  averaged across SSv2 videos within each model. M2 prediction has a
  pre-registered band of `[0.9, 1.4]` from `RESEARCH.md` §3.2; the H2
  threshold is the looser gate.
- **Mechanism**: Reverse playback inverts the *temporal-gradient*
  direction the encoder uses to build temporal features → M2 (which is
  defined on temporal differences) should be large. Per-frame features
  are roughly invariant to playback direction (still the same
  per-frame content) → M1 should be small.
- **Disconfirmation**: Mean `M2 ≤ 0.8` (encoder is insensitive to
  temporal direction) OR mean `M1 ≥ 0.10` (per-frame features depend on
  playback direction in a way M1 picks up).

### H3: Horizontal flip is a hard perturbation

- **Metric**: M1, M2
- **Models**: All four
- **Perturbation**: `horizontal_flip` at strength 1.0 (Tier 2 probe)
- **Direction / threshold**: Mean `M2 < 0.30` AND mean `M1 > 0.05`,
  averaged across SSv2 videos within each model.
- **Mechanism**: V-JEPA is not flip-augmented; flip changes per-pixel
  content (left becomes right) → M1 picks up the per-frame change.
  Temporal-gradient direction is preserved → M2 stays small.
- **Disconfirmation**: Mean `M2 ≥ 0.30` (V-JEPA features turn out to
  be flip-invariant — would be a *finding*, not a bug) OR mean
  `M1 ≤ 0.05` (per-frame features don't pick up the mirror).

### H4: Frame drops degrade M2 more than Gaussian noise

- **Metric**: M2
- **Models**: All four (averaged)
- **Perturbations**: `frame_drops` and `gaussian_noise`, both at
  strength 0.5
- **Direction / threshold**: `mean(M2 | frame_drops, s=0.5) >
  mean(M2 | gaussian_noise, s=0.5)`. Strict inequality.
- **Mechanism**: M2 is sensitive to temporal structure. Dropping
  frames removes temporal information directly; Gaussian noise mostly
  perturbs per-pixel signal but leaves frame ordering intact. Frame
  drops should bite M2 harder.
- **Disconfirmation**: The reverse direction (`frame_drops < gaussian_noise`)
  or equality. Refutation has *direction* implications: it would mean
  spatial noise reaches representational integrity faster than temporal
  ablation — informative either way.

### H5: ΔM3 correlates with M2 across the Tier 1 grid

- **Metric**: M3 (computed against DAVIS), M2 (computed against the
  same DAVIS clip)
- **Models**: All four
- **Perturbations**: Tier 1 set (`gaussian_noise`, `motion_blur`,
  `frame_drops`, `occlusion`, `low_light`)
- **Strengths**: 5 strengths `s ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`
  (DAVIS-side; SSv2 side uses 10 strengths but H5 is computed on the
  DAVIS rows)
- **Clips**: 30 DAVIS 2017 validation sequences
- **Quantity**: Pearson correlation `r` between `ΔM3` and `M2` across
  all `(model, perturbation, strength, clip)` tuples in the DAVIS
  sweep. `ΔM3 = M3_clean − M3_perturbed`, where `M3_clean` is the
  patch-correspondence accuracy on the unperturbed clip and
  `M3_perturbed` is the same on the perturbed clip. `M2` is the row's
  M2 value.
- **Direction / threshold**:
  - `r > 0.5` → **confirmed**
  - `0.3 ≤ r ≤ 0.5` → **ambiguous** (exploratory framing in the post)
  - `r < 0.3` → **refuted** (the post reframes around methodology
    rather than predictive validity of M2)
- **Mechanism**: If M2 measures representational drift that translates
  to downstream task failure, the correlation should be substantial.
  `r > 0.5` is the bar for "M2 is a meaningful proxy for downstream
  consequence"; below that, M2 is interesting but not predictive of
  action.
- **Disconfirmation**: `r < 0.5` against the DAVIS Tier 1 grid as
  defined above. The split between *ambiguous* and *refuted* exists
  because correlation strength is continuous and the post needs a
  defensible position for the borderline.

### H6: Robustness is non-monotonic in model size on at least one Tier 1 perturbation

- **Metric**: M2 (averaged across SSv2 videos and strengths within a
  given (model, perturbation) cell)
- **Models**: All four, comparison across the size order
  base → large → giant → gigantic
- **Perturbations**: Tier 1 set
- **Direction / threshold**: For at least one Tier 1 perturbation,
  there exists a consecutive (smaller, larger) model pair where
  `M2(smaller) < M2(larger)` — i.e. the larger model is *less robust*
  on that perturbation than the next smaller one. "Non-monotonic"
  means the M2-vs-size curve is not strictly monotone-decreasing.
- **Mechanism**: Larger V-JEPA variants are *not* trained with extra
  data augmentation relative to smaller ones; the family is identical
  except for parameter count. There is no a-priori reason a 2B model
  should be more robust to JPEG artifacts than a 1B model on the same
  pretraining data. Hypothesis: scaling doesn't monotonically deliver
  robustness for free.
- **Disconfirmation**: Every Tier 1 perturbation shows strictly
  monotone-improving M2 with size. Confirmation: at least one shows a
  reversal. (The "stronger than predicted" tier — *all five*
  perturbations non-monotonic — is not the bar; one is sufficient.)

---

## Notes on metric scope (post-hoc, non-binding)

A Stage 5 methodology revision (`RESEARCH.md` §3.1, dated 2026-04-30,
*after* the SSv2 sweep) clarified that **M1 is ill-defined for
perturbations that reorganize the time axis** (`reverse`,
`frame_drops`). The original predictions in H2 and H4 above are
*unmodified* — the pre-registered text is what it is. The Stage 5
verdict in `devlog.md` records:

- **H2 M1 component**: withdrawn as ill-defined (M1 measured 0.358 ≫
  0.10 threshold, but the metric is not measuring what the prediction
  thought it was measuring).
- **H2 M2 component**: confirmed (mean M2 = 1.036, well above 0.8).

This note is appended for transparency and does *not* edit the
binding text above.
