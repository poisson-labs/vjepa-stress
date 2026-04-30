# Case-study verification — `video_id=202588` (low_light, s=1.0)

## Status

**Verification deferred to coding agent.** The local environment has no SSv2
clips on disk (`~/data/ssv2/`, `$VJEPA_STRESS_SSV2_DIR`, and the RunPod path
`/workspace/data/ssv2/` all absent), so the raw frame for video 202588 cannot
be decoded here. Side-by-side PNG renders therefore have not been produced.
Verification is filed for the coding agent in
`analysis/coding_agent_requests.md` (request `R1`).

The substitute path described in the brief (pick a non-low-light candidate
and verify locally) does not unblock — every candidate in
`analysis/case_study_candidates.csv` requires the same SSv2 clip on disk to
render.

## What we can confirm without rendering frames

### 1. Correction to the Stage 5 headline

The Stage 5 summary characterized 202588 as "three of four models collapse
simultaneously". Rechecking the per-clip data:

| model     | M1     | M2     | M2-rank in population (low_light s=1.0) | M1-rank |
|-----------|--------|--------|------------------------------------------|---------|
| base      | 0.348  | 0.981  | **1 / 200**                              | 1 / 200 |
| large     | 0.618  | 0.977  | **1 / 200**                              | 1 / 200 |
| giant     | 0.612  | 0.978  | **1 / 200**                              | 1 / 200 |
| gigantic  | 0.560  | 0.977  | **1 / 200**                              | 1 / 200 |

Video 202588 is the **#1 worst clip on both M1 and M2 for all four models**.
All four collapse, not three. The earlier "three of four" framing came from
the catastrophic-degradation top-5 list, which uses a joint top-decile
filter on M1 *and* M2; vit-base's M1 (0.35) is below the M1 top-decile cut
on this perturbation, so its 202588 row didn't surface in the top-5. The
M2 collapse is identical across model sizes.

This finding is *more* striking than the earlier framing, not less.

### 2. Analytical inspection of the perturbation at s=1.0

Read `vjepa_stress/perturbations/sensor.py` (no edits). At `strength=1.0`
the LowLight perturbation applies, in this order:

- gamma = 2.5 (darken: `f^2.5`)
- gain  = 0.15 (multiplicative)
- additive Gaussian noise on `[0, 1]`, sigma = 0.02

For an originally bright pixel (8-bit value 200, `f = 0.784`):
  `0.784^2.5 × 0.15 ≈ 0.081`, then *255 ≈ **21**.

For a mid pixel (value 100, `f = 0.392`):
  `0.392^2.5 × 0.15 ≈ 0.0145`, then *255 ≈ **4**.

For a dark pixel (value 50, `f = 0.196`):
  `0.196^2.5 × 0.15 ≈ 0.0027`, then *255 ≈ **1**.

So at s=1.0 the perturbation produces a near-black image for *any* clip:
even the brightest highlights survive at ~8% of original luminance, mid
tones effectively go to zero. This is by design (`_LOW_LIGHT_GAMMA_MAX =
2.5` and `_LOW_LIGHT_GAIN_MIN = 0.15` are the s=1 endpoints) — the
perturbation library calls this the "obviously degraded" extreme.

### 3. Population context

At low_light s=1.0, the entire SSv2 sample is hit hard, not just 202588:

| model     | min M2 | mean M2 | median M2 | max M2 |
|-----------|--------|---------|-----------|--------|
| base      | 0.274  | 0.687   | 0.689     | 0.981  |
| large     | 0.231  | 0.640   | 0.639     | 0.977  |
| giant     | 0.287  | 0.620   | 0.611     | 0.978  |
| gigantic  | 0.266  | 0.662   | 0.662     | 0.977  |

So the perturbation analytically erases most of the temporal signal for
the population, and 202588 is the worst-case *within* that already-hit
population. The "all four models top this list with the same clip" finding
is the part worth keeping.

## Verdict

**Provisional: keep `video_id=202588` as the headline case study, pending
visual confirmation.** Three reasons:

1. The cross-model agreement is unusual: of 200 clips, the same one tops
   the M2 list for every model size. Random chance under independent
   per-model degradation would not reproduce that.
2. The analytical reading of LowLight at s=1.0 is "near-black image with
   mostly only highlight residue", which is an *expected* failure mode
   for a video encoder (no visible motion ⇒ no temporal signal). That's
   a finding, not a bug.
3. There is nothing in the M1/M2 numbers themselves that suggests a
   corrupted frame or NaN propagation; M1 and M2 are well-defined and
   their relative magnitudes match the analytical prediction.

The risk path is option B from the brief: "if pure black, this is itself a
finding (extreme gamma erases the signal entirely). Note for the post."
That outcome looks likely from the analytical inspection — the post should
frame this as a *combined* finding ("at the extreme s=1.0 setting the
perturbation analytically nears black; one specific clip nevertheless
distinguishes itself by going fully featureless first") rather than a
mysterious model failure on a normal-looking dim video.

## Outstanding for coding agent

Render `figures/case_study_202588_clean.png` and
`figures/case_study_202588_perturbed.png` from the SSv2 source on the pod
(or wherever the dataset is mounted). Frame index 8 of the 16-frame sample
that the runner used. Use `vjepa_stress.perturbations.sensor.LowLight` with
`strength=1.0` and the seed from the manifest (per-(video, perturbation)
seed; for video 202588 + low_light pull from `runs/sweep-b077627/manifest.json`
under `seeds["202588"]["low_light"]`). Do not modify the perturbation.

Once the PNGs land, re-open this file and decide:

- **If perturbed image is sensibly dim with visible structure**: keep
  202588, finding becomes "this clip's content goes blank faster than its
  peers under the same operator" — a content-distribution observation.
- **If perturbed image is uniformly black or near-black**: re-frame as
  "the s=1.0 endpoint of low_light analytically erases signal" and either
  (a) keep 202588 to illustrate the endpoint, or (b) substitute with a
  case study from another perturbation family for visual variety.
- **If perturbed image is corrupted (NaN propagation, hue artifacts,
  garbage)**: flag as a perturbation library bug; ask coding agent to
  open an issue. The M1/M2 numbers do not currently suggest this case.

The decision can be made by the user after looking at the rendered frames;
nothing in the analysis-side artifacts needs to change before that
inspection.

---

## 2026-04-29 — R1 gallery + cross-model collapse confirmation (coding agent)

Rendered the top-3-per-category gallery via
`scripts/render_case_study.py` on the pod (SSv2 mounted at
`/workspace/data/ssv2/`). Selection rule: top three unique
`(video_id, perturbation, strength)` tuples per failure-mode category
in `analysis/case_study_candidates.csv`, deduped across categories.
Net: nine unique candidates (no cross-category overlap, since each
category uses a different perturbation family). Twenty-seven PNGs
saved under `figures/case_study_*_(clean|perturbed|sidebyside).png`.
Seeds for each perturbation came from
`runs/sweep-b077627/manifest.json` so the rendered output is
bit-identical to what the encoder ingested.

### Cross-model collapse confirmation (Task 2)

Direct DuckDB lookup against `runs/sweep-b077627/*.parquet` for
`video_id=202588`, `perturbation='low_light'`, `strength=1.0`:

| model                  | M1     | M2     |
|------------------------|--------|--------|
| vjepa-2.1-vit-base     | 0.3478 | 0.9811 |
| vjepa-2.1-vit-large    | 0.6177 | 0.9773 |
| vjepa-2.1-vit-giant    | 0.6119 | 0.9785 |
| vjepa-2.1-vit-gigantic | 0.5603 | 0.9772 |

**Confirmed for M2:** all four models collapse uniformly — M2 spread
across the four is < 0.5% (0.977–0.981), well within noise. The
"three of four" framing is wrong; it's four of four.

**Partial nuance for M1:** large/giant/gigantic all sit in the
catastrophic ~0.6 band; vit-base lands at 0.348, which is still very
high (the population mean M1 at low_light s=1.0 is ~0.18) but visibly
less extreme than the larger models. So "all four models show
comparable *M2* collapse" is exact; "all four show identical M1
collapse" overstates it. The headline-rewriting just needs to be on
M2; the M1 breakdown is more nuanced and shouldn't be flattened.

### R1 deep-dive: `video_id=202588` low_light s=1.0

**Visual outcome.** The perturbed frame is **near-black**. The clean
frame is already a low-light scene (a person's face/profile with what
appears to be a phone or small bright object on the right); after the
gamma=2.5 + gain=0.15 saturation, the perturbed frame is essentially
solid black with only a faint hint of the brightest highlight (the
phone-screen rectangle) preserved as a slightly-less-black region.
This matches the analytical prediction in §2 above (a 200/255 input
pixel maps to ~21/255; mid-tones effectively go to zero).

**Verdict.** Keep `video_id=202588` as a *callout* in the post, but
**do not use it as the post's hero image**. Reasoning:

- The analytical framing — "intrinsic content failure mode at
  perturbation saturation" — still holds and is unchanged: a clip
  that starts dim, hit by a perturbation that drives gamma to 2.5,
  produces an information-poor input, and all four encoders fail on
  it identically. That's a content × perturbation interaction, not
  an encoder-size effect, and the cross-model M2 agreement bears
  that out.
- But "near-black square next to dim square" is not legible at
  thumbnail size. As a hero image in a blog post, a viewer skimming
  past it sees a broken render, not a finding. The deeper write-up
  in the body of the post can still use these PNGs (with the
  metric overlay, this is informative); the *hero* slot wants a
  perturbation whose visual effect is striking *and* legible.
- This is **not a perturbation library bug.** The numbers and the
  pixels both match the documented `_LOW_LIGHT_GAIN_MIN=0.15`,
  `_LOW_LIGHT_GAMMA_MAX=2.5` endpoints. No code review needed.

**Implication for the post.** The "intrinsic content failure mode at
perturbation saturation" framing should be tightened slightly to
"perturbation × content interaction at saturation": the failure isn't
intrinsic to the *clip* — it's that the perturbation at its s=1.0
endpoint, applied to an already-dim scene, drives the input below the
information floor for any encoder. This still supports the cross-model
collapse story; it just credits the perturbation operator rather than
the clip alone.

### Gallery — all 9 candidates

| video_id | perturbation | strength | category(ies)     | visual outcome                                                  | use as hero?                                            |
|----------|--------------|----------|-------------------|------------------------------------------------------------------|---------------------------------------------------------|
| 64685    | motion_blur  | 0.700    | texture_sticking  | strongly directionally smeared cup; legible, clearly perturbed   | Yes — clean visual reading of "what motion blur looks like" |
| 130607   | motion_blur  | 0.700    | texture_sticking  | cable smeared along its long axis; legible                       | Maybe — striking only if reader cares about texture     |
| 130607   | motion_blur  | 0.800    | texture_sticking  | essentially identical to s=0.7 at thumbnail; redundant           | No — drop in favor of the 0.700 render                  |
| 121085   | frame_drops  | 1.000    | identity_drift    | clean and perturbed look near-identical at frame 8 (frame 8 wasn't a dropped frame for this seed) | No — single-frame render hides the perturbation         |
| 41267    | frame_drops  | 0.900    | identity_drift    | striking — clean shows knife slicing an orange, perturbed shows the same orange whole (frame 8 froze on an earlier frame) | **Yes — strongest hero candidate**: shows the freezing mechanism in one image |
| 41267    | frame_drops  | 1.000    | identity_drift    | identical to s=0.9 here (same earlier frame got frozen)         | No — duplicate of the s=0.9 render                      |
| 202588   | low_light    | 1.000    | catastrophic      | near-black with faint highlight residue                          | No — informative in body w/ metric overlay, not as hero |
| 202588   | low_light    | 0.900    | catastrophic      | near-black, marginally more visible than s=1.0                   | No — same problem as s=1.0                              |
| 213048   | low_light    | 1.000    | catastrophic      | near-black; original was already dim bathroom scene             | No — same problem                                        |

**Hero recommendation:** `case_study_41267_frame_drops_0.900_sidebyside.png`. The "knife slicing an orange" → "whole orange" before/after captures the frame_drops mechanism in a single image and is striking at any size. Backup hero: `case_study_64685_motion_blur_0.700_sidebyside.png`. The 202588 / low_light renders earn space in the post body alongside the M2 collapse table, not as the hero.

### Anything unexpected during rendering

- **Frame 8 happens not to be a dropped frame for `121085 frame_drops s=1.0`.** This is a quirk of the random seed: the renderer is honest (it shows the same frame index the encoder saw), so the perturbation is invisible in this single-frame view even though M1/M2 confirm the clip *was* perturbed at other frame indices. If the post wants to use 121085 specifically, render multiple frame indices instead of just 8.
- **`41267 frame_drops s=0.9` and `s=1.0` produce visually identical renders** at frame 8 — both froze on the same earlier frame. The metric difference between s=0.9 and s=1.0 here comes from drops at *other* frame indices.
- **All 9 candidates loaded successfully** — no `FileNotFoundError`, no decode failures, no degenerate-frame rejections.
