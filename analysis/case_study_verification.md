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
