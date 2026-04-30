# Requests for the coding agent

Items the analysis agent surfaced during Stage 5 follow-up that fall outside
its file-ownership boundary. Each is self-contained — no analysis-side
context needed beyond what's written here. Pick up in any order.

---

## R1 — Render case-study clip 202588 (Stage 5, blocked locally)

**Why.** Stage 5 identified `video_id=202588` (low_light, s=1.0) as the #1
worst SSv2 clip on M2 for **all four** model sizes. Before the post or paper
uses it as the headline case study, the perturbed frame should be
inspected to confirm the result reflects sensible content collapse rather
than a corrupted output. The analysis agent cannot do this locally — SSv2
is not on the local disk (`~/data/ssv2/` is empty, `$VJEPA_STRESS_SSV2_DIR`
unset, `/workspace/data/ssv2/` is RunPod-only).

**What to do.**

1. On a machine with SSv2 mounted (the RunPod pod or wherever the dataset
   is now), use `vjepa_stress.data.ssv2.SSv2Dataset` (or the loader's path
   resolution) to decode `video_id=202588` to the standard 16-frame, 384²
   tensor.
2. Apply `vjepa_stress.perturbations.sensor.LowLight` at `strength=1.0`,
   passing the seed from `runs/sweep-b077627/manifest.json` →
   `seeds["202588"]["low_light"]`. Do not modify the perturbation.
3. Save frame 8 of both clean and perturbed videos as PNG:
   - `figures/case_study_202588_clean.png`
   - `figures/case_study_202588_perturbed.png`
4. Append the verdict (sensible dim / near-black / corrupted) to
   `analysis/case_study_verification.md` under a new "## Verification
   result" section. The decision tree is already written there.

**Constraints.**

- Do not edit `vjepa_stress/perturbations/sensor.py` — the perturbation is
  part of the pre-registered measurement chain.
- Do not modify the canonical run dir `runs/sweep-b077627/`.
- Frame index 8 is the convention; if the runner uses a different sampling,
  pick whichever frame index the runner saw at sample-position 8.

---

## R2 — M1 metric: scope-aware reporting on time-axis perturbations

**Why.** Stage 5 found that M1 is ill-defined under perturbations that
reorganize the time axis (`reverse`, `frame_drops`). The metric is well
defined for everything else. Documented in `RESEARCH.md` §3.1 ("Validity
scope") and `devlog.md` 2026-04-29 entry. The H2 M1 component was
withdrawn as a result. The pre-registered prediction text in §7 is locked.

**What to do (when the next sweep is wired).**

1. Add a `m1_valid: bool` (or equivalent) field to the per-row Parquet
   schema, set to `False` for `perturbation in {"reverse", "frame_drops"}`
   and `True` everywhere else. Bump schema version `1.1` → `1.2`.
2. In the runner, optionally NaN out M1 for the time-axis perturbations
   instead of writing the meaningless number. Either choice is fine; the
   contract is that downstream code can tell from the row whether the M1
   number is interpretable. NaN is probably cleaner since it's already
   used for M3 deferred cells.
3. Tests in `tests/test_metrics.py` should pin the contract: M1 is NaN
   (or `m1_valid=False`) on reverse and frame_drops; otherwise present
   and finite.
4. Do **not** change `compute_m1`'s implementation. The math is correct
   for in-scope perturbations; the change is at the schema / scope layer.

This is non-urgent — Stage 5 analysis is reportable as-is. Land it before
the next sweep so the data-side guard exists from then on.

**Constraint.** Do not modify §7 of `RESEARCH.md` (pre-registered
predictions are locked). The validity-scope note in §3.1 is the
canonical reference.

---

## How to flag a request back

Append a "## Resolution" subsection under the relevant request when done,
with the commit SHA and any caveats. Do not delete the request — keep this
file append-only so the audit trail survives.
