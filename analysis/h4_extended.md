# H4 follow-up — strength-curve descriptive shape

Pre-registered claim: frame_drops M2 at s=0.5 > gaussian_noise M2 at s=0.5.
Single-strength comparison was refuted (frame_drops=0.324, gaussian=0.499,
Δ=−0.175). The follow-up below extends the comparison across the full
strength range, faceted by model. Source data:
`analysis/h4_strength_table.csv` (40 rows = 4 models × 10 strengths). Plot:
`figures/h4_strength_comparison.{png,pdf}`.

## Shape

Gaussian dominates frame_drops at every strength tested for vit-base,
vit-large, and vit-gigantic — 30 of 30 strength points, no crossing. The
Δ (gaussian − frame_drops) is largest in the s≈0.2–0.4 mid-range
(Δ ≈ +0.19 to +0.24) and narrows monotonically toward s=1.0, where the
two corruptions begin to converge as both metrics saturate.

Vit-giant is the single exception: 9 of 10 strengths still favor
Gaussian, but at s=1.0 the means cross (Gaussian=0.609, frame_drops=0.635,
Δ=−0.026), and the bootstrap CIs at that point overlap by ~0.01. The H4
direction therefore holds at s=0.5 in every model and across the
strength range for 3 of 4 models; the only place the pre-registered
direction is even briefly observed is at the most extreme strength on a
single model size, with overlapping CIs.

The dominance of Gaussian over frame_drops widens with strength up to
s≈0.3 then contracts; on every model the gap is at its smallest at
s=1.0. The shape is consistent across model sizes — same opening,
peaking, and contracting pattern — even though the absolute heights
differ by a few percentage points between model variants.
