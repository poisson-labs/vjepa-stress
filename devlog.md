# vjepa-stress devlog

Append-only log of milestones, design decisions, and follow-ups. New
entries go at the **bottom**. Do not edit prior entries; if a decision
is reversed, write a new entry that supersedes the old one and links
back to it. Each entry is dated and references the relevant commit.

---

## 2026-04-27 — Repo bootstrap

**Commit:** `1c98f58 Initial scaffolding`

Bootstrapped the repo from scratch. 44 files, 55 KB tracked at first
commit. Layout matches the bootstrap spec: `src/` package with sub-
packages for data, models, perturbations, metrics, sweep, viz; `scripts/`
for stage-gated entry points; `tests/` for the contract suite;
`manifests/` for the reproducibility schema; `notebooks/` template; data
gitignored at `/data/` (leading slash, to avoid the previous-project bug
where `data/` also caught `src/data/`).

**Key decisions captured at bootstrap:**

- **Strength=0 ≡ identity** is the load-bearing perturbation contract.
  Encoded in `Perturbation._validate` and exhaustively tested. This makes
  "Identity at any strength" and "any perturbation at strength 0"
  exchangeable, which is what makes the calibration probes interpretable.
- **222-cell experiment matrix.** Tier 1: 4 models × 5 perturbations × 10
  strengths = 200. Tier 2 (calibration probes): 4 × 3 × 1 = 12. Tier 3
  (codec sanity): 1 × 1 × 10 = 10. Strengths are `linspace(0.1, 1.0, 10)`
  — strength=0 is covered by the Identity probe rather than duplicated
  per-perturbation.
- **`src/perturbations/learned/` reserved for Part 2.** Empty, with a
  `.gitkeep`. No imports from Monte; vjepa-stress is independent of the
  Monte codebase and will stay so.
- **`RESEARCH.md` and `EXPERIMENT.md` are flagged stubs.** The bootstrap
  spec referenced them but the source documents weren't in scope. The
  stubs explicitly point at the sections the rest of the code expects to
  exist (M1/M2/M3 math in §3, pre-registered hypotheses in §7, etc.).
- **Per-cell Parquet shards, not one big table.** Decided in
  `src/sweep/checkpoint.py`: each cell writes
  `{output_dir}/{cell_id}.parquet`; the sweep is resumable by skipping
  cells whose shard exists. Aggregation is via `duckdb`, not by
  appending to a single mutable file.

---

## 2026-04-27 — Venv-first quickstart

**Commit:** `c1bd6b5 Make venv-first the canonical quickstart in README`

`.venv/` (gitignored) is the canonical install target on both the M4 dev
box and RunPod. README updated to make this the documented workflow:
`python3 -m venv .venv && source .venv/bin/activate && pip install -e
".[dev]"`. The heavy ML deps (torch, transformers, opencv) shouldn't
live in system Python.

---

## 2026-04-27 — Stage 0: perturbations + metrics implemented

**Commit:** `cba982e Stage 0: implement all perturbations and feature-stability metrics`

All six numeric perturbations and all three metrics now have real
implementations. Tests: 35 → 67, all passing.

**Strength endpoints (s=1) for the numeric perturbations:**

| Perturbation     | s=1 setting                                              |
|------------------|----------------------------------------------------------|
| `gaussian_noise` | σ = 50 (8-bit units), additive                           |
| `motion_blur`    | kernel length = 25 px, angle drawn per video             |
| `frame_drops`    | drop probability = 0.5; frame 0 always retained          |
| `occlusion`      | square cutout, side = 0.5·min(H,W), gray fill (128)      |
| `low_light`      | gain=0.15, gamma=2.5, additive σ=0.02 on [0,1] scale     |
| `jpeg`           | JPEG quality = 5 (95 at s=0+, linear in strength)        |

These are placeholders — Stage 1 should empirically refine the s=1
endpoints so the strength sweep has a useful dynamic range on real
SSv2 clips. Picked by hand to span "still recoverable" through
"obviously degraded."

**M2 reformulation (this is the load-bearing decision of Stage 0).**

The original docstring described M2 as `S(v) - S(v')` where
`S(v) = mean cos(F_t, F_{t+1})`. This is **mathematically symmetric
under temporal reversal**: the multiset of consecutive token pairs is
identical when you reverse the clip, so `S(v) ≡ S(v_rev)` and the
"M2 > 0 on Reverse" test premise was incompatible with that math.

Resolved by redefining M2 as cosine distance between matched temporal
gradients:

```
δ(v)_t = F(v)_{t+1} - F(v)_t
M2 = mean over (t, p) of  1 - cos(δ(v)_t, δ(v')_t)
```

- Identity → δ_clean = δ_perturb everywhere → M2 = 0. ✓
- Reverse  → δ_perturb_t = -δ_clean_{T-2-t} → cosine collapses to ~0
  on most indices, -1 at the middle index, M2 lands around 1 on
  random features (verified: `1.066`).

Bounded in `[0, 2]` (same scale as M1). Saturating at 2 requires
strict periodic/symmetric structure in the deltas, not just reversal.

**TODO:** reconcile the new M2 definition with `RESEARCH.md §3` when
that document is filled in. The metric module docstring is the source
of truth until then.

**M3 signature deviation.** Added `seed: int = 0` kwarg to `compute_m3`
(not in the original placeholder signature) so keypoint sampling is
deterministic. Same seed across (clean, perturbed) evaluations of one
video so the metric is paired.

---

## 2026-04-27 — Devlog convention established

This file. Append-only, dated entries at the bottom, one per milestone
or design decision. Going forward, every commit that represents a
milestone or a non-trivial decision gets an entry here before or with
the commit that introduces it.

What counts as worth a devlog entry:

- A new feature reaching working state (Stage 0, Stage 1, ...).
- A decision that constrains future work (the M2 reformulation; the
  per-cell shard format; the strength endpoint table).
- A scope change or reversal — supersede the prior entry, don't edit it.
- An external dependency added or pinned (HF model revision, dataset
  version hash, RunPod image change).
- Any failure mode discovered during a sweep that informs the next run.

What does **not** belong here:

- Routine code cleanup, typo fixes, formatting passes.
- Per-test-run output (that lives in `manifests/manifest.json` and the
  Parquet shards under `checkpoints/`).
- TODOs (those live in code comments or a separate task file).

---
