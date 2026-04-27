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

## 2026-04-27 — Stage 0.5: runner + checkpoint glue against fakes

**Commit:** _stage 0.5 series_

The runner is now end-to-end against fakes. 87/87 tests green;
`python -m src.run_sweep --matrix tier2 --output ... --fake` writes 12
shards in <1 s and a re-run skips all 12. Tier 1 (200 cells × 2 fake
videos) writes 400 rows in ~0.4 s on the M4 dev box.

**What landed:**

- `src/sweep/schema.py` — Pydantic v2 `Row` model is the source of
  truth. `ARROW_SCHEMA` is derived from it; the writer fails fast if
  the table built from `rows_to_arrow_table` doesn't match. `SCHEMA_VERSION`
  is recorded in every row and in the manifest; the runner refuses to
  resume against a manifest whose version doesn't match the code.
- `src/sweep/manifest.py` — `build_manifest` + `write_manifest`. Seeds
  are precomputed up-front: `precompute_seeds(video_ids,
  perturbation_names, master_seed)` returns
  `dict[video_id][perturbation_name] -> int` derived from
  `blake2b(master_seed | video_id | perturbation_name)`. A resumed
  run reads the existing manifest and reuses its seeds, so resumed
  ≡ fresh bit-for-bit.
- `src/testing/fakes.py` — `FakeEncoder` (hashes frame bytes + model
  salt → deterministic L2-normed features; counts `forward_calls`),
  `FakeVideoDataset` (uint8 clips seeded by
  `(dataset_name, index, seed)`).
- `src/sweep/checkpoint.py` — `write_shard` is atomic
  (`.parquet.tmp` → fsync → rename), refuses to write empty shards,
  and `cell_completed` only returns True when the file exists, has
  non-zero size, parses as Parquet with the locked schema, and has
  ≥1 row. `cleanup_stale_tmp` is called at runner start.
- `src/sweep/runner.py` — `RunnerConfig` takes `encoder_factory` and
  `dataset_factory` callables. `run_sweep` groups cells by model, so
  the encoder for one model is loaded **once** regardless of how many
  cells reference it (verified by a test: 6 cells × 2 models →
  `encoder_log == ["vit-b", "vit-l"]`). A test-only
  `inject_crash_after_cell` hook drives the restart-semantics test.
- `src/run_sweep.py` — CLI: `--matrix {tier1,tier2,tier3,all,JSON
  path}`, `--output`, `--fake`, `--n-videos`, `--master-seed`,
  `--no-resume`, `--inject-crash-after-cell`, `--log-level`.

**Decisions captured at Stage 0.5:**

- **`vjepa-2.1-vit-G` renamed to `vjepa-2.1-vit-giant`.** The original
  matrix had `vit-g` (1 B params) and `vit-G` (8 B / "Giant") as
  case-distinct identifiers. macOS HFS+/APFS is case-insensitive by
  default, so both models tried to write to the same Parquet path
  (`...vit-g__...parquet`); the second model's cells were silently
  marked as already-completed. The smoke test caught this on the
  first tier-2 run (`cells_skipped=3` on a fresh dir). The rename
  makes the sweep portable across case-insensitive and case-sensitive
  filesystems without per-platform shimming. Updated in `matrix.py`,
  `fakes.py`, `models/vjepa.py`, `manifests/manifest.template.json`.
- **Atomic write contract: `.tmp` + fsync + rename, no empty shards.**
  `write_shard` raises on an empty row list rather than writing a
  zero-row file that would defeat `cell_completed`. Mocked-rename
  test confirms a failed write leaves no shard and no orphan tmp.
- **Manifest is written once, before any cell runs.** A precomputed
  seed dict for every (video_id, perturbation_name) pair lives in
  `manifest.seeds`. The runner looks seeds up by name; nobody else
  decides what seed a cell uses. This is what makes a resumed run
  bit-identical to a fresh run.
- **M3 = NaN until Stage 0.6.** `compute_m3` requires DAVIS-style
  masks. Until the dataset loaders land, the runner fills `m3` with
  `math.nan`. NaN survives the Parquet round-trip (test confirms);
  DuckDB aggregates ignore it.
- **CLI imports as `python -m src.run_sweep`, not
  `python -m vjepa_stress.run_sweep`.** A package rename
  (`src/` → `vjepa_stress/`) is out of scope for Stage 0.5; the
  spec's `vjepa_stress.run_sweep` form is deferred. The hatch wheel
  packaging already emits `src/`-rooted modules.

**Stage 0.6 prerequisites (deferred):**

- Real V-JEPA 2.1 loader in `src/models/vjepa.py` (HF
  `transformers.AutoModel`, pinned revision in the manifest).
- Real `src/data/{ssv2,davis}.py` loaders. The vl-jepa-repro project
  has a proven OpenCV pipeline that's numerically equivalent across
  Mac and Jetson — port that.
- Wire `compute_m3` into the runner once DAVIS masks are loadable.

---

## 2026-04-27 — Variant naming corrected to match Meta's torch.hub entries

**Commit:** _stage 0.5 follow-up_

**Supersedes** the earlier Stage 0.5 entry on the `vit-G` → `vit-giant`
rename. That rename was wrong: it collided with the canonical name
Meta uses for the **1 B** model, and the parameter counts in the
prior entry were also wrong (the 2 B model was reported as 8 B).

**What's actually canonical.** Meta's `facebookresearch/vjepa2`
torch.hub registry exposes the V-JEPA 2.1 variants under these names:

| Paper | Params | torch.hub entry           | Our `Cell.model`        |
|-------|--------|---------------------------|-------------------------|
| ViT-B | 80 M   | `vjepa2_1_vit_base_384`   | `vjepa-2.1-vit-b`       |
| ViT-L | 300 M  | `vjepa2_1_vit_large_384`  | `vjepa-2.1-vit-l`       |
| ViT-g | 1 B    | `vjepa2_1_vit_giant_384`  | `vjepa-2.1-vit-giant`   |
| ViT-G | 2 B    | `vjepa2_1_vit_gigantic_384` | `vjepa-2.1-vit-gigantic` |

`giant` = lowercase-g = 1 B; `gigantic` = uppercase-G = 2 B. Meta
solved the case-insensitive-filesystem problem by spelling the words
out, and we should match that spelling exactly so a future loader can
build torch.hub entries by string rule (`f"vjepa2_1_vit_{suffix}_384"`)
without a case-juggling lookup.

**Renames applied** (in `matrix.py`, `fakes.py`, `models/vjepa.py`,
`manifests/manifest.template.json`):

- `vjepa-2.1-vit-g` → `vjepa-2.1-vit-giant` (1 B)
- `vjepa-2.1-vit-giant` → `vjepa-2.1-vit-gigantic` (2 B)

The B/L identifiers are left as `vit-b`/`vit-l` for now; switching
those to `vit-base`/`vit-large` would match Meta's spelling end-to-end
but is bigger than the bug-fix that motivated this entry. **Open
question for Stage 0.6:** when we wire the real loader, decide whether
to harmonize the small/medium identifiers too. If yes, add a
superseding devlog entry then.

**Open question for Stage 0.6 (HF availability).** As of 2026-03-20,
V-JEPA 2.1 weights are **not yet on Hugging Face**; the integration
request is open and unfulfilled. Meta distributes the weights via
`torch.hub.load('facebookresearch/vjepa2', ...)`. This means:

- `src/models/vjepa.py` should not assume
  `transformers.AutoModel.from_pretrained(...)` will work; the
  `VARIANT_REPOS` table currently uses HF-style placeholders (`TODO/...`)
  that may need to become torch.hub entry strings instead.
- The first Stage 0.6 task should be a verification spike: pull one
  variant via torch.hub in the venv on the M4 dev box and confirm
  weights actually load. That spike decides the loader contract.

**macOS/decord note.** Meta's repo states V-JEPA 2 (and so 2.1) ties
to `decord`, which has no macOS build. The OpenCV decode path from
`vl-jepa-repro` Phase 7 is the proven workaround and should be the
default loader on `darwin`; `decord` stays gated behind the existing
`sys_platform != 'darwin'` marker in `pyproject.toml`. Document this
in `EXPERIMENT.md` when that doc is filled in.

---
