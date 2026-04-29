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

## 2026-04-27 — Stage 0.6 Task 2: V-JEPA 2.1 loading spike

**Commits:** `0ddf76c Rename src/ → vjepa_stress/`, `b4dfc6f Stage 0.6
Task 2: V-JEPA 2.1 loading spike — torch.hub primary`.

### Headline

D1 (loading mechanism) and D2 (resolution) are both pinned. Stage 0.6
loader will use `torch.hub.load('facebookresearch/vjepa2', ...)` at
384×384, with a one-line monkey-patch of `VJEPA_BASE_URL` because
Meta's hub repo ships with `http://localhost:8300` hardcoded as the
download URL. Full forward-pass contract captured in
`docs/loading_spike.md`.

### Structural fix: `src/` → `vjepa_stress/`

The first spike run crashed with `ModuleNotFoundError: No module named
'src.hub'`. Root cause: `facebookresearch/vjepa2` ships its own
top-level `src/` package (with `src/hub/...` for the hubconf entries),
and when `torch.hub.load(...)` clones the repo and adds the clone dir
to `sys.path`, Python resolved `src` to *our* package — which has no
`hub` submodule. Any future hub clone with a colliding top-level name
would have hit the same wall.

Fix: renamed the project package `src/` → `vjepa_stress/`. Mechanical
git mv + import find-replace across .py files; pyproject `packages` and
the Dockerfile `COPY` both updated. 87 → 91 tests still green (4 new in
`test_hub_isolation.py`). Single atomic commit so it can be reverted if
something regresses.

The defence-in-depth helper `vjepa_stress.utils.hub_isolation
.isolate_for_torch_hub()` is preserved as a context manager. Even with
`vjepa_stress` as the package name, future contributors should wrap any
`torch.hub.load()` in it — the next colliding repo we touch (Sentry,
Argus, anywhere) will be guarded by the same import-time scrub.

### Findings

| Mechanism | Outcome |
|-----------|---------|
| HF Hub | 0 V-JEPA 2.1 models published. 10 V-JEPA 2.0 fine-tunes only. Not a viable path. |
| torch.hub | All four 2.1 entries listed (`vjepa2_1_vit_{base,large,giant,gigantic}_384`). After URL fix, `vit-base` downloads in 19.5 s on Mac M4 and forwards a `(1, 3, 2, 384, 384)` clip to `(1, 576, 768)` float32 in 0.12 s. **Primary mechanism.** |
| Direct fbaipublicfiles | All four `.pt` files reachable. Filenames had to be recovered from the cached hub repo's `ARCH_NAME_MAP` — they're `vjepa2_1_vitb_dist_vitG_384.pt` (1.55 GB, base, distilled), `vjepa2_1_vitl_dist_vitG_384.pt` (4.80 GB, large, distilled), `vjepa2_1_vitg_384.pt` (15.7 GB, giant native), `vjepa2_1_vitG_384.pt` (28.2 GB, gigantic native). Useful as a loader fallback. |

### Tubelet flag for Task 3

The encoder uses `tubelet_size=2` by default — two consecutive frames
collapse into a single token row. Output is `(B, T_tubelet × 576, D)`.
Our M2 metric assumes per-frame deltas `δ_t = F_{t+1} - F_t`; this
either becomes per-tubelet deltas (cleanest, halves T) or we override
the kwarg if the entrypoint accepts it (unverified). Decision deferred
to Task 3 — flagging here so it doesn't get lost.

### Constraint: pyproject deps

`facebookresearch/vjepa2`'s hubconf raises `RuntimeError: Missing
dependencies: timm, einops` if either is absent. Both added to
`[project.dependencies]`. They're transitive; we shouldn't be calling
either directly.

### `STAGE_0.6_TASKS.md` drift

The Stage 0.6 task ledger at `docs/STAGE_0.6_TASKS.md` was written
before the rename and still references `src/perturbations/`,
`src/models/vjepa.py`, etc. Substitute `vjepa_stress/...` everywhere
when reading. Not editing the ledger mid-execution to keep the audit
trail clean — Task 1's "Files in scope" lines should be read as
historical pointers, not authoritative paths.

### Next

Task 3 (encoder loader) starts here:
- Implement `_load_via_hub(variant)` in `vjepa_stress/models/vjepa.py`:
  isolate, monkey-patch `VJEPA_BASE_URL`, `torch.hub.load`, return
  `(encoder, predictor)[0]`.
- Define an `Encoder` Protocol so `FakeEncoder` and the real loader are
  interchangeable in `RunnerConfig.encoder_factory`.
- Wire `try_forward_pass` so OOM on giant/gigantic surfaces at load
  time, not deep in the sweep.
- 4 unit tests; CI runs `vit-base` only (per the ledger).

Larger variants (large, giant, gigantic) deliberately *not* downloaded
to the M4 — RunPod is where their forward passes actually need to run
on a real GPU. Task 3 must work on `vit-base` end-to-end before we move.

---

## 2026-04-27 — Stage 0.6 Tasks 3 + 3.5: real V-JEPA 2.1 loader, M2 calibration deferred

### Task 3 — `vjepa_stress/models/vjepa.py`

`load_encoder("vit-base")` now end-to-end on Mac M4 CPU:

- `Encoder` Protocol (`@runtime_checkable`) exposes
  `embedding_dim`/`tubelet_size`/`patch_size`/`input_resolution` plus
  `output_temporal_dim(n_frames)` / `output_spatial_patches()` /
  `encode(frames)` / `forward_calls`. `VJEPAEncoder` and `FakeEncoder`
  both satisfy it; the runner's `_EncoderProtocol` was widened to match
  and `_features_to_token_grid` now calls
  `encoder.output_temporal_dim(T_input)` rather than assuming 1:1.
- `VARIANT_REPOS` maps the canonical short names
  (`vit-base`/`vit-large`/`vit-giant`/`vit-gigantic`) to the four
  `vjepa2_1_vit_<v>_384` hub entries. `_normalize_variant` accepts
  either short or `"vjepa-2.1-vit-..."` long form so both Cell.model
  styles flow through unchanged.
- `patch_meta_url()` is the structural fix for the localhost stub from
  Task 2's spike: it adds the cached repo to `sys.path`, evicts any
  prior `src.*`, imports `src.hub.backbones`, saves the original
  `VJEPA_BASE_URL`, swaps in `https://dl.fbaipublicfiles.com/vjepa2`,
  and unwinds *all four* (URL, sys.path, evicted modules, cache modules)
  on exit — including the failure path. The reversibility is unit-tested
  against a stubbed cache dir so the test runs in the fast suite.
- `try_forward_pass` runs a `(2, 384, 384, 3)` uint8 dummy clip
  through the encoder right after load and asserts the documented
  output shape, so any device/dtype misconfig surfaces at load time
  rather than 4h into a sweep. Counter is reset before returning so
  callers see clean `forward_calls` accounting.
- `default_dtype_for_device` picks bf16 on CUDA and fp32 elsewhere
  (MPS bf16 support is incomplete; Mac dev stays in fp32, CUDA prod
  gets the bf16 win).
- `encode()` takes `(T, H, W, 3)` uint8 (matching `FakeEncoder`),
  permutes/unsqueezes to the model's `(1, C, T, H, W)` convention,
  ImageNet-normalizes, runs the forward, validates the output shape
  against `(1, T_tubelet × 576, embedding_dim)`, reshapes to
  `(T_tubelet, P, D)`, casts back to fp32 on CPU, and L2-normalizes.

End-to-end forward: `(16, 384, 384, 3)` uint8 → `(8, 576, 768)`
fp32 in ~10 s on M4 CPU. 8 fast loader tests + 1 slow end-to-end
test + 99 fast + 2 slow tests total green (1 calibration test
skipped — see below).

### Task 3.5 — M2 reverse calibration: deferred to Task 4

The original spec was a slow test asserting `0.9 < M2 < 1.4` on
real-encoder features under reverse playback, against RESEARCH.md
§3.2's analytical ~1.14 expectation. Built it as
`test_m2_reverse_calibration_real_features` and ran it.

**Observation.** On a 16-frame synthetic video built as `base_image
+ cumulative_drift(±2 per pixel per frame)` (i.e., uniform low-amplitude
linear drift), real V-JEPA 2.1 vit-base produces **M2 ≈ 0.10**, an
order of magnitude *smaller* than the 1.14 prediction. Sign is right
(strictly positive — Reverse is *some* perturbation) but the magnitude
isn't.

**Two candidate mechanisms** (mechanism call deferred — see below):

1. *Tubelet pair-averaging.* `tubelet_size=2` collapses two consecutive
   frames into a single token row. Within a tubelet, the model may be
   roughly symmetric in frame ordering (averaging-style pooling rather
   than a sequenced read). For globally smooth drift, this swallows
   most of the temporal-direction signal: `delta_clean[t]` and
   `delta_reverse[t]` end up *aligned*, not anti-aligned, because the
   per-tubelet feature change tracks position in the input sequence
   more than direction of motion.
2. *Low-entropy collapse.* Synthetic uniform drift is far off-manifold
   for a model trained on real motion video. The encoder may map the
   whole clip into a low-entropy region where consecutive-tubelet
   features are nearly identical, so `‖delta‖ → 0` at *both* clean
   and reversed and the cosine becomes dominated by tiny aligned
   noise vectors.

These aren't mutually exclusive — both could contribute in different
ratios. Distinguishing them requires comparing M2 across (real
SSv2 motion clip) vs. (synthetic drift) vs. (synthetic high-entropy
content), which requires SSv2 plumbing.

**Decision.** Defer the mechanism call to Stage 0.6 Task 4 (SSv2
loader). The synthetic stimulus is just for plumbing tests; real
calibration only makes sense on real video. Concretely:

- `test_m2_reverse_calibration_real_features` is now
  `@pytest.mark.slow @pytest.mark.calibration` and `pytest.skip`s with
  a message tying it to RESEARCH.md §3.2 and Task 4. The body is intact
  so re-enabling is a one-line change once SSv2 exists.
- Added `test_m2_reverse_synthetic_observed_band` which pins the
  observed ~0.10 with a 0.3 ceiling. Docstring is explicit that this
  *captures observed behavior, mechanism TBD pending real-video
  calibration*. Its only job is to flag a regression in the loader /
  preprocessing path — not to validate any claim about M2's
  semantics.
- Existing synthetic test `test_m2_positive_on_reversed_features`
  (in `tests/test_metrics.py`) gets a docstring noting it covers
  algorithmic correctness only — *not* numerical calibration —
  pointing readers at the calibration test for the latter.
- New pytest marker `calibration` documented in `pyproject.toml`.

This is the "named stub with implementation notes for deferred work"
pattern from the feedback ledger — surfacing the gap rather than
silently widening the band to whatever number we observed.

### Suite state

| | count |
|---|---|
| fast tests | 99 (was 91; +8 new in `test_vjepa_loader.py`) |
| slow tests | 3 — `test_load_smallest_works_end_to_end` + `test_m2_reverse_synthetic_observed_band` pass; `test_m2_reverse_calibration_real_features` skipped |
| skipped | 1 (the calibration test, with documented reason) |

### Next

Task 4: SSv2 loader. When that lands:

1. Re-enable `test_m2_reverse_calibration_real_features` (delete the
   `pytest.skip`, add an SSv2 fixture).
2. Run the calibration → if `0.9 < M2 < 1.4` lands, mechanism
   probably *was* synthetic content collapse. If it doesn't, dig
   into tubelet pair-averaging (e.g. compare against a hub-loaded
   variant with `tubelet_size=1` if Meta's kwargs allow, or
   override and re-run).
3. Either way, fold the mechanism finding back into RESEARCH.md §3.2
   so the analytical band is supported by both prediction *and*
   observation.

---

## 2026-04-27 — Stage 0.6 Task 4: SSv2 loader (manual-download, OpenCV decode)

### Surface

`vjepa_stress/data/ssv2.py` replaces the Stage 0 placeholder:

- `SSv2Sample(video_id, frames)` — mirrors `FakeVideo`, so the runner
  iterates real and fake datasets through the same code path.
- `SSv2Dataset(*, video_paths, num_frames=16, image_size=384,
  dataset_name="ssv2")` — implements `video_ids()` / `__len__` /
  `__iter__` / `__getitem__`, satisfies the runner's `_DatasetProtocol`.
- `load_ssv2_dataset(*, n_videos=200, seed=0, split="validation",
  cache_dir=None, num_frames=16, image_size=384,
  file_patterns=("*.webm", "*.mp4"))` — the user-facing factory.

### Cache resolution

Priority order: explicit `cache_dir=` → `$VJEPA_STRESS_SSV2_DIR` →
`~/data/ssv2/`. The ergonomic env var is the path of least resistance
on RunPod where `~/data/...` would point at ephemeral storage; the
default keeps Mac dev simple.

### Manual-download error path (D3 closed)

D3 (manual-only vs. auto-with-token) is settled: manual-only.
`load_ssv2_dataset` raises `FileNotFoundError` with the install URL
(`https://developer.qualcomm.com/software/ai-datasets/something-something`)
in two cases:
1. Split dir doesn't exist.
2. Split dir exists but contains no matching videos.

Both messages include the cache path the loader looked at and remind
the user about the env-var override. Two unit tests pin the error
strings — anyone who later swaps the URL or rewrites the message
will get a flagged test rather than a silently-degraded UX.

### Decode

OpenCV (`cv2.VideoCapture`). decord is gated to non-darwin in
`pyproject.toml` and we deliberately don't add a darwin shim — same
code on Mac and on RunPod is more important than the speed bump
decord buys on Linux for this workload (we're encoder-bound, not
decode-bound).

Frame indices: `np.linspace(0, total-1, num_frames).round().astype(int)`,
clamped to `[0, total-1]` because some containers report inflated
frame counts. Per-frame: `cap.set(POS_FRAMES, idx)` + `read()`,
BGR→RGB convert, short-side resize + center-crop to `image_size`.
Resize uses `INTER_AREA` for downsampling and `INTER_LINEAR` for
upsampling — small thing, but the sweep is cumulative-error-sensitive
and `INTER_LINEAR` aliases on big downsample factors.

### Sampling

`random.Random(seed).sample(all_paths, n_videos)` then `sorted(...)`.
Sampling first with the seed, then sorting, is the right order: the
sample set is a function of `(pool, seed)`, and the sort makes
iteration deterministic and easy to audit. `n_videos > pool` falls
back to "use the whole pool" with a logged warning rather than
raising — synthetic test pools are smaller than 200 and we don't
want every test that constructs an SSv2 fixture to know about the
pool size.

### Tests (10 fast + 2 slow + 1 calibration)

`tests/test_ssv2_loader.py`:

| Test | Marker | What it covers |
|------|--------|----------------|
| `test_resolve_cache_dir_prefers_explicit_arg` | fast | priority of `cache_dir=` over env |
| `test_resolve_cache_dir_falls_back_to_env` | fast | env var is honoured |
| `test_resolve_cache_dir_default` | fast | default is `~/data/ssv2/` |
| `test_load_with_missing_split_dir_raises_with_install_url` | fast | error path 1 |
| `test_load_with_empty_split_dir_raises_with_install_url` | fast | error path 2 |
| `test_resize_and_center_crop_shape_and_aspect` | fast | resize math, both aspect ratios |
| `test_video_shape_with_synthetic_mp4` | fast | full decode of a writer-generated mp4 → `(16,384,384,3)` uint8 RGB |
| `test_load_is_deterministic_with_same_seed` | fast | seeded sampling repeats |
| `test_n_videos_capped_at_pool_size_emits_warning` | fast | small-pool warning path |
| `test_dataset_iteration_yields_runner_compatible_samples` | fast | runner protocol compatibility |
| `test_load_returns_correct_count_real_ssv2` | slow + soft-skip | real SSv2 |
| `test_load_is_deterministic_real_ssv2` | slow + soft-skip | real SSv2 |

The synthetic-mp4 path is the load-bearing one — it exercises the
full decode pipeline (open, frame seek, BGR→RGB, resize+crop, dtype)
without needing the gated dataset, so CI on a fresh box catches
regressions.

### M2 reverse calibration: now soft-skipped, not hard-skipped

`test_m2_reverse_calibration_real_features` is no longer an
unconditional `pytest.skip`. The body now:

1. Checks `_resolve_cache_dir(None) / "validation"` for SSv2 presence.
2. If absent → `pytest.skip` with the install URL and a pointer at
   the synthetic observed-band test.
3. If present → load one SSv2 clip, encode clean + reverse, assert
   `0.9 < M2 < 1.4`.

So on a Mac dev box with no SSv2 it still skips, but on RunPod (or
any host with the env var pointed at a real SSv2 tree) it runs
without code changes — which is exactly the "named stub with escape
hatch" pattern the feedback ledger calls for.

`test_m2_reverse_synthetic_observed_band` keeps pinning the ~0.10
synthetic value with a 0.3 ceiling. Once we observe the real-video
M2 it'll either confirm the synthetic-collapse hypothesis or send us
to investigate tubelet pair-averaging — the test pair gives us a
clean A/B for that.

### Suite state

| | count |
|---|---|
| fast tests | 109 (was 99; +10 in `test_ssv2_loader.py`) |
| slow tests | 4 — vit-base load + M2 synthetic-band pass; 2 SSv2 + 1 M2-calibration soft-skip on Mac (no SSv2) |

### Next

- Task 5: DAVIS loader. Smaller, ungated dataset; same iterable
  shape but yields `(video_id, frames, masks)` for M3.
- Final integration smoke (`test_real_pipeline_smoke`) once Task 5
  lands — one cell, real ViT-base, 2 SSv2 clips, assert non-NaN
  M1/M2 end-to-end.
- Once an SSv2 tree is plumbed somewhere we control (RunPod or local
  manual download), run the M2 calibration test and resolve the
  ~0.10 vs. ~1.14 discrepancy noted in the Task 3.5 entry.

---

## 2026-04-27 — Stage 0.6 Task 5: DAVIS 2017 loader

### Surface

`vjepa_stress/data/davis.py` replaces the Stage 0 placeholder:

- `DAVISSample(sequence_id, frames, masks)` with a `video_id` alias
  on `sequence_id` so the runner sees the same `.video_id` accessor
  as `FakeVideo` and `SSv2Sample`. `frames: (T, R, R, 3) uint8 RGB`,
  `masks: (T, R, R) int32` (0 = background, ≥1 = instance ID).
- `DAVISDataset(*, root, sequence_ids, num_frames=16, image_size=384)`
  exposes `video_ids() / sequence_ids() / __len__ / __iter__ /
  __getitem__`. Decoding happens lazily in `__getitem__` rather than
  at construction time — DAVIS is small (~1 GB) but we still don't
  want to materialise 30 sequences × 16 frames × 384² as soon as the
  dataset object exists.
- `load_davis_dataset(*, split="val", cache_dir=None, num_frames=16,
  image_size=384, sequence_ids=None) -> DAVISDataset` factory.

### Cache resolution + inner-DAVIS unwrap

Same priority order as SSv2 (arg → `$VJEPA_STRESS_DAVIS_DIR` →
`~/data/davis/`). DAVIS's official zip extracts to a directory tree
that includes an inner `DAVIS/` directory; the loader detects that
case and unwraps it, so users who unzip with the default settings
don't have to flatten the tree by hand. Both layouts work:

```
~/data/davis/JPEGImages/...                # already unwrapped
~/data/davis/DAVIS/JPEGImages/...          # default unzip layout
```

### Sequence list

Reads `ImageSets/2017/<split>.txt` first (the canonical DAVIS list,
30 entries for `val`). Falls back to a directory listing of
`JPEGImages/480p/` if the ImageSets file is missing — some
re-distributions strip the meta files, and we'd rather succeed
loudly with a deterministic order than fail mysteriously. An
explicit `sequence_ids=` override skips both, which is mostly a test
ergonomics affordance.

### Mask decoding — the load-bearing detail

DAVIS PNG masks are *paletted* PNGs (PIL mode "P"): each pixel byte
is a palette index, and the palette table maps that index to an RGB
colour for visualisation. **`cv2.imread` decodes the palette and
returns RGB**, which silently destroys the integer instance IDs that
M3 needs.

The loader uses PIL specifically because `np.array(Image.open(p))`
on a mode-"P" PNG returns the palette indices unchanged. There's a
defensive convert-to-grayscale path for re-saved RGB masks (a
well-meaning preprocessor might emit those), and a 2-D shape check
that surfaces the error early if the cache file is malformed.

This mistake is exactly the sort of thing that produces silently
garbage M3 numbers in production and is hard to debug from the
metric output. `test_mask_values_are_integers` pins it: the
synthetic fixture writes IDs 1 (left) and 2 (right) and the test
asserts both survive decode + resize unchanged, with no spurious
RGB-derived values.

### Resize geometry

Frames: short-side resize + center-crop, `INTER_AREA` for
downsampling and `INTER_LINEAR` for upsampling — same as SSv2.

Masks: same crop-resize geometry, `INTER_NEAREST`. Bilinear or
area-resize on integer instance IDs would produce non-integer
values that don't index cleanly. The two helpers are kept side by
side in `davis.py` rather than shared with SSv2 — a third loader
will earn the extraction; two doesn't yet.

### Frames and masks share `np.linspace` indices

Per the task spec, the same `np.linspace(0, total-1, num_frames)`
sample is used for both the JPEG and PNG file lists. M3 keypoints
in frame `t` correspond to mask annotations in frame `t` only if
those `t`s pull from the same on-disk indices.
`test_frame_and_mask_share_index_sample` pins this by writing JPEGs
that encode their on-disk index in pixel intensity (`value = i*30`),
then asserting the decoded values match `np.linspace(0, 11, 4)` and
that all sampled masks still carry both instance IDs. If the two
indexings ever drift apart, the assert fires.

### Pillow as an explicit dep

Pillow lands transitively via torchvision but the DAVIS loader
imports it directly. Added `Pillow>=10.0` to `[project.dependencies]`
so the dep is explicit and pinned to a floor that supports the
non-deprecated `Image.frombytes("P", ...)` API used by the test
fixture.

### Tests (12 fast + 2 slow)

`tests/test_davis_loader.py`:

| Test | Marker | What it covers |
|------|--------|----------------|
| `test_load_with_missing_cache_dir_raises_with_install_url` | fast | error path 1 |
| `test_load_with_missing_subdirs_raises_with_install_url` | fast | error path 2 |
| `test_load_with_empty_sequence_list_raises` | fast | error path 3 |
| `test_load_unwraps_inner_davis_directory` | fast | inner-DAVIS handling |
| `test_video_id_is_stable_string` | fast | `video_id` ≡ `sequence_id` |
| `test_mask_shape_matches_frame_shape` | fast | `(T,R,R,3)` / `(T,R,R)` |
| `test_mask_values_are_integers` | fast | palette-vs-RGB regression guard |
| `test_frame_and_mask_share_index_sample` | fast | linspace shared between JPEG & PNG |
| `test_dataset_iteration_yields_runner_compatible_samples` | fast | runner protocol parity |
| `test_explicit_sequence_ids_override_imagesets` | fast | `sequence_ids=` override |
| `test_imagesets_fallback_to_directory_listing` | fast | val.txt-missing fallback |
| `test_env_var_overrides_default` | fast | `$VJEPA_STRESS_DAVIS_DIR` |
| `test_load_returns_30_videos_real_davis` | slow + soft-skip | real DAVIS = 30 sequences |
| `test_load_is_deterministic_real_davis` | slow + soft-skip | real DAVIS stable order |

The synthetic-DAVIS scaffold uses `cv2.imwrite` for JPEGs and
`PIL.Image.frombytes("P", ...)` + `putpalette(...)` for the
paletted PNGs — i.e. the test files exercise the exact decode path
real DAVIS data hits.

### Suite state

| | count |
|---|---|
| fast tests | 121 (was 109; +12 in `test_davis_loader.py`) |
| slow tests | 6 — 2 pass (vit-base load + M2 synthetic-band); 4 soft-skip on this host (no SSv2 / no DAVIS) |

### Next

Stage 0.6 is now code-complete except for the **final integration
smoke** (`test_real_pipeline_smoke`): one cell, real ViT-base, 2
SSv2 clips, asserting non-NaN M1/M2 end-to-end. That's a small new
test file plus a small CLI affordance to run the runner against
the SSv2 dataset (the current `--fake` flag is the only switch).
After that, Stage 0.6 closes and we move to Stage 1 (Docker +
RunPod).

Open thread, deferred for later: M3 wiring needs a tubelet-aware
mask alignment — V-JEPA collapses 2 input frames to 1 token row, so
the mask sequence M3 sees needs to align with token rows, not raw
frames. The DAVIS loader hands back `(16, R, R)` masks; M3 will
either consume the first frame of each tubelet pair or majority-vote.
That decision lives in the M3 runner integration, not the loader.

---

## 2026-04-29 — Stage 0.6 Final integration smoke + real-mode CLI

**Stage 0.6 is now code-complete.** Added the integration smoke that
wires real V-JEPA + real SSv2 through the runner and writes a Parquet
shard, plus the CLI affordance for invoking the runner without `--fake`.

### CLI surface

`run_sweep.py` no longer rejects non-`--fake` invocations. Two new args:

- `--device` — `"auto"` (default) / `"cpu"` / `"cuda"` / `"mps"`. Threaded
  into `load_encoder(...)`.
- `--ssv2-cache-dir` — explicit override; falls back to
  `$VJEPA_STRESS_SSV2_DIR`, then `~/data/ssv2/`.

The factories are split into `_build_fake_factories(args)` and
`_build_real_factories(args)`. Real-mode imports (`load_encoder`,
`load_ssv2_dataset`) live inside `_build_real_factories` so a `--fake`
or `--help` invocation never pays the torch.hub / cv2 import cost.

The manifest's `extra` block now records `{"fake": false, "n_videos":
..., "device": ..., "ssv2_cache_dir": ...}` in real mode so a future
reader can tell which mode produced a shard.

**SSv2 subsample seed.** The dataset factory passes `seed=args.master_seed`
into `load_ssv2_dataset`. That deliberately couples the subsample to the
sweep's master seed: a sweep run is now uniquely identified by
`(master_seed, n_videos, on-disk pool)`, so changing master_seed
produces a different video set *and* a different per-(video,perturb)
seed dict — both of which is the right behavior for "rerun with a
different seed."

### `tests/test_real_pipeline_smoke.py`

One test, `@pytest.mark.slow`, soft-skips when no SSv2 tree is present.
Construction:

- One `Cell(model="vjepa-2.1-vit-base", perturbation="gaussian_noise",
  strength=0.5, tier=1)` — gaussian_noise so M1/M2 are *expected* to be
  strictly positive; identity would assert ≈0 but wouldn't prove the
  perturbation is reaching the encoder.
- Real `load_encoder("vjepa-2.1-vit-base", device="cpu")` and
  `load_ssv2_dataset(n_videos=2, seed=0, num_frames=16, image_size=384)`.
- `run_sweep(...)` writes one shard with 2 rows.

Assertions:

- `m1` finite + `> 0` on every row
- `m2` finite + `> 0` on every row
- `m3` is NaN on every row (DAVIS plumbing not yet in the runner; this
  is the pinned placeholder until M3 lands)

### Why a separate `_real_ssv2_available()` helper

Same soft-skip pattern as `test_ssv2_loader.py` and
`test_davis_loader.py`. A single dev box may or may not have SSv2; on
RunPod with cached SSv2 + cached weights, this is the canonical
"the real sweep actually works" check.

### Not yet a calibration test

The smoke deliberately doesn't pin a magnitude band on M1/M2 — those
calibration targets need real SSv2 statistics across many clips, not a
2-clip sample. Calibration is a separate question deferred to Stage 1
(devlog 2026-04-27 Task 3.5 entry covers the M2 reverse-band reasoning).

### Suite state

| | count |
|---|---|
| fast tests | 121 (unchanged) |
| slow tests | 7 — 2 pass on this dev box (vit-base load + M2 synthetic-band); 5 soft-skip (no SSv2 / no DAVIS / no cached calibration data) |

### Stage 0.6 close-out

Stage 0.6 ledger:

| Task | Status |
|---|---|
| 1. perturbations | ✅ Stage 0.5 |
| 2. metrics | ✅ Stage 0.5 |
| 3. V-JEPA loader | ✅ Task 3 (this stage) |
| 3.5. M2 reverse calibration | ⚠️ deferred — calibration test soft-skips, synthetic curiosity pinned at M2 < 0.3 |
| 4. SSv2 loader | ✅ Task 4 |
| 5. DAVIS loader | ✅ Task 5 |
| 6. Final integration smoke + CLI | ✅ this entry |

### Next

Stage 1 — Docker + RunPod. The CLI now actually invokes the real sweep
when `--fake` is not set, so the next entry point is "containerize this
and run a 12-cell tier2 slice on RunPod against real SSv2 + real
weights." Calibration follow-ups (M2 band, M3 mechanism) wait until
the real-feature corpus exists.

---

## 2026-04-29 — Real-feature M2 calibration on DAVIS reverse: prediction confirmed

The deferred Task 3.5 calibration question — *what does M2 actually look
like on real V-JEPA 2.1 features under reverse playback?* — is now
answered. Branch fired: **`in_band`**. Prediction confirmed.

### Setup

- Encoder: `vjepa-2.1-vit-base`, CPU, fp32, eval-mode + frozen.
- Dataset: DAVIS 2017 val (fixed 30 clips). Switched from SSv2 because
  SSv2 is gated, ~50 GB, and lands directly on RunPod in Stage 1+;
  DAVIS is local, ungated, real motion content, fixed 30-clip set.
- Per clip: 16 frames at 384², linspace-sampled. Reverse via `[::-1]`.
- Metric: `compute_m2(F_clean, F_reversed)` — locked, untouched.
- Script: `scripts/calibrate_m2_real.py` (commit fixed in this entry).
- Output: `data/calibration/m2_reverse_calibration.json` +
  `logs/m2_calibration.log`.

### Result

| metric | mean | median | std | min | max |
|---|---|---|---|---|---|
| **M2** | **1.0200** | 1.0304 | 0.0360 | 0.8949 | 1.0664 |
| M1 | 0.2751 | 0.2785 | 0.0459 | 0.1612 | 0.3840 |

All 30 per-clip M2 values fall inside `[0.9, 1.4]`. The std (0.036) is
remarkably tight — a 13% spread (0.89→1.07) over content as
heterogeneous as `bike-packing`, `breakdance`, `goldfish`,
`scooter-black`, `motocross-jump`. The prediction in RESEARCH.md §3.2 of
~1.14 is slightly above what we observed (1.02), but well within the
defined `[0.9, 1.4]` band.

### Per-clip lows worth noting

Three clips landed below 0.95:

- `dogs-jump` 0.8949
- `judo` 0.9434
- `pigs` 0.9541
- `bike-packing` 0.9644

These are the lowest M1 clips too (`dogs-jump` M1=0.16; `judo` M1=0.20),
which is consistent with the analytical story: when per-frame structural
drift is small, the temporal-gradient direction is harder to flip cleanly,
so M2 retreats slightly toward 0. None of them are pathologically low and
all sit comfortably above the H2 threshold (M2 > 0.8).

### What this means for the rest of the study

- **H2 (Reverse Sensitivity, threshold M2 > 0.8) is well-grounded.** All
  30 DAVIS clips clear it with margin. The H2 calibration probe will
  produce signal, not noise.
- **The synthetic ~0.10 result from Task 3.5 stays as a structural
  finding**, not a bug. The real-video number is an order of magnitude
  larger and lands where RESEARCH.md predicted, which is the only
  evidence we needed that the synthetic gap was about feature-entropy
  collapse on monotonic stimuli, not a metric / pipeline issue.
- **Diagnostics did not run** (the `below_floor` branch was not hit), so
  the script's identity / cross-distribution / random-features paths are
  untested in practice. They remain wired for the unlikely case where a
  Stage-1 follow-up on a different model variant or on SSv2 fires
  `below_floor`.

### Pre-registration discipline

We predicted [0.9, 1.4]. We observed mean=1.020, all 30 clips in band.
RESEARCH.md is unchanged. H2's 0.8 threshold is unchanged. M2 is
unchanged. The calibration JSON is the artifact-of-record:

    data/calibration/m2_reverse_calibration.json

### Next

Calibration unblocks Stage 1 (Dockerfile + RunPod image). The remaining
calibration follow-up — same number on a real-motion natural-class
distribution like SSv2 — happens on RunPod once SSv2 is downloaded
there. The script's diagnostic-2 slot is reserved for that follow-up.

---

## 2026-04-29 — Stage 1: Dockerfile, .dockerignore, RunPod runbook, pilot script

Stage 1 lands the operational surface for getting `vjepa-stress` onto a
RunPod GPU box. No new science — just plumbing so the calibrated pipeline
from Stage 0.6 can move off the laptop and onto a real GPU.

### What changed

1. **`.dockerignore` (new).** Without it, `docker build .` slurps the
   ~800 MB DAVIS tree under `data/`, the host `.venv/`, and the git
   history into the build context every time. The new file mirrors
   `.gitignore`'s defensive list (datasets, weights, archives, caches,
   editor cruft) plus excludes `*.md` (with `!README.md` allowlist) since
   docs aren't needed at runtime. Build context should now be single-digit
   MB.
2. **Dockerfile — `TORCH_HOME` fix (real bug).** The previous version set
   `HF_HOME=/workspace/.cache/huggingface` but did not set `TORCH_HOME`.
   V-JEPA 2.1 weights load via `torch.hub.load(...)`, which uses
   `~/.cache/torch/hub` by default — i.e. `/root/.cache/torch` inside
   the container, which is *not* on the persistent volume. Without the
   fix, every container restart would re-download ~1.5 GB (vit-large) or
   ~6 GB (vit-gigantic). Now pinned to `/workspace/.cache/torch` so the
   weight cache lives on the RunPod volume.
3. **Dockerfile — env contract.** Added `VJEPA_STRESS_SSV2_DIR=/workspace/data/ssv2`
   and `VJEPA_STRESS_DAVIS_DIR=/workspace/data/davis` so the loaders
   resolve to the conventional in-container paths without a CLI flag.
   Dropped the deprecated `TRANSFORMERS_CACHE` env (newer transformers
   reads `HF_HOME` directly).
4. **docker-compose parity.** Added a `torch-cache` named volume mounted
   at `/workspace/.cache/torch`, plus the `TORCH_HOME` env, so local
   `docker compose up` matches the RunPod env contract.
5. **`scripts/pilot_runpod.py` (rewrite).** The old version imported
   `vjepa_stress.config.SweepConfig`, which doesn't exist — Stage 0.6
   moved to `RunnerConfig` with explicit factories. The new pilot script:
   - Builds a 10-cell Tier-1 slice (1 model × 1 perturbation × all 10
     strengths) directly from `STRENGTHS`.
   - Loads the real `load_encoder` and `load_ssv2_dataset` lazily inside
     `main()` so `--help` is cheap.
   - Wires through `RunnerConfig` with the same `extra_manifest` shape
     as `vjepa_stress.run_sweep`'s real-mode path, plus a `pilot=True`
     flag so the manifest is unambiguous.
   - Defaults: `vit-large`, `gaussian_noise`, `n_videos=4`, `device=auto`.
6. **`docs/runpod.md` (new).** Operational runbook covering: build →
   push → provision → first-run bootstrap (warm V-JEPA caches up-front
   so a cold cache miss doesn't kill a cell mid-sweep) → pilot smoke →
   full sweep. Includes the explicit "if pilot M1 doesn't trend with
   strength, **stop**" gate.

### Layer caching trade-off (recorded, not fixed)

Current `Dockerfile` does:

    COPY pyproject.toml README.md /workspace/
    COPY vjepa_stress /workspace/vjepa_stress
    RUN pip install -e ".[dev]"

Any change under `vjepa_stress/` invalidates the COPY layer and re-runs
`pip install`. With pip's cache warm and deps unchanged, this is fast
(~10s — pip is just relinking the editable package). The "proper" fix
(BuildKit `--mount=type=bind` or a stub-package + `--no-deps` two-step)
is deferred until a build wall-time complaint actually appears.

### What's deferred to Stage 2+

- **Image SHA → manifest auto-capture.** Right now the runner records
  whatever `extra_manifest` we pass; the runbook says "record the digest
  manually." Stage 2 should pin this so a manifest can never be
  mis-attributed to the wrong image.
- **`scripts/run_full_sweep.py`.** Still a `NotImplementedError` stub.
  The Stage-0.6 CLI (`python -m vjepa_stress.run_sweep --matrix all`)
  covers the same ground; the dedicated entry-point is Stage 4.
- **A Dockerfile-level smoke test.** Building the image and running
  `pytest tests/test_real_pipeline_smoke.py -m slow` inside it is what
  validates the image, but we don't have CI for that — only the
  `docs/runpod.md` checklist.

### Verification done locally

- `python scripts/pilot_runpod.py --help` runs without importing torch
  (lazy imports confirmed).
- `python -c "import ast; ast.parse(open('scripts/pilot_runpod.py').read())"`
  parses clean.
- `.dockerignore` syntax validated by inspection (Docker treats it like
  `.gitignore`; same patterns).

The image itself is not built locally — Apple Silicon doesn't run the
NGC CUDA base. Build verification happens on RunPod.

### Next

Stage 1 unblocks the actual RunPod sweep run. Concrete next steps:

1. Build image on a Linux box (CI or local), push to a registry.
2. Spin up a RunPod H100 instance, attach a 200 GB volume.
3. Download SSv2 validation split to `/workspace/data/ssv2/`.
4. Warm V-JEPA weight caches (the `for variant in ...` block in
   `docs/runpod.md` §5).
5. Run `tests/test_real_pipeline_smoke.py -m slow` inside the image.
6. Run the pilot (`scripts/pilot_runpod.py`).
7. Inspect M1 vs strength; if monotonic, launch the 222-cell sweep.

---
