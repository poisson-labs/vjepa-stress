# Stage 0.6 — Task Ledger

**Goal of Stage 0.6:** replace `FakeEncoder` and `FakeVideoDataset` with a
real V-JEPA 2.1 loader and real SSv2 + DAVIS data pipelines so the
runner from Stage 0.5 can write real metrics. After this stage, the
runner is ready for the actual sweep; Stage 1 is then Docker + RunPod
configuration on top of working code.

This file is a handoff ledger. A fresh agent (or future-Taylor) should
be able to pick up mid-stream by reading top-to-bottom: each task lists
its preconditions, acceptance criteria, current status, and the
decision it unblocks. The **Findings** sub-sections are append-only —
results from Task 2's pre-flight, the loader spike, etc. land there.
Higher-level decisions and supersedes go in `devlog.md`, not here.

**Status legend:** `[ ]` not started · `[~]` in progress · `[x]` done ·
`[!]` blocked (see note).

---

## Decisions still open

| ID | Decision | Blocks | Resolved by |
|----|----------|--------|-------------|
| D1 | torch.hub vs HF vs direct-checkpoint download for V-JEPA 2.1 | Task 3 (encoder loader) | Task 2 pre-flight + spike |
| D2 | Input resolution 384×384 (recommended) vs 224×224 | Task 3, Task 4, Task 5 | Resolved during Task 3 once D1 lands; default 384 |
| D3 | SSv2 download UX: manual-only vs auto with token | Task 4 | During Task 4 — current prior is manual-only |
| D4 | Whether to harmonize `vit-b/vit-l` → `vit-base/vit-large` (Task 1) — recorded for reference, action item | Tasks 2-5 (loader uses suffix substitution) | Task 1 (this ledger) |

---

## Task 1 — Naming consistency rename

**Status:** `[ ]`

**Goal.** Rename `vjepa-2.1-vit-b` → `vjepa-2.1-vit-base` and
`vjepa-2.1-vit-l` → `vjepa-2.1-vit-large` everywhere so the four
identifiers follow a single suffix rule (`{base, large, giant,
gigantic}`). The Stage 0.6 loader will derive torch.hub entry names by
string substitution: `f"vjepa2_1_vit_{suffix}_384"`.

**Preconditions.** None — pure refactor.

**Files in scope.**
- `src/sweep/matrix.py` — `MODELS` tuple and `TIER3_MODEL`.
- `src/testing/fakes.py` — `FAKE_EMBED_DIMS`.
- `src/models/vjepa.py` — `VARIANT_REPOS`.
- `src/config.py` — `ModelConfig.variant` default.
- `manifests/manifest.template.json` — `model_revisions` keys.
- `tests/test_schema.py` and `tests/test_runner.py` — fixture strings.
- `RESEARCH.md`, `EXPERIMENT.md` if they reference the names.

**Watch-outs.**
- Use the **quoted** form (`"vjepa-2.1-vit-b"`) when matching, never
  bare `vit-b`. The string `vit-b` is a prefix of `vit-base`, so a
  naive replace_all introduces double-renames.

**Acceptance criteria.**
- All 87 Stage 0.5 tests still pass.
- `python -m src.run_sweep --matrix tier2 --output /tmp/x --fake
  --n-videos 2` writes 12 distinct shards on a fresh dir; filenames
  use the new identifiers.
- A `grep -n "vit-b\b\|vit-l\b"` returns nothing under `src/`,
  `tests/`, `manifests/`.

**Commit.** Single commit:
`rename: vit-b/l → vit-base/large for Meta canonical consistency`.

**Findings.** _(append after completion)_

---

## Task 2 — torch.hub spike on Mac M4

**Status:** `[ ]`

**Goal.** Decide D1 (loading mechanism) before writing loader code.

**Preconditions.** Task 1 complete (so the spike script uses canonical
suffixes).

### 2a — Pre-flight (no downloads)

These are all free — no weight downloads, ~5 min total:

1. **torch.hub.list V-JEPA 2.1 entries.** `torch.hub.list(
   'facebookresearch/vjepa2', trust_repo=True)`. Filter for entries
   matching `vjepa2_1_vit_*_384`. If `gigantic` is in the list, the
   path probably works for all four; if only `vjepa2_vit_*` (no `_1_`)
   is in the list, V-JEPA 2.1 isn't on torch.hub yet.
2. **torch.hub.list V-JEPA 2 entries** (no `_1_` infix). Establishes
   the contrast — confirms what's on hub today vs. what's been added.
3. **`HfApi().list_models(author='facebook', search='vjepa2-1')`**.
   The HF integration request is open as of 2026-03-20; cheap second
   source in case weights landed without doc updates.
4. **HEAD checks against `dl.fbaipublicfiles.com/vjepa2/`** for
   plausible 2.1 filenames (`vjepa2_1_vitbase_384.pt`,
   `vjepa2_1_vitlarge_384.pt`, `vjepa2_1_vitgiant_384.pt`,
   `vjepa2_1_vitgigantic_384.pt`, plus the `_1` and underscored
   variants). The V-JEPA 2 README confirmed `vitg-384.pt` exists at
   that URL pattern; if 2.1 weights are anywhere, this is the third
   most-likely place.

**Output.** Hand the four results back as a small table; await user
decision on whether to proceed with downloads (and which).

### 2b — Confirmation download (one variant only, prior is `vit-base`)

Conditional on 2a; user picks the variant.

**Goal.** Run a forward pass on synthetic `(16, 3, 384, 384)`, log
output shape + dtype + wallclock. Confirms the API contract; the
remaining variants are the same arch at scale.

**Hard stops.**
- If the chosen variant fails to download or load — stop and ask.
- If MPS rejects the model dtype (some HF defaults are bf16 which MPS
  does not fully support) — stop and ask. Likely workaround: force
  float32 on Mac, bf16 on CUDA.

**Output deliverable.** `docs/loading_spike.md` with:
- Result table (variant × {hub, HF, direct, none}).
- Forward-pass numbers.
- Recommended primary mechanism + one-sentence justification.
- Known constraints (memory ceiling, dtype quirks, decord skip on
  darwin).

**Decision.** D1 lands here. Update the table at the top of this file.

---

## Task 3 — Real encoder loader

**Status:** `[ ]`

**Preconditions.** Task 2 complete; D1 and D2 (resolution) recorded.

**Goal.** Replace the `NotImplementedError` body in
`src/models/vjepa.py` with a working `load_encoder(model_name, device,
dtype)` that returns an object satisfying the same protocol as
`FakeEncoder` (`encode(np.ndarray) -> torch.Tensor`,
`forward_calls: int`).

**Implementation notes.**
- Define an `Encoder` Protocol/ABC explicitly so `FakeEncoder` and the
  real encoder are interchangeable; the runner's `_EncoderProtocol`
  type hint should point at it.
- `_load_via_hub` / `_load_via_hf` / `_load_via_direct` private
  functions; `load_encoder` is a thin switch keyed off D1.
- `VARIANT_REPOS: dict[str, str]` is module-level, explicit, greppable
  (do not synthesize at runtime via f-string only).
- Expose `.embedding_dim` for sanity checks: 768 / 1024 / 1408 / 1664
  for base / large / giant / gigantic.
- At 384×384 with patch 16, expect 24×24 = 576 spatial patches per
  frame. (At 224×224 it would be 14×14 = 196.) Whichever D2 resolves
  to, lock the assertion in the loader.
- `try_forward_pass(model, device)` runs synthetic `(1, 3, 384, 384)`
  at load time so OOM surfaces immediately, not deep in the sweep.
- Default dtype: CUDA→bf16, MPS→f32, CPU→f32.

**Tests** in `tests/test_vjepa_loader.py`:
1. `test_load_smallest_works` — load `vit-base` on whatever device is
   available, run encode, check shape `(T, 576, embedding_dim)`.
2. `test_load_returns_correct_embedding_dim`.
3. `test_unknown_model_name_raises` — clear message naming valid keys.
4. `test_dtype_defaults_per_device` — pure unit, no real load.

CI runs only `vit-base`; bigger sizes are pilot-time concerns.

**Acceptance.** All four tests pass; the runner can swap
`FakeEncoder` → `load_encoder("vit-base")` and produce non-NaN M1/M2
on a synthetic clip.

---

## Task 4 — Real SSv2 loader

**Status:** `[ ]`

**Preconditions.** Task 3 complete (so we know the resolution).

**Goal.** `src/data/ssv2.py::load_ssv2_dataset(n_videos=200, seed=0,
split="validation", cache_dir=None) -> SSv2Dataset` matching the
`FakeVideoDataset` contract.

**Constraints.**
- SSv2 is gated; ~50 GB. **Strong prior: manual-only download** — clear
  error pointing at install instructions if `~/data/ssv2/` is missing.
  Document in README that SSv2 must be downloaded out-of-band before
  real sweeps.
- Decode via OpenCV (`cv2.VideoCapture` + `np.linspace`-spaced 16
  frames). decord is gated to non-darwin in `pyproject.toml`; do not
  add a darwin shim.
- Sample with `random.Random(seed).sample(all_video_ids, 200)` then
  `sorted(...)` for stability. Record the resulting list in the
  manifest's `extra` block.
- Output shape `(16, R, R, 3)` uint8 where R is the resolution from D2.

**Tests.**
1. `test_load_with_missing_data_raises` — clear error w/ install URL.
2. `test_load_returns_correct_count` — `@pytest.mark.slow`.
3. `test_load_is_deterministic` — `@pytest.mark.slow`.
4. `test_video_shape` — mock filesystem, runs in CI.

---

## Task 5 — Real DAVIS loader

**Status:** `[ ]`

**Preconditions.** Task 3 complete.

**Goal.** `src/data/davis.py::load_davis_dataset(seed=0, split="val",
cache_dir=None) -> DAVISDataset` yielding `(video_id, frames, masks)`
where `masks: (T, R, R) int32` (0 = bg, ≥1 = instance IDs).

**Constraints.**
- DAVIS 2017, ~1 GB, public download from
  `https://davischallenge.org/davis2017/code.html`. No gating.
- Frames and masks must share the same `np.linspace` index sample so
  M3 keypoint tracking is consistent.

**Tests.**
1. `test_load_returns_30_videos`.
2. `test_mask_shape_matches_frame_shape`.
3. `test_mask_values_are_integers`.
4. `test_video_id_is_stable_string`.

---

## Final integration smoke

**Status:** `[ ]`

**Preconditions.** Tasks 3–5 complete.

`tests/test_real_pipeline_smoke.py::test_smoke_real_pipeline`:

```python
@pytest.mark.slow
def test_smoke_real_pipeline():
    # 1 cell, real ViT-base, 2 SSv2 clips. Assert non-NaN m1, m2.
```

**Acceptance.** The CLI invocation
`python -m src.run_sweep --matrix tier2 --output /tmp/test_real/
--n-videos 2` (with `--fake` removed and a real-encoder switch wired
in) writes shards with non-NaN M1/M2 on real ViT-base + real SSv2.

---

## Devlog handoff checklist

When Stage 0.6 closes, the devlog entry should record:
- Outcome of Task 2 (mechanism + why).
- Resolution choice (D2).
- Naming-rename completion.
- Any constraints discovered (memory ceiling, dtype, decord, OOMs).
- Test count delta (from 87 → ?).
- One-cell wallclock on Mac M4 for Tier-1 ViT-base × 200 SSv2 (so
  RunPod estimates can be calibrated).

---

## Findings log _(append-only)_

_(empty)_
