# V-JEPA 2.1 loading spike — findings

Stage 0.6 Task 2 result. Goal was to decide D1 (loading mechanism) before
writing loader code. **Decision: `torch.hub` with a URL monkey-patch.**

The auto-generated raw output from `scripts/spike_vjepa_loading.py` is at
the bottom of this file; the human-edited summary above interprets it.

## TL;DR for D1

| Mechanism | Status | Verdict |
|-----------|--------|---------|
| Hugging Face Hub | ❌ no V-JEPA 2.1 models published as of 2026-04-27 | Not viable |
| `torch.hub.load('facebookresearch/vjepa2', '<entry>')` | ✅ entries exist; weights download succeeds **after a one-line URL fix** in the cached repo | **Primary** |
| Direct `dl.fbaipublicfiles.com/vjepa2/*.pt` | ✅ all four 2.1 checkpoints reachable; URLs discoverable from the hub repo's `ARCH_NAME_MAP` | Use as fallback inside the loader |

D2 (resolution) is also pinned by the spike: torch.hub function names end
in `_384`, the encoder constructs at `img_size=(384, 384)` only. **Lock D2
at 384×384.**

## What broke and how we worked around it

`facebookresearch/vjepa2` ships the V-JEPA 2.1 hubconf entries with
`VJEPA_BASE_URL = "http://localhost:8300"` — Meta left a dev-server stub
in place when they merged the 2.1 hub code. The production URL is
commented out one line above:

```python
# in the cached repo: src/hub/backbones.py:8-11
# VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
#
# for testing
VJEPA_BASE_URL = "http://localhost:8300"
```

A `torch.hub.load(..., pretrained=True)` therefore tries `http://localhost:8300/<filename>.pt` and fails with `Connection refused`.

**Workaround for the spike** (one-shot, manual):
```bash
sed -i.bak 's|VJEPA_BASE_URL = "http://localhost:8300"|VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"|' \
  ~/.cache/torch/hub/facebookresearch_vjepa2_main/src/hub/backbones.py
```

**Workaround for Task 3 (encoder loader)**: monkey-patch the constant in
`sys.modules` *before* calling `torch.hub.load(..., pretrained=True)`. The
helper module `vjepa_stress.utils.hub_isolation` already wraps the import
collision; the loader can import `src.hub.backbones`, set
`VJEPA_BASE_URL`, and call the entrypoint. Detail in the loader.

## V-JEPA 2.1 weights — direct URLs

Recovered from the hub repo's `ARCH_NAME_MAP`. All four are public on
`https://dl.fbaipublicfiles.com/vjepa2/`:

| Variant | Filename | Size | Notes |
|---------|----------|------|-------|
| `vjepa2-2.1-vit-base` | `vjepa2_1_vitb_dist_vitG_384.pt` | 1.55 GB | distilled from the ViT-G teacher |
| `vjepa2-2.1-vit-large` | `vjepa2_1_vitl_dist_vitG_384.pt` | 4.80 GB | distilled |
| `vjepa2-2.1-vit-giant` | `vjepa2_1_vitg_384.pt` | 15.7 GB | native 1B (`giant_xformers` arch) |
| `vjepa2-2.1-vit-gigantic` | `vjepa2_1_vitG_384.pt` | 28.2 GB | native 2B (`gigantic_xformers` arch) |

These were *not* in our pre-flight HEAD probe set — `dl.fbaipublicfiles.com`
returned 403 for every guess we made because we didn't know the
`vitb_dist_vitG`/`vitl_dist_vitG` naming until we read the cached source.
The fbaipublicfiles bucket is open; you just have to know the exact name.

## Forward-pass contract (vit-base, Mac M4 CPU)

```
torch.hub.load("facebookresearch/vjepa2", "vjepa2_1_vit_base_384", trust_repo=True)
  -> tuple(encoder, predictor)

encoder type   : VisionTransformer
encoder.embed_dim: 768
forward call   : encoder(torch.randn(1, 3, 2, 384, 384)) on CPU
output         : Tensor shape (1, 576, 768) dtype float32
forward dt     : 0.12 s for T=2 frames, batch=1, on M4 CPU
```

**Key shape facts:**

- Input convention is `(B, C, T, H, W)` (channels-first, 5-D), not
  `(B, T, C, H, W)`. The `FakeEncoder.encode` shim takes `(T, H, W, C)`
  uint8; the real loader needs to permute and unsqueeze.
- Patch grid at 384×384 is **24×24 = 576** spatial patches per tubelet.
  ✓ matches the loader-spec assertion in `STAGE_0.6_TASKS.md`.
- The encoder uses **`tubelet_size=2`** by default, so two consecutive
  frames are collapsed into a single token row. Output is
  `(B, T_tubelet × 576, D)` where `T_tubelet = T / 2`. At T=2 this
  yields one tubelet → `(1, 576, 768)`. At our planned T=16 it will be
  `(1, 8 × 576, 768) = (1, 4608, 768)`.
- **This is significant for M2.** Our metric definition assumes a
  per-frame delta `δ_t = F_{t+1} - F_t`. With tubelet collapsing the
  features live on tubelets, not frames. Either: (a) re-index our T axis
  in the row schema as "tubelet", (b) override `tubelet_size=1` if the
  hub kwargs allow (unverified — flag for Task 3).

## Hub deps (now pinned in pyproject)

`facebookresearch/vjepa2` raises `RuntimeError: Missing dependencies:
timm, einops` if either is absent. These are not in `torch>=2.4`; we
added them to `[project.dependencies]` in this commit.

## What the spike did *not* test

- Larger variants (large, giant, gigantic). Skipped per Taylor's prior:
  download those on RunPod where bandwidth is free and disk is ephemeral.
  Forward-pass APIs are identical; only `embed_dim` and arch keys change.
- MPS (Apple GPU). `model.to("mps")` may or may not survive the bf16
  defaults. Defer to Task 3 — Mac stays on CPU/f32 for development; CUDA
  is the production target.
- decord. Confirmed unbuildable on darwin/arm64; the `pyproject.toml`
  already has the `sys_platform != 'darwin'` guard. Stage 0.6 SSv2 loader
  will use OpenCV on darwin.

---

## Auto-generated raw output

(From the most recent `python scripts/spike_vjepa_loading.py --full` run.
Re-running the script overwrites this file — the human notes above are
preserved manually.)

### torch.hub.list — V-JEPA 2.1 entries (4 / 10)

```
vjepa2_1_vit_base_384
vjepa2_1_vit_giant_384
vjepa2_1_vit_gigantic_384
vjepa2_1_vit_large_384
```

### torch.hub.list — V-JEPA 2.0 entries (contrast)

```
vjepa2_ac_vit_giant
vjepa2_preprocessor
vjepa2_vit_giant
vjepa2_vit_giant_384
vjepa2_vit_huge
vjepa2_vit_large
```

### HF `list_models(author='facebook', search='vjepa', limit=100)`

V-JEPA 2.1 (strict filter — requires `vjepa2-1`/`vjepa-2-1`/etc. token):
**0 hits.**

V-JEPA 2.0 (10 hits, contrast):
```
facebook/sparsh-vjepa-base
facebook/sparsh-vjepa-small
facebook/vjepa2-vitg-fpc32-384-diving48
facebook/vjepa2-vitg-fpc64-256
facebook/vjepa2-vitg-fpc64-384
facebook/vjepa2-vitg-fpc64-384-ssv2
facebook/vjepa2-vith-fpc64-256
facebook/vjepa2-vitl-fpc16-256-ssv2
facebook/vjepa2-vitl-fpc32-256-diving48
facebook/vjepa2-vitl-fpc64-256
```

### `dl.fbaipublicfiles.com/vjepa2/<file>.pt` HEAD probes

Initial guesses (all 403 except the V-JEPA-2.0 sanity probe):
```
vjepa2_1_vitbase_384.pt        403
vjepa2_1_vitlarge_384.pt       403
vjepa2_1_vitgiant_384.pt       403
vjepa2_1_vitgigantic_384.pt    403
vitbase_2_1_384.pt             403
vitlarge_2_1_384.pt            403
vitgiant_2_1_384.pt            403
vitgigantic_2_1_384.pt         403
v2_1_vitbase_384.pt            403
v2_1_vitlarge_384.pt           403
vitg-384.pt                    200  16,464,301,382  (V-JEPA 2.0 sanity)
vitl-384.pt                    403
```

Filenames recovered from the cached hub repo (all 200, sizes from `Content-Length`):
```
vjepa2_1_vitb_dist_vitG_384.pt   200    1,664,223,428
vjepa2_1_vitl_dist_vitG_384.pt   200    5,151,198,524
vjepa2_1_vitg_384.pt             200   16,878,318,788
vjepa2_1_vitG_384.pt             200   30,238,058,912
```

### hub.load `vjepa2_1_vit_base_384` (after URL patch)

```
loaded in 19.5 s, type=tuple
  encoder: VisionTransformer (embed_dim=768)
  predictor: VisionTransformerPredictor

encoder(torch.randn(1, 3, 2, 384, 384))
  → Tensor (1, 576, 768) float32, dt=0.12 s
```
