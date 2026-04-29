# RunPod runbook

This is the Stage-1 operational doc for getting `vjepa-stress` onto a RunPod
GPU box. The image is reproducible, the data is mounted at runtime, and a
single-cell pilot exists to gate the 222-cell sweep.

The flow is: **build locally → push to a registry → run on RunPod with a
persistent volume**. Datasets and model weights never live in the image —
they stay on the volume so they survive container restarts.

---

## 1. What goes where

| Artifact                 | Lives in                      | How it gets there                |
| ------------------------ | ----------------------------- | -------------------------------- |
| Source + deps            | The image                     | `docker build` (this doc)        |
| V-JEPA 2.1 weights       | `/workspace/.cache/torch`     | `torch.hub.load(...)` on first run |
| HuggingFace cache        | `/workspace/.cache/huggingface` | First run that touches HF        |
| SSv2 validation videos   | `/workspace/data/ssv2/validation/` | Manual download (see §4)        |
| DAVIS 2017 val (optional)| `/workspace/data/davis/DAVIS/` | Public download (see §4)         |
| Sweep output (Parquet)   | `/workspace/runs/<sweep>/`    | The runner writes shards here    |

The Dockerfile pins `TORCH_HOME`, `HF_HOME`, `VJEPA_STRESS_SSV2_DIR`, and
`VJEPA_STRESS_DAVIS_DIR` to paths under `/workspace`, so as long as the
RunPod volume is mounted at `/workspace` everything persists across
container restarts.

---

## 2. Build the image

From the repo root:

```bash
docker build -t vjepa-stress:latest .
```

The base is `nvcr.io/nvidia/pytorch:24.10-py3` (CUDA 12.x + PyTorch 2.4+).
First build pulls ~10 GB; subsequent builds reuse the cached deps layer
unless `pyproject.toml` or `vjepa_stress/` source changes.

The `.dockerignore` keeps the build context lean — `data/`, `logs/`,
`.venv/`, `*.zip`, and the git history are all excluded. Verify if
suspicious:

```bash
docker build --no-cache --progress=plain -t vjepa-stress:latest . 2>&1 | grep "load build context"
```

A healthy build context is single-digit MB.

---

## 3. Push to a registry

RunPod pulls from Docker Hub, GHCR, or any registry that accepts a
`docker pull` with credentials configured on the pod. Tag with a SHA so
each sweep is anchored to an exact image:

```bash
SHA=$(git rev-parse --short HEAD)
docker tag vjepa-stress:latest <your-registry>/vjepa-stress:${SHA}
docker push <your-registry>/vjepa-stress:${SHA}
```

Record the digest in the sweep manifest (the runner writes `manifest.json`
into the output dir; image SHA goes there manually for now — Stage 2 will
auto-capture it).

---

## 4. Provision the pod

**Template choice.** "Custom Container" with the pushed image. Don't pick
a generic PyTorch template — the deps inside it will fight with ours.

**GPU sizing.** Per [EXPERIMENT.md](../EXPERIMENT.md), the heaviest cell
is `vit-gigantic` at 16 frames @ 384²; that fits on a single H100 in
fp32. A4000/A5000 work for `vit-base`/`vit-large` only. The full Tier-1+2+3
sweep on a single H100 is on the order of 1–2 hours of GPU time.

**Volume.** 200 GB is plenty (SSv2 is ~50 GB, weights are ~10 GB across
all four variants, sweep output Parquet is single-digit GB). Mount at
`/workspace`.

**Ports.** None needed. The runner is a CLI; ssh in and run things.

---

## 5. First-run bootstrap

Once the pod is up:

```bash
# Inside the container
cd /workspace

# Sanity: deps installed, runner imports cleanly
python -c "from vjepa_stress.sweep.runner import run_sweep; print('ok')"

# Warm the V-JEPA weight cache. Each variant is ~1.5–6 GB, so do it
# once up-front rather than during the sweep where a download failure
# would abort a cell mid-run.
python - <<'PY'
from vjepa_stress.models.vjepa import load_encoder
for variant in ("vit-base", "vit-large", "vit-giant", "vit-gigantic"):
    print(f"loading vjepa-2.1-{variant} ...", flush=True)
    load_encoder(f"vjepa-2.1-{variant}", device="cpu")
PY

# SSv2 — the loader fails fast with this URL when the tree is missing.
# It's gated; download the validation split manually:
#   https://developer.qualcomm.com/software/ai-datasets/something-something
# Extract to /workspace/data/ssv2/validation/<id>.webm
ls /workspace/data/ssv2/validation | head -5
```

Once SSv2 is on disk, run the smoke test inside the container:

```bash
pytest tests/test_real_pipeline_smoke.py -m slow -v
```

This loads `vit-base`, runs one cell on two SSv2 clips, and asserts the
M1/M2 contract. If it passes, the box is wired correctly.

---

## 6. Run the pilot

The pilot exercises one model × one perturbation × all 10 strengths
against a small SSv2 subsample. It catches per-cell-strength bugs that a
single-cell smoke wouldn't:

```bash
python scripts/pilot_runpod.py \
    --output /workspace/runs/pilot \
    --device cuda \
    --n-videos 4
```

Expected: 10 shards under `/workspace/runs/pilot/`, `manifest.json`
recording `pilot=true`, M1 monotonically increasing with strength.
Inspect quickly:

```bash
python - <<'PY'
from vjepa_stress.sweep.checkpoint import load_shards
df = load_shards("/workspace/runs/pilot").df()
print(df[["cell_id", "strength", "m1", "m2"]].sort_values("strength"))
PY
```

If `m1` doesn't trend up with strength, **stop** — don't launch the full
sweep. Either the perturbation isn't reaching the encoder, the seed dict
is wrong, or the metric implementation drifted.

---

## 7. Run the full sweep

Stage 4 wiring (the 222-cell driver) lands separately; for now the
canonical entry point is the CLI:

```bash
python -m vjepa_stress.run_sweep \
    --matrix all \
    --output /workspace/runs/sweep-${SHA} \
    --device cuda \
    --n-videos 200
```

`--n-videos` here is the SSv2 subsample size *per cell* — the runner
reuses one fixed subsample across all cells (deterministic via
`master_seed`), so the actual data cost is one subsample, not 222 of them.

The runner is resumable: cells whose Parquet shard already exists are
skipped. Crash recovery is just re-running the same command.

---

## 8. Tear down

Datasets and weights live on the persistent volume — leave them. Stop or
delete the pod when done; next sweep just re-attaches the same volume to
a fresh image SHA.

---

## Troubleshooting

| Symptom                                                     | Likely cause                                                                                   |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| V-JEPA weights re-download every restart                    | `TORCH_HOME` not set, or `/workspace/.cache/torch` not on the persistent volume                |
| `FileNotFoundError: SSv2 split dir not found at /workspace/data/ssv2/validation` | Download SSv2 manually; see §5                                                                 |
| `Use of `localhost:8300`` in V-JEPA hub URL                 | `vjepa_stress.models.vjepa.patch_meta_url` should fix this; if not, check the hub cache repo   |
| pip install slow on rebuild                                 | Either `pyproject.toml` changed, or the build context is too large — check `.dockerignore`     |
| Pilot M1 flat across strengths                              | Perturbation isn't reaching the encoder; bisect with `--matrix tier2` (identity/reverse probes)|
