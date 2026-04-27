# vjepa-stress

Frozen-encoder feature-stability sweep across all four V-JEPA 2.1 variants under
nine controlled perturbations, with downstream validation on DAVIS patch
correspondence. The aim is to characterize how representation geometry deforms
under sensor, temporal, spatial, and codec stress, and to test whether scale
correlates with robustness once the encoder is held fixed.

## What's in this repo

- [`RESEARCH.md`](RESEARCH.md) — methodology, metrics (M1, M2, M3), and the
  theoretical justification for the perturbation taxonomy.
- [`EXPERIMENT.md`](EXPERIMENT.md) — the staged execution plan, the experiment
  matrix, and the pre-registered predictions.
- [`manifests/predictions.md`](manifests/predictions.md) — pre-registered
  hypotheses, recorded before any sweep is run.
- `src/` — perturbations, metrics, sweep harness, model loaders.
- `scripts/` — Stage 0 verification, RunPod pilot, full sweep entry point.
- `tests/` — unit tests for perturbation contract and metric sanity.

## Status

Pre-registration; sweep not yet executed. Stage 0 (perturbation and metric
implementation) is the next task.

## Quickstart

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Verify the perturbation pipeline on a sample clip
python scripts/verify_perturbations.py

# Verify metric sanity on synthetic features
python scripts/verify_metrics.py

# Sanity-check the experiment matrix
python -m src.sweep.matrix
```

## Reproducing

Every sweep run writes a manifest to `manifests/manifest.json` capturing seeds,
HuggingFace model commit hashes, dataset version hashes, library versions, and
the exact list of video IDs used. See
[`manifests/manifest.template.json`](manifests/manifest.template.json) for the
schema. Reproducing a result means re-running with the same manifest pinned.

## Citation

```bibtex
@misc{kolasinski2026vjepastress,
  author = {Kolasinski, Taylor},
  title  = {V-JEPA 2.1 Robustness: Frozen-Encoder Feature Stability Under
            Controlled Perturbations},
  year   = {2026},
  note   = {Poisson Labs technical report. URL pending.}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
