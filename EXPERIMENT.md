# EXPERIMENT.md

> **Placeholder.** This file should contain the full experiment plan for
> vjepa-stress. The scaffolding expects the following structure:
>
> - §1 — Stage breakdown (Stage 0 verification, Stage 1 dry run, Stage 2 RunPod
>   pilot, Stage 3 calibration, Stage 4 full sweep, Stage 5 analysis)
> - §2 — Experiment matrix (must total 222 cells; see `src/sweep/matrix.py`)
> - §3 — Compute budget and RunPod plan (A100 80GB Community Cloud)
> - §4 — Per-cell artifact schema (Parquet rows the runner emits)
> - §5 — Stop conditions and failure modes
> - §6 — Analysis plan (notebooks, headline plots)
> - §7 — Pre-registered predictions (mirrored into `manifests/predictions.md`)
>
> Drop the source document into this path before the first sweep is launched.
> The matrix in `src/sweep/matrix.py` and the schema in
> `manifests/manifest.template.json` should stay in sync with §2 and §4.
