# RESEARCH.md

> **Placeholder.** This file should contain the full research methodology
> document for vjepa-stress. The scaffolding tooling expects the following
> sections to exist (matrix builder, metric modules, and `predictions.md` all
> reference them):
>
> - §1 — Motivation and prior work
> - §2 — Perturbation taxonomy (P1–P9, with strength parameterizations)
> - §3 — Metrics (M1 per-frame structural drift, M2 temporal consistency
>   degradation, M3 DAVIS patch correspondence)
> - §4 — Models under test (V-JEPA 2.1 ViT-B / L / g / G)
> - §5 — Datasets (SSv2 evaluation split, DAVIS 2017 validation split)
> - §6 — Threats to validity
>
> Drop the source document into this path before the first sweep is launched.
> The tests in `tests/test_metrics.py` enforce the metric contracts described
> in §3, so the math there is load-bearing for the codebase.
