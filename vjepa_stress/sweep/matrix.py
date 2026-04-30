"""The experiment matrix (222 SSv2 cells; +100 DAVIS cells with M3 enabled).

Tier breakdown (matches EXPERIMENT.md §2):

- **Tier 1** — robustness sweep: 4 models x 5 perturbations x 10 strengths
  = 200 cells. Perturbations: gaussian_noise, motion_blur, frame_drops,
  occlusion, low_light.
- **Tier 2** — calibration probes: 4 models x 3 probes = 12 cells. Probes:
  identity, reverse, horizontal_flip. Each runs at a single strength
  (``1.0``); see RESEARCH.md §2 for why intermediate strengths are not
  defined for these.
- **Tier 3** — codec sanity sweep: 1 model (ViT-L) x 1 perturbation
  (jpeg) x 10 strengths = 10 cells.

Total SSv2: 200 + 12 + 10 = **222** cells.

When ``include_davis_m3=True``, an additional 100 DAVIS cells are
appended for the H5 / M3 follow-up sweep:

- 4 models x 5 Tier-1 perturbations x 5 strengths
  (``{0.1, 0.3, 0.5, 0.7, 0.9}``, a subset of the SSv2 grid). Per
  ``manifests/predictions.md`` H5, that's enough points to estimate the
  Pearson r between ΔM3 and M2 reliably across (model, perturbation,
  strength, clip) tuples.

The runner consumes a flat list of ``Cell`` objects in canonical order
so that resumption from a partial sweep is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Variant identifiers; must match keys of ``vjepa_stress.models.vjepa.VARIANT_REPOS``.
MODELS: tuple[str, ...] = (
    "vjepa-2.1-vit-base",
    "vjepa-2.1-vit-large",
    "vjepa-2.1-vit-giant",
    "vjepa-2.1-vit-gigantic",
)

TIER1_PERTURBATIONS: tuple[str, ...] = (
    "gaussian_noise",
    "motion_blur",
    "frame_drops",
    "occlusion",
    "low_light",
)

TIER2_PROBES: tuple[str, ...] = (
    "identity",
    "reverse",
    "horizontal_flip",
)

TIER3_MODEL: str = "vjepa-2.1-vit-large"
TIER3_PERTURBATIONS: tuple[str, ...] = ("jpeg",)

#: Datasets we sweep over. Kept here (rather than as a free string) so
#: the matrix builder, runner dispatch, and Cell validator agree on the
#: closed set. New datasets only require touching this tuple plus a
#: factory in ``run_sweep.py``.
DATASETS: tuple[str, ...] = ("ssv2", "davis")

#: 10 strengths spanning the open interval after 0; strength=0 is covered by
#: the Identity probe, so we don't duplicate it here.
STRENGTHS: tuple[float, ...] = tuple(round(float(s), 4) for s in np.linspace(0.1, 1.0, 10))

#: 5 strengths used for the DAVIS / M3 sweep (subset of STRENGTHS). H5 does
#: not need every strength to estimate a Pearson r; halving the grid keeps
#: total compute under one hour while still giving 100 row-cell tuples per
#: (model, perturbation) for the correlation.
DAVIS_STRENGTHS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)

#: Calibration probes are evaluated at a single strength.
PROBE_STRENGTH: float = 1.0


@dataclass(frozen=True)
class Cell:
    """One sweep cell: model x perturbation x strength x dataset.

    ``dataset`` defaults to ``"ssv2"`` so existing call sites that don't
    care about the axis (the entire pre-DAVIS history) continue to work
    untouched. The Tier 2 calibration probes and Tier 3 codec sweep are
    SSv2-only by construction; the DAVIS axis only fires for Tier 1
    perturbations under ``include_davis_m3=True``.
    """

    model: str
    perturbation: str
    strength: float
    tier: int
    dataset: str = "ssv2"

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(
                f"dataset must be one of {DATASETS}; got {self.dataset!r}"
            )

    @property
    def cell_id(self) -> str:
        """Stable identifier used for the Parquet shard filename.

        SSv2 cell IDs preserve the pre-DAVIS format
        (``t{tier}__{model}__{perturbation}__s{strength:.4f}``) so
        existing shard collections (e.g. ``runs/sweep-b077627/``) and
        prior tests continue to resolve without churn. DAVIS cells get
        an explicit ``davis`` token so the two never collide on disk
        even when ``(model, perturbation, strength, tier)`` overlaps.
        """
        if self.dataset == "ssv2":
            return f"t{self.tier}__{self.model}__{self.perturbation}__s{self.strength:.4f}"
        return (
            f"t{self.tier}__{self.dataset}__{self.model}__{self.perturbation}"
            f"__s{self.strength:.4f}"
        )


def build_experiment_matrix(*, include_davis_m3: bool = False) -> list[Cell]:
    """Return the experiment matrix in canonical order.

    Args:
        include_davis_m3: when ``True``, append 100 DAVIS cells (4 models
            x 5 Tier-1 perturbations x 5 DAVIS_STRENGTHS) for the H5 /
            M3 sweep. Default ``False`` preserves the historical 222-cell
            SSv2-only matrix.

    Returns:
        222 cells with ``include_davis_m3=False``, or 322 cells with
        ``include_davis_m3=True``. Counts:

        - Tier 1 SSv2: 200 (4x5x10)
        - Tier 2 SSv2: 12 (4x3x1)
        - Tier 3 SSv2: 10 (1x1x10)
        - DAVIS / M3: 100 (4x5x5), only if requested
    """
    cells: list[Cell] = []

    # Tier 1 — 4 x 5 x 10 = 200, all SSv2
    for model in MODELS:
        for perturbation in TIER1_PERTURBATIONS:
            for strength in STRENGTHS:
                cells.append(
                    Cell(model, perturbation, strength, tier=1, dataset="ssv2")
                )

    # Tier 2 — 4 x 3 x 1 = 12, all SSv2 (calibration probes)
    for model in MODELS:
        for probe in TIER2_PROBES:
            cells.append(
                Cell(model, probe, PROBE_STRENGTH, tier=2, dataset="ssv2")
            )

    # Tier 3 — 1 x 1 x 10 = 10, all SSv2 (codec sanity sweep)
    for perturbation in TIER3_PERTURBATIONS:
        for strength in STRENGTHS:
            cells.append(
                Cell(TIER3_MODEL, perturbation, strength, tier=3, dataset="ssv2")
            )

    # DAVIS / M3 — 4 x 5 x 5 = 100 (Tier 1 perturbations only), gated on flag.
    if include_davis_m3:
        for model in MODELS:
            for perturbation in TIER1_PERTURBATIONS:
                for strength in DAVIS_STRENGTHS:
                    cells.append(
                        Cell(model, perturbation, strength, tier=1, dataset="davis")
                    )

    return cells


def _matrix_as_markdown(cells: list[Cell]) -> str:
    """Render the matrix as a Markdown table for sanity-checking on stdout."""
    lines = [
        "| # | tier | model | perturbation | strength |",
        "|---|------|-------|--------------|----------|",
    ]
    for i, c in enumerate(cells, start=1):
        lines.append(f"| {i} | {c.tier} | {c.model} | {c.perturbation} | {c.strength:.4f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    cells = build_experiment_matrix()
    print(_matrix_as_markdown(cells))
    print()
    print(f"Total cells: {len(cells)}")
    by_tier = {1: 0, 2: 0, 3: 0}
    for c in cells:
        by_tier[c.tier] += 1
    for tier, count in by_tier.items():
        print(f"  Tier {tier}: {count}")
