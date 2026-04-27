"""The 222-cell experiment matrix.

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

Total: 200 + 12 + 10 = **222** cells.

The runner consumes a flat list of ``Cell`` objects in canonical order so
that resumption from a partial sweep is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Variant identifiers; must match keys of ``src.models.vjepa.VARIANT_REPOS``.
MODELS: tuple[str, ...] = (
    "vjepa-2.1-vit-b",
    "vjepa-2.1-vit-l",
    "vjepa-2.1-vit-g",
    "vjepa-2.1-vit-giant",
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

TIER3_MODEL: str = "vjepa-2.1-vit-l"
TIER3_PERTURBATIONS: tuple[str, ...] = ("jpeg",)

#: 10 strengths spanning the open interval after 0; strength=0 is covered by
#: the Identity probe, so we don't duplicate it here.
STRENGTHS: tuple[float, ...] = tuple(round(float(s), 4) for s in np.linspace(0.1, 1.0, 10))

#: Calibration probes are evaluated at a single strength.
PROBE_STRENGTH: float = 1.0


@dataclass(frozen=True)
class Cell:
    """One sweep cell: model x perturbation x strength."""

    model: str
    perturbation: str
    strength: float
    tier: int

    @property
    def cell_id(self) -> str:
        """Stable identifier used for the Parquet shard filename."""
        return f"t{self.tier}__{self.model}__{self.perturbation}__s{self.strength:.4f}"


def build_experiment_matrix() -> list[Cell]:
    """Return the full 222-cell experiment matrix in canonical order."""
    cells: list[Cell] = []

    # Tier 1 — 4 x 5 x 10 = 200
    for model in MODELS:
        for perturbation in TIER1_PERTURBATIONS:
            for strength in STRENGTHS:
                cells.append(Cell(model, perturbation, strength, tier=1))

    # Tier 2 — 4 x 3 x 1 = 12
    for model in MODELS:
        for probe in TIER2_PROBES:
            cells.append(Cell(model, probe, PROBE_STRENGTH, tier=2))

    # Tier 3 — 1 x 1 x 10 = 10
    for perturbation in TIER3_PERTURBATIONS:
        for strength in STRENGTHS:
            cells.append(Cell(TIER3_MODEL, perturbation, strength, tier=3))

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
