"""Sanity tests for the experiment matrix.

The 222-cell SSv2 total is part of the pre-registration; if it changes,
``manifests/predictions.md`` and EXPERIMENT.md \u00a72 must change with it.
The 322-cell SSv2 + DAVIS total mirrors H5's spec in
``manifests/predictions.md``.
"""

from __future__ import annotations

import pytest

from vjepa_stress.sweep.matrix import (
    DAVIS_STRENGTHS,
    DATASETS,
    Cell,
    build_experiment_matrix,
)


def test_matrix_has_222_cells():
    cells = build_experiment_matrix()
    assert len(cells) == 222


def test_matrix_tier_breakdown():
    cells = build_experiment_matrix()
    counts = {1: 0, 2: 0, 3: 0}
    for c in cells:
        counts[c.tier] += 1
    assert counts == {1: 200, 2: 12, 3: 10}


def test_matrix_cells_are_unique():
    cells = build_experiment_matrix()
    keys = {
        (c.model, c.perturbation, round(c.strength, 6), c.tier, c.dataset)
        for c in cells
    }
    assert len(keys) == len(cells)


def test_default_matrix_is_ssv2_only():
    cells = build_experiment_matrix()
    assert {c.dataset for c in cells} == {"ssv2"}


def test_matrix_with_davis_has_322_cells():
    cells = build_experiment_matrix(include_davis_m3=True)
    assert len(cells) == 322
    ssv2 = [c for c in cells if c.dataset == "ssv2"]
    davis = [c for c in cells if c.dataset == "davis"]
    assert len(ssv2) == 222
    assert len(davis) == 100  # 4 models x 5 Tier-1 perturbations x 5 strengths


def test_davis_subset_uses_only_tier1_and_davis_strengths():
    cells = build_experiment_matrix(include_davis_m3=True)
    davis = [c for c in cells if c.dataset == "davis"]
    assert all(c.tier == 1 for c in davis)
    strengths = {round(c.strength, 4) for c in davis}
    assert strengths == set(round(s, 4) for s in DAVIS_STRENGTHS)


def test_davis_cells_are_unique_per_axis():
    cells = build_experiment_matrix(include_davis_m3=True)
    keys = {
        (c.model, c.perturbation, round(c.strength, 6), c.tier, c.dataset)
        for c in cells
    }
    assert len(keys) == len(cells)


def test_cell_id_format_back_compat_for_ssv2():
    """SSv2 cell_id must match the pre-DAVIS format byte-for-byte; the
    canonical sweep-b077627 shards depend on it."""
    c = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="gaussian_noise",
        strength=0.5,
        tier=1,
    )
    assert c.dataset == "ssv2"  # default
    assert c.cell_id == "t1__vjepa-2.1-vit-base__gaussian_noise__s0.5000"


def test_cell_id_includes_davis_token_for_davis():
    """DAVIS cells get an explicit ``davis`` token so they never collide
    with SSv2 cells on disk even if (model, perturbation, strength, tier)
    overlap."""
    c = Cell(
        model="vjepa-2.1-vit-base",
        perturbation="gaussian_noise",
        strength=0.5,
        tier=1,
        dataset="davis",
    )
    assert c.cell_id == "t1__davis__vjepa-2.1-vit-base__gaussian_noise__s0.5000"


def test_cell_rejects_unknown_dataset():
    with pytest.raises(ValueError):
        Cell(
            model="vjepa-2.1-vit-base",
            perturbation="gaussian_noise",
            strength=0.5,
            tier=1,
            dataset="kinetics",
        )


def test_dataset_constants_are_locked():
    assert DATASETS == ("ssv2", "davis")
