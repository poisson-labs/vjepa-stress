"""Sanity tests for the experiment matrix.

The 222-cell total is part of the pre-registration; if it changes,
``manifests/predictions.md`` and EXPERIMENT.md \u00a72 must change with it.
"""

from __future__ import annotations

from vjepa_stress.sweep.matrix import build_experiment_matrix


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
    keys = {(c.model, c.perturbation, round(c.strength, 6), c.tier) for c in cells}
    assert len(keys) == len(cells)
