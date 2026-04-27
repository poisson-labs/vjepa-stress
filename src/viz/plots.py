"""Headline plot helpers.

Each function takes a tidy ``pandas.DataFrame`` (one row per sweep cell) and
returns a ``matplotlib.figure.Figure``. The runner does not call these — they
are invoked from ``notebooks/analysis.ipynb`` during Stage 5.
"""

from __future__ import annotations

import pandas as pd
from matplotlib.figure import Figure


def plot_drift_vs_strength(df: pd.DataFrame) -> Figure:  # pragma: no cover - placeholder
    """M1 vs strength, one line per ``(model, perturbation)`` pair."""
    raise NotImplementedError


def plot_temporal_consistency(df: pd.DataFrame) -> Figure:  # pragma: no cover - placeholder
    """M2 across perturbations, faceted by model variant."""
    raise NotImplementedError


def plot_correspondence_decay(df: pd.DataFrame) -> Figure:  # pragma: no cover - placeholder
    """M3 (DAVIS patch correspondence) vs strength."""
    raise NotImplementedError
