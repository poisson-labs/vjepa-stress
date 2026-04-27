"""M1 — Per-Frame Structural Drift.

Per RESEARCH.md §3:

    M1(v, v') = (1 / TP) * sum_{t,p} (1 - cos(F(v)_{t,p}, F(v')_{t,p}))

where ``F(v)_{t,p}`` is the L2-normalized token at frame ``t`` and spatial
position ``p`` from the frozen encoder. M1 is the average cosine distance
between paired tokens, taken over frames and spatial positions. It is bounded
in ``[0, 2]`` and equals zero iff every pair of tokens is identical (modulo
sign), which is the calibration anchor.
"""

from __future__ import annotations

import torch


def compute_m1(
    features_clean: torch.Tensor,
    features_perturbed: torch.Tensor,
) -> torch.Tensor:
    """Compute the per-frame structural drift between two clip encodings.

    Args:
        features_clean: Tensor of shape ``(T, P, D)``, L2-normalized along ``D``.
        features_perturbed: Tensor of the same shape and normalization.

    Returns:
        Scalar tensor in ``[0, 2]`` giving the M1 drift value.
    """
    raise NotImplementedError("M1: implement in Stage 0.")
