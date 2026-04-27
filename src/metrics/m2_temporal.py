"""M2 — Temporal Consistency Degradation.

Per RESEARCH.md §3, M2 measures whether the per-token *temporal gradient* of
the perturbed clip agrees with that of the clean clip:

    delta(v)_{t,p}  = F(v)_{t+1,p} - F(v)_{t,p}             # (T-1, P, D)
    M2(v, v') = (1 / (T-1)P) * sum_{t,p}
                  (1 - cos(delta(v)_{t,p}, delta(v')_{t,p}))

Why a gradient-based form rather than a pure smoothness scalar?  A scalar
``S(v) = mean cos(F_t, F_{t+1})`` is invariant under temporal reversal
because the multiset of consecutive token pairs is identical.  M2 needs to
detect Reverse as a probe — see RESEARCH.md §2 — so we instead compare the
*direction of change* across consecutive frames.  Identity gives M2 = 0 by
construction (the gradient sequences match exactly); Reverse flips the sign
of every gradient, so cosine collapses and M2 saturates.

M2 is non-negative, lives in ``[0, 2]`` per token, and equals zero iff the
two trajectories advance in the same direction at every (frame, position).
"""

from __future__ import annotations

import torch

_EPS: float = 1e-12


def compute_m2(
    features_clean: torch.Tensor,
    features_perturbed: torch.Tensor,
) -> torch.Tensor:
    """Compute the temporal consistency degradation between two clip encodings.

    Args:
        features_clean: Tensor of shape ``(T, P, D)``, L2-normalized along
            ``D``, with ``T >= 2``.
        features_perturbed: Tensor of the same shape and normalization.

    Returns:
        Scalar tensor in ``[0, 2]`` giving the average cosine distance
        between paired temporal gradients.
    """
    if features_clean.shape != features_perturbed.shape:
        raise ValueError(
            f"shape mismatch: {features_clean.shape} vs {features_perturbed.shape}"
        )
    if features_clean.shape[0] < 2:
        raise ValueError(f"need T >= 2 frames; got T={features_clean.shape[0]}")

    delta_clean = features_clean[1:] - features_clean[:-1]
    delta_perturb = features_perturbed[1:] - features_perturbed[:-1]
    norm_clean = torch.linalg.vector_norm(delta_clean, dim=-1).clamp_min(_EPS)
    norm_perturb = torch.linalg.vector_norm(delta_perturb, dim=-1).clamp_min(_EPS)
    cos = (delta_clean * delta_perturb).sum(dim=-1) / (norm_clean * norm_perturb)
    return (1.0 - cos).mean()
