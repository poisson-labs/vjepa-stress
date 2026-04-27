"""M2 — Temporal Consistency Degradation.

Per RESEARCH.md §3:

    S(v)  = (1 / (T-1)P) * sum_{t,p} cos(F(v)_{t,p}, F(v)_{t+1,p})
    M2(v, v') = S(v) - S(v')

``S`` is the clip-internal temporal smoothness: average cosine similarity
between tokens at the same spatial position across consecutive frames. M2 is
the drop in smoothness induced by perturbation. Positive M2 means the
perturbed clip is *less* temporally consistent than the clean clip.

A reversed clip should produce a measurable, positive M2 jump on every
encoder, which is what makes Reverse a useful calibration probe in §4.
"""

from __future__ import annotations

import torch


def compute_m2(
    features_clean: torch.Tensor,
    features_perturbed: torch.Tensor,
) -> torch.Tensor:
    """Compute the temporal consistency degradation between two clip encodings.

    Args:
        features_clean: Tensor of shape ``(T, P, D)``, L2-normalized along ``D``,
            with ``T >= 2``.
        features_perturbed: Tensor of the same shape and normalization.

    Returns:
        Scalar tensor giving ``S(v) - S(v')``. Sign convention: positive
        values mean the perturbed clip lost temporal smoothness.
    """
    raise NotImplementedError("M2: implement in Stage 0.")
