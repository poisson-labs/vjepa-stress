"""M3 — DAVIS Patch Correspondence.

Per RESEARCH.md §3:

    For each instance mask in the first frame, sample K keypoints inside the
    mask. Lift each keypoint into V-JEPA token space (the token whose
    receptive-field center contains the keypoint). For each subsequent frame
    t > 0, find the nearest neighbor token under cosine similarity inside an
    L x L spatial search window centered on the previous matched location.
    M3 is the fraction of matched tokens whose receptive-field center falls
    inside the corresponding ground-truth mask in frame t.

M3 is the only metric that requires labels (DAVIS instance masks). It is the
downstream-validity check: M1 and M2 are intrinsic stability metrics, but M3
ties them to a task we know V-JEPA encoders can do well on the clean clip,
so degradation under perturbation is unambiguously a representation failure.
"""

from __future__ import annotations

import torch


def compute_m3(
    features: torch.Tensor,
    masks: torch.Tensor,
    *,
    keypoints_per_mask: int = 8,
    search_window: int = 7,
) -> torch.Tensor:
    """Compute DAVIS patch correspondence accuracy.

    Args:
        features: Tensor of shape ``(T, H_t, W_t, D)``, L2-normalized along ``D``,
            where ``H_t`` x ``W_t`` is the spatial token grid.
        masks: Tensor of shape ``(T, H, W)`` with integer instance ids; ``0``
            is background. ``H`` x ``W`` is the original image resolution.
        keypoints_per_mask: How many keypoints to sample per instance from
            the first frame.
        search_window: Side length ``L`` of the spatial search window used
            for the nearest-neighbor lookup in subsequent frames.

    Returns:
        Scalar tensor in ``[0, 1]`` giving the average correspondence
        accuracy across all (instance, keypoint, frame) tuples.
    """
    raise NotImplementedError("M3: implement in Stage 0.")
