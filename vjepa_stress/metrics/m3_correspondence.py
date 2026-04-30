"""M3 — DAVIS Patch Correspondence.

Per RESEARCH.md §3:

    For each instance mask in the first frame, sample K keypoints inside the
    mask. Lift each keypoint into V-JEPA token space (the token whose
    receptive-field center contains the keypoint). For each subsequent frame
    t > 0, find the nearest neighbor token under cosine similarity inside an
    L x L spatial search window centered on the previous matched location.
    M3 is the fraction of matched tokens whose receptive-field center falls
    inside the corresponding ground-truth mask in frame t.

M3 is the only metric that requires labels. It ties M1/M2 — which are
intrinsic stability scores — to a downstream task we already know V-JEPA
encoders do well on the clean clip, so a degradation under perturbation is
unambiguously a representation failure rather than a metric artifact.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def compute_m3(
    features: torch.Tensor,
    masks: torch.Tensor,
    *,
    keypoints_per_mask: int = 8,
    search_window: int = 7,
    seed: int = 0,
) -> torch.Tensor:
    """Compute DAVIS patch correspondence accuracy.

    Args:
        features: Tensor of shape ``(T, H_t, W_t, D)``, L2-normalized along
            ``D``, where ``H_t`` x ``W_t`` is the spatial token grid.
        masks: Tensor of shape ``(T, H, W)`` with integer instance ids; ``0``
            is background. ``H`` x ``W`` is the original image resolution.
        keypoints_per_mask: How many keypoints to sample per instance from
            the first frame.
        search_window: Side length ``L`` of the spatial search window used
            for the nearest-neighbor lookup in subsequent frames.
        seed: Per-video seed for keypoint sampling. Held independent of the
            sweep's perturbation seed so the same keypoints are used for
            clean and perturbed evaluations of the same video.

    Returns:
        Scalar tensor in ``[0, 1]`` giving the average correspondence
        accuracy across all (instance, keypoint, frame) tuples. Returns NaN
        if the first-frame mask has no instances.
    """
    if features.ndim != 4:
        raise ValueError(f"features must be (T, H_t, W_t, D); got {features.shape}")
    if masks.ndim != 3:
        raise ValueError(f"masks must be (T, H, W); got {masks.shape}")
    if features.shape[0] != masks.shape[0]:
        raise ValueError(
            f"feature/mask T mismatch: {features.shape[0]} vs {masks.shape[0]}"
        )

    t_total, h_t, w_t, _ = features.shape
    h_img, w_img = masks.shape[-2:]
    if t_total < 2:
        return torch.tensor(float("nan"))

    scale_h = h_img / h_t
    scale_w = w_img / w_t
    half = search_window // 2

    instances = torch.unique(masks[0])
    instances = instances[instances != 0]
    if instances.numel() == 0:
        return torch.tensor(float("nan"))

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    correct = 0
    total = 0

    for inst in instances.tolist():
        coords = torch.nonzero(masks[0] == inst, as_tuple=False)  # (N, 2): (y, x)
        if coords.shape[0] == 0:
            continue
        k = min(keypoints_per_mask, coords.shape[0])
        idx = torch.randperm(coords.shape[0], generator=gen)[:k]
        for kp in coords[idx]:
            y_img = int(kp[0].item())
            x_img = int(kp[1].item())
            y_tok = max(0, min(h_t - 1, int(math.floor(y_img / scale_h))))
            x_tok = max(0, min(w_t - 1, int(math.floor(x_img / scale_w))))

            for t in range(1, t_total):
                anchor = features[t - 1, y_tok, x_tok]  # (D,)
                y_lo, y_hi = max(0, y_tok - half), min(h_t, y_tok + half + 1)
                x_lo, x_hi = max(0, x_tok - half), min(w_t, x_tok + half + 1)
                window = features[t, y_lo:y_hi, x_lo:x_hi]  # (h, w, D)
                cos = window @ anchor  # (h, w)
                flat = torch.argmax(cos.reshape(-1)).item()
                wh, ww = window.shape[:2]
                y_tok = y_lo + flat // ww
                x_tok = x_lo + flat % ww

                y_match = min(h_img - 1, max(0, int((y_tok + 0.5) * scale_h)))
                x_match = min(w_img - 1, max(0, int((x_tok + 0.5) * scale_w)))
                if int(masks[t, y_match, x_match].item()) == inst:
                    correct += 1
                total += 1

    if total == 0:
        return torch.tensor(float("nan"))
    return torch.tensor(correct / total)


def _tubelet_pool_masks(masks: np.ndarray, tubelet_size: int) -> torch.Tensor:
    """Collapse a per-frame mask stack into a per-tubelet mask stack.

    The encoder emits one token per tubelet; ``compute_m3`` needs one
    mask per output time step. Per-pixel max (i.e. per-pixel union) is
    the right pool: a pixel that belongs to instance ``k`` for any frame
    in the tubelet is part of instance ``k`` for that tubelet. Background
    (``0``) loses all ties, which matches DAVIS J&F evaluation
    semantics where any-frame foreground counts as foreground.

    Args:
        masks: ``(T, H, W)`` int array; ``T`` must be a multiple of
            ``tubelet_size``.
        tubelet_size: encoder's tubelet size; ``1`` is the no-op case.

    Returns:
        ``(T // tubelet_size, H, W)`` int32 tensor on CPU.
    """
    if masks.ndim != 3:
        raise ValueError(f"masks must be (T, H, W); got shape {masks.shape}")
    t, h, w = masks.shape
    if tubelet_size <= 0:
        raise ValueError(f"tubelet_size must be positive; got {tubelet_size}")
    if t % tubelet_size != 0:
        raise ValueError(
            f"masks T={t} must be a multiple of tubelet_size={tubelet_size}"
        )
    if tubelet_size == 1:
        return torch.from_numpy(np.ascontiguousarray(masks.astype(np.int32, copy=False)))
    grouped = masks.reshape(t // tubelet_size, tubelet_size, h, w)
    pooled = grouped.max(axis=1).astype(np.int32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(pooled))


def compute_m3_for_clip(
    *,
    features: torch.Tensor,
    masks: np.ndarray,
    tubelet_size: int,
    keypoints_per_mask: int = 8,
    search_window: int = 7,
    seed: int = 0,
) -> torch.Tensor:
    """Adapter: reshape encoder output to a token grid and dispatch to ``compute_m3``.

    The frozen V-JEPA 2.1 encoder returns ``(T_tubelet, P, D)``. M3
    operates on a ``(T_tubelet, H_t, W_t, D)`` token grid plus a
    per-tubelet ``(T_tubelet, H, W)`` mask, so this adapter:

    - infers ``H_t = W_t = sqrt(P)`` (V-JEPA uses square patch grids),
    - L2-normalizes per token (idempotent if already normalized; required
      by ``compute_m3``'s cosine-similarity argmax),
    - tubelet-pools the per-frame ground-truth masks,
    - delegates to :func:`compute_m3`.

    Returns the same scalar in ``[0, 1]`` as ``compute_m3`` (or NaN if
    the first tubelet's mask has no foreground instances).
    """
    if features.ndim != 3:
        raise ValueError(
            f"features must be (T_tubelet, P, D); got shape {tuple(features.shape)}"
        )
    t_tubelet, p, d = features.shape
    side = int(round(math.sqrt(p)))
    if side * side != p:
        raise ValueError(
            f"feature patch count P={p} is not a perfect square; M3 requires "
            f"a square token grid (got side={side})"
        )

    pooled_masks = _tubelet_pool_masks(masks, tubelet_size)
    if pooled_masks.shape[0] != t_tubelet:
        raise ValueError(
            f"after pooling, masks T={pooled_masks.shape[0]} != features "
            f"T_tubelet={t_tubelet}"
        )

    grid = features.reshape(t_tubelet, side, side, d)
    grid = torch.nn.functional.normalize(grid, dim=-1)

    return compute_m3(
        features=grid,
        masks=pooled_masks,
        keypoints_per_mask=keypoints_per_mask,
        search_window=search_window,
        seed=seed,
    )
