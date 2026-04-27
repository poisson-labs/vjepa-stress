"""Stage-0 verification: sanity-check M1 and M2 on synthetic features.

Once Stage 0 lands the metric implementations, this script confirms the
calibration anchors hold:

- M1 on identical features is zero.
- M2 on identical features is zero.
- M2 on a reversed feature sequence is positive and non-zero.

Run before launching any sweep.

Usage:
    python scripts/verify_metrics.py
"""

from __future__ import annotations

import torch

from vjepa_stress.metrics import compute_m1, compute_m2


def make_synthetic_features(t: int = 16, p: int = 196, d: int = 768) -> torch.Tensor:
    """Random L2-normalized token features of shape ``(T, P, D)``."""
    g = torch.Generator().manual_seed(0)
    f = torch.randn(t, p, d, generator=g)
    return torch.nn.functional.normalize(f, dim=-1)


def main() -> int:
    f = make_synthetic_features()

    print("M1(identity, identity) — expect 0:")
    try:
        print(f"  -> {compute_m1(f, f).item():.6e}")
    except NotImplementedError:
        print("  -> NotImplemented (Stage 0 work)")

    print("M2(identity, identity) — expect 0:")
    try:
        print(f"  -> {compute_m2(f, f).item():.6e}")
    except NotImplementedError:
        print("  -> NotImplemented (Stage 0 work)")

    print("M2(clean, reverse(clean)) — expect > 0:")
    f_rev = torch.flip(f, dims=[0])
    try:
        print(f"  -> {compute_m2(f, f_rev).item():.6e}")
    except NotImplementedError:
        print("  -> NotImplemented (Stage 0 work)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
