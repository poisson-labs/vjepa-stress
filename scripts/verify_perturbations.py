"""Stage-0 verification: visualize every perturbation on a test clip.

Loads a small synthetic clip, applies each perturbation at a few strengths,
and writes a contact-sheet PNG to ``checkpoints/verify_perturbations.png``.
The intent is to eyeball the perturbations once Stage 0 implements them; the
shape/dtype contract is already enforced by ``tests/test_perturbations.py``.

Usage:
    python scripts/verify_perturbations.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vjepa_stress.perturbations import all_perturbations


def make_synthetic_clip(num_frames: int = 16, size: int = 128) -> np.ndarray:
    """A deterministic synthetic clip for sanity checks."""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    frames = np.empty((num_frames, size, size, 3), dtype=np.uint8)
    for t in range(num_frames):
        frames[t] = np.roll(base, shift=t * 4, axis=1)
    return frames


def main() -> int:
    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    clip = make_synthetic_clip()
    print(f"Synthetic clip: shape={clip.shape}, dtype={clip.dtype}")

    strengths = [0.0, 0.5, 1.0]
    for p in all_perturbations():
        line = [f"{p.name:>18s} (tier {p.tier}):"]
        for s in strengths:
            try:
                out = p.apply(clip, strength=s, seed=0)
                line.append(f"s={s:.1f}=ok({out.shape},{out.dtype})")
            except NotImplementedError:
                line.append(f"s={s:.1f}=NotImplemented")
            except ValueError as err:
                line.append(f"s={s:.1f}=ValueError({err.args[0][:40]}...)")
        print("  ".join(line))

    # TODO Stage 0: render a contact sheet to ``out_dir`` once the
    # perturbations are implemented end-to-end.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
