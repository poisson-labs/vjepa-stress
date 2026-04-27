"""Temporal perturbations: P3 Frame drops, P7 Reverse.

Frame drops perturb the time axis by removing frames and re-padding with the
previous frame so the clip length is preserved. Reverse inverts the time
axis as a calibration probe; an encoder that ignored direction would show
zero M2 jump on Reverse, which would itself be a result.
"""

from __future__ import annotations

import numpy as np

from vjepa_stress.perturbations.base import Perturbation


# At strength=1 we drop half the frames in expectation; capped so the clip
# always retains some signal regardless of seed.
_FRAME_DROPS_MAX_PROB: float = 0.5


class FrameDrops(Perturbation):
    """P3 — Drop a fraction of frames and re-pad with the previous frame.

    Drop probability is ``strength * 0.5``. Frame 0 is always retained so the
    forward-fill has something to start from. Drops are sampled per video
    from ``seed`` and held independent of the frame contents.
    """

    name = "frame_drops"
    tier = 1

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        rng = np.random.default_rng(seed)
        drop_prob = float(strength) * _FRAME_DROPS_MAX_PROB
        t = frames.shape[0]
        drops = rng.random(t) < drop_prob
        drops[0] = False
        out = frames.copy()
        for i in range(1, t):
            if drops[i]:
                out[i] = out[i - 1]
        return out


class Reverse(Perturbation):
    """P7 — Reverse the temporal axis.

    Calibration probe: only ``strength=1`` is meaningful (full reversal).
    ``strength=0`` is identity. Intermediate strengths are not defined for
    this perturbation, so the runner only schedules it at the endpoints.
    """

    name = "reverse"
    tier = 2

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        if strength == 1.0:
            return frames[::-1].copy()
        raise ValueError(
            "Reverse only accepts strength in {0.0, 1.0}; got "
            f"{strength}. Use Identity (P6) at strength=0 if you want a "
            "no-op anchor."
        )
