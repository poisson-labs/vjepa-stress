"""Codec-domain perturbations: P9 JPEG compression.

JPEG is a tier-3 sanity sweep — we only run it on ViT-L. The point is to
confirm that codec artifacts behave like the other low-frequency stressors
and don't surface a separate failure mode.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.perturbations.base import Perturbation


# JPEG quality at strength=0 vs strength=1. ``95`` is visually lossless;
# ``5`` is severely blocky. The runner clamps to libjpeg's [1, 100] range.
_JPEG_QUALITY_HIGH: int = 95
_JPEG_QUALITY_LOW: int = 5


class JPEGCompression(Perturbation):
    """P9 — Per-frame JPEG re-encode at a strength-controlled quality factor.

    Quality interpolates linearly from ``95`` at ``strength=0+`` to ``5`` at
    ``strength=1``. Encoding is deterministic, so ``seed`` is unused but
    accepted for API uniformity.
    """

    name = "jpeg"
    tier = 3

    def apply(self, frames: np.ndarray, strength: float, seed: int) -> np.ndarray:
        self._validate(frames, strength)
        if strength == 0.0:
            return frames.copy()
        del seed
        s = float(strength)
        quality = int(round(_JPEG_QUALITY_HIGH + s * (_JPEG_QUALITY_LOW - _JPEG_QUALITY_HIGH)))
        quality = max(1, min(100, quality))
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        out = np.empty_like(frames)
        for t in range(frames.shape[0]):
            bgr = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, encode_params)
            if not ok:
                raise RuntimeError(f"JPEG encode failed at frame {t}, quality {quality}")
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            out[t] = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return out
