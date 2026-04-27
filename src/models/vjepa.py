"""Loader for the four V-JEPA 2.1 variants.

The encoder is frozen for every cell of the sweep — we only need a forward
pass that returns per-token features. The HF repo IDs are placeholders here;
fill them in once Meta publishes the V-JEPA 2.1 weights and pin the
``revision`` (commit hash) in ``manifests/manifest.json``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


# Map the variant name used in ``Cell.model`` to its HF repo id. The exact
# repo paths will land here once V-JEPA 2.1 is released; do not guess them.
VARIANT_REPOS: dict[str, str] = {
    "vjepa-2.1-vit-base": "TODO/vjepa-2.1-vit-base",
    "vjepa-2.1-vit-large": "TODO/vjepa-2.1-vit-large",
    "vjepa-2.1-vit-giant": "TODO/vjepa-2.1-vit-giant",
    "vjepa-2.1-vit-gigantic": "TODO/vjepa-2.1-vit-gigantic",
}


@dataclass
class VJEPAFeatures:
    """Per-token features for one clip.

    Attributes:
        features: shape ``(T, P, D)`` where ``T`` is the number of frames the
            encoder consumed, ``P`` is the number of spatial tokens per frame,
            and ``D`` is the token embedding dimension. L2-normalized along
            ``D`` so cosine similarity is a dot product.
        variant: the variant string this came from; useful for downstream
            tagging in the Parquet output.
    """

    features: torch.Tensor
    variant: str


class VJEPAEncoder:
    """Stage-0 placeholder for the V-JEPA 2.1 frozen-encoder wrapper.

    The real implementation needs to:

    1. Resolve ``variant`` against ``VARIANT_REPOS`` and load via
       ``transformers.AutoModel.from_pretrained(repo, revision=revision)``.
    2. Move to the requested device, set ``.eval()`` and ``requires_grad=False``.
    3. Apply the variant-specific normalization to ``frames``.
    4. Run the encoder and return ``VJEPAFeatures`` with L2-normalized tokens.
    """

    def __init__(
        self,
        variant: str,
        revision: str | None = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        if variant not in VARIANT_REPOS:
            raise ValueError(
                f"Unknown V-JEPA 2.1 variant {variant!r}. "
                f"Expected one of {sorted(VARIANT_REPOS)}."
            )
        self.variant = variant
        self.revision = revision
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> VJEPAFeatures:  # pragma: no cover - placeholder
        """Encode frames of shape ``(T, H, W, 3)``, uint8 RGB, to per-token features."""
        raise NotImplementedError
