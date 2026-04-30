"""V-JEPA 2.1 frozen-encoder loader.

Loads one of the four V-JEPA 2.1 variants via ``torch.hub``, freezes it,
and exposes a small protocol surface (:class:`Encoder`) that the runner
relies on to stay agnostic to whether it is talking to a real model or a
:class:`vjepa_stress.testing.fakes.FakeEncoder`.

Two non-obvious contracts:

1. **Tubelet collapse.** V-JEPA 2.1 uses ``tubelet_size=2``: every two
   consecutive input frames collapse into a single token row. A 16-frame
   clip therefore yields 8 tubelet rows, not 16. The runner uses
   :meth:`Encoder.output_temporal_dim` to translate input frame counts
   into output token-row counts so it never assumes a 1:1 mapping.

2. **Hub repo URL is broken upstream.** ``facebookresearch/vjepa2``
   ships V-JEPA 2.1's ``hubconf`` with a localhost dev-server URL
   (``http://localhost:8300``). We patch ``VJEPA_BASE_URL`` to the real
   ``dl.fbaipublicfiles.com`` bucket inside :func:`patch_meta_url`,
   which restores the original on exit so subsequent imports of the
   module see the cache repo unchanged.

Background and direct download URLs are documented in
``docs/loading_spike.md``.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from vjepa_stress.utils.hub_isolation import isolate_for_torch_hub

logger = logging.getLogger(__name__)


HUB_REPO: str = "facebookresearch/vjepa2"

PATCH_SIZE: int = 16
TUBELET_SIZE: int = 2
INPUT_RESOLUTION: int = 384

#: Canonical variant short name → torch.hub entrypoint name.
VARIANT_REPOS: dict[str, str] = {
    "vit-base": "vjepa2_1_vit_base_384",
    "vit-large": "vjepa2_1_vit_large_384",
    "vit-giant": "vjepa2_1_vit_giant_384",
    "vit-gigantic": "vjepa2_1_vit_gigantic_384",
}

#: Embedding dim for each canonical variant. Pinned by Meta's hubconf.
VARIANT_EMBED_DIMS: dict[str, int] = {
    "vit-base": 768,
    "vit-large": 1024,
    "vit-giant": 1408,
    "vit-gigantic": 1664,
}

#: Real production URL for the V-JEPA 2.1 weights bucket. Hub repo ships a
#: localhost dev-server URL; we patch this in at load time.
VJEPA_PRODUCTION_URL: str = "https://dl.fbaipublicfiles.com/vjepa2"

#: ImageNet mean/std — V-JEPA's standard preprocessing.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@runtime_checkable
class Encoder(Protocol):
    """Protocol the runner expects from any encoder (real or fake).

    The runner never calls ``forward`` directly; it goes through
    :meth:`encode` and uses the attribute/method surface below to size
    output tensors and validate shapes.
    """

    embedding_dim: int
    tubelet_size: int
    patch_size: int
    input_resolution: int
    forward_calls: int

    def output_temporal_dim(self, n_frames: int) -> int: ...

    def output_spatial_patches(self) -> int: ...

    def encode(self, frames: np.ndarray) -> torch.Tensor: ...


@dataclass
class VJEPAFeatures:
    """Per-token features for one clip.

    Attributes:
        features: shape ``(T_tubelet, P, D)``, L2-normalized along ``D``.
        variant: the canonical variant short name (e.g. ``"vit-base"``).
    """

    features: torch.Tensor
    variant: str


def _normalize_variant(name: str) -> str:
    """Accept either ``"vit-base"`` or ``"vjepa-2.1-vit-base"`` style names."""
    if name in VARIANT_REPOS:
        return name
    prefix = "vjepa-2.1-"
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix in VARIANT_REPOS:
            return suffix
    valid = sorted(VARIANT_REPOS) + [f"vjepa-2.1-{v}" for v in sorted(VARIANT_REPOS)]
    raise ValueError(
        f"Unknown V-JEPA 2.1 variant {name!r}. Expected one of {valid}."
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve ``"auto"`` to the best available device on this host."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def default_dtype_for_device(device: str | torch.device) -> torch.dtype:
    """Pick a sensible default dtype for the inference target.

    CUDA gets bfloat16 (numerically stable for ViT inference, half the
    memory). MPS bf16 support is incomplete so MPS stays in float32. CPU
    stays in float32 too.
    """
    dev = _resolve_device(device)
    if dev.type == "cuda":
        return torch.bfloat16
    return torch.float32


def _hub_cache_dir() -> Path:
    """Path to the cached ``facebookresearch/vjepa2`` clone."""
    hub_dir = Path(torch.hub.get_dir())
    # torch.hub names the dir ``<owner>_<repo>_<branch>``.
    return hub_dir / "facebookresearch_vjepa2_main"


def _ensure_repo_cloned() -> None:
    """Trigger torch.hub to clone the repo if it isn't cached yet.

    ``torch.hub.list`` does the clone (or no-op if already present) and
    does *not* download weights, so it is safe and fast as a one-shot.
    """
    if _hub_cache_dir().is_dir():
        return
    with isolate_for_torch_hub():
        torch.hub.list(HUB_REPO, trust_repo=True)


@contextmanager
def patch_meta_url(target_url: str = VJEPA_PRODUCTION_URL) -> Iterator[None]:
    """Patch the cached repo's ``VJEPA_BASE_URL`` for the duration of the block.

    ``facebookresearch/vjepa2`` hubconf imports ``src.hub.backbones`` and
    reads ``VJEPA_BASE_URL`` at construction time. The shipped value is
    ``http://localhost:8300`` (a dev-server stub Meta forgot to flip back
    when merging the 2.1 hub code). We import the module from the cache
    dir, save+set the constant, then restore it on exit so we leave the
    cache as we found it.
    """
    cache_dir = _hub_cache_dir()
    if not cache_dir.is_dir():
        raise RuntimeError(
            f"hub cache dir does not exist: {cache_dir}. "
            f"Call _ensure_repo_cloned() first."
        )

    import sys

    cache_dir_str = str(cache_dir)
    added_to_path = False
    if cache_dir_str not in sys.path:
        sys.path.insert(0, cache_dir_str)
        added_to_path = True

    # Evict any prior import so we re-resolve against the cache dir, not
    # whatever happened to win the import lottery earlier.
    evicted: dict[str, object] = {}
    for mod_name in list(sys.modules):
        if mod_name == "src" or mod_name.startswith("src."):
            evicted[mod_name] = sys.modules.pop(mod_name)

    backbones = importlib.import_module("src.hub.backbones")
    original = backbones.VJEPA_BASE_URL
    backbones.VJEPA_BASE_URL = target_url
    try:
        yield
    finally:
        backbones.VJEPA_BASE_URL = original
        if added_to_path and cache_dir_str in sys.path:
            sys.path.remove(cache_dir_str)
        # Restore prior ``src.*`` modules so we don't leak the cache-dir
        # version into surrounding code that may have its own ``src``.
        for mod_name in list(sys.modules):
            if mod_name == "src" or mod_name.startswith("src."):
                del sys.modules[mod_name]
        for mod_name, mod in evicted.items():
            sys.modules.setdefault(mod_name, mod)


class VJEPAEncoder:
    """Thin wrapper around a hub-loaded V-JEPA 2.1 encoder.

    Takes ``(T, H, W, 3)`` uint8 frames in :meth:`encode` (matching
    :class:`vjepa_stress.testing.fakes.FakeEncoder`), runs the model on
    its configured device/dtype, and returns L2-normalized
    ``(T_tubelet, P, D)`` float32 tensors on CPU so downstream metric
    code is device-agnostic.
    """

    def __init__(
        self,
        *,
        variant: str,
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.variant = variant
        self.model = model
        self.device = device
        self.dtype = dtype
        self.embedding_dim: int = VARIANT_EMBED_DIMS[variant]
        self.tubelet_size: int = TUBELET_SIZE
        self.patch_size: int = PATCH_SIZE
        self.input_resolution: int = INPUT_RESOLUTION
        self.forward_calls: int = 0

    @property
    def model_name(self) -> str:
        return self.variant

    def output_temporal_dim(self, n_frames: int) -> int:
        if n_frames <= 0 or n_frames % self.tubelet_size != 0:
            raise ValueError(
                f"n_frames must be a positive multiple of tubelet_size="
                f"{self.tubelet_size}; got {n_frames}"
            )
        return n_frames // self.tubelet_size

    def output_spatial_patches(self) -> int:
        return (self.input_resolution // self.patch_size) ** 2

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """Encode ``(T, H, W, 3)`` uint8 RGB frames to ``(T_tubelet, P, D)`` float32.

        Frames are expected to already be at ``input_resolution`` — the
        loader stage (Stage 0.6 Task 4 / SSv2 loader) is responsible for
        resize/crop. We reject mis-sized inputs with a clear error rather
        than silently letting the model interpolate.
        """
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise ValueError(
                f"VJEPAEncoder.encode expects (T,H,W,3) uint8; got "
                f"shape={frames.shape} dtype={frames.dtype}"
            )
        t, h, w, _ = frames.shape
        if h != self.input_resolution or w != self.input_resolution:
            raise ValueError(
                f"frame size must be {self.input_resolution}x{self.input_resolution}; "
                f"got {h}x{w}"
            )
        t_tubelet = self.output_temporal_dim(t)

        # (T, H, W, 3) uint8 → (1, 3, T, H, W) float32 in [0, 1]
        x = torch.from_numpy(frames).to(torch.float32).div_(255.0)
        x = x.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1, C, T, H, W)
        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1, 1)
        x = (x - mean) / std

        x = x.to(device=self.device, dtype=self.dtype)
        self.forward_calls += 1
        out = self.model(x)
        # Some hubconf entries return ``(features, ...)``; defensively unwrap.
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim != 3:
            raise RuntimeError(
                f"unexpected encoder output rank {out.ndim}; expected 3-D "
                f"(B, T_tubelet*P, D), got shape {tuple(out.shape)}"
            )
        b, n_tokens, d = out.shape
        p = self.output_spatial_patches()
        if b != 1 or n_tokens != t_tubelet * p or d != self.embedding_dim:
            raise RuntimeError(
                f"encoder output shape {(b, n_tokens, d)} does not match "
                f"expected (1, {t_tubelet * p}, {self.embedding_dim})"
            )
        out = out.reshape(t_tubelet, p, d).to(dtype=torch.float32, device="cpu")
        return torch.nn.functional.normalize(out, dim=-1)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        return self.encode(frames)


def try_forward_pass(encoder: Encoder, *, n_frames: int = 16) -> torch.Tensor:
    """Run a single dummy forward pass to validate the encoder is wired up.

    Catches misconfigured devices/dtypes early — failing fast at load
    time is far better than failing 4 hours into a sweep.
    """
    h = w = encoder.input_resolution
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n_frames, h, w, 3), dtype=np.uint8)
    out = encoder.encode(frames)
    expected_shape = (
        encoder.output_temporal_dim(n_frames),
        encoder.output_spatial_patches(),
        encoder.embedding_dim,
    )
    if tuple(out.shape) != expected_shape:
        raise RuntimeError(
            f"try_forward_pass: encoder produced shape {tuple(out.shape)}; "
            f"expected {expected_shape}"
        )
    return out


def load_encoder(
    model_name: str,
    *,
    device: str | torch.device = "auto",
    dtype: torch.dtype | None = None,
) -> VJEPAEncoder:
    """Load a frozen V-JEPA 2.1 encoder by canonical name.

    Args:
        model_name: One of ``"vit-base"`` / ``"vit-large"`` / ``"vit-giant"`` /
            ``"vit-gigantic"`` (or the longer ``"vjepa-2.1-..."`` form).
        device: ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``, or a
            :class:`torch.device`.
        dtype: Override for the inference dtype. Default is bfloat16 on
            CUDA, float32 elsewhere.

    Returns:
        A :class:`VJEPAEncoder` ready for :meth:`encode` calls. The
        underlying ``nn.Module`` is in ``eval()`` mode with all params
        frozen, and a startup forward-pass sanity check has already run.
    """
    variant = _normalize_variant(model_name)
    hub_entry = VARIANT_REPOS[variant]
    resolved_device = _resolve_device(device)
    resolved_dtype = dtype if dtype is not None else default_dtype_for_device(resolved_device)

    logger.info(
        "loading V-JEPA 2.1 variant=%s entry=%s device=%s dtype=%s",
        variant,
        hub_entry,
        resolved_device,
        resolved_dtype,
    )

    _ensure_repo_cloned()

    with isolate_for_torch_hub(), patch_meta_url():
        loaded = torch.hub.load(
            HUB_REPO,
            hub_entry,
            trust_repo=True,
            pretrained=True,
        )

    # The hubconf returns ``(encoder, predictor)``; we only need the encoder.
    if isinstance(loaded, (tuple, list)):
        model = loaded[0]
    else:
        model = loaded

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model.to(device=resolved_device, dtype=resolved_dtype)

    # Patch upstream RoPE to keep its output in the activation dtype; without
    # this, bf16 inference fails inside SDPA on V-JEPA 2.1's RoPE attention.
    _patch_rope_dtype(model)

    encoder = VJEPAEncoder(
        variant=variant,
        model=model,
        device=resolved_device,
        dtype=resolved_dtype,
    )
    try_forward_pass(encoder, n_frames=2 * TUBELET_SIZE)
    # Reset the counter so callers see a clean count of their own forwards.
    encoder.forward_calls = 0
    return encoder


def _patch_rope_dtype(model: torch.nn.Module) -> None:
    """Coerce ``rotate_queries_or_keys`` output back to its input dtype.

    Upstream ``facebookresearch/vjepa2`` builds RoPE position frequencies
    as ``1.0 * int_tensor`` (fp32) and combines them with omega via
    ``torch.einsum``, producing fp32 ``emb_sin`` / ``emb_cos`` regardless
    of the input dtype. The rotated query/key therefore leaves the
    function in fp32 even when ``x`` arrives in bf16, so when SDPA later
    sees ``q``/``k`` in fp32 and ``v`` still in bf16 it raises::

        RuntimeError: Expected query, key, and value to have the same
        dtype, but got query.dtype: float key.dtype: float and
        value.dtype: c10::BFloat16 instead.

    The fix is local: cast the rotated output back to ``x.dtype``. The
    rotation math still runs in fp32 (so we keep position precision), but
    the result is round-tripped to the activation dtype, which is what
    every other path in the model already assumes. We patch the upstream
    module's namespace once per process; subsequent loads are no-ops.
    """
    # V-JEPA ships RoPE in two places: ``src.models.utils.modules`` (older
    # signature ``rotate_queries_or_keys(x, pos)``) and
    # ``app.vjepa_2_1.models.utils.modules`` (newer signature with
    # ``n_registers`` and ``has_cls_first`` kwargs). The instance type tells
    # us which one this model is wired to.
    rope_modules: list[object] = []
    for sub in model.modules():
        if type(sub).__name__ == "RoPEAttention":
            mod = inspect.getmodule(type(sub))
            if mod is not None and mod not in rope_modules:
                rope_modules.append(mod)
    if not rope_modules:
        return  # variant doesn't use RoPE — nothing to patch.

    for rope_module in rope_modules:
        if getattr(rope_module, "_vjepa_stress_rope_patched", False):
            continue
        original_fn = rope_module.rotate_queries_or_keys

        def _make_dtype_safe(orig):
            def _rope_dtype_safe(*args, **kwargs):
                # First positional / ``x`` keyword is the activation tensor
                # whose dtype we want to preserve. The signature differs
                # across V-JEPA versions, so accept arbitrary args.
                if args:
                    x = args[0]
                else:
                    x = kwargs.get("x")
                out = orig(*args, **kwargs)
                if isinstance(x, torch.Tensor):
                    return out.to(x.dtype)
                return out

            return _rope_dtype_safe

        rope_module.rotate_queries_or_keys = _make_dtype_safe(original_fn)
        rope_module._vjepa_stress_rope_patched = True
        logger.info(
            "patched %s.rotate_queries_or_keys to preserve input dtype",
            rope_module.__name__,
        )


def vjepa_encoder_factory(model_name: str) -> VJEPAEncoder:
    """Top-level factory the runner can use as ``encoder_factory``."""
    return load_encoder(model_name)
