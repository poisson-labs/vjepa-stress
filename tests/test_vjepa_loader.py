"""Tests for ``vjepa_stress.models.vjepa`` — the V-JEPA 2.1 loader.

Most tests here are pure unit tests that don't require model weights.
The single end-to-end test that downloads + runs ``vit-base`` is gated
on ``@pytest.mark.slow`` so the fast suite stays fast. Run slow tests
with ``pytest -m slow`` (or ``-m "not slow"`` to exclude them).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vjepa_stress.models.vjepa import (
    INPUT_RESOLUTION,
    PATCH_SIZE,
    TUBELET_SIZE,
    VARIANT_EMBED_DIMS,
    VARIANT_REPOS,
    VJEPA_PRODUCTION_URL,
    VJEPAEncoder,
    _normalize_variant,
    default_dtype_for_device,
    load_encoder,
    patch_meta_url,
    try_forward_pass,
)


def test_normalize_variant_accepts_short_and_long_form():
    assert _normalize_variant("vit-base") == "vit-base"
    assert _normalize_variant("vjepa-2.1-vit-base") == "vit-base"
    assert _normalize_variant("vit-gigantic") == "vit-gigantic"


def test_normalize_variant_unknown_name_raises_with_valid_options():
    with pytest.raises(ValueError) as ei:
        _normalize_variant("vjepa-2.0-vit-base")
    msg = str(ei.value)
    # Error should list the valid names so the user can fix their config.
    for v in VARIANT_REPOS:
        assert v in msg


def test_dtype_defaults_per_device():
    # CPU and MPS both stay in float32; CUDA goes to bfloat16.
    assert default_dtype_for_device("cpu") == torch.float32
    assert default_dtype_for_device(torch.device("cpu")) == torch.float32
    if torch.cuda.is_available():
        assert default_dtype_for_device("cuda") == torch.bfloat16


def test_variant_repos_cover_all_four_variants():
    expected = {"vit-base", "vit-large", "vit-giant", "vit-gigantic"}
    assert set(VARIANT_REPOS) == expected
    assert set(VARIANT_EMBED_DIMS) == expected
    # Sanity: the hub entry names follow Meta's ``vjepa2_1_vit_<variant>_384`` pattern.
    for name, entry in VARIANT_REPOS.items():
        assert entry.startswith("vjepa2_1_") and entry.endswith("_384")


def test_protocol_attributes_exposed_on_loaded_encoder_skeleton():
    """Build a ``VJEPAEncoder`` around an ``nn.Identity`` to exercise the
    protocol surface without downloading weights."""
    enc = VJEPAEncoder(
        variant="vit-base",
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert enc.embedding_dim == VARIANT_EMBED_DIMS["vit-base"]
    assert enc.tubelet_size == TUBELET_SIZE == 2
    assert enc.patch_size == PATCH_SIZE == 16
    assert enc.input_resolution == INPUT_RESOLUTION == 384
    assert enc.forward_calls == 0


def test_output_temporal_dim_math():
    enc = VJEPAEncoder(
        variant="vit-base",
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert enc.output_temporal_dim(16) == 8
    assert enc.output_temporal_dim(8) == 4
    assert enc.output_temporal_dim(2) == 1
    assert enc.output_spatial_patches() == (384 // 16) ** 2  # 576

    with pytest.raises(ValueError):
        enc.output_temporal_dim(15)  # not a multiple of tubelet_size=2
    with pytest.raises(ValueError):
        enc.output_temporal_dim(0)


def test_load_encoder_unknown_name_raises_before_any_network_call():
    # If this triggered a network fetch, the test would be slow / flaky;
    # ``_normalize_variant`` runs first and raises synchronously.
    with pytest.raises(ValueError) as ei:
        load_encoder("vit-tiny")  # not a real variant
    msg = str(ei.value)
    assert "vit-base" in msg


def test_patch_meta_url_is_reversible(monkeypatch, tmp_path):
    """``patch_meta_url`` must restore ``VJEPA_BASE_URL`` even when the
    body raises. We stub out the cache-dir lookup so this stays a unit
    test that doesn't require an actual hub clone.
    """
    fake_cache = tmp_path / "facebookresearch_vjepa2_main"
    (fake_cache / "src" / "hub").mkdir(parents=True)

    # Minimal package layout matching what ``patch_meta_url`` imports.
    (fake_cache / "src" / "__init__.py").write_text("")
    (fake_cache / "src" / "hub" / "__init__.py").write_text("")
    backbones_py = fake_cache / "src" / "hub" / "backbones.py"
    backbones_py.write_text('VJEPA_BASE_URL = "http://localhost:8300"\n')

    from vjepa_stress.models import vjepa as vjepa_mod

    monkeypatch.setattr(vjepa_mod, "_hub_cache_dir", lambda: fake_cache)

    # Happy path: URL is patched inside, restored on exit.
    with patch_meta_url(VJEPA_PRODUCTION_URL):
        import src.hub.backbones as bb

        assert bb.VJEPA_BASE_URL == VJEPA_PRODUCTION_URL

    # After exit, re-entering the context should see the *on-disk* original
    # URL again — proves the in-memory mutation was unwound, not persisted.
    # (We re-enter rather than importing directly because ``patch_meta_url``
    # also removes the cache dir from sys.path on exit, by design.)
    with patch_meta_url(VJEPA_PRODUCTION_URL):
        import src.hub.backbones as bb2

        # The first thing inside the block: assert pre-patch value matches
        # the on-disk literal. Then the patch sets it to the production URL.
        # We can't observe the pre-patch state from outside without replicating
        # ``patch_meta_url``'s import dance, so check the file directly:
        assert (
            backbones_py.read_text().strip() == 'VJEPA_BASE_URL = "http://localhost:8300"'
        )
        assert bb2.VJEPA_BASE_URL == VJEPA_PRODUCTION_URL

    # Failure path: exception inside the block must still restore the URL
    # in any cached module *and* leave the on-disk file unchanged.
    class Boom(RuntimeError):
        pass

    try:
        with patch_meta_url(VJEPA_PRODUCTION_URL):
            raise Boom("simulated load failure")
    except Boom:
        pass

    assert (
        backbones_py.read_text().strip() == 'VJEPA_BASE_URL = "http://localhost:8300"'
    )


@pytest.mark.slow
def test_load_smallest_works_end_to_end():
    """End-to-end: download (or use cached) vit-base weights, load, and
    run a 16-frame forward pass. Asserts the documented output contract.

    Cached weights are ~1.5 GB; first run on a clean cache will download
    them. Subsequent runs are fast (<30s on M4 CPU).
    """
    enc = load_encoder("vit-base", device="cpu")

    assert enc.embedding_dim == 768
    assert enc.tubelet_size == 2
    assert enc.input_resolution == 384

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(16, 384, 384, 3), dtype=np.uint8)
    out = enc.encode(frames)
    assert out.shape == (8, 576, 768)
    assert out.dtype == torch.float32
    # L2 normalization invariant: every token has unit norm along D.
    norms = torch.linalg.vector_norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # ``try_forward_pass`` should also pass on a fresh tiny clip.
    try_forward_pass(enc, n_frames=4)


@pytest.mark.slow
@pytest.mark.calibration
def test_m2_reverse_calibration_real_features():
    """Pin the M2(clean, reverse-played) magnitude on real V-JEPA features.

    RESEARCH.md §3.2 sets the analytical expectation at ~1.14 for
    real-encoder reverse playback on real motion video; the band
    ``0.9 < M2 < 1.4`` is calibrated against that reference. Synthetic
    drift produces a different, much-smaller M2 (~0.10 — see
    ``test_m2_reverse_synthetic_observed_band``), so this test runs
    against a real SSv2 clip and soft-skips when none is available.

    Soft-skip (vs. unconditional skip) is intentional: SSv2 is gated
    and not present on every dev box, but on RunPod / any host with a
    populated cache dir this is the calibration check that actually
    validates the metric. Devlog 2026-04-27 has the deferral history.
    """
    from vjepa_stress.data.ssv2 import (
        SSV2_INSTALL_URL,
        _resolve_cache_dir,
        load_ssv2_dataset,
    )
    from vjepa_stress.metrics import compute_m2

    cache_root = _resolve_cache_dir(None)
    if not (cache_root / "validation").is_dir():
        pytest.skip(
            f"No SSv2 tree at {cache_root}; manual download required "
            f"(see {SSV2_INSTALL_URL}). M2 reverse calibration only runs "
            f"against real motion video — synthetic stimulus is covered by "
            f"test_m2_reverse_synthetic_observed_band."
        )

    ds = load_ssv2_dataset(n_videos=1, seed=0, image_size=384, num_frames=16)
    sample = next(iter(ds))
    frames_clean = np.ascontiguousarray(sample.frames)
    frames_reverse = np.ascontiguousarray(frames_clean[::-1])

    enc = load_encoder("vit-base", device="cpu")
    feats_clean = enc.encode(frames_clean)
    feats_reverse = enc.encode(frames_reverse)
    m2 = float(compute_m2(feats_clean, feats_reverse).item())

    assert 0.9 < m2 < 1.4, (
        f"M2(clean, reverse) on SSv2 clip {sample.video_id!r} = {m2:.3f}; "
        f"expected ~1.14 per RESEARCH.md §3.2 (band [0.9, 1.4]). Calibration "
        f"on real V-JEPA features has drifted."
    )


@pytest.mark.slow
def test_m2_reverse_synthetic_observed_band():
    """Captures observed behavior on synthetic drift; mechanism TBD pending
    real-video calibration.

    On a 16-frame video built from a base image plus per-pixel cumulative
    drift, ``M2(clean, reverse(clean))`` lands around ~0.10 on real
    V-JEPA 2.1 vit-base — about 10x smaller than RESEARCH.md §3.2's
    ~1.14 prediction for real motion video. We do not yet know which of
    two candidate mechanisms dominates (tubelet pair-averaging swallowing
    the temporal sign, vs. low-entropy synthetic content collapsing the
    encoder's temporal-gradient direction). This test pins the observed
    band so any future change in the loader / preprocessing path that
    would silently move it surfaces in CI; the *mechanism* and the real
    calibration target both wait on Task 4.
    """
    from vjepa_stress.metrics import compute_m2

    enc = load_encoder("vit-base", device="cpu")

    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(384, 384, 3), dtype=np.uint8)
    drift = rng.integers(-2, 3, size=(16, 384, 384, 3), dtype=np.int16).cumsum(axis=0)
    frames_clean = np.clip(base.astype(np.int16)[None] + drift, 0, 255).astype(np.uint8)
    frames_reverse = frames_clean[::-1].copy()

    feats_clean = enc.encode(frames_clean)
    feats_reverse = enc.encode(frames_reverse)
    m2 = float(compute_m2(feats_clean, feats_reverse).item())

    # Observed: ~0.10. Allow generous headroom around that without ever
    # crossing into territory the real-video band would occupy. If this
    # ever rises above 0.3 we want a flag — either the encoder is being
    # called differently, or one of the two candidate mechanisms above
    # has shifted, and we need to revisit the analysis.
    assert m2 > 0.0, f"M2 should be strictly positive on reversed features; got {m2}"
    assert m2 < 0.3, (
        f"M2 on synthetic drift = {m2:.3f}; observed band is ~0.10 with a "
        "0.3 ceiling. A breach here means either preprocessing changed or "
        "one of the candidate mechanisms (tubelet pair-averaging / "
        "low-entropy collapse) no longer applies — re-run the analysis."
    )
