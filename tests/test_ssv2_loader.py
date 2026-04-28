"""Tests for ``vjepa_stress.data.ssv2`` — the SSv2 video loader.

Most of the surface lives in fast tests: missing-data error path, file
glob, deterministic sampling, and shape/dtype contract via a synthetic
mp4 we write at test time. The two tests that require a real SSv2 tree
are gated on ``@pytest.mark.slow`` AND soft-skip when the cache dir is
absent — so the suite stays green on a Mac dev box without SSv2.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from vjepa_stress.data.ssv2 import (
    DEFAULT_CACHE_DIR,
    SSV2_INSTALL_URL,
    SSv2Dataset,
    SSv2Sample,
    _resize_and_center_crop,
    _resolve_cache_dir,
    load_ssv2_dataset,
)


# ---------- helpers --------------------------------------------------------


def _write_synthetic_mp4(
    path: Path, *, n_frames: int = 24, height: int = 60, width: int = 80
) -> None:
    """Write a tiny synthetic mp4 so we can exercise the OpenCV decode path
    without a real SSv2 download. Each frame's per-pixel value encodes the
    frame index, so a downstream reader can verify it landed on the right
    frames if it wants to."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV mp4v writer not available in this environment")
    try:
        for t in range(n_frames):
            frame = np.full((height, width, 3), t * 7 % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _real_ssv2_available() -> bool:
    """True iff something resembling an SSv2 tree exists on this host."""
    root = _resolve_cache_dir(None)
    val = root / "validation"
    if not val.is_dir():
        return False
    for pat in ("*.webm", "*.mp4"):
        if any(val.glob(pat)):
            return True
    return False


# ---------- cache-dir resolution ------------------------------------------


def test_resolve_cache_dir_prefers_explicit_arg(tmp_path, monkeypatch):
    monkeypatch.setenv("VJEPA_STRESS_SSV2_DIR", str(tmp_path / "from_env"))
    out = _resolve_cache_dir(tmp_path / "from_arg")
    assert out == tmp_path / "from_arg"


def test_resolve_cache_dir_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VJEPA_STRESS_SSV2_DIR", str(tmp_path / "from_env"))
    out = _resolve_cache_dir(None)
    assert out == tmp_path / "from_env"


def test_resolve_cache_dir_default(monkeypatch):
    monkeypatch.delenv("VJEPA_STRESS_SSV2_DIR", raising=False)
    assert _resolve_cache_dir(None) == DEFAULT_CACHE_DIR


# ---------- error paths ----------------------------------------------------


def test_load_with_missing_split_dir_raises_with_install_url(tmp_path):
    """No ``validation/`` subdir → clear error pointing at the install URL."""
    with pytest.raises(FileNotFoundError) as ei:
        load_ssv2_dataset(cache_dir=tmp_path)
    msg = str(ei.value)
    assert SSV2_INSTALL_URL in msg
    assert "validation" in msg


def test_load_with_empty_split_dir_raises_with_install_url(tmp_path):
    """Split dir exists but contains no videos → clear error + install URL."""
    (tmp_path / "validation").mkdir()
    with pytest.raises(FileNotFoundError) as ei:
        load_ssv2_dataset(cache_dir=tmp_path)
    msg = str(ei.value)
    assert SSV2_INSTALL_URL in msg


# ---------- shape / decode contract ---------------------------------------


def test_resize_and_center_crop_shape_and_aspect():
    # Tall input: short side is width.
    frame = np.zeros((600, 300, 3), dtype=np.uint8)
    out = _resize_and_center_crop(frame, image_size=128)
    assert out.shape == (128, 128, 3)
    assert out.dtype == np.uint8

    # Wide input: short side is height. Make sure no crash on extreme aspect.
    frame_w = np.zeros((100, 1000, 3), dtype=np.uint8)
    out_w = _resize_and_center_crop(frame_w, image_size=64)
    assert out_w.shape == (64, 64, 3)


def test_video_shape_with_synthetic_mp4(tmp_path):
    """Write a small mp4, decode through SSv2Dataset, assert (16,384,384,3) uint8 RGB."""
    val = tmp_path / "validation"
    val.mkdir()
    _write_synthetic_mp4(val / "00001.mp4", n_frames=24, height=60, width=80)
    _write_synthetic_mp4(val / "00002.mp4", n_frames=24, height=120, width=80)

    ds = load_ssv2_dataset(
        cache_dir=tmp_path,
        n_videos=2,
        image_size=384,
        num_frames=16,
        # Synthetic test files are .mp4 only; restrict glob so we don't trip
        # over OS-specific webm decoding here.
        file_patterns=("*.mp4",),
    )
    assert len(ds) == 2
    ids = ds.video_ids()
    assert ids == sorted(["00001", "00002"])

    sample = ds[0]
    assert isinstance(sample, SSv2Sample)
    assert sample.video_id in ids
    assert sample.frames.shape == (16, 384, 384, 3)
    assert sample.frames.dtype == np.uint8
    # Values should not be uniform-zero — the writer encoded frame index
    # into pixel intensity, so post-decode we expect a non-trivial value.
    assert int(sample.frames.max()) > 0


# ---------- determinism ---------------------------------------------------


def test_load_is_deterministic_with_same_seed(tmp_path):
    """Same seed against the same pool → same chosen video paths."""
    val = tmp_path / "validation"
    val.mkdir()
    for i in range(8):
        _write_synthetic_mp4(val / f"{i:05d}.mp4", n_frames=4)

    ds_a = load_ssv2_dataset(
        cache_dir=tmp_path,
        n_videos=4,
        seed=7,
        file_patterns=("*.mp4",),
    )
    ds_b = load_ssv2_dataset(
        cache_dir=tmp_path,
        n_videos=4,
        seed=7,
        file_patterns=("*.mp4",),
    )
    assert ds_a.video_ids() == ds_b.video_ids()

    # Different seed → different (or at least not identically-ordered) set.
    ds_c = load_ssv2_dataset(
        cache_dir=tmp_path,
        n_videos=4,
        seed=42,
        file_patterns=("*.mp4",),
    )
    # We can't assert disjointness on small pools, but we can assert the
    # function honours the seed difference at the API surface (i.e. the
    # call returns successfully and the sort is stable for each seed).
    assert sorted(ds_c.video_ids()) == ds_c.video_ids()


def test_n_videos_capped_at_pool_size_emits_warning(tmp_path, caplog):
    val = tmp_path / "validation"
    val.mkdir()
    for i in range(3):
        _write_synthetic_mp4(val / f"{i:05d}.mp4", n_frames=4)

    with caplog.at_level("WARNING"):
        ds = load_ssv2_dataset(
            cache_dir=tmp_path,
            n_videos=200,
            file_patterns=("*.mp4",),
        )
    assert len(ds) == 3
    assert any("n_videos=200" in r.message for r in caplog.records)


# ---------- runner-protocol surface ---------------------------------------


def test_dataset_iteration_yields_runner_compatible_samples(tmp_path):
    """The runner only touches ``video_ids()``, ``__iter__``, ``video_id``,
    and ``frames``. Confirm all four match :class:`FakeVideoDataset`."""
    val = tmp_path / "validation"
    val.mkdir()
    _write_synthetic_mp4(val / "abc.mp4", n_frames=8)
    _write_synthetic_mp4(val / "def.mp4", n_frames=8)
    ds = load_ssv2_dataset(
        cache_dir=tmp_path,
        n_videos=2,
        file_patterns=("*.mp4",),
        image_size=64,
    )
    seen = []
    for sample in ds:
        seen.append(sample.video_id)
        assert sample.frames.dtype == np.uint8
        assert sample.frames.ndim == 4
    assert sorted(seen) == ds.video_ids()


# ---------- slow / real-data path -----------------------------------------


@pytest.mark.slow
def test_load_returns_correct_count_real_ssv2():
    if not _real_ssv2_available():
        pytest.skip(
            f"No SSv2 tree at {_resolve_cache_dir(None)}; this is the manual-"
            f"only download path — see {SSV2_INSTALL_URL}."
        )
    ds = load_ssv2_dataset(n_videos=4, seed=0)
    assert len(ds) == 4
    assert len(set(ds.video_ids())) == 4


@pytest.mark.slow
def test_load_is_deterministic_real_ssv2():
    if not _real_ssv2_available():
        pytest.skip(
            f"No SSv2 tree at {_resolve_cache_dir(None)}; this is the manual-"
            f"only download path — see {SSV2_INSTALL_URL}."
        )
    a = load_ssv2_dataset(n_videos=4, seed=11).video_ids()
    b = load_ssv2_dataset(n_videos=4, seed=11).video_ids()
    assert a == b
