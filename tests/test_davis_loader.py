"""Tests for ``vjepa_stress.data.davis`` — the DAVIS 2017 loader.

Most of the surface is exercised against a synthetic DAVIS-style tree
written at test time so the fast suite doesn't depend on a real DAVIS
download. The two tests that *do* need the real archive are gated on
``@pytest.mark.slow`` and soft-skip when no DAVIS tree is on the host.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from vjepa_stress.data.davis import (
    DAVIS_2017_VAL_COUNT,
    DAVIS_INSTALL_URL,
    DAVISDataset,
    DAVISSample,
    _resolve_cache_dir,
    load_davis_dataset,
)


# ---------- helpers --------------------------------------------------------


def _write_synthetic_jpeg(path: Path, *, height: int = 60, width: int = 80,
                           value: int = 0) -> None:
    img = np.full((height, width, 3), value % 256, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_synthetic_palette_png(
    path: Path, *, height: int = 60, width: int = 80, instance_pattern: str = "split"
) -> None:
    """Write a paletted PNG with deterministic instance IDs.

    ``instance_pattern="split"`` puts instance 1 on the left half and
    instance 2 on the right half; ``"empty"`` is all-background.
    """
    arr = np.zeros((height, width), dtype=np.uint8)
    if instance_pattern == "split":
        arr[:, : width // 2] = 1
        arr[:, width // 2 :] = 2
    elif instance_pattern == "empty":
        pass
    else:
        raise ValueError(instance_pattern)
    # ``frombytes`` lets us specify mode "P" without the deprecated
    # ``Image.fromarray(..., mode=...)`` call. The palette table is just
    # a sanity stand-in — the loader reads palette *indices*, not the
    # mapped RGB values, so a synthetic palette is enough.
    im = Image.frombytes("P", (width, height), arr.tobytes())
    palette = bytes([i for i in range(256)] * 3)
    im.putpalette(palette)
    im.save(str(path))


def _build_synthetic_davis(
    root: Path, *, sequences: list[str], n_frames: int = 6
) -> None:
    """Create a minimal but real-shaped DAVIS layout under ``root``."""
    (root / "ImageSets" / "2017").mkdir(parents=True)
    (root / "ImageSets" / "2017" / "val.txt").write_text(
        "\n".join(sequences) + "\n"
    )
    for seq in sequences:
        jpeg_dir = root / "JPEGImages" / "480p" / seq
        mask_dir = root / "Annotations" / "480p" / seq
        jpeg_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        for i in range(n_frames):
            _write_synthetic_jpeg(
                jpeg_dir / f"{i:05d}.jpg", height=60, width=80, value=i * 30
            )
            _write_synthetic_palette_png(
                mask_dir / f"{i:05d}.png", height=60, width=80,
                instance_pattern="split",
            )


def _real_davis_available() -> bool:
    root = _resolve_cache_dir(None)
    if not root.is_dir():
        return False
    return (root / "JPEGImages" / "480p").is_dir() or (
        root / "DAVIS" / "JPEGImages" / "480p"
    ).is_dir()


# ---------- error paths ----------------------------------------------------


def test_load_with_missing_cache_dir_raises_with_install_url(tmp_path):
    with pytest.raises(FileNotFoundError) as ei:
        load_davis_dataset(cache_dir=tmp_path / "does_not_exist")
    msg = str(ei.value)
    assert DAVIS_INSTALL_URL in msg
    assert "VJEPA_STRESS_DAVIS_DIR" in msg


def test_load_with_missing_subdirs_raises_with_install_url(tmp_path):
    """Cache root exists but lacks JPEGImages/Annotations → clear error."""
    with pytest.raises(FileNotFoundError) as ei:
        load_davis_dataset(cache_dir=tmp_path)
    msg = str(ei.value)
    assert DAVIS_INSTALL_URL in msg
    assert "JPEGImages" in msg or "Annotations" in msg


def test_load_with_empty_sequence_list_raises(tmp_path):
    """Subdirs exist but val.txt is empty → clear error."""
    (tmp_path / "ImageSets" / "2017").mkdir(parents=True)
    (tmp_path / "ImageSets" / "2017" / "val.txt").write_text("")
    (tmp_path / "JPEGImages" / "480p").mkdir(parents=True)
    (tmp_path / "Annotations" / "480p").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as ei:
        load_davis_dataset(cache_dir=tmp_path)
    assert DAVIS_INSTALL_URL in str(ei.value)


# ---------- inner-DAVIS unwrap --------------------------------------------


def test_load_unwraps_inner_davis_directory(tmp_path):
    """Some downloads keep an inner ``DAVIS/`` dir; loader should unwrap."""
    inner = tmp_path / "DAVIS"
    _build_synthetic_davis(inner, sequences=["alpha"])
    ds = load_davis_dataset(
        cache_dir=tmp_path, image_size=32, num_frames=4
    )
    assert ds.video_ids() == ["alpha"]


# ---------- shape / dtype contract ----------------------------------------


def test_video_id_is_stable_string(tmp_path):
    _build_synthetic_davis(
        tmp_path, sequences=["bmx-trees", "blackswan"]
    )
    ds = load_davis_dataset(cache_dir=tmp_path, image_size=32, num_frames=4)
    ids = ds.video_ids()
    # Order should match val.txt — the loader does NOT re-sort, so the
    # caller controls iteration order via the ImageSets file.
    assert ids == ["bmx-trees", "blackswan"]
    assert all(isinstance(i, str) for i in ids)

    sample = ds[0]
    assert isinstance(sample.video_id, str)
    assert sample.video_id == "bmx-trees"
    assert sample.sequence_id == sample.video_id


def test_mask_shape_matches_frame_shape(tmp_path):
    _build_synthetic_davis(tmp_path, sequences=["alpha"], n_frames=8)
    ds = load_davis_dataset(cache_dir=tmp_path, image_size=64, num_frames=4)
    s = ds[0]
    assert s.frames.shape == (4, 64, 64, 3)
    assert s.masks.shape == (4, 64, 64)
    assert s.frames.dtype == np.uint8


def test_mask_values_are_integers(tmp_path):
    """Mask must come back as integer instance IDs, not RGB. The synthetic
    fixture writes IDs 1 (left) and 2 (right) — both must survive
    decode+resize unchanged."""
    _build_synthetic_davis(tmp_path, sequences=["alpha"])
    ds = load_davis_dataset(cache_dir=tmp_path, image_size=32, num_frames=4)
    s = ds[0]
    assert np.issubdtype(s.masks.dtype, np.integer)
    unique = sorted(np.unique(s.masks).tolist())
    # Background may or may not survive after the center-crop, but the
    # two foreground IDs must — and there must be no spurious values
    # like 64/128/etc. that would suggest RGB-decoding went wrong.
    assert 1 in unique
    assert 2 in unique
    for v in unique:
        assert v in {0, 1, 2}, f"unexpected mask value {v}: {unique}"


def test_frame_and_mask_share_index_sample(tmp_path):
    """Frames and masks must use the *same* np.linspace indices into the
    on-disk file list — otherwise M3 keypoints in frame ``t`` wouldn't
    correspond to mask annotations in frame ``t``.

    The fixture encodes the on-disk frame index into pixel intensity
    (``value=i*30``). After uniform sampling 4 of 12 source frames we
    expect to see exactly the indices np.linspace produces — and the
    masks should pair 1:1 with those indices.
    """
    _build_synthetic_davis(tmp_path, sequences=["alpha"], n_frames=12)
    ds = load_davis_dataset(cache_dir=tmp_path, image_size=8, num_frames=4)
    s = ds[0]
    expected = np.linspace(0, 11, 4).round().astype(int)
    # value=i*30 → mean pixel ~i*30 (post-crop center-pixel intensity).
    observed_intensity = s.frames[:, 0, 0, 0].astype(int)  # one corner
    expected_intensity = (expected * 30) % 256
    assert (observed_intensity == expected_intensity).all(), (
        f"frame indices look wrong: observed {observed_intensity.tolist()} "
        f"vs expected {expected_intensity.tolist()}"
    )
    # Masks at all 4 indices keep the split pattern → both 1 and 2 present.
    for t in range(4):
        u = sorted(np.unique(s.masks[t]).tolist())
        assert 1 in u and 2 in u, f"mask {t} lost an instance: {u}"


# ---------- runner protocol -----------------------------------------------


def test_dataset_iteration_yields_runner_compatible_samples(tmp_path):
    _build_synthetic_davis(tmp_path, sequences=["a", "b", "c"])
    ds = load_davis_dataset(
        cache_dir=tmp_path, image_size=16, num_frames=4
    )
    seen = []
    for sample in ds:
        seen.append(sample.video_id)
        assert isinstance(sample, DAVISSample)
        assert sample.frames.dtype == np.uint8
        assert sample.frames.ndim == 4
        assert sample.masks.ndim == 3
    assert seen == ds.video_ids() == ["a", "b", "c"]


def test_explicit_sequence_ids_override_imagesets(tmp_path):
    _build_synthetic_davis(tmp_path, sequences=["a", "b", "c"])
    ds = load_davis_dataset(
        cache_dir=tmp_path,
        image_size=16,
        num_frames=4,
        sequence_ids=["c", "a"],  # subset, custom order
    )
    assert ds.video_ids() == ["c", "a"]


def test_imagesets_fallback_to_directory_listing(tmp_path):
    """If val.txt is missing, list JPEGImages/480p/ instead."""
    _build_synthetic_davis(tmp_path, sequences=["alpha", "beta"])
    (tmp_path / "ImageSets" / "2017" / "val.txt").unlink()
    ds = load_davis_dataset(cache_dir=tmp_path, image_size=16, num_frames=4)
    # Directory listing is sorted lexicographically.
    assert ds.video_ids() == ["alpha", "beta"]


# ---------- env-var resolution --------------------------------------------


def test_env_var_overrides_default(tmp_path, monkeypatch):
    _build_synthetic_davis(tmp_path, sequences=["alpha"])
    monkeypatch.setenv("VJEPA_STRESS_DAVIS_DIR", str(tmp_path))
    ds = load_davis_dataset(image_size=16, num_frames=4)
    assert ds.video_ids() == ["alpha"]


# ---------- slow / real-data path -----------------------------------------


@pytest.mark.slow
def test_load_returns_30_videos_real_davis():
    if not _real_davis_available():
        pytest.skip(
            f"No DAVIS tree at {_resolve_cache_dir(None)}; download from "
            f"{DAVIS_INSTALL_URL}."
        )
    ds = load_davis_dataset()
    assert len(ds) == DAVIS_2017_VAL_COUNT == 30


@pytest.mark.slow
def test_load_is_deterministic_real_davis():
    if not _real_davis_available():
        pytest.skip(
            f"No DAVIS tree at {_resolve_cache_dir(None)}; download from "
            f"{DAVIS_INSTALL_URL}."
        )
    a = load_davis_dataset().video_ids()
    b = load_davis_dataset().video_ids()
    assert a == b
