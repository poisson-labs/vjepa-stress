"""Render a gallery of case study candidates from the Stage 5 analysis.

R1 in ``analysis/coding_agent_requests.md``: produce side-by-side
clean/perturbed renders for the top three candidates in each of the
three failure-mode categories (texture-sticking, identity-drift,
catastrophic). Selection is *top-three-unique-tuples-per-category* (so
multiple sweep rows that share the same ``(video_id, perturbation,
strength)`` tuple count once); we then dedupe across categories on the
same key. Cross-category overlap is logically impossible here because
each category uses a different perturbation, but the dedup is wired in
so the script stays correct if that ever changes.

The seeds we feed into the perturbation are the *exact same* seeds the
sweep used (looked up from ``runs/sweep-b077627/manifest.json``), so
the rendered output is bit-identical to what the encoder saw.

Usage:
    python scripts/render_case_study.py

Side-effects:
    Writes ``figures/case_study_<videoid>_<perturbation>_<sNNN>_*.png``.

We deliberately do **not** auto-normalize, contrast-stretch, or
otherwise post-process the perturbed frame. ``cv2.imwrite`` writes the
raw uint8 bytes (no gamma correction, no tone mapping). If a
``low_light`` frame at strength 1.0 looks near-black, the saved PNG is
near-black — that's what the encoder ingested.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vjepa_stress.data.ssv2 import _load_frames_with_opencv, _resolve_cache_dir
from vjepa_stress.perturbations import all_perturbations
from vjepa_stress.sweep.manifest import lookup_seed, read_manifest

CANDIDATES_CSV = Path("analysis/case_study_candidates.csv")
OUT_DIR = Path("figures")
RUN_DIR = Path("runs/sweep-b077627")
NUM_FRAMES = 16
IMAGE_SIZE = 384

# Frame index used for the single-frame render. Frame 8 is the middle
# of the 16-frame sample, which is a defensible default for any
# perturbation that's roughly uniform along the time axis (gaussian,
# motion_blur, low_light). For frame_drops a single-frame render may
# show no visible change if the chosen index isn't a dropped frame; we
# accept that — frame_drops is a temporal effect that doesn't
# necessarily look striking in a still anyway.
FRAME_IDX = 8

PERTURBATION_BY_NAME = {p.name: p for p in all_perturbations()}


def select_top_three_per_category(rows: list[dict]) -> list[dict]:
    """Take top 3 unique `(video_id, perturbation, strength)` per category,
    then dedupe across categories.

    The CSV is already sorted by ratio descending within each category
    (per the analysis agent's contract), so we just walk the rows in
    order and pick the first three new tuples per category.
    """
    seen_per_cat: dict[str, set[tuple]] = {}
    selected: list[dict] = []
    seen_global: set[tuple] = set()

    for row in rows:
        cat = row["failure_mode"]
        tup = (row["video_id"], row["perturbation"], float(row["strength"]))
        cat_set = seen_per_cat.setdefault(cat, set())
        if len(cat_set) >= 3 or tup in cat_set:
            continue
        cat_set.add(tup)
        if tup in seen_global:
            continue
        seen_global.add(tup)
        # Keep a copy of the row but augment with the categories it satisfies
        # (helps the verdict table downstream).
        out_row = dict(row)
        out_row["_categories"] = [cat]
        selected.append(out_row)
    # Second pass: any tuple that fit a later category too gets the extra
    # tag attached.
    by_tuple = {
        (r["video_id"], r["perturbation"], float(r["strength"])): r
        for r in selected
    }
    for row in rows:
        tup = (row["video_id"], row["perturbation"], float(row["strength"]))
        if tup in by_tuple:
            cat = row["failure_mode"]
            cats = by_tuple[tup]["_categories"]
            if cat not in cats:
                cats.append(cat)
    return selected


def render_sidebyside(
    clean: np.ndarray, perturbed: np.ndarray, strength: float
) -> np.ndarray:
    """Compose ``clean | perturbed`` with white labels.

    A small white separator (``8`` pixels wide) sits between the two
    panels so they don't bleed into each other when the perturbed side
    is near-black.
    """
    h, w = clean.shape[:2]
    canvas = np.zeros((h, 2 * w + 8, 3), dtype=np.uint8)
    canvas[:, :w] = clean
    canvas[:, w + 8 :] = perturbed
    canvas[:, w : w + 8] = 255  # separator
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    # DejaVu is in the standard Linux font path on RunPod; fall back to
    # PIL's built-in if absent (mac dev box).
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    font = None
    for fp in font_paths:
        try:
            font = ImageFont.truetype(fp, 18)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    # White text with a thin black halo so it stays legible against any
    # background — including a near-black perturbed panel.
    def _label(xy, text):
        x, y = xy
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

    _label((10, 10), "clean")
    _label((w + 8 + 10, 10), f"perturbed @ s={strength:.3f}")
    return np.asarray(img)


def _resolve_video_path(split_dir: Path, video_id: str) -> Path:
    candidates = sorted(split_dir.glob(f"{video_id}.*"))
    if not candidates:
        raise FileNotFoundError(
            f"No video file found for video_id={video_id!r} under "
            f"{split_dir} (tried glob {video_id}.*). Confirm the SSv2 "
            f"validation tree is mounted at the expected path."
        )
    return candidates[0]


def _save_png(out_dir: Path, stem: str, suffix: str, frame_rgb: np.ndarray) -> Path:
    """Write a PNG with no auto post-processing.

    ``cv2.imwrite`` on PNG with default parameters does not apply gamma
    correction or color management — it writes the raw uint8 bytes
    sRGB-untagged. That's the right choice here: we want to document
    what the encoder ingested, not what looks nice.
    """
    out = out_dir / f"{stem}_{suffix}.png"
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out), bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {out}")
    return out


def main() -> int:
    if not CANDIDATES_CSV.is_file():
        print(f"ERROR: {CANDIDATES_CSV} not found", file=sys.stderr)
        return 2
    if not RUN_DIR.is_dir():
        print(f"ERROR: {RUN_DIR} not found (need it for seed lookup)", file=sys.stderr)
        return 2

    manifest = read_manifest(RUN_DIR)
    with open(CANDIDATES_CSV) as f:
        rows = list(csv.DictReader(f))

    selected = select_top_three_per_category(rows)
    print(f"Selected {len(selected)} unique candidates:")
    for r in selected:
        print(
            f"  {r['video_id']:>8s}  {r['perturbation']:14s}  "
            f"s={float(r['strength']):.3f}  cats={','.join(r['_categories'])}"
        )

    cache_root = _resolve_cache_dir(None)
    split_dir = cache_root / "validation"
    if not split_dir.is_dir():
        print(
            f"ERROR: SSv2 split dir not found at {split_dir}. "
            f"Set $VJEPA_STRESS_SSV2_DIR.",
            file=sys.stderr,
        )
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for row in selected:
        video_id = row["video_id"]
        pert_name = row["perturbation"]
        strength = float(row["strength"])

        path = _resolve_video_path(split_dir, video_id)
        frames_clean = _load_frames_with_opencv(path, NUM_FRAMES, IMAGE_SIZE)
        seed = lookup_seed(manifest, video_id, pert_name)
        pert = PERTURBATION_BY_NAME[pert_name]
        frames_perturbed = pert.apply(
            frames_clean, strength=strength, seed=seed
        )

        clean = frames_clean[FRAME_IDX]
        perturbed = frames_perturbed[FRAME_IDX]
        s_str = f"{strength:.3f}"
        stem = f"case_study_{video_id}_{pert_name}_{s_str}"

        written.append(_save_png(OUT_DIR, stem, "clean", clean))
        written.append(_save_png(OUT_DIR, stem, "perturbed", perturbed))
        side = render_sidebyside(clean, perturbed, strength)
        written.append(_save_png(OUT_DIR, stem, "sidebyside", side))
        print(f"  wrote: {stem}_(clean|perturbed|sidebyside).png")

    print(f"\nDone — {len(written)} PNGs in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
