"""Real-feature M2 calibration on SSv2 reverse playback.

Answers the deferred Stage 0.6 Task 3.5 question: what does M2 actually
look like on real V-JEPA 2.1 features when a clean clip is compared to
its reverse-played counterpart?

RESEARCH.md §3.2 predicts the band ``[0.9, 1.4]`` with center near 1.14
for 16-frame input at tubelet=2. The synthetic-drift companion test
pinned ~0.10 — a structural finding (V-JEPA represents monotonic
synthetic stimuli with reversal-symmetric gradient fields), not a bug —
and left the real-video calibration question open.

This script is **standalone**. It is not a pytest test: the calibration
*is* an empirical observation, not an assertion against a predicted
band. The band itself is what we're calibrating. The result lands in:

- stdout (the user pipes through ``tee logs/m2_calibration.log``);
- ``data/calibration/m2_reverse_calibration.json`` (per-clip + summary);
- the devlog (a follow-up entry written by the operator after running).

Decision-tree branches handled at the bottom of ``main()``:

    M2 ∈ [0.9, 1.4]     → "in_band"      — prediction confirmed
    M2 ∈ [0.5, 0.9)     → "low_band"     — user picks band/threshold
    M2 < 0.5            → "below_floor"  — surprise; auto-runs diagnostics
    M2 > 1.4            → "above_band"   — anti-symmetric gradients; non-blocking

Pre-registration discipline: this script does NOT edit RESEARCH.md, H2
thresholds, or M2 itself, regardless of the result. We report what we
predicted and what we found; the user calls the next move.

Usage:
    mkdir -p logs
    python scripts/calibrate_m2_real.py 2>&1 | tee logs/m2_calibration.log
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from vjepa_stress.data.ssv2 import (
    SSV2_INSTALL_URL,
    _resolve_cache_dir,
    load_ssv2_dataset,
)
from vjepa_stress.metrics import compute_m1, compute_m2
from vjepa_stress.models.vjepa import load_encoder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "calibration" / "m2_reverse_calibration.json"

# RESEARCH.md §3.2 — predicted band for 16-frame V-JEPA 2.1 reverse playback.
PREDICTED_BAND: tuple[float, float] = (0.9, 1.4)

# Tubelet-collapsed shape for V-JEPA 2.1 vit-base on 16-frame, 384-px input.
RANDOM_FEATURES_SHAPE: tuple[int, int, int] = (8, 576, 768)


def _check_ssv2_present() -> None:
    """Fail fast with the registration URL if SSv2 isn't on disk yet."""
    root = _resolve_cache_dir(None)
    if not (root / "validation").is_dir():
        sys.stderr.write(
            f"\nERROR: SSv2 validation tree not found at {root / 'validation'}.\n"
            f"\n"
            f"SSv2 is registration-gated and must be downloaded manually:\n"
            f"  1. Register at {SSV2_INSTALL_URL}\n"
            f"  2. Download something-something-v2 (~50 GB)\n"
            f"  3. Extract validation videos to {root}/validation/\n"
            f"\n"
            f"You can also point the loader elsewhere via "
            f"$VJEPA_STRESS_SSV2_DIR.\n"
        )
        sys.exit(2)


def _summary(values: list[float]) -> dict[str, float]:
    """mean / median / pop-std / min / max for a list of metric values."""
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _classify(mean_m2: float) -> tuple[str, str]:
    """Map mean M2 to (branch_id, interpretation_text). See module docstring."""
    if PREDICTED_BAND[0] <= mean_m2 <= PREDICTED_BAND[1]:
        return (
            "in_band",
            "Prediction confirmed. H2 thresholds (M2 > 0.8) are well-grounded. "
            "Archive this calibration and proceed to Stage 1 (Dockerfile).",
        )
    if 0.5 <= mean_m2 < PREDICTED_BAND[0]:
        return (
            "low_band",
            "Lower than predicted but substantially nonzero. User decides between "
            "(a) keep H2's 0.8 threshold (real videos may fail H2 sometimes), or "
            "(b) revise H2 downward. The choice depends on per-clip variance.",
        )
    if mean_m2 < 0.5:
        return (
            "below_floor",
            "Real surprise. Diagnostics auto-run (DAVIS reverse, identity, random "
            "features). Do NOT modify RESEARCH.md / H2 / M2 here — surface the "
            "full picture; the user calls the next move.",
        )
    return (
        "above_band",
        "Higher than predicted. Likely anti-symmetric gradient structure on real "
        "motion (the rare case in §3.2). Non-blocking; H2 still holds. Note for "
        "the eventual blog post and proceed to Stage 1.",
    )


def _run_main_calibration(
    *, encoder, n_videos: int, seed: int
) -> tuple[list[dict], dict[str, float], dict[str, float]]:
    """Encode N SSv2 clips clean+reverse and return per-clip + summary stats."""
    print("\nLoading SSv2 dataset ...")
    ds = load_ssv2_dataset(n_videos=n_videos, seed=seed)
    print(f"  loaded {len(ds)} clips")

    per_clip: list[dict] = []
    for i, sample in enumerate(ds):
        frames_clean = np.ascontiguousarray(sample.frames)
        # ``[::-1]`` returns a non-contiguous view; ``copy()`` materializes a
        # clean uint8 buffer so ``encode``'s preprocessing stride math is sane.
        frames_reverse = np.ascontiguousarray(frames_clean[::-1])

        feats_clean = encoder.encode(frames_clean)
        feats_reverse = encoder.encode(frames_reverse)
        m1 = float(compute_m1(feats_clean, feats_reverse).item())
        m2 = float(compute_m2(feats_clean, feats_reverse).item())
        print(
            f"  [{i + 1}/{len(ds)}] video_id={sample.video_id}  "
            f"m1={m1:.4f}  m2={m2:.4f}"
        )
        per_clip.append({"video_id": sample.video_id, "m1": m1, "m2": m2})

    m1_stats = _summary([r["m1"] for r in per_clip])
    m2_stats = _summary([r["m2"] for r in per_clip])
    return per_clip, m1_stats, m2_stats


def _run_diagnostics(*, encoder, n_videos: int, seed: int) -> dict:
    """M2<0.5 branch: identity / DAVIS reverse / random-features sanity.

    Each diagnostic is independent. Failures inside one don't kill the
    others — the user wants the full picture.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTICS — branch=below_floor (M2 < 0.5 fired)")
    print("=" * 70)
    out: dict = {}

    # 1) Identity on the same SSv2 clips. M2(F, F) must be ~0.
    print("\n[diagnostic 1/3] Identity check on SSv2 clips")
    ds = load_ssv2_dataset(n_videos=n_videos, seed=seed)
    identity_m2: list[float] = []
    for i, sample in enumerate(ds):
        frames = np.ascontiguousarray(sample.frames)
        f1 = encoder.encode(frames)
        f2 = encoder.encode(frames)
        m2 = float(compute_m2(f1, f2).item())
        identity_m2.append(m2)
        print(f"  [{i + 1}/{len(ds)}] {sample.video_id}: M2(F,F)={m2:.6e}")
    identity_stats = _summary(identity_m2)
    print(
        f"  identity stats: mean={identity_stats['mean']:.6e}  "
        f"max={identity_stats['max']:.6e}  "
        f"(>0 would mean encode() is non-deterministic)"
    )
    out["identity_m2"] = {"per_clip": identity_m2, "stats": identity_stats}

    # 2) DAVIS reverse — different content distribution to SSv2.
    print("\n[diagnostic 2/3] DAVIS reverse playback")
    try:
        from vjepa_stress.data.davis import load_davis_dataset
        davis_ds = load_davis_dataset()
        davis_iter = iter(davis_ds)
        davis_m2: list[float] = []
        for i in range(min(n_videos, len(davis_ds))):
            sample = next(davis_iter)
            frames_clean = np.ascontiguousarray(sample.frames)
            frames_rev = np.ascontiguousarray(frames_clean[::-1])
            f_clean = encoder.encode(frames_clean)
            f_rev = encoder.encode(frames_rev)
            m2 = float(compute_m2(f_clean, f_rev).item())
            davis_m2.append(m2)
            print(f"  [{i + 1}] {sample.sequence_id}: M2={m2:.4f}")
        davis_stats = _summary(davis_m2)
        print(
            f"  DAVIS reverse stats: mean={davis_stats['mean']:.4f}  "
            f"median={davis_stats['median']:.4f}  std={davis_stats['std']:.4f}"
        )
        out["davis_reverse_m2"] = {"per_clip": davis_m2, "stats": davis_stats}
    except Exception as e:  # noqa: BLE001
        print(f"  DAVIS diagnostic failed: {type(e).__name__}: {e}")
        out["davis_reverse_m2"] = {"error": f"{type(e).__name__}: {e}"}

    # 3) Random features sanity at the encoder's output shape.
    # RESEARCH.md §3.2 predicts ~1.14 on truly random gradient directions.
    print(
        f"\n[diagnostic 3/3] Random L2-normalized features sanity at "
        f"shape={RANDOM_FEATURES_SHAPE}"
    )
    g = torch.Generator().manual_seed(0)
    f = torch.randn(*RANDOM_FEATURES_SHAPE, generator=g)
    f = torch.nn.functional.normalize(f, dim=-1)
    f_rev = torch.flip(f, dims=[0])
    rand_m2 = float(compute_m2(f, f_rev).item())
    print(
        f"  random M2(F, reverse(F))={rand_m2:.4f}  "
        f"(RESEARCH §3.2 predicts ~1.14 for genuinely random gradients)"
    )
    out["random_features_m2"] = rand_m2

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Real-feature M2 calibration on SSv2 reverse playback."
    )
    parser.add_argument("--n-videos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="V-JEPA inference device. CPU is fine for 10 clips.",
    )
    parser.add_argument("--variant", default="vjepa-2.1-vit-base")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to JSON output file (parent dirs auto-created).",
    )
    args = parser.parse_args(argv)

    _check_ssv2_present()

    manifest_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"calibrate_m2_real: manifest_id={manifest_id} started_at={started_at}")
    print(f"  variant={args.variant} device={args.device}")
    print(f"  n_videos={args.n_videos} seed={args.seed}")

    print("\nLoading encoder ...")
    encoder = load_encoder(args.variant, device=args.device)
    print(
        f"  embedding_dim={encoder.embedding_dim} "
        f"input_resolution={encoder.input_resolution} "
        f"tubelet_size={encoder.tubelet_size}"
    )

    per_clip, m1_stats, m2_stats = _run_main_calibration(
        encoder=encoder, n_videos=args.n_videos, seed=args.seed
    )

    branch, interpretation = _classify(m2_stats["mean"])
    print()
    print("=" * 70)
    print(
        f"M2(clean, reverse) on real {args.variant} / "
        f"{len(per_clip)} SSv2 validation clips"
    )
    print("=" * 70)
    m2_formatted = [f"{r['m2']:.4f}" for r in per_clip]
    m1_formatted = [f"{r['m1']:.4f}" for r in per_clip]
    print(f"  per-clip M2: {m2_formatted}")
    print(
        f"  M2 stats: mean={m2_stats['mean']:.4f}  median={m2_stats['median']:.4f}  "
        f"std={m2_stats['std']:.4f}  min={m2_stats['min']:.4f}  "
        f"max={m2_stats['max']:.4f}"
    )
    print(f"  per-clip M1: {m1_formatted}")
    print(
        f"  M1 stats: mean={m1_stats['mean']:.4f}  median={m1_stats['median']:.4f}  "
        f"std={m1_stats['std']:.4f}  min={m1_stats['min']:.4f}  "
        f"max={m1_stats['max']:.4f}"
    )
    print(
        f"  predicted band: [{PREDICTED_BAND[0]}, {PREDICTED_BAND[1]}]  "
        f"→ branch={branch}"
    )
    print(f"  interpretation: {interpretation}")
    print("=" * 70)

    diagnostics: dict | None = None
    if branch == "below_floor":
        diagnostics = _run_diagnostics(
            encoder=encoder, n_videos=args.n_videos, seed=args.seed
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_id": manifest_id,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": args.variant,
        "device": args.device,
        "n_clips": len(per_clip),
        "seed": args.seed,
        "predicted_band": list(PREDICTED_BAND),
        "branch": branch,
        "interpretation": interpretation,
        "m2_stats": m2_stats,
        "m1_stats": m1_stats,
        "per_clip": per_clip,
        "diagnostics": diagnostics,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
