"""Stage 0.6 Task 2 — torch.hub / HF / direct-checkpoint availability spike.

Determines, without committing to multi-GB weight downloads, where (if
anywhere) V-JEPA 2.1 checkpoints can be loaded from. Output is a
table that decides D1 in `docs/STAGE_0.6_TASKS.md`:

    - hub-primary    : `torch.hub.list('facebookresearch/vjepa2')` lists
                       `vjepa2_1_vit_*_384` entries.
    - HF-primary     : `HfApi().list_models(...)` returns a `facebook/vjepa2-1-*`
                       collection.
    - direct-primary : `dl.fbaipublicfiles.com/vjepa2/*.pt` HEAD returns 200 for
                       2.1-named files, neither hub nor HF lists 2.1.
    - none           : no public source found. Stop and ask Taylor.

Usage:
    python scripts/spike_vjepa_loading.py [--full] 2>&1 | tee logs/loading_spike.log

Without --full this runs only Phase A (the four free pre-flight checks
and the optional vit-base download). With --full it also attempts a
forward pass on every variant returned by Phase A; expect ~10 GB of
downloads. The `--full` path is the one Taylor will gate on after
seeing Phase A.

Findings are written to `docs/loading_spike.md`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vjepa_stress.utils.hub_isolation import isolate_for_torch_hub

REPO_OWNER_REPO = "facebookresearch/vjepa2"

# 2.1 entry names we expect to find on torch.hub if 2.1 is integrated.
EXPECTED_21_ENTRIES = (
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384",
)

# 2.0 entry names — if present and 2.1 is absent, we know 2.1 hasn't been
# integrated yet but the broader hub setup works.
EXPECTED_20_ENTRIES = (
    "vjepa2_vit_large_384",
    "vjepa2_vit_huge_384",
    "vjepa2_vit_giant_384",
)

# Plausible HF repo name patterns to probe.
HF_PROBES = (
    {"author": "facebook", "search": "vjepa2-1"},
    {"author": "facebook", "search": "vjepa2.1"},
    {"author": "facebook", "search": "v-jepa-2-1"},
)

# Plausible direct .pt URLs at dl.fbaipublicfiles.com. The V-JEPA 2 README
# confirmed the path pattern dl.fbaipublicfiles.com/vjepa2/<filename>.pt
# (e.g. vitg-384.pt). Probe several plausible 2.1 spellings.
FBAI_BASE = "https://dl.fbaipublicfiles.com/vjepa2/"
FBAI_PROBES = (
    "vjepa2_1_vitbase_384.pt",
    "vjepa2_1_vitlarge_384.pt",
    "vjepa2_1_vitgiant_384.pt",
    "vjepa2_1_vitgigantic_384.pt",
    "vitbase_2_1_384.pt",
    "vitlarge_2_1_384.pt",
    "vitgiant_2_1_384.pt",
    "vitgigantic_2_1_384.pt",
    "v2_1_vitbase_384.pt",
    "v2_1_vitlarge_384.pt",
    # Sanity probes — confirm the bucket itself works:
    "vitg-384.pt",
    "vitl-384.pt",
)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    raw: Any | None = None


def banner(msg: str) -> None:
    bar = "=" * len(msg)
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


# ---------------------------------------------------------------------------
# Phase A — pre-flight (no weight downloads)
# ---------------------------------------------------------------------------


def check_torch_hub_list_21() -> CheckResult:
    banner("A1) torch.hub.list — V-JEPA 2.1 entries")
    try:
        import torch

        with isolate_for_torch_hub():
            entries = torch.hub.list(REPO_OWNER_REPO, trust_repo=True, skip_validation=True)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="torch.hub.list (2.1)",
            ok=False,
            detail=f"{type(e).__name__}: {e}",
            raw=traceback.format_exc(),
        )
    matches = sorted(
        e for e in entries if e in EXPECTED_21_ENTRIES or "_1_" in e
    )
    detail = (
        f"hub returned {len(entries)} entries; "
        f"{len(matches)} match a V-JEPA-2.1 pattern"
    )
    print(f"  all entries ({len(entries)}): {entries}")
    print(f"  2.1 matches: {matches}")
    return CheckResult(
        name="torch.hub.list (2.1)",
        ok=len(matches) > 0,
        detail=detail,
        raw=entries,
    )


def check_torch_hub_list_20() -> CheckResult:
    banner("A2) torch.hub.list — V-JEPA 2.0 entries (contrast)")
    try:
        import torch

        with isolate_for_torch_hub():
            entries = torch.hub.list(REPO_OWNER_REPO, trust_repo=True, skip_validation=True)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="torch.hub.list (2.0)",
            ok=False,
            detail=f"{type(e).__name__}: {e}",
            raw=traceback.format_exc(),
        )
    twentys = sorted(
        e for e in entries if e in EXPECTED_20_ENTRIES or (
            e.startswith("vjepa2_") and "_1_" not in e
        )
    )
    print(f"  2.0 matches: {twentys}")
    return CheckResult(
        name="torch.hub.list (2.0)",
        ok=len(twentys) > 0,
        detail=f"{len(twentys)} V-JEPA-2.0 entries on hub",
        raw=twentys,
    )


def _is_v21_model_id(model_id: str) -> bool:
    """Strict 2.1 marker: vjepa2-1, vjepa-2-1, vjepa2.1, etc.

    Plain ``vjepa2-...`` (V-JEPA 2.0) does *not* count.
    """
    mid = model_id.lower()
    return any(token in mid for token in ("vjepa2-1", "vjepa2.1", "vjepa-2-1", "vjepa-2.1"))


def check_hf_list_models() -> CheckResult:
    banner("A3) huggingface_hub.HfApi().list_models — strict 2.1 filter")
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        all_ids: set[str] = set()
        rows: list[str] = []
        # Pull every facebook/vjepa* model the API knows about.
        try:
            models = list(api.list_models(author="facebook", search="vjepa", limit=100))
            for m in models:
                mid = m.modelId if hasattr(m, "modelId") else m.id
                all_ids.add(mid)
        except Exception as e:  # noqa: BLE001
            rows.append(f"  list_models(facebook,vjepa) -> {type(e).__name__}: {e}")

        v21_ids = sorted(mid for mid in all_ids if _is_v21_model_id(mid))
        v2_only_ids = sorted(mid for mid in all_ids if not _is_v21_model_id(mid))

        print(f"  total facebook/vjepa* models: {len(all_ids)}")
        print(f"  V-JEPA 2.1 (strict): {v21_ids}")
        print(f"  V-JEPA 2.0 (contrast): {v2_only_ids[:10]}{' ...' if len(v2_only_ids) > 10 else ''}")

        return CheckResult(
            name="HF list_models",
            ok=len(v21_ids) > 0,
            detail=(
                f"{len(v21_ids)} V-JEPA-2.1 models on HF"
                + (f"; {len(v2_only_ids)} V-JEPA-2.0 models present" if v2_only_ids else "")
            ),
            raw={"v21": v21_ids, "v2": v2_only_ids[:20]},
        )
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name="HF list_models",
            ok=False,
            detail=f"{type(e).__name__}: {e}",
            raw=traceback.format_exc(),
        )


def head_request(url: str, timeout: float = 10.0) -> tuple[int, str]:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            length = r.headers.get("Content-Length", "?")
            return r.status, f"OK Content-Length={length}"
    except urllib.error.HTTPError as e:
        return e.code, f"HTTPError {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return -1, f"URLError {e.reason}"
    except Exception as e:  # noqa: BLE001
        return -2, f"{type(e).__name__}: {e}"


def check_fbai_direct() -> CheckResult:
    banner("A4) dl.fbaipublicfiles.com HEAD probes for V-JEPA 2.1 .pt files")
    rows: list[tuple[str, int, str]] = []
    for fname in FBAI_PROBES:
        url = FBAI_BASE + fname
        status, detail = head_request(url)
        rows.append((fname, status, detail))
        marker = "OK" if status == 200 else "  "
        print(f"  [{marker}] {status:>4}  {fname} -> {detail}")
    found_21 = any(
        status == 200 and ("vjepa2_1" in fname or "2_1" in fname)
        for (fname, status, _) in rows
    )
    sanity_ok = any(
        status == 200 and "2_1" not in fname
        for (fname, status, _) in rows
    )
    detail = []
    if found_21:
        detail.append("found at least one V-JEPA-2.1 checkpoint")
    if sanity_ok:
        detail.append("bucket is reachable (V-JEPA-2 sanity probe OK)")
    if not detail:
        detail.append("no V-JEPA-2.1 checkpoints; bucket may also be unreachable")
    return CheckResult(
        name="dl.fbaipublicfiles direct",
        ok=found_21,
        detail="; ".join(detail),
        raw=rows,
    )


# ---------------------------------------------------------------------------
# Phase B — confirmation download (gated, opt-in)
# ---------------------------------------------------------------------------


def maybe_load_via_hub(entry: str) -> CheckResult:
    banner(f"B) torch.hub.load {entry} (downloads weights)")
    import torch

    t0 = time.time()
    try:
        with isolate_for_torch_hub():
            model = torch.hub.load(REPO_OWNER_REPO, entry, trust_repo=True)
    except Exception as e:  # noqa: BLE001
        return CheckResult(
            name=f"hub.load {entry}",
            ok=False,
            detail=f"{type(e).__name__}: {e}",
            raw=traceback.format_exc(),
        )
    dt = time.time() - t0
    print(f"  loaded in {dt:.1f}s, type={type(model).__name__}")
    return CheckResult(
        name=f"hub.load {entry}",
        ok=True,
        detail=f"loaded in {dt:.1f}s, type={type(model).__name__}",
        raw=None,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def render_loading_spike_md(results: list[CheckResult]) -> str:
    lines = [
        "# V-JEPA 2.1 loading spike",
        "",
        "Output of `python scripts/spike_vjepa_loading.py`. Records what was",
        "available at the time of the spike; re-run to refresh.",
        "",
        "## Result summary",
        "",
        "| Check | Outcome | Detail |",
        "|-------|---------|--------|",
    ]
    for r in results:
        outcome = "OK" if r.ok else "fail"
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| {r.name} | {outcome} | {detail} |")
    lines.append("")
    lines.append("## Raw output")
    for r in results:
        lines.append("")
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append("```")
        if isinstance(r.raw, str):
            lines.append(r.raw)
        elif isinstance(r.raw, list):
            for item in r.raw:
                lines.append(repr(item))
        elif r.raw is None:
            lines.append("(no raw output)")
        else:
            lines.append(json.dumps(r.raw, default=str, indent=2))
        lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run Phase B (downloads vit-base ~300 MB; gate this on Phase A).",
    )
    parser.add_argument(
        "--phase-b-entry",
        default="vjepa2_1_vit_base_384",
        help="torch.hub entry to test in Phase B (only with --full).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/loading_spike.md"),
        help="Where to write the findings markdown.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results: list[CheckResult] = []
    results.append(check_torch_hub_list_21())
    results.append(check_torch_hub_list_20())
    results.append(check_hf_list_models())
    results.append(check_fbai_direct())

    if args.full:
        results.append(maybe_load_via_hub(args.phase_b_entry))

    # Summary table
    banner("Summary")
    for r in results:
        marker = "OK" if r.ok else "FAIL"
        print(f"  [{marker:4}] {r.name}: {r.detail}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_loading_spike_md(results))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
