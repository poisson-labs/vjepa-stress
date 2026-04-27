"""Isolate ``sys.path`` so ``torch.hub`` can import third-party repos cleanly.

When ``torch.hub.load(...)`` clones a repo, the clone directory is added to
``sys.path``. If that repo has its own top-level package whose name collides
with one in our project (the V-JEPA 2 repo at ``facebookresearch/vjepa2``
ships a ``src/`` package, for example), Python's import system resolves the
name to whichever entry comes first on ``sys.path`` — usually ours, because
the project root is added by ``pip install -e .``. The hub clone then fails
with ``ModuleNotFoundError: No module named 'src.hub'`` (or similar) and the
load aborts.

We discovered this when running the Stage 0.6 V-JEPA 2.1 loading spike:
torch.hub crashed importing ``src.hub`` from Meta's repo because our top-
level ``src/`` shadowed it. The structural fix was to rename our package
from ``src/`` to ``vjepa_stress/`` (commit on 2026-04-27), which removed the
specific collision with V-JEPA 2.

This helper is the belt-and-braces guard for the next time the same class
of bug shows up — any future hub clone whose internal layout collides with
``vjepa_stress`` (or anything else on the project root) will be safe inside
``isolate_for_torch_hub()``. Use it as a context manager around every
``torch.hub.list()`` and ``torch.hub.load()`` call where we want defence in
depth.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path

# Resolve once at import time so the project-root path is stable across calls.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

_PROJECT_PACKAGE = "vjepa_stress"


@contextmanager
def isolate_for_torch_hub() -> Iterator[None]:
    """Temporarily remove the project root from ``sys.path``.

    On entry, removes the project root and the ``""``/``"."`` cwd entries
    from ``sys.path``, and evicts any cached imports of the project package
    so a fresh resolution against the hub clone wins. On exit, restores the
    original entries (idempotent — does not duplicate if the caller has
    already re-added them).
    """
    removed: list[str] = []
    targets = {_PROJECT_ROOT, "", "."}
    for entry in list(sys.path):
        if entry in targets:
            sys.path.remove(entry)
            removed.append(entry)

    evicted: dict[str, object] = {}
    for mod_name in list(sys.modules):
        if mod_name == _PROJECT_PACKAGE or mod_name.startswith(_PROJECT_PACKAGE + "."):
            evicted[mod_name] = sys.modules.pop(mod_name)

    try:
        yield
    finally:
        for entry in removed:
            if entry not in sys.path:
                sys.path.append(entry)
        for mod_name, mod in evicted.items():
            sys.modules.setdefault(mod_name, mod)
