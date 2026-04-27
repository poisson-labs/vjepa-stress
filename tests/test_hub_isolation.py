"""Unit tests for ``vjepa_stress.utils.hub_isolation``.

The helper is the structural fix for the ``src/`` shadowing bug we hit
during the Stage 0.6 V-JEPA 2.1 loading spike. These tests cover the
two invariants we care about — sys.path is restored on exit, and
project-package imports are evicted (and restored) — without invoking
torch.hub itself.
"""

from __future__ import annotations

import sys

from vjepa_stress.utils.hub_isolation import isolate_for_torch_hub


def _project_root() -> str:
    from vjepa_stress.utils import hub_isolation

    return hub_isolation._PROJECT_ROOT  # noqa: SLF001 — test of internal


def test_project_root_removed_inside_context_and_restored_after():
    root = _project_root()
    sys.path.insert(0, root)  # simulate the editable-install entry

    with isolate_for_torch_hub():
        assert root not in sys.path

    assert root in sys.path
    sys.path.remove(root)


def test_project_package_evicted_inside_and_restored_after():
    import vjepa_stress.sweep.matrix  # noqa: F401 — populates sys.modules

    assert "vjepa_stress" in sys.modules
    assert "vjepa_stress.sweep.matrix" in sys.modules

    with isolate_for_torch_hub():
        assert "vjepa_stress" not in sys.modules
        assert "vjepa_stress.sweep.matrix" not in sys.modules

    assert "vjepa_stress" in sys.modules
    assert "vjepa_stress.sweep.matrix" in sys.modules


def test_idempotent_when_project_root_already_absent():
    root = _project_root()
    saved = [p for p in sys.path if p == root]
    for _ in saved:
        sys.path.remove(root)

    try:
        with isolate_for_torch_hub():
            assert root not in sys.path
        assert root not in sys.path  # nothing to restore — must not duplicate
    finally:
        for p in saved:
            sys.path.append(p)


def test_restores_after_exception():
    root = _project_root()
    sys.path.insert(0, root)

    class Boom(RuntimeError):
        pass

    try:
        with isolate_for_torch_hub():
            raise Boom("simulated hub failure")
    except Boom:
        pass

    assert root in sys.path
    sys.path.remove(root)
