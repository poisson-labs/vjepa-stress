"""Sweep manifest: the reproducibility record for one run.

The manifest is written **before** any cell executes and lives at
``{output_dir}/manifest.json``. It records:

- A unique ``manifest_id`` (UUID4) identifying this sweep run.
- The ``schema_version`` the shards conform to.
- Library versions and the source commit hash.
- The exact list of (video_id, perturbation_name) seeds used. Seeds are
  precomputed up-front from a master seed so a partial run can be
  resumed with bit-identical perturbation noise.

Resuming a crashed run reads the existing manifest and reuses its seeds.
The runner refuses to resume against a manifest with a different
``schema_version`` than the current code expects.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from src.sweep.schema import SCHEMA_VERSION

#: Libraries whose version is recorded in every manifest. Missing libraries
#: are recorded as ``"missing"`` rather than failing the run.
_TRACKED_LIBRARIES: tuple[str, ...] = (
    "torch",
    "transformers",
    "numpy",
    "opencv-python-headless",
    "pyarrow",
    "duckdb",
    "pydantic",
    "pandas",
)


@dataclass
class Manifest:
    """In-memory representation of ``manifest.json``."""

    manifest_id: str
    schema_version: str
    code_commit: str
    library_versions: dict[str, str]
    python_version: str
    platform: str
    matrix_fingerprint: str
    n_videos: int
    seeds: dict[str, dict[str, int]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "code_commit": self.code_commit,
            "library_versions": self.library_versions,
            "python_version": self.python_version,
            "platform": self.platform,
            "matrix_fingerprint": self.matrix_fingerprint,
            "n_videos": self.n_videos,
            "seeds": self.seeds,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        return cls(
            manifest_id=data["manifest_id"],
            schema_version=data["schema_version"],
            code_commit=data["code_commit"],
            library_versions=dict(data.get("library_versions", {})),
            python_version=data["python_version"],
            platform=data["platform"],
            matrix_fingerprint=data["matrix_fingerprint"],
            n_videos=data["n_videos"],
            seeds={
                vid: dict(per_pert) for vid, per_pert in data.get("seeds", {}).items()
            },
            extra=dict(data.get("extra", {})),
        )


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _library_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def _matrix_fingerprint(
    video_ids: list[str], perturbation_names: list[str]
) -> str:
    """Stable hash over the (video_ids, perturbations) the manifest covers."""
    h = hashlib.blake2b(digest_size=16)
    for v in video_ids:
        h.update(v.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"||")
    for p in perturbation_names:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def precompute_seeds(
    video_ids: list[str],
    perturbation_names: list[str],
    master_seed: int,
) -> dict[str, dict[str, int]]:
    """Return ``{video_id: {perturbation_name: int_seed}}``.

    Seeds are derived from a deterministic blake2b of ``(master_seed,
    video_id, perturbation_name)``. Same inputs always yield the same
    seeds; this is what makes a resumed run bit-identical to a fresh run.
    """
    seeds: dict[str, dict[str, int]] = {}
    for vid in video_ids:
        per_pert: dict[str, int] = {}
        for pname in perturbation_names:
            h = hashlib.blake2b(digest_size=8)
            h.update(master_seed.to_bytes(8, "little", signed=False))
            h.update(b"|")
            h.update(vid.encode("utf-8"))
            h.update(b"|")
            h.update(pname.encode("utf-8"))
            per_pert[pname] = int.from_bytes(h.digest(), "little", signed=False) % (
                2**31
            )
        seeds[vid] = per_pert
    return seeds


def build_manifest(
    *,
    video_ids: list[str],
    perturbation_names: list[str],
    master_seed: int,
    repo_root: Path,
    extra: dict[str, Any] | None = None,
) -> Manifest:
    """Construct a fresh ``Manifest`` for this run."""
    return Manifest(
        manifest_id=str(uuid.uuid4()),
        schema_version=SCHEMA_VERSION,
        code_commit=_git_commit(repo_root),
        library_versions=_library_versions(),
        python_version=sys.version.split()[0],
        platform=f"{platform.system()}-{platform.release()}-{platform.machine()}",
        matrix_fingerprint=_matrix_fingerprint(video_ids, perturbation_names),
        n_videos=len(video_ids),
        seeds=precompute_seeds(video_ids, perturbation_names, master_seed),
        extra=extra or {},
    )


def manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / "manifest.json"


def write_manifest(output_dir: Path, manifest: Manifest) -> Path:
    """Write the manifest atomically (.tmp + fsync + rename)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = manifest_path(output_dir)
    tmp = final.with_suffix(final.suffix + ".tmp")
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    with open(tmp, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, final)
    return final


def read_manifest(output_dir: Path) -> Manifest:
    """Read the existing manifest from ``output_dir``."""
    with open(manifest_path(output_dir)) as f:
        data = json.load(f)
    return Manifest.from_dict(data)


def manifest_exists(output_dir: Path) -> bool:
    return manifest_path(output_dir).exists()


def lookup_seed(manifest: Manifest, video_id: str, perturbation_name: str) -> int:
    """Return the precomputed seed for one (video, perturbation) pair."""
    try:
        return manifest.seeds[video_id][perturbation_name]
    except KeyError as e:
        raise KeyError(
            f"No seed for ({video_id!r}, {perturbation_name!r}) in manifest "
            f"{manifest.manifest_id}"
        ) from e
