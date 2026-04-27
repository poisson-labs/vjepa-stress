"""Per-cell checkpointing to Parquet.

Each completed cell writes one Parquet shard at
``{output_dir}/{cell_id}.parquet`` containing one row per video. Shards
are written atomically: data goes to ``{cell_id}.parquet.tmp``, fsync'd,
and renamed into place. A crash mid-write therefore leaves no
half-written shard that ``cell_completed`` would mistake for done.

The shard schema is locked by ``src.sweep.schema.ARROW_SCHEMA``. Reads
go through ``load_shards`` which returns a DuckDB relation so callers
can run analytical queries without materializing the full table.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.sweep.matrix import Cell
from src.sweep.schema import ARROW_SCHEMA, Row, rows_to_arrow_table


def shard_path(output_dir: Path, cell: Cell) -> Path:
    """Resolve the canonical Parquet path for one cell."""
    return Path(output_dir) / f"{cell.cell_id}.parquet"


def _tmp_path(final: Path) -> Path:
    return final.with_suffix(final.suffix + ".tmp")


def cell_completed(output_dir: Path, cell: Cell) -> bool:
    """True iff a usable shard already exists for this cell.

    A shard counts as complete only when:

    - the file exists,
    - it has non-zero size,
    - it parses as a Parquet file with the expected schema,
    - it contains at least one row.

    Anything weaker would let a crash that happened mid-rename be
    misread as completion. The runner uses this to skip already-done
    cells on resume.
    """
    path = shard_path(output_dir, cell)
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        meta = pq.read_metadata(path)
    except (OSError, pa.ArrowInvalid):
        return False
    if meta.num_rows < 1:
        return False
    schema = pq.read_schema(path)
    if schema.names != ARROW_SCHEMA.names:
        return False
    return True


def write_shard(output_dir: Path, cell: Cell, rows: list[Row]) -> Path:
    """Atomically write one cell's rows to ``{cell_id}.parquet``.

    Refuses to write an empty shard: an empty cell is a runner bug, and
    a zero-row Parquet file would defeat the ``cell_completed`` guard.
    """
    if not rows:
        raise ValueError(f"refusing to write empty shard for cell {cell.cell_id}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final = shard_path(output_dir, cell)
    tmp = _tmp_path(final)

    table = rows_to_arrow_table(rows)
    if table.schema != ARROW_SCHEMA:
        raise ValueError(
            f"row table schema does not match ARROW_SCHEMA for cell {cell.cell_id}"
        )

    # Stale tmp from a previous crash would shadow ours; remove first.
    if tmp.exists():
        tmp.unlink()

    with pq.ParquetWriter(tmp, ARROW_SCHEMA) as writer:
        writer.write_table(table)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

    os.rename(tmp, final)
    return final


def load_shards(output_dir: Path) -> "duckdb.DuckDBPyRelation":
    """Return a DuckDB relation over every shard in ``output_dir``.

    DuckDB scans Parquet directly without copying into memory; callers
    can chain ``.df()`` or further SQL on the returned relation.
    """
    output_dir = Path(output_dir)
    pattern = str(output_dir / "*.parquet")
    con = duckdb.connect()
    return con.from_query(f"SELECT * FROM read_parquet('{pattern}')")


def cleanup_stale_tmp(output_dir: Path) -> int:
    """Remove any ``*.parquet.tmp`` left behind by a crashed writer.

    Returns the number of files removed. Safe to call before a run; an
    in-flight writer never has a competing tmp because the writer
    deletes a stale tmp before opening its own.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    n = 0
    for p in output_dir.glob("*.parquet.tmp"):
        p.unlink()
        n += 1
    return n
