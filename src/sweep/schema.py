"""Row schema lock for sweep shards.

The sweep writes one Parquet shard per cell. Every row in every shard
conforms to the schema defined here. Code that produces rows must build
them through ``Row``; code that reads shards relies on ``ARROW_SCHEMA``.

Both directions go through the same source of truth so a drifted writer
and a drifted reader can never agree silently. Schema changes bump
``SCHEMA_VERSION``; the manifest records the version each shard was
written with so readers can refuse to load mixed-version sweeps.
"""

from __future__ import annotations

import math
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Bumped whenever the row schema changes in a way that breaks readers.
SCHEMA_VERSION: str = "1.0"


class Row(BaseModel):
    """One (cell, video) row.

    The metric fields use ``float`` rather than ``Optional[float]`` because
    a missing metric is encoded as NaN, not as null. NaN survives Parquet
    round-trips and DuckDB aggregates ignore it; null would force every
    consumer to write a coalesce.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(default=SCHEMA_VERSION)
    cell_id: str
    tier: int = Field(ge=1, le=3)
    model: str
    perturbation: str
    strength: float = Field(ge=0.0, le=1.0)
    video_id: str
    seed: int
    m1: float
    m2: float
    m3: float
    notes: str = ""

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"Row schema_version {v!r} does not match current SCHEMA_VERSION {SCHEMA_VERSION!r}"
            )
        return v


#: PyArrow schema derived from ``Row``. Field order is the canonical column
#: order in the written Parquet shards.
ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("tier", pa.int32(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("perturbation", pa.string(), nullable=False),
        pa.field("strength", pa.float64(), nullable=False),
        pa.field("video_id", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("m1", pa.float64(), nullable=False),
        pa.field("m2", pa.float64(), nullable=False),
        pa.field("m3", pa.float64(), nullable=False),
        pa.field("notes", pa.string(), nullable=False),
    ]
)


COLUMNS: tuple[str, ...] = tuple(f.name for f in ARROW_SCHEMA)


def make_arrow_schema() -> pa.Schema:
    """Return the canonical Arrow schema (returns the module-level constant)."""
    return ARROW_SCHEMA


def validate_row(data: dict[str, Any]) -> Row:
    """Validate one row dict against the schema lock.

    Re-raises ``pydantic.ValidationError`` on failure so callers can decide
    whether to abort the sweep or skip the offending cell.
    """
    return Row(**data)


def rows_to_arrow_table(rows: list[Row]) -> pa.Table:
    """Convert a list of validated rows to a PyArrow table with the locked schema."""
    if not rows:
        empty: dict[str, list[Any]] = {name: [] for name in COLUMNS}
        return pa.Table.from_pydict(empty, schema=ARROW_SCHEMA)

    columns: dict[str, list[Any]] = {name: [] for name in COLUMNS}
    for r in rows:
        d = r.model_dump()
        for name in COLUMNS:
            columns[name].append(d[name])
    return pa.Table.from_pydict(columns, schema=ARROW_SCHEMA)


def is_nan(x: float) -> bool:
    """Return True iff ``x`` is NaN. Useful for callers checking metric presence."""
    return isinstance(x, float) and math.isnan(x)
