"""Schema lock tests: Row <-> Arrow <-> Parquet <-> DuckDB round-trip."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from src.sweep.schema import (
    ARROW_SCHEMA,
    COLUMNS,
    SCHEMA_VERSION,
    Row,
    rows_to_arrow_table,
    validate_row,
)


def _example_row(**overrides) -> dict:
    base = {
        "cell_id": "t1__vjepa-2.1-vit-base__gaussian_noise__s0.1000",
        "tier": 1,
        "model": "vjepa-2.1-vit-base",
        "perturbation": "gaussian_noise",
        "strength": 0.1,
        "video_id": "fake_0001",
        "seed": 12345,
        "m1": 0.42,
        "m2": 0.17,
        "m3": float("nan"),
        "notes": "",
    }
    base.update(overrides)
    return base


def test_schema_version_exported_and_nonempty():
    assert isinstance(SCHEMA_VERSION, str)
    assert SCHEMA_VERSION


def test_columns_match_arrow_schema():
    assert COLUMNS == tuple(ARROW_SCHEMA.names)


def test_validate_row_accepts_well_formed_dict():
    row = validate_row(_example_row())
    assert isinstance(row, Row)
    assert row.cell_id.startswith("t1__")


def test_missing_required_field_raises():
    bad = _example_row()
    del bad["m1"]
    with pytest.raises(ValidationError):
        validate_row(bad)


def test_wrong_dtype_raises():
    bad = _example_row(tier="one")  # tier must be int
    with pytest.raises(ValidationError):
        validate_row(bad)


def test_strength_out_of_range_raises():
    with pytest.raises(ValidationError):
        validate_row(_example_row(strength=1.5))
    with pytest.raises(ValidationError):
        validate_row(_example_row(strength=-0.1))


def test_extra_field_raises():
    bad = _example_row(extra_unexpected_field=1)
    with pytest.raises(ValidationError):
        validate_row(bad)


def test_nan_m3_round_trips_through_parquet(tmp_path: Path):
    rows = [Row(**_example_row())]
    table = rows_to_arrow_table(rows)
    out = tmp_path / "single.parquet"
    pq.write_table(table, out)

    read_back = pq.read_table(out)
    assert read_back.schema == ARROW_SCHEMA
    m3_col = read_back.column("m3").to_pylist()
    assert len(m3_col) == 1
    assert math.isnan(m3_col[0])


def test_full_round_trip_through_duckdb(tmp_path: Path):
    payload = [
        Row(**_example_row(m1=0.0, m2=0.0, m3=float("nan"))),
        Row(
            **_example_row(
                cell_id="t1__vjepa-2.1-vit-base__gaussian_noise__s0.5000",
                strength=0.5,
                m1=0.3,
                m2=0.2,
                m3=0.95,
            )
        ),
    ]
    table = rows_to_arrow_table(payload)
    shard = tmp_path / "round_trip.parquet"
    pq.write_table(table, shard)

    con = duckdb.connect()
    rel = con.from_query(f"SELECT * FROM read_parquet('{shard}')")
    df = rel.df()

    assert list(df.columns) == list(COLUMNS)
    assert len(df) == 2
    assert df["schema_version"].iloc[0] == SCHEMA_VERSION
    assert df["seed"].iloc[0] == 12345


def test_schema_version_field_must_match_constant():
    bad = _example_row(schema_version="0.0")
    with pytest.raises(ValidationError):
        validate_row(bad)


def test_rows_to_arrow_table_empty_returns_locked_schema():
    table = rows_to_arrow_table([])
    assert table.schema == ARROW_SCHEMA
    assert table.num_rows == 0
