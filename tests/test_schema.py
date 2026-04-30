"""Schema lock tests: Row <-> Arrow <-> Parquet <-> DuckDB round-trip."""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from vjepa_stress.sweep.schema import (
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
        "dataset": "ssv2",
        "model": "vjepa-2.1-vit-base",
        "perturbation": "gaussian_noise",
        "strength": 0.1,
        "video_id": "fake_0001",
        "seed": 12345,
        "m1": 0.42,
        "m2": 0.17,
        "m3_clean": float("nan"),
        "m3_perturbed": float("nan"),
        "wallclock_sec": 1.234,
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
    for col in ("m3_clean", "m3_perturbed"):
        values = read_back.column(col).to_pylist()
        assert len(values) == 1
        assert math.isnan(values[0])


def test_full_round_trip_through_duckdb(tmp_path: Path):
    payload = [
        Row(
            **_example_row(
                m1=0.0, m2=0.0, m3_clean=float("nan"), m3_perturbed=float("nan")
            )
        ),
        Row(
            **_example_row(
                cell_id="t1__davis__vjepa-2.1-vit-base__gaussian_noise__s0.5000",
                dataset="davis",
                strength=0.5,
                m1=0.3,
                m2=0.2,
                m3_clean=0.95,
                m3_perturbed=0.71,
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


def test_wallclock_sec_round_trips_through_parquet(tmp_path: Path):
    """Schema v1.1: wallclock_sec is a required, non-null float64 column."""
    rows = [
        Row(**_example_row(wallclock_sec=0.123)),
        Row(**_example_row(video_id="fake_0002", wallclock_sec=4.567)),
    ]
    table = rows_to_arrow_table(rows)
    out = tmp_path / "wallclock.parquet"
    pq.write_table(table, out)

    read_back = pq.read_table(out)
    assert "wallclock_sec" in read_back.schema.names
    field = read_back.schema.field("wallclock_sec")
    assert str(field.type) == "double"
    assert not field.nullable
    values = read_back.column("wallclock_sec").to_pylist()
    assert values == [0.123, 4.567]


def test_wallclock_sec_negative_rejected():
    """Field(ge=0.0) — negative wallclock is a logic bug, not a measurement."""
    with pytest.raises(ValidationError):
        validate_row(_example_row(wallclock_sec=-0.001))


def test_wallclock_sec_required():
    """Adding wallclock_sec is a v1.1 break: writers must populate it."""
    bad = _example_row()
    del bad["wallclock_sec"]
    with pytest.raises(ValidationError):
        validate_row(bad)


def test_schema_version_is_one_two():
    """Tripwire: forces a deliberate touch of this test on the next bump."""
    assert SCHEMA_VERSION == "1.2"


def test_dataset_axis_required_and_validated():
    """Schema v1.2: dataset is required and must be 'ssv2' or 'davis'."""
    bad = _example_row()
    del bad["dataset"]
    with pytest.raises(ValidationError):
        validate_row(bad)
    with pytest.raises(ValidationError):
        validate_row(_example_row(dataset="kinetics"))


def test_m3_split_columns_required():
    """v1.2 break: writers must populate m3_clean + m3_perturbed (legacy m3 dropped)."""
    bad = _example_row()
    del bad["m3_clean"]
    with pytest.raises(ValidationError):
        validate_row(bad)
    bad2 = _example_row()
    del bad2["m3_perturbed"]
    with pytest.raises(ValidationError):
        validate_row(bad2)
    # Legacy ``m3`` is now an unexpected extra under extra="forbid".
    with pytest.raises(ValidationError):
        validate_row(_example_row(m3=0.5))


def test_schema_version_field_must_match_constant():
    bad = _example_row(schema_version="0.0")
    with pytest.raises(ValidationError):
        validate_row(bad)


def test_rows_to_arrow_table_empty_returns_locked_schema():
    table = rows_to_arrow_table([])
    assert table.schema == ARROW_SCHEMA
    assert table.num_rows == 0
