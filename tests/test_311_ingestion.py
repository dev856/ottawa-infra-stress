"""Tests for privacy-safe 311 event ingestion."""

from pathlib import Path
from unittest.mock import MagicMock

import h3
import pandas as pd
import pytest

from src.arcgis_client import ArcGISRestClient
from src.errors import DataValidationError
from src.ingest_311_service_requests import (
    PERSISTED_FIELDS,
    build_date_filter,
    fetch_311_service_requests,
    is_water_break_record,
    minimize_features,
    upsert_events,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_water_break_matching_uses_controlled_request_type() -> None:
    """Explicit break types match regardless of casing or whitespace."""
    assert is_water_break_record({"reqtype": " water   main break "}) is True


def test_broad_water_category_does_not_match() -> None:
    """The broad WATER category cannot label every water-related request a break."""
    record = {"reqtype": "WATER SERVICE", "category": "WATER"}
    assert is_water_break_record(record) is False


def test_free_text_is_not_used_for_classification() -> None:
    """Potentially identifying descriptions are not collected or inspected."""
    record = {"reqtype": "OTHER", "description": "water main break at my home"}
    assert is_water_break_record(record) is False


def test_build_date_filter_rejects_injection_text() -> None:
    """Only strict ISO dates can enter an ArcGIS SQL-like expression."""
    with pytest.raises(DataValidationError, match="YYYY-MM-DD"):
        build_date_filter("2024-01-01' OR 1=1 --", None)


def test_minimize_features_removes_exact_and_free_text_fields() -> None:
    """Persisted events contain only the reviewed field allowlist plus H3."""
    features = [
        {
            "attributes": {
                "objectid": 1,
                "reqtype": "WATER MAIN BREAK",
                "category": "WATER",
                "createddate": 1688169600000,
                "description": "person and exact address",
                "email": "private@example.com",
            },
            "geometry": {"x": -75.6972, "y": 45.4215},
        }
    ]

    result = minimize_features(features)

    assert tuple(result.columns) == PERSISTED_FIELDS
    assert "description" not in result.columns
    assert "email" not in result.columns
    assert "longitude" not in result.columns
    assert "latitude" not in result.columns
    assert h3.is_valid_cell(result.loc[0, "h3_index"])


def test_minimize_features_rejects_invalid_geometry() -> None:
    """Missing and out-of-city points are skipped rather than persisted."""
    features = [
        {
            "attributes": {
                "objectid": 1,
                "reqtype": "WATER MAIN BREAK",
                "createddate": 1688169600000,
            },
            "geometry": {"x": 0, "y": 0},
        },
        {
            "attributes": {
                "objectid": 2,
                "reqtype": "WATER MAIN BREAK",
                "createddate": 1688169600000,
            }
        },
    ]

    assert minimize_features(features).empty


def test_fetch_mock_returns_only_explicit_breaks() -> None:
    """The fixture produces minimized H3 rows, not source records."""
    client = ArcGISRestClient(
        "https://example.com/arcgis/311",
        mode="mock",
        fixture_path=FIXTURES_DIR / "arcgis_311_sample.json",
    )

    result = fetch_311_service_requests(client=client)

    assert len(result) == 2
    assert set(result["objectid"]) == {2001, 2004}
    assert tuple(result.columns) == PERSISTED_FIELDS


def test_duplicate_object_ids_are_idempotent() -> None:
    """Duplicate source rows collapse to one stable event."""
    feature = {
        "attributes": {
            "objectid": 1,
            "reqtype": "WATER MAIN BREAK",
            "category": "WATER",
            "createddate": 1688169600000,
        },
        "geometry": {"x": -75.6972, "y": 45.4215},
    }

    result = minimize_features([feature, feature])

    assert len(result) == 1


def test_upsert_uses_bound_parameters() -> None:
    """Database values are supplied separately from the SQL statement."""
    events = pd.DataFrame(
        [
            {
                "objectid": 1,
                "reqtype": "WATER MAIN BREAK",
                "category": "WATER",
                "createddate": pd.Timestamp("2023-07-01", tz="UTC"),
                "h3_index": "881f1d4887fffff",
            }
        ]
    )
    connection = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = connection
    engine = MagicMock()
    engine.begin.return_value = context

    count = upsert_events(engine, events)

    assert count == 1
    statement, parameters = connection.execute.call_args.args
    assert ":objectid" in str(statement)
    assert parameters[0]["objectid"] == 1
