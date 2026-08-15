"""Tests for ArcGIS REST client."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.arcgis_client import ArcGISRestClient
from src.errors import DataValidationError, ExternalServiceError

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_query_all_paginates():
    """Test that query_all handles pagination."""
    client = ArcGISRestClient(
        "https://example.com/arcgis/rest/services/test/MapServer/0",
        mode="live",
    )

    # Mock responses
    responses = [
        {"features": [{"attributes": {"id": 1}}], "exceededTransferLimit": True},
        {"features": [{"attributes": {"id": 2}}], "exceededTransferLimit": False},
    ]

    with patch.object(client, "query", side_effect=responses):
        result = client.query_all(result_record_count=1)

    assert len(result["features"]) == 2
    assert result["metadata"]["count"] == 2
    assert result["metadata"]["exceededTransferLimit"] is True


def test_query_all_stops_when_no_more_records():
    """Test that query_all stops when fewer records are returned."""
    client = ArcGISRestClient(
        "https://example.com/arcgis/rest/services/test/MapServer/0",
        mode="live",
    )

    with patch.object(client, "query", return_value={"features": [{"attributes": {"id": 1}}]}):
        result = client.query_all(result_record_count=1000)

    assert len(result["features"]) == 1
    assert result["metadata"]["count"] == 1


def test_query_mock_mode_loads_fixture():
    """Test mock mode reads from local JSON fixture."""
    fixture_path = FIXTURES_DIR / "arcgis_page_1.json"
    client = ArcGISRestClient(
        "https://example.com/arcgis/rest/services/test/MapServer/0",
        mode="mock",
        fixture_path=fixture_path,
    )

    res = client.query()
    assert len(res["features"]) == 2
    assert res["features"][0]["attributes"]["objectid"] == 1001


def test_query_mock_empty_fixture():
    """Test mock mode with empty fixture."""
    fixture_path = FIXTURES_DIR / "arcgis_empty.json"
    client = ArcGISRestClient(
        "https://example.com/arcgis/rest/services/test/MapServer/0",
        mode="mock",
        fixture_path=fixture_path,
    )

    res = client.query()
    assert res["features"] == []


def test_get_metadata_mock():
    """Test metadata retrieval in mock mode."""
    client = ArcGISRestClient("https://example.com/arcgis/test", mode="mock")
    metadata = client.get_metadata()
    assert metadata["objectIdField"] == "objectid"
    assert client.get_object_id_field() == "objectid"


def test_query_hybrid_fallback_on_network_error():
    """Test hybrid mode falls back to fixture on HTTP error."""
    client = ArcGISRestClient(
        "https://example.com/arcgis/test",
        mode="hybrid",
        fixture_path=FIXTURES_DIR / "arcgis_311_sample.json",
    )

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.ConnectionError("Network unreachable")
    with patch.object(client.session, "get", return_value=mock_resp):
        res = client.query()

    assert len(res["features"]) == 4


def test_query_rejects_arcgis_error_envelope():
    """HTTP 200 responses containing an ArcGIS error are still failures."""
    client = ArcGISRestClient("https://example.com/arcgis/test", mode="live")
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": {"code": 400, "message": "bad"}}

    with patch.object(client.session, "get", return_value=mock_response):
        with pytest.raises(ExternalServiceError, match="code 400"):
            client.query()


def test_query_rejects_malformed_feature_payload():
    """A missing feature list cannot be mistaken for an empty page."""
    client = ArcGISRestClient("https://example.com/arcgis/test", mode="live")
    mock_response = MagicMock()
    mock_response.json.return_value = {"not_features": []}

    with patch.object(client.session, "get", return_value=mock_response):
        with pytest.raises(DataValidationError, match="features list"):
            client.query()


def test_query_all_rejects_repeated_page():
    """A server that ignores offsets must not create an infinite loop."""
    client = ArcGISRestClient(
        "https://example.com/arcgis/test",
        mode="live",
        max_pages=3,
    )
    repeated = {
        "features": [{"attributes": {"id": 1}}],
        "exceededTransferLimit": True,
    }

    with patch.object(client, "query", side_effect=[repeated, repeated]):
        with pytest.raises(DataValidationError, match="repeated"):
            client.query_all(result_record_count=1)


def test_live_client_requires_https():
    """Live source traffic cannot be downgraded to plaintext HTTP."""
    with pytest.raises(ValueError, match="HTTPS"):
        ArcGISRestClient("http://example.com/arcgis/test", mode="live")
