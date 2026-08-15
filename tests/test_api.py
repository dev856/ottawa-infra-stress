"""API contract and security tests using aggregate-only in-memory services."""

from typing import Any

import h3
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import (
    app,
    get_predictor,
    get_repository,
    get_risk_level,
    get_weather_features,
)
from api.services import ModelBundle, RiskMapRows, RiskPredictor
from src.errors import ArtifactError

VALID_CELL = h3.latlng_to_cell(45.4215, -75.6972, 8)
MISSING_CELL = next(cell for cell in h3.grid_ring(VALID_CELL, 1) if cell != VALID_CELL)


class FakeClassifier:
    """Small deterministic classifier matching the joblib bundle interface."""

    def predict_proba(self, dataframe: Any) -> np.ndarray:
        """Return a stable high-risk probability for every aggregate row."""
        return np.array([[0.2, 0.8] for _ in range(len(dataframe))])


class FakeRepository:
    """Return one reviewed aggregate row and no raw service-request data."""

    aggregate = {
        "centroid_lat": 45.4215,
        "centroid_lon": -75.6972,
        "line_length_km_water": 1.2,
        "line_count_water": 2,
        "line_length_km_road": 1.8,
        "line_count_road": 3,
        "building_count": 10,
        "median_year_built": 1975.0,
        "pct_pre_1980": 0.6,
    }

    def get_bbox(self, bbox_values: Any, limit: int) -> RiskMapRows:
        """Return a small GeoJSON polygon row."""
        del bbox_values, limit
        return RiskMapRows(
            rows=[
                {
                    "h3_index": VALID_CELL,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-75.70, 45.41],
                                [-75.68, 45.41],
                                [-75.68, 45.43],
                                [-75.70, 45.41],
                            ]
                        ],
                    },
                    "features": self.aggregate,
                }
            ],
            truncated=False,
        )

    def get_h3(self, h3_index: str) -> dict[str, float | int] | None:
        """Return aggregate features only for the fixture cell."""
        return self.aggregate if h3_index == VALID_CELL else None


def fake_predictor() -> RiskPredictor:
    """Return a predictor with explicit synthetic provenance."""
    bundle = ModelBundle(
        model=FakeClassifier(),
        feature_columns=("line_length_km_water", "heat_index_c"),
        decision_threshold=0.5,
        metadata={
            "artifact_schema_version": 1,
            "target_mode": "synthetic",
            "model_version": "test-version",
        },
    )
    return RiskPredictor(bundle)


def fake_weather() -> dict[str, Any]:
    """Return one synthetic aggregate weather row."""
    return {
        "timestamp": "2024-07-01T12:00:00",
        "station_name": "SYNTHETIC OTTAWA WEATHER",
        "temperature_c": 30.0,
        "relative_humidity": 60.0,
        "heat_index_c": 32.0,
        "consecutive_dry_days": 3.0,
        "rainfall_48h": 0.0,
    }


@pytest.fixture(autouse=True)
def api_dependencies() -> None:
    """Isolate API tests from local files, models, databases, and the network."""
    app.dependency_overrides[get_repository] = FakeRepository
    app.dependency_overrides[get_predictor] = fake_predictor
    app.dependency_overrides[get_weather_features] = fake_weather
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def test_health_and_security_headers() -> None:
    """Liveness is cheap and every response includes browser hardening headers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ottawa-infra-stress"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["cache-control"] == "no-store"


def test_cors_allows_local_origin_not_arbitrary_site() -> None:
    """CORS does not reflect or wildcard an untrusted browser origin."""
    local = client.get("/health", headers={"Origin": "http://127.0.0.1:3000"})
    remote = client.get("/health", headers={"Origin": "https://evil.example"})
    assert local.headers["access-control-allow-origin"] == "http://127.0.0.1:3000"
    assert "access-control-allow-origin" not in remote.headers


def test_weather_summary_is_explicitly_synthetic() -> None:
    """Weather output labels mock mode instead of presenting a live observation."""
    response = client.get("/weather-summary")
    assert response.status_code == 200
    data = response.json()
    assert data["temperature_c"] == 30.0
    assert data["is_synthetic"] is True
    assert data["forecast_horizon_hours"] == 48


def test_metrics_always_warn_about_synthetic_target() -> None:
    """Model metadata never represents synthetic metrics as operational evidence."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "Synthetic" in data["target_warning"]
    assert data["prediction_horizon_hours"] == 48


def test_risk_map_returns_bounded_aggregate_geojson() -> None:
    """Risk map returns H3 aggregates with model provenance."""
    response = client.get(
        "/risk-map?min_lon=-75.72&min_lat=45.40&max_lon=-75.68&max_lat=45.44"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1
    properties = data["features"][0]["properties"]
    assert properties["risk_score"] == 0.8
    assert properties["target_mode"] == "synthetic"
    assert "description" not in properties
    assert data["metadata"]["truncated"] is False


def test_risk_map_accepts_bounded_scenario_inputs() -> None:
    """What-if inputs are labeled in response metadata."""
    response = client.get(
        "/risk-map?min_lon=-75.72&min_lat=45.40&max_lon=-75.68&max_lat=45.44"
        "&sim_temp_c=38.5&sim_humidity=70&sim_dry_days=8"
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["scenario_applied"] is True


@pytest.mark.parametrize(
    "query, expected_detail",
    [
        (
            "min_lon=-75.60&min_lat=45.40&max_lon=-75.70&max_lat=45.44",
            "min_lon",
        ),
        (
            "min_lon=-75.70&min_lat=45.50&max_lon=-75.60&max_lat=45.40",
            "min_lat",
        ),
        (
            "min_lon=-76.35&min_lat=45.25&max_lon=-75.50&max_lat=45.55",
            "too large",
        ),
        (
            "min_lon=-80&min_lat=45.40&max_lon=-79.9&max_lat=45.44",
            "Ottawa",
        ),
    ],
)
def test_risk_map_rejects_unsafe_bbox(query: str, expected_detail: str) -> None:
    """Reversed, oversized, and out-of-project bboxes fail before storage access."""
    response = client.get(f"/risk-map?{query}")
    assert response.status_code == 400
    assert expected_detail in response.json()["detail"]


def test_risk_map_enforces_server_feature_limit() -> None:
    """Clients cannot override the configured response-size ceiling."""
    response = client.get(
        "/risk-map?min_lon=-75.72&min_lat=45.40&max_lon=-75.68&max_lat=45.44"
        "&max_features=9999"
    )
    assert response.status_code == 400


def test_single_h3_returns_aggregate_features() -> None:
    """A valid present cell returns aggregate inputs and provenance."""
    response = client.get(f"/risk/{VALID_CELL}")
    assert response.status_code == 200
    data = response.json()
    assert data["risk_score"] == 0.8
    assert data["features"]["building_count"] == 10
    assert data["target_mode"] == "synthetic"


def test_valid_missing_h3_returns_404() -> None:
    """A syntactically valid cell absent from storage is a 404."""
    response = client.get(f"/risk/{MISSING_CELL}")
    assert response.status_code == 404
    assert response.json()["detail"] == "H3 cell not found"


def test_invalid_h3_returns_422_before_storage() -> None:
    """Malformed indexes are rejected by request validation."""
    response = client.get("/risk/not-an-h3-cell")
    assert response.status_code == 422


def test_artifact_error_is_generic() -> None:
    """Model failures do not expose paths or Python exception details."""
    def broken_predictor() -> RiskPredictor:
        raise ArtifactError("secret path C:/private/model.joblib")

    app.dependency_overrides[get_predictor] = broken_predictor
    response = client.get(
        "/risk-map?min_lon=-75.72&min_lat=45.40&max_lon=-75.68&max_lat=45.44"
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Prediction artifacts are unavailable"}
    assert "private" not in response.text


def test_get_risk_level_thresholds() -> None:
    """Display bands use named constants and exact boundary behavior."""
    assert get_risk_level(0.85) == "high"
    assert get_risk_level(0.70) == "high"
    assert get_risk_level(0.69) == "medium"
    assert get_risk_level(0.30) == "medium"
    assert get_risk_level(0.29) == "low"
    with pytest.raises(ValueError):
        get_risk_level(float("nan"))
