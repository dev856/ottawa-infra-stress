"""Unit tests for trusted model loading and aggregate API storage services."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from api.services import (
    ModelBundle,
    RiskPredictor,
    RiskRepository,
    load_latest_weather_features,
    load_model_bundle,
)
from src.errors import ArtifactError


class FixedClassifier:
    """Return one stable probability for each input row."""

    def predict_proba(self, dataframe: Any) -> np.ndarray:
        """Return probabilities using the scikit-learn classifier shape."""
        return np.array([[0.25, 0.75] for _ in range(len(dataframe))])


class InvalidClassifier:
    """Return invalid probabilities to exercise artifact validation."""

    def predict_proba(self, dataframe: Any) -> np.ndarray:
        """Return out-of-range values for every input row."""
        return np.array([[-0.2, 1.2] for _ in range(len(dataframe))])


def _write_model_bundle(path: Path, *, threshold: float = 0.5) -> None:
    """Write one trusted test bundle using the production artifact schema."""
    joblib.dump(
        {
            "model": FixedClassifier(),
            "feature_columns": ["temperature_c", "heat_index_c"],
            "decision_threshold": threshold,
            "metadata": {"artifact_schema_version": 1, "target_mode": "synthetic"},
        },
        path,
    )


def test_load_model_bundle_validates_trusted_artifact(tmp_path: Path) -> None:
    """A local schema-versioned joblib bundle is loaded with typed fields."""
    artifact_path = tmp_path / "model.joblib"
    _write_model_bundle(artifact_path)

    bundle = load_model_bundle(artifact_path)

    assert bundle.feature_columns == ("temperature_c", "heat_index_c")
    assert bundle.decision_threshold == 0.5
    assert bundle.metadata["target_mode"] == "synthetic"


def test_load_model_bundle_rejects_missing_and_invalid_artifacts(tmp_path: Path) -> None:
    """Missing and malformed bundles fail with the project's safe error type."""
    with pytest.raises(ArtifactError, match="missing"):
        load_model_bundle(tmp_path / "missing.joblib")

    invalid_path = tmp_path / "invalid.joblib"
    joblib.dump({"model": FixedClassifier()}, invalid_path)
    with pytest.raises(ArtifactError, match="feature contract"):
        load_model_bundle(invalid_path)

    bad_threshold_path = tmp_path / "bad-threshold.joblib"
    _write_model_bundle(bad_threshold_path, threshold=2.0)
    with pytest.raises(ArtifactError, match="threshold"):
        load_model_bundle(bad_threshold_path)


def test_latest_weather_features_are_available_offline(tmp_path: Path) -> None:
    """Mock mode derives a deterministic latest row when no Parquet file exists."""
    latest = load_latest_weather_features(tmp_path, "mock")

    assert latest["station_name"] == "SYNTHETIC OTTAWA WEATHER"
    assert isinstance(latest["temperature_c"], float)
    assert "heat_index_c" in latest


def test_mock_repository_returns_only_aggregate_rows(tmp_path: Path) -> None:
    """The Parquet backend joins H3 geometry to reviewed numeric aggregates."""
    h3_index = "882baac88dfffff"
    polygon = Polygon(
        [
            (-75.71, 45.41),
            (-75.69, 45.41),
            (-75.69, 45.43),
            (-75.71, 45.41),
        ]
    )
    grid = gpd.GeoDataFrame(
        {"h3_index": [h3_index], "geometry": [polygon]},
        crs="EPSG:4326",
    )
    infrastructure = pd.DataFrame(
        {
            "h3_index": [h3_index],
            "centroid_lat": [45.42],
            "centroid_lon": [-75.70],
            "building_count": [12],
            "private_note": ["must never be returned"],
        }
    )
    grid.to_parquet(tmp_path / "h3_grid.parquet")
    infrastructure.to_parquet(tmp_path / "infrastructure_features.parquet")
    repository = RiskRepository(tmp_path, "mock")

    result = repository.get_bbox((-75.72, 45.40, -75.68, 45.44), limit=10)

    assert result.truncated is False
    assert result.rows[0]["h3_index"] == h3_index
    assert result.rows[0]["features"]["building_count"] == 12
    assert "private_note" not in result.rows[0]["features"]
    assert repository.get_h3(h3_index) == {
        "centroid_lat": 45.42,
        "centroid_lon": -75.70,
        "building_count": 12,
    }
    assert repository.get_h3("missing") is None


def test_predictor_applies_scenario_and_rejects_invalid_scores() -> None:
    """Scenario values are usable while non-probabilities fail securely."""
    bundle = ModelBundle(
        model=FixedClassifier(),
        feature_columns=("temperature_c", "heat_index_c", "consecutive_dry_days"),
        decision_threshold=0.5,
        metadata={"artifact_schema_version": 1},
    )
    predictor = RiskPredictor(bundle)

    scores = predictor.predict_many(
        [{"building_count": 10}],
        {"temperature_c": 20.0, "relative_humidity": 50.0},
        {"temperature_c": 35.0, "relative_humidity": 65.0, "dry_days": 5.0},
    )

    assert scores == [0.75]
    assert predictor.predict_many([], {}, {}) == []

    invalid_predictor = RiskPredictor(
        ModelBundle(
            model=InvalidClassifier(),
            feature_columns=("temperature_c",),
            decision_threshold=0.5,
            metadata={"artifact_schema_version": 1},
        )
    )
    with pytest.raises(ArtifactError, match="invalid probability"):
        invalid_predictor.predict_many([{"temperature_c": 20.0}], {}, {})
