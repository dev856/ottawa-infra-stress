"""Storage, trusted model loading, and batch inference services for the API.

The public routes depend on H3-level aggregate records only. Mock mode reads
repository-local Parquet artifacts; live mode uses one parameterized batch query.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from shapely.geometry import box, mapping
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from src.config import DATA_DIR, DATA_SOURCE_MODE, MODEL_PATH, get_database_url
from src.errors import ArtifactError, DataValidationError, ExternalServiceError
from src.fetch_weather_features import calculate_heat_index_c, calculate_weather_features
from src.mock_data_sources import generate_mock_weather_observations

logger = logging.getLogger(__name__)

AGGREGATE_FEATURE_COLUMNS = (
    "centroid_lat",
    "centroid_lon",
    "line_length_km_water",
    "line_count_water",
    "line_length_km_road",
    "line_count_road",
    "building_count",
    "median_year_built",
    "pct_pre_1980",
)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Validated trusted model artifact components used for inference."""

    model: Any
    feature_columns: tuple[str, ...]
    decision_threshold: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RiskMapRows:
    """Aggregate grid rows plus response truncation status."""

    rows: list[dict[str, Any]]
    truncated: bool


def load_model_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    """Load and validate a trusted joblib bundle from the configured models path."""
    if not path.is_file():
        raise ArtifactError("The trained model artifact is missing")
    try:
        payload = joblib.load(path)
    except Exception as exc:  # joblib can wrap several format-specific errors
        raise ArtifactError("The trusted model artifact could not be loaded") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("The model artifact uses an unsupported legacy format")

    model = payload.get("model")
    feature_columns = payload.get("feature_columns")
    threshold = payload.get("decision_threshold")
    metadata = payload.get("metadata")
    if not hasattr(model, "predict_proba"):
        raise ArtifactError("The model artifact has no classifier")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ArtifactError("The model artifact has no feature contract")
    if not all(isinstance(column, str) and column for column in feature_columns):
        raise ArtifactError("The model artifact feature contract is invalid")
    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise ArtifactError("The model artifact threshold is invalid")
    if not isinstance(metadata, dict) or metadata.get("artifact_schema_version") != 1:
        raise ArtifactError("The model artifact metadata is invalid")
    return ModelBundle(
        model=model,
        feature_columns=tuple(feature_columns),
        decision_threshold=float(threshold),
        metadata=metadata,
    )


def load_latest_weather_features(
    data_dir: Path = DATA_DIR,
    data_source_mode: str = DATA_SOURCE_MODE,
) -> dict[str, Any]:
    """Load the latest weather feature row or explicit deterministic mock data."""
    weather_path = data_dir / "weather_features.parquet"
    if weather_path.is_file():
        try:
            weather = pd.read_parquet(weather_path)
        except (OSError, ValueError) as exc:
            raise ArtifactError("The weather feature artifact cannot be read") from exc
        if weather.empty:
            raise ArtifactError("The weather feature artifact is empty")
    elif data_source_mode == "mock":
        weather = calculate_weather_features(generate_mock_weather_observations())
    else:
        raise ArtifactError("The weather feature artifact is missing")

    latest = weather.sort_values("timestamp").iloc[-1].to_dict()
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in latest.items()
    }


class RiskRepository:
    """Read aggregate H3 data from one configured storage backend."""

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        data_source_mode: str = DATA_SOURCE_MODE,
    ) -> None:
        self.data_dir = data_dir
        self.data_source_mode = data_source_mode

    @staticmethod
    def _numeric_features(row: Mapping[str, Any]) -> dict[str, float | int]:
        """Select only reviewed aggregate numeric columns from a storage row."""
        features: dict[str, float | int] = {}
        for column in AGGREGATE_FEATURE_COLUMNS:
            value = row.get(column)
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                features[column] = value
        return features

    def _read_parquet_tables(self) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
        """Read and validate local aggregate artifacts."""
        grid_path = self.data_dir / "h3_grid.parquet"
        features_path = self.data_dir / "infrastructure_features.parquet"
        if not grid_path.is_file() or not features_path.is_file():
            raise ArtifactError("Run the pipeline to create aggregate risk-map artifacts")
        try:
            grid = gpd.read_parquet(grid_path)
            features = pd.read_parquet(features_path)
        except (OSError, ValueError) as exc:
            raise ArtifactError("Aggregate risk-map artifacts cannot be read") from exc
        required_grid = {"h3_index", "geometry"}
        if not required_grid.issubset(grid.columns) or "h3_index" not in features:
            raise DataValidationError("Aggregate risk-map artifacts have an invalid schema")
        return grid, features

    def _bbox_from_parquet(
        self,
        bbox_values: tuple[float, float, float, float],
        limit: int,
    ) -> RiskMapRows:
        """Return one joined Parquet batch for a bounding box."""
        grid, features = self._read_parquet_tables()
        requested_area = box(*bbox_values)
        selected = grid[grid.geometry.intersects(requested_area)].sort_values("h3_index")
        truncated = len(selected) > limit
        selected = selected.head(limit)
        joined = selected[["h3_index", "geometry"]].merge(
            features,
            on="h3_index",
            how="left",
            validate="one_to_one",
        )
        rows: list[dict[str, Any]] = []
        for row in joined.to_dict(orient="records"):
            rows.append(
                {
                    "h3_index": row["h3_index"],
                    "geometry": mapping(row["geometry"]),
                    "features": self._numeric_features(row),
                }
            )
        return RiskMapRows(rows=rows, truncated=truncated)

    def _bbox_from_database(
        self,
        bbox_values: tuple[float, float, float, float],
        limit: int,
    ) -> RiskMapRows:
        """Return geometry and aggregate features in one parameterized query."""
        query = text(
            """
            SELECT
                grid.h3_index,
                ST_AsGeoJSON(grid.geometry) AS geometry_json,
                infra.centroid_lat,
                infra.centroid_lon,
                infra.line_length_km_water,
                infra.line_count_water,
                infra.line_length_km_road,
                infra.line_count_road,
                infra.building_count,
                infra.median_year_built,
                infra.pct_pre_1980
            FROM features.h3_grid AS grid
            LEFT JOIN features.infrastructure_features AS infra USING (h3_index)
            WHERE ST_Intersects(
                grid.geometry,
                ST_MakeEnvelope(:west, :south, :east, :north, 4326)
            )
            ORDER BY grid.h3_index
            LIMIT :row_limit
            """
        )
        west, south, east, north = bbox_values
        engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
        try:
            with engine.connect() as connection:
                result = connection.execute(
                    query,
                    {
                        "west": west,
                        "south": south,
                        "east": east,
                        "north": north,
                        "row_limit": limit + 1,
                    },
                )
                database_rows = result.mappings().all()
        except SQLAlchemyError as exc:
            raise ExternalServiceError("Aggregate database query failed") from exc
        finally:
            engine.dispose()

        truncated = len(database_rows) > limit
        rows = [
            {
                "h3_index": row["h3_index"],
                "geometry": json.loads(row["geometry_json"]),
                "features": self._numeric_features(dict(row)),
            }
            for row in database_rows[:limit]
        ]
        return RiskMapRows(rows=rows, truncated=truncated)

    def get_bbox(
        self,
        bbox_values: tuple[float, float, float, float],
        limit: int,
    ) -> RiskMapRows:
        """Read a bbox from the configured backend with explicit hybrid fallback."""
        if self.data_source_mode == "mock":
            return self._bbox_from_parquet(bbox_values, limit)
        try:
            return self._bbox_from_database(bbox_values, limit)
        except ExternalServiceError:
            if self.data_source_mode != "hybrid":
                raise
            logger.warning("Database risk query failed; using local aggregate artifacts")
            return self._bbox_from_parquet(bbox_values, limit)

    def get_h3(self, h3_index: str) -> dict[str, float | int] | None:
        """Return one H3 aggregate feature dictionary or ``None``."""
        if self.data_source_mode == "mock":
            grid, features = self._read_parquet_tables()
            if not (grid["h3_index"] == h3_index).any():
                return None
            matched = features[features["h3_index"] == h3_index]
            return self._numeric_features(matched.iloc[0].to_dict()) if not matched.empty else {}

        query = text(
            """
            SELECT
                centroid_lat, centroid_lon, line_length_km_water, line_count_water,
                line_length_km_road, line_count_road, building_count,
                median_year_built, pct_pre_1980
            FROM features.infrastructure_features
            WHERE h3_index = :h3_index
            """
        )
        engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
        try:
            with engine.connect() as connection:
                row = connection.execute(query, {"h3_index": h3_index}).mappings().first()
        except SQLAlchemyError as exc:
            if self.data_source_mode == "hybrid":
                fallback = RiskRepository(self.data_dir, "mock")
                return fallback.get_h3(h3_index)
            raise ExternalServiceError("Aggregate database query failed") from exc
        finally:
            engine.dispose()
        return self._numeric_features(dict(row)) if row else None


class RiskPredictor:
    """Apply one validated model feature contract to aggregate H3 rows."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle

    def _apply_scenario(
        self,
        features: dict[str, Any],
        scenario: Mapping[str, float | None],
    ) -> dict[str, Any]:
        """Apply optional, bounded what-if inputs to a copied feature row."""
        result = features.copy()
        temperature = scenario.get("temperature_c")
        humidity = scenario.get("relative_humidity")
        dry_days = scenario.get("dry_days")
        if temperature is not None:
            result["temperature_c"] = temperature
            result["temp_max_24h"] = temperature
            result["temp_max_48h"] = temperature
        if humidity is not None:
            result["relative_humidity"] = humidity
        if temperature is not None or humidity is not None:
            current_temperature = float(result.get("temperature_c", 20.0))
            current_humidity = float(result.get("relative_humidity", 50.0))
            calculated = calculate_heat_index_c(current_temperature, current_humidity)
            result["heat_index_c"] = calculated
            result["heat_index_max_24h"] = calculated
            result["heat_index_max_48h"] = calculated
        if dry_days is not None:
            result["consecutive_dry_days"] = dry_days
        return result

    def predict_many(
        self,
        aggregate_rows: list[dict[str, float | int]],
        weather_features: Mapping[str, Any],
        scenario: Mapping[str, float | None],
    ) -> list[float]:
        """Return one finite probability per H3 aggregate in a batch."""
        if not aggregate_rows:
            return []
        model_rows = [
            self._apply_scenario({**row, **weather_features}, scenario)
            for row in aggregate_rows
        ]
        dataframe = pd.DataFrame(model_rows)
        for column in self.bundle.feature_columns:
            if column not in dataframe:
                dataframe[column] = np.nan
        model_input = dataframe[list(self.bundle.feature_columns)].apply(
            pd.to_numeric,
            errors="coerce",
        )
        try:
            probabilities = self.bundle.model.predict_proba(model_input)[:, 1]
        except Exception as exc:
            raise ArtifactError("Model inference failed") from exc
        scores = [float(score) for score in probabilities]
        if any(not np.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ArtifactError("Model inference returned an invalid probability")
        return scores
