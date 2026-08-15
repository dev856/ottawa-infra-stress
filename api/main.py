"""Serve bounded, aggregate-only risk responses through FastAPI.

The API validates H3 and bounding-box inputs, restricts CORS to configured local
origins, emits basic security headers, and maps internal failures to generic
client messages while retaining sanitized server logs.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Annotated, Any, Literal

import h3
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from api.services import (
    ModelBundle,
    RiskPredictor,
    RiskRepository,
    load_latest_weather_features,
    load_model_bundle,
)
from src.config import (
    API_CORS_ORIGINS,
    API_HOST,
    API_PORT,
    DATA_SOURCE_MODE,
    FRONTEND_DIR,
    H3_RESOLUTION,
    MAX_API_FEATURES,
    MAX_BBOX_SPAN_DEGREES,
    MODELS_DIR,
    OTTAWA_BBOX_EAST,
    OTTAWA_BBOX_NORTH,
    OTTAWA_BBOX_SOUTH,
    OTTAWA_BBOX_WEST,
    PREDICTION_HORIZON_HOURS,
)
from src.errors import ArtifactError, DataValidationError, ExternalServiceError

logger = logging.getLogger(__name__)

HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30
DEFAULT_MAP_FEATURES = min(500, MAX_API_FEATURES)

app = FastAPI(
    title="Ottawa Infrastructure Stress Predictor",
    description=(
        "Local synthetic demonstration of aggregate 48-hour water-main-break risk. "
        "Not an operational municipal forecast."
    ),
    version="2.0.0",
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "testserver"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(API_CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["Accept", "Content-Type"],
    max_age=600,
)


class HealthResponse(BaseModel):
    """Process liveness response."""

    status: Literal["ok"]
    service: Literal["ottawa-infra-stress"]


class WeatherSummaryResponse(BaseModel):
    """Latest aggregate weather indicators used by inference."""

    model_config = ConfigDict(extra="forbid")

    temperature_c: float
    relative_humidity: float
    heat_index_c: float
    consecutive_dry_days: float
    rainfall_48h: float
    station_name: str
    observed_at: str
    data_source_mode: str
    is_synthetic: bool
    forecast_horizon_hours: int


class HexagonRiskResponse(BaseModel):
    """One aggregate H3 risk response."""

    h3_index: str
    risk_score: float = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    target_mode: str
    model_version: str
    features: dict[str, float | int]


def get_risk_level(score: float) -> Literal["low", "medium", "high"]:
    """Convert a finite probability into documented display bands."""
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("score must be a finite probability")
    if score >= HIGH_RISK_THRESHOLD:
        return "high"
    if score >= MEDIUM_RISK_THRESHOLD:
        return "medium"
    return "low"


@lru_cache(maxsize=1)
def get_repository() -> RiskRepository:
    """Return the process-wide aggregate storage reader."""
    return RiskRepository()


@lru_cache(maxsize=1)
def get_predictor() -> RiskPredictor:
    """Load the trusted configured model once per API process."""
    return RiskPredictor(load_model_bundle())


def get_weather_features() -> dict[str, Any]:
    """Return latest weather inputs; isolated as an injectable dependency."""
    return load_latest_weather_features()


@app.middleware("http")
async def add_security_headers(request: Request, call_next: Any) -> Any:
    """Add simple browser security headers to every API response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.exception_handler(ArtifactError)
async def handle_artifact_error(request: Request, exc: ArtifactError) -> JSONResponse:
    """Hide model/filesystem details from clients while logging safe context."""
    logger.error("Artifact operation failed for %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": "Prediction artifacts are unavailable"})


@app.exception_handler(ExternalServiceError)
async def handle_external_error(request: Request, exc: ExternalServiceError) -> JSONResponse:
    """Map storage/service failures to a generic availability response."""
    logger.error("External dependency failed for %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": "Prediction service is unavailable"})


@app.exception_handler(DataValidationError)
async def handle_internal_data_error(request: Request, exc: DataValidationError) -> JSONResponse:
    """Treat malformed trusted artifacts as an internal error without leaking rows."""
    logger.error("Internal data validation failed for %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal data validation failed"})


def _validate_bbox(west: float, south: float, east: float, north: float) -> None:
    """Reject reversed, oversized, or out-of-project bounding boxes."""
    if west >= east:
        raise HTTPException(status_code=400, detail="min_lon must be less than max_lon")
    if south >= north:
        raise HTTPException(status_code=400, detail="min_lat must be less than max_lat")
    if east - west > MAX_BBOX_SPAN_DEGREES or north - south > MAX_BBOX_SPAN_DEGREES:
        raise HTTPException(status_code=400, detail="Bounding box is too large")
    if not (
        OTTAWA_BBOX_WEST <= west <= OTTAWA_BBOX_EAST
        and OTTAWA_BBOX_WEST <= east <= OTTAWA_BBOX_EAST
        and OTTAWA_BBOX_SOUTH <= south <= OTTAWA_BBOX_NORTH
        and OTTAWA_BBOX_SOUTH <= north <= OTTAWA_BBOX_NORTH
    ):
        raise HTTPException(status_code=400, detail="Bounding box must stay within Ottawa")


def _model_metadata(bundle: ModelBundle) -> tuple[str, str]:
    """Return required public provenance fields from a validated bundle."""
    return (
        str(bundle.metadata.get("target_mode", "unknown")),
        str(bundle.metadata.get("model_version", "unknown")),
    )


@app.get("/health", response_model=HealthResponse)
def health_check() -> dict[str, str]:
    """Report process liveness without accessing storage or models."""
    return {"status": "ok", "service": "ottawa-infra-stress"}


@app.get("/ready")
def readiness_check() -> JSONResponse:
    """Report non-secret artifact readiness separately from liveness."""
    try:
        bundle = load_model_bundle()
        load_latest_weather_features()
    except (ArtifactError, DataValidationError):
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "model_version": bundle.metadata["model_version"],
            "target_mode": bundle.metadata["target_mode"],
        },
    )


@app.get("/weather-summary", response_model=WeatherSummaryResponse)
def weather_summary(
    weather: Annotated[dict[str, Any], Depends(get_weather_features)],
) -> dict[str, Any]:
    """Return the latest aggregate weather row with explicit source labeling."""
    timestamp = pd.Timestamp(weather["timestamp"])
    return {
        "temperature_c": round(float(weather["temperature_c"]), 1),
        "relative_humidity": round(float(weather["relative_humidity"]), 1),
        "heat_index_c": round(float(weather["heat_index_c"]), 1),
        "consecutive_dry_days": round(float(weather["consecutive_dry_days"]), 2),
        "rainfall_48h": round(float(weather["rainfall_48h"]), 2),
        "station_name": str(weather.get("station_name", "Unknown station")),
        "observed_at": timestamp.isoformat(),
        "data_source_mode": DATA_SOURCE_MODE,
        "is_synthetic": DATA_SOURCE_MODE == "mock",
        "forecast_horizon_hours": PREDICTION_HORIZON_HOURS,
    }


@app.get("/metrics")
def model_metrics() -> dict[str, Any]:
    """Return reviewed model metadata and up to ten aggregate importances."""
    metrics_path = MODELS_DIR / "model_metrics.json"
    importance_path = MODELS_DIR / "feature_importance.csv"
    metrics: dict[str, Any] = {}
    features: list[dict[str, Any]] = []
    if metrics_path.is_file():
        try:
            parsed = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                metrics = parsed
        except (OSError, json.JSONDecodeError):
            logger.error("Model metrics artifact is invalid")
    if importance_path.is_file():
        try:
            features = pd.read_csv(importance_path).head(10).to_dict(orient="records")
        except (OSError, ValueError):
            logger.error("Feature-importance artifact is invalid")
    return {
        "metrics": metrics,
        "features": features,
        "model_type": "LightGBM binary classifier",
        "target_warning": "Synthetic demonstration target; not an operational forecast.",
        "spatial_resolution": f"H3 resolution {H3_RESOLUTION}",
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
    }


@app.get("/risk-map")
def risk_map(
    repository: Annotated[RiskRepository, Depends(get_repository)],
    predictor: Annotated[RiskPredictor, Depends(get_predictor)],
    weather: Annotated[dict[str, Any], Depends(get_weather_features)],
    min_lon: float = Query(..., ge=-180, le=180),
    min_lat: float = Query(..., ge=-90, le=90),
    max_lon: float = Query(..., ge=-180, le=180),
    max_lat: float = Query(..., ge=-90, le=90),
    max_features: int = Query(DEFAULT_MAP_FEATURES, ge=1, le=10_000),
    sim_temp_c: float | None = Query(None, ge=-50, le=60),
    sim_humidity: float | None = Query(None, ge=0, le=100),
    sim_dry_days: float | None = Query(None, ge=0, le=365),
) -> dict[str, Any]:
    """Return bounded H3 aggregate predictions as a GeoJSON FeatureCollection."""
    _validate_bbox(min_lon, min_lat, max_lon, max_lat)
    if max_features > MAX_API_FEATURES:
        raise HTTPException(status_code=400, detail="max_features exceeds the server limit")
    result = repository.get_bbox((min_lon, min_lat, max_lon, max_lat), max_features)
    scenario = {
        "temperature_c": sim_temp_c,
        "relative_humidity": sim_humidity,
        "dry_days": sim_dry_days,
    }
    scores = predictor.predict_many(
        [row["features"] for row in result.rows],
        weather,
        scenario,
    )
    target_mode, model_version = _model_metadata(predictor.bundle)
    features = []
    for row, score in zip(result.rows, scores, strict=True):
        aggregate = row["features"]
        features.append(
            {
                "type": "Feature",
                "geometry": row["geometry"],
                "properties": {
                    "h3_index": row["h3_index"],
                    "risk_score": round(score, 4),
                    "risk_level": get_risk_level(score),
                    "water_km": round(float(aggregate.get("line_length_km_water", 0)), 2),
                    "vintage_pre1980_pct": round(
                        float(aggregate.get("pct_pre_1980", 0)) * 100,
                        1,
                    ),
                    "target_mode": target_mode,
                    "model_version": model_version,
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "feature_count": len(features),
            "truncated": result.truncated,
            "target_mode": target_mode,
            "model_version": model_version,
            "scenario_applied": any(value is not None for value in scenario.values()),
        },
    }


@app.get("/risk/{h3_index}", response_model=HexagonRiskResponse)
def risk_for_hexagon(
    repository: Annotated[RiskRepository, Depends(get_repository)],
    predictor: Annotated[RiskPredictor, Depends(get_predictor)],
    weather: Annotated[dict[str, Any], Depends(get_weather_features)],
    h3_index: str = Path(..., min_length=15, max_length=20, pattern=r"^[0-9a-f]+$"),
) -> dict[str, Any]:
    """Return one validated H3 cell's aggregate features and prediction."""
    if not h3.is_valid_cell(h3_index) or h3.get_resolution(h3_index) != H3_RESOLUTION:
        raise HTTPException(status_code=422, detail="h3_index is invalid for this API")
    aggregate = repository.get_h3(h3_index)
    if aggregate is None:
        raise HTTPException(status_code=404, detail="H3 cell not found")
    score = predictor.predict_many(
        [aggregate],
        weather,
        {"temperature_c": None, "relative_humidity": None, "dry_days": None},
    )[0]
    target_mode, model_version = _model_metadata(predictor.bundle)
    return {
        "h3_index": h3_index,
        "risk_score": round(score, 4),
        "risk_level": get_risk_level(score),
        "target_mode": target_mode,
        "model_version": model_version,
        "features": aggregate,
    }


if FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
