"""Serve Ottawa 311 insights and the dashboard through FastAPI.

The API restricts CORS to configured local origins, emits basic security
headers, and maps internal failures to generic client messages while keeping
sanitized server logs.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Annotated, Any

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.services import ModelBundle, Snapshot, load_model_bundle, load_snapshot
from src.config import API_CORS_ORIGINS, API_HOST, API_PORT, FRONTEND_DIR, MODELS_DIR
from src.errors import ArtifactError, DataValidationError
from src.insights import remaining_wait, ward_metrics, ward_report
from src.train_model import MATURITY_DAYS

logger = logging.getLogger(__name__)

ATTRIBUTION = "Contains information licensed under the Open Government Licence – City of Ottawa."

app = FastAPI(
    title="Ottawa 311 Insights",
    description=(
        "Ward report cards, request closing times, and unusual spikes from City of Ottawa "
        "311 open data. Not affiliated with or endorsed by the City of Ottawa."
    ),
    version="4.0.0",
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


@lru_cache(maxsize=1)
def get_model() -> ModelBundle:
    """Load the trusted configured model once per API process."""
    return load_model_bundle()


@lru_cache(maxsize=1)
def get_snapshot() -> Snapshot:
    """Load the 311 snapshot and its derived insights once per API process."""
    return load_snapshot()


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
    return JSONResponse(status_code=503, content={"detail": "The data is unavailable"})


@app.exception_handler(DataValidationError)
async def handle_internal_data_error(request: Request, exc: DataValidationError) -> JSONResponse:
    """Treat malformed trusted snapshot files as an internal error."""
    logger.error("Internal data validation failed for %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal data validation failed"})


SnapshotDependency = Annotated[Snapshot, Depends(get_snapshot)]
TypeQuery = Query(..., alias="type", min_length=1, max_length=200)


def require_known_type(snapshot: Snapshot, request_type: str) -> None:
    """Reject request types too rare to say anything reliable about."""
    if request_type not in {item["type"] for item in snapshot.request_types}:
        raise HTTPException(status_code=404, detail="Unknown or rare request type")


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report process liveness without accessing storage or models."""
    return {"status": "ok", "service": "ottawa-infra-stress"}


@app.get("/ready")
def readiness_check() -> JSONResponse:
    """Report whether the model and snapshot files load."""
    try:
        bundle = get_model()
        get_snapshot()
    except (ArtifactError, DataValidationError):
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "model_version": bundle.metadata["model_version"]},
    )


@app.get("/metrics")
def model_metrics() -> dict[str, Any]:
    """Return the reviewed model evaluation written at training time."""
    metrics: dict[str, Any] = {}
    metrics_path = MODELS_DIR / "model_metrics.json"
    if metrics_path.is_file():
        try:
            parsed = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                metrics = parsed
        except (OSError, json.JSONDecodeError):
            logger.error("Model metrics artifact is invalid")
    return {"metrics": metrics, "model_type": "LightGBM quantile regression (50th and 90th)"}


@app.get("/wards")
def wards(snapshot: SnapshotDependency) -> dict[str, Any]:
    """Return ward boundaries with 12-month request totals and current spike counts."""
    return {
        **snapshot.wards,
        "as_of": snapshot.end.date().isoformat(),
        "attribution": ATTRIBUTION,
    }


@app.get("/wards/{ward}")
def ward_report_card(
    snapshot: SnapshotDependency,
    ward: int = Path(..., ge=1, le=24),
) -> dict[str, Any]:
    """Return one ward's most common requests and closing times compared with the city."""
    report = ward_report(snapshot.requests, ward, snapshot.end, snapshot.batch_types)
    spikes = [spike for spike in snapshot.spikes if spike["ward"] == ward]
    return {**report, "name": snapshot.ward_name(ward), "spikes": spikes}


@app.get("/city")
def city_report_card(snapshot: SnapshotDependency) -> dict[str, Any]:
    """Return the city-wide report card and every unusual spike this week."""
    report = ward_report(snapshot.requests, None, snapshot.end, snapshot.batch_types)
    return {**report, "name": "All of Ottawa", "spikes": snapshot.spikes}


@app.get("/request-types")
def request_types(snapshot: SnapshotDependency) -> list[dict[str, Any]]:
    """Return request types common enough to estimate, most frequent first."""
    return snapshot.request_types


@app.get("/estimate")
def estimate(
    snapshot: SnapshotDependency,
    model: Annotated[ModelBundle, Depends(get_model)],
    request_type: str = TypeQuery,
    ward: int = Query(..., ge=1, le=24),
) -> dict[str, Any]:
    """Estimate how long a new request of this type in this ward takes to be closed."""
    require_known_type(snapshot, request_type)
    batch_closed = request_type in snapshot.batch_types
    usual: float | None
    upper: float | None
    if batch_closed:
        # The model is not trained on these types; only their requests since the City's
        # mass closure have real closing dates, so the answer is read from those.
        mature_before = snapshot.end - pd.Timedelta(days=MATURITY_DAYS)
        history = remaining_wait(snapshot.timed, request_type, ward, 0, mature_before)
        usual, upper = history["usual_more_days"], history["nine_in_ten_more_days"]
    else:
        days = model.estimate(request_type, ward, pd.Timestamp.now())
        usual, upper = round(days["usual_days"], 1), round(days["nine_in_ten_days"], 1)
    return {
        "type": request_type,
        "ward": ward,
        "usual_days": usual,
        "nine_in_ten_days": upper,
        "batch_closed": batch_closed,
        "note": model.metadata["warning"],
    }


@app.get("/remaining")
def remaining(
    snapshot: SnapshotDependency,
    request_type: str = TypeQuery,
    ward: int = Query(..., ge=1, le=24),
    waited: int = Query(..., ge=1, le=365),
) -> dict[str, Any]:
    """Say how much longer requests like this took once they had waited ``waited`` days."""
    require_known_type(snapshot, request_type)
    mature_before = snapshot.end - pd.Timedelta(days=MATURITY_DAYS)
    result = remaining_wait(snapshot.timed, request_type, ward, waited, mature_before)
    return {"type": request_type, "ward": ward, "waited_days": waited, **result}


@app.get("/ward-metrics")
def ward_metrics_for_map(
    snapshot: SnapshotDependency,
    request_type: str | None = Query(None, alias="type", min_length=1, max_length=200),
) -> dict[str, Any]:
    """Return what the map can be coloured by per ward, for one request type or all."""
    if request_type is not None:
        require_known_type(snapshot, request_type)
    metrics = ward_metrics(snapshot.requests, snapshot.end, request_type)
    wards = []
    for feature in snapshot.wards["features"]:
        ward = feature["properties"]["ward"]
        spikes = sum(
            spike["ward"] == ward and request_type in (None, spike["type"])
            for spike in snapshot.spikes
        )
        wards.append(
            {
                "ward": ward,
                **metrics.get(ward, {"requests": 0, "median_days": None}),
                "spikes": spikes,
            }
        )
    return {"type": request_type, "wards": wards}


@app.get("/spikes")
def spikes(snapshot: SnapshotDependency) -> dict[str, Any]:
    """Return ward/request-type pairs with far more requests this week than usual."""
    return {"week_ending": snapshot.end.date().isoformat(), "spikes": snapshot.spikes}


if FRONTEND_DIR.is_dir() and (FRONTEND_DIR / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=False)
