"""Load the trusted model and the open-data snapshot for the API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, REQUESTS_PATH, WARDS_PATH
from src.errors import ArtifactError, DataValidationError
from src.insights import (
    REPORT_WINDOW_DAYS,
    batch_closed_types,
    bulk_closed,
    bulk_closure_end,
    find_spikes,
    last_days,
    snapshot_end,
    ward_totals,
)
from src.train_model import QUANTILES, load_requests, model_inputs

logger = logging.getLogger(__name__)

# Residents mostly report online; the model was trained across all channels.
RESIDENT_CHANNEL = "Web"
MIN_REQUESTS_TO_ESTIMATE = 100


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Validated trusted model artifact components used for inference."""

    models: dict[str, Any]
    categories: dict[str, list[str]]
    metadata: dict[str, Any]

    def estimate(self, request_type: str, ward: int, opened: pd.Timestamp) -> dict[str, float]:
        """Return the usual and 9-in-10 days to close for one new request."""
        frame = pd.DataFrame(
            {
                "type": [request_type],
                "ward": [ward],
                "channel": [RESIDENT_CHANNEL],
                "opened": [opened],
            }
        )
        inputs = model_inputs(frame, self.categories)
        try:
            days = {name: float(model.predict(inputs)[0]) for name, model in self.models.items()}
        except Exception as exc:
            raise ArtifactError("Model inference failed") from exc
        if not all(np.isfinite(value) for value in days.values()):
            raise ArtifactError("Model inference returned an invalid estimate")
        usual = max(0.0, days["usual_days"])
        # Two separately trained quantiles can cross; the upper one is never below the median.
        return {"usual_days": usual, "nine_in_ten_days": max(usual, days["nine_in_ten_days"])}


def load_model_bundle(path: Path = MODEL_PATH) -> ModelBundle:
    """Load and validate a trusted joblib bundle from the configured models path."""
    if not path.is_file():
        raise ArtifactError("The trained model artifact is missing")
    try:
        payload = joblib.load(path)
    except Exception as exc:  # joblib can wrap several format-specific errors
        raise ArtifactError("The trusted model artifact could not be loaded") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("The model artifact uses an unsupported format")

    models = payload.get("models")
    categories = payload.get("categories")
    metadata = payload.get("metadata")
    if not isinstance(models, dict) or set(models) != set(QUANTILES):
        raise ArtifactError("The model artifact has the wrong models")
    if not all(hasattr(model, "predict") for model in models.values()):
        raise ArtifactError("The model artifact has an invalid model")
    if not isinstance(categories, dict) or not categories.get("type"):
        raise ArtifactError("The model artifact has no categories")
    if not isinstance(metadata, dict) or metadata.get("artifact_schema_version") != 3:
        raise ArtifactError("The model artifact metadata is invalid")
    return ModelBundle(models=models, categories=categories, metadata=metadata)


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The 311 snapshot plus everything derived from it once at startup."""

    requests: pd.DataFrame
    end: pd.Timestamp
    batch_types: set[str]
    request_types: list[dict[str, Any]]
    spikes: list[dict[str, Any]]
    wards: dict[str, Any]
    # ``requests`` without the rows whose closing date was set in the City's mass closure.
    timed: pd.DataFrame

    def ward_name(self, ward: int) -> str:
        for feature in self.wards["features"]:
            if feature["properties"]["ward"] == ward:
                return str(feature["properties"]["name"])
        return f"Ward {ward}"


def load_snapshot(
    requests_path: Path = REQUESTS_PATH,
    wards_path: Path = WARDS_PATH,
) -> Snapshot:
    """Load the snapshot and precompute ward totals, request types, and spikes."""
    requests = load_requests(requests_path)
    try:
        wards = json.loads(wards_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError("The ward boundary snapshot cannot be read") from exc
    if not isinstance(wards, dict) or not wards.get("features"):
        raise DataValidationError("The ward boundary snapshot has no wards")

    end = snapshot_end(requests)
    totals = ward_totals(requests, end)
    spikes = find_spikes(requests, end)
    for feature in wards["features"]:
        ward = feature["properties"]["ward"]
        feature["properties"]["total_requests_12_months"] = totals.get(ward, 0)
        feature["properties"]["spikes"] = sum(spike["ward"] == ward for spike in spikes)

    recent = last_days(requests, REPORT_WINDOW_DAYS, end)
    counts = recent["type"].value_counts()
    departments = recent.groupby("type")["category"].first()
    request_types = [
        {"type": request_type, "category": str(departments[request_type]), "count": int(count)}
        for request_type, count in counts[counts >= MIN_REQUESTS_TO_ESTIMATE].items()
    ]
    batch_types = batch_closed_types(requests)
    untrusted = bulk_closed(requests, batch_types, bulk_closure_end(requests))
    # Report cards only need these; dropping the other text columns halves memory use.
    kept = requests[["opened", "type", "ward", "status", "days_to_close"]].copy()
    # Still counted as requests, but left out of every "days to close" figure.
    kept.loc[untrusted, "days_to_close"] = float("nan")
    snapshot = Snapshot(kept, end, batch_types, request_types, spikes, wards, kept[~untrusted])
    for spike in spikes:
        spike["ward_name"] = snapshot.ward_name(spike["ward"])
    return snapshot
