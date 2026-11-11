"""API contract and security tests using an in-memory snapshot and model."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_model, get_snapshot
from api.services import ModelBundle, Snapshot, load_model_bundle, load_snapshot
from src.errors import ArtifactError

END = pd.Timestamp("2026-10-31")


class FakeModel:
    """Return a fixed number of days for every request."""

    def __init__(self, days: float) -> None:
        self.days = days

    def predict(self, inputs: Any) -> np.ndarray:
        return np.full(len(inputs), self.days)


def fake_model(usual: float = 3.0, nine_in_ten: float = 12.0) -> ModelBundle:
    return ModelBundle(
        models={"usual_days": FakeModel(usual), "nine_in_ten_days": FakeModel(nine_in_ten)},
        categories={"type": ["Noise", "Pothole"], "ward": ["1", "2"], "channel": ["Web"]},
        metadata={"artifact_schema_version": 3, "model_version": "test", "warning": "Closed date."},
    )


def fake_wards() -> dict[str, Any]:
    square = {
        "type": "Polygon",
        "coordinates": [[[-75.7, 45.4], [-75.6, 45.4], [-75.6, 45.5], [-75.7, 45.4]]],
    }
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": square,
                "properties": {"ward": 1, "name": "Somerset", "sector": "Urban"},
            },
            {
                "type": "Feature",
                "geometry": square,
                "properties": {"ward": 2, "name": "Kanata North", "sector": "Suburban"},
            },
        ],
    }


def fake_requests() -> pd.DataFrame:
    rows = [("2026-10-30", "2026-10-31", "Noise", 1)] * 120 + [
        ("2026-10-30", None, "Pothole", 2)
    ] * 3
    frame = pd.DataFrame(rows, columns=["opened", "closed", "type", "ward"])
    frame["opened"] = pd.to_datetime(frame["opened"])
    frame["closed"] = pd.to_datetime(frame["closed"])
    frame["ward"] = frame["ward"].astype("Int64")
    frame["status"] = "Resolved"
    frame["days_to_close"] = (frame["closed"] - frame["opened"]).dt.days
    return frame


def fake_snapshot() -> Snapshot:
    return Snapshot(
        requests=fake_requests(),
        end=END,
        batch_types={"Pothole"},
        request_types=[{"type": "Noise", "category": "Bylaw", "count": 120}],
        spikes=[
            {
                "ward": 1,
                "ward_name": "Somerset",
                "type": "Noise",
                "this_week": 120,
                "usual_per_week": 2.0,
            }
        ],
        wards=fake_wards(),
        timed=fake_requests(),
    )


@pytest.fixture(autouse=True)
def api_dependencies() -> None:
    """Isolate API tests from local files and models."""
    app.dependency_overrides[get_model] = fake_model
    app.dependency_overrides[get_snapshot] = fake_snapshot
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


def test_wards_include_date_and_attribution() -> None:
    data = client.get("/wards").json()
    assert data["as_of"] == "2026-10-31"
    assert "City of Ottawa" in data["attribution"]
    assert len(data["features"]) == 2


def test_ward_report_card_lists_top_requests_and_spikes() -> None:
    data = client.get("/wards/1").json()
    assert data["name"] == "Somerset"
    assert data["total_requests"] == 120
    assert data["top_requests"][0]["type"] == "Noise"
    assert data["spikes"][0]["this_week"] == 120


@pytest.mark.parametrize("ward", ["0", "25", "abc"])
def test_ward_report_rejects_unknown_wards(ward: str) -> None:
    assert client.get(f"/wards/{ward}").status_code == 422


def test_estimate_returns_rounded_days_and_flags() -> None:
    data = client.get("/estimate", params={"type": "Noise", "ward": 1}).json()
    assert data["usual_days"] == 3.0
    assert data["nine_in_ten_days"] == 12.0
    assert data["batch_closed"] is False


def test_estimate_never_returns_upper_below_usual() -> None:
    """Separately trained quantiles can cross; the API keeps them ordered."""
    app.dependency_overrides[get_model] = lambda: fake_model(usual=10.0, nine_in_ten=4.0)
    data = client.get("/estimate", params={"type": "Noise", "ward": 1}).json()
    assert data["nine_in_ten_days"] == 10.0


def test_estimate_rejects_unknown_types_and_wards() -> None:
    assert client.get("/estimate", params={"type": "Pothole", "ward": 1}).status_code == 404
    assert client.get("/estimate", params={"type": "Noise", "ward": 30}).status_code == 422


def test_request_types_and_spikes_endpoints() -> None:
    assert client.get("/request-types").json() == [
        {"type": "Noise", "category": "Bylaw", "count": 120}
    ]
    assert client.get("/spikes").json()["week_ending"] == "2026-10-31"


def test_artifact_error_is_generic() -> None:
    """Model failures do not expose paths or Python exception details."""

    def broken_model() -> ModelBundle:
        raise ArtifactError("secret path C:/private/model.joblib")

    app.dependency_overrides[get_model] = broken_model
    response = client.get("/estimate", params={"type": "Noise", "ward": 1})
    assert response.status_code == 503
    assert response.json() == {"detail": "The data is unavailable"}
    assert "private" not in response.text


def test_model_bundle_round_trip_and_validation(tmp_path: Path) -> None:
    bundle = fake_model()
    good = {"models": bundle.models, "categories": bundle.categories, "metadata": bundle.metadata}
    joblib.dump(good, tmp_path / "good.joblib")
    joblib.dump({**good, "metadata": {"artifact_schema_version": 2}}, tmp_path / "old.joblib")

    assert (
        load_model_bundle(tmp_path / "good.joblib").estimate("Noise", 1, END)["usual_days"] == 3.0
    )
    with pytest.raises(ArtifactError, match="metadata"):
        load_model_bundle(tmp_path / "old.joblib")
    with pytest.raises(ArtifactError, match="missing"):
        load_model_bundle(tmp_path / "absent.joblib")


def test_load_snapshot_adds_ward_totals_and_common_types(tmp_path: Path) -> None:
    requests_path = tmp_path / "requests.csv.gz"
    frame = fake_requests().drop(columns="days_to_close")
    frame["category"] = "X"
    frame["channel"] = "Web"
    frame.to_csv(requests_path, index=False)
    wards_path = tmp_path / "wards.geojson"
    wards_path.write_text(json.dumps(fake_wards()), encoding="utf-8")

    snapshot = load_snapshot(requests_path, wards_path)

    totals = {
        f["properties"]["ward"]: f["properties"]["total_requests_12_months"]
        for f in snapshot.wards["features"]
    }
    assert totals == {1: 120, 2: 3}
    assert snapshot.request_types == [{"type": "Noise", "category": "X", "count": 120}]
    assert snapshot.end == pd.Timestamp("2026-10-30")


def test_city_report_card_covers_every_ward_and_spike() -> None:
    data = client.get("/city").json()
    assert data["name"] == "All of Ottawa"
    assert data["ward"] is None
    assert data["total_requests"] == 123
    assert len(data["spikes"]) == 1


def test_ward_metrics_cover_every_ward_and_filter_by_type() -> None:
    data = client.get("/ward-metrics").json()
    assert data["wards"] == [
        {"ward": 1, "requests": 120, "median_days": 1.0, "spikes": 1},
        {"ward": 2, "requests": 3, "median_days": None, "spikes": 0},
    ]
    noise = client.get("/ward-metrics", params={"type": "Noise"}).json()
    assert noise["wards"][1] == {"ward": 2, "requests": 0, "median_days": None, "spikes": 0}
    assert client.get("/ward-metrics", params={"type": "Pothole"}).status_code == 404


def test_remaining_reports_thin_history_and_validates_input() -> None:
    """The fake snapshot has no requests old enough to count, so there is no estimate."""
    data = client.get("/remaining", params={"type": "Noise", "ward": 1, "waited": 10}).json()
    assert data["enough_history"] is False
    assert data["usual_more_days"] is None
    no_wait = {"type": "Noise", "ward": 1, "waited": 0}
    rare_type = {"type": "Pothole", "ward": 1, "waited": 5}
    assert client.get("/remaining", params=no_wait).status_code == 422
    assert client.get("/remaining", params=rare_type).status_code == 404


def test_estimate_for_batch_closed_type_comes_from_history_not_the_model() -> None:
    recent = pd.DataFrame(
        {
            "opened": pd.Timestamp("2026-06-01"),
            "type": "Pothole",
            "ward": pd.array([1] * 40, dtype="Int64"),
            "status": "Resolved",
            "days_to_close": 4.0,
        }
    )
    snapshot = fake_snapshot()
    snapshot.request_types.append({"type": "Pothole", "category": "Roads", "count": 400})
    app.dependency_overrides[get_snapshot] = lambda: Snapshot(
        snapshot.requests,
        snapshot.end,
        snapshot.batch_types,
        snapshot.request_types,
        snapshot.spikes,
        snapshot.wards,
        timed=recent,
    )
    data = client.get("/estimate", params={"type": "Pothole", "ward": 1}).json()
    assert data["batch_closed"] is True
    assert (data["usual_days"], data["nine_in_ten_days"]) == (4.0, 4.0)  # the fake model says 3
    # Types with trustworthy history still use the model.
    app.dependency_overrides[get_snapshot] = lambda: fake_snapshot()
    assert client.get("/estimate", params={"type": "Noise", "ward": 2}).json()["usual_days"] == 3.0
