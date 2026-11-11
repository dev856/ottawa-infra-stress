"""Tests for the request closing-time model."""

import numpy as np
import pandas as pd
import pytest

from src.errors import DataValidationError
from src.train_model import (
    MATURITY_DAYS,
    TEST_DAYS,
    evaluate,
    load_requests,
    model_inputs,
    split_by_date,
    train,
    type_lookup,
)


def make_requests(days: int = 400, per_day: int = 30) -> pd.DataFrame:
    """Return requests where noise closes in ~1 day and potholes in ~30, slower in ward 2."""
    rng = np.random.default_rng(0)
    rows = []
    for day in pd.date_range("2025-01-01", periods=days, freq="D"):
        for _ in range(per_day):
            request_type = rng.choice(["Noise", "Pothole"])
            ward = int(rng.integers(1, 3))
            mean = (1 if request_type == "Noise" else 30) * (2 if ward == 2 else 1)
            rows.append(
                {
                    "opened": day,
                    "closed": day + pd.Timedelta(days=int(rng.poisson(mean))),
                    "type": request_type,
                    "ward": ward,
                    "channel": "Web",
                    "status": "Resolved",
                }
            )
    requests = pd.DataFrame(rows)
    requests["days_to_close"] = (requests["closed"] - requests["opened"]).dt.days
    return requests


def test_load_requests_adds_days_to_close(tmp_path) -> None:
    path = tmp_path / "requests.csv.gz"
    pd.DataFrame(
        {
            "opened": ["2025-01-01", "2025-01-02"],
            "closed": ["2025-01-04", None],
            "category": ["A", "B"],
            "type": ["X", "Y"],
            "ward": [1, None],
            "channel": ["Web", "Phone"],
            "status": ["Resolved", "Active"],
        }
    ).to_csv(path, index=False)

    requests = load_requests(path)

    assert requests["days_to_close"].iloc[0] == 3
    assert pd.isna(requests["days_to_close"].iloc[1])


def test_split_holds_out_mature_recent_requests_and_keeps_open_ones_in_test() -> None:
    """Too-recent requests are dropped; still-open mature ones stay in test only."""
    requests = make_requests()
    end = requests["opened"].max()
    open_day = end - pd.Timedelta(days=MATURITY_DAYS + 5)
    requests.loc[requests["opened"] == open_day, ["closed", "days_to_close"]] = [pd.NaT, np.nan]
    requests.loc[requests.index[:5], "status"] = "Cancelled"

    train_rows, test_rows = split_by_date(requests)

    assert train_rows["days_to_close"].notna().all()
    assert test_rows["days_to_close"].isna().any()
    assert test_rows["opened"].max() <= end - pd.Timedelta(days=MATURITY_DAYS)
    assert test_rows["opened"].min() >= end - pd.Timedelta(days=MATURITY_DAYS + TEST_DAYS)
    assert (train_rows["status"] != "Cancelled").all()


def test_split_rejects_too_little_history() -> None:
    with pytest.raises(DataValidationError, match="Not enough history"):
        split_by_date(make_requests(days=100))


def test_unseen_categories_become_missing_not_errors() -> None:
    inputs = model_inputs(
        pd.DataFrame(
            {
                "type": ["New"],
                "ward": [9],
                "channel": ["Fax"],
                "opened": [pd.Timestamp("2025-01-06")],
            }
        ),
        {"type": ["Noise"], "ward": ["1"], "channel": ["Web"]},
    )
    assert inputs[["type", "ward", "channel"]].isna().all(axis=None)
    assert inputs["weekday"].iloc[0] == 0


def test_model_learns_type_and_ward_and_is_calibrated() -> None:
    """The model separates fast and slow types, sees the slower ward, and its 90th holds."""
    train_rows, test_rows = split_by_date(make_requests())
    models, categories = train(train_rows)
    metrics = evaluate(models, train_rows, test_rows, categories)

    def usual(request_type: str, ward: int) -> float:
        frame = pd.DataFrame(
            {
                "type": [request_type],
                "ward": [ward],
                "channel": ["Web"],
                "opened": [pd.Timestamp("2025-06-02")],
            }
        )
        return float(models["usual_days"].predict(model_inputs(frame, categories))[0])

    assert usual("Pothole", 1) > 10 * usual("Noise", 1)
    assert usual("Pothole", 2) > 1.5 * usual("Pothole", 1)
    assert 0.85 <= metrics["nine_in_ten_days"]["share_closed_within"] <= 0.95
    # Ward matters here, so the model must beat a lookup that ignores it.
    assert metrics["usual_days"]["pinball_loss"] < metrics["usual_days"]["pinball_loss_type_lookup"]


def test_type_lookup_falls_back_for_new_types() -> None:
    train_rows = pd.DataFrame({"type": ["A", "A", "B"], "days_to_close": [1.0, 3.0, 10.0]})
    test_rows = pd.DataFrame({"type": ["A", "C"]})

    lookup = type_lookup(train_rows, test_rows, 0.5)

    assert lookup.tolist() == [2.0, 3.0]
