"""Tests for temporal model training and artifact behavior."""

import numpy as np
import pandas as pd
import pytest

from src.errors import DataValidationError
from src.train_model import (
    TARGET_COLUMN,
    assemble_feature_matrix,
    create_target_variable,
    evaluate_model,
    get_feature_columns,
    select_decision_threshold,
    temporal_train_test_split,
    train_lightgbm,
)


def make_temporal_dataset(days: int = 30, cells: int = 4) -> pd.DataFrame:
    """Create deterministic multi-cell daily examples with both synthetic classes."""
    records: list[dict[str, object]] = []
    for day in range(days):
        for cell in range(cells):
            records.append(
                {
                    "timestamp": pd.Timestamp("2024-05-01") + pd.Timedelta(days=day),
                    "h3_index": f"cell-{cell}",
                    "line_length_km_water": float(cell + 1),
                    "line_length_km_road": float(cell + 2),
                    "building_count": 10 + cell,
                    "pct_pre_1980": 0.2 + cell * 0.2,
                    "heat_index_c": 24.0 + (day % 10),
                    "consecutive_dry_days": float(day % 6),
                    "rainfall_48h": 0.0 if day % 5 else 5.0,
                }
            )
    return pd.DataFrame(records)


def test_assemble_feature_matrix_is_temporal_cross_join() -> None:
    """Every H3 cell receives every daily weather snapshot."""
    infrastructure = pd.DataFrame(
        {
            "h3_index": ["a", "b"],
            "centroid_lat": [45.4, 45.5],
            "centroid_lon": [-75.7, -75.8],
            "line_length_km_water": [1.0, 2.0],
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "heat_index_c": np.arange(10, dtype=float),
        }
    )

    result = assemble_feature_matrix(infrastructure, weather)

    assert len(result) == 20
    assert result["timestamp"].nunique() == 10


def test_create_target_variable_is_explicitly_binary() -> None:
    """Synthetic mode creates a deterministic binary label."""
    result = create_target_variable(make_temporal_dataset(), mode="synthetic")
    assert set(result[TARGET_COLUMN].unique()) == {0, 1}


def test_observed_mode_requires_joined_outcomes() -> None:
    """Observed mode cannot silently create an all-negative target."""
    with pytest.raises(DataValidationError, match="pre-joined"):
        create_target_variable(make_temporal_dataset(), mode="observed")


def test_get_feature_columns_excludes_identifiers_and_time() -> None:
    """Identifiers, timestamp, and target do not become model inputs."""
    dataframe = create_target_variable(make_temporal_dataset())
    columns = get_feature_columns(dataframe)
    assert "h3_index" not in columns
    assert "timestamp" not in columns
    assert TARGET_COLUMN not in columns
    assert "line_length_km_water" in columns


def test_temporal_split_has_prediction_horizon_gaps() -> None:
    """Validation and test start at least 48 hours after preceding partitions."""
    dataframe = create_target_variable(make_temporal_dataset())
    feature_columns = get_feature_columns(dataframe)
    split = temporal_train_test_split(dataframe, feature_columns, purge_hours=48)
    x_train, x_validation, x_test, *_ = split

    train_times = dataframe.loc[x_train.index, "timestamp"]
    validation_times = dataframe.loc[x_validation.index, "timestamp"]
    test_times = dataframe.loc[x_test.index, "timestamp"]
    assert validation_times.min() - train_times.max() >= pd.Timedelta(hours=48)
    assert test_times.min() - validation_times.max() >= pd.Timedelta(hours=48)


def test_training_threshold_and_test_evaluation() -> None:
    """Threshold selection uses validation before final test evaluation."""
    dataframe = create_target_variable(make_temporal_dataset(days=40, cells=6))
    feature_columns = get_feature_columns(dataframe)
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        temporal_train_test_split(dataframe, feature_columns)
    )
    model = train_lightgbm(x_train, y_train, x_validation, y_validation)
    threshold = select_decision_threshold(model, x_validation, y_validation)
    metrics = evaluate_model(model, x_test, y_test, threshold)

    assert 0 <= threshold <= 1
    assert metrics["decision_threshold"] == threshold
    assert 0 <= metrics["pr_auc"] <= 1


def test_temporal_split_rejects_missing_timestamp() -> None:
    """Sorting by H3 is not accepted as a temporal split."""
    dataframe = create_target_variable(make_temporal_dataset().drop(columns="timestamp"))
    with pytest.raises(DataValidationError, match="timestamp"):
        temporal_train_test_split(dataframe, get_feature_columns(dataframe))
