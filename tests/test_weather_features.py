"""Tests for weather feature calculation."""

import numpy as np
import pandas as pd
import pytest

from src.errors import DataValidationError
from src.fetch_weather_features import (
    calculate_heat_index_c,
    calculate_weather_features,
    heat_index,
)


def test_heat_index_returns_temperature_below_threshold():
    """Test heat index returns temperature when below threshold."""
    assert heat_index(20.0, 50.0) == 20.0


def test_heat_index_rothfusz_above_threshold():
    """Test heat index calculates Rothfusz regression above 80°F (~26.7°C)."""
    # 32°C at 70% humidity produces an elevated heat index
    hi = heat_index(32.0, 70.0)
    assert hi > 32.0


def test_heat_index_missing_inputs():
    """Test heat index returns NaN for missing values."""
    assert np.isnan(heat_index(None, 50.0))
    assert np.isnan(heat_index(30.0, None))
    assert np.isnan(heat_index(np.nan, 50.0))


def test_calculate_weather_features_adds_columns():
    """Test weather feature calculation adds expected columns."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=48, freq="h"),
        "temperature_c": np.arange(48, dtype=float),
        "relative_humidity": np.full(48, 50.0),
        "precip_mm": np.zeros(48),
    })

    result = calculate_weather_features(df)

    expected_cols = {
        "temp_delta_24h",
        "temp_delta_48h",
        "temp_max_24h",
        "temp_max_48h",
        "temp_min_24h",
        "heat_index_c",
        "consecutive_dry_hours",
        "consecutive_dry_days",
        "rainfall_24h",
        "rainfall_48h",
        "rainfall_7d",
        "humidity_delta_24h",
        "humidity_min_24h",
    }

    assert expected_cols.issubset(result.columns)
    assert len(result) == len(df)


def test_calculate_weather_features_dry_streak_resets_on_rain():
    """Test consecutive dry hours increments during dry hours and resets on rain."""
    precip = [0.0] * 5 + [2.5] + [0.0] * 3
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(precip), freq="h"),
        "temperature_c": np.full(len(precip), 25.0),
        "relative_humidity": np.full(len(precip), 50.0),
        "precip_mm": precip,
    })

    res = calculate_weather_features(df)
    dry_hours = res["consecutive_dry_hours"].tolist()
    assert dry_hours[:5] == [1, 2, 3, 4, 5]
    assert dry_hours[5] == 0
    assert dry_hours[6:] == [1, 2, 3]


def test_calculate_weather_features_empty():
    """Test empty dataframe returns empty dataframe."""
    res = calculate_weather_features(pd.DataFrame())
    assert res.empty


def test_calculate_heat_index_rejects_invalid_humidity():
    """Physically invalid humidity fails instead of producing a model feature."""
    with pytest.raises(DataValidationError, match="relative_humidity"):
        calculate_heat_index_c(30.0, 120.0)


def test_weather_features_use_elapsed_time_for_gaps():
    """A missing hour resets dry streaks and does not shift row-count deltas."""
    timestamps = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 03:00"]
    )
    dataframe = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_c": [20.0, 21.0, 23.0],
            "relative_humidity": [50.0, 50.0, 50.0],
            "precip_mm": [0.0, 0.0, 0.0],
        }
    )

    result = calculate_weather_features(dataframe)

    assert result["consecutive_dry_hours"].tolist() == [1, 2, 1]


def test_weather_features_reject_duplicate_timestamps():
    """Duplicate observations are ambiguous and must be resolved by ingestion."""
    dataframe = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")] * 2,
            "temperature_c": [20.0, 21.0],
            "relative_humidity": [50.0, 50.0],
            "precip_mm": [0.0, 0.0],
        }
    )

    with pytest.raises(DataValidationError, match="duplicate"):
        calculate_weather_features(dataframe)
