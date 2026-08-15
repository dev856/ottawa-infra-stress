"""Tests for the ECCC historic weather client."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import eccc_client
from src.errors import DataValidationError, ExternalServiceError

# A realistic (truncated) ECCC CSV response. Note the leading BOM.
SAMPLE_CSV = (
    "\ufeff"
    '"Longitude (x)","Latitude (y)","Station Name","Climate ID",'
    '"Date/Time (LST)","Year","Month","Day","Time (LST)","Flag","Temp (°C)",'
    '"Temp Flag","Dew Point Temp (°C)","Dew Point Temp Flag","Rel Hum (%)",'
    '"Rel Hum Flag","Precip. Amount (mm)","Precip. Amount Flag",'
    '"Wind Dir (10s deg)","Wind Dir Flag","Wind Spd (km/h)","Wind Spd Flag",'
    '"Visibility (km)","Visibility Flag","Stn Press (kPa)","Stn Press Flag",'
    '"Hmdx","Hmdx Flag","Wind Chill","Wind Chill Flag","Weather"\r\n'
    '"-75.67","45.32","OTTAWA INTL A","6106001","2023-07-01 00:00","2023","07","01","00:00","",'
    '"19.7","","17.8","","89","","","","11","","13","","6.4","","100.04","",'
    '"","","","","Smoke"\r\n'
    '"-75.67","45.32","OTTAWA INTL A","6106001","2023-07-01 01:00","2023","07","01","01:00","",'
    '"","","17.9","","91","","","","13","","13","","6.4","","100.01","",'
    '"","","","",""\r\n'
)


def test_parse_csv_text_maps_columns():
    """ECCC columns are renamed to our internal schema."""
    df = eccc_client._parse_csv_text(SAMPLE_CSV, "49568")

    assert list(df.columns) == [
        "timestamp",
        "station_id",
        "climate_id",
        "station_name",
        "temperature_c",
        "dew_point_c",
        "relative_humidity",
        "precip_mm",
        "wind_speed_kmh",
        "pressure_kpa",
    ]
    assert len(df) == 2
    assert df["temperature_c"].iloc[0] == pytest.approx(19.7)
    # Empty cell should become NaN, not 0 or a string.
    assert pd.isna(df["temperature_c"].iloc[1])
    assert df["relative_humidity"].iloc[0] == 89
    # The BOM must not corrupt the timestamp column.
    assert df["timestamp"].iloc[0] == pd.Timestamp("2023-07-01 00:00")


def test_parse_csv_text_empty():
    """An empty CSV returns an empty dataframe with the expected columns."""
    df = eccc_client._parse_csv_text("", "49568")
    assert df.empty
    assert "timestamp" in df.columns
    assert "temperature_c" in df.columns


def test_monthly_periods_inclusive():
    """_monthly_periods covers every month between the bounds inclusive."""
    periods = eccc_client._monthly_periods("2023-12-01", "2024-02-15")
    assert periods == [(2023, 12), (2024, 1), (2024, 2)]


def test_monthly_periods_invalid_range():
    """end before start raises ValueError."""
    with pytest.raises(DataValidationError):
        eccc_client._monthly_periods("2024-02-01", "2024-01-01")


def test_build_url_format():
    """The download URL contains the station id, year and month."""
    url = eccc_client._build_url("49568", 2023, 7)
    assert "stationID=49568" in url
    assert "Year=2023" in url
    assert "Month=7" in url
    assert "timeframe=1" in url


def test_fetch_weather_data_uses_months_override(monkeypatch, tmp_path):
    """fetch_weather_data honors an explicit months list and caches to disk."""
    # Redirect the cache directory to a temp dir so the test does not pollute
    # the real data dir.
    monkeypatch.setattr(eccc_client, "WEATHER_CACHE_DIR", tmp_path)

    def fake_download(station_id, year, month, **kwargs):
        return SAMPLE_CSV.encode("utf-8")

    monkeypatch.setattr(eccc_client, "_download_month", fake_download)

    df = eccc_client.fetch_weather_data(months=[(2023, 7)], mode="live")

    assert len(df) == 2
    assert df["station_id"].iloc[0] == "49568"
    # Cache file should have been written.
    cache_file = tmp_path / "49568_2023_07.csv"
    assert cache_file.exists()


def test_fetch_weather_data_uses_cache(monkeypatch, tmp_path):
    """A cached CSV is reused instead of re-downloading."""
    monkeypatch.setattr(eccc_client, "WEATHER_CACHE_DIR", tmp_path)
    (tmp_path / "49568_2023_07.csv").write_bytes(SAMPLE_CSV.encode("utf-8"))

    def boom(*args, **kwargs):
        raise AssertionError("should not download when cache exists")

    monkeypatch.setattr(eccc_client, "_download_month", boom)

    df = eccc_client.fetch_weather_data(months=[(2023, 7)], mode="live")
    assert len(df) == 2


def test_fetch_weather_data_force_redownloads(monkeypatch, tmp_path):
    """force=True bypasses the cache and re-downloads."""
    monkeypatch.setattr(eccc_client, "WEATHER_CACHE_DIR", tmp_path)
    (tmp_path / "49568_2023_07.csv").write_bytes(b"stale")

    calls = {"n": 0}

    def fake_download(station_id, year, month, **kwargs):
        calls["n"] += 1
        return SAMPLE_CSV.encode("utf-8")

    monkeypatch.setattr(eccc_client, "_download_month", fake_download)

    df = eccc_client.fetch_weather_data(months=[(2023, 7)], force=True, mode="live")
    assert calls["n"] == 1
    assert len(df) == 2


def test_fetch_weather_data_rejects_incomplete_result(monkeypatch, tmp_path):
    """A failed month aborts by default instead of silently changing the dataset."""
    monkeypatch.setattr(eccc_client, "WEATHER_CACHE_DIR", tmp_path)

    def fake_download(station_id, year, month, **kwargs):
        if year == 2023 and month == 6:
            raise ExternalServiceError("boom")
        return SAMPLE_CSV.encode("utf-8")

    monkeypatch.setattr(eccc_client, "_download_month", fake_download)

    with pytest.raises(ExternalServiceError, match="incomplete"):
        eccc_client.fetch_weather_data(
            months=[(2023, 6), (2023, 7)],
            mode="live",
        )


def test_fetch_weather_data_allows_explicit_partial_result(monkeypatch, tmp_path):
    """A caller must opt in before successful months can hide a failed month."""
    monkeypatch.setattr(eccc_client, "WEATHER_CACHE_DIR", tmp_path)

    def fake_download(station_id, year, month, **kwargs):
        if month == 6:
            raise ExternalServiceError("boom")
        return SAMPLE_CSV.encode("utf-8")

    monkeypatch.setattr(eccc_client, "_download_month", fake_download)

    result = eccc_client.fetch_weather_data(
        months=[(2023, 6), (2023, 7)],
        mode="live",
        allow_partial=True,
    )

    assert len(result) == 2


def test_fetch_weather_data_dedupes_timestamps(monkeypatch, tmp_path):
    """Duplicate timestamps across months are collapsed to one row."""
    monkeypatch.setattr(eccc_client, "WEATHER_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        eccc_client,
        "_download_month",
        lambda *a, **k: SAMPLE_CSV.encode("utf-8"),
    )
    df = eccc_client.fetch_weather_data(
        months=[(2023, 7), (2023, 7)],
        mode="live",
    )
    assert len(df) == 2


def test_mock_mode_never_downloads(monkeypatch, tmp_path):
    """The default-safe mode reads one fixture and makes no HTTP call."""
    fixture = tmp_path / "weather.csv"
    fixture.write_text(SAMPLE_CSV, encoding="utf-8")
    monkeypatch.setattr(
        eccc_client,
        "_download_month",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    result = eccc_client.fetch_weather_data(mode="mock", fixture_path=fixture)

    assert len(result) == 2


def test_default_mock_weather_is_long_enough_for_temporal_tests(monkeypatch):
    """Offline training receives many chronological observations without HTTP."""
    monkeypatch.setattr(
        eccc_client,
        "_download_month",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network called")),
    )

    result = eccc_client.fetch_weather_data(mode="mock")

    assert len(result) > 24 * 30
    assert result["timestamp"].is_monotonic_increasing


def test_station_id_cannot_escape_cache_directory():
    """Station IDs are constrained before they become cache filenames."""
    with pytest.raises(DataValidationError, match="station_id"):
        eccc_client._cache_path("../secret", 2023, 7)


def test_missing_required_column_is_rejected():
    """A changed ECCC schema fails clearly instead of filling a required field."""
    malformed = '"Date/Time (LST)","Temp (°C)"\n"2023-07-01 00:00","20"\n'
    with pytest.raises(DataValidationError, match="missing required columns"):
        eccc_client._parse_csv_text(malformed, "49568")
