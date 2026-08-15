"""Fetch, cache, and normalize ECCC hourly weather CSV data.

Mock mode reads one checked-in fixture without network access. Approved live use
adds finite timeouts/retries, a named user agent, safe cache filenames, atomic
writes, required-column validation, and explicit partial-result behavior.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal, cast
from urllib.parse import urlencode

import pandas as pd
import requests

if TYPE_CHECKING:
    from .config import (
        DATA_SOURCE_MODE,
        DEFAULT_HTTP_MAX_RETRIES,
        DEFAULT_HTTP_TIMEOUT_SECONDS,
        DEFAULT_USER_AGENT,
        ECCC_BASE_URL,
        ECCC_STATION_ID,
        ECCC_WEATHER_END_DATE,
        ECCC_WEATHER_START_DATE,
        FIXTURES_DIR,
        WEATHER_CACHE_DIR,
    )
    from .errors import DataValidationError, ExternalServiceError
    from .mock_data_sources import generate_mock_weather_observations
else:
    try:
        from .config import (
            DATA_SOURCE_MODE,
            DEFAULT_HTTP_MAX_RETRIES,
            DEFAULT_HTTP_TIMEOUT_SECONDS,
            DEFAULT_USER_AGENT,
            ECCC_BASE_URL,
            ECCC_STATION_ID,
            ECCC_WEATHER_END_DATE,
            ECCC_WEATHER_START_DATE,
            FIXTURES_DIR,
            WEATHER_CACHE_DIR,
        )
        from .errors import DataValidationError, ExternalServiceError
        from .mock_data_sources import generate_mock_weather_observations
    except ImportError:  # pragma: no cover - supports direct script imports
        from config import (
            DATA_SOURCE_MODE,
            DEFAULT_HTTP_MAX_RETRIES,
            DEFAULT_HTTP_TIMEOUT_SECONDS,
            DEFAULT_USER_AGENT,
            ECCC_BASE_URL,
            ECCC_STATION_ID,
            ECCC_WEATHER_END_DATE,
            ECCC_WEATHER_START_DATE,
            FIXTURES_DIR,
            WEATHER_CACHE_DIR,
        )
        from errors import DataValidationError, ExternalServiceError
        from mock_data_sources import generate_mock_weather_observations

logger = logging.getLogger(__name__)

ClientMode = Literal["mock", "hybrid", "live"]

ECCC_COLUMN_MAP = {
    "Date/Time (LST)": "timestamp",
    "Climate ID": "climate_id",
    "Station Name": "station_name",
    "Temp (°C)": "temperature_c",
    "Dew Point Temp (°C)": "dew_point_c",
    "Rel Hum (%)": "relative_humidity",
    "Precip. Amount (mm)": "precip_mm",
    "Wind Spd (km/h)": "wind_speed_kmh",
    "Stn Press (kPa)": "pressure_kpa",
}

REQUIRED_SOURCE_COLUMNS = {
    "Date/Time (LST)",
    "Temp (°C)",
    "Rel Hum (%)",
    "Precip. Amount (mm)",
}
NUMERIC_FIELDS = (
    "temperature_c",
    "dew_point_c",
    "relative_humidity",
    "precip_mm",
    "wind_speed_kmh",
    "pressure_kpa",
)
OUTPUT_COLUMNS = (
    "timestamp",
    "station_id",
    "climate_id",
    "station_name",
    *NUMERIC_FIELDS,
)


def _validate_station_id(station_id: str) -> str:
    """Restrict station IDs used in query parameters and cache filenames."""
    normalized = station_id.strip()
    if not normalized.isascii() or not normalized.isdigit():
        raise DataValidationError("ECCC station_id must contain ASCII digits only")
    return normalized


def _monthly_periods(start_date: str, end_date: str) -> list[tuple[int, int]]:
    """Return every calendar month between two inclusive ISO dates."""
    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("ECCC date range must use valid ISO dates") from exc
    if end < start:
        raise DataValidationError("ECCC end_date must be on or after start_date")

    periods: list[tuple[int, int]] = []
    current = start.to_period("M")
    last = end.to_period("M")
    while current <= last:
        periods.append((current.year, current.month))
        current += 1
    return periods


def _build_url(station_id: str, year: int, month: int) -> str:
    """Build an encoded ECCC bulk hourly CSV URL."""
    station = _validate_station_id(station_id)
    if not 1 <= month <= 12 or not 1900 <= year <= 2100:
        raise DataValidationError("ECCC year/month is outside the supported range")
    query = urlencode(
        {
            "format": "csv",
            "stationID": station,
            "Year": year,
            "Month": month,
            "Day": 14,
            "timeframe": 1,
            "submit": "Download Data",
        }
    )
    return f"{ECCC_BASE_URL}?{query}"


def _download_month(
    station_id: str,
    year: int,
    month: int,
    *,
    timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
    retries: int = DEFAULT_HTTP_MAX_RETRIES,
    backoff_seconds: float = 0.5,
    session: requests.Session | None = None,
) -> bytes:
    """Download one month with a finite number of conservative retries."""
    url = _build_url(station_id, year, month)
    http = session or requests.Session()
    attempts = retries + 1

    for attempt in range(1, attempts + 1):
        try:
            response = http.get(
                url,
                headers={"Accept": "text/csv", "User-Agent": DEFAULT_USER_AGENT},
                timeout=timeout,
            )
            response.raise_for_status()
            if not response.content:
                raise DataValidationError("ECCC returned an empty CSV response")
            return response.content
        except DataValidationError:
            raise
        except requests.RequestException as exc:
            if attempt == attempts:
                raise ExternalServiceError(
                    f"ECCC download failed for {year}-{month:02d}"
                ) from exc
            logger.warning(
                "ECCC download attempt %s/%s failed for %s-%02d",
                attempt,
                attempts,
                year,
                month,
            )
            time.sleep(backoff_seconds * attempt)
    raise AssertionError("unreachable")


def _cache_path(station_id: str, year: int, month: int) -> Path:
    """Return a cache path under the dedicated weather-cache directory."""
    station = _validate_station_id(station_id)
    if not 1 <= month <= 12:
        raise DataValidationError("ECCC month must be between 1 and 12")
    return WEATHER_CACHE_DIR / f"{station}_{year}_{month:02d}.csv"


def _normalize_columns(dataframe: pd.DataFrame, station_id: str) -> pd.DataFrame:
    """Rename required ECCC fields and coerce calculations to numeric values."""
    missing_columns = REQUIRED_SOURCE_COLUMNS - set(dataframe.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"ECCC CSV is missing required columns: {names}")

    normalized = dataframe.rename(columns=ECCC_COLUMN_MAP).copy()
    for field in NUMERIC_FIELDS:
        if field in normalized.columns:
            normalized[field] = pd.to_numeric(normalized[field], errors="coerce")
        else:
            normalized[field] = pd.NA

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    normalized = normalized.dropna(subset=["timestamp"])
    normalized["station_id"] = station_id
    return normalized[list(OUTPUT_COLUMNS)]


def _parse_csv_text(text: str | bytes, station_id: str) -> pd.DataFrame:
    """Parse one UTF-8 ECCC CSV response into the stable internal schema."""
    station = _validate_station_id(station_id)
    decoded = text.decode("utf-8-sig") if isinstance(text, bytes) else text.lstrip("\ufeff")
    if not decoded.strip():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    reader = csv.DictReader(io.StringIO(decoded))
    if reader.fieldnames is None:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    rows = list(reader)
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return _normalize_columns(pd.DataFrame(rows), station)


def _write_cache_atomically(cache_file: Path, content: bytes) -> None:
    """Replace a cache file only after the full response is written."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = cache_file.with_suffix(".csv.tmp")
    try:
        temporary_file.write_bytes(content)
        temporary_file.replace(cache_file)
    finally:
        if temporary_file.exists():
            temporary_file.unlink()


def fetch_monthly_data(
    station_id: str,
    year: int,
    month: int,
    force: bool = False,
) -> pd.DataFrame:
    """Load one live-approved month from cache or the ECCC endpoint."""
    cache_file = _cache_path(station_id, year, month)
    if cache_file.is_file() and cache_file.stat().st_size > 0 and not force:
        logger.info("Using cached ECCC data for %s-%02d", year, month)
        return _parse_csv_text(cache_file.read_bytes(), station_id)

    logger.info("Downloading ECCC data for %s-%02d", year, month)
    content = _download_month(station_id, year, month)
    parsed = _parse_csv_text(content, station_id)
    _write_cache_atomically(cache_file, content)
    return parsed


def _load_mock_weather(fixture_path: Path, station_id: str) -> pd.DataFrame:
    """Load the deterministic offline weather fixture exactly once."""
    try:
        content = fixture_path.read_bytes()
    except OSError as exc:
        raise DataValidationError(f"ECCC fixture cannot be read: {fixture_path.name}") from exc
    return _parse_csv_text(content, station_id)


def fetch_weather_data(
    station_id: str = ECCC_STATION_ID,
    start_date: str = ECCC_WEATHER_START_DATE,
    end_date: str = ECCC_WEATHER_END_DATE,
    force: bool = False,
    months: Iterable[tuple[int, int]] | None = None,
    *,
    mode: str | None = None,
    fixture_path: Path | None = None,
    allow_partial: bool = False,
) -> pd.DataFrame:
    """Return normalized weather rows for the requested period.

    ``allow_partial`` is false by default so a missing live month cannot silently
    change the training population. Hybrid mode falls back to the local fixture
    only when the live operation fails.
    """
    station = _validate_station_id(station_id)
    selected_mode = mode or DATA_SOURCE_MODE
    if selected_mode not in {"mock", "hybrid", "live"}:
        raise ValueError("mode must be one of: mock, hybrid, live")
    typed_mode = cast(ClientMode, selected_mode)
    if typed_mode == "mock":
        if fixture_path is not None:
            return _load_mock_weather(fixture_path, station)
        return generate_mock_weather_observations(station_id=station)

    selected_fixture = fixture_path or FIXTURES_DIR / "eccc_monthly_sample.csv"

    periods = list(months) if months is not None else _monthly_periods(start_date, end_date)
    if not periods:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    frames: list[pd.DataFrame] = []
    failures: list[tuple[int, int]] = []
    for year, month in periods:
        try:
            frames.append(fetch_monthly_data(station, year, month, force=force))
        except (ExternalServiceError, DataValidationError, OSError):
            failures.append((year, month))
            logger.error("ECCC month failed: %s-%02d", year, month)

    if failures and typed_mode == "hybrid":
        logger.warning("ECCC live operation failed; using the configured local fixture")
        return _load_mock_weather(selected_fixture, station)
    if failures and not allow_partial:
        raise ExternalServiceError(
            f"ECCC data is incomplete: {len(failures)} requested month(s) failed"
        )
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
    return combined.reset_index(drop=True)
