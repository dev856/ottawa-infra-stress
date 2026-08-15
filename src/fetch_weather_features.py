"""Calculate weather features from normalized ECCC or mock hourly observations.

Pure calculations are separate from source I/O. Rolling windows use elapsed time
rather than row counts, and missing-hour gaps reset consecutive dry-hour streaks.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .config import (
        DATA_DIR,
        ECCC_STATION_ID,
        ECCC_WEATHER_END_DATE,
        ECCC_WEATHER_START_DATE,
        SETTINGS,
        ensure_directories,
    )
    from .eccc_client import fetch_weather_data as _fetch_eccc_data
    from .errors import DataValidationError
else:
    try:
        from .config import (
            DATA_DIR,
            ECCC_STATION_ID,
            ECCC_WEATHER_END_DATE,
            ECCC_WEATHER_START_DATE,
            SETTINGS,
            ensure_directories,
        )
        from .eccc_client import fetch_weather_data as _fetch_eccc_data
        from .errors import DataValidationError
    except ImportError:  # pragma: no cover - supports direct script imports
        from config import (
            DATA_DIR,
            ECCC_STATION_ID,
            ECCC_WEATHER_END_DATE,
            ECCC_WEATHER_START_DATE,
            SETTINGS,
            ensure_directories,
        )
        from eccc_client import fetch_weather_data as _fetch_eccc_data
        from errors import DataValidationError

logger = logging.getLogger(__name__)

DRY_HOUR_PRECIP_THRESHOLD_MM = 0.2
REQUIRED_WEATHER_COLUMNS = {
    "timestamp",
    "temperature_c",
    "relative_humidity",
    "precip_mm",
}


def calculate_heat_index_c(
    temperature_c: float | None,
    relative_humidity: float | None,
) -> float:
    """Calculate the Rothfusz heat index in Celsius.

    Below 26.7 Celsius the function returns the air temperature. Missing inputs
    return NaN because guessing humidity or temperature would distort features.
    """
    if temperature_c is None or relative_humidity is None:
        return math.nan
    if pd.isna(temperature_c) or pd.isna(relative_humidity):
        return math.nan
    if not 0 <= relative_humidity <= 100:
        raise DataValidationError("relative_humidity must be between 0 and 100")

    temperature_f = temperature_c * 9 / 5 + 32
    if temperature_f < 80:
        return float(temperature_c)

    humidity = relative_humidity
    heat_index_f = (
        -42.379
        + 2.04901523 * temperature_f
        + 10.14333127 * humidity
        - 0.22475541 * temperature_f * humidity
        - 0.00683783 * temperature_f**2
        - 0.05481717 * humidity**2
        + 0.00122874 * temperature_f**2 * humidity
        + 0.00085282 * temperature_f * humidity**2
        - 0.00000199 * temperature_f**2 * humidity**2
    )
    return float((heat_index_f - 32) * 5 / 9)


def heat_index(temperature_c: float | None, relative_humidity: float | None) -> float:
    """Compatibility alias for ``calculate_heat_index_c``."""
    return calculate_heat_index_c(temperature_c, relative_humidity)


def validate_weather_observations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return sorted weather observations after schema and range validation."""
    missing_columns = REQUIRED_WEATHER_COLUMNS - set(dataframe.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Weather data is missing required columns: {names}")

    validated = dataframe.copy()
    validated["timestamp"] = pd.to_datetime(validated["timestamp"], errors="coerce")
    if validated["timestamp"].isna().any():
        raise DataValidationError("Weather data contains an invalid timestamp")
    if validated["timestamp"].duplicated().any():
        raise DataValidationError("Weather data contains duplicate timestamps")

    for column in ("temperature_c", "relative_humidity", "precip_mm"):
        validated[column] = pd.to_numeric(validated[column], errors="coerce")
    humidity = validated["relative_humidity"].dropna()
    if not humidity.between(0, 100).all():
        raise DataValidationError("Weather humidity values must be between 0 and 100")
    precipitation = validated["precip_mm"].dropna()
    if (precipitation < 0).any():
        raise DataValidationError("Weather precipitation values cannot be negative")

    return validated.sort_values("timestamp").reset_index(drop=True)


def _elapsed_delta(series: pd.Series, hours: int) -> pd.Series:
    """Subtract the observation at the exact elapsed-hour offset when present."""
    shifted = series.shift(freq=f"{hours}h")
    return series - shifted


def _consecutive_dry_hours(index: pd.DatetimeIndex, precipitation: pd.Series) -> pd.Series:
    """Count dry hourly observations and reset at rain, missing data, or time gaps."""
    dry = precipitation.notna() & (precipitation < DRY_HOUR_PRECIP_THRESHOLD_MM)
    gap = index.to_series().diff().ne(pd.Timedelta(hours=1)).to_numpy()
    reset = (~dry).to_numpy() | gap
    group = pd.Series(reset, index=index).cumsum()
    streak = dry.astype(int).groupby(group).cumsum()
    return streak.where(dry, 0).astype(int)


def calculate_weather_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate time-based temperature, heat, dryness, rain, and humidity features."""
    if dataframe.empty:
        return dataframe.copy()

    validated = validate_weather_observations(dataframe)
    result = validated.set_index("timestamp")
    temperature = result["temperature_c"]
    humidity = result["relative_humidity"]
    precipitation = result["precip_mm"]

    result["temp_delta_24h"] = _elapsed_delta(temperature, 24)
    result["temp_delta_48h"] = _elapsed_delta(temperature, 48)
    result["temp_max_24h"] = temperature.rolling("24h", min_periods=1).max()
    result["temp_max_48h"] = temperature.rolling("48h", min_periods=1).max()
    result["temp_min_24h"] = temperature.rolling("24h", min_periods=1).min()

    result["heat_index_c"] = [
        calculate_heat_index_c(temperature_c, relative_humidity)
        for temperature_c, relative_humidity in zip(temperature, humidity, strict=True)
    ]
    result["heat_index_max_24h"] = result["heat_index_c"].rolling(
        "24h", min_periods=1
    ).max()
    result["heat_index_max_48h"] = result["heat_index_c"].rolling(
        "48h", min_periods=1
    ).max()

    result["is_dry_hour"] = (
        precipitation.notna() & (precipitation < DRY_HOUR_PRECIP_THRESHOLD_MM)
    ).astype(int)
    result["consecutive_dry_hours"] = _consecutive_dry_hours(result.index, precipitation)
    result["consecutive_dry_days"] = result["consecutive_dry_hours"] / 24

    precipitation_for_sum = precipitation.fillna(0)
    result["rainfall_24h"] = precipitation_for_sum.rolling("24h", min_periods=1).sum()
    result["rainfall_48h"] = precipitation_for_sum.rolling("48h", min_periods=1).sum()
    result["rainfall_7d"] = precipitation_for_sum.rolling("168h", min_periods=1).sum()
    result["humidity_delta_24h"] = _elapsed_delta(humidity, 24)
    result["humidity_min_24h"] = humidity.rolling("24h", min_periods=1).min()

    return result.reset_index()


def fetch_weather_data(
    station_id: str = ECCC_STATION_ID,
    start_date: str = ECCC_WEATHER_START_DATE,
    end_date: str = ECCC_WEATHER_END_DATE,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch normalized observations through the configured safe data mode."""
    logger.info("Loading weather observations for station %s", station_id)
    return _fetch_eccc_data(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
        force=force,
        mode=SETTINGS.data_source_mode,
    )


def main(force: bool = False) -> None:
    """Create the local weather-feature Parquet artifact."""
    ensure_directories()
    observations = fetch_weather_data(force=force)
    if observations.empty:
        raise DataValidationError("No weather observations are available")
    features = calculate_weather_features(observations)
    output_path = DATA_DIR / "weather_features.parquet"
    features.to_parquet(output_path, index=False)
    logger.info("Saved %s weather feature row(s) to %s", len(features), output_path.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main(force=SETTINGS.weather_force_refresh)
