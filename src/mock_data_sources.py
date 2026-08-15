"""Generate deterministic synthetic data for offline development and tests.

Generated values are clearly synthetic and do not contain copied municipal
records, exact addresses, personal information, or live service dependencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOCK_WEATHER_START = "2024-05-01 00:00:00"
MOCK_WEATHER_HOURS = 24 * 92


def generate_mock_weather_observations(
    start: str = MOCK_WEATHER_START,
    hours: int = MOCK_WEATHER_HOURS,
    station_id: str = "49568",
) -> pd.DataFrame:
    """Return a reproducible hourly summer-like weather time series."""
    if hours < 1:
        raise ValueError("hours must be at least one")
    timestamps = pd.date_range(start, periods=hours, freq="h")
    elapsed = np.arange(hours, dtype=float)
    daily_cycle = np.sin(2 * np.pi * (elapsed % 24) / 24 - np.pi / 2)
    seasonal_cycle = np.sin(2 * np.pi * elapsed / (24 * 31))
    temperature = 23.0 + 7.0 * daily_cycle + 4.0 * seasonal_cycle
    humidity = np.clip(63.0 - 18.0 * daily_cycle - 6.0 * seasonal_cycle, 25.0, 98.0)
    precipitation = np.zeros(hours, dtype=float)
    storm_hours = (elapsed.astype(int) % (24 * 9)).astype(int)
    precipitation[storm_hours == 0] = 5.0
    precipitation[storm_hours == 1] = 2.0

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "station_id": station_id,
            "climate_id": "SYNTHETIC",
            "station_name": "SYNTHETIC OTTAWA WEATHER",
            "temperature_c": temperature.round(2),
            "dew_point_c": (temperature - 5.0).round(2),
            "relative_humidity": humidity.round(1),
            "precip_mm": precipitation,
            "wind_speed_kmh": 12.0 + 3.0 * np.cos(2 * np.pi * elapsed / 24),
            "pressure_kpa": 101.0 + 0.5 * np.sin(2 * np.pi * elapsed / (24 * 5)),
        }
    )
