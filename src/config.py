"""Parse and validate environment configuration for local project entrypoints.

The safe default is offline mock mode. This module exposes a typed ``Settings``
object and temporary module-level aliases for the existing pipeline modules.
Secrets are never included in validation messages or log output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, cast
from urllib.parse import urlparse

from dotenv import load_dotenv
from sqlalchemy import URL

if TYPE_CHECKING:
    from .errors import ConfigError
else:
    try:
        from .errors import ConfigError
    except ImportError:  # pragma: no cover - supports direct script imports
        from errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SQL_DIR = PROJECT_ROOT / "sql"
WEATHER_CACHE_DIR = DATA_DIR / "weather_cache"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

DEFAULT_H3_RESOLUTION = 8
DEFAULT_PREDICTION_HORIZON_HOURS = 48
DEFAULT_API_PORT = 8000
DEFAULT_MAX_API_FEATURES = 2_000
DEFAULT_MAX_BBOX_SPAN_DEGREES = 0.5
DEFAULT_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_HTTP_MAX_RETRIES = 3
DEFAULT_HTTP_MAX_PAGES = 500
DEFAULT_USER_AGENT = "ottawa-infra-stress/1.0 (local educational project)"
DEFAULT_CORS_ORIGINS = ("http://127.0.0.1:3000", "http://localhost:3000")

OTTAWA_BBOX_WEST = -76.35
OTTAWA_BBOX_SOUTH = 45.25
OTTAWA_BBOX_EAST = -75.50
OTTAWA_BBOX_NORTH = 45.60

DataSourceMode = Literal["mock", "hybrid", "live"]
TargetMode = Literal["synthetic", "observed"]


def _parse_int(
    values: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Parse an integer setting and enforce an inclusive range."""
    raw_value = values.get(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _parse_float(
    values: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Parse a floating-point setting and enforce an inclusive range."""
    raw_value = values.get(name, str(default)).strip()
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}")
    return value


def _parse_bool(values: Mapping[str, str], name: str, default: bool) -> bool:
    """Parse a conventional environment boolean."""
    raw_value = values.get(name, "1" if default else "0").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be one of: 0, 1, false, true, no, yes, off, on")


def _parse_date(values: Mapping[str, str], name: str, default: str) -> date:
    """Parse an ISO calendar date without exposing unrelated settings."""
    raw_value = values.get(name, default).strip()
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise ConfigError(f"{name} must use YYYY-MM-DD format") from exc


def _validate_https_url(name: str, value: str, *, required: bool) -> str:
    """Validate an optional or required public HTTPS endpoint."""
    normalized = value.strip()
    if not normalized:
        if required:
            raise ConfigError(f"{name} is required outside mock mode")
        return ""
    parsed = urlparse(normalized)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ConfigError(f"{name} must be a valid HTTPS URL")
    return normalized.rstrip("/")


def _trusted_path(raw_path: str, trusted_directory: Path, setting_name: str) -> Path:
    """Resolve a configured file path and keep it inside a trusted directory."""
    candidate = Path(raw_path.strip())
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    trusted_root = trusted_directory.resolve()
    if not resolved.is_relative_to(trusted_root):
        raise ConfigError(f"{setting_name} must resolve inside {trusted_root.name}/")
    return resolved


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated application settings loaded from environment variables."""

    data_source_mode: DataSourceMode
    target_mode: TargetMode
    postgres_host: str
    postgres_port: int
    postgres_db: str
    postgres_user: str
    postgres_password: str
    ottawa_311_service_url: str
    ottawa_water_network_url: str
    ottawa_roads_url: str
    ottawa_buildings_url: str
    h3_resolution: int
    prediction_horizon_hours: int
    model_path: Path
    eccc_station_id: str
    eccc_base_url: str
    eccc_weather_start_date: date
    eccc_weather_end_date: date
    weather_force_refresh: bool
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]
    max_api_features: int
    max_bbox_span_degrees: float
    http_timeout_seconds: float
    http_max_retries: int
    http_max_pages: int
    user_agent: str
    city_data_license_confirmed: bool
    eccc_data_terms_accepted: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> Settings:
        """Build settings from a mapping, usually ``os.environ``."""
        raw_mode = values.get("DATA_SOURCE_MODE", "mock").strip().lower()
        if raw_mode not in {"mock", "hybrid", "live"}:
            raise ConfigError("DATA_SOURCE_MODE must be one of: mock, hybrid, live")
        mode = cast(DataSourceMode, raw_mode)
        live_sources_required = mode != "mock"

        raw_target_mode = values.get("TARGET_MODE", "synthetic").strip().lower()
        if raw_target_mode not in {"synthetic", "observed"}:
            raise ConfigError("TARGET_MODE must be one of: synthetic, observed")
        target_mode = cast(TargetMode, raw_target_mode)

        start_date = _parse_date(values, "ECCC_WEATHER_START_DATE", "2019-01-01")
        end_date = _parse_date(values, "ECCC_WEATHER_END_DATE", "2025-12-31")
        if end_date < start_date:
            raise ConfigError(
                "ECCC_WEATHER_END_DATE must be on or after ECCC_WEATHER_START_DATE"
            )

        station_id = values.get("ECCC_STATION_ID", "49568").strip()
        if not station_id.isascii() or not station_id.isdigit():
            raise ConfigError("ECCC_STATION_ID must contain ASCII digits only")

        cors_values = values.get("API_CORS_ORIGINS", ",".join(DEFAULT_CORS_ORIGINS))
        cors_origins = tuple(
            origin.strip().rstrip("/")
            for origin in cors_values.split(",")
            if origin.strip()
        )
        if not cors_origins or "*" in cors_origins:
            raise ConfigError("API_CORS_ORIGINS must list explicit origins and cannot contain '*'")
        for origin in cors_origins:
            parsed_origin = urlparse(origin)
            if parsed_origin.scheme not in {"http", "https"} or not parsed_origin.netloc:
                raise ConfigError("API_CORS_ORIGINS contains an invalid HTTP(S) origin")

        city_licence_confirmed = _parse_bool(
            values, "CITY_DATA_LICENSE_CONFIRMED", False
        )
        eccc_terms_accepted = _parse_bool(values, "ECCC_DATA_TERMS_ACCEPTED", False)
        if live_sources_required and not city_licence_confirmed:
            raise ConfigError(
                "CITY_DATA_LICENSE_CONFIRMED must be enabled after source-specific legal review"
            )
        if live_sources_required and not eccc_terms_accepted:
            raise ConfigError(
                "ECCC_DATA_TERMS_ACCEPTED must be enabled after terms review"
            )

        return cls(
            data_source_mode=mode,
            target_mode=target_mode,
            postgres_host=values.get("POSTGRES_HOST", "localhost").strip() or "localhost",
            postgres_port=_parse_int(
                values, "POSTGRES_PORT", 5432, minimum=1, maximum=65_535
            ),
            postgres_db=values.get("POSTGRES_DB", "ottawa_infra").strip()
            or "ottawa_infra",
            postgres_user=values.get("POSTGRES_USER", "change_me").strip()
            or "change_me",
            postgres_password=values.get("POSTGRES_PASSWORD", "change_me"),
            ottawa_311_service_url=_validate_https_url(
                "OTTAWA_311_SERVICE_URL",
                values.get("OTTAWA_311_SERVICE_URL", ""),
                required=live_sources_required,
            ),
            ottawa_water_network_url=_validate_https_url(
                "OTTAWA_WATER_NETWORK_URL",
                values.get("OTTAWA_WATER_NETWORK_URL", ""),
                required=live_sources_required,
            ),
            ottawa_roads_url=_validate_https_url(
                "OTTAWA_ROADS_URL",
                values.get("OTTAWA_ROADS_URL", ""),
                required=live_sources_required,
            ),
            ottawa_buildings_url=_validate_https_url(
                "OTTAWA_BUILDINGS_URL",
                values.get("OTTAWA_BUILDINGS_URL", ""),
                required=live_sources_required,
            ),
            h3_resolution=_parse_int(
                values,
                "H3_RESOLUTION",
                DEFAULT_H3_RESOLUTION,
                minimum=1,
                maximum=15,
            ),
            prediction_horizon_hours=_parse_int(
                values,
                "PREDICTION_HORIZON_HOURS",
                DEFAULT_PREDICTION_HORIZON_HOURS,
                minimum=1,
                maximum=24 * 30,
            ),
            model_path=_trusted_path(
                values.get("MODEL_PATH", "models/infra_stress_model.joblib"),
                MODELS_DIR,
                "MODEL_PATH",
            ),
            eccc_station_id=station_id,
            eccc_base_url=_validate_https_url(
                "ECCC_BASE_URL",
                values.get(
                    "ECCC_BASE_URL",
                    "https://climate.weather.gc.ca/climate_data/bulk_data_e.html",
                ),
                required=live_sources_required,
            ),
            eccc_weather_start_date=start_date,
            eccc_weather_end_date=end_date,
            weather_force_refresh=_parse_bool(values, "WEATHER_FORCE_REFRESH", False),
            api_host=values.get("API_HOST", "127.0.0.1").strip() or "127.0.0.1",
            api_port=_parse_int(
                values, "API_PORT", DEFAULT_API_PORT, minimum=1, maximum=65_535
            ),
            cors_origins=cors_origins,
            max_api_features=_parse_int(
                values,
                "MAX_API_FEATURES",
                DEFAULT_MAX_API_FEATURES,
                minimum=1,
                maximum=10_000,
            ),
            max_bbox_span_degrees=_parse_float(
                values,
                "MAX_BBOX_SPAN_DEGREES",
                DEFAULT_MAX_BBOX_SPAN_DEGREES,
                minimum=0.01,
                maximum=5.0,
            ),
            http_timeout_seconds=_parse_float(
                values,
                "HTTP_TIMEOUT_SECONDS",
                DEFAULT_HTTP_TIMEOUT_SECONDS,
                minimum=1.0,
                maximum=120.0,
            ),
            http_max_retries=_parse_int(
                values,
                "HTTP_MAX_RETRIES",
                DEFAULT_HTTP_MAX_RETRIES,
                minimum=0,
                maximum=5,
            ),
            http_max_pages=_parse_int(
                values,
                "HTTP_MAX_PAGES",
                DEFAULT_HTTP_MAX_PAGES,
                minimum=1,
                maximum=5_000,
            ),
            user_agent=values.get("HTTP_USER_AGENT", DEFAULT_USER_AGENT).strip()
            or DEFAULT_USER_AGENT,
            city_data_license_confirmed=city_licence_confirmed,
            eccc_data_terms_accepted=eccc_terms_accepted,
        )

    def database_url(self) -> URL:
        """Return a SQLAlchemy URL object whose string form masks the password."""
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )


load_dotenv()
SETTINGS = Settings.from_mapping(os.environ)

# Compatibility aliases. New modules should accept ``Settings`` or the specific
# dependency they need instead of importing many globals.
DATA_SOURCE_MODE = SETTINGS.data_source_mode
TARGET_MODE = SETTINGS.target_mode
POSTGRES_HOST = SETTINGS.postgres_host
POSTGRES_PORT = SETTINGS.postgres_port
POSTGRES_DB = SETTINGS.postgres_db
POSTGRES_USER = SETTINGS.postgres_user
POSTGRES_PASSWORD = SETTINGS.postgres_password
OTTAWA_311_SERVICE_URL = SETTINGS.ottawa_311_service_url
OTTAWA_WATER_NETWORK_URL = SETTINGS.ottawa_water_network_url
OTTAWA_ROADS_URL = SETTINGS.ottawa_roads_url
OTTAWA_BUILDINGS_URL = SETTINGS.ottawa_buildings_url
H3_RESOLUTION = SETTINGS.h3_resolution
PREDICTION_HORIZON_HOURS = SETTINGS.prediction_horizon_hours
MODEL_PATH = SETTINGS.model_path
ECCC_STATION_ID = SETTINGS.eccc_station_id
ECCC_BASE_URL = SETTINGS.eccc_base_url
ECCC_WEATHER_START_DATE = SETTINGS.eccc_weather_start_date.isoformat()
ECCC_WEATHER_END_DATE = SETTINGS.eccc_weather_end_date.isoformat()
WEATHER_FORCE_REFRESH = int(SETTINGS.weather_force_refresh)
API_HOST = SETTINGS.api_host
API_PORT = SETTINGS.api_port
API_CORS_ORIGINS = SETTINGS.cors_origins
MAX_API_FEATURES = SETTINGS.max_api_features
MAX_BBOX_SPAN_DEGREES = SETTINGS.max_bbox_span_degrees


def validate_config() -> None:
    """Validate compatibility globals used by older callers and tests."""
    values = dict(os.environ)
    values.update(
        {
            "DATA_SOURCE_MODE": str(DATA_SOURCE_MODE),
            "H3_RESOLUTION": str(H3_RESOLUTION),
            "ECCC_WEATHER_START_DATE": str(ECCC_WEATHER_START_DATE),
            "ECCC_WEATHER_END_DATE": str(ECCC_WEATHER_END_DATE),
        }
    )
    # Validation callers should be able to check the four historical fields in
    # isolation even if they monkeypatch live mode for a negative test.
    values.setdefault("CITY_DATA_LICENSE_CONFIRMED", "1")
    values.setdefault("ECCC_DATA_TERMS_ACCEPTED", "1")
    Settings.from_mapping(values)


def ensure_directories() -> None:
    """Create only the repository-local runtime directories."""
    for directory in (DATA_DIR, MODELS_DIR, LOGS_DIR, WEATHER_CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_database_url() -> URL:
    """Return a safe SQLAlchemy connection URL without rendering its password."""
    return SETTINGS.database_url()
