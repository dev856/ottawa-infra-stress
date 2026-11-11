"""Parse and validate environment configuration for the project entrypoints.

Everything has a safe default, so the project runs with no ``.env`` file.
Validation messages never include secret values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv

from .errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPEN_DATA_DIR = PROJECT_ROOT / "open_data"
MODELS_DIR = PROJECT_ROOT / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

REQUESTS_PATH = OPEN_DATA_DIR / "service_requests.csv.gz"
WARDS_PATH = OPEN_DATA_DIR / "wards.geojson"

DEFAULT_API_PORT = 8000
DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0
DEFAULT_USER_AGENT = "ottawa-infra-stress/2.0 (educational portfolio project)"
DEFAULT_CORS_ORIGINS = ("http://127.0.0.1:3000", "http://localhost:3000")


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

    model_path: Path
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> Settings:
        """Build settings from a mapping, usually ``os.environ``."""
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

        return cls(
            model_path=_trusted_path(
                values.get("MODEL_PATH", "models/resolution_time_model.joblib"),
                MODELS_DIR,
                "MODEL_PATH",
            ),
            api_host=values.get("API_HOST", "127.0.0.1").strip() or "127.0.0.1",
            api_port=_parse_int(
                values, "API_PORT", DEFAULT_API_PORT, minimum=1, maximum=65_535
            ),
            cors_origins=cors_origins,
        )


load_dotenv()
SETTINGS = Settings.from_mapping(os.environ)

MODEL_PATH = SETTINGS.model_path
API_HOST = SETTINGS.api_host
API_PORT = SETTINGS.api_port
API_CORS_ORIGINS = SETTINGS.cors_origins
