"""Tests for typed, secure environment configuration."""

from pathlib import Path

import pytest
from sqlalchemy import URL

from src import config
from src.errors import ConfigError


def settings_values(**overrides: str) -> dict[str, str]:
    """Return a minimal valid mock configuration for focused tests."""
    values = {
        "DATA_SOURCE_MODE": "mock",
        "MODEL_PATH": "models/test-model.joblib",
        "ECCC_WEATHER_START_DATE": "2020-01-01",
        "ECCC_WEATHER_END_DATE": "2021-01-01",
    }
    values.update(overrides)
    return values


def test_config_defaults_are_safe() -> None:
    """Defaults stay offline, local, restricted, and typed."""
    settings = config.Settings.from_mapping({})

    assert settings.data_source_mode == "mock"
    assert settings.api_host == "127.0.0.1"
    assert settings.h3_resolution == 8
    assert settings.prediction_horizon_hours == 48
    assert "*" not in settings.cors_origins
    assert settings.model_path.parent == config.MODELS_DIR.resolve()


def test_database_url_masks_password_when_rendered() -> None:
    """SQLAlchemy URL rendering must not accidentally reveal a password."""
    settings = config.Settings.from_mapping(
        settings_values(POSTGRES_PASSWORD="do-not-print-this")
    )

    url = settings.database_url()

    assert isinstance(url, URL)
    assert "do-not-print-this" not in str(url)
    assert "***" in str(url)


def test_model_path_must_stay_inside_models_directory() -> None:
    """Model artifacts cannot be redirected to an arbitrary path."""
    with pytest.raises(ConfigError, match="MODEL_PATH must resolve inside models/"):
        config.Settings.from_mapping(settings_values(MODEL_PATH="../unsafe.joblib"))


@pytest.mark.parametrize("port", ["zero", "0", "65536"])
def test_invalid_port_is_rejected(port: str) -> None:
    """API ports must be numeric and inside the TCP port range."""
    with pytest.raises(ConfigError, match="API_PORT"):
        config.Settings.from_mapping(settings_values(API_PORT=port))


def test_wildcard_cors_is_rejected() -> None:
    """A wildcard CORS origin is never a safe default."""
    with pytest.raises(ConfigError, match="API_CORS_ORIGINS"):
        config.Settings.from_mapping(settings_values(API_CORS_ORIGINS="*"))


def test_invalid_date_range_is_rejected() -> None:
    """The weather end date cannot precede the start date."""
    with pytest.raises(ConfigError, match="must be on or after"):
        config.Settings.from_mapping(
            settings_values(
                ECCC_WEATHER_START_DATE="2024-01-01",
                ECCC_WEATHER_END_DATE="2023-01-01",
            )
        )


def test_live_mode_requires_legal_acknowledgements() -> None:
    """Unreviewed sources cannot be enabled by changing one mode setting."""
    with pytest.raises(ConfigError, match="CITY_DATA_LICENSE_CONFIRMED"):
        config.Settings.from_mapping(settings_values(DATA_SOURCE_MODE="live"))


def test_live_mode_requires_all_source_urls() -> None:
    """Approved live mode still requires explicit HTTPS endpoints."""
    with pytest.raises(ConfigError, match="OTTAWA_311_SERVICE_URL"):
        config.Settings.from_mapping(
            settings_values(
                DATA_SOURCE_MODE="live",
                CITY_DATA_LICENSE_CONFIRMED="1",
                ECCC_DATA_TERMS_ACCEPTED="1",
            )
        )


def test_ensure_directories_uses_project_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Directory creation is explicit and limited to configured runtime roots."""
    runtime_paths = [tmp_path / name for name in ("data", "models", "logs", "cache")]
    monkeypatch.setattr(config, "DATA_DIR", runtime_paths[0])
    monkeypatch.setattr(config, "MODELS_DIR", runtime_paths[1])
    monkeypatch.setattr(config, "LOGS_DIR", runtime_paths[2])
    monkeypatch.setattr(config, "WEATHER_CACHE_DIR", runtime_paths[3])

    config.ensure_directories()

    assert all(path.is_dir() for path in runtime_paths)
