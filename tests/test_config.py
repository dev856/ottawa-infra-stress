"""Tests for typed, secure environment configuration."""

import pytest

from src import config
from src.errors import ConfigError


def test_config_defaults_are_safe() -> None:
    """Defaults stay local, restricted, and inside the project."""
    settings = config.Settings.from_mapping({})

    assert settings.api_host == "127.0.0.1"
    assert "*" not in settings.cors_origins
    assert settings.model_path.parent == config.MODELS_DIR.resolve()


def test_model_path_must_stay_inside_models_directory() -> None:
    """Model artifacts cannot be redirected to an arbitrary path."""
    with pytest.raises(ConfigError, match="MODEL_PATH must resolve inside models/"):
        config.Settings.from_mapping({"MODEL_PATH": "../unsafe.joblib"})


@pytest.mark.parametrize("port", ["zero", "0", "65536"])
def test_invalid_port_is_rejected(port: str) -> None:
    """API ports must be numeric and inside the TCP port range."""
    with pytest.raises(ConfigError, match="API_PORT"):
        config.Settings.from_mapping({"API_PORT": port})


@pytest.mark.parametrize("origins", ["*", "", "ftp://example.com", "not-a-url"])
def test_unsafe_cors_origins_are_rejected(origins: str) -> None:
    """CORS must name explicit HTTP(S) origins."""
    with pytest.raises(ConfigError, match="API_CORS_ORIGINS"):
        config.Settings.from_mapping({"API_CORS_ORIGINS": origins})
