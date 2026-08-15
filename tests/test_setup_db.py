"""Unit tests for read-only database readiness checks."""

from unittest.mock import MagicMock

from sqlalchemy.exc import OperationalError

from src import setup_db


def result_with_scalar(value: object) -> MagicMock:
    """Return a SQLAlchemy-like mock scalar result."""
    result = MagicMock()
    result.scalar.return_value = value
    result.scalar_one.return_value = value
    return result


def test_connection_success() -> None:
    """A scalar one indicates connectivity."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    connection.execute.return_value = result_with_scalar(1)
    assert setup_db.test_connection(engine) is True


def test_connection_failure_is_sanitized() -> None:
    """Connection failures return false without escaping the health boundary."""
    engine = MagicMock()
    engine.connect.side_effect = OperationalError("statement", {}, Exception("secret"))
    assert setup_db.test_connection(engine) is False


def test_postgis_requires_all_schemas() -> None:
    """PostGIS alone is insufficient when application schemas are missing."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    schemas = MagicMock()
    schemas.__iter__.return_value = [("raw",), ("features",), ("model",)]
    connection.execute.side_effect = [result_with_scalar("3.4"), schemas]
    assert setup_db.test_postgis(engine) is True

    missing = MagicMock()
    missing.__iter__.return_value = [("raw",)]
    connection.execute.side_effect = [result_with_scalar("3.4"), missing]
    assert setup_db.test_postgis(engine) is False


def test_tables_exist_requires_exact_contract() -> None:
    """A populated schema does not pass unless every required table exists."""
    engine = MagicMock()
    connection = engine.connect.return_value.__enter__.return_value
    rows = MagicMock()
    rows.__iter__.return_value = [(name,) for name in sorted(setup_db.REQUIRED_TABLES)]
    connection.execute.return_value = rows
    assert setup_db.test_tables_exist(engine) is True


def test_main_allows_offline_mock(monkeypatch) -> None:
    """Mock mode remains locally runnable without Docker."""
    monkeypatch.setattr(setup_db, "run_all_checks", lambda engine: False)
    monkeypatch.setattr(setup_db, "DATA_SOURCE_MODE", "mock")
    assert setup_db.main(argv=[]) == 0


def test_main_rejects_offline_live(monkeypatch) -> None:
    """Live mode cannot silently switch storage behavior."""
    monkeypatch.setattr(setup_db, "run_all_checks", lambda engine: False)
    monkeypatch.setattr(setup_db, "DATA_SOURCE_MODE", "live")
    assert setup_db.main(argv=[]) == 1
