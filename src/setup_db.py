#!/usr/bin/env python
"""Check optional local PostgreSQL/PostGIS readiness without changing schemas.

Docker initializes schemas from ``sql/init.sql``. This command performs bounded,
read-only checks and permits an offline Parquet workflow only in mock mode or
when ``--allow-offline`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import TYPE_CHECKING

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from .config import DATA_SOURCE_MODE, get_database_url
else:
    try:
        from .config import DATA_SOURCE_MODE, get_database_url
    except ImportError:  # pragma: no cover - supports direct script imports
        from config import DATA_SOURCE_MODE, get_database_url

logger = logging.getLogger(__name__)

REQUIRED_SCHEMAS = {"raw", "features", "model"}
REQUIRED_TABLES = {
    "raw.h3_311_events",
    "features.h3_grid",
    "features.infrastructure_features",
    "model.risk_predictions",
}


def test_connection(engine: Engine) -> bool:
    """Return true when a bounded ``SELECT 1`` succeeds."""
    try:
        with engine.connect() as connection:
            return bool(connection.execute(text("SELECT 1")).scalar_one() == 1)
    except (SQLAlchemyError, OSError):
        logger.info("PostgreSQL connectivity check failed")
        return False


def test_postgis(engine: Engine) -> bool:
    """Verify PostGIS and all required schemas without logging server internals."""
    try:
        with engine.connect() as connection:
            postgis_version = connection.execute(text("SELECT PostGIS_Version()"))
            if not postgis_version.scalar():
                return False
            schema_rows = connection.execute(
                text(
                    """
                    SELECT schema_name
                    FROM information_schema.schemata
                    WHERE schema_name IN ('raw', 'features', 'model')
                    """
                )
            )
            schemas = {row[0] for row in schema_rows}
    except (SQLAlchemyError, OSError):
        logger.info("PostGIS/schema readiness check failed")
        return False
    missing = REQUIRED_SCHEMAS - schemas
    if missing:
        logger.warning("Database is missing required schema(s): %s", sorted(missing))
        return False
    return True


def test_tables_exist(engine: Engine) -> bool:
    """Verify the exact tables needed by ingestion, feature, and API code."""
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT table_schema || '.' || table_name
                    FROM information_schema.tables
                    WHERE table_schema IN ('raw', 'features', 'model')
                    """
                )
            )
            tables = {row[0] for row in rows}
    except (SQLAlchemyError, OSError):
        logger.info("Database table readiness check failed")
        return False
    missing = REQUIRED_TABLES - tables
    if missing:
        logger.warning("Database is missing required table(s): %s", sorted(missing))
        return False
    return True


def run_all_checks(engine: Engine) -> bool:
    """Run every read-only readiness check, stopping after a failed prerequisite."""
    return test_connection(engine) and test_postgis(engine) and test_tables_exist(engine)


def main(argv: list[str] | None = None) -> int:
    """Run the database readiness CLI and return a process exit code."""
    parser = argparse.ArgumentParser(description="Check local PostGIS readiness")
    parser.add_argument(
        "--allow-offline",
        action="store_true",
        help="Permit a file-only run even outside the default mock mode",
    )
    arguments = parser.parse_args(argv)

    try:
        engine = create_engine(
            get_database_url(),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3},
        )
    except (SQLAlchemyError, ModuleNotFoundError):
        engine = None

    try:
        if engine is not None and run_all_checks(engine):
            logger.info("PostGIS readiness checks passed")
            return 0
    finally:
        if engine is not None:
            engine.dispose()

    if DATA_SOURCE_MODE == "mock" or arguments.allow_offline:
        logger.info("PostGIS is offline; continuing with repository-local Parquet artifacts")
        return 0
    logger.error("PostGIS readiness checks failed and offline operation is not allowed")
    return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
