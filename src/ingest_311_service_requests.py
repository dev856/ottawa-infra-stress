"""Fetch, validate, minimize, and H3-index water-main-break 311 events.

Exact coordinates are used in memory only to assign an H3 cell. The persisted
schema excludes coordinates, geometry, addresses, descriptions, notes, names,
and contact information.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import h3
import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from .arcgis_client import ArcGISRestClient
    from .config import (
        DATA_DIR,
        H3_RESOLUTION,
        OTTAWA_311_SERVICE_URL,
        OTTAWA_BBOX_EAST,
        OTTAWA_BBOX_NORTH,
        OTTAWA_BBOX_SOUTH,
        OTTAWA_BBOX_WEST,
        ensure_directories,
        get_database_url,
    )
    from .errors import DataValidationError
else:
    try:
        from .arcgis_client import ArcGISRestClient
        from .config import (
            DATA_DIR,
            H3_RESOLUTION,
            OTTAWA_311_SERVICE_URL,
            OTTAWA_BBOX_EAST,
            OTTAWA_BBOX_NORTH,
            OTTAWA_BBOX_SOUTH,
            OTTAWA_BBOX_WEST,
            ensure_directories,
            get_database_url,
        )
        from .errors import DataValidationError
    except ImportError:  # pragma: no cover - supports direct script imports
        from arcgis_client import ArcGISRestClient
        from config import (
            DATA_DIR,
            H3_RESOLUTION,
            OTTAWA_311_SERVICE_URL,
            OTTAWA_BBOX_EAST,
            OTTAWA_BBOX_NORTH,
            OTTAWA_BBOX_SOUTH,
            OTTAWA_BBOX_WEST,
            ensure_directories,
            get_database_url,
        )
        from errors import DataValidationError

logger = logging.getLogger(__name__)

SOURCE_FIELDS = ("objectid", "reqtype", "category", "createddate")
PERSISTED_FIELDS = (*SOURCE_FIELDS, "h3_index")
WATER_BREAK_REQUEST_TYPES = {
    "MAIN BREAK",
    "WATER BREAK",
    "WATER MAIN",
    "WATER MAIN BREAK",
}
MAX_SOURCE_RECORDS = 1_000_000


def normalize_text(value: object) -> str:
    """Normalize a controlled categorical value for exact matching."""
    if value is None:
        return ""
    return " ".join(str(value).strip().upper().split())


def is_water_break_record(record: dict[str, Any]) -> bool:
    """Return true only for an explicit water-main-break request type.

    Category names such as ``WATER`` are intentionally insufficient, and free
    text is intentionally ignored because it can contain personal information.
    """
    attributes = record.get("attributes", record)
    if not isinstance(attributes, dict):
        return False
    request_type = normalize_text(
        attributes.get("reqtype")
        or attributes.get("req_type")
        or attributes.get("request_type")
    )
    return request_type in WATER_BREAK_REQUEST_TYPES


def _parse_iso_date(value: str, field_name: str) -> date:
    """Validate a date before embedding it in the ArcGIS query expression."""
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DataValidationError(f"{field_name} must use YYYY-MM-DD format") from exc


def build_date_filter(start_date: str | None, end_date: str | None) -> str:
    """Build a date-only ArcGIS filter from strictly validated ISO dates."""
    clauses = ["1=1"]
    parsed_start = _parse_iso_date(start_date, "start_date") if start_date else None
    parsed_end = _parse_iso_date(end_date, "end_date") if end_date else None
    if parsed_start and parsed_end and parsed_end < parsed_start:
        raise DataValidationError("end_date must be on or after start_date")
    if parsed_start:
        clauses.append(f"createddate >= DATE '{parsed_start.isoformat()}'")
    if parsed_end:
        exclusive_end = parsed_end + timedelta(days=1)
        clauses.append(f"createddate < DATE '{exclusive_end.isoformat()}'")
    return " AND ".join(clauses)


def _parse_timestamp(value: object) -> pd.Timestamp:
    """Parse an ArcGIS epoch-millisecond or ISO timestamp into UTC."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        timestamp = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    else:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise DataValidationError("311 event has an invalid createddate")
    return pd.Timestamp(timestamp)


def _parse_coordinates(feature: dict[str, Any]) -> tuple[float, float]:
    """Return a validated Ottawa longitude/latitude pair from an ArcGIS point."""
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise DataValidationError("311 event has no point geometry")
    try:
        longitude = float(geometry["x"])
        latitude = float(geometry["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataValidationError("311 event has invalid point coordinates") from exc
    if not (
        OTTAWA_BBOX_WEST <= longitude <= OTTAWA_BBOX_EAST
        and OTTAWA_BBOX_SOUTH <= latitude <= OTTAWA_BBOX_NORTH
    ):
        raise DataValidationError("311 event coordinates are outside the Ottawa boundary")
    return longitude, latitude


def minimize_feature(feature: dict[str, Any], resolution: int) -> dict[str, Any] | None:
    """Convert one source feature to the privacy-safe persisted event schema."""
    attributes = feature.get("attributes")
    if not isinstance(attributes, dict) or not is_water_break_record(attributes):
        return None
    if attributes.get("objectid") is None:
        raise DataValidationError("311 event is missing objectid")

    longitude, latitude = _parse_coordinates(feature)
    return {
        "objectid": int(attributes["objectid"]),
        "reqtype": normalize_text(attributes.get("reqtype")),
        "category": normalize_text(attributes.get("category")),
        "createddate": _parse_timestamp(attributes.get("createddate")),
        "h3_index": h3.latlng_to_cell(latitude, longitude, resolution),
    }


def minimize_features(
    features: list[dict[str, Any]],
    resolution: int = H3_RESOLUTION,
) -> pd.DataFrame:
    """Minimize valid water-main-break features and report rejected records."""
    if not 1 <= resolution <= 15:
        raise ValueError("resolution must be between 1 and 15")

    records: list[dict[str, Any]] = []
    rejected_count = 0
    for feature in features:
        try:
            record = minimize_feature(feature, resolution)
        except (DataValidationError, TypeError, ValueError):
            rejected_count += 1
            continue
        if record is not None:
            records.append(record)

    dataframe = pd.DataFrame(records, columns=PERSISTED_FIELDS)
    if not dataframe.empty:
        dataframe = dataframe.drop_duplicates(subset=["objectid"], keep="first")
        dataframe = dataframe.sort_values(["createddate", "objectid"]).reset_index(drop=True)
    if rejected_count:
        logger.warning("Rejected %s malformed 311 event(s)", rejected_count)
    return dataframe


def fetch_311_service_requests(
    url: str = OTTAWA_311_SERVICE_URL,
    start_date: str | None = None,
    end_date: str | None = None,
    max_records: int | None = None,
    client: ArcGISRestClient | None = None,
) -> pd.DataFrame:
    """Fetch only required source fields and return minimized H3 event rows."""
    if max_records is not None and not 1 <= max_records <= MAX_SOURCE_RECORDS:
        raise ValueError(f"max_records must be between 1 and {MAX_SOURCE_RECORDS}")
    arcgis_client = client or ArcGISRestClient(url)
    response = arcgis_client.query_all(
        where=build_date_filter(start_date, end_date),
        out_fields=",".join(SOURCE_FIELDS),
        result_record_count=1_000,
        order_by_fields="createddate,objectid",
        return_geometry=True,
    )
    features = response["features"]
    if max_records is not None:
        features = features[:max_records]
    logger.info("Fetched %s candidate 311 event(s)", len(features))
    events = minimize_features(features)
    logger.info("Retained %s validated water-main-break event(s)", len(events))
    return events


def upsert_events(engine: Engine, events: pd.DataFrame) -> int:
    """Idempotently upsert minimized events with parameterized SQL."""
    if events.empty:
        return 0
    statement = text(
        """
        INSERT INTO raw.h3_311_events (
            objectid, reqtype, category, createddate, h3_index
        ) VALUES (
            :objectid, :reqtype, :category, :createddate, :h3_index
        )
        ON CONFLICT (objectid) DO UPDATE SET
            reqtype = EXCLUDED.reqtype,
            category = EXCLUDED.category,
            createddate = EXCLUDED.createddate,
            h3_index = EXCLUDED.h3_index
        """
    )
    parameters = events[list(PERSISTED_FIELDS)].to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(statement, parameters)
    return len(parameters)


def load_events_to_postgis(events: pd.DataFrame) -> bool:
    """Write minimized events when local PostGIS is available."""
    if events.empty:
        return True
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    try:
        count = upsert_events(engine, events)
    except SQLAlchemyError:
        logger.info("PostGIS is unavailable; kept the local minimized Parquet artifact")
        return False
    finally:
        engine.dispose()
    logger.info("Upserted %s minimized 311 event(s) into PostGIS", count)
    return True


def main() -> None:
    """Create the local privacy-safe 311 H3 event artifact."""
    ensure_directories()
    events = fetch_311_service_requests(
        start_date="2019-01-01",
        end_date="2025-12-31",
    )
    output_path = DATA_DIR / "h3_311_events.parquet"
    events.to_parquet(output_path, index=False)
    logger.info("Saved %s minimized H3 event(s) to %s", len(events), output_path.name)
    load_events_to_postgis(events)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
