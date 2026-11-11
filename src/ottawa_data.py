"""Download the open-data snapshot that the insights and model are built from.

Writes two files to ``open_data/`` that are committed to the repository, so the
whole project works offline. Run again to refresh them:

    python -m src.ottawa_data

Each 311 request keeps only its dates, category, type, ward, channel, and
status: no addresses, coordinates, or request IDs.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pandas as pd
import requests

from .config import (
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_USER_AGENT,
    OPEN_DATA_DIR,
    REQUESTS_PATH,
    WARDS_PATH,
)
from .errors import DataValidationError, ExternalServiceError

logger = logging.getLogger(__name__)

SERVICE_REQUEST_URLS = (
    "https://311opendatastorage.blob.core.windows.net/311data/311opendata_lastyear.csv",
    "https://311opendatastorage.blob.core.windows.net/311data/311opendata_currentyear.csv",
)
WARDS_URL = (
    "https://services.arcgis.com/G6F8XLCl5KtAlZ2G/arcgis/rest/services/"
    "Wards_2022_2026/FeatureServer/0/query"
)
WARD_COUNT = 24
SOURCE_COLUMNS = {
    "Opened Date": "opened",
    "Closed Date": "closed",
    "Type": "category",
    "Description": "type",
    "Ward": "ward",
    "Channel": "channel",
    "Status": "status",
}


def _download(url: str, params: dict[str, str] | None = None) -> bytes:
    """Fetch one public file with a finite timeout and a named user agent."""
    try:
        response = requests.get(
            url,
            params=params,
            timeout=DEFAULT_HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": DEFAULT_USER_AGENT},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ExternalServiceError(f"Download failed: {url}") from exc
    return response.content


def parse_service_requests(content: bytes) -> pd.DataFrame:
    """Return the kept columns of one City of Ottawa 311 CSV, English labels only."""
    # The City's CSV has a few rows with unquoted commas; they are skipped, not guessed.
    raw = pd.read_csv(
        io.BytesIO(content),
        dtype=str,
        na_values=[r"\N"],
        on_bad_lines="skip",
        encoding="utf-8-sig",
    )
    raw.columns = [column.split(" | ")[0].strip() for column in raw.columns]
    missing = set(SOURCE_COLUMNS) - set(raw.columns)
    if missing:
        raise DataValidationError(f"311 file is missing columns: {', '.join(sorted(missing))}")

    result = raw[list(SOURCE_COLUMNS)].rename(columns=SOURCE_COLUMNS)
    result["opened"] = pd.to_datetime(result["opened"], errors="coerce")
    result["closed"] = pd.to_datetime(result["closed"], errors="coerce")
    # Bilingual labels read "English | French"; keep the English half.
    result["type"] = result["type"].str.split(" | ", regex=False).str[0].str.strip()
    ward = pd.to_numeric(result["ward"], errors="coerce")
    result["ward"] = ward.where(ward.between(1, WARD_COUNT)).astype("Int64")
    result = result.dropna(subset=["opened", "type", "category"])
    return result.sort_values("opened").reset_index(drop=True)


def fetch_service_requests() -> pd.DataFrame:
    """Download last year's and this year's 311 files."""
    frames = [parse_service_requests(_download(url)) for url in SERVICE_REQUEST_URLS]
    return pd.concat(frames, ignore_index=True)


def fetch_wards() -> dict[str, Any]:
    """Download simplified 2022-2026 ward boundaries as GeoJSON in WGS84."""
    geojson = json.loads(
        _download(
            WARDS_URL,
            {
                "where": "1=1",
                "outFields": "WARD,NAME,SECTOR_EN",
                "outSR": "4326",
                "maxAllowableOffset": "0.0003",
                "geometryPrecision": "5",
                "f": "geojson",
            },
        )
    )
    features = geojson.get("features", [])
    if len(features) != WARD_COUNT:
        raise DataValidationError(f"Expected {WARD_COUNT} wards, received {len(features)}")
    for feature in features:
        attributes = feature["properties"]
        feature["properties"] = {
            "ward": int(attributes["WARD"]),
            "name": str(attributes["NAME"]),
            "sector": str(attributes["SECTOR_EN"]),
        }
    features.sort(key=lambda feature: feature["properties"]["ward"])
    return {"type": "FeatureCollection", "features": features}


def main() -> None:
    """Refresh every file in the committed open-data snapshot."""
    OPEN_DATA_DIR.mkdir(exist_ok=True)
    service_requests = fetch_service_requests()
    if service_requests.empty:
        raise DataValidationError("No 311 service requests were found")
    wards = fetch_wards()

    service_requests.to_csv(REQUESTS_PATH, index=False, date_format="%Y-%m-%d")
    WARDS_PATH.write_text(json.dumps(wards, separators=(",", ":")), encoding="utf-8")
    logger.info(
        "Saved %s service requests (%s to %s) and %s wards",
        len(service_requests),
        service_requests["opened"].min().date(),
        service_requests["opened"].max().date(),
        len(wards["features"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
