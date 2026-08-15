"""Fetch and validate ArcGIS REST JSON with offline fixture support.

Live requests use a named user agent, finite timeouts, conservative retries, and
bounded pagination. Mock mode reads deterministic JSON and never uses the network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    from .config import (
        DATA_SOURCE_MODE,
        DEFAULT_HTTP_MAX_PAGES,
        DEFAULT_HTTP_MAX_RETRIES,
        DEFAULT_HTTP_TIMEOUT_SECONDS,
        DEFAULT_USER_AGENT,
        FIXTURES_DIR,
    )
    from .errors import DataValidationError, ExternalServiceError
else:
    try:
        from .config import (
            DATA_SOURCE_MODE,
            DEFAULT_HTTP_MAX_PAGES,
            DEFAULT_HTTP_MAX_RETRIES,
            DEFAULT_HTTP_TIMEOUT_SECONDS,
            DEFAULT_USER_AGENT,
            FIXTURES_DIR,
        )
        from .errors import DataValidationError, ExternalServiceError
    except ImportError:  # pragma: no cover - supports direct script imports
        from config import (
            DATA_SOURCE_MODE,
            DEFAULT_HTTP_MAX_PAGES,
            DEFAULT_HTTP_MAX_RETRIES,
            DEFAULT_HTTP_TIMEOUT_SECONDS,
            DEFAULT_USER_AGENT,
            FIXTURES_DIR,
        )
        from errors import DataValidationError, ExternalServiceError

logger = logging.getLogger(__name__)

ClientMode = Literal["mock", "hybrid", "live"]
DEFAULT_PAGE_SIZE = 1_000


class ArcGISRestClient:
    """Small ArcGIS FeatureServer/MapServer client with explicit data modes."""

    def __init__(
        self,
        base_url: str,
        timeout: float = DEFAULT_HTTP_TIMEOUT_SECONDS,
        mode: str | None = None,
        fixture_path: Path | str | None = None,
        *,
        max_retries: int = DEFAULT_HTTP_MAX_RETRIES,
        max_pages: int = DEFAULT_HTTP_MAX_PAGES,
        user_agent: str = DEFAULT_USER_AGENT,
        session: requests.Session | None = None,
    ) -> None:
        selected_mode = mode or DATA_SOURCE_MODE
        if selected_mode not in {"mock", "hybrid", "live"}:
            raise ValueError("mode must be one of: mock, hybrid, live")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if max_pages < 1:
            raise ValueError("max_pages must be at least one")

        normalized_url = base_url.strip().rstrip("/")
        if selected_mode != "mock":
            parsed_url = urlparse(normalized_url)
            if parsed_url.scheme != "https" or not parsed_url.netloc:
                raise ValueError("Live ArcGIS base_url must be a valid HTTPS URL")

        self.base_url = normalized_url
        self.timeout = timeout
        self.mode = cast(ClientMode, selected_mode)
        self.fixture_path = Path(fixture_path).resolve() if fixture_path else None
        self.max_pages = max_pages
        self.session = session or self._create_session(max_retries, user_agent)

    @staticmethod
    def _create_session(max_retries: int, user_agent: str) -> requests.Session:
        """Create an HTTP session with bounded GET-only retries."""
        retry_strategy = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.headers.update({"Accept": "application/json", "User-Agent": user_agent})
        session.mount("https://", adapter)
        return session

    @staticmethod
    def _validate_payload(payload: object, *, require_features: bool) -> dict[str, Any]:
        """Reject non-object payloads, ArcGIS error envelopes, and bad features."""
        if not isinstance(payload, dict):
            raise DataValidationError("ArcGIS response must be a JSON object")
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code", "unknown")
            raise ExternalServiceError(f"ArcGIS returned service error code {code}")
        if require_features:
            features = payload.get("features")
            if not isinstance(features, list):
                raise DataValidationError("ArcGIS query response must contain a features list")
            if any(not isinstance(feature, dict) for feature in features):
                raise DataValidationError("ArcGIS features must be JSON objects")
        return cast(dict[str, Any], payload)

    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        require_features: bool,
    ) -> dict[str, Any]:
        """Perform one request and map transport/JSON failures to project errors."""
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise ExternalServiceError("ArcGIS request failed") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise DataValidationError("ArcGIS response was not valid JSON") from exc
        return self._validate_payload(payload, require_features=require_features)

    def get_metadata(self) -> dict[str, Any]:
        """Return layer metadata or deterministic fixture metadata."""
        if self.mode == "mock":
            return {
                "name": "MockLayer",
                "type": "Feature Layer",
                "objectIdField": "objectid",
                "fields": [{"name": "objectid", "type": "esriFieldTypeOID"}],
            }
        return self._request_json(
            self.base_url,
            params={"f": "json"},
            require_features=False,
        )

    def _load_mock_data(self) -> dict[str, Any]:
        """Load and validate the selected local JSON fixture."""
        fixture_path = self.fixture_path or FIXTURES_DIR / "arcgis_311_sample.json"
        try:
            payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DataValidationError(
                f"ArcGIS fixture does not exist: {fixture_path.name}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise DataValidationError(f"ArcGIS fixture is invalid: {fixture_path.name}") from exc
        return self._validate_payload(payload, require_features=True)

    def query(
        self,
        where: str = "1=1",
        out_fields: str = "*",
        geometry: str | None = None,
        geometry_type: str | None = None,
        spatial_rel: str | None = None,
        result_offset: int = 0,
        result_record_count: int = DEFAULT_PAGE_SIZE,
        order_by_fields: str | None = None,
        return_geometry: bool = True,
        out_sr: int = 4326,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """Query one ArcGIS result page using structured request parameters."""
        if self.mode == "mock":
            return self._load_mock_data()
        if result_offset < 0 or result_record_count < 1:
            raise ValueError("ArcGIS pagination values must be positive")

        params: dict[str, Any] = {
            "f": "json",
            "where": where,
            "outFields": out_fields,
            "resultOffset": result_offset,
            "resultRecordCount": result_record_count,
            "returnGeometry": str(return_geometry).lower(),
            "outSR": out_sr,
        }
        optional_params = {
            "geometry": geometry,
            "geometryType": geometry_type,
            "spatialRel": spatial_rel,
            "orderByFields": order_by_fields,
        }
        params.update({key: value for key, value in optional_params.items() if value is not None})
        params.update(extra_params)

        try:
            return self._request_json(
                f"{self.base_url}/query",
                params=params,
                require_features=True,
            )
        except (ExternalServiceError, DataValidationError):
            if self.mode == "hybrid":
                logger.warning("ArcGIS live query failed; using the configured local fixture")
                return self._load_mock_data()
            raise

    def query_all(
        self,
        where: str = "1=1",
        out_fields: str = "*",
        result_record_count: int = DEFAULT_PAGE_SIZE,
        order_by_fields: str | None = None,
        return_geometry: bool = True,
        **extra_params: Any,
    ) -> dict[str, Any]:
        """Fetch every page up to ``max_pages`` and detect stalled pagination."""
        if result_record_count < 1:
            raise ValueError("result_record_count must be at least one")

        all_features: list[dict[str, Any]] = []
        previous_page: list[dict[str, Any]] | None = None
        exceeded_transfer_limit = False

        for page_number in range(self.max_pages):
            result_offset = page_number * result_record_count
            logger.info("Fetching ArcGIS page %s", page_number + 1)
            payload = self.query(
                where=where,
                out_fields=out_fields,
                result_offset=result_offset,
                result_record_count=result_record_count,
                order_by_fields=order_by_fields,
                return_geometry=return_geometry,
                **extra_params,
            )
            features = cast(list[dict[str, Any]], payload["features"])
            if previous_page is not None and features and features == previous_page:
                raise DataValidationError("ArcGIS pagination repeated the previous page")
            all_features.extend(features)

            page_exceeded = payload.get("exceededTransferLimit") is True
            exceeded_transfer_limit = exceeded_transfer_limit or page_exceeded
            if self.mode == "mock" or not features:
                break
            if len(features) < result_record_count or not page_exceeded:
                break
            previous_page = features
        else:
            raise ExternalServiceError("ArcGIS pagination exceeded the configured page limit")

        return {
            "features": all_features,
            "metadata": {
                "count": len(all_features),
                "exceededTransferLimit": exceeded_transfer_limit,
            },
        }

    def get_object_id_field(self) -> str:
        """Return the object-ID field declared by the layer metadata."""
        metadata = self.get_metadata()
        object_id_field = metadata.get("objectIdField") or metadata.get("objectIdFieldName")
        if not isinstance(object_id_field, str) or not object_id_field:
            raise DataValidationError("ArcGIS metadata has no object-ID field")
        return object_id_field
