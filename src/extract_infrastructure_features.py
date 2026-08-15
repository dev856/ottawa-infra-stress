"""Aggregate infrastructure geometry into H3-level model features.

Mock mode builds deterministic synthetic layers. Line geometries are clipped to
each H3 polygon before metre-based length calculation in EPSG:3347.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from .config import DATA_DIR, DATA_SOURCE_MODE, ensure_directories, get_database_url
    from .errors import DataValidationError, ExternalServiceError
    from .generate_h3_grid import generate_h3_grid
else:
    try:
        from .config import DATA_DIR, DATA_SOURCE_MODE, ensure_directories, get_database_url
        from .errors import DataValidationError, ExternalServiceError
        from .generate_h3_grid import generate_h3_grid
    except ImportError:  # pragma: no cover - supports direct script imports
        from config import DATA_DIR, DATA_SOURCE_MODE, ensure_directories, get_database_url
        from errors import DataValidationError, ExternalServiceError
        from generate_h3_grid import generate_h3_grid

logger = logging.getLogger(__name__)

METRIC_CRS = "EPSG:3347"
GEOGRAPHIC_CRS = "EPSG:4326"
DEFAULT_BUILDING_YEAR = 1980.0
INFRASTRUCTURE_COLUMNS = (
    "h3_index",
    "centroid_lat",
    "centroid_lon",
    "line_length_km_water",
    "line_count_water",
    "line_length_km_road",
    "line_count_road",
    "building_count",
    "median_year_built",
    "pct_pre_1980",
)


def load_h3_grid(mode: str = DATA_SOURCE_MODE) -> gpd.GeoDataFrame:
    """Load the local grid, optionally checking PostGIS outside mock mode."""
    parquet_path = DATA_DIR / "h3_grid.parquet"
    if parquet_path.is_file():
        grid = gpd.read_parquet(parquet_path)
        logger.info("Loaded %s H3 cell(s) from %s", len(grid), parquet_path.name)
        return grid

    if mode != "mock":
        engine = create_engine(get_database_url(), pool_pre_ping=True)
        try:
            query = text(
                """
                SELECT h3_index, h3_resolution, centroid_lat, centroid_lon, geometry
                FROM features.h3_grid
                ORDER BY h3_index
                """
            )
            grid = gpd.read_postgis(query, engine, geom_col="geometry")
            if not grid.empty:
                return grid
        except SQLAlchemyError:
            logger.info("PostGIS H3 grid is unavailable; generating a local grid")
        finally:
            engine.dispose()

    dataframe = generate_h3_grid()
    return gpd.GeoDataFrame(dataframe, geometry="geometry", crs=GEOGRAPHIC_CRS)


def generate_mock_infrastructure(
    h3_grid: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build deterministic, visibly synthetic infrastructure geometries with spatial density."""
    water_lines: list[dict[str, Any]] = []
    road_lines: list[dict[str, Any]] = []
    buildings: list[dict[str, Any]] = []
    downtown_lat = 45.4215
    downtown_lon = -75.6972
    obj_id = 1

    for index, row in enumerate(h3_grid.itertuples(index=False)):
        polygon = row.geometry
        min_x, min_y, max_x, max_y = polygon.bounds
        centroid = polygon.centroid

        dx = (centroid.x - downtown_lon) * 78.0
        dy = (centroid.y - downtown_lat) * 111.0
        dist_km = float(np.sqrt(dx * dx + dy * dy))

        pseudo_seed = float(((index * 2654435761) % (2**31)) / (2**31))
        density_factor = float(np.exp(-dist_km / 10.0))

        target_buildings = max(4, int(15 + 260 * density_factor + (pseudo_seed - 0.5) * 30))
        base_year = int(1955 + min(55, dist_km * 2.2) + (pseudo_seed - 0.5) * 20)
        base_year = max(1910, min(2020, base_year))

        for b_idx in range(target_buildings):
            angle = float(b_idx * 2.399963)
            radius = float(np.sqrt((b_idx + 0.5) / target_buildings) * (max_x - min_x) * 0.38)
            bx = float(centroid.x + radius * np.cos(angle))
            by = float(centroid.y + radius * np.sin(angle))
            b_year = int(base_year + ((b_idx * 13 + index * 7) % 31) - 15)
            b_year = max(1890, min(2024, b_year))

            buildings.append({
                "objectid": obj_id,
                "year_built": b_year,
                "geometry": Point(bx, by),
            })
            obj_id += 1

        num_pipes = max(2, int(2 + 8 * density_factor))
        for p_idx in range(num_pipes):
            fraction = (p_idx + 1) / (num_pipes + 1)
            py = min_y + (max_y - min_y) * fraction
            water_lines.append({
                "objectid": obj_id,
                "feature_type": "SYNTHETIC_WATER_MAIN",
                "geometry": LineString([(min_x, py), (max_x, py)]),
            })
            obj_id += 1

        num_roads = max(2, int(2 + 7 * density_factor))
        for r_idx in range(num_roads):
            fraction = (r_idx + 1) / (num_roads + 1)
            rx = min_x + (max_x - min_x) * fraction
            road_lines.append({
                "objectid": obj_id,
                "road_class": "SYNTHETIC_COLLECTOR",
                "geometry": LineString([(rx, min_y), (rx, max_y)]),
            })
            obj_id += 1

    return (
        gpd.GeoDataFrame(water_lines, geometry="geometry", crs=GEOGRAPHIC_CRS),
        gpd.GeoDataFrame(road_lines, geometry="geometry", crs=GEOGRAPHIC_CRS),
        gpd.GeoDataFrame(buildings, geometry="geometry", crs=GEOGRAPHIC_CRS),
    )


def fetch_live_infrastructure() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Refuse unimplemented live adapters instead of silently returning mock data."""
    raise ExternalServiceError(
        "Live infrastructure layers are disabled until exact source schemas are approved"
    )


def calculate_line_density(
    lines: gpd.GeoDataFrame,
    h3_grid: gpd.GeoDataFrame,
    prefix: str = "water",
) -> pd.DataFrame:
    """Clip lines to cells, then aggregate segment length and unique source count."""
    if prefix not in {"water", "road"}:
        raise ValueError("prefix must be 'water' or 'road'")
    length_column = f"line_length_km_{prefix}"
    count_column = f"line_count_{prefix}"
    empty = pd.DataFrame(columns=["h3_index", length_column, count_column])
    if lines.empty or h3_grid.empty:
        return empty
    if lines.crs is None or h3_grid.crs is None:
        raise DataValidationError("Infrastructure and H3 geometry must declare a CRS")
    if "objectid" not in lines or "h3_index" not in h3_grid:
        raise DataValidationError("Infrastructure line inputs are missing required identifiers")

    metric_lines = lines[["objectid", "geometry"]].to_crs(METRIC_CRS)
    metric_cells = h3_grid[["h3_index", "geometry"]].to_crs(METRIC_CRS)
    clipped = gpd.overlay(metric_lines, metric_cells, how="intersection", keep_geom_type=True)
    if clipped.empty:
        return empty
    clipped["clipped_length_km"] = clipped.geometry.length / 1_000
    clipped = clipped[clipped["clipped_length_km"] > 0]
    if clipped.empty:
        return empty

    aggregated = (
        clipped.groupby("h3_index", as_index=False)
        .agg(
            line_length_km=("clipped_length_km", "sum"),
            line_count=("objectid", "nunique"),
        )
        .rename(columns={"line_length_km": length_column, "line_count": count_column})
    )
    return aggregated


def calculate_building_features(
    buildings: gpd.GeoDataFrame,
    h3_grid: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Aggregate building count and vintage using one representative point each."""
    output_columns = (
        "h3_index",
        "building_count",
        "median_year_built",
        "pct_pre_1980",
    )
    if buildings.empty or h3_grid.empty:
        return pd.DataFrame(columns=output_columns)
    required = {"objectid", "year_built", "geometry"}
    if not required.issubset(buildings.columns):
        raise DataValidationError("Building inputs are missing required fields")

    normalized = buildings[["objectid", "year_built", "geometry"]].copy()
    normalized["year_built"] = pd.to_numeric(normalized["year_built"], errors="coerce")
    normalized = normalized[normalized["year_built"].between(1600, 2100)]
    normalized["geometry"] = normalized.geometry.representative_point()
    joined = gpd.sjoin(
        normalized,
        h3_grid[["h3_index", "geometry"]],
        predicate="within",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame(columns=output_columns)

    joined["is_pre_1980"] = joined["year_built"] < 1980
    return (
        joined.groupby("h3_index", as_index=False)
        .agg(
            building_count=("objectid", "nunique"),
            median_year_built=("year_built", "median"),
            pct_pre_1980=("is_pre_1980", "mean"),
        )
        .loc[:, output_columns]
    )


def create_infrastructure_features(mode: str = DATA_SOURCE_MODE) -> pd.DataFrame:
    """Return one complete infrastructure-feature row per H3 cell."""
    if mode not in {"mock", "hybrid", "live"}:
        raise ValueError("mode must be one of: mock, hybrid, live")
    h3_grid = load_h3_grid(mode)
    if mode == "mock":
        water, roads, buildings = generate_mock_infrastructure(h3_grid)
    else:
        try:
            water, roads, buildings = fetch_live_infrastructure()
        except ExternalServiceError:
            if mode != "hybrid":
                raise
            logger.warning("Live infrastructure failed; using deterministic mock layers")
            water, roads, buildings = generate_mock_infrastructure(h3_grid)

    features = h3_grid[["h3_index", "centroid_lat", "centroid_lon"]].copy()
    features = features.merge(calculate_line_density(water, h3_grid, "water"), how="left")
    features = features.merge(calculate_line_density(roads, h3_grid, "road"), how="left")
    features = features.merge(calculate_building_features(buildings, h3_grid), how="left")

    integer_columns = ("line_count_water", "line_count_road", "building_count")
    float_defaults = {
        "line_length_km_water": 0.0,
        "line_length_km_road": 0.0,
        "median_year_built": DEFAULT_BUILDING_YEAR,
        "pct_pre_1980": 0.0,
    }
    for column in integer_columns:
        features[column] = features[column].fillna(0).astype(int)
    for column, default in float_defaults.items():
        features[column] = features[column].fillna(default).astype(float)
    return features[list(INFRASTRUCTURE_COLUMNS)].sort_values("h3_index").reset_index(drop=True)


def upsert_infrastructure_features(engine: Engine, features: pd.DataFrame) -> int:
    """Idempotently upsert aggregate features using bound SQL values."""
    if features.empty:
        return 0
    assignments = ",\n            ".join(
        f"{column} = EXCLUDED.{column}" for column in INFRASTRUCTURE_COLUMNS[1:]
    )
    statement = text(
        f"""
        INSERT INTO features.infrastructure_features ({", ".join(INFRASTRUCTURE_COLUMNS)})
        VALUES ({", ".join(f":{column}" for column in INFRASTRUCTURE_COLUMNS)})
        ON CONFLICT (h3_index) DO UPDATE SET
            {assignments}
        """  # noqa: S608 - identifiers are fixed module constants, values are bound
    )
    parameters = features[list(INFRASTRUCTURE_COLUMNS)].to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(statement, parameters)
    return len(parameters)


def load_to_postgis(features: pd.DataFrame) -> bool:
    """Write aggregate features when optional local PostGIS is available."""
    if features.empty:
        return True
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    try:
        count = upsert_infrastructure_features(engine, features)
    except SQLAlchemyError:
        logger.info("PostGIS is unavailable; kept the local infrastructure artifact")
        return False
    finally:
        engine.dispose()
    logger.info("Upserted %s infrastructure row(s) into PostGIS", count)
    return True


def main() -> None:
    """Create and persist aggregate infrastructure features."""
    ensure_directories()
    features = create_infrastructure_features()
    output_path = DATA_DIR / "infrastructure_features.parquet"
    features.to_parquet(output_path, index=False)
    logger.info("Saved %s infrastructure row(s) to %s", len(features), output_path.name)
    load_to_postgis(features)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
