"""Generate complete H3 polygon coverage for the configured Ottawa extent.

The grid is created with H3 polygon filling plus boundary neighbors, not sparse
point sampling. Database writes use an idempotent parameterized upsert.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import h3
import pandas as pd
from shapely.geometry import Polygon, box, mapping
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from .config import (
        DATA_DIR,
        H3_RESOLUTION,
        OTTAWA_BBOX_EAST,
        OTTAWA_BBOX_NORTH,
        OTTAWA_BBOX_SOUTH,
        OTTAWA_BBOX_WEST,
        ensure_directories,
        get_database_url,
    )
else:
    try:
        from .config import (
            DATA_DIR,
            H3_RESOLUTION,
            OTTAWA_BBOX_EAST,
            OTTAWA_BBOX_NORTH,
            OTTAWA_BBOX_SOUTH,
            OTTAWA_BBOX_WEST,
            ensure_directories,
            get_database_url,
        )
    except ImportError:  # pragma: no cover - supports direct script imports
        from config import (
            DATA_DIR,
            H3_RESOLUTION,
            OTTAWA_BBOX_EAST,
            OTTAWA_BBOX_NORTH,
            OTTAWA_BBOX_SOUTH,
            OTTAWA_BBOX_WEST,
            ensure_directories,
            get_database_url,
        )

logger = logging.getLogger(__name__)

OTTAWA_BBOX = (
    OTTAWA_BBOX_WEST,
    OTTAWA_BBOX_SOUTH,
    OTTAWA_BBOX_EAST,
    OTTAWA_BBOX_NORTH,
)


def latlng_to_h3_cell(latitude: float, longitude: float, resolution: int) -> str:
    """Convert a validated latitude/longitude pair to one H3 index."""
    if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
        raise ValueError("latitude/longitude is outside the valid world range")
    if not 0 <= resolution <= 15:
        raise ValueError("resolution must be between 0 and 15")
    return str(h3.latlng_to_cell(latitude, longitude, resolution))


def h3_cell_to_polygon(cell: str) -> Polygon:
    """Convert one valid H3 index into an EPSG:4326 Shapely polygon."""
    if not h3.is_valid_cell(cell):
        raise ValueError("cell must be a valid H3 index")
    latitude_longitude = h3.cell_to_boundary(cell)
    longitude_latitude = [(longitude, latitude) for latitude, longitude in latitude_longitude]
    polygon = Polygon(longitude_latitude)
    if not polygon.is_valid:
        raise ValueError("H3 produced an invalid polygon")
    return polygon


def h3_resolution_to_edge_length_km(resolution: int = H3_RESOLUTION) -> float:
    """Return H3's documented average hexagon edge length in kilometres."""
    if not 0 <= resolution <= 15:
        raise ValueError("resolution must be between 0 and 15")
    return float(h3.average_hexagon_edge_length(resolution, unit="km"))


def validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    """Validate west/south/east/north order and world-coordinate limits."""
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox must be ordered west, south, east, north")


def generate_h3_grid(
    bbox: tuple[float, float, float, float] = OTTAWA_BBOX,
    resolution: int = H3_RESOLUTION,
    lat_step: float | None = None,
    lon_step: float | None = None,
) -> pd.DataFrame:
    """Return every H3 cell whose polygon intersects the bounding rectangle.

    ``lat_step`` and ``lon_step`` remain accepted for command compatibility but
    are ignored because polygon filling no longer samples points.
    """
    del lat_step, lon_step
    validate_bbox(bbox)
    if not 0 <= resolution <= 15:
        raise ValueError("resolution must be between 0 and 15")

    west, south, east, north = bbox
    requested_polygon = box(west, south, east, north)
    centre_cells = set(h3.geo_to_cells(mapping(requested_polygon), resolution))
    candidate_cells = set(centre_cells)
    for cell in centre_cells:
        candidate_cells.update(h3.grid_disk(cell, 1))

    records: list[dict[str, Any]] = []
    for cell in sorted(candidate_cells):
        polygon = h3_cell_to_polygon(cell)
        if not polygon.intersects(requested_polygon):
            continue
        centroid = polygon.centroid
        records.append(
            {
                "h3_index": cell,
                "h3_resolution": resolution,
                "centroid_lat": float(centroid.y),
                "centroid_lon": float(centroid.x),
                "geometry": polygon,
            }
        )
    logger.info("Generated %s H3 cell(s) at resolution %s", len(records), resolution)
    return pd.DataFrame(records)


def upsert_h3_grid(engine: Engine, grid: gpd.GeoDataFrame) -> int:
    """Idempotently upsert grid cells while preserving table indexes."""
    if grid.empty:
        return 0
    statement = text(
        """
        INSERT INTO features.h3_grid (
            h3_index, h3_resolution, centroid_lat, centroid_lon, geometry
        ) VALUES (
            :h3_index, :h3_resolution, :centroid_lat, :centroid_lon,
            ST_GeomFromText(:geometry_wkt, 4326)
        )
        ON CONFLICT (h3_index) DO UPDATE SET
            h3_resolution = EXCLUDED.h3_resolution,
            centroid_lat = EXCLUDED.centroid_lat,
            centroid_lon = EXCLUDED.centroid_lon,
            geometry = EXCLUDED.geometry
        """
    )
    parameters = [
        {
            "h3_index": row.h3_index,
            "h3_resolution": int(row.h3_resolution),
            "centroid_lat": float(row.centroid_lat),
            "centroid_lon": float(row.centroid_lon),
            "geometry_wkt": row.geometry.wkt,
        }
        for row in grid.itertuples(index=False)
    ]
    with engine.begin() as connection:
        connection.execute(statement, parameters)
    return len(parameters)


def load_to_postgis(grid: gpd.GeoDataFrame) -> bool:
    """Write the grid when optional local PostGIS is available."""
    if grid.empty:
        return True
    engine = create_engine(get_database_url(), pool_pre_ping=True)
    try:
        count = upsert_h3_grid(engine, grid)
    except SQLAlchemyError:
        logger.info("PostGIS is unavailable; kept the local H3 Parquet artifact")
        return False
    finally:
        engine.dispose()
    logger.info("Upserted %s H3 grid cell(s) into PostGIS", count)
    return True


def main() -> None:
    """Generate and persist the local H3 grid artifact."""
    ensure_directories()
    dataframe = generate_h3_grid()
    grid = gpd.GeoDataFrame(dataframe, geometry="geometry", crs="EPSG:4326")
    output_path = DATA_DIR / "h3_grid.parquet"
    grid.to_parquet(output_path, index=False)
    logger.info("Saved %s H3 grid cell(s) to %s", len(grid), output_path.name)
    load_to_postgis(grid)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
