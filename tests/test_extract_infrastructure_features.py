"""Tests for infrastructure feature extraction."""

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from src.extract_infrastructure_features import (
    calculate_building_features,
    calculate_line_density,
    create_infrastructure_features,
)


def _make_sample_h3_gdf():
    poly = Polygon([(-75.70, 45.41), (-75.68, 45.41), (-75.68, 45.43), (-75.70, 45.43)])
    return gpd.GeoDataFrame(
        [{
            "h3_index": "882bb2c51ffffff",
            "centroid_lat": 45.42,
            "centroid_lon": -75.69,
            "geometry": poly,
        }],
        crs="EPSG:4326",
    )


def test_calculate_line_density_metric():
    """Test line density computes metric length (km) in EPSG:3347."""
    h3_gdf = _make_sample_h3_gdf()
    line = LineString([(-75.695, 45.415), (-75.685, 45.425)])
    lines_gdf = gpd.GeoDataFrame([{"objectid": 1, "geometry": line}], crs="EPSG:4326")

    df = calculate_line_density(lines_gdf, h3_gdf, prefix="water")
    assert not df.empty
    assert df["line_count_water"].iloc[0] == 1
    assert df["line_length_km_water"].iloc[0] > 0.5


def test_line_density_clips_geometry_to_cell():
    """A long crossing line contributes only the portion inside the cell."""
    h3_gdf = _make_sample_h3_gdf()
    line = LineString([(-76.0, 45.42), (-75.0, 45.42)])
    lines_gdf = gpd.GeoDataFrame(
        [{"objectid": 1, "geometry": line}],
        crs="EPSG:4326",
    )
    full_length_km = lines_gdf.to_crs(3347).geometry.length.iloc[0] / 1000

    result = calculate_line_density(lines_gdf, h3_gdf, prefix="water")

    assert 0 < result["line_length_km_water"].iloc[0] < full_length_km / 10


def test_calculate_building_features():
    """Test building vintage calculation."""
    h3_gdf = _make_sample_h3_gdf()
    b1 = Point(-75.69, 45.42)
    b2 = Point(-75.691, 45.421)
    buildings_gdf = gpd.GeoDataFrame(
        [
            {"objectid": 101, "year_built": 1965, "geometry": b1},
            {"objectid": 102, "year_built": 1995, "geometry": b2},
        ],
        crs="EPSG:4326",
    )

    df = calculate_building_features(buildings_gdf, h3_gdf)
    assert not df.empty
    assert df["building_count"].iloc[0] == 2
    assert df["median_year_built"].iloc[0] == 1980.0
    assert df["pct_pre_1980"].iloc[0] == 0.5


def test_create_infrastructure_features_mock_mode():
    """Test end-to-end infrastructure feature generation in mock mode."""
    df = create_infrastructure_features(mode="mock")
    assert not df.empty
    assert "h3_index" in df.columns
    assert "line_length_km_water" in df.columns
    assert "line_length_km_road" in df.columns
    assert "building_count" in df.columns
    assert "pct_pre_1980" in df.columns
