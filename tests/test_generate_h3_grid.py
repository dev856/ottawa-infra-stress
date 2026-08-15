"""Tests for H3 grid generation."""

import h3
import pytest
from shapely.geometry import Polygon, box

from src.generate_h3_grid import (
    generate_h3_grid,
    h3_cell_to_polygon,
    h3_resolution_to_edge_length_km,
    latlng_to_h3_cell,
)


def test_latlng_to_h3_cell():
    """Test converting lat/lon to H3 cell string."""
    cell = latlng_to_h3_cell(45.4215, -75.6972, 8)
    assert isinstance(cell, str)
    assert len(cell) > 5


def test_h3_cell_to_polygon():
    """Test converting H3 cell to Shapely Polygon."""
    cell = latlng_to_h3_cell(45.4215, -75.6972, 8)
    poly = h3_cell_to_polygon(cell)
    assert isinstance(poly, Polygon)
    assert poly.is_valid
    assert -76.0 < poly.centroid.x < -75.0
    assert 45.0 < poly.centroid.y < 46.0


def test_generate_h3_grid_small_bbox():
    """Test generating H3 grid over a small bounded area."""
    small_bbox = (-75.72, 45.40, -75.68, 45.44)
    df = generate_h3_grid(bbox=small_bbox, resolution=8)

    assert not df.empty
    assert "h3_index" in df.columns
    assert "geometry" in df.columns
    assert "centroid_lat" in df.columns
    assert "centroid_lon" in df.columns
    assert all(isinstance(geom, Polygon) for geom in df["geometry"])
    assert all(geom.intersects(box(*small_bbox)) for geom in df["geometry"])
    assert all(h3.is_valid_cell(cell) for cell in df["h3_index"])


def test_h3_resolution_edge_length():
    """Test resolution edge length helper."""
    assert h3_resolution_to_edge_length_km(8) == pytest.approx(
        h3.average_hexagon_edge_length(8, unit="km")
    )


def test_generate_h3_grid_rejects_reversed_bbox():
    """A reversed bbox fails before any expensive spatial work."""
    with pytest.raises(ValueError, match="bbox"):
        generate_h3_grid(bbox=(-75.0, 45.0, -76.0, 46.0), resolution=8)
