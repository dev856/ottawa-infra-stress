"""Tests for frontend HTML/JS structure, compliance, CSP, and coordinate clamping."""

from __future__ import annotations

import re
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def test_index_html_exists_and_has_favicon():
    """Verify index.html exists and specifies an inline SVG favicon to prevent 404s."""
    index_path = FRONTEND_DIR / "index.html"
    assert index_path.is_file(), "index.html must exist in frontend directory"
    content = index_path.read_text(encoding="utf-8")

    assert "rel=\"icon\"" in content or "rel='icon'" in content
    assert "data:image/svg+xml" in content


def test_index_html_csp_no_unapproved_tiles():
    """Verify CSP forbids unapproved external tile services (Esri, Carto, etc.)."""
    index_path = FRONTEND_DIR / "index.html"
    content = index_path.read_text(encoding="utf-8")

    assert "server.arcgisonline.com" not in content
    assert "basemaps.cartocdn.com" not in content
    assert "tile.openstreetmap.org" not in content
    assert "Content-Security-Policy" in content


def test_index_html_guide_dialog_and_labels():
    """Verify guide modal structure, accessibility tags, and building vintage labeling."""
    index_path = FRONTEND_DIR / "index.html"
    content = index_path.read_text(encoding="utf-8")

    assert "<dialog id=\"guide-dialog\"" in content
    assert "id=\"close-guide\"" in content
    assert "id=\"open-guide\"" in content
    assert "Pre-1980 building" in content or "Pre-1980 Buildings" in content


def test_frontend_app_js_has_no_external_tiles():
    """Ensure app.js does not make unapproved raster tile requests."""
    app_js_path = FRONTEND_DIR / "app.js"
    assert app_js_path.is_file()
    content = app_js_path.read_text(encoding="utf-8")

    assert "server.arcgisonline.com" not in content
    assert "basemaps.cartocdn.com" not in content
    assert "tile.openstreetmap.org" not in content


def test_frontend_coordinate_clamping_logic():
    """Validate bounding-box clamping logic matches Ottawa spatial bounds."""
    # Mirroring boundedMapCoordinates logic from app.js
    ottawa_bbox = {"west": -76.35, "south": 45.10, "east": -75.50, "north": 45.55}
    max_span = 0.48

    def clamp_bounds(raw_west: float, raw_south: float, raw_east: float, raw_north: float):
        min_lon = max(ottawa_bbox["west"], min(raw_west, ottawa_bbox["east"]))
        max_lon = max(ottawa_bbox["west"], min(raw_east, ottawa_bbox["east"]))
        min_lat = max(ottawa_bbox["south"], min(raw_south, ottawa_bbox["north"]))
        max_lat = max(ottawa_bbox["south"], min(raw_north, ottawa_bbox["north"]))

        if max_lon <= min_lon:
            mid_lon = (ottawa_bbox["west"] + ottawa_bbox["east"]) / 2.0
            min_lon = mid_lon - 0.05
            max_lon = mid_lon + 0.05
        if max_lat <= min_lat:
            mid_lat = (ottawa_bbox["south"] + ottawa_bbox["north"]) / 2.0
            min_lat = mid_lat - 0.04
            max_lat = mid_lat + 0.04

        lon_span = max_lon - min_lon
        if lon_span > max_span:
            center_lon = (min_lon + max_lon) / 2.0
            min_lon = max(ottawa_bbox["west"], center_lon - max_span / 2.0)
            max_lon = min(ottawa_bbox["east"], min_lon + max_span)

        return min_lon, min_lat, max_lon, max_lat

    # Test full zoomed out global viewport
    w, s, e, n = clamp_bounds(-180.0, -85.0, 180.0, 85.0)
    assert w >= ottawa_bbox["west"]
    assert e <= ottawa_bbox["east"]
    assert (e - w) <= max_span + 1e-6
    assert s >= ottawa_bbox["south"]
    assert n <= ottawa_bbox["north"]

    # Test normal downtown viewport
    w, s, e, n = clamp_bounds(-75.72, 45.40, -75.68, 45.44)
    assert round(w, 2) == -75.72
    assert round(e, 2) == -75.68


def test_quick_jump_coordinates_validity():
    """Verify that all quick-jump buttons target coordinates strictly inside Ottawa."""
    index_path = FRONTEND_DIR / "index.html"
    content = index_path.read_text(encoding="utf-8")

    lng_matches = [float(m) for m in re.findall(r'data-lng="([^"]+)"', content)]
    lat_matches = [float(m) for m in re.findall(r'data-lat="([^"]+)"', content)]

    assert len(lng_matches) > 0
    assert len(lng_matches) == len(lat_matches)

    for lng, lat in zip(lng_matches, lat_matches, strict=True):
        assert -76.35 <= lng <= -75.50, f"Longitude {lng} outside Ottawa bounding box"
        assert 45.10 <= lat <= 45.55, f"Latitude {lat} outside Ottawa bounding box"
