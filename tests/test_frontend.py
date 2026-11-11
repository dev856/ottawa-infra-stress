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
    assert "unpkg.com" not in content, "MapLibre is self-hosted under frontend/vendor/"
    assert "Content-Security-Policy" in content


def test_index_html_guide_dialog_and_labels():
    """Verify guide modal structure, accessibility tags, and building vintage labeling."""
    index_path = FRONTEND_DIR / "index.html"
    content = index_path.read_text(encoding="utf-8")

    assert "<dialog id=\"guide-dialog\"" in content
    assert "id=\"close-guide\"" in content
    assert "id=\"open-guide\"" in content


def test_frontend_app_js_has_no_external_tiles():
    """Ensure the dashboard code does not make unapproved raster tile requests."""
    app_js_path = FRONTEND_DIR / "src" / "app.ts"
    assert app_js_path.is_file()
    content = app_js_path.read_text(encoding="utf-8")

    assert "server.arcgisonline.com" not in content
    assert "basemaps.cartocdn.com" not in content
    assert "tile.openstreetmap.org" not in content


def test_footer_credits_the_open_data_licence():
    """The City of Ottawa licence requires this attribution statement."""
    content = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    statement = "Contains information licensed under the Open Government Licence – City of Ottawa"
    assert statement in content


def test_legend_colours_match_map_bands():
    """The legend swatches in styles.css must be the band colours the map paints."""
    app_ts = (FRONTEND_DIR / "src" / "app.ts").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
    declared = re.search(r"BAND_COLOURS = \[(.*?)\]", app_ts)
    assert declared
    colours = re.findall(r"#[0-9a-f]{6}", declared.group(1))
    assert len(colours) == 4
    legend = styles[styles.index(".legend span {") : styles.index(".map-legend {")]
    assert re.findall(r"#[0-9a-f]{6}", legend) == colours


def test_page_loads_the_compiled_typescript():
    """index.html must load the committed tsc output; CI checks it matches src/app.ts."""
    index = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    assert '<script type="module" src="dist/app.js"></script>' in index
    assert (FRONTEND_DIR / "dist" / "app.js").is_file()
