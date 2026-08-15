const API_BASE_URL = (() => {
    if (window.location.protocol === "http:" || window.location.protocol === "https:") {
        const hostname = window.location.hostname || "127.0.0.1";
        const port = window.location.port;
        if (port === "8000") {
            return "";
        }
        return `${window.location.protocol}//${hostname}:8000`;
    }
    return "http://127.0.0.1:8000";
})();
const MAP_SOURCE_ID = "h3-risk-source";
const MAP_FILL_LAYER_ID = "h3-risk-fill";
const MAP_LINE_LAYER_ID = "h3-risk-line";
const MAX_REQUEST_SPAN = 0.48;

const state = {
    map: null,
    currentGeoJSON: null,
    minimumRisk: 0,
    hexagonOpacity: 0.70,
    currentView: "neutral",
    scenario: null,
    abortController: null,
    debounceTimer: null,
    hoverPopup: null,
};

function element(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const target = element(id);
    if (target) {
        target.textContent = String(value);
    }
}

function setStatus(message, className) {
    const status = element("api-status");
    status.textContent = message;
    status.className = `status ${className}`;
}

function showError(message) {
    const target = element("error-message");
    target.textContent = message;
    target.classList.remove("hidden");
}

function clearError() {
    const target = element("error-message");
    target.textContent = "";
    target.classList.add("hidden");
}

function setLoading(isLoading) {
    element("loading-message").classList.toggle("hidden", !isLoading);
}

async function fetchJson(path, options = {}) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal: options.signal,
    });
    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            const body = await response.json();
            if (typeof body.detail === "string") {
                detail = body.detail;
            }
        } catch (_error) {
            // The HTTP status remains a safe fallback when the body is not JSON.
        }
        throw new Error(detail);
    }
    return response.json();
}

function formatPercent(value) {
    return `${(Number(value) * 100).toFixed(1)}%`;
}

async function loadWeather() {
    const weather = await fetchJson("/weather-summary");
    setText("weather-temperature", `${Number(weather.temperature_c).toFixed(1)} °C`);
    setText("weather-heat-index", `${Number(weather.heat_index_c).toFixed(1)} °C`);
    setText("weather-dry-days", Number(weather.consecutive_dry_days).toFixed(1));
    setText("weather-rainfall", `${Number(weather.rainfall_48h).toFixed(1)} mm`);
    setText("weather-mode", weather.is_synthetic ? "Synthetic" : "Approved source");
    setText(
        "weather-observed-at",
        `${weather.station_name} · row timestamp ${new Date(weather.observed_at).toLocaleString()}`,
    );
}

async function loadMetrics() {
    const response = await fetchJson("/metrics");
    const metrics = response.metrics || {};
    setText("model-warning", response.target_warning);
    setText("metric-pr-auc", metrics.pr_auc == null ? "—" : Number(metrics.pr_auc).toFixed(3));
    setText(
        "metric-f1",
        metrics.f1_at_threshold == null ? "—" : Number(metrics.f1_at_threshold).toFixed(3),
    );
    setText(
        "metric-threshold",
        metrics.decision_threshold == null ? "—" : Number(metrics.decision_threshold).toFixed(3),
    );
}

const OTTAWA_BBOX = {
    west: -76.35,
    south: 45.25,
    east: -75.50,
    north: 45.60,
};

function boundedMapCoordinates() {
    const bounds = state.map.getBounds();
    const center = state.map.getCenter();
    const halfSpan = MAX_REQUEST_SPAN / 2;

    let west = Math.max(bounds.getWest(), center.lng - halfSpan);
    let east = Math.min(bounds.getEast(), center.lng + halfSpan);
    let south = Math.max(bounds.getSouth(), center.lat - halfSpan);
    let north = Math.min(bounds.getNorth(), center.lat + halfSpan);

    west = Math.max(OTTAWA_BBOX.west, Math.min(OTTAWA_BBOX.east - 0.02, west));
    east = Math.min(OTTAWA_BBOX.east, Math.max(OTTAWA_BBOX.west + 0.02, east));
    south = Math.max(OTTAWA_BBOX.south, Math.min(OTTAWA_BBOX.north - 0.02, south));
    north = Math.min(OTTAWA_BBOX.north, Math.max(OTTAWA_BBOX.south + 0.02, north));

    if (east - west > MAX_REQUEST_SPAN) {
        const midLon = (east + west) / 2;
        west = midLon - halfSpan;
        east = midLon + halfSpan;
    }
    if (north - south > MAX_REQUEST_SPAN) {
        const midLat = (north + south) / 2;
        south = midLat - halfSpan;
        north = midLat + halfSpan;
    }

    if (west < OTTAWA_BBOX.west) {
        west = OTTAWA_BBOX.west;
        east = Math.min(OTTAWA_BBOX.east, west + MAX_REQUEST_SPAN);
    }
    if (east > OTTAWA_BBOX.east) {
        east = OTTAWA_BBOX.east;
        west = Math.max(OTTAWA_BBOX.west, east - MAX_REQUEST_SPAN);
    }
    if (south < OTTAWA_BBOX.south) {
        south = OTTAWA_BBOX.south;
        north = Math.min(OTTAWA_BBOX.north, south + MAX_REQUEST_SPAN);
    }
    if (north > OTTAWA_BBOX.north) {
        north = OTTAWA_BBOX.north;
        south = Math.max(OTTAWA_BBOX.south, north - MAX_REQUEST_SPAN);
    }

    return { west, south, east, north };
}

function riskMapPath() {
    const bounds = boundedMapCoordinates();
    const parameters = new URLSearchParams({
        min_lon: bounds.west.toFixed(6),
        min_lat: bounds.south.toFixed(6),
        max_lon: bounds.east.toFixed(6),
        max_lat: bounds.north.toFixed(6),
        max_features: "500",
    });
    if (state.scenario) {
        parameters.set("sim_temp_c", String(state.scenario.temperature));
        parameters.set("sim_humidity", String(state.scenario.humidity));
        parameters.set("sim_dry_days", String(state.scenario.dryDays));
    }
    return `/risk-map?${parameters.toString()}`;
}

function updateStatistics(geojson) {
    const features = Array.isArray(geojson.features) ? geojson.features : [];
    const scores = features.map((feature) => Number(feature.properties.risk_score));
    const highCount = scores.filter((score) => score >= 0.7).length;
    const average = scores.length
        ? scores.reduce((total, score) => total + score, 0) / scores.length
        : null;
    const maximum = scores.length ? Math.max(...scores) : null;
    setText("cell-count", features.length.toLocaleString());
    setText("high-count", highCount.toLocaleString());
    setText("average-risk", average == null ? "—" : formatPercent(average));
    setText("maximum-risk", maximum == null ? "—" : formatPercent(maximum));

    const metadata = geojson.metadata || {};
    const truncation = metadata.truncated ? " · response limited" : "";
    setText(
        "model-provenance",
        `Target: ${metadata.target_mode || "unknown"} · model ${metadata.model_version || "unknown"}${truncation}`,
    );
}

function ensureRiskLayers() {
    if (!state.map.getSource(MAP_SOURCE_ID)) {
        state.map.addSource(MAP_SOURCE_ID, {
            type: "geojson",
            data: state.currentGeoJSON,
        });
    }
    if (!state.map.getLayer(MAP_FILL_LAYER_ID)) {
        state.map.addLayer({
            id: MAP_FILL_LAYER_ID,
            type: "fill",
            source: MAP_SOURCE_ID,
            paint: {
                "fill-color": [
                    "interpolate",
                    ["linear"],
                    ["get", "risk_score"],
                    0,
                    "#2b8a66",
                    0.3,
                    "#d48a10",
                    0.7,
                    "#c13d3d",
                    1,
                    "#701c28",
                ],
                "fill-opacity": state.hexagonOpacity,
            },
        });
    }
    if (!state.map.getLayer(MAP_LINE_LAYER_ID)) {
        state.map.addLayer({
            id: MAP_LINE_LAYER_ID,
            type: "line",
            source: MAP_SOURCE_ID,
            paint: {
                "line-color": "#ffffff",
                "line-opacity": 0.85,
                "line-width": 0.8,
            },
        });
    }
    applyRiskFilter();
}

function applyRiskFilter() {
    if (!state.map || !state.map.getLayer(MAP_FILL_LAYER_ID)) {
        return;
    }
    const filter = state.minimumRisk > 0
        ? [">=", ["get", "risk_score"], state.minimumRisk]
        : null;
    state.map.setFilter(MAP_FILL_LAYER_ID, filter);
    state.map.setFilter(MAP_LINE_LAYER_ID, filter);
}

async function loadRiskMap() {
    if (!state.map) {
        return;
    }
    if (state.abortController) {
        state.abortController.abort();
    }
    state.abortController = new AbortController();
    setLoading(true);
    clearError();
    try {
        const geojson = await fetchJson(riskMapPath(), {
            signal: state.abortController.signal,
        });
        state.currentGeoJSON = geojson;
        updateStatistics(geojson);
        if (state.map.getSource(MAP_SOURCE_ID)) {
            state.map.getSource(MAP_SOURCE_ID).setData(geojson);
        } else {
            ensureRiskLayers();
        }
        setStatus("Local API connected · synthetic target", "status-ready");
    } catch (error) {
        if (error.name !== "AbortError") {
            if (error.message && (error.message.includes("Ottawa") || error.message.includes("Bounding box"))) {
                showError("Pan or zoom towards Ottawa to load risk data.");
            } else {
                setStatus("Local API unavailable", "status-error");
                showError(`Could not load aggregate risk data: ${error.message}`);
            }
        }
    } finally {
        setLoading(false);
    }
}

function popupContent(properties) {
    const container = document.createElement("div");
    const title = document.createElement("strong");
    const details = document.createElement("p");
    title.textContent = `${properties.risk_level} band · ${formatPercent(properties.risk_score)}`;
    details.textContent = `${properties.h3_index} · ${Number(properties.water_km || 0).toFixed(2)} km water mains`;
    details.className = "supporting-text";
    container.append(title, details);
    return container;
}

const MAP_HIGHLIGHT_LAYER_ID = "h3-selected-highlight";

function ensureHighlightLayer() {
    if (state.map && !state.map.getLayer(MAP_HIGHLIGHT_LAYER_ID) && state.map.getSource(MAP_SOURCE_ID)) {
        state.map.addLayer({
            id: MAP_HIGHLIGHT_LAYER_ID,
            type: "line",
            source: MAP_SOURCE_ID,
            paint: {
                "line-color": "#ffff00",
                "line-width": 3.0,
                "line-opacity": 0.95,
            },
            filter: ["==", ["get", "h3_index"], ""],
        });
    }
}

function highlightSelectedCell(h3Index) {
    ensureHighlightLayer();
    if (state.map && state.map.getLayer(MAP_HIGHLIGHT_LAYER_ID)) {
        state.map.setFilter(MAP_HIGHLIGHT_LAYER_ID, ["==", ["get", "h3_index"], h3Index]);
    }
}

async function selectCell(properties) {
    setText("selected-h3", properties.h3_index);
    setText("selected-score", formatPercent(properties.risk_score));
    setText("selected-level", `${properties.risk_level} display band`);
    setText("selected-water", `${Number(properties.water_km || 0).toFixed(2)} km`);

    if (state.map) {
        highlightSelectedCell(properties.h3_index);
    }

    try {
        const response = await fetchJson(`/risk/${encodeURIComponent(properties.h3_index)}`);
        const features = response.features || {};
        const bCount = Number(features.building_count || 0);
        const prePct = Number(features.pct_pre_1980 || 0) * 100;
        const postPct = Math.max(0, 100 - prePct);

        setText("selected-roads", `${Number(features.line_length_km_road || 0).toFixed(2)} km`);
        setText("selected-buildings", `${bCount.toLocaleString()} structures`);
        setText("selected-year", Math.round(Number(features.median_year_built || 0)) || "—");
        setText("selected-vintage", `${prePct.toFixed(1)}%`);

        setText("vintage-pre-pct", `${prePct.toFixed(1)}%`);
        setText("vintage-post-pct", `${postPct.toFixed(1)}%`);
        const barFill = element("vintage-bar-fill");
        if (barFill) {
            barFill.style.width = `${prePct}%`;
        }

        const densityBadgeContainer = element("density-badge-container");
        const densityBadge = element("density-badge");
        const demandBadge = element("demand-badge");
        const vintageBreakdown = element("vintage-breakdown");

        if (densityBadgeContainer) {
            densityBadgeContainer.classList.remove("hidden");
        }
        if (vintageBreakdown) {
            vintageBreakdown.classList.remove("hidden");
        }

        if (densityBadge) {
            if (bCount >= 130) {
                densityBadge.textContent = "🏢 Dense Urban Core";
                densityBadge.className = "badge badge-high";
            } else if (bCount >= 40) {
                densityBadge.textContent = "🏘️ Suburban Medium Density";
                densityBadge.className = "badge badge-med";
            } else {
                densityBadge.textContent = "🌲 Low Density Fringe";
                densityBadge.className = "badge badge-low";
            }
        }

        if (demandBadge) {
            const waterKm = Number(properties.water_km || 0);
            if (bCount > 90 && waterKm > 3.0) {
                demandBadge.textContent = "🏢 High Asset Concentration";
                demandBadge.className = "badge badge-high";
            } else {
                demandBadge.textContent = "💧 Moderate Asset Density";
                demandBadge.className = "badge badge-low";
            }
        }

        setText(
            "selected-provenance",
            `Target: ${response.target_mode} · model ${response.model_version}. Spatial building & pipe density proxy.`,
        );
    } catch (error) {
        showError(`Could not load the selected H3 aggregate: ${error.message}`);
    }
}

const MAP_THEMES = {
    neutral: { bg: "#dfe7e6", line: "#ffffff" },
    dark: { bg: "#182226", line: "#2c3e44" },
    slate: { bg: "#d4dcdd", line: "#ffffff" },
    terrain: { bg: "#d2ded9", line: "#ffffff" },
};

function initializeMap() {
    if (typeof window.maplibregl === "undefined") {
        setStatus("MapLibre failed to load", "status-error");
        showError("MapLibre GL JS could not be loaded. Check the pinned CDN asset or self-host it.");
        setLoading(false);
        return;
    }
    state.map = new window.maplibregl.Map({
        container: "map",
        center: [-75.6972, 45.4215],
        zoom: 11.2,
        minZoom: 9.0,
        maxZoom: 16,
        maxBounds: [
            [-76.50, 45.15],
            [-75.35, 45.70],
        ],
        attributionControl: false,
        style: {
            version: 8,
            sources: {},
            layers: [
                {
                    id: "neutral-background",
                    type: "background",
                    paint: { "background-color": "#dfe7e6" },
                },
            ],
        },
    });
    state.map.addControl(new window.maplibregl.NavigationControl({ showCompass: false }), "top-right");
    state.hoverPopup = new window.maplibregl.Popup({ closeButton: false, closeOnClick: false });
    state.map.on("load", () => {
        loadRiskMap();
    });
    state.map.on("moveend", () => {
        window.clearTimeout(state.debounceTimer);
        state.debounceTimer = window.setTimeout(loadRiskMap, 350);
    });
    state.map.on("click", MAP_FILL_LAYER_ID, (event) => {
        if (!event.features || event.features.length === 0) {
            return;
        }
        selectCell(event.features[0].properties);
    });
    state.map.on("mousemove", MAP_FILL_LAYER_ID, (event) => {
        if (!event.features || event.features.length === 0) {
            return;
        }
        state.map.getCanvas().style.cursor = "pointer";
        state.hoverPopup
            .setLngLat(event.lngLat)
            .setDOMContent(popupContent(event.features[0].properties))
            .addTo(state.map);
    });
    state.map.on("mouseleave", MAP_FILL_LAYER_ID, () => {
        state.map.getCanvas().style.cursor = "";
        state.hoverPopup.remove();
    });
}

function setMapView(viewName) {
    state.currentView = viewName;
    document.querySelectorAll(".view-btn").forEach((btn) => {
        const isActive = btn.getAttribute("data-view") === viewName;
        btn.classList.toggle("active", isActive);
        btn.setAttribute("aria-checked", isActive ? "true" : "false");
    });

    if (!state.map) {
        return;
    }

    const theme = MAP_THEMES[viewName] || MAP_THEMES.neutral;
    if (state.map.getLayer("neutral-background")) {
        state.map.setPaintProperty("neutral-background", "background-color", theme.bg);
    }
    if (state.map.getLayer(MAP_LINE_LAYER_ID)) {
        state.map.setPaintProperty(MAP_LINE_LAYER_ID, "line-color", theme.line);
    }
}

function bindControls() {
    const riskFilter = element("risk-filter");
    riskFilter.addEventListener("input", () => {
        state.minimumRisk = Number(riskFilter.value) / 100;
        setText("risk-filter-value", `${riskFilter.value}%`);
        applyRiskFilter();
    });

    const opacityFilter = element("opacity-filter");
    if (opacityFilter) {
        opacityFilter.addEventListener("input", () => {
            const opacity = Number(opacityFilter.value) / 100;
            state.hexagonOpacity = opacity;
            setText("opacity-filter-value", `${opacityFilter.value}%`);
            if (state.map && state.map.getLayer(MAP_FILL_LAYER_ID)) {
                state.map.setPaintProperty(MAP_FILL_LAYER_ID, "fill-opacity", opacity);
            }
        });
    }

    document.querySelectorAll(".view-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const view = btn.getAttribute("data-view");
            if (view) {
                setMapView(view);
            }
        });
    });

    document.querySelectorAll(".jump-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const lng = Number(btn.getAttribute("data-lng"));
            const lat = Number(btn.getAttribute("data-lat"));
            const zoom = Number(btn.getAttribute("data-zoom") || "13.0");
            if (state.map && Number.isFinite(lng) && Number.isFinite(lat)) {
                state.map.flyTo({
                    center: [lng, lat],
                    zoom: zoom,
                    essential: true,
                    speed: 1.4,
                });
            }
        });
    });

    const temperature = element("scenario-temperature");
    const humidity = element("scenario-humidity");
    const dryDays = element("scenario-dry-days");
    temperature.addEventListener("input", () => {
        setText("scenario-temperature-value", `${temperature.value} °C`);
    });
    humidity.addEventListener("input", () => {
        setText("scenario-humidity-value", `${humidity.value}%`);
    });
    dryDays.addEventListener("input", () => {
        setText("scenario-dry-days-value", dryDays.value);
    });

    element("apply-scenario").addEventListener("click", () => {
        state.scenario = {
            temperature: Number(temperature.value),
            humidity: Number(humidity.value),
            dryDays: Number(dryDays.value),
        };
        loadRiskMap();
    });
    element("reset-scenario").addEventListener("click", () => {
        state.scenario = null;
        temperature.value = "34";
        humidity.value = "65";
        dryDays.value = "6";
        temperature.dispatchEvent(new Event("input"));
        humidity.dispatchEvent(new Event("input"));
        dryDays.dispatchEvent(new Event("input"));
        loadRiskMap();
    });

    bindGuideModal();
}

function bindGuideModal() {
    const dialog = element("guide-dialog");
    const openBtn = element("open-guide");
    const closeBtn = element("close-guide");
    if (!dialog || !openBtn || !closeBtn) {
        return;
    }
    openBtn.addEventListener("click", () => {
        if (typeof dialog.showModal === "function") {
            dialog.showModal();
        } else {
            dialog.setAttribute("open", "");
        }
    });
    closeBtn.addEventListener("click", () => {
        if (typeof dialog.close === "function") {
            dialog.close();
        } else {
            dialog.removeAttribute("open");
        }
    });
    dialog.addEventListener("click", (event) => {
        if (event.target === dialog) {
            if (typeof dialog.close === "function") {
                dialog.close();
            } else {
                dialog.removeAttribute("open");
            }
        }
    });
}

document.addEventListener("DOMContentLoaded", async () => {
    bindControls();
    initializeMap();
    try {
        await Promise.all([loadWeather(), loadMetrics()]);
    } catch (error) {
        setStatus("Local API unavailable", "status-error");
        showError(`Could not load API summary data: ${error.message}`);
    }
});
