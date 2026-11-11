import type * as MapLibre from "maplibre-gl";

declare global {
    interface Window {
        maplibregl?: typeof MapLibre;
    }
}

// Response shapes from the FastAPI backend (api/main.py).
interface WardProperties {
    ward: number;
    name: string;
    sector: string;
    total_requests_12_months: number;
    spikes: number;
    /** Colour band for the chosen map metric; -1 or absent means "too few to say". */
    band?: number;
}

type Metric = "requests" | "median_days" | "spikes";

interface WardMetric {
    ward: number;
    requests: number;
    median_days: number | null;
    spikes: number;
}

interface Remaining {
    waited_days: number;
    scope: "ward" | "city";
    sample_size: number;
    enough_history: boolean;
    usual_more_days: number | null;
    nine_in_ten_more_days: number | null;
}

type WardFeature = GeoJSON.Feature<GeoJSON.Polygon | GeoJSON.MultiPolygon, WardProperties>;

interface WardCollection extends GeoJSON.FeatureCollection<WardFeature["geometry"], WardProperties> {
    features: WardFeature[];
    as_of: string;
    attribution: string;
}

interface Spike {
    ward: number;
    ward_name: string;
    type: string;
    this_week: number;
    usual_per_week: number;
}

interface ReportEntry {
    type: string;
    count: number;
    ward_median_days: number | null;
    city_median_days: number | null;
    batch_closed: boolean;
}

interface Report {
    ward: number | null;
    name: string;
    total_requests: number;
    top_requests: ReportEntry[];
    spikes: Spike[];
}

interface RequestType {
    type: string;
    category: string;
    count: number;
}

interface Estimate {
    /** Null when too few requests with a trustworthy closing date exist yet. */
    usual_days: number | null;
    nine_in_ten_days: number | null;
    batch_closed: boolean;
}

const API_BASE_URL = (() => {
    const { protocol, hostname, port } = window.location;
    if (protocol === "http:" || protocol === "https:") {
        return port === "8000" ? "" : `${protocol}//${hostname || "127.0.0.1"}:8000`;
    }
    return "http://127.0.0.1:8000";
})();
const MAP_SOURCE_ID = "wards";
const MAP_FILL_LAYER_ID = "wards-fill";
const MAP_LINE_LAYER_ID = "wards-line";
const MAP_HIGHLIGHT_LAYER_ID = "wards-selected";
const SPIKES_SHOWN = 5;
// Lightest to darkest; the legend in styles.css uses the same four colours.
const BAND_COLOURS = ["#c9e3d6", "#7fbf9f", "#3d8f6b", "#1f5c43"] as const;
const NO_DATA_COLOUR = "#d9dedc";
const METRIC_LABELS: Record<Metric, string> = {
    requests: "Requests, last 12 months",
    median_days: "Typical days to close, last 12 months",
    spikes: "Unusual jumps this week",
};
const OTTAWA_VIEW = { center: [-75.75, 45.3] as [number, number], zoom: 9.2 };
// City labels end in short codes; spelled out so residents can read them.
const ABBREVIATIONS: Record<string, string> = {
    SWC: "missed curbside pickup",
    SW: "Waste",
};

const state: {
    map: MapLibre.Map | null;
    mapReady: Promise<void>;
    wards: WardCollection | null;
    selectedWard: number | null;
    hoverPopup: MapLibre.Popup | null;
    showAllSpikes: boolean;
    spikes: Spike[];
    metric: Metric;
    wardMetrics: Map<number, WardMetric>;
} = {
    map: null,
    mapReady: Promise.resolve(),
    wards: null,
    selectedWard: null,
    hoverPopup: null,
    showAllSpikes: false,
    spikes: [],
    metric: "requests",
    wardMetrics: new Map(),
};

function element<T extends HTMLElement = HTMLElement>(id: string): T {
    const found = document.getElementById(id);
    if (!found) {
        throw new Error(`Missing page element #${id}`);
    }
    return found as T;
}

function setText(id: string, value: string | number): void {
    element(id).textContent = String(value);
}

function setStatus(message: string, className: string): void {
    const status = element("api-status");
    status.textContent = message;
    status.className = `status ${className}`;
}

function showError(message: string): void {
    const target = element("error-message");
    target.textContent = message;
    target.classList.remove("hidden");
}

function clearError(): void {
    const target = element("error-message");
    target.textContent = "";
    target.classList.add("hidden");
}

function setLoading(isLoading: boolean): void {
    element("loading-message").classList.toggle("hidden", !isLoading);
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

async function fetchJson<T>(path: string): Promise<T> {
    const response = await fetch(`${API_BASE_URL}${path}`, {
        method: "GET",
        headers: { Accept: "application/json" },
    });
    if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
            const body: unknown = await response.json();
            if (typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string") {
                detail = body.detail;
            }
        } catch {
            // The HTTP status remains a safe fallback when the body is not JSON.
        }
        throw new Error(detail);
    }
    return (await response.json()) as T;
}

/** "Light Out" -> "Light out", keeping acronyms such as "BBQ" in capitals. */
function sentenceCase(text: string): string {
    return text
        .split(" ")
        .map((word, index) => (index === 0 || /^[A-Z0-9]{2,}$/.test(word) ? word : word.toLowerCase()))
        .join(" ");
}

/** Split a City label like "Traffic Operations - Light Out" into a title and a group. */
function friendlyType(raw: string): { title: string; group: string } {
    const parts = raw.split(" - ").map((part) => part.trim()).filter(Boolean);
    const last = parts.at(-1) ?? raw;
    const rest = parts.slice(0, -1).map((part) => ABBREVIATIONS[part] ?? part);
    const expanded = ABBREVIATIONS[last];
    if (expanded) {
        return { title: `${sentenceCase(rest.join(" - ") || last)} (${expanded})`, group: "" };
    }
    return { title: sentenceCase(last), group: rest.join(" · ") };
}

function formatDays(days: number): string {
    if (days < 1) {
        return "same day";
    }
    const rounded = Math.round(days);
    return `${rounded} day${rounded === 1 ? "" : "s"}`;
}

function formatUsual(perWeek: number): string {
    return perWeek < 1 ? "normally less than 1" : `normally about ${Math.round(perWeek)}`;
}

function wardFeature(wardNumber: number): WardFeature | null {
    return state.wards?.features.find((feature) => feature.properties.ward === wardNumber) ?? null;
}

function wardName(wardNumber: number): string {
    return wardFeature(wardNumber)?.properties.name ?? `Ward ${wardNumber}`;
}

function addWardLayers(map: MapLibre.Map, wards: WardCollection): void {
    map.addSource(MAP_SOURCE_ID, { type: "geojson", data: wards });
    map.addLayer({
        id: MAP_FILL_LAYER_ID,
        type: "fill",
        source: MAP_SOURCE_ID,
        paint: {
            "fill-color": [
                "match",
                ["get", "band"],
                0,
                BAND_COLOURS[0],
                1,
                BAND_COLOURS[1],
                2,
                BAND_COLOURS[2],
                3,
                BAND_COLOURS[3],
                NO_DATA_COLOUR,
            ],
            "fill-opacity": 0.75,
        },
    });
    map.addLayer({
        id: MAP_LINE_LAYER_ID,
        type: "line",
        source: MAP_SOURCE_ID,
        paint: { "line-color": "#ffffff", "line-width": 1.2 },
    });
    map.addLayer({
        id: MAP_HIGHLIGHT_LAYER_ID,
        type: "line",
        source: MAP_SOURCE_ID,
        paint: { "line-color": "#d4700f", "line-width": 3.5 },
        filter: ["==", ["get", "ward"], -1],
    });
}

/** 12400 -> "12.4k", 2.5 -> "2.5": short enough for a legend band. */
function compact(value: number): string {
    return value >= 1000 ? `${Math.round(value / 100) / 10}k` : String(Math.round(value * 10) / 10);
}

/** Split the wards into up to four equal-sized colour bands for the chosen metric. */
function paintMap(): void {
    const { wards, metric } = state;
    if (!wards) {
        return;
    }
    const values = wards.features.map((feature) => state.wardMetrics.get(feature.properties.ward)?.[metric] ?? null);
    const known = values.filter((value): value is number => value !== null).sort((a, b) => a - b);
    const lowest = known[0] ?? 0;
    // Quartile cut points; equal ones merge, so wards that all share a value get one band.
    const cuts = [0.25, 0.5, 0.75].map((share) => known[Math.floor(share * known.length)] ?? lowest);
    const breaks = [...new Set(cuts)].filter((cut) => cut > lowest);
    wards.features.forEach((feature, index) => {
        const value = values[index] ?? null;
        feature.properties.band = value === null ? -1 : breaks.filter((cut) => value >= cut).length;
    });
    (state.map?.getSource(MAP_SOURCE_ID) as MapLibre.GeoJSONSource | undefined)?.setData(wards);

    const requestType = element<HTMLSelectElement>("type-picker").value;
    const scope = requestType ? ` · ${friendlyType(requestType).title}` : "";
    setText("legend-title", `${METRIC_LABELS[metric]}${scope}${known.length ? "" : " · too few to say"}`);
    const labels = (known.length ? [lowest, ...breaks] : []).map((edge, index) => {
        const next = breaks[index];
        if (next === undefined) {
            return index === 0 ? compact(edge) : `${compact(edge)}+`;
        }
        return index === 0 ? `Under ${compact(next)}` : `${compact(edge)}–${compact(next)}`;
    });
    element("legend-bands").replaceChildren(
        ...labels.map((label) => {
            const band = document.createElement("span");
            band.textContent = label;
            return band;
        }),
    );
}

async function loadWardMetrics(): Promise<void> {
    const picker = element<HTMLSelectElement>("type-picker");
    const requestType = picker.value;
    const query = requestType ? `?${new URLSearchParams({ type: requestType }).toString()}` : "";
    try {
        const result = await fetchJson<{ wards: WardMetric[] }>(`/ward-metrics${query}`);
        // A slow answer for an earlier choice must not overwrite the current one.
        if (requestType !== picker.value) {
            return;
        }
        state.wardMetrics = new Map(result.wards.map((ward) => [ward.ward, ward]));
        paintMap();
    } catch (error) {
        showError(`Could not colour the map: ${errorMessage(error)}`);
    }
}

async function loadWards(): Promise<void> {
    setLoading(true);
    try {
        const collection = await fetchJson<WardCollection>("/wards");
        state.wards = collection;
        setText("as-of", ` up to ${new Date(`${collection.as_of}T12:00:00`).toLocaleDateString()}`);
        setText("attribution", collection.attribution);
        const picker = element<HTMLSelectElement>("ward-picker");
        for (const feature of [...collection.features].sort((a, b) => a.properties.name.localeCompare(b.properties.name))) {
            picker.append(new Option(feature.properties.name, String(feature.properties.ward)));
        }
        // Add the layers only once the map has finished loading, whichever finishes first.
        await state.mapReady;
        if (state.map && !state.map.getSource(MAP_SOURCE_ID)) {
            addWardLayers(state.map, collection);
        }
        setStatus("Connected", "status-ready");
    } finally {
        setLoading(false);
    }
}

async function loadRequestTypes(): Promise<void> {
    const types = await fetchJson<RequestType[]>("/request-types");
    const byDepartment = new Map<string, RequestType[]>();
    for (const item of types) {
        byDepartment.set(item.category, [...(byDepartment.get(item.category) ?? []), item]);
    }
    const picker = element<HTMLSelectElement>("type-picker");
    for (const department of [...byDepartment.keys()].sort()) {
        const group = document.createElement("optgroup");
        group.label = department;
        const items = (byDepartment.get(department) ?? [])
            .map((item) => ({ item, label: friendlyType(item.type) }))
            .sort((a, b) => a.label.title.localeCompare(b.label.title));
        for (const { item, label } of items) {
            const text = label.group ? `${label.title} — ${label.group}` : label.title;
            group.append(new Option(text, item.type));
        }
        picker.append(group);
    }
}

function renderSpikes(): void {
    const list = element<HTMLUListElement>("spike-list");
    const toggle = element<HTMLButtonElement>("show-all-spikes");
    const spikes = state.spikes;
    const shown = state.showAllSpikes ? spikes : spikes.slice(0, SPIKES_SHOWN);
    list.replaceChildren();
    if (spikes.length === 0) {
        const item = document.createElement("li");
        item.className = "supporting-text";
        item.textContent = "Nothing unusual this week.";
        list.append(item);
    }
    for (const spike of shown) {
        const { title, group } = friendlyType(spike.type);
        const button = document.createElement("button");
        button.type = "button";
        button.className = "spike-item";
        const heading = document.createElement("span");
        heading.className = "spike-title";
        const name = document.createElement("strong");
        name.textContent = title;
        const badge = document.createElement("span");
        badge.className = "badge badge-high";
        badge.textContent = `${Math.round(spike.this_week / spike.usual_per_week)}× more`;
        heading.append(name, badge);
        const detail = document.createElement("span");
        const where = state.selectedWard === null ? `${spike.ward_name} · ` : "";
        detail.textContent = `${where}${spike.this_week} this week, ${formatUsual(spike.usual_per_week)}${group ? ` · ${group}` : ""}`;
        button.append(heading, detail);
        button.addEventListener("click", () => void selectWard(spike.ward));
        list.append(wrap(button));
    }
    toggle.classList.toggle("hidden", spikes.length <= SPIKES_SHOWN);
    toggle.textContent = state.showAllSpikes ? "Show fewer" : `Show all ${spikes.length}`;
}

function wrap(child: HTMLElement): HTMLLIElement {
    const item = document.createElement("li");
    item.append(child);
    return item;
}

function reportRow(entry: ReportEntry, forWard: boolean): HTMLTableRowElement {
    const { title, group } = friendlyType(entry.type);
    const row = document.createElement("tr");

    const problem = document.createElement("td");
    const name = document.createElement("span");
    name.textContent = entry.batch_closed ? `${title} ⚠️` : title;
    problem.append(name);
    if (group) {
        const small = document.createElement("small");
        small.textContent = group;
        problem.append(small);
    }

    const count = document.createElement("td");
    count.textContent = entry.count.toLocaleString();

    const days = document.createElement("td");
    const here = forWard ? entry.ward_median_days : entry.city_median_days;
    days.textContent = here === null ? "—" : formatDays(here);
    if (forWard && entry.city_median_days !== null) {
        const city = document.createElement("small");
        city.textContent = `city: ${formatDays(entry.city_median_days)}`;
        days.append(city);
    }
    row.append(problem, count, days);
    return row;
}

async function showReport(wardNumber: number | null): Promise<void> {
    clearError();
    try {
        const report = await fetchJson<Report>(wardNumber === null ? "/city" : `/wards/${wardNumber}`);
        setText("selected-name", report.name);
        const sector = wardNumber === null ? "" : `${wardFeature(wardNumber)?.properties.sector ?? ""} ward · `;
        setText("selected-summary", `${sector}${report.total_requests.toLocaleString()} requests in the last 12 months`);
        element("report-rows").replaceChildren(...report.top_requests.map((entry) => reportRow(entry, wardNumber !== null)));
        state.spikes = report.spikes;
        state.showAllSpikes = false;
        renderSpikes();
    } catch (error) {
        showError(`Could not load the report card: ${errorMessage(error)}`);
    }
}

async function updateEstimate(): Promise<void> {
    const requestType = element<HTMLSelectElement>("type-picker").value;
    const card = element("estimate-card");
    const batchNote = element("estimate-batch");
    const ward = state.selectedWard;
    batchNote.classList.add("hidden");
    element("waited-block").classList.toggle("hidden", !requestType || ward === null);
    void updateRemaining();
    if (!requestType) {
        card.classList.add("hidden");
        return;
    }
    card.classList.remove("hidden");
    if (ward === null) {
        setText("estimate-usual", "Choose your ward first");
        setText("estimate-range", "Times differ between wards, so step 1 is needed for an estimate.");
        return;
    }
    try {
        const parameters = new URLSearchParams({ type: requestType, ward: String(ward) });
        const result = await fetchJson<Estimate>(`/estimate?${parameters.toString()}`);
        if (result.usual_days === null) {
            setText("estimate-usual", "Not enough reliable history yet");
            setText("estimate-range", "Too few of these have been reported and closed since the City's bulk clean-up to say.");
        } else {
            setText("estimate-usual", formatDays(result.usual_days));
            setText(
                "estimate-range",
                result.nine_in_ten_days === null
                    ? "More than 1 in 10 were still open when this data was published."
                    : `9 in 10 are closed within ${formatDays(result.nine_in_ten_days)} in ${wardName(ward)}.`,
            );
        }
        batchNote.classList.toggle("hidden", !result.batch_closed);
    } catch (error) {
        showError(`Could not estimate this request: ${errorMessage(error)}`);
    }
}

function remainingText(result: Remaining, ward: number): string {
    const waited = formatDays(result.waited_days);
    if (!result.enough_history) {
        return `Too few past requests like this were still open after ${waited} to say how much longer it takes.`;
    }
    const where = result.scope === "ward" ? `in ${wardName(ward)}` : "across Ottawa";
    const group = `Of ${result.sample_size.toLocaleString()} past requests like this ${where} that were still open after ${waited}`;
    const unclosed = "had still not been closed when this data was published";
    if (result.usual_more_days === null) {
        return `${group}, more than half ${unclosed}.`;
    }
    const half = `${group}, half were closed within another ${formatDays(result.usual_more_days)}`;
    return result.nine_in_ten_more_days === null
        ? `${half}, but more than 1 in 10 ${unclosed}.`
        : `${half} and 9 in 10 within another ${formatDays(result.nine_in_ten_more_days)}.`;
}

async function updateRemaining(): Promise<void> {
    const target = element("remaining-text");
    const requestType = element<HTMLSelectElement>("type-picker").value;
    const ward = state.selectedWard;
    const waited = Number(element<HTMLInputElement>("waited-input").value);
    if (!requestType || ward === null || !Number.isInteger(waited) || waited < 1 || waited > 365) {
        target.classList.add("hidden");
        return;
    }
    try {
        const parameters = new URLSearchParams({ type: requestType, ward: String(ward), waited: String(waited) });
        const result = await fetchJson<Remaining>(`/remaining?${parameters.toString()}`);
        target.textContent = remainingText(result, ward);
        target.classList.remove("hidden");
    } catch (error) {
        showError(`Could not look up the remaining wait: ${errorMessage(error)}`);
    }
}

function wardBounds(maplibregl: typeof MapLibre, feature: WardFeature): MapLibre.LngLatBounds {
    const bounds = new maplibregl.LngLatBounds();
    const polygons = feature.geometry.type === "Polygon" ? [feature.geometry.coordinates] : feature.geometry.coordinates;
    for (const polygon of polygons) {
        for (const [lng, lat] of polygon[0] ?? []) {
            if (lng !== undefined && lat !== undefined) {
                bounds.extend([lng, lat]);
            }
        }
    }
    return bounds;
}

async function selectWard(wardNumber: number | null, moveMap = true): Promise<void> {
    const feature = wardNumber === null ? null : wardFeature(wardNumber);
    if (wardNumber !== null && !feature) {
        return;
    }
    state.selectedWard = wardNumber;
    element<HTMLSelectElement>("ward-picker").value = wardNumber === null ? "" : String(wardNumber);
    // Keep the ward in the address so the page can be bookmarked or shared.
    history.replaceState(null, "", wardNumber === null ? location.pathname : `#ward=${wardNumber}`);
    const { map } = state;
    if (map && window.maplibregl && map.getLayer(MAP_HIGHLIGHT_LAYER_ID)) {
        map.setFilter(MAP_HIGHLIGHT_LAYER_ID, ["==", ["get", "ward"], wardNumber ?? -1]);
        if (moveMap) {
            if (feature) {
                map.fitBounds(wardBounds(window.maplibregl, feature), { padding: 40, maxZoom: 13 });
            } else {
                map.flyTo(OTTAWA_VIEW);
            }
        }
    }
    void updateEstimate();
    await showReport(wardNumber);
}

function popupContent(properties: WardProperties): HTMLElement {
    const container = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = properties.name;
    const details = document.createElement("p");
    details.className = "supporting-text";
    const metrics = state.wardMetrics.get(properties.ward);
    const shown = {
        requests: `${(metrics?.requests ?? 0).toLocaleString()} requests in 12 months`,
        median_days:
            metrics?.median_days == null ? "too few closed requests to say" : `usually closed in ${formatDays(metrics.median_days)}`,
        spikes: metrics?.spikes ? `${metrics.spikes} unusual this week` : "nothing unusual this week",
    }[state.metric];
    details.textContent = `${shown} · click to open`;
    container.append(title, details);
    return container;
}

function initializeMap(): void {
    const { maplibregl } = window;
    if (!maplibregl) {
        setStatus("Map failed to load", "status-error");
        showError("The map could not load. Try refreshing the page.");
        setLoading(false);
        return;
    }
    const map = new maplibregl.Map({
        container: "map",
        ...OTTAWA_VIEW,
        minZoom: 8.5,
        maxZoom: 14,
        maxBounds: [
            [-76.8, 44.75],
            [-74.9, 45.8],
        ],
        attributionControl: false,
        style: {
            version: 8,
            sources: {},
            layers: [{ id: "background", type: "background", paint: { "background-color": "#eef2f1" } }],
        },
    });
    state.map = map;
    state.mapReady = new Promise((resolve) => map.once("load", () => resolve()));
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false });
    state.hoverPopup = popup;

    // MapLibre turns GeoJSON properties into plain values, so they match WardProperties.
    const hovered = (event: MapLibre.MapLayerMouseEvent): WardProperties | null =>
        (event.features?.[0]?.properties as WardProperties | undefined) ?? null;
    map.on("click", MAP_FILL_LAYER_ID, (event) => {
        const properties = hovered(event);
        if (properties) {
            void selectWard(properties.ward, false);
        }
    });
    map.on("mousemove", MAP_FILL_LAYER_ID, (event) => {
        const properties = hovered(event);
        if (!properties) {
            return;
        }
        map.getCanvas().style.cursor = "pointer";
        popup.setLngLat(event.lngLat).setDOMContent(popupContent(properties)).addTo(map);
    });
    map.on("mouseleave", MAP_FILL_LAYER_ID, () => {
        map.getCanvas().style.cursor = "";
        popup.remove();
    });
}

function bindGuideModal(): void {
    const dialog = element<HTMLDialogElement>("guide-dialog");
    element("open-guide").addEventListener("click", () => dialog.showModal());
    element("close-guide").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
        if (event.target === dialog) {
            dialog.close();
        }
    });
}

function bindControls(): void {
    element<HTMLSelectElement>("ward-picker").addEventListener("change", (event) => {
        const value = (event.target as HTMLSelectElement).value;
        void selectWard(value ? Number(value) : null);
    });
    element<HTMLSelectElement>("type-picker").addEventListener("change", () => {
        void updateEstimate();
        void loadWardMetrics();
    });
    element<HTMLSelectElement>("metric-picker").addEventListener("change", (event) => {
        state.metric = (event.target as HTMLSelectElement).value as Metric;
        paintMap();
    });
    element<HTMLInputElement>("waited-input").addEventListener("input", () => void updateRemaining());
    element("show-all-spikes").addEventListener("click", () => {
        state.showAllSpikes = !state.showAllSpikes;
        renderSpikes();
    });
    bindGuideModal();
}

document.addEventListener("DOMContentLoaded", async () => {
    bindControls();
    initializeMap();
    try {
        await Promise.all([loadWards(), loadRequestTypes()]);
        void loadWardMetrics();
        const linked = Number(new URLSearchParams(location.hash.slice(1)).get("ward"));
        await selectWard(Number.isInteger(linked) && wardFeature(linked) ? linked : null);
    } catch (error) {
        setStatus("Not connected", "status-error");
        showError(`Could not load the 311 data: ${errorMessage(error)}`);
    }
});
