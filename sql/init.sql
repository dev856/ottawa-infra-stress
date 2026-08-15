-- Local-development PostGIS schema. This script is idempotent and runs as the
-- database owner configured through Docker Compose environment variables.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS model;

-- Minimized 311 events. Coordinates, geometry, addresses, descriptions, notes,
-- names, and contact fields are intentionally absent.
CREATE TABLE IF NOT EXISTS raw.h3_311_events (
    objectid BIGINT PRIMARY KEY,
    reqtype VARCHAR(100) NOT NULL,
    category VARCHAR(100) NOT NULL DEFAULT '',
    createddate TIMESTAMPTZ NOT NULL,
    h3_index VARCHAR(20) NOT NULL
);

CREATE INDEX IF NOT EXISTS h3_311_events_h3_time_idx
ON raw.h3_311_events (h3_index, createddate);

CREATE TABLE IF NOT EXISTS features.h3_grid (
    h3_index VARCHAR(20) PRIMARY KEY,
    h3_resolution SMALLINT NOT NULL CHECK (h3_resolution BETWEEN 0 AND 15),
    centroid_lat DOUBLE PRECISION NOT NULL CHECK (centroid_lat BETWEEN -90 AND 90),
    centroid_lon DOUBLE PRECISION NOT NULL CHECK (centroid_lon BETWEEN -180 AND 180),
    geometry GEOMETRY(Polygon, 4326) NOT NULL
);

CREATE INDEX IF NOT EXISTS h3_grid_geometry_idx
ON features.h3_grid USING GIST (geometry);

CREATE TABLE IF NOT EXISTS features.infrastructure_features (
    h3_index VARCHAR(20) PRIMARY KEY REFERENCES features.h3_grid(h3_index)
        ON UPDATE CASCADE ON DELETE CASCADE,
    centroid_lat DOUBLE PRECISION NOT NULL,
    centroid_lon DOUBLE PRECISION NOT NULL,
    line_length_km_water DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (line_length_km_water >= 0),
    line_count_water INTEGER NOT NULL DEFAULT 0 CHECK (line_count_water >= 0),
    line_length_km_road DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (line_length_km_road >= 0),
    line_count_road INTEGER NOT NULL DEFAULT 0 CHECK (line_count_road >= 0),
    building_count INTEGER NOT NULL DEFAULT 0 CHECK (building_count >= 0),
    median_year_built DOUBLE PRECISION NOT NULL DEFAULT 1980,
    pct_pre_1980 DOUBLE PRECISION NOT NULL DEFAULT 0
        CHECK (pct_pre_1980 BETWEEN 0 AND 1)
);

CREATE TABLE IF NOT EXISTS model.risk_predictions (
    h3_index VARCHAR(20) PRIMARY KEY REFERENCES features.h3_grid(h3_index)
        ON UPDATE CASCADE ON DELETE CASCADE,
    risk_score DOUBLE PRECISION NOT NULL CHECK (risk_score BETWEEN 0 AND 1),
    generated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    model_version VARCHAR(100) NOT NULL,
    target_mode VARCHAR(20) NOT NULL CHECK (target_mode IN ('synthetic', 'observed'))
);

-- The raw schema is not a public application interface.
REVOKE ALL ON SCHEMA raw FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA raw FROM PUBLIC;
