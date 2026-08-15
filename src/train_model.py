"""Assemble temporal examples and train the synthetic LightGBM demonstration.

Training uses chronological train/validation/test partitions with 48-hour purge
gaps. The validation partition selects a threshold; the test partition is used
once for final metrics. Saved artifacts are trusted local joblib output bundles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

if TYPE_CHECKING:
    from .config import (
        DATA_DIR,
        MODEL_PATH,
        MODELS_DIR,
        PREDICTION_HORIZON_HOURS,
        TARGET_MODE,
        ensure_directories,
    )
    from .errors import ArtifactError, DataValidationError
else:
    try:
        from .config import (
            DATA_DIR,
            MODEL_PATH,
            MODELS_DIR,
            PREDICTION_HORIZON_HOURS,
            TARGET_MODE,
            ensure_directories,
        )
        from .errors import ArtifactError, DataValidationError
    except ImportError:  # pragma: no cover - supports direct script imports
        from config import (
            DATA_DIR,
            MODEL_PATH,
            MODELS_DIR,
            PREDICTION_HORIZON_HOURS,
            TARGET_MODE,
            ensure_directories,
        )
        from errors import ArtifactError, DataValidationError

logger = logging.getLogger(__name__)

TARGET_COLUMN = "break_in_next_48h"
ARTIFACT_SCHEMA_VERSION = 1
RANDOM_SEED = 42
MAX_TRAINING_DAYS = 92
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20


def _read_required_parquet(path: Path, description: str) -> pd.DataFrame:
    """Load one required local artifact with a clear safe error."""
    if not path.is_file():
        raise ArtifactError(f"Missing {description} artifact: {path.name}")
    try:
        dataframe = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Could not read {description} artifact: {path.name}") from exc
    if dataframe.empty:
        raise DataValidationError(f"{description} artifact is empty")
    return dataframe


def _daily_weather_snapshots(weather: pd.DataFrame) -> pd.DataFrame:
    """Select the final observation per day to limit cross-join size."""
    if "timestamp" not in weather:
        raise DataValidationError("Weather features must contain timestamp")
    normalized = weather.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    if normalized["timestamp"].isna().any():
        raise DataValidationError("Weather features contain invalid timestamps")
    normalized = normalized.sort_values("timestamp")
    daily = normalized.set_index("timestamp").resample("24h").last().dropna(how="all")
    return daily.tail(MAX_TRAINING_DAYS).reset_index()


def assemble_feature_matrix(
    infrastructure: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Cross each H3 aggregate with chronological daily weather snapshots."""
    required_infrastructure = {"h3_index", "centroid_lat", "centroid_lon"}
    missing = required_infrastructure - set(infrastructure.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise DataValidationError(f"Infrastructure features are missing: {names}")
    if infrastructure["h3_index"].duplicated().any():
        raise DataValidationError("Infrastructure features contain duplicate H3 indexes")

    daily_weather = _daily_weather_snapshots(weather)
    if len(daily_weather) < 10:
        raise DataValidationError("At least 10 daily weather snapshots are required")
    matrix = infrastructure.merge(daily_weather, how="cross", suffixes=("", "_weather"))
    return matrix.sort_values(["timestamp", "h3_index"]).reset_index(drop=True)


def load_feature_matrix() -> pd.DataFrame:
    """Load required pipeline artifacts and assemble temporal model examples."""
    infrastructure = _read_required_parquet(
        DATA_DIR / "infrastructure_features.parquet",
        "infrastructure features",
    )
    weather = _read_required_parquet(DATA_DIR / "weather_features.parquet", "weather features")
    return assemble_feature_matrix(infrastructure, weather)


def create_target_variable(
    dataframe: pd.DataFrame,
    mode: str = TARGET_MODE,
) -> pd.DataFrame:
    """Create an explicitly synthetic target or validate an observed target."""
    if dataframe.empty:
        return dataframe.copy()
    if mode not in {"synthetic", "observed"}:
        raise ValueError("mode must be one of: synthetic, observed")
    result = dataframe.copy()
    if mode == "observed":
        if TARGET_COLUMN not in result:
            raise DataValidationError(
                "Observed target mode requires a reviewed, pre-joined outcome column"
            )
        result[TARGET_COLUMN] = result[TARGET_COLUMN].astype(int)
        return result

    water_length = result.get("line_length_km_water", pd.Series(0.0, index=result.index))
    vintage = result.get("pct_pre_1980", pd.Series(0.0, index=result.index))
    heat = result.get("heat_index_c", pd.Series(20.0, index=result.index))
    dry_days = result.get("consecutive_dry_days", pd.Series(0.0, index=result.index))
    recent_rain = result.get("rainfall_48h", pd.Series(0.0, index=result.index))

    synthetic_score = (
        (water_length > water_length.median()).astype(int) * 2
        + (vintage >= 0.4).astype(int) * 2
        + (heat >= 28.0).astype(int)
        + (dry_days >= 2.0).astype(int)
        + (recent_rain < 1.0).astype(int)
    )
    result[TARGET_COLUMN] = (synthetic_score >= 4).astype(int)
    logger.info("Synthetic target distribution: %s", result[TARGET_COLUMN].value_counts().to_dict())
    return result


def get_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return sorted numeric model inputs while excluding IDs, target, and time."""
    excluded = {
        "h3_index",
        TARGET_COLUMN,
        "geometry",
        "centroid_lat",
        "centroid_lon",
        "timestamp",
    }
    return sorted(
        column
        for column in dataframe.select_dtypes(include="number").columns
        if column not in excluded
    )


def temporal_train_test_split(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = TARGET_COLUMN,
    purge_hours: int = PREDICTION_HORIZON_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split by timestamp and remove prediction-horizon gaps between partitions."""
    if "timestamp" not in dataframe:
        raise DataValidationError("Temporal split requires a timestamp column")
    if target_col not in dataframe:
        raise DataValidationError(f"Temporal split requires target column {target_col}")
    if purge_hours < 0:
        raise ValueError("purge_hours cannot be negative")

    ordered = dataframe.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise DataValidationError("Temporal split received an invalid timestamp")
    ordered = ordered.sort_values(["timestamp", "h3_index"]).reset_index(drop=True)
    unique_times = ordered["timestamp"].drop_duplicates().sort_values().tolist()
    if len(unique_times) < 10:
        raise DataValidationError("Temporal split requires at least 10 unique timestamps")

    train_count = max(1, int(len(unique_times) * TRAIN_FRACTION))
    validation_end_count = max(
        train_count + 1,
        int(len(unique_times) * (TRAIN_FRACTION + VALIDATION_FRACTION)),
    )
    train_end = pd.Timestamp(unique_times[train_count - 1])
    validation_end = pd.Timestamp(unique_times[validation_end_count - 1])
    purge_gap = pd.Timedelta(hours=purge_hours)

    train_rows = ordered[ordered["timestamp"] <= train_end]
    validation_rows = ordered[
        (ordered["timestamp"] >= train_end + purge_gap)
        & (ordered["timestamp"] <= validation_end)
    ]
    test_rows = ordered[ordered["timestamp"] >= validation_end + purge_gap]
    if train_rows.empty or validation_rows.empty or test_rows.empty:
        raise DataValidationError("Temporal split is empty after applying purge gaps")

    return (
        train_rows[feature_columns],
        validation_rows[feature_columns],
        test_rows[feature_columns],
        train_rows[target_col],
        validation_rows[target_col],
        test_rows[target_col],
    )


def train_lightgbm(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> lgb.LGBMClassifier:
    """Train a small deterministic classifier with imbalance compensation."""
    if y_train.nunique() < 2:
        raise DataValidationError("Training target must contain both classes")
    positive_count = int(y_train.sum())
    negative_count = len(y_train) - positive_count
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=5,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float(negative_count / positive_count),
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="binary_logloss",
    )
    return model


def select_decision_threshold(
    model: lgb.LGBMClassifier,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> float:
    """Choose an F1-maximizing threshold using validation data only."""
    if y_validation.nunique() < 2:
        return 0.5
    probabilities = model.predict_proba(x_validation)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_validation, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def evaluate_model(
    model: lgb.LGBMClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    decision_threshold: float = 0.5,
    target_mode: str = TARGET_MODE,
) -> dict[str, Any]:
    """Evaluate the untouched test partition at a preselected threshold."""
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= decision_threshold).astype(int)
    pr_auc = (
        float(average_precision_score(y_test, probabilities))
        if y_test.nunique() > 1
        else 0.0
    )
    return {
        "pr_auc": pr_auc,
        "f1_at_threshold": float(f1_score(y_test, predictions, zero_division=0)),
        "decision_threshold": decision_threshold,
        "target_mode": target_mode,
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
        "test_rows": len(y_test),
    }


def _artifact_version(feature_columns: list[str], row_count: int) -> str:
    """Create a stable short version from schema-relevant training inputs."""
    payload = json.dumps(
        {"features": feature_columns, "rows": row_count, "target_mode": TARGET_MODE},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def save_artifacts(
    model: lgb.LGBMClassifier,
    feature_columns: list[str],
    decision_threshold: float,
    metrics: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    """Atomically save a trusted model bundle and human-readable metadata."""
    ensure_directories()
    model_version = _artifact_version(feature_columns, row_count)
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": model_version,
        "target_mode": TARGET_MODE,
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
        "training_rows": row_count,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "warning": "Synthetic demonstration target; not an operational forecast.",
    }
    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "decision_threshold": decision_threshold,
        "metadata": metadata,
    }
    temporary_model = MODEL_PATH.with_suffix(f"{MODEL_PATH.suffix}.tmp")
    joblib.dump(bundle, temporary_model)
    temporary_model.replace(MODEL_PATH)

    metrics_payload = {**metrics, **metadata}
    metrics_path = MODELS_DIR / "model_metrics.json"
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    temporary_metrics.replace(metrics_path)

    importance = pd.DataFrame(
        {"feature": feature_columns, "importance": model.feature_importances_}
    ).sort_values(["importance", "feature"], ascending=[False, True])
    importance.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    return metadata


def main() -> None:
    """Train, evaluate, and save the local demonstration model."""
    matrix = create_target_variable(load_feature_matrix())
    feature_columns = get_feature_columns(matrix)
    if not feature_columns:
        raise DataValidationError("No numeric model features are available")
    x_train, x_validation, x_test, y_train, y_validation, y_test = (
        temporal_train_test_split(matrix, feature_columns)
    )
    model = train_lightgbm(x_train, y_train, x_validation, y_validation)
    threshold = select_decision_threshold(model, x_validation, y_validation)
    metrics = evaluate_model(model, x_test, y_test, threshold)
    metadata = save_artifacts(model, feature_columns, threshold, metrics, len(matrix))
    logger.info("Saved trusted synthetic model artifact version %s", metadata["model_version"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
