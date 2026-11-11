"""Train the "how long will my request take?" model on the committed 311 snapshot.

Two LightGBM quantile models predict the days until the City marks a request
closed: the median ("usually") and the 90th percentile ("9 in 10 within").
Requests opened in the last ``MATURITY_DAYS`` are left out, because many of them
have not had time to close and would bias the times downward. The latest
``TEST_DAYS`` of the remaining requests are held out, and both models are
compared with simply looking up each request type's historical quantile.

    python -m src.train_model
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from .config import MODEL_PATH, MODELS_DIR, REQUESTS_PATH
from .errors import ArtifactError, DataValidationError
from .insights import batch_closed_types

logger = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 3
RANDOM_SEED = 42
QUANTILES = {"usual_days": 0.5, "nine_in_ten_days": 0.9}
CATEGORICAL_FEATURES = ["type", "ward", "channel"]
FEATURE_COLUMNS = [*CATEGORICAL_FEATURES, "weekday"]
MATURITY_DAYS = 120
TEST_DAYS = 120


def load_requests(path: Any = REQUESTS_PATH) -> pd.DataFrame:
    """Load the 311 snapshot with parsed dates and days to close."""
    try:
        requests = pd.read_csv(path, parse_dates=["opened", "closed"], dtype={"ward": "Int64"})
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Could not read the 311 snapshot: {path}") from exc
    if requests.empty:
        raise DataValidationError("The 311 snapshot is empty")
    requests["days_to_close"] = (requests["closed"] - requests["opened"]).dt.days
    return requests


def model_inputs(frame: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Encode features with the training categories, so unseen values become missing."""
    inputs = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL_FEATURES:
        inputs[column] = pd.Categorical(frame[column].astype(str), categories=categories[column])
    inputs["weekday"] = frame["opened"].dt.weekday
    return inputs


def split_by_date(requests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return non-cancelled train and test requests, test being the latest mature days.

    Training keeps closed requests only. Test also keeps the few still open after
    ``MATURITY_DAYS``, because dropping them would hide exactly the slow cases.
    """
    snapshot_end = requests["opened"].max()
    mature_end = snapshot_end - pd.Timedelta(days=MATURITY_DAYS)
    test_start = mature_end - pd.Timedelta(days=TEST_DAYS)
    usable = requests[(requests["status"] != "Cancelled") & (requests["opened"] <= mature_end)]
    train = usable[usable["opened"] < test_start].dropna(subset=["days_to_close"])
    test = usable[usable["opened"] >= test_start]
    if train.empty or test.empty:
        raise DataValidationError("Not enough history to train and test")
    return train, test


def _new_model(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=15,
        min_child_samples=50,
        random_state=RANDOM_SEED,
        # shortcut: LightGBM 4.6.0's col-wise histograms crashed natively on this project's
        # data (Windows, Python 3.13), so the timing-based auto choice failed ~1 in 3 runs.
        # Revisit after a LightGBM upgrade.
        force_row_wise=True,
        n_jobs=1,
        verbosity=-1,
    )


def type_lookup(train: pd.DataFrame, test: pd.DataFrame, alpha: float) -> pd.Series:
    """Baseline: each request type's historical quantile, or the overall one for new types."""
    by_type = train.groupby("type")["days_to_close"].quantile(alpha)
    fallback = float(train["days_to_close"].quantile(alpha))
    return test["type"].map(by_type).fillna(fallback)


def evaluate(
    models: dict[str, lgb.LGBMRegressor],
    train: pd.DataFrame,
    test: pd.DataFrame,
    categories: dict[str, list[str]],
) -> dict[str, Any]:
    """Score both quantile models and the per-type lookup on the held-out requests."""
    closed = test["days_to_close"].notna()
    actual = test.loc[closed, "days_to_close"]
    inputs = model_inputs(test, categories)
    metrics: dict[str, Any] = {
        "test_requests": len(test),
        "test_requests_still_open": int((~closed).sum()),
        "test_start": str(test["opened"].min().date()),
        "test_end": str(test["opened"].max().date()),
    }
    for name, alpha in QUANTILES.items():
        predicted = pd.Series(np.clip(models[name].predict(inputs), 0, None), index=test.index)
        baseline = type_lookup(train, test, alpha)
        # A still-open request counts as not closed within the estimate (NaN <= x is False).
        within = test["days_to_close"] <= predicted
        metrics[name] = {
            "pinball_loss": float(mean_pinball_loss(actual, predicted[closed], alpha=alpha)),
            "pinball_loss_type_lookup": float(
                mean_pinball_loss(actual, baseline[closed], alpha=alpha)
            ),
            "share_closed_within": float(within.mean()),
            "share_closed_within_type_lookup": float((test["days_to_close"] <= baseline).mean()),
            "median_absolute_error_days": float(np.median(np.abs(actual - predicted[closed]))),
        }
    return metrics


def train(train_rows: pd.DataFrame) -> tuple[dict[str, lgb.LGBMRegressor], dict[str, list[str]]]:
    """Fit one quantile model per entry in ``QUANTILES``."""
    categories = {
        column: sorted(train_rows[column].astype(str).unique()) for column in CATEGORICAL_FEATURES
    }
    inputs = model_inputs(train_rows, categories)
    models = {
        name: _new_model(alpha).fit(inputs, train_rows["days_to_close"])
        for name, alpha in QUANTILES.items()
    }
    return models, categories


def save_artifacts(bundle: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Atomically save the trusted model bundle and readable metrics."""
    MODELS_DIR.mkdir(exist_ok=True)
    temporary_model = MODEL_PATH.with_suffix(f"{MODEL_PATH.suffix}.tmp")
    joblib.dump(bundle, temporary_model)
    temporary_model.replace(MODEL_PATH)

    metrics_path = MODELS_DIR / "model_metrics.json"
    temporary_metrics = metrics_path.with_suffix(".json.tmp")
    temporary_metrics.write_text(
        json.dumps({**metrics, **bundle["metadata"]}, indent=2), encoding="utf-8"
    )
    temporary_metrics.replace(metrics_path)


def main() -> None:
    """Evaluate on held-out requests, then refit on everything mature and save."""
    requests = load_requests()
    # shortcut: types the City closed in bulk are left out, because their closing dates
    # before the May 2026 mass closure are not real and too few have closed since to hold
    # out a test set. The API answers for them from recent history instead. Train on their
    # post-closure requests once there is a year of them.
    requests = requests[~requests["type"].isin(batch_closed_types(requests))]
    train_rows, test_rows = split_by_date(requests)
    models, categories = train(train_rows)
    metrics = evaluate(models, train_rows, test_rows, categories)

    final_models, final_categories = train(pd.concat([train_rows, test_rows]))
    version = hashlib.sha256(str(requests["opened"].max()).encode()).hexdigest()[:12]
    bundle = {
        "models": final_models,
        "categories": final_categories,
        "metadata": {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_version": version,
            "training_requests": len(train_rows) + len(test_rows),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "warning": "Days until the City marks a request closed, not when work was done.",
        },
    }
    save_artifacts(bundle, metrics)
    logger.info("Saved model %s: %s", version, json.dumps(metrics))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
