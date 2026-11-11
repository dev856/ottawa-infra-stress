"""Descriptive insights from the 311 snapshot: ward report cards and unusual spikes.

These are counts and dates from City records, not predictions.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

REPORT_WINDOW_DAYS = 365
SPIKE_WINDOW_DAYS = 7
# shortcut: "usual" is the previous 8 weeks, so a seasonal ramp-up (leaf pickup in the fall)
# can still flag; compare with the same weeks last year once 3+ years of data are kept.
SPIKE_BASELINE_DAYS = 56
SPIKE_MIN_COUNT = 5
SPIKE_MIN_RATIO = 3.0
# About 24 wards x 800 request types are tested each week, so the bar is strict.
SPIKE_MAX_PROBABILITY = 1e-4
BATCH_TOP_DAYS = 5
BATCH_SHARE = 0.30
BATCH_MIN_CLOSED = 200
# One day counts as part of a mass closure when it holds this share (and this many) of all
# closings of requests that had been open for BULK_MIN_AGE_DAYS or more.
BULK_MIN_AGE_DAYS = 30
BULK_DAY_SHARE = 0.02
BULK_DAY_MIN_CLOSED = 1000
WARD_MEDIAN_MIN_CLOSED = 5
REMAINING_MIN_SAMPLE = 30


def snapshot_end(requests: pd.DataFrame) -> pd.Timestamp:
    """Return the last day with requests, which every 'last N days' is measured from."""
    return pd.Timestamp(requests["opened"].max())


def last_days(requests: pd.DataFrame, days: int, end: pd.Timestamp) -> pd.DataFrame:
    """Return requests opened in the ``days`` days up to and including ``end``."""
    return requests[
        (requests["opened"] > end - pd.Timedelta(days=days)) & (requests["opened"] <= end)
    ]


def batch_closed_types(requests: pd.DataFrame) -> set[str]:
    """Return request types the City often closes in bulk on a few days.

    For these, days-to-close reflects paperwork more than when work was done.
    """
    closed = requests.dropna(subset=["closed"])
    counts = closed.groupby(["type", "closed"]).size()
    totals = closed.groupby("type").size()
    top_days = counts.groupby(level="type").nlargest(BATCH_TOP_DAYS).groupby(level=0).sum()
    share = (top_days / totals).loc[totals >= BATCH_MIN_CLOSED]
    return set(share[share >= BATCH_SHARE].index)


def bulk_closure_end(requests: pd.DataFrame) -> pd.Timestamp | None:
    """Return the last day of the City's mass closure of long-open requests, if there was one.

    In May 2026 the City closed about 130,000 old requests over nine days. For the request
    types caught up in it, those closing dates say when the records were tidied, not when
    the work was done.
    """
    old = requests[requests["days_to_close"] >= BULK_MIN_AGE_DAYS]
    per_day = old.groupby("closed").size()
    bulk_days = per_day[per_day >= max(BULK_DAY_MIN_CLOSED, BULK_DAY_SHARE * len(old))]
    return None if bulk_days.empty else pd.Timestamp(bulk_days.index.max())


def bulk_closed(
    requests: pd.DataFrame, batch_types: set[str], bulk_end: pd.Timestamp | None
) -> pd.Series:
    """Mark requests whose closing date cannot be trusted.

    These are batch-closed types opened before the mass closure ended: nearly all of them
    were closed in it, and the few that were not are too odd a sample to learn from.
    """
    if bulk_end is None:
        return pd.Series(False, index=requests.index)
    return requests["type"].isin(batch_types) & (requests["opened"] <= bulk_end)


def ward_totals(requests: pd.DataFrame, end: pd.Timestamp) -> dict[int, int]:
    """Return each ward's request count over the report window."""
    counts = last_days(requests, REPORT_WINDOW_DAYS, end)["ward"].dropna().value_counts()
    return {int(ward): int(count) for ward, count in counts.items()}


def ward_report(
    requests: pd.DataFrame,
    ward: int | None,
    end: pd.Timestamp,
    batch_types: set[str],
    top: int = 5,
) -> dict[str, Any]:
    """Summarize a ward's (or, with ``ward=None``, the city's) most common requests.

    Each request type's median days to close is shown next to the city-wide median.
    """
    window = last_days(requests, REPORT_WINDOW_DAYS, end)
    window = window[window["status"] != "Cancelled"]
    in_ward = window if ward is None else window[window["ward"] == ward]
    top_types = in_ward["type"].value_counts().head(top)
    ward_days = in_ward.groupby("type")["days_to_close"].median()
    city_days = window.groupby("type")["days_to_close"].median()
    return {
        "ward": ward,
        "total_requests": len(in_ward),
        "top_requests": [
            {
                "type": request_type,
                "count": int(count),
                "ward_median_days": _rounded(ward_days.get(request_type)),
                "city_median_days": _rounded(city_days.get(request_type)),
                "batch_closed": request_type in batch_types,
            }
            for request_type, count in top_types.items()
        ],
    }


def _rounded(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 1)


def ward_metrics(
    requests: pd.DataFrame, end: pd.Timestamp, request_type: str | None = None
) -> dict[int, dict[str, Any]]:
    """Return each ward's request count and median days to close, for one type or all."""
    window = last_days(requests, REPORT_WINDOW_DAYS, end)
    window = window[window["status"] != "Cancelled"].dropna(subset=["ward"])
    if request_type is not None:
        window = window[window["type"] == request_type]
    days = window.groupby("ward")["days_to_close"]
    closed = days.count()
    medians = days.median()
    return {
        int(ward): {
            "requests": int(count),
            # A median of a handful of requests is noise, so it is withheld.
            "median_days": _rounded(medians[ward])
            if closed[ward] >= WARD_MEDIAN_MIN_CLOSED
            else None,
        }
        for ward, count in days.size().items()
    }


def remaining_wait(
    requests: pd.DataFrame,
    request_type: str,
    ward: int,
    waited_days: int,
    mature_before: pd.Timestamp,
) -> dict[str, Any]:
    """Return how much longer requests still open on day ``waited_days`` took to close.

    With ``waited_days=0`` this is simply how long requests of this type take.

    Uses the ward's own history when it has enough such requests, else the whole city's.
    Only requests opened by ``mature_before`` count, so recent ones that have not had
    time to close do not make waits look shorter. ``None`` days mean "still not closed".
    """
    similar = requests[
        (requests["type"] == request_type)
        & (requests["status"] != "Cancelled")
        & (requests["opened"] <= mature_before)
    ]
    # A request still open in the snapshot is an endless wait, not a row to drop.
    days = similar["days_to_close"].astype(float).fillna(math.inf)
    more = days[days >= waited_days] - waited_days
    in_ward = more[similar.loc[more.index, "ward"] == ward]
    scope, sample = ("ward", in_ward) if len(in_ward) >= REMAINING_MIN_SAMPLE else ("city", more)
    enough = len(sample) >= REMAINING_MIN_SAMPLE

    def quantile(share: float) -> float | None:
        if not enough:
            return None
        # "lower" picks a real observation, so an endless wait stays infinite, not NaN.
        value = float(sample.quantile(share, interpolation="lower"))
        return None if math.isinf(value) else value

    return {
        "scope": scope,
        "sample_size": len(sample),
        "enough_history": enough,
        "usual_more_days": quantile(0.5),
        "nine_in_ten_more_days": quantile(0.9),
    }


def poisson_tail(count: int, expected: float) -> float:
    """Return P(X >= count) for X ~ Poisson(expected)."""
    if count <= 0:
        return 1.0
    term = math.exp(-expected)
    below = term
    for k in range(1, count):
        term *= expected / k
        below += term
    return max(0.0, 1.0 - below)


def find_spikes(requests: pd.DataFrame, end: pd.Timestamp) -> list[dict[str, Any]]:
    """Return ward/type pairs with far more requests this week than in recent weeks."""
    week = last_days(requests, SPIKE_WINDOW_DAYS, end).dropna(subset=["ward"])
    before = last_days(
        requests, SPIKE_BASELINE_DAYS, end - pd.Timedelta(days=SPIKE_WINDOW_DAYS)
    ).dropna(subset=["ward"])
    current = week.groupby(["ward", "type"]).size()
    usual = before.groupby(["ward", "type"]).size() * SPIKE_WINDOW_DAYS / SPIKE_BASELINE_DAYS

    spikes = []
    for (ward, request_type), count in current[current >= SPIKE_MIN_COUNT].items():
        # Half a request per week stands in for "none before", so new types are not infinite.
        expected = max(float(usual.get((ward, request_type), 0.0)), 0.5)
        if (
            count / expected >= SPIKE_MIN_RATIO
            and poisson_tail(int(count), expected) <= SPIKE_MAX_PROBABILITY
        ):
            spikes.append(
                {
                    "ward": int(ward),
                    "type": request_type,
                    "this_week": int(count),
                    "usual_per_week": round(expected, 1),
                }
            )
    return sorted(
        spikes, key=lambda spike: spike["this_week"] / spike["usual_per_week"], reverse=True
    )
