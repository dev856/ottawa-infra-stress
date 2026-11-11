"""Tests for ward report cards, batch-closure detection, and spike alerts."""

import math

import pandas as pd
import pytest

from src.insights import (
    batch_closed_types,
    bulk_closed,
    bulk_closure_end,
    find_spikes,
    poisson_tail,
    remaining_wait,
    ward_metrics,
    ward_report,
    ward_totals,
)

END = pd.Timestamp("2026-10-31")


def requests_frame(rows: list[tuple[str, str | None, str, int | None, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["opened", "closed", "type", "ward", "status"])
    frame["opened"] = pd.to_datetime(frame["opened"])
    frame["closed"] = pd.to_datetime(frame["closed"])
    frame["ward"] = frame["ward"].astype("Int64")
    frame["days_to_close"] = (frame["closed"] - frame["opened"]).dt.days
    return frame


def test_poisson_tail_matches_closed_form() -> None:
    assert poisson_tail(0, 2.0) == 1.0
    assert poisson_tail(1, 2.0) == pytest.approx(1 - math.exp(-2))
    assert poisson_tail(3, 2.0) == pytest.approx(1 - math.exp(-2) * (1 + 2 + 2))


def test_spike_needs_count_ratio_and_improbability() -> None:
    """Ten reports against a usual one a week flags; a steady type does not."""
    rows = []
    # Steady: 7 a week in ward 1 for 9 weeks.
    for day in pd.date_range(END - pd.Timedelta(days=62), END, freq="D"):
        rows.append((str(day.date()), None, "Steady", 1, "Active"))
    # Jumpy: one a week for 8 weeks, then 10 this week.
    for week in range(1, 9):
        rows.append(
            (str((END - pd.Timedelta(days=7 * week + 1)).date()), None, "Light Out", 2, "Active")
        )
    rows += [(str(END.date()), None, "Light Out", 2, "Active")] * 10

    spikes = find_spikes(requests_frame(rows), END)

    assert spikes == [{"ward": 2, "type": "Light Out", "this_week": 10, "usual_per_week": 1.0}]


def test_small_counts_never_flag() -> None:
    rows = [(str(END.date()), None, "Rare", 3, "Active")] * 4
    assert find_spikes(requests_frame(rows), END) == []


def test_batch_closed_types_flags_bulk_closures_only() -> None:
    rows = []
    for i in range(300):
        opened = (pd.Timestamp("2025-01-01") + pd.Timedelta(days=i % 200)).date()
        rows.append((str(opened), "2026-05-15", "Pothole", 1, "Resolved"))
        rows.append(
            (
                str(opened),
                str((pd.Timestamp(opened) + pd.Timedelta(days=2)).date()),
                "Noise",
                1,
                "Resolved",
            )
        )

    assert batch_closed_types(requests_frame(rows)) == {"Pothole"}


def test_ward_report_compares_ward_with_city_and_skips_cancelled() -> None:
    rows = [
        ("2026-10-01", "2026-10-03", "Noise", 1, "Resolved"),
        ("2026-10-01", "2026-10-05", "Noise", 1, "Resolved"),
        ("2026-10-01", "2026-10-11", "Noise", 2, "Resolved"),
        ("2026-10-01", None, "Noise", 1, "Cancelled"),
        ("2026-10-01", "2026-10-02", "Garbage", 1, "Resolved"),
        ("2024-01-01", "2024-01-02", "Old", 1, "Resolved"),
    ]

    report = ward_report(requests_frame(rows), 1, END, batch_types={"Garbage"})

    assert report["total_requests"] == 3
    assert report["top_requests"] == [
        {
            "type": "Noise",
            "count": 2,
            "ward_median_days": 3.0,
            "city_median_days": 4.0,
            "batch_closed": False,
        },
        {
            "type": "Garbage",
            "count": 1,
            "ward_median_days": 1.0,
            "city_median_days": 1.0,
            "batch_closed": True,
        },
    ]


def test_ward_totals_ignore_requests_without_ward_or_older_than_a_year() -> None:
    rows = [
        ("2026-10-01", None, "A", 1, "Active"),
        ("2026-10-01", None, "A", None, "Active"),
        ("2025-01-01", None, "A", 1, "Active"),
    ]
    assert ward_totals(requests_frame(rows), END) == {1: 1}


def test_ward_metrics_counts_per_ward_and_withholds_thin_medians() -> None:
    rows = [("2026-10-01", "2026-10-03", "Noise", 1, "Resolved")] * 6
    rows += [("2026-10-01", "2026-10-09", "Noise", 2, "Resolved")] * 2
    rows += [("2026-10-01", "2026-10-02", "Pothole", 2, "Resolved")] * 6
    rows += [("2026-10-01", "2026-10-02", "Noise", 1, "Cancelled")]
    frame = requests_frame(rows)

    noise = ward_metrics(frame, END, "Noise")
    assert noise[1] == {"requests": 6, "median_days": 2.0}
    assert noise[2] == {"requests": 2, "median_days": None}  # too few closed to say
    assert ward_metrics(frame, END)[2] == {"requests": 8, "median_days": 1.0}


def test_remaining_wait_conditions_on_days_already_waited() -> None:
    mature = pd.Timestamp("2026-07-01")
    rows = [("2026-01-01", "2026-01-03", "Noise", 1, "Resolved")] * 50  # closed in 2 days
    rows += [("2026-01-01", "2026-01-21", "Noise", 1, "Resolved")] * 20  # closed in 20 days
    rows += [("2026-01-01", None, "Noise", 1, "Open")] * 20  # never closed
    rows += [("2026-10-01", "2026-10-02", "Noise", 1, "Resolved")] * 99  # too recent to count
    frame = requests_frame(rows)

    result = remaining_wait(frame, "Noise", 1, 5, mature)
    # After 5 days the quick 50 are gone: 20 close 15 days later, 20 never do.
    assert result["scope"] == "ward"
    assert result["sample_size"] == 40
    assert result["usual_more_days"] == 15.0
    assert result["nine_in_ten_more_days"] is None

    elsewhere = remaining_wait(frame, "Noise", 2, 5, mature)
    assert elsewhere["scope"] == "city"
    assert elsewhere["sample_size"] == 40

    thin = remaining_wait(frame, "Pothole", 1, 5, mature)
    assert thin["enough_history"] is False
    assert thin["usual_more_days"] is None


def test_bulk_closure_marks_only_batch_types_opened_before_it_ended() -> None:
    rows = [("2025-01-01", "2026-05-19", "Pothole", 1, "Resolved")] * 1500
    rows += [("2025-02-01", "2026-05-20", "Pothole", 1, "Resolved")] * 1500
    rows += [("2026-06-01", "2026-06-05", "Pothole", 1, "Resolved")] * 10
    rows += [("2025-01-01", "2025-01-02", "Noise", 1, "Resolved")] * 3000
    frame = requests_frame(rows)

    end = bulk_closure_end(frame)
    assert end == pd.Timestamp("2026-05-20")
    untrusted = bulk_closed(frame, {"Pothole"}, end)
    assert untrusted.sum() == 3000
    assert not untrusted[frame["opened"] > end].any()
    assert not untrusted[frame["type"] == "Noise"].any()

    quiet = frame[frame["type"] == "Noise"]
    assert bulk_closure_end(quiet) is None
    assert not bulk_closed(quiet, {"Noise"}, None).any()
