"""Tests for parsing the City of Ottawa 311 download."""

import pandas as pd
import pytest

from src.errors import DataValidationError
from src.ottawa_data import parse_service_requests

HEADER = (
    "﻿Service Request ID | Numéro de demande,Status | État,Type | Type,"
    "Description | Description,Opened Date | Date d'ouverture,"
    "Closed Date | Date de fermeture,Address | Adresse,Latitude | Latitude,"
    "Longitude | Longitude,Ward | Quartier,Channel | Voie de service\n"
)
ROWS = (
    "1,Resolved,Water and the Environment,Possible Leak - Leak Investigation | Fuite,"
    "2025-07-01,2025-07-09,12 Main St,45.4,-75.7,14,Phone\n"
    "2,Active,Garbage and Recycling,Blue Box - SWC | Boîte bleue,"
    "2025-07-02,\\N,\\N,\\N,\\N,\\N,Web\n"
    "3,Resolved,Bylaw Services,Noise - Music | Bruit,"
    "2025-06-30,2025-07-01,\\N,\\N,\\N,99,Web\n"
    "4,Resolved,Bylaw Services,Noise - Music | Bruit,not-a-date,2025-07-01,\\N,\\N,\\N,3,Web\n"
)


def test_service_requests_keep_only_safe_columns_in_english() -> None:
    """Addresses, coordinates, and IDs are dropped; labels keep the English half."""
    result = parse_service_requests((HEADER + ROWS).encode("utf-8"))

    assert list(result.columns) == [
        "opened",
        "closed",
        "category",
        "type",
        "ward",
        "channel",
        "status",
    ]
    assert result["type"].tolist() == [
        "Noise - Music",
        "Possible Leak - Leak Investigation",
        "Blue Box - SWC",
    ]
    assert result["opened"].tolist() == list(
        pd.to_datetime(["2025-06-30", "2025-07-01", "2025-07-02"])
    )


def test_missing_and_impossible_wards_become_blank_not_dropped() -> None:
    """City-wide counts keep requests without a valid ward."""
    result = parse_service_requests((HEADER + ROWS).encode("utf-8")).set_index("type")

    assert result.loc["Possible Leak - Leak Investigation", "ward"] == 14
    assert pd.isna(result.loc["Blue Box - SWC", "ward"])
    assert pd.isna(result.loc["Noise - Music", "ward"])
    assert pd.isna(result.loc["Blue Box - SWC", "closed"])


def test_service_requests_reject_unexpected_schema() -> None:
    """A renamed City file fails loudly instead of producing an empty snapshot."""
    with pytest.raises(DataValidationError, match="Ward"):
        parse_service_requests(b"Description,Opened Date\nNoise,2025-01-01\n")
