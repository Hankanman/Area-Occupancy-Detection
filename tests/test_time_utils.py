"""Tests for time normalization utilities."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from custom_components.area_occupancy.time_utils import (
    from_db_utc,
    to_db_utc,
    to_local,
    to_utc,
)
from homeassistant.util import dt as dt_util


@pytest.fixture
def set_tz_america_new_york():
    """Temporarily set HA default timezone to America/New_York for DST tests."""
    original = dt_util.DEFAULT_TIME_ZONE
    dt_util.set_default_time_zone(ZoneInfo("America/New_York"))
    try:
        yield
    finally:
        dt_util.set_default_time_zone(original)


class TestTimeUtils:
    def test_to_utc_naive_assumed_utc(self) -> None:
        value = datetime(2025, 1, 1, 12, 0, 0)  # naive
        out = to_utc(value)
        assert out.tzinfo == dt_util.UTC
        assert out.year == 2025 and out.hour == 12

    def test_to_utc_converts_non_utc(self) -> None:
        value = datetime(2025, 1, 1, 12, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        out = to_utc(value)
        assert out.tzinfo == dt_util.UTC
        # 12:00 in NY (winter, UTC-5) => 17:00 UTC
        assert out.hour == 17

    def test_to_db_utc_returns_naive(self) -> None:
        value = datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt_util.UTC)
        out = to_db_utc(value)
        assert out.tzinfo is None
        assert out == datetime(2025, 1, 1, 12, 0, 0)

    def test_from_db_utc_returns_aware(self) -> None:
        value = datetime(2025, 1, 1, 12, 0, 0)  # naive from DB
        out = from_db_utc(value)
        assert out.tzinfo == dt_util.UTC
        assert out.hour == 12

    def test_to_local_uses_default_timezone(self, set_tz_america_new_york) -> None:
        value = datetime(2025, 1, 1, 12, 0, 0, tzinfo=dt_util.UTC)
        out = to_local(value)
        assert out.tzinfo is not None
        # 12:00 UTC => 07:00 local (winter, UTC-5)
        assert out.hour == 7
