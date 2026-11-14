"""Time normalization utilities.

Timezone policy for this integration:

- Runtime arithmetic/comparisons: **timezone-aware UTC** datetimes.
- Database persistence (SQLite): **naive UTC** datetimes (tzinfo=None, interpreted as UTC).
- Wall-clock bucketing (time priors, daily/weekly/monthly grouping): **Home Assistant local timezone**.
"""

from __future__ import annotations

from datetime import datetime

from homeassistant.util import dt as dt_util


def to_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime.

    - If `value` is naive, it is assumed to already be UTC.
    - If `value` is timezone-aware in a non-UTC timezone, it is converted to UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=dt_util.UTC)
    return value.astimezone(dt_util.UTC)


def to_db_utc(value: datetime) -> datetime:
    """Return a naive UTC datetime suitable for SQLite persistence."""
    return to_utc(value).replace(tzinfo=None)


def from_db_utc(value: datetime) -> datetime:
    """Return a timezone-aware UTC datetime from a DB value.

    SQLite frequently returns naive datetimes even when columns are declared
    with timezone support. We interpret naive datetimes as UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=dt_util.UTC)
    return value.astimezone(dt_util.UTC)


def to_local(value: datetime) -> datetime:
    """Return a timezone-aware local datetime for wall-clock bucketing."""
    return dt_util.as_local(to_utc(value))


def assert_utc_aware(value: datetime, context: str = "") -> None:
    """Assert that a datetime is timezone-aware UTC (debug helper)."""
    if value.tzinfo is None or value.tzinfo != dt_util.UTC:
        ctx = f" ({context})" if context else ""
        raise ValueError(
            f"Expected UTC-aware datetime{ctx}, got: {value} (tz={value.tzinfo})"
        )


# Backwards-compatible aliases (keep time-related helpers out of utils.py)
def ensure_timezone_aware(value: datetime) -> datetime:
    """Ensure a datetime is timezone-aware, assuming UTC if naive.

    NOTE: This does NOT convert timezone-aware values to UTC.
    Prefer `to_utc()` for runtime arithmetic/comparisons.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=dt_util.UTC)
    return value


def ensure_utc_datetime(value: datetime) -> datetime:
    """Backwards-compatible alias for `to_utc()`."""
    return to_utc(value)
