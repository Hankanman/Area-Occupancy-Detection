"""Temporary manual occupancy overrides."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from homeassistant.util import dt as dt_util


@dataclass(frozen=True)
class ManualOccupancyOverride:
    """A user-supplied occupancy state with an optional expiry."""

    occupied: bool
    set_at: datetime
    expires_at: datetime

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ManualOccupancyOverride | None:
        """Restore an override, rejecting malformed or already expired data."""
        try:
            occupied = data["occupied"]
            set_at = dt_util.parse_datetime(str(data["set_at"]))
            expires_at = dt_util.parse_datetime(str(data["expires_at"]))
        except (KeyError, TypeError, ValueError):
            return None
        if not isinstance(occupied, bool) or set_at is None or expires_at is None:
            return None
        if expires_at <= dt_util.utcnow():
            return None
        return cls(occupied, set_at, expires_at)

    def to_dict(self) -> dict[str, object]:
        """Return the storage representation."""
        return {
            "occupied": self.occupied,
            "set_at": self.set_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }
