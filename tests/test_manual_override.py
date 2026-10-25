"""Tests for temporary manual occupancy overrides."""

from datetime import timedelta

from custom_components.area_occupancy.data.manual_override import (
    ManualOccupancyOverride,
)
from homeassistant.util import dt as dt_util


def test_override_round_trips() -> None:
    """A valid override survives storage serialization."""
    now = dt_util.utcnow()
    override = ManualOccupancyOverride(True, now, now + timedelta(hours=1))

    restored = ManualOccupancyOverride.from_dict(override.to_dict())

    assert restored is not None
    assert restored.occupied is True
    assert restored.expires_at == override.expires_at


def test_expired_override_is_not_restored() -> None:
    """Expired state safely returns to automatic detection after restart."""
    now = dt_util.utcnow()
    data = ManualOccupancyOverride(False, now - timedelta(hours=2), now).to_dict()

    assert ManualOccupancyOverride.from_dict(data) is None


def test_malformed_override_is_ignored() -> None:
    """Hand-edited or corrupt storage cannot create a manual state."""
    assert ManualOccupancyOverride.from_dict({"occupied": True}) is None
