"""Tests for point-in-time manual occupancy references."""

from datetime import timedelta

from custom_components.area_occupancy.data.manual_override import (
    ManualOccupancyFeedback,
)
from homeassistant.util import dt as dt_util


def test_feedback_round_trips() -> None:
    """A valid reference survives storage serialization."""
    now = dt_util.utcnow()
    feedback = ManualOccupancyFeedback("id", "living_room", True, now, 0.75)

    restored = ManualOccupancyFeedback.from_dict(feedback.to_dict())

    assert restored == feedback


def test_feedback_without_probability_round_trips() -> None:
    """Unavailable automatic predictions are represented as null."""
    now = dt_util.utcnow()
    feedback = ManualOccupancyFeedback("id", "living_room", False, now, None)

    assert ManualOccupancyFeedback.from_dict(feedback.to_dict()) == feedback


def test_malformed_feedback_is_ignored() -> None:
    """Hand-edited or corrupt storage cannot create a reference."""
    assert ManualOccupancyFeedback.from_dict({"occupied": True}) is None


def test_invalid_probability_is_ignored() -> None:
    """A probability outside the model range is rejected."""
    now = dt_util.utcnow()
    data = ManualOccupancyFeedback("id", "living_room", True, now, 0.5).to_dict()
    data["automatic_probability"] = 2

    assert ManualOccupancyFeedback.from_dict(data) is None


def test_old_expiry_shape_is_ignored() -> None:
    """Old state-changing override data is not reinterpreted as feedback."""
    now = dt_util.utcnow()
    data = {
        "occupied": True,
        "set_at": (now - timedelta(hours=1)).isoformat(),
        "expires_at": now.isoformat(),
    }

    assert ManualOccupancyFeedback.from_dict(data) is None
