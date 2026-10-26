"""Tests for point-in-time manual occupancy references."""

from datetime import UTC, datetime, timedelta

import pytest

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


def test_feedback_normalizes_ids_and_timestamp_to_utc() -> None:
    """Identifiers are trimmed and aware timestamps are stored in UTC."""
    feedback = ManualOccupancyFeedback(
        "  feedback  ",
        "  living_room  ",
        True,
        datetime(2026, 10, 25, 22, 0, tzinfo=UTC),
        None,
    )

    assert feedback.feedback_id == "feedback"
    assert feedback.area_id == "living_room"
    assert feedback.recorded_at.tzinfo == UTC


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("feedback_id", 1),
        ("area_id", 1),
        ("occupied", 1),
        ("recorded_at", "2026-10-25T22:00:00+00:00"),
    ],
)
def test_direct_construction_rejects_wrong_types(field: str, value: object) -> None:
    """Direct callers receive a clear type error for invalid fields."""
    values: dict[str, object] = {
        "feedback_id": "id",
        "area_id": "living_room",
        "occupied": True,
        "recorded_at": dt_util.utcnow(),
        "automatic_probability": None,
    }
    values[field] = value

    with pytest.raises(TypeError):
        ManualOccupancyFeedback(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("probability", ["0.5", True, float("nan"), float("inf")])
def test_direct_construction_rejects_invalid_probability(
    probability: object,
) -> None:
    """Probabilities must be finite numeric values in the model range."""
    with pytest.raises(ValueError):
        ManualOccupancyFeedback(
            "id",
            "living_room",
            True,
            dt_util.utcnow(),
            probability,  # type: ignore[arg-type]
        )


def test_naive_timestamp_is_ignored() -> None:
    """Stored timestamps without a timezone are rejected."""
    data = ManualOccupancyFeedback(
        "id", "living_room", True, dt_util.utcnow(), None
    ).to_dict()
    data["recorded_at"] = "2026-10-25T22:00:00"

    assert ManualOccupancyFeedback.from_dict(data) is None


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
