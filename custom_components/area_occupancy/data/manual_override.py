"""Point-in-time manual occupancy reference feedback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from homeassistant.util import dt as dt_util


@dataclass(frozen=True)
class ManualOccupancyFeedback:
    """A user-confirmed occupancy state at one point in time."""

    feedback_id: str
    area_id: str
    occupied: bool
    recorded_at: datetime
    automatic_probability: float | None

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ManualOccupancyFeedback | None:
        """Restore feedback, rejecting malformed records."""
        try:
            feedback_id = data["feedback_id"]
            area_id = data["area_id"]
            occupied = data["occupied"]
            recorded_at = dt_util.parse_datetime(str(data["recorded_at"]))
            probability = data.get("automatic_probability")
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not isinstance(feedback_id, str)
            or not feedback_id
            or not isinstance(area_id, str)
            or not area_id
            or not isinstance(occupied, bool)
            or recorded_at is None
        ):
            return None
        if probability is not None and (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1
        ):
            return None
        return cls(
            feedback_id,
            area_id,
            occupied,
            recorded_at,
            float(probability) if probability is not None else None,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the storage representation."""
        return {
            "feedback_id": self.feedback_id,
            "area_id": self.area_id,
            "occupied": self.occupied,
            "recorded_at": self.recorded_at.isoformat(),
            "automatic_probability": self.automatic_probability,
        }
