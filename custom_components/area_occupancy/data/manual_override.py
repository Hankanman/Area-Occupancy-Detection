"""Point-in-time manual occupancy reference feedback."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math

from homeassistant.util import dt as dt_util


@dataclass(frozen=True)
class ManualOccupancyFeedback:
    """A user-confirmed occupancy state at one point in time."""

    feedback_id: str
    area_id: str
    occupied: bool
    recorded_at: datetime
    automatic_probability: float | None

    def __post_init__(self) -> None:
        """Validate and normalize feedback created directly or restored."""
        if not isinstance(self.feedback_id, str):
            raise TypeError("feedback_id must be a string")
        feedback_id = self.feedback_id.strip()
        if not feedback_id:
            raise ValueError("feedback_id must be a non-empty string")
        object.__setattr__(self, "feedback_id", feedback_id)

        if not isinstance(self.area_id, str):
            raise TypeError("area_id must be a string")
        area_id = self.area_id.strip()
        if not area_id:
            raise ValueError("area_id must be a non-empty string")
        object.__setattr__(self, "area_id", area_id)

        if not isinstance(self.occupied, bool):
            raise TypeError("occupied must be a boolean")

        recorded_at = self.recorded_at
        if not isinstance(recorded_at, datetime):
            raise TypeError("recorded_at must be a datetime")
        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be a timezone-aware datetime")

        # Normalize timestamps so comparisons and persistence use UTC.
        object.__setattr__(self, "recorded_at", dt_util.as_utc(recorded_at))

        probability = self.automatic_probability
        if probability is not None:
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    "automatic_probability must be None or a finite "
                    "number between 0 and 1"
                )

            object.__setattr__(
                self,
                "automatic_probability",
                float(probability),
            )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> ManualOccupancyFeedback | None:
        """Restore feedback, rejecting malformed stored records."""
        if not isinstance(data, dict):
            return None

        try:
            feedback_id = data["feedback_id"]
            area_id = data["area_id"]
            occupied = data["occupied"]
            raw_timestamp = data["recorded_at"]
            probability = data.get("automatic_probability")

            if (
                not isinstance(feedback_id, str)
                or not feedback_id.strip()
                or not isinstance(area_id, str)
                or not area_id.strip()
                or not isinstance(occupied, bool)
                or not isinstance(raw_timestamp, str)
            ):
                return None

            recorded_at = dt_util.parse_datetime(raw_timestamp)

            # A timestamp without a timezone is ambiguous and is rejected.
            if (
                recorded_at is None
                or recorded_at.tzinfo is None
                or recorded_at.utcoffset() is None
            ):
                return None

            if probability is not None and (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                return None

            return cls(
                feedback_id=feedback_id,
                area_id=area_id,
                occupied=occupied,
                recorded_at=recorded_at,
                automatic_probability=(
                    float(probability) if probability is not None else None
                ),
            )

        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    def to_dict(self) -> dict[str, object]:
        """Return the storage representation."""
        return {
            "feedback_id": self.feedback_id,
            "area_id": self.area_id,
            "occupied": self.occupied,
            "recorded_at": self.recorded_at.isoformat(),
            "automatic_probability": self.automatic_probability,
        }
