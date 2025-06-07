"""Probability model for Area Occupancy Detection."""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Any

from homeassistant.util import dt as dt_util

from ..const import MAX_PROBABILITY, MIN_PROBABILITY
from ..exceptions import StateError
from ..utils import validate_datetime, validate_prob
from .decay import Decay

_LOGGER = logging.getLogger(__name__)


@dataclass
class Probability:
    """Probability model for Area Occupancy Detection."""

    probability: float
    decayed_probability: float
    decay_factor: float
    last_updated: datetime
    is_active: bool

    def __post_init__(self) -> None:
        """Validate properties after initialization."""
        self.probability = validate_prob(self.probability)
        self.decayed_probability = validate_prob(self.decayed_probability)
        self.last_updated = validate_datetime(self.last_updated)

    def update(
        self,
        probability: float | None = None,
        decayed_probability: float | None = None,
        decay_factor: float | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Update probability values.

        Args:
            probability: New probability value
            decayed_probability: New decayed probability value
            decay_factor: New decay factor
            is_active: New active status

        """
        if probability is not None:
            self.probability = validate_prob(probability)
        if decayed_probability is not None:
            self.decayed_probability = validate_prob(decayed_probability)
        if is_active is not None:
            self.is_active = is_active
        self.last_updated = dt_util.utcnow()

    def reset(self) -> None:
        """Reset probability state to defaults."""
        self.probability = 0.0
        self.decayed_probability = 0.0
        self.decay_factor = 1.0
        self.is_active = False
        self.last_updated = dt_util.utcnow()

    def calculate_probability(
        self,
        prior: float,
        prob_true: float,
        prob_false: float,
    ) -> None:
        """Calculate probability using Bayesian update.

        Args:
            prior: The prior probability
            prob_true: Probability given true
            prob_false: Probability given false

        """
        # Calculate probability using Bayesian update
        probability = self._calculate_bayesian_probability(
            prior,
            prob_true,
            prob_false,
            self.is_active,
        )

        # Update the instance
        self.update(
            probability=probability,
            decayed_probability=probability,  # Initially same as probability
            decay_factor=1.0,  # No decay initially
        )

    @staticmethod
    def _calculate_bayesian_probability(
        prior: float,
        prob_true: float,
        prob_false: float,
        is_active: bool,
    ) -> float:
        """Calculate probability using Bayesian update.

        Args:
            prior: The prior probability
            prob_true: Probability given true
            prob_false: Probability given false
            is_active: Whether the entity is active

        Returns:
            float: The calculated probability

        """
        if is_active:
            # P(occupied | active) = P(active | occupied) * P(occupied) / P(active)
            numerator = prob_true * prior
            denominator = (prob_true * prior) + (prob_false * (1 - prior))
        else:
            # P(occupied | inactive) = P(inactive | occupied) * P(occupied) / P(inactive)
            numerator = (1 - prob_true) * prior
            denominator = ((1 - prob_true) * prior) + ((1 - prob_false) * (1 - prior))

        if denominator == 0:
            return MIN_PROBABILITY

        probability = numerator / denominator
        return validate_prob(probability)

    def apply_decay(self, decay: Decay) -> None:
        """Apply decay to the probability.

        Args:
            decay: The decay configuration to apply

        Raises:
            StateError: If there's an error applying the decay

        """
        try:
            if not decay.decay_enabled:
                return

            # Calculate time difference
            now = dt_util.utcnow()
            if decay.decay_start_time is None:
                decay.decay_start_time = now
                decay.decay_start_probability = self.probability
                return

            time_diff = (now - decay.decay_start_time).total_seconds()
            if time_diff < decay.decay_window:
                return

            # Calculate decay factor
            decay_factor = max(
                MIN_PROBABILITY,
                min(
                    MAX_PROBABILITY,
                    1.0 - (time_diff / decay.decay_window) * decay.decay_factor,
                ),
            )

            # Apply decay
            self.decay_factor = decay_factor
            self.decayed_probability = max(
                MIN_PROBABILITY,
                min(MAX_PROBABILITY, self.probability * decay_factor),
            )

        except Exception as err:
            raise StateError(f"Error applying decay: {err}") from err

    def to_dict(self) -> dict[str, Any]:
        """Convert probability to dictionary for storage."""
        return {
            "probability": self.probability,
            "decayed_probability": self.decayed_probability,
            "decay_factor": self.decay_factor,
            "last_updated": self.last_updated.isoformat(),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Probability":
        """Create probability from dictionary."""
        last_updated = dt_util.parse_datetime(data["last_updated"])
        if last_updated is None:
            last_updated = dt_util.utcnow()

        return cls(
            probability=data["probability"],
            decayed_probability=data["decayed_probability"],
            decay_factor=data["decay_factor"],
            last_updated=last_updated,
            is_active=data["is_active"],
        )
