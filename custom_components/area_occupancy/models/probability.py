"""Probability model for Area Occupancy Detection."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from homeassistant.util import dt as dt_util

from ..const import MIN_PROBABILITY
from ..utils import validate_datetime, validate_prob

_LOGGER = logging.getLogger(__name__)


@dataclass
class Probability:
    """Represents the probability state for an entity."""

    probability: float
    decayed_probability: float
    decay_factor: float
    last_updated: datetime
    is_active: bool

    def __post_init__(self):
        """Validate properties after initialization."""
        self.probability = validate_prob(self.probability)
        self.decayed_probability = validate_prob(self.decayed_probability)
        self.decay_factor = validate_prob(self.decay_factor)
        self.last_updated = validate_datetime(self.last_updated)

    def update(
        self,
        probability: Optional[float] = None,
        decayed_probability: Optional[float] = None,
        decay_factor: Optional[float] = None,
        is_active: Optional[bool] = None,
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
        if decay_factor is not None:
            self.decay_factor = validate_prob(decay_factor)
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
