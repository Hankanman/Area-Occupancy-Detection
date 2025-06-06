"""Decay model for Area Occupancy Detection."""

from dataclasses import dataclass, replace
from datetime import datetime
import logging
import math
from typing import Final

from homeassistant.util import dt as dt_util

from ..const import (
    DECAY_LAMBDA,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from ..utils import validate_prob, validate_datetime

_LOGGER = logging.getLogger(__name__)

# Validation constants
MIN_DECAY_WINDOW: Final[int] = 1  # Minimum decay window in seconds
MAX_DECAY_WINDOW: Final[int] = 3600  # Maximum decay window in seconds


@dataclass
class Decay:
    """Represents decay state for an entity."""

    entity_id: str
    is_decaying: bool = False
    decay_start_time: datetime | None = None
    decay_start_probability: float | None = None
    decay_window: int = 300  # Default 5 minutes
    decay_enabled: bool = True

    def __post_init__(self):
        """Validate properties after initialization."""
        if self.decay_start_probability is not None:
            self.decay_start_probability = validate_prob(self.decay_start_probability)
        self.decay_start_time = validate_datetime(self.decay_start_time)
        
        # Validate decay window
        if not isinstance(self.decay_window, (int, float)):
            raise ValueError(f"Decay window must be a number, got {type(self.decay_window)}")
        if not MIN_DECAY_WINDOW <= self.decay_window <= MAX_DECAY_WINDOW:
            raise ValueError(
                f"Decay window must be between {MIN_DECAY_WINDOW} and {MAX_DECAY_WINDOW} seconds"
            )

    def calculate_decay(
        self,
        current_probability: float,
        previous_probability: float,
    ) -> tuple[float, float, bool, datetime | None, float | None]:
        """Calculate decay factor and apply it to probability.

        Args:
            current_probability: The current probability value (used to detect increase)
            previous_probability: The probability value from the previous step

        Returns:
            Tuple of (
                decayed_probability: The new probability after applying decay
                decay_factor: The calculated decay factor (1.0 if no decay applied)
                new_is_decaying: The updated decaying status (bool)
                new_decay_start_time: The updated decay start time (datetime | None)
                new_decay_start_probability: The updated probability when decay started (float | None)
            )

        Raises:
            ValueError: If probabilities are invalid
        """
        # Validate inputs
        if not self._validate_probabilities(current_probability, previous_probability):
            raise ValueError("Probabilities must be between 0 and 1")

        if not self.decay_enabled:
            _LOGGER.debug("Decay disabled for %s", self.entity_id)
            return current_probability, 1.0, False, None, None

        # --- Determine new decay state ---
        new_decay_start_time = self.decay_start_time
        new_is_decaying = self.is_decaying
        new_decay_start_probability = self.decay_start_probability

        # Condition 1: Probability increased or stayed the same
        if current_probability >= previous_probability:
            if self.is_decaying:  # Log only if decay was previously active
                _LOGGER.debug(
                    "Probability increased/stable for %s: %.3f -> %.3f. Stopping decay",
                    self.entity_id,
                    previous_probability,
                    current_probability,
                )
            new_decay_start_time = None
            new_is_decaying = False
            new_decay_start_probability = None
            return (
                current_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )

        # Condition 2: Probability decreased, decay wasn't active before
        if not self.is_decaying:
            _LOGGER.debug(
                "Probability decreased for %s: %.3f -> %.3f. Starting decay",
                self.entity_id,
                previous_probability,
                current_probability,
            )
            new_decay_start_time = dt_util.utcnow()
            new_is_decaying = True
            new_decay_start_probability = previous_probability
            return (
                previous_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )

        # Condition 3: Probability decreased, decay was already active
        if not self.decay_start_time or self.decay_start_probability is None:
            _LOGGER.error(
                "Inconsistent state for %s: is_decaying=True but decay_start_time (%s) or decay_start_probability (%s) is None. Resetting decay",
                self.entity_id,
                self.decay_start_time,
                self.decay_start_probability,
            )
            new_decay_start_time = dt_util.utcnow()
            new_is_decaying = True
            new_decay_start_probability = previous_probability
            return (
                previous_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )

        # --- Calculate decay based on existing start time and start probability ---
        try:
            elapsed = (dt_util.utcnow() - self.decay_start_time).total_seconds()
            decay_factor = self._calculate_decay_factor(elapsed)

            # Calculate the potential decayed value based on the probability when decay *started*
            potential_decayed_prob = self._apply_decay_factor(
                self.decay_start_probability, decay_factor
            )

            # The final decayed probability should not be lower than what current sensors suggest
            decayed_probability = max(potential_decayed_prob, current_probability)

            # Check if decay reached minimum threshold
            if decayed_probability <= MIN_PROBABILITY:
                _LOGGER.debug("Decay complete for %s: reached minimum probability", self.entity_id)
                new_decay_start_time = None
                new_is_decaying = False
                new_decay_start_probability = None
                return (
                    MIN_PROBABILITY,
                    decay_factor,
                    new_is_decaying,
                    new_decay_start_time,
                    new_decay_start_probability,
                )

            # Decay continues
            _LOGGER.debug(
                "Decay active for %s: t=%.1fs factor=%.3f p_start=%.3f -> p_decayed=%.3f (floor=%.3f)",
                self.entity_id,
                elapsed,
                decay_factor,
                self.decay_start_probability,
                decayed_probability,
                current_probability,
            )

        except (ValueError, ZeroDivisionError) as err:
            _LOGGER.error("Error in decay calculation for %s: %s", self.entity_id, err)
            new_is_decaying = False
            new_decay_start_time = None
            new_decay_start_probability = None
            return (
                current_probability,
                1.0,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )
        else:
            return (
                decayed_probability,
                decay_factor,
                new_is_decaying,
                new_decay_start_time,
                new_decay_start_probability,
            )

    def _validate_probabilities(
        self, current_probability: float, previous_probability: float
    ) -> bool:
        """Validate probability values.

        Args:
            current_probability: The current probability value
            previous_probability: The previous probability value

        Returns:
            True if all probabilities are valid, False otherwise
        """
        return all(0 <= p <= 1 for p in (current_probability, previous_probability))

    def _calculate_decay_factor(self, elapsed_seconds: float) -> float:
        """Calculate the decay factor based on elapsed time.

        Args:
            elapsed_seconds: Number of seconds elapsed since decay start

        Returns:
            Calculated decay factor
        """
        return math.exp(-DECAY_LAMBDA * (elapsed_seconds / self.decay_window))

    def _apply_decay_factor(self, probability: float, decay_factor: float) -> float:
        """Apply decay factor to probability with bounds checking.

        Args:
            probability: The probability to decay
            decay_factor: The decay factor to apply

        Returns:
            The decayed probability, bounded between MIN_PROBABILITY and MAX_PROBABILITY
        """
        return max(
            MIN_PROBABILITY,
            min(probability * decay_factor, MAX_PROBABILITY),
        )


class DecayManager:
    """Manages decay calculations for multiple entities."""

    def __init__(self, coordinator) -> None:
        """Initialize the decay manager."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self._decays: dict[str, Decay] = {}

    @property
    def decays(self) -> dict[str, Decay]:
        """Get all decay states."""
        return self._decays

    def get_decay(self, entity_id: str) -> Decay:
        """Get decay state for an entity."""
        if entity_id not in self._decays:
            self._decays[entity_id] = Decay(
                entity_id=entity_id,
                decay_enabled=self.config.decay.enabled,
                decay_window=self.config.decay.window,
            )
        return self._decays[entity_id]

    def update_decay(
        self,
        entity_id: str,
        current_probability: float,
        previous_probability: float,
    ) -> tuple[float, float]:
        """Update decay state and get new probability for an entity.

        Args:
            entity_id: The entity ID to update decay for
            current_probability: Current probability value
            previous_probability: Previous probability value

        Returns:
            Tuple of (decayed_probability, decay_factor)
        """
        decay = self.get_decay(entity_id)
        (
            decayed_probability,
            decay_factor,
            new_is_decaying,
            new_decay_start_time,
            new_decay_start_probability,
        ) = decay.calculate_decay(current_probability, previous_probability)

        # Update decay state
        self._decays[entity_id] = replace(
            decay,
            is_decaying=new_is_decaying,
            decay_start_time=new_decay_start_time,
            decay_start_probability=new_decay_start_probability,
        )

        return decayed_probability, decay_factor

    def reset_decay(self, entity_id: str) -> None:
        """Reset decay state for an entity."""
        if entity_id in self._decays:
            self._decays[entity_id] = Decay(
                entity_id=entity_id,
                decay_enabled=self.config.decay.enabled,
                decay_window=self.config.decay.window,
            )

    def reset_all_decays(self) -> None:
        """Reset all decay states."""
        self._decays.clear() 