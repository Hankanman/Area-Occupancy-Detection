"""Probability decay handling for Area Occupancy Detection."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from homeassistant.util import dt as dt_util

from ..const import MAX_PROBABILITY, MIN_PROBABILITY
from ..state.containers import ProbabilityState

_LOGGER = logging.getLogger(__name__)


class Decay:
    """Handle probability decay over time.

    This class manages the decay of occupancy probability over time when no
    new sensor events are detected. It implements a configurable decay window
    and minimum delay before decay starts.

    Attributes:
        decay_window: Time window for probability decay in minutes
        decay_min_delay: Minimum delay before decay starts in minutes

    """

    def __init__(
        self,
        decay_window: int = 30,
        decay_min_delay: int = 5,
    ) -> None:
        """Initialize the decay handler.

        Args:
            decay_window: Time window for probability decay in minutes
            decay_min_delay: Minimum delay before decay starts in minutes

        """
        self.decay_window = decay_window
        self.decay_min_delay = decay_min_delay

    def should_start_decay(
        self,
        state: ProbabilityState,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Check if probability decay should start.

        This method determines if probability decay should start based on the
        current state and time since the last update.

        Args:
            state: Current probability state
            current_time: Current time (defaults to UTC now)

        Returns:
            True if decay should start, False otherwise

        """
        if not state.decay_start_time:
            return False

        if current_time is None:
            current_time = dt_util.utcnow()

        time_since_update = current_time - state.decay_start_time
        min_delay = timedelta(minutes=self.decay_min_delay)

        return time_since_update >= min_delay

    def calculate_decay(
        self,
        state: ProbabilityState,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Calculate decayed probability value.

        This method calculates the decayed probability value based on the
        current state and time since decay started.

        Args:
            state: Current probability state
            current_time: Current time (defaults to UTC now)

        Returns:
            Decayed probability value

        """
        if not state.decay_start_time or not state.decay_start_probability:
            return state.probability

        if current_time is None:
            current_time = dt_util.utcnow()

        # Calculate time since decay started
        time_since_decay = current_time - state.decay_start_time
        decay_minutes = time_since_decay.total_seconds() / 60

        # Calculate decay factor (0 to 1)
        decay_factor = min(1.0, decay_minutes / self.decay_window)

        # Calculate decayed probability
        decayed_probability = state.decay_start_probability * (1.0 - decay_factor)

        return max(MIN_PROBABILITY, min(decayed_probability, MAX_PROBABILITY))

    def update_decay_state(
        self,
        state: ProbabilityState,
        current_time: Optional[datetime] = None,
    ) -> ProbabilityState:
        """Update state with decay calculations.

        This method updates the probability state with decay calculations,
        including decay status and whether decay is active.

        Args:
            state: Current probability state
            current_time: Current time (defaults to UTC now)

        Returns:
            Updated probability state

        """
        if current_time is None:
            current_time = dt_util.utcnow()

        # Check if decay should start
        should_decay = self.should_start_decay(state, current_time)

        if should_decay:
            # Calculate decayed probability
            decayed_probability = self.calculate_decay(state, current_time)

            # Calculate decay status (0-100)
            if state.decay_start_probability:
                decay_status = (
                    (state.decay_start_probability - decayed_probability)
                    / state.decay_start_probability
                    * 100
                )
            else:
                decay_status = 0.0

            # Update state
            return ProbabilityState(
                probability=decayed_probability,
                previous_probability=state.previous_probability,
                threshold=state.threshold,
                prior_probability=state.prior_probability,
                sensor_probabilities=state.sensor_probabilities,
                decay_status=decay_status,
                current_states=state.current_states,
                previous_states=state.previous_states,
                is_occupied=state.is_occupied,
                decaying=True,
                decay_start_time=state.decay_start_time,
                decay_start_probability=state.decay_start_probability,
            )
        else:
            # Reset decay state if not decaying
            return ProbabilityState(
                probability=state.probability,
                previous_probability=state.previous_probability,
                threshold=state.threshold,
                prior_probability=state.prior_probability,
                sensor_probabilities=state.sensor_probabilities,
                decay_status=0.0,
                current_states=state.current_states,
                previous_states=state.previous_states,
                is_occupied=state.is_occupied,
                decaying=False,
                decay_start_time=None,
                decay_start_probability=None,
            )

    def start_decay(
        self,
        state: ProbabilityState,
        current_time: Optional[datetime] = None,
    ) -> ProbabilityState:
        """Start probability decay.

        This method starts the probability decay process by setting the
        decay start time and probability.

        Args:
            state: Current probability state
            current_time: Current time (defaults to UTC now)

        Returns:
            Updated probability state

        """
        if current_time is None:
            current_time = dt_util.utcnow()

        return ProbabilityState(
            probability=state.probability,
            previous_probability=state.previous_probability,
            threshold=state.threshold,
            prior_probability=state.prior_probability,
            sensor_probabilities=state.sensor_probabilities,
            decay_status=0.0,
            current_states=state.current_states,
            previous_states=state.previous_states,
            is_occupied=state.is_occupied,
            decaying=True,
            decay_start_time=current_time,
            decay_start_probability=state.probability,
        )
