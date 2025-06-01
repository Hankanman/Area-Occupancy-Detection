"""State update handlers."""

from typing import Any, Dict

from .containers import PriorState, ProbabilityState
from .validation import StateValidator


class StateUpdater:
    """Handles state updates with validation."""

    def __init__(self, validator: StateValidator):
        """Initialize the state updater with a validator."""
        self._validator = validator

    def update_probability_state(
        self, state: ProbabilityState, updates: Dict[str, Any]
    ) -> ProbabilityState:
        """Update probability state with validation."""
        for key, value in updates.items():
            if hasattr(state, key):
                if key in [
                    "probability",
                    "previous_probability",
                    "threshold",
                    "prior_probability",
                    "decay_status",
                ]:
                    self._validator.validate_probability(value, key)
                setattr(state, key, value)
        return state

    def update_prior_state(
        self, state: PriorState, updates: Dict[str, Any]
    ) -> PriorState:
        """Update prior state with validation."""
        for key, value in updates.items():
            if hasattr(state, key):
                if key.endswith("_prior"):
                    self._validator.validate_probability(value, key)
                setattr(state, key, value)
        return state

    def update_sensor_probabilities(
        self, state: ProbabilityState, probabilities: Dict[str, Any]
    ) -> ProbabilityState:
        """Update sensor probabilities with validation."""
        self._validator.validate_sensor_probabilities(probabilities)
        state.sensor_probabilities.update(probabilities)
        return state
