"""State validation logic."""

from typing import Any, Dict

from .containers import PriorState, ProbabilityState


class StateValidator:
    """Centralized state validation."""

    @staticmethod
    def validate_probability(value: float, name: str) -> None:
        """Validate that a probability value is between 0 and 1."""
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    @staticmethod
    def validate_prior_state(state: PriorState) -> None:
        """Validate all probability fields in a prior state."""
        for field in [
            "overall_prior",
            "motion_prior",
            "media_prior",
            "appliance_prior",
            "door_prior",
            "window_prior",
            "light_prior",
            "environmental_prior",
            "wasp_in_box_prior",
        ]:
            StateValidator.validate_probability(getattr(state, field), field)

    @staticmethod
    def validate_probability_state(state: ProbabilityState) -> None:
        """Validate all probability fields in a probability state."""
        for field in [
            "probability",
            "previous_probability",
            "threshold",
            "prior_probability",
            "decay_status",
        ]:
            StateValidator.validate_probability(getattr(state, field), field)

    @staticmethod
    def validate_sensor_probabilities(probabilities: Dict[str, Any]) -> None:
        """Validate sensor probabilities dictionary."""
        for sensor_id, prob in probabilities.items():
            if not isinstance(sensor_id, str):
                raise TypeError("Sensor ID must be a string")
            if not isinstance(prob, (int, float)):
                raise TypeError(
                    f"Probability value for sensor {sensor_id} must be a number"
                )
            if not 0 <= prob <= 1:
                raise ValueError(
                    f"Probability value for sensor {sensor_id} must be between 0 and 1"
                )
