"""State serialization for Area Occupancy Detection."""

from typing import Any, Dict

from homeassistant.util import dt as dt_util

from .containers import PriorState, ProbabilityState


class StateSerializer:
    """Handles state serialization and deserialization."""

    @staticmethod
    def serialize_probability_state(state: ProbabilityState) -> Dict[str, Any]:
        """Serialize a probability state to a dictionary."""
        return {
            "probability": state.probability,
            "sensor_probabilities": {
                sensor_id: {
                    "probability": prob["probability"],
                    "weight": prob["weight"],
                    "weighted_probability": prob["weighted_probability"],
                }
                for sensor_id, prob in state.sensor_probabilities.items()
            },
            "current_states": {
                sensor_id: {
                    "state": info["state"],
                    "last_changed": info["last_changed"],
                    "availability": info["availability"],
                }
                for sensor_id, info in state.current_states.items()
            },
            "previous_states": {
                sensor_id: {
                    "state": info["state"],
                    "last_changed": info["last_changed"],
                    "availability": info["availability"],
                }
                for sensor_id, info in state.previous_states.items()
            },
            "last_updated": state.decay_start_time.isoformat()
            if state.decay_start_time
            else None,
            "is_occupied": state.is_occupied,
        }

    @staticmethod
    def deserialize_probability_state(data: Dict[str, Any]) -> ProbabilityState:
        """Deserialize a dictionary to a probability state."""
        return ProbabilityState(
            probability=data.get("probability", 0.0),
            sensor_probabilities={
                sensor_id: {
                    "probability": prob["probability"],
                    "weight": prob["weight"],
                    "weighted_probability": prob["weighted_probability"],
                }
                for sensor_id, prob in data.get("sensor_probabilities", {}).items()
            },
            current_states={
                sensor_id: {
                    "state": info["state"],
                    "last_changed": info["last_changed"],
                    "availability": info["availability"],
                }
                for sensor_id, info in data.get("current_states", {}).items()
            },
            previous_states={
                sensor_id: {
                    "state": info["state"],
                    "last_changed": info["last_changed"],
                    "availability": info["availability"],
                }
                for sensor_id, info in data.get("previous_states", {}).items()
            },
            decay_start_time=dt_util.parse_datetime(data["last_updated"])
            if data.get("last_updated")
            else None,
            is_occupied=data.get("is_occupied", False),
        )

    @staticmethod
    def serialize_prior_state(state: PriorState) -> Dict[str, Any]:
        """Serialize a prior state to a dictionary."""
        return {
            "overall_prior": state.overall_prior,
            "motion_prior": state.motion_prior,
            "media_prior": state.media_prior,
            "appliance_prior": state.appliance_prior,
            "door_prior": state.door_prior,
            "window_prior": state.window_prior,
            "light_prior": state.light_prior,
            "environmental_prior": state.environmental_prior,
            "wasp_in_box_prior": state.wasp_in_box_prior,
            "entity_priors": {
                entity_id: {
                    "prob_given_true": prior.prob_given_true,
                    "prob_given_false": prior.prob_given_false,
                    "last_updated": prior.last_updated,
                }
                for entity_id, prior in state.entity_priors.items()
            },
            "type_priors": {
                type_id: {
                    "prob_given_true": prior.prob_given_true,
                    "prob_given_false": prior.prob_given_false,
                    "last_updated": prior.last_updated,
                }
                for type_id, prior in state.type_priors.items()
            },
            "analysis_period": state.analysis_period,
        }

    @staticmethod
    def deserialize_prior_state(data: Dict[str, Any]) -> PriorState:
        """Deserialize a dictionary to a prior state."""
        return PriorState.from_dict(data)
