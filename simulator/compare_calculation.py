"""Script to compare probability calculations between simulator and real integration.

This script loads YAML data, runs both the simulator and real integration calculations,
and compares the step-by-step results to identify where differences occur.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys

# Add parent directory to path to import custom_components
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from custom_components.area_occupancy.const import (
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_WINDOW_ACTIVE_STATE,
    MAX_PRIOR,
    MAX_WEIGHT,
    MIN_PRIOR,
    MIN_WEIGHT,
)
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.data.prior import PRIOR_FACTOR
from custom_components.area_occupancy.data.purpose import (
    PURPOSE_DEFINITIONS,
    AreaPurpose,
)
from custom_components.area_occupancy.utils import bayesian_probability, combine_priors
from homeassistant.const import STATE_ON

# Configure logging to show debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s - %(name)s - %(message)s",
)

_LOGGER = logging.getLogger(__name__)

# Global state store for entity states (used by state_provider)
entity_state_store: dict[str, str | float] = {}


def create_state_provider():
    """Create a state provider function for the simulator."""

    def get_state(entity_id: str):
        """Get state for entity_id from state store."""
        state_value = entity_state_store.get(entity_id)
        if state_value is None:
            return None
        # Return simple object with .state attribute (mimics HA State object)
        return type("State", (), {"state": state_value})()

    return get_state


def get_half_life_from_purpose(purpose_str: str | None) -> float:
    """Get half-life in seconds from area purpose."""
    if purpose_str is None:
        return PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life

    try:
        purpose_enum = AreaPurpose(purpose_str)
        return PURPOSE_DEFINITIONS[purpose_enum].half_life
    except (ValueError, KeyError):
        return PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life


def create_simulator_config():
    """Create a minimal config object with correct sensor states for the simulator."""

    class SimulatorConfig:
        def __init__(self):
            self.sensor_states = type(
                "SensorStates",
                (),
                {
                    "motion": [STATE_ON],
                    "door": [DEFAULT_DOOR_ACTIVE_STATE],  # STATE_CLOSED
                    "window": [DEFAULT_WINDOW_ACTIVE_STATE],  # STATE_OPEN
                    "appliance": [STATE_ON, "standby"],
                    "media": ["playing", "paused"],
                },
            )()

    return SimulatorConfig()


def create_entity_type(input_type_str: str, config=None) -> EntityType:
    """Create EntityType from string input type."""
    try:
        input_type = InputType(input_type_str)
    except ValueError:
        input_type = InputType.UNKNOWN

    if config is None:
        config = create_simulator_config()

    return EntityType.create(input_type, config)


def create_simulator_entities(
    entity_states: dict[str, str],
    likelihoods: dict[str, dict[str, any]],
    half_life: float = 720.0,
) -> dict[str, Entity]:
    """Create Entity objects from parsed data using state provider."""
    # Clear and populate state store
    entity_state_store.clear()

    # Create simulator config once for all entities
    simulator_config = create_simulator_config()

    # Convert states to appropriate types and store them
    for entity_id, state_str in entity_states.items():
        likelihood_data = likelihoods.get(entity_id, {})
        input_type_str = likelihood_data.get("type", "unknown")
        entity_type = create_entity_type(input_type_str, simulator_config)

        # Convert state to appropriate type
        if entity_type.active_range is not None:
            # Numeric sensor
            try:
                entity_state_store[entity_id] = float(state_str)
            except (ValueError, TypeError):
                entity_state_store[entity_id] = None
        else:
            # Binary sensor
            entity_state_store[entity_id] = state_str

    # Create state provider
    state_provider = create_state_provider()

    entities = {}
    for entity_id, likelihood_data in likelihoods.items():
        input_type_str = likelihood_data.get("type", "unknown")
        entity_type = create_entity_type(input_type_str, simulator_config)

        # Apply weight from YAML if provided
        weight = likelihood_data.get("weight")
        if weight is not None:
            try:
                weight_float = float(weight)
                if MIN_WEIGHT <= weight_float <= MAX_WEIGHT:
                    entity_type.weight = weight_float
            except (TypeError, ValueError):
                pass

        # Use provided values or fall back to defaults
        prob_given_true = likelihood_data.get(
            "prob_given_true", entity_type.prob_given_true
        )
        prob_given_false = likelihood_data.get(
            "prob_given_false", entity_type.prob_given_false
        )

        decay = Decay(half_life=half_life)
        entity = Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            decay=decay,
            state_provider=state_provider,
        )
        # Initialize previous_evidence
        entity.previous_evidence = entity.evidence

        entities[entity_id] = entity

    return entities


def calculate_simulator_probability(
    entities: dict[str, Entity],
    area_prior: float,
    time_prior: float,
) -> float:
    """Calculate probability using simulator logic."""
    _LOGGER.info("=== SIMULATOR CALCULATION ===")

    # Combine priors and apply PRIOR_FACTOR
    combined_prior = combine_priors(area_prior, time_prior)
    prior = combined_prior * PRIOR_FACTOR
    prior = max(MIN_PRIOR, min(MAX_PRIOR, prior))

    # Calculate probability
    overall_prob = bayesian_probability(entities, prior)

    _LOGGER.info("Simulator result: %.10f (%.2f%%)", overall_prob, overall_prob * 100)
    return overall_prob


def main():
    """Main comparison function."""
    # YAML input from user
    yaml_text = """
area_name: Study
current_prior: 0.17068439531076843
global_prior: 0.28897248975495615
time_prior: 0.01
area_purpose: working
prior_entity_ids:
  - binary_sensor.study_motion_1_occupancy
total_entities: 8
entity_states:
  media_player.echojumper: unavailable
  sensor.study_illuminance: "3.0"
  binary_sensor.study_motion_1_occupancy: "on"
  sensor.study_temperature: "22.28"
  binary_sensor.study_window_left_contact: "on"
  binary_sensor.study_window_right_contact: "on"
  binary_sensor.study_door_contact: "on"
  sensor.study_humidity: "67.23"
likelihoods:
  binary_sensor.study_motion_1_occupancy:
    type: motion
    weight: 0.85
    prob_given_true: 0.3347714595509412
    prob_given_false: 0.01
  media_player.echojumper:
    type: media
    weight: 0.3
    prob_given_true: 0.532496125321511
    prob_given_false: 0.16648228285205913
  binary_sensor.study_door_contact:
    type: door
    weight: 0.3
    prob_given_true: 0.2
    prob_given_false: 0.02
  binary_sensor.study_window_right_contact:
    type: window
    weight: 0.1
    prob_given_true: 0.2
    prob_given_false: 0.02
  binary_sensor.study_window_left_contact:
    type: window
    weight: 0.1
    prob_given_true: 0.2
    prob_given_false: 0.02
  sensor.study_illuminance:
    type: illuminance
    weight: 0.1
    prob_given_true: 0.38197670712177
    prob_given_false: 0.0996518341204216
  sensor.study_humidity:
    type: humidity
    weight: 0.1
    prob_given_true: 0.023250551567983942
    prob_given_false: 0.04606638314676945
  sensor.study_temperature:
    type: temperature
    weight: 0.1
    prob_given_true: 0.5451733074826627
    prob_given_false: 0.742173492423621
update_timestamp: "2025-11-06T21:28:26.891114+00:00"

"""

    data = yaml.safe_load(yaml_text)

    # Extract required fields
    global_prior = data.get("global_prior", 0.5)
    time_prior = data.get("time_prior", 0.5)
    area_purpose = data.get("area_purpose", "social")
    entity_states = data.get("entity_states", {})
    likelihoods = data.get("likelihoods", {})

    # Get half-life from purpose
    half_life = get_half_life_from_purpose(area_purpose)

    # Create entities
    entities = create_simulator_entities(entity_states, likelihoods, half_life)

    # Calculate probability using simulator logic
    simulator_prob = calculate_simulator_probability(entities, global_prior, time_prior)

    _LOGGER.info("")
    _LOGGER.info("=== SUMMARY ===")
    _LOGGER.info(
        "Simulator probability: %.10f (%.2f%%)", simulator_prob, simulator_prob * 100
    )
    _LOGGER.info("")
    _LOGGER.info("Check the debug logs above to compare step-by-step calculations.")
    _LOGGER.info("Look for differences in:")
    _LOGGER.info("  1. Prior calculation (Phase 1)")
    _LOGGER.info("  2. Entity filtering (Phase 2.1)")
    _LOGGER.info("  3. Entity contributions (Phase 2.2, 3.2)")
    _LOGGER.info("  4. Final probability (Phase 3.3)")


if __name__ == "__main__":
    main()
