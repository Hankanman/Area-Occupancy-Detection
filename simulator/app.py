"""Flask web application for area occupancy simulator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
from typing import Any

# Add parent directory to path to import custom_components
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request
import yaml

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.data.purpose import (
    PURPOSE_DEFINITIONS,
    AreaPurpose,
)
from custom_components.area_occupancy.utils import bayesian_probability
from homeassistant.const import STATE_OFF, STATE_ON

# ruff: noqa: PLW0603, PLW0602, BLE001, INP001

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# Global state for the simulator
simulator_state: dict[str, any] | None = None

# Global state store for entity states (used by state_provider)
entity_state_store: dict[str, str | float] = {}


def create_state_provider() -> Callable[[str], Any]:
    """Create a state provider function for the simulator.

    Returns:
        Callable that returns state for a given entity_id
    """

    def get_state(entity_id: str) -> Any:
        """Get state for entity_id from state store."""
        state_value = entity_state_store.get(entity_id)
        if state_value is None:
            return None
        # Return simple object with .state attribute (mimics HA State object)
        return type("State", (), {"state": state_value})()

    return get_state


def get_half_life_from_purpose(purpose_str: str | None) -> float:
    """Get half-life in seconds from area purpose.

    Args:
        purpose_str: Purpose string (e.g., "social", "sleeping")

    Returns:
        Half-life in seconds
    """
    if purpose_str is None:
        return PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life

    try:
        purpose_enum = AreaPurpose(purpose_str)
        return PURPOSE_DEFINITIONS[purpose_enum].half_life
    except (ValueError, KeyError):
        # Fallback to social
        return PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life


def parse_yaml_input(yaml_text: str) -> dict:
    """Parse YAML input and extract simulation data.

    Args:
        yaml_text: YAML string from user input

    Returns:
        Dictionary with parsed data
    """
    return yaml.safe_load(yaml_text)


def create_entity_type(input_type_str: str) -> EntityType:
    """Create EntityType from string input type.

    Args:
        input_type_str: String representation of input type

    Returns:
        EntityType object
    """
    try:
        input_type = InputType(input_type_str)
    except ValueError:
        input_type = InputType.UNKNOWN

    return EntityType.create(input_type)


def create_simulator_entities(
    entity_states: dict[str, str],
    likelihoods: dict[str, dict[str, any]],
    half_life: float = 720.0,
) -> dict[str, Entity]:
    """Create Entity objects from parsed data using state provider.

    Args:
        entity_states: Dictionary mapping entity_id to state string
        likelihoods: Dictionary mapping entity_id to likelihood data
        half_life: Decay half-life in seconds (default: 720.0)

    Returns:
        Dictionary mapping entity_id to Entity
    """
    global entity_state_store

    # Clear and populate state store
    entity_state_store.clear()

    # Convert states to appropriate types and store them
    for entity_id, state_str in entity_states.items():
        likelihood_data = likelihoods.get(entity_id, {})
        input_type_str = likelihood_data.get("type", "unknown")
        entity_type = create_entity_type(input_type_str)

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
        entity_type = create_entity_type(input_type_str)

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


def update_decay_states(entities: dict[str, Entity]) -> None:
    """Update decay states for all entities based on evidence transitions.

    Args:
        entities: Dictionary of entities to update
    """
    for entity in entities.values():
        current_evidence = entity.evidence
        previous_evidence = entity.previous_evidence

        # Skip if current or previous evidence is None
        if current_evidence is None or previous_evidence is None:
            entity.previous_evidence = current_evidence
            continue

        # Detect transitions
        if current_evidence != previous_evidence:
            if current_evidence:
                # False → True: stop decay
                entity.decay.stop_decay()
            else:
                # True → False: start decay
                entity.decay.start_decay()

        # Update previous_evidence for next check
        entity.previous_evidence = current_evidence


def calculate_probability_breakdown(
    entities: dict[str, Entity],
    area_prior: float,
    time_prior: float,
) -> tuple[float, dict[str, float]]:
    """Calculate overall probability and per-sensor contributions.

    Args:
        entities: Dictionary of entities
        area_prior: Area prior probability
        time_prior: Time prior probability

    Returns:
        Tuple of (overall_probability, breakdown_dict)
    """
    # Calculate probability with all sensors
    overall_prob = bayesian_probability(entities, area_prior, time_prior)

    # Calculate contribution of each sensor
    breakdown = {}
    for entity_id in entities:
        # Calculate probability without this sensor
        entities_without = {k: v for k, v in entities.items() if k != entity_id}
        prob_without = bayesian_probability(entities_without, area_prior, time_prior)
        contribution = overall_prob - prob_without
        breakdown[entity_id] = contribution

    return overall_prob, breakdown


@app.route("/")
def index():
    """Serve the main HTML page."""
    return render_template("index.html")


@app.route("/api/load", methods=["POST"])
def load_data():
    """Parse YAML input and initialize simulation.

    Returns:
        JSON response with initialized simulation data
    """
    global simulator_state

    try:
        yaml_text = request.json.get("yaml", "")
        if not yaml_text:
            return jsonify({"error": "No YAML data provided"}), 400

        data = parse_yaml_input(yaml_text)

        # Extract required fields
        area_name = data.get("area_name", "Unknown Area")
        current_prior = data.get("current_prior", 0.5)
        global_prior = data.get("global_prior", 0.5)
        time_prior = data.get("time_prior", 0.5)
        area_purpose = data.get("area_purpose", "social")
        entity_states = data.get("entity_states", {})
        likelihoods = data.get("likelihoods", {})

        # Get half-life from purpose
        half_life = get_half_life_from_purpose(area_purpose)

        # Create entities with half-life
        entities = create_simulator_entities(entity_states, likelihoods, half_life)

        # Calculate initial probability
        overall_prob, breakdown = calculate_probability_breakdown(
            entities, current_prior, time_prior
        )

        # Store simulator state
        simulator_state = {
            "area_name": area_name,
            "current_prior": current_prior,
            "global_prior": global_prior,
            "time_prior": time_prior,
            "area_purpose": area_purpose,
            "half_life": half_life,
            "entities": entities,
        }

        # Prepare response
        response = {
            "area_name": area_name,
            "current_prior": current_prior,
            "global_prior": global_prior,
            "time_prior": time_prior,
            "area_purpose": area_purpose,
            "half_life": half_life,
            "probability": overall_prob,
            "breakdown": breakdown,
            "entities": [
                {
                    "entity_id": entity_id,
                    "type": entity.type.input_type.value,
                    "weight": entity.weight,
                    "prob_given_true": entity.prob_given_true,
                    "prob_given_false": entity.prob_given_false,
                    "current_state": str(entity.state)
                    if entity.state is not None
                    else None,
                    "evidence": entity.evidence,
                    "is_numeric": entity.type.active_range is not None,
                    "active_states": entity.type.active_states,
                    "active_range": entity.type.active_range,
                }
                for entity_id, entity in entities.items()
            ],
        }

        return jsonify(response)

    except yaml.YAMLError as e:
        return jsonify({"error": f"Invalid YAML: {e!s}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error loading data: {e!s}"}), 500


@app.route("/api/toggle", methods=["POST"])
def toggle_sensor():
    """Toggle binary sensor state and recalculate probability.

    Returns:
        JSON response with updated probability
    """
    global simulator_state

    if simulator_state is None:
        return jsonify({"error": "No simulation loaded"}), 400

    try:
        entity_id = request.json.get("entity_id")
        new_state = request.json.get("state")  # "on" or "off"

        if entity_id not in simulator_state["entities"]:
            return jsonify({"error": f"Entity {entity_id} not found"}), 404

        entity = simulator_state["entities"][entity_id]

        # Update state in state store
        if new_state == "on":
            # For binary sensors, set to appropriate active state
            if entity.type.active_states and len(entity.type.active_states) > 0:
                entity_state_store[entity_id] = entity.type.active_states[0]
            else:
                entity_state_store[entity_id] = "on"
        # Set to inactive state
        # Find a state that is NOT in active_states
        elif entity.type.active_states and len(entity.type.active_states) > 0:
            # Common inactive states to try
            inactive_states = [STATE_OFF, STATE_ON, "closed", "open"]
            for inactive in inactive_states:
                if inactive not in entity.type.active_states:
                    entity_state_store[entity_id] = inactive
                    break
            else:
                # Fallback: use "off" if no inactive state found
                entity_state_store[entity_id] = STATE_OFF
        else:
            entity_state_store[entity_id] = STATE_OFF

        # Recalculate probability
        overall_prob, breakdown = calculate_probability_breakdown(
            simulator_state["entities"],
            simulator_state["current_prior"],
            simulator_state["time_prior"],
        )

        return jsonify(
            {
                "probability": overall_prob,
                "breakdown": breakdown,
                "entity_state": str(entity.state) if entity.state is not None else None,
                "evidence": entity.evidence,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error toggling sensor: {e!s}"}), 500


@app.route("/api/update", methods=["POST"])
def update_sensor():
    """Update numeric sensor value and recalculate probability.

    Returns:
        JSON response with updated probability
    """
    global simulator_state

    if simulator_state is None:
        return jsonify({"error": "No simulation loaded"}), 400

    try:
        entity_id = request.json.get("entity_id")
        new_value = request.json.get("value")

        if entity_id not in simulator_state["entities"]:
            return jsonify({"error": f"Entity {entity_id} not found"}), 404

        entity = simulator_state["entities"][entity_id]

        # Update state in state store
        try:
            entity_state_store[entity_id] = float(new_value)
        except (ValueError, TypeError):
            return jsonify({"error": f"Invalid numeric value: {new_value}"}), 400

        # Recalculate probability
        overall_prob, breakdown = calculate_probability_breakdown(
            simulator_state["entities"],
            simulator_state["current_prior"],
            simulator_state["time_prior"],
        )

        return jsonify(
            {
                "probability": overall_prob,
                "breakdown": breakdown,
                "entity_state": str(entity.state) if entity.state is not None else None,
                "evidence": entity.evidence,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error updating sensor: {e!s}"}), 500


@app.route("/api/probability", methods=["GET"])
def get_probability():
    """Get current probability breakdown.

    Returns:
        JSON response with current probability and breakdown
    """
    global simulator_state

    if simulator_state is None:
        return jsonify({"error": "No simulation loaded"}), 400

    try:
        overall_prob, breakdown = calculate_probability_breakdown(
            simulator_state["entities"],
            simulator_state["current_prior"],
            simulator_state["time_prior"],
        )

        return jsonify({"probability": overall_prob, "breakdown": breakdown})

    except Exception as e:
        return jsonify({"error": f"Error calculating probability: {e!s}"}), 500


@app.route("/api/update-priors", methods=["POST"])
def update_priors():
    """Update global and time priors and recalculate probability.

    Returns:
        JSON response with updated probability
    """
    global simulator_state

    if simulator_state is None:
        return jsonify({"error": "No simulation loaded"}), 400

    try:
        global_prior = request.json.get("global_prior")
        time_prior = request.json.get("time_prior")

        if global_prior is None or time_prior is None:
            return jsonify(
                {"error": "Both global_prior and time_prior are required"}
            ), 400

        # Validate ranges
        if not 0.0 <= global_prior <= 1.0 or not 0.0 <= time_prior <= 1.0:
            return jsonify({"error": "Priors must be between 0.0 and 1.0"}), 400

        # Update simulator state
        simulator_state["global_prior"] = float(global_prior)
        simulator_state["time_prior"] = float(time_prior)
        # Use global_prior as current_prior for calculation
        simulator_state["current_prior"] = float(global_prior)

        # Recalculate probability
        overall_prob, breakdown = calculate_probability_breakdown(
            simulator_state["entities"],
            simulator_state["current_prior"],
            simulator_state["time_prior"],
        )

        return jsonify(
            {
                "probability": overall_prob,
                "breakdown": breakdown,
                "global_prior": simulator_state["global_prior"],
                "time_prior": simulator_state["time_prior"],
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error updating priors: {e!s}"}), 500


@app.route("/api/tick", methods=["POST"])
def tick():
    """Update decay states and recalculate probability (called every second).

    Returns:
        JSON response with updated probability and breakdown
    """
    global simulator_state

    if simulator_state is None:
        return jsonify({"error": "No simulation loaded"}), 400

    try:
        # Update decay states (detect transitions)
        update_decay_states(simulator_state["entities"])

        # Recalculate probability
        overall_prob, breakdown = calculate_probability_breakdown(
            simulator_state["entities"],
            simulator_state["current_prior"],
            simulator_state["time_prior"],
        )

        # Get decay information for each entity
        entity_decay_info = {}
        for entity_id, entity in simulator_state["entities"].items():
            entity_decay_info[entity_id] = {
                "is_decaying": entity.decay.is_decaying,
                "decay_factor": entity.decay.decay_factor,
                "evidence": entity.evidence,
            }

        return jsonify(
            {
                "probability": overall_prob,
                "breakdown": breakdown,
                "entity_decay": entity_decay_info,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error in tick: {e!s}"}), 500


@app.route("/api/update-purpose", methods=["POST"])
def update_purpose():
    """Update area purpose and half-life for all entities.

    Returns:
        JSON response with updated probability and breakdown
    """
    global simulator_state

    if simulator_state is None:
        return jsonify({"error": "No simulation loaded"}), 400

    try:
        new_purpose = request.json.get("purpose")
        if new_purpose is None:
            return jsonify({"error": "Purpose is required"}), 400

        # Get new half-life
        new_half_life = get_half_life_from_purpose(new_purpose)

        # Update all entities' decay half-life
        for entity in simulator_state["entities"].values():
            entity.decay.half_life = new_half_life

        # Update simulator state
        simulator_state["area_purpose"] = new_purpose
        simulator_state["half_life"] = new_half_life

        # Recalculate probability
        overall_prob, breakdown = calculate_probability_breakdown(
            simulator_state["entities"],
            simulator_state["current_prior"],
            simulator_state["time_prior"],
        )

        return jsonify(
            {
                "probability": overall_prob,
                "breakdown": breakdown,
                "area_purpose": new_purpose,
                "half_life": new_half_life,
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error updating purpose: {e!s}"}), 500


@app.route("/api/get-purposes", methods=["GET"])
def get_purposes():
    """Get list of available purposes.

    Returns:
        JSON response with purpose options
    """
    purposes = []
    for purpose_enum, purpose_def in PURPOSE_DEFINITIONS.items():
        purposes.append(
            {
                "value": purpose_enum.value,
                "name": purpose_def.name,
                "half_life": purpose_def.half_life,
            }
        )
    return jsonify({"purposes": purposes})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
