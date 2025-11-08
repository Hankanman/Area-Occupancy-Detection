"""Flask web application for area occupancy simulator."""

# ruff: noqa: E402, I001, PLW0603, PLW0602, BLE001

from __future__ import annotations

from collections.abc import Callable
import logging
import os
from pathlib import Path
import sys
from typing import Any

# Ensure project root is on sys.path so local custom_components can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS  # type: ignore[import]
from homeassistant.const import STATE_OFF, STATE_ON
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

_LOGGER = logging.getLogger(__name__)

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def get_allowed_origins() -> list[str] | str:
    """Return allowed origins for CORS configuration."""

    env_value = os.getenv("SIMULATOR_ALLOWED_ORIGINS")
    if env_value is None:
        # Default origins cover published docs site and local development
        return [
            "https://hankanman.github.io",
            "https://hankanman.github.io/Area-Occupancy-Detection",
            "https://hankanman.github.io/Area-Occupancy-Detection/",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

    origins = [origin.strip() for origin in env_value.split(",") if origin.strip()]
    if not origins:
        return "*"

    if "*" in origins:
        return "*"

    return origins


CORS(
    app,
    resources={r"/api/*": {"origins": get_allowed_origins()}},
    supports_credentials=False,
)

# Global state for the simulator
simulator_state: dict[str, Any] | None = None

# Global state store for entity states (used by state_provider)
entity_state_store: dict[str, str | float] = {}


def _json_success(payload: dict[str, Any], status: int = 200):
    """Return a successful JSON response."""

    return jsonify(payload), status


def _json_error(message: str, status: int = 400):
    """Return a standardized JSON error response."""

    return jsonify({"error": message}), status


def _require_simulator_state() -> dict[str, Any]:
    """Ensure the simulator has been initialized before proceeding."""

    if simulator_state is None:
        raise RuntimeError("No simulation loaded")

    return simulator_state


def _require_entity(state: dict[str, Any], entity_id: str) -> Entity:
    """Return an entity from the simulator state or raise a KeyError."""

    try:
        return state["entities"][entity_id]
    except KeyError as exc:
        raise KeyError(entity_id) from exc


def _simulation_payload(
    *,
    probability: float | None = None,
    breakdown: dict[str, float] | None = None,
    include_decay: bool = False,
) -> dict[str, Any]:
    """Build a simulation payload, recalculating if necessary."""

    state = _require_simulator_state()

    if probability is None or breakdown is None:
        probability, breakdown = calculate_probability_breakdown(
            state["entities"], state["global_prior"], state["time_prior"]
        )

    return _build_simulation_payload(
        probability, breakdown, include_decay=include_decay
    )


def _simulation_response(
    *,
    probability: float | None = None,
    breakdown: dict[str, float] | None = None,
    include_decay: bool = False,
):
    """Return a standardized simulation response."""

    payload = _simulation_payload(
        probability=probability, breakdown=breakdown, include_decay=include_decay
    )
    return _json_success(payload)


def _friendly_name(entity_id: str) -> str:
    """Create a human-friendly name for an entity."""

    object_id = entity_id.split(".")[-1]
    return object_id.replace("_", " ").title()


def _format_state_display(entity: Entity) -> str:
    """Format the entity state for UI display."""

    state = entity.state
    if state is None:
        return "Unavailable"

    if entity.type.active_range is not None:
        try:
            return f"{float(state):.2f}"
        except (TypeError, ValueError):
            return str(state)

    return str(state)


def _build_entity_actions(entity: Entity) -> list[dict[str, Any]]:
    """Create UI action descriptors for an entity."""

    if entity.type.active_range is not None:
        # Numeric entities are adjusted via sliders/inputs in the UI
        return []

    is_active = entity.evidence is True
    is_inactive = entity.evidence is False

    return [
        {
            "label": "Active",
            "state": "on",
            "active": is_active,
        },
        {
            "label": "Inactive",
            "state": "off",
            "active": is_inactive,
        },
    ]


def _serialize_entities(entities: dict[str, Entity]) -> list[dict[str, Any]]:
    """Serialize Entity objects for the simulator UI."""

    serialized = []
    for entity_id, entity in entities.items():
        state_display = _format_state_display(entity)
        details_parts = [
            f"Type: {entity.type.input_type.value}",
            f"Weight: {entity.weight:.2f}",
            f"P(active|occupied): {entity.prob_given_true:.2f}",
            f"P(active|vacant): {entity.prob_given_false:.2f}",
        ]

        if entity.type.active_states:
            active_states = ", ".join(entity.type.active_states)
            details_parts.append(f"Active states: {active_states}")
        if entity.type.active_range:
            min_val, max_val = entity.type.active_range
            details_parts.append(f"Active range: {min_val} – {max_val}")

        serialized.append(
            {
                "entity_id": entity_id,
                "name": entity.name or _friendly_name(entity_id),
                "type": entity.type.input_type.value,
                "weight": entity.weight,
                "state": entity.state,
                "state_display": state_display,
                "details": " • ".join(details_parts),
                "is_numeric": entity.type.active_range is not None,
                "actions": _build_entity_actions(entity),
                "evidence": entity.evidence,
            }
        )

    return serialized


def _build_breakdown_list(
    breakdown: dict[str, float], entities: dict[str, Entity]
) -> list[dict[str, Any]]:
    """Convert breakdown dict into sorted list for the UI."""

    breakdown_list = []
    for entity_id, contribution in breakdown.items():
        entity = entities.get(entity_id)
        if entity is None:
            continue

        state_display = _format_state_display(entity)
        evidence = entity.evidence
        if evidence is True:
            likelihood = entity.prob_given_true
        elif evidence is False:
            likelihood = entity.prob_given_false
        else:
            likelihood = entity.prob_given_true

        breakdown_list.append(
            {
                "entity_id": entity_id,
                "name": entity.name or _friendly_name(entity_id),
                "description": f"State: {state_display}",
                "likelihood": likelihood,
                "contribution": contribution,
            }
        )

    breakdown_list.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return breakdown_list


def _current_weights(state: dict[str, Any]) -> dict[str, float]:
    """Return a mapping of entity type -> weight for UI consumption."""

    weights = {**state.get("entity_type_weights", {})}
    for entity in state["entities"].values():
        weights.setdefault(entity.type.input_type.value, entity.weight)
    return weights


def _calculate_priors(state: dict[str, Any]) -> tuple[float, float, float]:
    """Return combined, adjusted, and final priors."""

    combined_prior = combine_priors(state["global_prior"], state["time_prior"])
    adjusted_prior = combined_prior * PRIOR_FACTOR
    final_prior = max(MIN_PRIOR, min(MAX_PRIOR, adjusted_prior))
    return combined_prior, adjusted_prior, final_prior


def _build_simulation_payload(
    probability: float,
    breakdown: dict[str, float],
    *,
    include_decay: bool = False,
) -> dict[str, Any]:
    """Construct the payload returned to the frontend."""

    if simulator_state is None:
        raise ValueError("Simulator has not been initialized")

    combined_prior, adjusted_prior, final_prior = _calculate_priors(simulator_state)

    simulator_state["current_prior"] = adjusted_prior

    entities = simulator_state["entities"]
    payload: dict[str, Any] = {
        "area_name": simulator_state["area_name"],
        "area_purpose": simulator_state["area_purpose"],
        "probability": probability,
        "breakdown": _build_breakdown_list(breakdown, entities),
        "entities": _serialize_entities(entities),
        "weights": _current_weights(simulator_state),
        "global_prior": simulator_state["global_prior"],
        "time_prior": simulator_state["time_prior"],
        "combined_prior": combined_prior,
        "current_prior": adjusted_prior,
        "final_prior": final_prior,
        "half_life": simulator_state["half_life"],
    }

    if include_decay:
        payload["entity_decay"] = {
            entity_id: {
                "is_decaying": entity.decay.is_decaying,
                "decay_factor": entity.decay.decay_factor,
                "evidence": entity.evidence,
            }
            for entity_id, entity in entities.items()
        }

    return payload


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


def create_simulator_config():
    """Create a minimal config object with correct sensor states for the simulator.

    Returns:
        Object with sensor_states attribute matching real integration defaults
    """

    # Create a minimal config-like object with sensor_states
    # This ensures EntityType.create uses the same active states as the real integration
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
    """Create EntityType from string input type.

    Args:
        input_type_str: String representation of input type
        config: Optional config object with sensor_states (defaults to simulator config)

    Returns:
        EntityType object
    """
    try:
        input_type = InputType(input_type_str)
    except ValueError:
        input_type = InputType.UNKNOWN

    # Use simulator config if no config provided
    if config is None:
        config = create_simulator_config()

    return EntityType.create(input_type, config)


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

        # Apply weight from YAML if provided (matching create_from_db behavior)
        weight = likelihood_data.get("weight")
        if weight is not None:
            try:
                weight_float = float(weight)
                # Clamp weight to valid range (MIN_WEIGHT to MAX_WEIGHT)
                if MIN_WEIGHT <= weight_float <= MAX_WEIGHT:
                    entity_type.weight = weight_float
            except (TypeError, ValueError):
                # Weight is invalid, keep the default from EntityType.create
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
        area_prior: Area prior probability (global_prior from YAML)
        time_prior: Time prior probability

    Returns:
        Tuple of (overall_probability, breakdown_dict)
    """
    _LOGGER.debug("=== SIMULATOR PRIOR CALCULATION TRACE START ===")
    _LOGGER.debug("Phase 1.1: Extract Input Values")
    _LOGGER.debug("  global_prior = %.10f", area_prior)
    _LOGGER.debug("  time_prior = %.10f", time_prior)
    _LOGGER.debug("  PRIOR_FACTOR = %.10f", PRIOR_FACTOR)
    _LOGGER.debug("  MIN_PRIOR = %.10f, MAX_PRIOR = %.10f", MIN_PRIOR, MAX_PRIOR)

    # Combine priors and apply PRIOR_FACTOR to match real integration behavior
    _LOGGER.debug("Phase 1.2: Combine Priors")
    combined_prior = combine_priors(area_prior, time_prior)
    _LOGGER.debug(
        "  combined_prior = combine_priors(%.10f, %.10f) = %.10f",
        area_prior,
        time_prior,
        combined_prior,
    )

    _LOGGER.debug("Phase 1.3: Apply PRIOR_FACTOR")
    prior_before_factor = combined_prior
    prior = combined_prior * PRIOR_FACTOR
    _LOGGER.debug(
        "  prior = %.10f * %.10f = %.10f", prior_before_factor, PRIOR_FACTOR, prior
    )

    _LOGGER.debug("Phase 1.4: Clamp Prior")
    prior_before_clamp = prior
    prior = max(MIN_PRIOR, min(MAX_PRIOR, prior))
    if prior != prior_before_clamp:
        _LOGGER.debug("  prior clamped: %.10f -> %.10f", prior_before_clamp, prior)
    else:
        _LOGGER.debug("  prior (no clamping needed): %.10f", prior)

    _LOGGER.debug("Phase 1.5: Final Prior Value = %.10f", prior)
    _LOGGER.debug("=== SIMULATOR PRIOR CALCULATION TRACE END ===")

    # Calculate probability with all sensors
    overall_prob = bayesian_probability(entities, prior)

    # Calculate contribution of each sensor
    breakdown = {}
    for entity_id in entities:
        # Calculate probability without this sensor
        entities_without = {k: v for k, v in entities.items() if k != entity_id}
        prob_without = bayesian_probability(entities_without, prior)
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
            return _json_error("No YAML data provided", 400)

        data = parse_yaml_input(yaml_text)

        # Extract required fields
        area_name = data.get("area_name", "Unknown Area")
        current_prior = data.get("current_prior", 0.5)  # Keep for display
        global_prior = data.get("global_prior", 0.5)  # Use this for calculation
        time_prior = data.get("time_prior", 0.5)
        area_purpose = data.get("area_purpose", "social")
        entity_states = data.get("entity_states", {})
        likelihoods = data.get("likelihoods", {})

        # Get half-life from purpose
        half_life = get_half_life_from_purpose(area_purpose)

        # Create entities with half-life
        entities = create_simulator_entities(entity_states, likelihoods, half_life)

        # Calculate initial probability using global_prior (not current_prior)
        # current_prior is already combined and adjusted, we need the raw global_prior
        overall_prob, breakdown = calculate_probability_breakdown(
            entities, global_prior, time_prior
        )

        # Initialize weights per entity type from YAML or use defaults
        # Extract weights from entities (they may have been set from YAML)
        entity_type_weights = {}
        for entity in entities.values():
            input_type = entity.type.input_type.value
            if input_type not in entity_type_weights:
                entity_type_weights[input_type] = entity.weight

        # Calculate final prior (after PRIOR_FACTOR and clamping) for display
        combined_prior = combine_priors(global_prior, time_prior)
        final_prior = combined_prior * PRIOR_FACTOR
        final_prior = max(MIN_PRIOR, min(MAX_PRIOR, final_prior))

        # Store simulator state
        simulator_state = {
            "area_name": area_name,
            "current_prior": current_prior,
            "global_prior": global_prior,
            "time_prior": time_prior,
            "area_purpose": area_purpose,
            "half_life": half_life,
            "entities": entities,
            "entity_type_weights": entity_type_weights,
        }

        return _simulation_response(probability=overall_prob, breakdown=breakdown)

    except yaml.YAMLError as exc:
        return _json_error(f"Invalid YAML: {exc!s}", 400)
    except Exception as exc:
        return _json_error(f"Error loading data: {exc!s}", 500)


@app.route("/api/toggle", methods=["POST"])
def toggle_sensor():
    """Toggle binary sensor state and recalculate probability.

    Returns:
        JSON response with updated probability
    """
    global simulator_state

    try:
        state = _require_simulator_state()
    except RuntimeError as exc:
        return _json_error(str(exc), 400)

    payload = request.get_json(silent=True) or {}
    entity_id = payload.get("entity_id")
    new_state = payload.get("state")

    if entity_id is None:
        return _json_error("Entity ID is required", 400)

    try:
        entity = _require_entity(state, entity_id)
    except KeyError:
        return _json_error(f"Entity {entity_id} not found", 404)

    try:
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

        return _simulation_response()
    except Exception as exc:
        return _json_error(f"Error toggling sensor: {exc!s}", 500)


@app.route("/api/update", methods=["POST"])
def update_sensor():
    """Update numeric sensor value and recalculate probability.

    Returns:
        JSON response with updated probability
    """
    global simulator_state

    try:
        state = _require_simulator_state()
    except RuntimeError as exc:
        return _json_error(str(exc), 400)

    payload = request.get_json(silent=True) or {}
    entity_id = payload.get("entity_id")
    new_value = payload.get("value")

    if entity_id is None:
        return _json_error("Entity ID is required", 400)

    try:
        _require_entity(state, entity_id)
    except KeyError:
        return _json_error(f"Entity {entity_id} not found", 404)

    try:
        entity_state_store[entity_id] = float(new_value)
    except (ValueError, TypeError):
        return _json_error(f"Invalid numeric value: {new_value}", 400)

    return _simulation_response()


@app.route("/api/probability", methods=["GET"])
def get_probability():
    """Get current probability breakdown.

    Returns:
        JSON response with current probability and breakdown
    """
    try:
        _require_simulator_state()
    except RuntimeError as exc:
        return _json_error(str(exc), 400)

    try:
        return _simulation_response()
    except Exception as exc:
        return _json_error(f"Error calculating probability: {exc!s}", 500)


@app.route("/api/update-priors", methods=["POST"])
def update_priors():
    """Update global and time priors and recalculate probability.

    Returns:
        JSON response with updated probability
    """
    try:
        state = _require_simulator_state()
    except RuntimeError as exc:
        return _json_error(str(exc), 400)

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Invalid request payload", 400)

    updated = False

    if "global_prior" in payload:
        try:
            new_global = float(payload["global_prior"])
        except (TypeError, ValueError) as exc:
            return _json_error(f"Invalid global_prior: {exc}", 400)

        if not 0.0 <= new_global <= 1.0:
            return _json_error("global_prior must be between 0.0 and 1.0", 400)
        state["global_prior"] = new_global
        updated = True

    if "time_prior" in payload:
        try:
            new_time = float(payload["time_prior"])
        except (TypeError, ValueError) as exc:
            return _json_error(f"Invalid time_prior: {exc}", 400)

        if not 0.0 <= new_time <= 1.0:
            return _json_error("time_prior must be between 0.0 and 1.0", 400)
        state["time_prior"] = new_time
        updated = True

    if not updated:
        return _json_error("At least one prior value is required", 400)

    try:
        return _simulation_response()
    except Exception as exc:
        return _json_error(f"Error updating priors: {exc!s}", 500)


@app.route("/api/tick", methods=["POST"])
def tick():
    """Update decay states and recalculate probability (called every second).

    Returns:
        JSON response with updated probability and breakdown
    """
    try:
        state = _require_simulator_state()
    except RuntimeError as exc:
        return _json_error(str(exc), 400)

    try:
        # Update decay states (detect transitions)
        update_decay_states(state["entities"])

        return _simulation_response(include_decay=True)
    except Exception as exc:
        return _json_error(f"Error in tick: {exc!s}", 500)


@app.route("/api/update-purpose", methods=["POST"])
def update_purpose():
    """Update area purpose and half-life for all entities.

    Returns:
        JSON response with updated probability and breakdown
    """
    try:
        state = _require_simulator_state()
    except RuntimeError as exc:
        return _json_error(str(exc), 400)

    payload = request.get_json(silent=True) or {}
    new_purpose = payload.get("purpose")
    if new_purpose is None:
        return _json_error("Purpose is required", 400)

    try:
        # Get new half-life
        new_half_life = get_half_life_from_purpose(new_purpose)

        # Update all entities' decay half-life
        for entity in state["entities"].values():
            entity.decay.half_life = new_half_life

        # Update simulator state
        state["area_purpose"] = new_purpose
        state["half_life"] = new_half_life

        return _simulation_response()
    except Exception as exc:
        return _json_error(f"Error updating purpose: {exc!s}", 500)


@app.route("/api/update-weights", methods=["POST"])
def update_weights():
    """Update weights for entity types and recalculate probability.

    Returns:
        JSON response with updated probability
    """
    try:
        state = _require_simulator_state()
    except RuntimeError as exc:
        return _json_error(str(exc), 400)

    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return _json_error("Invalid request payload", 400)

    weight_type = payload.get("weight_type")
    weight_value = payload.get("value")

    if weight_type is None or weight_value is None:
        return _json_error("weight_type and value are required", 400)

    try:
        weight_float = float(weight_value)
    except (TypeError, ValueError) as exc:
        return _json_error(f"Invalid weight value: {exc}", 400)

    weight_float = max(MIN_WEIGHT, min(MAX_WEIGHT, weight_float))

    updated = False
    for entity in state["entities"].values():
        if entity.type.input_type.value == weight_type:
            entity.type.weight = weight_float
            updated = True

    if not updated:
        return _json_error(f"No entities found for type {weight_type}", 404)

    state.setdefault("entity_type_weights", {})[weight_type] = weight_float

    try:
        return _simulation_response()
    except Exception as exc:
        return _json_error(f"Error updating weights: {exc!s}", 500)


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
                "label": purpose_def.name,
                "half_life": purpose_def.half_life,
            }
        )
    return _json_success({"purposes": purposes})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
