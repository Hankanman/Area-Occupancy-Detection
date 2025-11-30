"""Flask web application for the Area Occupancy stateless simulator."""

# ruff: noqa: I001

from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import sys
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS  # type: ignore[import]
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util
import yaml

from custom_components.area_occupancy.const import (
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_WINDOW_ACTIVE_STATE,
    MAX_PRIOR,
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PRIOR,
    MIN_PROBABILITY,
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
app = Flask(__name__)


def get_allowed_origins() -> list[str] | str:
    """Return allowed origins for CORS configuration."""

    env_value = os.getenv("SIMULATOR_ALLOWED_ORIGINS")
    if env_value is None:
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


def _json_success(payload: dict[str, Any], status: int = 200):
    return jsonify(payload), status


def _json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _friendly_name(entity_id: str) -> str:
    object_id = entity_id.split(".")[-1]
    return object_id.replace("_", " ").title()


def _format_state_display(entity: Entity) -> str:
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
    if entity.type.active_range is not None:
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


def _serialize_entities(
    entities: dict[str, Entity],
    breakdown: dict[str, float] | None = None,
    extra_fields: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []

    for entity_id, entity in entities.items():
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

        if entity.evidence is True:
            likelihood = entity.prob_given_true
        elif entity.evidence is False:
            likelihood = entity.prob_given_false
        else:
            likelihood = entity.prob_given_true

        entity_dict = {
            "entity_id": entity_id,
            "name": entity.name or _friendly_name(entity_id),
            "type": entity.type.input_type.value,
            "weight": entity.weight,
            "state": entity.state,
            "state_display": _format_state_display(entity),
            "details": " • ".join(details_parts),
            "is_numeric": entity.type.active_range is not None,
            "actions": _build_entity_actions(entity),
            "evidence": entity.evidence,
            "likelihood": likelihood,
            "contribution": breakdown.get(entity_id) if breakdown else None,
        }

        # Merge extra fields if available (analysis_error, analysis_data, correlation_type)
        if extra_fields and entity_id in extra_fields:
            entity_dict.update(extra_fields[entity_id])

        serialized.append(entity_dict)

    return serialized


def _serialize_breakdown(
    entities: dict[str, Entity], breakdown: dict[str, float]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for entity_id, contribution in breakdown.items():
        entity = entities.get(entity_id)
        if entity is None:
            continue

        if entity.evidence is True:
            likelihood = entity.prob_given_true
        elif entity.evidence is False:
            likelihood = entity.prob_given_false
        else:
            likelihood = entity.prob_given_true

        result.append(
            {
                "entity_id": entity_id,
                "name": entity.name or _friendly_name(entity_id),
                "description": f"State: {_format_state_display(entity)}",
                "likelihood": likelihood,
                "contribution": contribution,
            }
        )

    result.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return result


def _serialize_decay_map(entities: dict[str, Entity]) -> dict[str, dict[str, Any]]:
    decay_map: dict[str, dict[str, Any]] = {}

    for entity_id, entity in entities.items():
        decay_start = entity.decay.decay_start
        if not isinstance(decay_start, datetime):
            decay_start = dt_util.utcnow()

        decay_map[entity_id] = {
            "is_decaying": entity.decay.is_decaying,
            "decay_factor": entity.decay.decay_factor,
            "decay_start": decay_start.isoformat(),
            "evidence": entity.evidence,
        }

    return decay_map


def _create_simulator_config():
    class SimulatorConfig:
        def __init__(self):
            self.sensor_states = type(
                "SensorStates",
                (),
                {
                    "motion": [STATE_ON],
                    "door": [DEFAULT_DOOR_ACTIVE_STATE],
                    "window": [DEFAULT_WINDOW_ACTIVE_STATE],
                    "appliance": [STATE_ON, "standby"],
                    "media": ["playing", "paused"],
                },
            )()

    return SimulatorConfig()


def _get_half_life_from_purpose(purpose_str: str | None) -> float:
    if purpose_str is None:
        return PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life

    try:
        purpose_enum = AreaPurpose(purpose_str)
        return PURPOSE_DEFINITIONS[purpose_enum].half_life
    except (ValueError, KeyError):
        return PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life


def _create_entity_type(input_type_str: str, config=None) -> EntityType:
    try:
        input_type = InputType(input_type_str)
    except ValueError:
        input_type = InputType.UNKNOWN

    if config is None:
        config = _create_simulator_config()

    # Extract overrides from config
    weight = None
    active_states = None
    active_range = None

    weights = getattr(config, "weights", None)
    if weights:
        weight_attr = getattr(weights, input_type.value, None)
        if weight_attr is not None:
            weight = weight_attr

    sensor_states = getattr(config, "sensor_states", None)
    if sensor_states:
        states_attr = getattr(sensor_states, input_type.value, None)
        if states_attr is not None:
            active_states = states_attr

    range_config_attr = f"{input_type.value}_active_range"
    range_attr = getattr(config, range_config_attr, None)
    if range_attr is not None:
        active_range = range_attr

    return EntityType(
        input_type,
        weight=weight,
        active_states=active_states,
        active_range=active_range,
    )


def _coerce_state(entity_type: EntityType, raw_state: Any) -> Any:
    if entity_type.active_range is not None:
        try:
            return float(raw_state)
        except (TypeError, ValueError):
            return None

    if raw_state is None:
        return None

    return str(raw_state)


def _build_state_provider(state_map: dict[str, Any]):
    def get_state(entity_id: str) -> Any:
        value = state_map.get(entity_id)
        if value is None:
            return None
        return type("State", (), {"state": value})()

    return get_state


def _parse_decay_payload(
    decay_payload: dict[str, Any] | None, *, default_half_life: float
) -> Decay:
    if not decay_payload:
        return Decay(half_life=default_half_life, is_decaying=False)

    is_decaying = bool(decay_payload.get("is_decaying", False))

    decay_start_raw = decay_payload.get("decay_start")
    decay_start = dt_util.parse_datetime(decay_start_raw) if decay_start_raw else None
    if decay_start is None:
        decay_start = dt_util.utcnow()

    return Decay(
        decay_start=decay_start,
        half_life=default_half_life,
        is_decaying=is_decaying,
    )


def _normalize_probability(value: float) -> float:
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, float(value)))


def _build_entity_inputs_from_entities(
    entities: dict[str, Entity],
    state_map: dict[str, Any],
    weights_map: dict[str, float],
) -> list[dict[str, Any]]:
    entity_inputs: list[dict[str, Any]] = []

    for entity_id, entity in entities.items():
        entity_type_value = entity.type.input_type.value
        weight_value = entity.type.weight
        weights_map.setdefault(entity_type_value, weight_value)

        decay_start = entity.decay.decay_start
        if not isinstance(decay_start, datetime):
            decay_start = dt_util.utcnow()

        entity_inputs.append(
            {
                "entity_id": entity_id,
                "type": entity_type_value,
                "state": state_map.get(entity_id),
                "prob_given_true": float(entity.prob_given_true),
                "prob_given_false": float(entity.prob_given_false),
                "weight": weight_value,
                "previous_evidence": entity.evidence,
                "decay": {
                    "is_decaying": entity.decay.is_decaying,
                    "decay_start": decay_start.isoformat(),
                },
            }
        )

    return entity_inputs


def _create_entities_from_inputs(
    entity_inputs: list[dict[str, Any]],
    area_half_life: float,
    weights_map: dict[str, float],
) -> tuple[dict[str, Entity], dict[str, Any]]:
    config = _create_simulator_config()
    state_map: dict[str, Any] = {}
    entities: dict[str, Entity] = {}

    for entity_input in entity_inputs:
        entity_id = entity_input["entity_id"]
        entity_type = _create_entity_type(entity_input.get("type", "unknown"), config)

        weight_override = entity_input.get("weight")
        type_weight_override = weights_map.get(entity_type.input_type.value)

        # Prioritize type weights over individual entity weights
        # This ensures that when a user changes a type weight slider,
        # it applies to all entities of that type
        if type_weight_override is not None:
            try:
                weight_value = float(type_weight_override)
                if MIN_WEIGHT <= weight_value <= MAX_WEIGHT:
                    entity_type.weight = weight_value
            except (TypeError, ValueError):
                pass
        elif weight_override is not None:
            try:
                weight_value = float(weight_override)
                if MIN_WEIGHT <= weight_value <= MAX_WEIGHT:
                    entity_type.weight = weight_value
            except (TypeError, ValueError):
                pass

        raw_state = entity_input.get("state")
        coerced_state = _coerce_state(entity_type, raw_state)
        state_map[entity_id] = coerced_state

        try:
            prob_true = float(
                entity_input.get("prob_given_true", entity_type.prob_given_true)
            )
        except (TypeError, ValueError):
            prob_true = entity_type.prob_given_true

        try:
            prob_false = float(
                entity_input.get("prob_given_false", entity_type.prob_given_false)
            )
        except (TypeError, ValueError):
            prob_false = entity_type.prob_given_false

        prob_true = _normalize_probability(prob_true)
        prob_false = _normalize_probability(prob_false)

        decay = _parse_decay_payload(
            entity_input.get("decay"),
            default_half_life=area_half_life,
        )

        entity = Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=prob_true,
            prob_given_false=prob_false,
            decay=decay,
            state_provider=_build_state_provider(state_map),
        )

        previous_evidence = entity_input.get("previous_evidence")
        if previous_evidence in (True, False, None):
            entity.previous_evidence = previous_evidence
        else:
            entity.previous_evidence = entity.evidence

        entities[entity_id] = entity

    return entities, state_map


def update_decay_states(entities: dict[str, Entity]) -> None:
    """Update per-entity decay state based on evidence transitions."""
    for entity in entities.values():
        current_evidence = entity.evidence
        previous_evidence = entity.previous_evidence

        if current_evidence is None or previous_evidence is None:
            entity.previous_evidence = current_evidence
            continue

        if current_evidence != previous_evidence:
            if current_evidence:
                entity.decay.stop_decay()
            else:
                entity.decay.start_decay()

        entity.previous_evidence = current_evidence


def calculate_probability_breakdown(
    entities: dict[str, Entity],
    area_prior: float,
    time_prior: float,
) -> tuple[float, dict[str, float]]:
    """Calculate total probability and per-entity contribution deltas."""
    _LOGGER.debug("=== SIMULATOR PRIOR CALCULATION TRACE START ===")
    _LOGGER.debug("global_prior = %.5f, time_prior = %.5f", area_prior, time_prior)

    combined_prior = combine_priors(area_prior, time_prior)
    _LOGGER.debug("combined_prior = %.5f", combined_prior)

    prior_before_clamp = combined_prior * PRIOR_FACTOR
    prior = max(MIN_PRIOR, min(MAX_PRIOR, prior_before_clamp))
    if prior != prior_before_clamp:
        _LOGGER.debug("prior clamped: %.5f -> %.5f", prior_before_clamp, prior)

    overall_prob = bayesian_probability(entities, prior)
    breakdown: dict[str, float] = {}

    for entity_id in entities:
        entities_without = {k: v for k, v in entities.items() if k != entity_id}
        prob_without = bayesian_probability(entities_without, prior)
        breakdown[entity_id] = overall_prob - prob_without

    return overall_prob, breakdown


def _calculate_priors(global_prior: float, time_prior: float) -> dict[str, float]:
    combined_prior = combine_priors(global_prior, time_prior)
    current_prior = combined_prior * PRIOR_FACTOR
    final_prior = max(MIN_PRIOR, min(MAX_PRIOR, current_prior))

    return {
        "combined": combined_prior,
        "current": current_prior,
        "final": final_prior,
    }


def _run_simulation(simulation_input: dict[str, Any]) -> dict[str, Any]:
    area = simulation_input["area"]
    weights_map = {k: float(v) for k, v in simulation_input.get("weights", {}).items()}

    entities, state_map = _create_entities_from_inputs(
        simulation_input["entities"],
        area["half_life"],
        weights_map,
    )

    # Extract extra fields (analysis_error, analysis_data, correlation_type) from entity inputs
    # These are stored in the entity input dicts but not in Entity objects
    entity_extra_fields: dict[str, dict[str, Any]] = {}
    for entity_input in simulation_input.get("entities", []):
        entity_id = entity_input.get("entity_id")
        if entity_id:
            extra = {}
            if "analysis_error" in entity_input:
                extra["analysis_error"] = entity_input["analysis_error"]
            if "analysis_data" in entity_input:
                extra["analysis_data"] = entity_input["analysis_data"]
            if "correlation_type" in entity_input:
                extra["correlation_type"] = entity_input["correlation_type"]
            if extra:
                entity_extra_fields[entity_id] = extra

    update_decay_states(entities)

    probability, breakdown = calculate_probability_breakdown(
        entities,
        area["global_prior"],
        area["time_prior"],
    )

    priors = _calculate_priors(area["global_prior"], area["time_prior"])

    # Build weights from entities first
    normalized_weights: dict[str, float] = {
        entity.type.input_type.value: float(entity.type.weight)
        for entity in entities.values()
    }

    # Override with requested type weights to ensure user's weight settings take precedence
    # This ensures that when a user changes a type weight slider, that value is returned
    # even if individual entities have different weights
    normalized_weights.update(weights_map)

    entity_inputs_next = _build_entity_inputs_from_entities(
        entities,
        state_map,
        normalized_weights,
    )

    return {
        "area": {
            "name": area["name"],
            "purpose": area.get("purpose"),
            "half_life": area["half_life"],
            "priors": {
                "global": area["global_prior"],
                "time": area["time_prior"],
                "combined": priors["combined"],
                "current": priors["current"],
                "final": priors["final"],
            },
        },
        "probability": probability,
        "entities": _serialize_entities(entities, breakdown, entity_extra_fields),
        "breakdown": _serialize_breakdown(entities, breakdown),
        "weights": normalized_weights,
        "entity_decay": _serialize_decay_map(entities),
        "entity_inputs": entity_inputs_next,
    }


def _normalize_area_payload(area_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(area_payload, dict):
        raise TypeError("Area payload must be an object")

    name = area_payload.get("name") or "Area"
    purpose = area_payload.get("purpose")

    try:
        global_prior = float(area_payload.get("global_prior", 0.5))
        time_prior = float(area_payload.get("time_prior", 0.5))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prior value: {exc}") from exc

    global_prior = max(0.0, min(1.0, global_prior))
    time_prior = max(0.0, min(1.0, time_prior))

    half_life = area_payload.get("half_life")
    if half_life is None:
        half_life = _get_half_life_from_purpose(purpose)
    else:
        try:
            half_life = max(1.0, float(half_life))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid half-life value: {exc}") from exc

    return {
        "name": str(name),
        "purpose": purpose,
        "global_prior": global_prior,
        "time_prior": time_prior,
        "half_life": half_life,
    }


def _normalize_entity_payloads(
    entity_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not entity_payloads:
        raise ValueError("At least one entity is required")

    normalized: list[dict[str, Any]] = []

    for payload in entity_payloads:
        if not isinstance(payload, dict):
            raise TypeError("Each entity must be an object")

        entity_id = payload.get("entity_id")
        if not entity_id:
            raise ValueError("Entity ID is required")

        entity_type = payload.get("type", "unknown")

        normalized.append(
            {
                "entity_id": str(entity_id),
                "type": entity_type,
                "state": payload.get("state"),
                "prob_given_true": payload.get("prob_given_true"),
                "prob_given_false": payload.get("prob_given_false"),
                "weight": payload.get("weight"),
                "previous_evidence": payload.get("previous_evidence"),
                "decay": payload.get("decay"),
            }
        )

    return normalized


def _normalize_simulation_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("Simulation payload must be an object")

    area_payload = payload.get("area", {})
    area = _normalize_area_payload(area_payload)

    weights_payload = payload.get("weights", {})
    weights: dict[str, float] = {}
    if isinstance(weights_payload, dict):
        for type_key, raw_weight in weights_payload.items():
            try:
                weight_value = float(raw_weight)
            except (TypeError, ValueError):
                continue
            weight_value = max(MIN_WEIGHT, min(MAX_WEIGHT, weight_value))
            weights[str(type_key)] = weight_value

    entities_payload = payload.get("entities")
    if not isinstance(entities_payload, list):
        raise TypeError("Field 'entities' must be an array")

    entities = _normalize_entity_payloads(entities_payload)

    return {
        "area": area,
        "weights": weights,
        "entities": entities,
    }


def _build_simulation_from_yaml(
    data: dict[str, Any], area_name: str | None = None
) -> dict[str, Any]:
    """Build simulation from YAML data, supporting both old and new formats."""
    if not isinstance(data, dict):
        raise TypeError("YAML payload must describe an object")

    # Check if this is the new multi-area format
    areas_dict = data.get("areas")
    if isinstance(areas_dict, dict) and areas_dict:
        # New format: multiple areas
        available_areas = list(areas_dict.keys())
        if not available_areas:
            raise ValueError("No areas found in YAML data")

        # Select area: use provided area_name, or first area, or first available
        selected_area_name = area_name or available_areas[0]
        if selected_area_name not in areas_dict:
            raise ValueError(
                f"Area '{selected_area_name}' not found. Available areas: {', '.join(available_areas)}"
            )

        area_data = areas_dict[selected_area_name]
        if not isinstance(area_data, dict):
            raise TypeError(f"Area data for '{selected_area_name}' must be an object")

        # Extract area information from new format
        area = {
            "name": area_data.get("area_name", selected_area_name),
            "purpose": None,  # New format doesn't include purpose
            "global_prior": float(area_data.get("global_prior", 0.5)),
            "time_prior": float(area_data.get("time_prior", 0.5)),
        }
        # Try to infer half_life from purpose if available, otherwise use default
        area["half_life"] = _get_half_life_from_purpose(area.get("purpose"))

        entity_states = area_data.get("entity_states", {}) or {}
        likelihoods = area_data.get("likelihoods", {}) or {}

    else:
        # Old format: single area at root level (backward compatibility)
        area = {
            "name": data.get("area_name", "Area"),
            "purpose": data.get("area_purpose"),
            "global_prior": float(data.get("global_prior", 0.5)),
            "time_prior": float(data.get("time_prior", 0.5)),
        }
        area["half_life"] = _get_half_life_from_purpose(area.get("purpose"))

        entity_states = data.get("entity_states", {}) or {}
        likelihoods = data.get("likelihoods", {}) or {}
        available_areas = None
        selected_area_name = None

    weights: dict[str, float] = {}
    entities: list[dict[str, Any]] = []

    for entity_id, raw_state in entity_states.items():
        likelihood = likelihoods.get(entity_id, {})
        if not isinstance(likelihood, dict):
            likelihood = {}

        entity_type = likelihood.get("type", "unknown")

        weight = likelihood.get("weight")
        if weight is not None:
            try:
                weight_value = float(weight)
                if MIN_WEIGHT <= weight_value <= MAX_WEIGHT:
                    weights.setdefault(str(entity_type), weight_value)
            except (TypeError, ValueError):
                pass

        # Build entity with all available fields, including new ones
        entity_dict = {
            "entity_id": entity_id,
            "type": entity_type,
            "state": raw_state,
            "prob_given_true": likelihood.get("prob_given_true"),
            "prob_given_false": likelihood.get("prob_given_false"),
            "weight": likelihood.get("weight"),
            "previous_evidence": None,
            "decay": {
                "is_decaying": False,
                "decay_start": dt_util.utcnow().isoformat(),
            },
        }

        # Preserve new fields from likelihood if present
        if "analysis_data" in likelihood:
            entity_dict["analysis_data"] = likelihood["analysis_data"]
        if "analysis_error" in likelihood:
            entity_dict["analysis_error"] = likelihood["analysis_error"]
        if "correlation_type" in likelihood:
            entity_dict["correlation_type"] = likelihood["correlation_type"]

        entities.append(entity_dict)

    simulation_payload = {
        "area": area,
        "weights": weights,
        "entities": entities,
    }

    normalized = _normalize_simulation_payload(simulation_payload)

    # Add metadata for new format
    if available_areas is not None:
        normalized["available_areas"] = available_areas
        normalized["selected_area_name"] = selected_area_name

    return normalized


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Execute a stateless simulation analysis."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _json_error("Invalid request payload", 400)

    try:
        simulation_input = _normalize_simulation_payload(payload)
        result = _run_simulation(simulation_input)
        return _json_success(result)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        _LOGGER.exception("Error analyzing simulation")
        return _json_error(f"Error analyzing simulation: {exc!s}", 500)


@app.route("/api/load", methods=["POST"])
def load_data():
    """Convert YAML analysis output to simulation input and run initial analysis."""
    payload = request.get_json(silent=True) or {}
    yaml_text = payload.get("yaml")
    if not yaml_text:
        return _json_error("No YAML data provided", 400)

    area_name = payload.get("area_name")  # Optional: for selecting specific area

    try:
        parsed_yaml = yaml.safe_load(yaml_text)
    except yaml.YAMLError as exc:
        return _json_error(f"Invalid YAML: {exc!s}", 400)

    try:
        simulation_input = _build_simulation_from_yaml(parsed_yaml or {}, area_name)
        result = _run_simulation(simulation_input)

        # Build response with available areas if in new format
        response = {"simulation": simulation_input, "result": result}
        if "available_areas" in simulation_input:
            response["available_areas"] = simulation_input["available_areas"]
            response["selected_area_name"] = simulation_input.get("selected_area_name")

        return _json_success(response)
    except ValueError as exc:
        return _json_error(str(exc), 400)
    except Exception as exc:
        _LOGGER.exception("Error loading simulation")
        return _json_error(f"Error loading simulation: {exc!s}", 500)


@app.route("/api/get-purposes", methods=["GET"])
def get_purposes():
    """Return available area purposes."""
    purposes = [
        {
            "value": purpose_enum.value,
            "label": purpose_def.name,
            "half_life": purpose_def.half_life,
        }
        for purpose_enum, purpose_def in PURPOSE_DEFINITIONS.items()
    ]
    return _json_success({"purposes": purposes})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
