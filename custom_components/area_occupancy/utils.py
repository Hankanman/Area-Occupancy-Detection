"""Utility functions for Area Occupancy Detection."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.device_registry import ChildDeviceInfo, DeviceInfo

from .const import (
    ACTIVITY_BOOST_FADE_LOGIT,
    DOMAIN,
    GROUND_TRUTH_ACTIVE_FLOOR,
    GROUND_TRUTH_HOLD_HALF_LIVES,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    ROUNDING_PRECISION,
)

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator
    from .data.entity import Entity

_LOGGER = logging.getLogger(__name__)


def _get_device_identifiers(
    device_info: DeviceInfo | ChildDeviceInfo | dict[str, Any] | None,
) -> set[tuple[str, str]] | None:
    """Read identifiers from legacy dictionaries or device info dataclasses."""
    if device_info is None:
        return None
    if isinstance(device_info, dict):
        return device_info.get("identifiers")
    identifiers = device_info.identifiers
    # An unset dataclass field holds Home Assistant's UNDEFINED sentinel.
    return identifiers if isinstance(identifiers, set) else None


def assign_device_to_ha_area(
    hass: HomeAssistant,
    device_info: DeviceInfo | ChildDeviceInfo | None,
    area_id: str | None,
    config_entry_id: str,
) -> None:
    """Assign a registered device to its configured Home Assistant area."""
    if not area_id or not (identifiers := _get_device_identifiers(device_info)):
        return

    registry = dr.async_get(hass)
    is_child = (
        "parent_device_id" in device_info
        if isinstance(device_info, dict)
        else isinstance(device_info, ChildDeviceInfo)
    )
    lookup = (
        registry.async_get_child_device_by_identifier
        if is_child
        else registry.async_get_device_by_identifier
    )
    matched_device: dr.DeviceEntry | dr.ChildDeviceEntry | None = None
    for identifier in identifiers:
        device = lookup(identifier, config_entry_id)
        if device is None:
            continue
        if matched_device is not None and device.id != matched_device.id:
            _LOGGER.warning(
                "Device identifiers match multiple devices in config entry %s; "
                "skipping area assignment",
                config_entry_id,
            )
            return
        matched_device = device

    if matched_device is None or matched_device.area_id == area_id:
        return
    if is_child:
        registry.async_update_child_device(matched_device.id, area_id=area_id)
    else:
        registry.async_update_device(matched_device.id, area_id=area_id)


def format_float(value: float, precision: int = ROUNDING_PRECISION) -> float:
    """Round a numeric value to the requested decimal precision."""
    return round(float(value), precision)


def format_percentage(value: float) -> str:
    """Format a fraction as a percentage with two decimal places."""
    return f"{value * 100:.2f}%"


def is_valid_number(value: float) -> bool:
    """Return whether the value is finite."""
    return math.isfinite(value)


def clamp_probability(
    value: float, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Clamp a probability to the default or caller-supplied bounds.

    Custom bounds are expected to be finite and ordered. Preserve the existing
    NaN fallback to the upper bound.
    """
    min_bound = MIN_PROBABILITY if min_val is None else min_val
    max_bound = MAX_PROBABILITY if max_val is None else max_val
    if math.isnan(value):
        _LOGGER.warning(
            "clamp_probability received NaN; using upper bound %s", max_bound
        )
        return max_bound
    if value == math.inf:
        return max_bound
    if value == -math.inf:
        return min_bound
    return max(min_bound, min(max_bound, value))


def map_binary_state_to_semantic(state: str, active_states: list[str]) -> str:
    """Map on/off to open/closed when the configuration uses those states."""
    if "open" in active_states or "closed" in active_states:
        if state == "on":
            return "open"
        if state == "off":
            return "closed"
    return state


def sigmoid(z: float) -> float:
    """Convert log-odds to a probability without overflowing exp()."""
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)


def logit(p: float) -> float:
    """Convert a probability to log-odds after clamping it away from 0 and 1."""
    p = clamp_probability(p)
    return math.log(p / (1 - p))


def evidence_value(entity: Entity) -> float:
    """Return the evidence factor shared by live and shadow fusion."""
    if getattr(entity, "is_stuck", False) is True:
        return 0.0
    if entity.evidence is True:
        return 1.0
    if entity.decay.is_decaying:
        return entity.decay_factor
    return 0.0


def sigmoid_probability(
    entities: dict[str, Entity],
    prior: float = 0.5,
    correlations: dict[str, float] | None = None,
    threshold: float | None = None,
) -> float:
    """Calculate occupancy from a prior and weighted evidence in logit space.

    At configured weight and correlation of 1, an active motion or sleep
    sensor with positive effective weight can lift the estimate to the active
    floor. If the threshold is above the prior, use a threshold-based signal
    during decay so an isolated full-weight sensor holds it for the configured
    number of decay half-lives. Count only the strongest motion or sleep
    channel per physical device.
    """
    from .data.entity_type import InputType  # noqa: PLC0415

    if not entities:
        return clamp_probability(prior)

    bias = logit(prior)
    ground_truth_floor = logit(GROUND_TRUTH_ACTIVE_FLOOR) - bias
    hold: float | None = None
    if threshold is not None:
        hold = (logit(threshold) - bias) * (2**GROUND_TRUTH_HOLD_HALF_LIVES)
        ground_truth_floor = max(ground_truth_floor, hold)

    z = bias
    device_contributions: dict[str, float] = {}
    for entity_id, entity in entities.items():
        weight = entity.weight
        if weight <= 0:
            continue

        correlation = correlations.get(entity_id, 1.0) if correlations else 1.0
        evidence = evidence_value(entity)
        entity_type = entity.type
        is_ground_truth = entity_type.input_type in (InputType.MOTION, InputType.SLEEP)
        effective_weight = getattr(entity, "effective_weight", weight)
        signal = entity.prob_given_true * getattr(
            entity_type, "strength_multiplier", 2.0
        )

        if is_ground_truth:
            # Undo information-gain attenuation for the floor/hold only.
            gain = effective_weight / weight if effective_weight > 0 else 1.0
            if hold is not None and hold > 0 and 0.0 < evidence < 1.0:
                signal = hold / gain
            else:
                signal = max(signal, ground_truth_floor / gain)

        contribution = effective_weight * evidence * correlation * signal
        device_id = getattr(entity, "device_id", None) if is_ground_truth else None
        if isinstance(device_id, str):
            device_contributions[device_id] = max(
                device_contributions.get(device_id, 0.0), contribution
            )
        else:
            z += contribution

    z += sum(device_contributions.values())
    return clamp_probability(sigmoid(z))


def presence_probability(
    entities: dict[str, Entity],
    prior: float = 0.5,
    correlations: dict[str, float] | None = None,
    threshold: float | None = None,
) -> float:
    """Calculate probability from presence sensors, or use a reduced prior."""
    from .data.entity_type import PRESENCE_INPUT_TYPES  # noqa: PLC0415

    presence_entities = {
        entity_id: entity
        for entity_id, entity in entities.items()
        if entity.type.input_type in PRESENCE_INPUT_TYPES
    }
    if not presence_entities:
        return clamp_probability(prior * 0.5)
    return sigmoid_probability(presence_entities, prior, correlations, threshold)


def environmental_confidence(
    entities: dict[str, Entity],
    correlations: dict[str, float] | None = None,
) -> float:
    """Calculate environmental support relative to a neutral prior of 0.5."""
    from .data.entity_type import ENVIRONMENTAL_INPUT_TYPES  # noqa: PLC0415

    env_entities = {
        entity_id: entity
        for entity_id, entity in entities.items()
        if entity.type.input_type in ENVIRONMENTAL_INPUT_TYPES
    }
    if not env_entities:
        return 0.5
    return sigmoid_probability(env_entities, prior=0.5, correlations=correlations)


def combined_probability(presence: float, environmental: float) -> float:
    """Update presence with 20% of the environmental logit contribution."""
    z_presence = logit(presence)
    z_env = logit(environmental)
    return clamp_probability(sigmoid(z_presence + 0.2 * z_env))


def activity_boost_fade(base_probability: float, threshold: float) -> float:
    """Ramp the boost from zero at the threshold to full strength above it."""
    headroom = logit(base_probability) - logit(threshold)
    return max(0.0, min(1.0, headroom / ACTIVITY_BOOST_FADE_LOGIT))


def apply_activity_boost(
    base_probability: float,
    activity_boost: float,
    activity_confidence: float,
) -> float:
    """Apply a confidence-weighted activity boost in logit space."""
    effective_boost = activity_boost * activity_confidence
    if effective_boost <= 0:
        return clamp_probability(base_probability)
    z = logit(base_probability) + effective_boost
    return clamp_probability(sigmoid(z))


def combine_priors(
    area_prior: float, time_prior: float, time_weight: float = 0.4
) -> float:
    """Interpolate between area and time priors in logit space."""
    if time_weight == 0.0:
        return clamp_probability(area_prior)
    if time_weight == 1.0:
        return clamp_probability(time_prior)

    # Normalize exact endpoints before testing for nearly identical priors.
    if area_prior == 0.0:
        area_prior = MIN_PROBABILITY
    elif area_prior == 1.0:
        area_prior = MAX_PROBABILITY
    if time_prior == 0.0:
        time_prior = MIN_PROBABILITY
    elif time_prior == 1.0:
        time_prior = MAX_PROBABILITY

    if abs(area_prior - time_prior) < 1e-10:
        return clamp_probability(area_prior)

    time_weight = max(0.0, min(1.0, time_weight))
    area_logit = logit(area_prior)
    time_logit = logit(time_prior)
    combined_logit = (1.0 - time_weight) * area_logit + time_weight * time_logit
    return clamp_probability(sigmoid(combined_logit))


def format_area_names(coordinator: AreaOccupancyCoordinator) -> str:
    """Format the names, falling back while the coordinator is unavailable."""
    try:
        area_names = coordinator.get_area_names()
        return ", ".join(area_names) if area_names else "no areas"
    except Exception:  # noqa: BLE001
        return "no areas"


def get_coordinator(hass: HomeAssistant) -> AreaOccupancyCoordinator:
    """Return the global coordinator or raise if it has not been configured."""
    coordinator = hass.data.get(DOMAIN)
    if coordinator is None:
        raise HomeAssistantError(
            "Area Occupancy coordinator not found. Ensure integration is configured."
        )
    return coordinator


def extract_device_identifier_from_device_info(
    device_info: DeviceInfo | ChildDeviceInfo | dict[str, Any],
) -> str | None:
    """Extract the first identifier without changing legacy unique-ID selection."""
    identifiers = _get_device_identifiers(device_info)
    if not identifiers:
        return None
    try:
        identifier = next(iter(identifiers), None)
    except TypeError:
        return None
    if isinstance(identifier, tuple) and len(identifier) >= 2:
        return str(identifier[1])
    return None


def generate_entity_unique_id(
    entry_id: str,
    device_info: DeviceInfo | ChildDeviceInfo | dict[str, Any] | None,
    entity_name: str,
) -> str:
    """Generate the existing platform unique-ID format."""
    device_id = (
        extract_device_identifier_from_device_info(device_info)
        if device_info is not None
        else None
    )
    if device_id is None:
        device_id = entry_id
    normalized_name = entity_name.lower().replace(" ", "_")
    return f"{entry_id}_{device_id}_{normalized_name}"
