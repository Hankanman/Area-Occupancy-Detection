"""Utility functions for Area Occupancy Detection."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo

from .const import DOMAIN, MAX_PROBABILITY, MIN_PROBABILITY, ROUNDING_PRECISION

_LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator
    from .data.entity import Entity


def format_float(value: float) -> float:
    """Format float value."""
    return round(float(value), ROUNDING_PRECISION)


def format_percentage(value: float) -> str:
    """Format float value as percentage."""
    return f"{value * 100:.2f}%"


def is_valid_number(value: float) -> bool:
    """Check if value is a valid finite number (not NaN or infinity).

    Args:
        value: Value to check

    Returns:
        True if value is finite and not NaN, False otherwise
    """
    return not (math.isnan(value) or math.isinf(value))


def clamp_probability(
    value: float, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Clamp probability value to valid range.

    Args:
        value: Probability value to clamp
        min_val: Minimum value (default: MIN_PROBABILITY from const)
        max_val: Maximum value (default: MAX_PROBABILITY from const)

    Returns:
        Clamped probability value (always a valid finite number)
    """
    # Handle NaN and infinity values explicitly
    if not is_valid_number(value):
        if math.isinf(value):
            if value > 0:
                # Positive infinity -> clamp to MAX_PROBABILITY
                return max_val if max_val is not None else MAX_PROBABILITY
            # Negative infinity -> clamp to MIN_PROBABILITY
            return min_val if min_val is not None else MIN_PROBABILITY
        # NaN -> clamp to MAX_PROBABILITY (matching existing test behavior)
        _LOGGER.warning(
            "clamp_probability received invalid value (NaN): %s, using MAX_PROBABILITY",
            value,
        )
        return max_val if max_val is not None else MAX_PROBABILITY

    min_bound = min_val if min_val is not None else MIN_PROBABILITY
    max_bound = max_val if max_val is not None else MAX_PROBABILITY
    return max(min_bound, min(max_bound, value))


def map_binary_state_to_semantic(state: str, active_states: list[str]) -> str:
    """Map binary sensor state ('on'/'off') to semantic state ('open'/'closed') if needed.

    Home Assistant binary sensors always report 'on'/'off', but some configs use
    semantic states like 'open'/'closed'. This function maps between them.

    Args:
        state: The actual state from the sensor ('on' or 'off')
        active_states: List of active states expected by the config

    Returns:
        The mapped state if mapping is needed, otherwise the original state
    """
    # If active_states contains semantic states, map binary states
    if "closed" in active_states or "open" in active_states:
        # Map binary states to semantic states
        # For doors: 'off' means closed, 'on' means open
        # For windows: 'off' means closed, 'on' means open
        if state == "off":
            return "closed"
        if state == "on":
            return "open"
    # No mapping needed, return original state
    return state


# ────────────────────────────────────── Sigmoid Functions ───────────────────────────


def sigmoid(z: float) -> float:
    """Compute sigmoid function with numerical stability.

    Args:
        z: Input value (log-odds)

    Returns:
        Probability in range (0, 1)
    """
    # Handle edge cases for numerical stability
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1 + exp_z)


def logit(p: float) -> float:
    """Compute logit (inverse sigmoid) with bounds protection.

    Args:
        p: Probability value

    Returns:
        Log-odds value
    """
    p = clamp_probability(p)  # Ensure 0.01-0.99 range
    return math.log(p / (1 - p))


def sigmoid_probability(
    entities: dict[str, Entity],
    prior: float = 0.5,
    correlations: dict[str, float] | None = None,
) -> float:
    """Calculate occupancy probability using weighted sigmoid model.

    This replaces Bayesian calculation with logistic regression-style probability:
    z = bias + Σ(weight_i × evidence_i × correlation_i × strength_factor)
    P = sigmoid(z)

    Args:
        entities: Dict of Entity objects
        prior: Learned prior probability for this area (0.0-1.0)
        correlations: Optional dict of entity_id -> correlation strength (0-1)
                     If None or missing entries, defaults to 1.0

    Returns:
        Probability in range MIN_PROBABILITY to MAX_PROBABILITY
    """
    if not entities:
        return clamp_probability(prior)

    # Start with bias from prior (logit transforms prior to log-odds space)
    # logit(0.5) = 0, logit(0.7) = 0.85, logit(0.3) = -0.85
    bias = logit(prior)

    # Sum weighted contributions from all entities
    z = bias

    for entity_id, entity in entities.items():
        if entity.weight <= 0:
            continue

        # Get correlation multiplier (learned or default)
        correlation = 1.0
        if correlations and entity_id in correlations:
            correlation = correlations[entity_id]

        # Determine evidence contribution
        # Active = full contribution, Decaying = partial, Inactive = zero
        if entity.evidence is True:
            evidence = 1.0
        elif entity.decay.is_decaying:
            evidence = entity.decay_factor  # Gradual fade (0.0 to 1.0)
        else:
            evidence = 0.0  # Inactive = no contribution (not negative!)

        # Scale by sensor type strength (prob_given_true indicates signal strength)
        # Motion (0.95) contributes more than door (0.2)
        strength = entity.prob_given_true

        # Use effective_weight (weight × information_gain) so uninformative sensors
        # contribute less. Falls back to weight if effective_weight is not available.
        ew = getattr(entity, "effective_weight", entity.weight)

        # Add to z: effective_weight × evidence × correlation × strength_factor.
        # strength_multiplier is per-type (e.g., 3.0 for motion, 2.0 for others)
        # to give ground-truth sensors a stronger logit-space contribution.
        strength_multiplier = getattr(entity.type, "strength_multiplier", 2.0)
        contribution = ew * evidence * correlation * (strength * strength_multiplier)
        z += contribution

    return clamp_probability(sigmoid(z))


def presence_probability(
    entities: dict[str, Entity],
    prior: float = 0.5,
    correlations: dict[str, float] | None = None,
) -> float:
    """Calculate presence probability from strong binary indicators.

    Filters entities to only include presence-related sensors (motion, media,
    appliances, doors, windows, covers, power) and calculates probability
    using the sigmoid model.

    Args:
        entities: Dict of Entity objects
        prior: Learned prior probability for this area
        correlations: Optional dict of entity_id -> correlation strength

    Returns:
        Probability in range MIN_PROBABILITY to MAX_PROBABILITY
    """
    from .data.entity_type import PRESENCE_INPUT_TYPES  # noqa: PLC0415

    presence_entities = {
        eid: e
        for eid, e in entities.items()
        if e.type.input_type in PRESENCE_INPUT_TYPES
    }

    if not presence_entities:
        # No presence sensors - return reduced prior (uncertain state)
        return clamp_probability(prior * 0.5)

    return sigmoid_probability(presence_entities, prior, correlations)


def environmental_confidence(
    entities: dict[str, Entity],
    correlations: dict[str, float] | None = None,
) -> float:
    """Calculate environmental support as 0-1 confidence.

    Uses sigmoid centered at 0.5 (neutral prior) so the result represents
    how much environmental data supports vs opposes occupancy.

    Args:
        entities: Dict of Entity objects
        correlations: Optional dict of entity_id -> correlation strength

    Returns:
        Confidence value 0.0-1.0 (0.5 = neutral, >0.5 = supports, <0.5 = opposes)
    """
    from .data.entity_type import ENVIRONMENTAL_INPUT_TYPES  # noqa: PLC0415

    env_entities = {
        eid: e
        for eid, e in entities.items()
        if e.type.input_type in ENVIRONMENTAL_INPUT_TYPES
    }

    if not env_entities:
        return 0.5  # Neutral - no environmental data

    # Use 0.5 prior so result is purely from environmental evidence
    return sigmoid_probability(env_entities, prior=0.5, correlations=correlations)


def combined_probability(
    presence: float,
    environmental: float,
) -> float:
    """Combine presence and environmental using weighted average in logit space.

    This preserves the smooth sigmoid properties while combining both signals.
    Presence is weighted more heavily (80%) than environmental (20%).

    Args:
        presence: Presence probability (0.0-1.0)
        environmental: Environmental confidence (0.0-1.0)

    Returns:
        Combined probability in range MIN_PROBABILITY to MAX_PROBABILITY
    """
    # Convert to logit space for principled combination
    z_presence = logit(presence)
    z_env = logit(environmental)

    # Weighted combination (presence dominates at 80%, environmental at 20%)
    z_combined = 0.8 * z_presence + 0.2 * z_env

    return clamp_probability(sigmoid(z_combined))


def apply_activity_boost(
    base_probability: float,
    activity_boost: float,
    activity_confidence: float,
) -> float:
    """Apply an activity-based occupancy boost in logit space.

    When a strong activity is detected (e.g. watching TV, showering), the
    combination of signals is a stronger occupancy indicator than any single
    sensor. This function boosts the base probability proportionally to the
    activity's strength and confidence.

    Args:
        base_probability: Sensor-only probability (0.0-1.0).
        activity_boost: Logit-space boost magnitude from the activity definition.
        activity_confidence: Activity detection confidence (0.0-1.0).

    Returns:
        Boosted probability in range MIN_PROBABILITY to MAX_PROBABILITY.
    """
    effective_boost = activity_boost * activity_confidence
    if effective_boost <= 0:
        return clamp_probability(base_probability)
    z = logit(base_probability) + effective_boost
    return clamp_probability(sigmoid(z))


# ────────────────────────────────────── Core Bayes ───────────────────────────
def _filter_invalid_entity_likelihoods(
    active_entities: dict[str, Entity],
) -> dict[str, Entity]:
    """Validate entity likelihoods and filter out invalid entities.

    Args:
        active_entities: Dictionary of entities to validate

    Returns:
        Dictionary of entities with valid likelihoods
    """
    entities_to_remove = []
    for entity_id, entity in active_entities.items():
        # Check if entity uses continuous likelihood (densities can be > 1.0)
        is_continuous = getattr(entity, "is_continuous_likelihood", False)
        if not isinstance(is_continuous, bool):
            is_continuous = False

        # Validate likelihoods based on sensor type
        # First check for NaN/inf explicitly (comparison operators don't catch NaN)
        if not is_valid_number(entity.prob_given_true) or not is_valid_number(
            entity.prob_given_false
        ):
            # Mark entities with NaN/inf likelihoods for removal
            entities_to_remove.append(entity_id)
            continue

        if is_continuous:
            # For continuous likelihoods (densities), only check > 0
            # Densities can be > 1.0, so we don't check upper bound
            if entity.prob_given_true <= 0.0 or entity.prob_given_false <= 0.0:
                # Mark entities with invalid likelihoods for removal
                entities_to_remove.append(entity_id)
        # For standard probabilities, validate [0, 1] range
        elif (
            entity.prob_given_true <= 0.0
            or entity.prob_given_true >= 1.0
            or entity.prob_given_false <= 0.0
            or entity.prob_given_false >= 1.0
        ):
            # Mark entities with invalid likelihoods for removal
            entities_to_remove.append(entity_id)

    # Remove invalid entities after iteration
    for entity_id in entities_to_remove:
        active_entities.pop(entity_id, None)

    return active_entities


def _get_entity_likelihoods(
    entity: Entity, is_continuous: bool, effective_evidence: bool
) -> tuple[float, float] | None:
    """Get likelihoods for an entity, handling continuous and binary sensors.

    Args:
        entity: Entity to get likelihoods for
        is_continuous: Whether entity uses continuous likelihood
        effective_evidence: Whether entity has effective evidence

    Returns:
        Tuple of (p_t, p_f) or None if entity should be skipped
    """
    if effective_evidence:
        # Evidence is present (either current or decaying) - use likelihoods
        # Use dynamic likelihoods if available (for Gaussian sensors)
        if is_continuous and hasattr(entity, "get_likelihoods"):
            # Ensure get_likelihoods is callable (Mocks make everything callable but check anyway)
            if callable(entity.get_likelihoods):
                p_t, p_f = entity.get_likelihoods()
                # Validate return values for NaN/inf
                if not is_valid_number(p_t) or not is_valid_number(p_f):
                    # Fallback to static values if get_likelihoods() returned invalid values
                    _LOGGER.warning(
                        "get_likelihoods() returned invalid values (NaN/inf) for %s, using static probabilities",
                        entity.entity_id if hasattr(entity, "entity_id") else "unknown",
                    )
                    p_t = entity.prob_given_true
                    p_f = entity.prob_given_false
                    # If static values are also invalid, skip this entity
                    if not is_valid_number(p_t) or not is_valid_number(p_f):
                        return None
                return (p_t, p_f)
            return (entity.prob_given_true, entity.prob_given_false)
        return (entity.prob_given_true, entity.prob_given_false)

    # No evidence present
    if is_continuous:
        # For continuous sensors with no effective evidence, use get_likelihoods()
        if hasattr(entity, "get_likelihoods") and callable(entity.get_likelihoods):
            p_t, p_f = entity.get_likelihoods()
            # Validate return values for NaN/inf
            if not is_valid_number(p_t) or not is_valid_number(p_f):
                # Fallback to neutral values if get_likelihoods() returned invalid values
                _LOGGER.warning(
                    "get_likelihoods() returned invalid values (NaN/inf) for %s, using neutral values",
                    entity.entity_id if hasattr(entity, "entity_id") else "unknown",
                )
                # Can't use inverse for densities, so use neutral values
                return (0.5, 0.5)
            return (p_t, p_f)
        # Fallback: use neutral values
        return (0.5, 0.5)

    # Binary sensors: use inverse probabilities
    return (1.0 - entity.prob_given_true, 1.0 - entity.prob_given_false)


def _apply_decay_interpolation(
    p_t: float, p_f: float, is_decaying: bool, decay_factor: float
) -> tuple[float, float]:
    """Apply decay interpolation to likelihoods.

    Args:
        p_t: Probability given true
        p_f: Probability given false
        is_decaying: Whether entity is decaying
        decay_factor: Decay factor (0.0 to 1.0)

    Returns:
        Tuple of (adjusted p_t, adjusted p_f)
    """
    if is_decaying and decay_factor < 1.0:
        # When decaying, interpolate between neutral and full evidence based on decay factor
        neutral_prob = 0.5
        p_t = neutral_prob + (p_t - neutral_prob) * decay_factor
        p_f = neutral_prob + (p_f - neutral_prob) * decay_factor
    return (p_t, p_f)


def bayesian_probability(entities: dict[str, Entity], prior: float = 0.5) -> float:
    """Compute posterior probability of occupancy given current features and prior.

    Args:
        entities: Dict mapping entity_id to Entity objects containing evidence and likelihood
        prior: Prior probability of occupancy for this area (default: 0.5)

    """
    # Handle edge cases first
    if not entities:
        # No entities provided - return prior
        return clamp_probability(prior)

    # Check for entities with zero weights (they contribute nothing)
    active_entities = {k: v for k, v in entities.items() if v.weight > 0.0}

    if not active_entities:
        # All entities have zero weight - return prior
        return clamp_probability(prior)

    # Check for entities with invalid likelihoods
    active_entities = _filter_invalid_entity_likelihoods(active_entities)

    if not active_entities:
        # All entities had invalid likelihoods - return prior
        return clamp_probability(prior)

    # Clamp prior
    prior = clamp_probability(prior)

    # log-space for numerical stability
    log_true = math.log(prior)
    log_false = math.log(1 - prior)

    for entity in active_entities.values():
        value = entity.evidence
        # Use entity.decay_factor property which handles evidence=True case correctly
        # Clamp decay factor locally to avoid mutating entity state
        # This is defensive programming: decay.decay_factor should always return [0, 1],
        # but we clamp to handle edge cases (e.g., clock skew, invalid half_life values)
        decay_factor = entity.decay_factor
        if decay_factor < 0.0 or decay_factor > 1.0:
            decay_factor = max(0.0, min(1.0, decay_factor))
        is_decaying = entity.decay.is_decaying

        # Skip entities with no evidence (unavailable) unless they're decaying
        # Unavailable entities should not contribute to the calculation
        if value is None and not is_decaying:
            continue

        # Determine effective evidence: True if evidence is True OR if decaying
        effective_evidence = value or is_decaying

        # Check if entity supports continuous likelihood (Gaussian density)
        # Continuous likelihoods are probability densities and can be > 1.0
        # Use simple getattr to be safe with Mocks
        is_continuous = getattr(entity, "is_continuous_likelihood", False)
        # Verify it's actually a boolean and True (Mocks can return Mocks for attributes)
        if not isinstance(is_continuous, bool):
            is_continuous = False

        # Get likelihoods for this entity
        likelihoods = _get_entity_likelihoods(entity, is_continuous, effective_evidence)
        if likelihoods is None:
            continue
        p_t, p_f = likelihoods

        # Apply decay interpolation if needed
        p_t, p_f = _apply_decay_interpolation(p_t, p_f, is_decaying, decay_factor)

        # Clamp probabilities to avoid log(0) or log(1)
        # For continuous likelihoods (densities), we only clamp the lower bound > 0
        # For standard probabilities, we clamp to [MIN_PROBABILITY, MAX_PROBABILITY]
        if is_continuous:
            # Ensure strictly positive for log()
            # Don't clamp upper bound as densities can be > 1
            p_t = max(1e-9, p_t)
            p_f = max(1e-9, p_f)
        else:
            p_t = clamp_probability(p_t)
            p_f = clamp_probability(p_f)

        log_p_t = math.log(p_t)
        log_p_f = math.log(p_f)
        # Use effective_weight so uninformative sensors contribute less.
        ew = getattr(entity, "effective_weight", entity.weight)
        contribution_true = log_p_t * ew
        contribution_false = log_p_f * ew

        log_true += contribution_true
        log_false += contribution_false

    # convert back
    max_log = max(log_true, log_false)
    true_prob = math.exp(log_true - max_log)
    false_prob = math.exp(log_false - max_log)

    # Handle numerical overflow/underflow edge case
    total_prob = true_prob + false_prob
    if total_prob == 0.0:
        # Both probabilities are zero - return prior as fallback
        return prior

    return clamp_probability(true_prob / total_prob)


def combine_priors(
    area_prior: float, time_prior: float, time_weight: float = 0.4
) -> float:
    """Combine area prior and time prior using weighted averaging in logit space.

    Args:
        area_prior: Base prior probability of occupancy for this area
        time_prior: Time-based modifier for the prior
        time_weight: Weight given to time_prior (0.0 to 1.0, default: 0.4)

    Returns:
        float: Combined prior probability

    """
    # Handle edge cases first
    if time_weight == 0.0:
        # No time influence, return area_prior
        return clamp_probability(area_prior)

    if time_weight == 1.0:
        # Full time influence, return time_prior (with clamping)
        return clamp_probability(time_prior)

    if time_prior == 0.0:
        # Time slot has never been occupied - this is strong evidence
        # Use a very small probability but not zero
        time_prior = MIN_PROBABILITY
    elif time_prior == 1.0:
        # Time slot has always been occupied - this is strong evidence
        time_prior = MAX_PROBABILITY

    # Handle area_prior edge cases
    if area_prior == 0.0:
        # Area has never been occupied - this is strong evidence
        area_prior = MIN_PROBABILITY
    elif area_prior == 1.0:
        # Area has always been occupied - this is strong evidence
        area_prior = MAX_PROBABILITY

    # Handle identical priors case
    if abs(area_prior - time_prior) < 1e-10:
        # Priors are essentially identical, return the common value
        return area_prior

    # Clamp inputs to valid ranges to prevent division by zero in logit conversion
    # prob_to_logit(p) = log(p / (1 - p)) requires p ∈ (0, 1)
    area_prior = clamp_probability(area_prior)
    time_prior = clamp_probability(time_prior)
    time_weight = max(0.0, min(1.0, time_weight))

    area_weight = 1.0 - time_weight

    # Interpolate in logit space for more principled combination
    area_logit = logit(area_prior)
    time_logit = logit(time_prior)

    # Weighted combination in logit space
    combined_logit = area_weight * area_logit + time_weight * time_logit
    combined_prior = sigmoid(combined_logit)

    return clamp_probability(combined_prior)


# ────────────────────────────────────── Coordinator Utilities ───────────────────────────


def format_area_names(coordinator: AreaOccupancyCoordinator) -> str:
    """Format area names as a comma-separated string.

    Args:
        coordinator: The coordinator instance containing areas

    Returns:
        Comma-separated string of area names, or "no areas" if empty
    """
    try:
        if not hasattr(coordinator, "get_area_names"):
            return "no areas"
        area_names = coordinator.get_area_names()
        return ", ".join(area_names) if area_names else "no areas"
    except Exception:  # noqa: BLE001
        # Handle case where coordinator isn't fully initialized or get_area_names fails
        return "no areas"


def get_coordinator(hass: HomeAssistant) -> AreaOccupancyCoordinator:
    """Get global coordinator from hass.data with error handling.

    Args:
        hass: Home Assistant instance

    Returns:
        AreaOccupancyCoordinator instance

    Raises:
        HomeAssistantError: If coordinator not found
    """
    coordinator = hass.data.get(DOMAIN)
    if coordinator is None:
        raise HomeAssistantError(
            "Area Occupancy coordinator not found. Ensure integration is configured."
        )
    return coordinator


def extract_device_identifier_from_device_info(device_info: DeviceInfo) -> str | None:
    """Extract device identifier from DeviceInfo identifiers.

    Args:
        device_info: DeviceInfo object containing identifiers

    Returns:
        Device identifier string (second element of tuple) or None if not found
    """
    identifiers = device_info.get("identifiers")
    if not identifiers:
        return None
    # Identifiers is a set of tuples like {(DOMAIN, device_identifier)}
    try:
        identifier_tuple = next(iter(identifiers))
        if isinstance(identifier_tuple, tuple) and len(identifier_tuple) >= 2:
            return str(identifier_tuple[1])
    except (StopIteration, TypeError):
        pass
    return None


def generate_entity_unique_id(
    entry_id: str,
    device_info: DeviceInfo | dict[str, Any] | None,
    entity_name: str,
) -> str:
    """Generate consistent unique_id for platform entities.

    Args:
        entry_id: Config entry identifier associated with the entity.
        device_info: DeviceInfo for the parent device (may be None).
        entity_name: Entity name constant (will be normalized).

    Returns:
        Unique ID in format: {entry_id}_{device_id}_{normalized_entity_name}
    """
    device_id = extract_device_identifier_from_device_info(device_info or {})
    if device_id is None:
        device_id = entry_id

    normalized_name = entity_name.lower().replace(" ", "_")

    return f"{entry_id}_{device_id}_{normalized_name}"
