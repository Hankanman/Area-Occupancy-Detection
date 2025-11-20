"""Utility functions for Area Occupancy Detection."""

from __future__ import annotations

from datetime import datetime
import logging
import math
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.util import dt as dt_util

from .const import DOMAIN, MAX_PROBABILITY, MIN_PROBABILITY, ROUNDING_PRECISION

_LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator
    from .data.entity import Entity


def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure a datetime is timezone-aware, assuming UTC if naive.

    Args:
        dt: The datetime object to make timezone-aware

    Returns:
        A timezone-aware datetime object

    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=dt_util.UTC)
    return dt


def format_float(value: float) -> float:
    """Format float value."""
    return round(float(value), ROUNDING_PRECISION)


def format_percentage(value: float) -> str:
    """Format float value as percentage."""
    return f"{value * 100:.2f}%"


def clamp_probability(value: float) -> float:
    """Clamp probability value to valid range [MIN_PROBABILITY, MAX_PROBABILITY]."""
    return max(MIN_PROBABILITY, min(MAX_PROBABILITY, value))


# ────────────────────────────────────── Core Bayes ───────────────────────────
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
    entities_to_remove = []
    for entity_id, entity in active_entities.items():
        if (
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
        # Clamp decay factor locally to avoid mutating entity state
        decay_factor = entity.decay.decay_factor
        if decay_factor < 0.0 or decay_factor > 1.0:
            decay_factor = max(0.0, min(1.0, decay_factor))
        is_decaying = entity.decay.is_decaying

        # Skip entities with no evidence (unavailable) unless they're decaying
        # Unavailable entities should not contribute to the calculation
        if value is None and not is_decaying:
            continue

        # Determine effective evidence: True if evidence is True OR if decaying
        effective_evidence = value or is_decaying

        if effective_evidence:
            # Evidence is present (either current or decaying) - use likelihoods with decay applied
            p_t = entity.prob_given_true
            p_f = entity.prob_given_false

            # Apply decay factor to reduce the strength of the evidence
            if is_decaying and decay_factor < 1.0:
                # When decaying, interpolate between neutral (0.5) and full evidence based on decay factor
                neutral_prob = 0.5
                p_t = neutral_prob + (p_t - neutral_prob) * decay_factor
                p_f = neutral_prob + (p_f - neutral_prob) * decay_factor
        else:
            # No evidence present - use neutral probabilities
            p_t = 0.5
            p_f = 0.5

        # Clamp probabilities to avoid log(0) or log(1)
        p_t = clamp_probability(p_t)
        p_f = clamp_probability(p_f)

        log_p_t = math.log(p_t)
        log_p_f = math.log(p_f)
        contribution_true = log_p_t * entity.weight
        contribution_false = log_p_f * entity.weight

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

    return true_prob / total_prob


def combine_priors(
    area_prior: float, time_prior: float, time_weight: float = 0.2
) -> float:
    """Combine area prior and time prior using weighted averaging in logit space.

    Args:
        area_prior: Base prior probability of occupancy for this area
        time_prior: Time-based modifier for the prior
        time_weight: Weight given to time_prior (0.0 to 1.0, default: 0.2)

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

    # Clamp other inputs to valid ranges
    area_prior = clamp_probability(area_prior)
    time_weight = max(0.0, min(1.0, time_weight))

    area_weight = 1.0 - time_weight

    # Convert to logit space for better interpolation
    def prob_to_logit(p: float) -> float:
        return math.log(p / (1 - p))

    def logit_to_prob(logit: float) -> float:
        return 1 / (1 + math.exp(-logit))

    # Interpolate in logit space for more principled combination
    area_logit = prob_to_logit(area_prior)
    time_logit = prob_to_logit(time_prior)

    # Weighted combination in logit space
    combined_logit = area_weight * area_logit + time_weight * time_logit
    combined_prior = logit_to_prob(combined_logit)

    return clamp_probability(combined_prior)


# ────────────────────────────────────── Coordinator Utilities ───────────────────────────


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
