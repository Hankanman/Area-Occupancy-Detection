"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, TypedDict

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util

from .const import (
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PROBABILITY,
    MIN_WEIGHT,
    ROUNDING_PRECISION,
)

if TYPE_CHECKING:
    from .data.entity import Entity

_LOGGER = logging.getLogger(__name__)


class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str


# ──────────────────────────────────── History Utilities ──────────────────────────────────
async def get_states_from_recorder(
    hass: HomeAssistant, entity_id: str, start_time: datetime, end_time: datetime
) -> list[State | dict[str, Any]] | None:
    """Fetch states history from recorder.

    Args:
        hass: Home Assistant instance
        entity_id: Entity ID to fetch history for
        start_time: Start time window
        end_time: End time window

    Returns:
        List of states or minimal state dicts if successful, None if error occurred

    Raises:
        HomeAssistantError: If recorder access fails
        SQLAlchemyError: If database query fails

    """
    _LOGGER.debug("Fetching states: %s [%s -> %s]", entity_id, start_time, end_time)

    # Check if recorder is available
    recorder = get_instance(hass)
    if recorder is None:
        _LOGGER.debug("Recorder not available for %s", entity_id)
        return None

    try:
        states = await recorder.async_add_executor_job(
            lambda: get_significant_states(
                hass,
                start_time,
                end_time,
                [entity_id],
                minimal_response=False,  # Must be false to include last_changed attribute
            )
        )

        entity_states = states.get(entity_id) if states else None

        if entity_states:
            _LOGGER.debug("Found %d states for %s", len(entity_states), entity_id)
        else:
            _LOGGER.debug("No states found for %s", entity_id)

    except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
        _LOGGER.error("Error getting states for %s: %s", entity_id, err)
        # Re-raise the exception as documented, let the caller handle fallback
        raise

    else:
        return entity_states


async def states_to_intervals(
    states: Sequence[State], start: datetime, end: datetime
) -> list[TimeInterval]:
    """Convert state history to time intervals.

    Args:
        states: List of State objects
        start: Start time for analysis
        end: End time for analysis

    Returns:
        List of TimeInterval objects

    """
    intervals: list[TimeInterval] = []
    if not states:
        return intervals

    # Sort states chronologically
    sorted_states = sorted(states, key=lambda x: x.last_changed)

    # Determine the state that was active at the start time
    current_state = sorted_states[0].state
    for state in sorted_states:
        if state.last_changed <= start:
            current_state = state.state
        else:
            break

    current_time = start

    # Build intervals between state changes
    for state in sorted_states:
        if state.last_changed <= start:
            continue
        if state.last_changed > end:
            break
        intervals.append(
            TimeInterval(
                start=current_time, end=state.last_changed, state=current_state
            )
        )
        current_state = state.state
        current_time = state.last_changed

    # Final interval until end
    intervals.append(TimeInterval(start=current_time, end=end, state=current_state))

    return intervals


# ───────────────────────────────────────── Validation ────────────────────────
def validate_prob(value: complex) -> float:
    """Validate probability value, handling complex numbers."""
    # Handle complex numbers by taking the real part
    if isinstance(value, complex):
        _LOGGER.warning(
            "Complex number detected in probability calculation: %s, using real part",
            value,
        )
        value = value.real

    # Ensure it's a valid float
    if not isinstance(value, (int, float)) or not (-1e10 < value < 1e10):
        _LOGGER.warning("Invalid probability value: %s, using default", value)
        return 0.5

    return max(0.001, min(float(value), 1.0))


def validate_prior(value: float) -> float:
    """Validate prior probability value."""
    return max(0.000001, min(value, 0.999999))


def validate_datetime(value: datetime | None) -> datetime:
    """Validate datetime value."""
    return value if isinstance(value, datetime) else dt_util.utcnow()


def validate_weight(value: float) -> float:
    """Validate weight value."""
    return max(MIN_WEIGHT, min(value, MAX_WEIGHT))


def validate_decay_factor(value: float) -> float:
    """Validate decay factor value."""
    return max(MIN_PROBABILITY, min(value, MAX_PROBABILITY))


def format_float(value: float) -> float:
    """Format float value."""
    return round(float(value), ROUNDING_PRECISION)


def format_percentage(value: float) -> str:
    """Format float value as percentage."""
    return f"{value * 100:.2f}%"


def apply_decay(
    prob_given_true: float, prob_given_false: float, decay_factor: float
) -> tuple[float, float]:
    """Apply decay factor to likelihood probabilities.

    This maintains mathematical equivalence with applying decay as an exponent
    to the Bayes factor in the original bayesian_probability function.

    Args:
        prob_given_true: Original probability given true
        prob_given_false: Original probability given false
        decay_factor: Decay factor (0.0 = full decay, 1.0 = no decay)

    Returns:
        Tuple of (effective_prob_given_true, effective_prob_given_false)

    """
    if decay_factor == 1.0:
        return prob_given_true, prob_given_false

    if decay_factor == 0.0:
        # Full decay - return neutral probabilities
        return 0.5, 0.5

    # Ensure inputs are in valid range
    prob_given_true = max(0.001, min(prob_given_true, 0.999))
    prob_given_false = max(0.001, min(prob_given_false, 0.999))

    # Calculate the original bayes factor
    original_bf = prob_given_true / prob_given_false

    # Apply decay to the bayes factor (this is what bayesian_probability was doing)
    decayed_bf = original_bf**decay_factor

    # Calculate geometric mean to preserve overall magnitude
    geo_mean = (prob_given_true * prob_given_false) ** 0.5

    # Calculate new probabilities that give the decayed bayes factor
    # p_t_eff / p_f_eff = decayed_bf
    # p_t_eff * p_f_eff = geo_mean^2 (preserve geometric mean)
    # Solving: p_t_eff = geo_mean * sqrt(decayed_bf), p_f_eff = geo_mean / sqrt(decayed_bf)

    sqrt_bf = decayed_bf**0.5
    p_true_eff = geo_mean * sqrt_bf
    p_false_eff = geo_mean / sqrt_bf

    # Ensure probabilities are in valid range
    p_true_eff = max(0.001, min(0.999, p_true_eff))
    p_false_eff = max(0.001, min(0.999, p_false_eff))

    return p_true_eff, p_false_eff


EPS = 1e-12


# ────────────────────────────────────── Core Bayes ───────────────────────────
def bayesian_probability(
    *,  # keyword-only → prevents accidental positional mix-ups
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    evidence: bool | None,
) -> float:
    """Pure Bayesian probability update.

    This function now focuses solely on Bayesian calculation.
    Decay should be applied to prob_given_true and prob_given_false before calling this.

    Args:
        prior: Prior probability
        prob_given_true: Probability of evidence given true (decay already applied if needed)
        prob_given_false: Probability of evidence given false (decay already applied if needed)
        evidence: Evidence (True/False/None)

    Returns:
        Posterior probability

    """
    if evidence is None:
        return prior

    # Validate inputs first
    prob_given_true = max(0.001, min(prob_given_true, 0.999))
    prob_given_false = max(0.001, min(prob_given_false, 0.999))
    prior = max(0.001, min(prior, 0.999))

    # Calculate Bayes factor
    bayes_factor = (
        (prob_given_true + EPS) / (prob_given_false + EPS)
        if evidence
        else (1 - prob_given_true + EPS) / (1 - prob_given_false + EPS)
    )

    # Ensure bayes_factor is positive to avoid complex numbers
    bayes_factor = max(EPS, bayes_factor)

    # Calculate posterior odds
    odds = prior / (1.0 - prior + EPS)
    posterior_odds = odds * bayes_factor

    # Return posterior probability
    result = posterior_odds / (1.0 + posterior_odds)
    return validate_prob(result)


# ─────────────────────────────── Area-level fusion ───────────────────────────
def overall_probability(entities: dict[str, Entity], prior: float) -> float:
    """Combine weighted posteriors from active/decaying sensors."""

    contributing_entities = [
        e for e in entities.values() if e.evidence or e.decay.is_decaying
    ]

    if not contributing_entities:
        return validate_prob(prior)

    product = 1.0
    for e in contributing_entities:
        # Use Entity's effective probabilities (decay already applied)
        posterior = bayesian_probability(
            prior=prior,
            prob_given_true=e.effective_prob_given_true,
            prob_given_false=e.effective_prob_given_false,
            evidence=True,
        )
        product *= 1 - posterior

    return validate_prob(1 - product)
