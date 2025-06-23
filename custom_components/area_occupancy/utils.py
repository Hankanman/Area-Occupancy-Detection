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
    hass: HomeAssistant,
    entity_id: str,
    start_time: datetime,
    end_time: datetime,
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
    _LOGGER.debug(
        "Fetching states: %s [%s -> %s]",
        entity_id,
        start_time,
        end_time,
    )

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
            _LOGGER.debug(
                "Found %d states for %s",
                len(entity_states),
                entity_id,
            )
        else:
            _LOGGER.debug("No states found for %s", entity_id)

    except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
        _LOGGER.error("Error getting states for %s: %s", entity_id, err)
        # Re-raise the exception as documented, let the caller handle fallback
        raise

    else:
        return entity_states


async def states_to_intervals(
    states: Sequence[State],
    start: datetime,
    end: datetime,
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

    # Sort states by last_changed
    sorted_states = sorted(states, key=lambda x: x.last_changed)

    # Create intervals from states
    for i, state in enumerate(sorted_states):
        interval_start = state.last_changed
        interval_end = (
            sorted_states[i + 1].last_changed if i < len(sorted_states) - 1 else end
        )
        intervals.append(
            TimeInterval(
                start=interval_start,
                end=interval_end,
                state=state.state,
            )
        )

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


EPS = 1e-12


# ────────────────────────────────────── Core Bayes ───────────────────────────
def bayesian_probability(
    *,  # keyword-only → prevents accidental positional mix-ups
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    evidence: bool | None,
    decay_factor: float,  # Remove weight parameter
) -> float:
    """Simplified Bayesian update - weight is already applied in likelihood values.

    Args:
        prior: Prior probability
        prob_given_true: Weighted probability of evidence given true
        prob_given_false: Weighted probability of evidence given false
        evidence: Evidence
        decay_factor: Decay factor

    Returns:
        Posterior probability

    """
    if evidence is None or decay_factor == 0:
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

    # Apply only decay factor (weight already applied in likelihood)
    if decay_factor != 1.0:
        bayes_factor = bayes_factor**decay_factor

    # Calculate posterior odds
    odds = prior / (1.0 - prior + EPS)
    posterior_odds = odds * bayes_factor

    # Return posterior probability
    result = posterior_odds / (1.0 + posterior_odds)
    return validate_prob(result)


# ─────────────────────────────── Area-level fusion ───────────────────────────
def overall_probability(
    entities: dict[str, Entity],
    prior: float,
) -> float:
    """Combine current beliefs of all entities into one room posterior."""
    posterior = prior
    for e in entities.values():
        # Determine evidence and decay handling
        evidence = e.evidence
        decay_factor = 1.0
        if e.decay.is_decaying:
            # A decaying sensor should gradually reduce the probability
            evidence = False
            decay_factor = e.decay.decay_factor

        # Update posterior probability
        posterior = bayesian_probability(
            prior=posterior,
            prob_given_true=e.likelihood.prob_given_true,  # Already weighted
            prob_given_false=e.likelihood.prob_given_false,  # Already weighted
            evidence=evidence,
            decay_factor=decay_factor,
        )

    return validate_prob(posterior)
