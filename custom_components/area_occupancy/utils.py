"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
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

# Interval filtering thresholds to exclude anomalous data
# Exclude intervals shorter than 10 seconds (false triggers)
MIN_INTERVAL_SECONDS = 10
# Exclude intervals longer than 13 hours (stuck sensors)
MAX_INTERVAL_SECONDS = 13 * 3600
# States to exclude from intervals
INVALID_STATES = {"unknown", "unavailable", None, "", "NaN"}


class Interval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime


class StateInterval(Interval):
    """Time interval with time information."""

    state: str
    entity_id: str


class PriorInterval:
    """Time interval with prior information."""

    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    prior: float | None

    def __init__(
        self,
        start_hour: int,
        start_minute: int,
        end_hour: int,
        end_minute: int,
        prior: float | None,
    ) -> None:
        """Initialize prior interval."""
        self.start_hour = start_hour
        self.start_minute = start_minute
        self.end_hour = end_hour
        self.end_minute = end_minute
        self.prior = prior

    def to_interval(self) -> Interval:
        """Convert to interval."""
        return Interval(
            start=datetime(
                datetime.now().year,
                datetime.now().month,
                datetime.now().day,
                self.start_hour,
                self.start_minute,
            ),
            end=datetime(
                datetime.now().year,
                datetime.now().month,
                datetime.now().day,
                self.end_hour,
                self.end_minute,
            ),
        )


class LikelihoodInterval(Interval):
    """Time interval with likelihood information."""

    prob_given_true: float
    prob_given_false: float


def is_valid_state(state: Any) -> bool:
    """Check if a state is valid."""
    return state not in INVALID_STATES


async def init_times_of_day(hass: HomeAssistant) -> list[PriorInterval]:
    """Initialize times of day for a given entity as 24 hourly intervals."""
    return [
        PriorInterval(
            start_hour=hour,
            start_minute=minute,
            end_hour=hour if minute == 0 else (hour + 1) % 24,
            end_minute=29 if minute == 0 else 59,
            prior=None,
        )
        for hour in range(24)
        for minute in (0, 30)
    ]


# ──────────────────────────────────── Time-Based Prior Utilities ──────────────────────────────────


def datetime_to_time_slot(dt: datetime) -> tuple[int, int]:
    """Convert datetime to day of week and time slot.

    Args:
        dt: Datetime to convert

    Returns:
        Tuple of (day_of_week, time_slot) where:
        - day_of_week: 0=Monday, 6=Sunday
        - time_slot: 0-47 (30-minute intervals, 00:00-00:29=0, 00:30-00:59=1, etc.)

    """
    # weekday() already returns Monday=0, Tuesday=1, ..., Sunday=6
    day_of_week = dt.weekday()

    # Calculate time slot (0-47 for 30-minute intervals)
    hour = dt.hour
    minute = dt.minute
    time_slot = hour * 2 + (1 if minute >= 30 else 0)

    return day_of_week, time_slot


def time_slot_to_datetime_range(
    day_of_week: int, time_slot: int, base_date: datetime | None = None
) -> tuple[datetime, datetime]:
    """Convert day of week and time slot to datetime range.

    Args:
        day_of_week: 0=Monday, 6=Sunday
        time_slot: 0-47 (30-minute intervals)
        base_date: Base date to use (defaults to current date)

    Returns:
        Tuple of (start_datetime, end_datetime) for the 30-minute slot

    """
    if base_date is None:
        base_date = dt_util.utcnow()

    # Calculate start hour and minute
    start_hour = time_slot // 2
    start_minute = (time_slot % 2) * 30

    # Calculate end hour and minute
    if start_minute == 30:
        end_hour = (start_hour + 1) % 24
        end_minute = 0
    else:
        end_hour = start_hour
        end_minute = 30

    # Find the target day of week
    current_weekday = base_date.weekday()  # weekday() already returns Monday=0
    days_diff = (day_of_week - current_weekday) % 7

    # Calculate target date
    target_date = base_date.date() + timedelta(days=days_diff)

    # Create start and end datetimes
    start_dt = datetime.combine(
        target_date, datetime.min.time().replace(hour=start_hour, minute=start_minute)
    )
    end_dt = datetime.combine(
        target_date, datetime.min.time().replace(hour=end_hour, minute=end_minute)
    )

    return start_dt, end_dt


def get_current_time_slot() -> tuple[int, int]:
    """Get the current day of week and time slot.

    Returns:
        Tuple of (day_of_week, time_slot) for current time

    """
    return datetime_to_time_slot(dt_util.utcnow())


def get_time_slot_name(day_of_week: int, time_slot: int) -> str:
    """Get a human-readable name for a time slot.

    Args:
        day_of_week: 0=Monday, 6=Sunday
        time_slot: 0-47 (30-minute intervals)

    Returns:
        Human-readable time slot name (e.g., "Monday 13:00-13:29")

    """
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    start_hour = time_slot // 2
    start_minute = (time_slot % 2) * 30

    if start_minute == 30:
        end_hour = (start_hour + 1) % 24
        end_minute = 0
    else:
        end_hour = start_hour
        end_minute = 30

    return f"{day_names[day_of_week]} {start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"


def get_all_time_slots() -> list[tuple[int, int]]:
    """Get all possible day of week and time slot combinations.

    Returns:
        List of (day_of_week, time_slot) tuples for all 336 slots (7 days × 48 slots)

    """
    return [(day, slot) for day in range(7) for slot in range(48)]


# ──────────────────────────────────── History Utilities ──────────────────────────────────
async def get_intervals_from_recorder(
    hass: HomeAssistant, entity_id: str, start_time: datetime, end_time: datetime
) -> list[StateInterval]:
    """Get intervals from HA recorder."""
    _LOGGER.debug(
        "Getting intervals from recorder for %s from %s to %s",
        entity_id,
        start_time,
        end_time,
    )

    states = await get_states_from_recorder(hass, entity_id, start_time, end_time)

    _LOGGER.debug(
        "Got %d states from recorder for %s", len(states) if states else 0, entity_id
    )

    if not states:
        _LOGGER.debug("No states found in recorder for %s", entity_id)
        return []

    # Filter to only State objects and exclude unavailable/unknown states
    state_objects: list[State] = [
        s for s in states if isinstance(s, State) and is_valid_state(s.state)
    ]

    _LOGGER.debug(
        "Filtered to %d valid State objects for %s", len(state_objects), entity_id
    )

    intervals = await states_to_intervals(state_objects, start_time, end_time)

    _LOGGER.debug("Converted to %d intervals for %s", len(intervals), entity_id)

    return filter_intervals(intervals)


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
        _LOGGER.debug("Executing recorder query for %s", entity_id)
        states = await recorder.async_add_executor_job(
            lambda: get_significant_states(
                hass,
                start_time,
                end_time,
                [entity_id],
                minimal_response=False,  # Must be false to include last_changed attribute
            )
        )

        _LOGGER.debug("Recorder query completed for %s, processing results", entity_id)
        entity_states = states.get(entity_id) if states else None

        if entity_states:
            _LOGGER.debug("Found %d states for %s", len(entity_states), entity_id)
        else:
            _LOGGER.debug("No states found for %s in recorder query result", entity_id)

    except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
        _LOGGER.error("Error getting states for %s: %s", entity_id, err)
        # Re-raise the exception as documented, let the caller handle fallback
        raise

    else:
        return entity_states


async def states_to_intervals(
    states: Sequence[State], start: datetime, end: datetime
) -> list[StateInterval]:
    """Convert state history to time intervals.

    Args:
        states: List of State objects
        start: Start time for analysis
        end: End time for analysis

    Returns:
        List of StateInterval objects

    """
    intervals: list[StateInterval] = []
    if not states:
        return intervals

    # Filter out any invalid states that might have slipped through
    valid_states = [state for state in states if is_valid_state(state.state)]

    if not valid_states:
        _LOGGER.debug("No valid states after filtering unavailable/unknown")
        return intervals

    # Sort states chronologically
    sorted_states = sorted(valid_states, key=lambda x: x.last_changed)

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

        # Only create intervals for valid states
        if is_valid_state(current_state):
            intervals.append(
                StateInterval(
                    start=current_time,
                    end=state.last_changed,
                    state=current_state,
                    entity_id=state.entity_id,
                )
            )
        current_state = state.state
        current_time = state.last_changed

    # Final interval until end (only if state is valid)
    if is_valid_state(current_state):
        intervals.append(
            StateInterval(
                start=current_time,
                end=end,
                state=current_state,
                entity_id=state.entity_id,
            )
        )

    return intervals


def filter_intervals(
    intervals: list[StateInterval],
) -> list[StateInterval]:
    """Filter intervals to remove anomalous data and invalid states.

    - For binary sensors ("on" state): filter out intervals that are too short or too long.
    - For numeric sensors and other valid states: only remove intervals with invalid states, do not filter by duration.
    """
    # Remove intervals with invalid states
    valid_intervals = [
        interval for interval in intervals if is_valid_state(interval["state"])
    ]

    filtered_intervals = []
    for interval in valid_intervals:
        state = interval["state"]
        # Only apply duration filtering to binary "on" state
        if state == "on":
            duration_seconds = (interval["end"] - interval["start"]).total_seconds()
            if MIN_INTERVAL_SECONDS <= duration_seconds <= MAX_INTERVAL_SECONDS:
                filtered_intervals.append(interval)
            # else: skip this interval (too short/long)
        else:
            # For all other states (including numerics), keep as long as state is valid
            filtered_intervals.append(interval)

    return filtered_intervals


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
    return posterior_odds / (1.0 + posterior_odds)


# ─────────────────────────────── Area-level fusion ───────────────────────────
# Not used
def complementary_probability(entities: dict[str, Entity], prior: float) -> float:
    """Calculate the complementary probability.

    This function computes the probability that at least ONE entity provides
    evidence for occupancy, using the complement rule:
    P(at least one) = 1 - product(P(not each)). For each contributing entity,
    a Bayesian update is performed assuming evidence is True (or decaying),
    and the complement of the posterior is multiplied across all such entities.
    Is not affected by the order of the entities.
    Does not consider negative evidence.

    Args:
        entities: Dictionary of Entity objects to consider.
        prior: The prior probability of occupancy.

    Returns:
        The combined probability that at least one contributing entity
        indicates occupancy, after Bayesian updates and decay are applied.

    """

    contributing_entities = [
        e for e in entities.values() if e.evidence or e.decay.is_decaying
    ]

    product = 1.0
    for e in contributing_entities:
        posterior = bayesian_probability(
            prior=prior,
            prob_given_true=e.likelihood.prob_given_true,
            prob_given_false=e.likelihood.prob_given_false,
            evidence=True,
            decay_factor=e.decay_factor,
        )
        weighted_posterior = posterior * e.type.weight
        product *= 1 - weighted_posterior

    return 1 - product


# Not used
def conditional_probability(entities: dict[str, Entity], prior: float) -> float:
    """Return conditional probability, accounting for entity weights.

    Sequentially update the prior probability by applying Bayes' theorem for each entity,
    using the entity's evidence and likelihoods. The posterior from each step becomes the
    prior for the next entity. Each entity's weight is used to interpolate between the
    previous posterior and the new posterior, so that higher-weight entities have more
    influence on the result.

    Args:
        entities: Dictionary of Entity objects to process.
        prior: Initial prior probability.

    Returns:
        The final posterior probability after all updates.

    """

    posterior = prior
    for e in entities.values():
        # Use effective evidence: True if evidence is True OR if decaying
        effective_evidence = e.evidence or e.decay.is_decaying
        entity_posterior = bayesian_probability(
            prior=posterior,
            prob_given_true=e.likelihood.prob_given_true,
            prob_given_false=e.likelihood.prob_given_false,
            evidence=effective_evidence,
            decay_factor=e.decay_factor,
        )
        # Interpolate between previous posterior and entity_posterior using entity weight
        weight = e.type.weight
        posterior = posterior * (1 - weight) + entity_posterior * weight

    return posterior


def conditional_sorted_probability(entities: dict[str, Entity], prior: float) -> float:
    """Return conditional sorted probability.

    Sequentially update the prior probability by applying Bayes' theorem for each entity,
    using the entity's evidence and likelihoods. The posterior from each step becomes the
    prior for the next entity. This method reflects the effect of each entity's evidence
    (and decay, if applicable) on the overall probability. The entities are sorted by
    evidence status (active first) and then by weight (highest weight first) to ensure
    that the most relevant entities are considered first.

    Args:
        entities: Dictionary of Entity objects to process.
        prior: Initial prior probability.

    Returns:
        The final posterior probability after all updates.

    """

    sorted_entities = sorted(
        entities.values(),
        key=lambda x: (not (x.evidence or x.decay.is_decaying), -x.type.weight),
    )
    posterior = prior
    for e in sorted_entities:
        # Use effective evidence: True if evidence is True OR if decaying
        effective_evidence = e.evidence or e.decay.is_decaying
        entity_posterior = bayesian_probability(
            prior=posterior,
            prob_given_true=e.likelihood.prob_given_true,
            prob_given_false=e.likelihood.prob_given_false,
            evidence=effective_evidence,
            decay_factor=e.decay_factor,
        )
        # Interpolate between previous posterior and entity_posterior using entity weight
        weight = e.type.weight
        posterior = posterior * (1 - weight) + entity_posterior * weight

    return posterior
