"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

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

# ──────────────────────────────────── Time-Based Prior Utilities ──────────────────────────────────


def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware."""
    if dt.tzinfo is None:
        # If naive, assume UTC (Home Assistant stores times in UTC)

        return dt.replace(tzinfo=dt_util.UTC)
    return dt


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


# ───────────────────────────────────────── Validation ────────────────────────
def validate_prob(value: complex) -> float:
    """Validate probability value, handling complex numbers."""
    # Handle complex numbers by taking the real part
    if isinstance(value, complex):
        value = value.real

    # Ensure it's a valid float
    if not isinstance(value, (int, float)) or not (-1e10 < value < 1e10):
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
            prob_given_true=e.prob_given_true,
            prob_given_false=e.prob_given_false,
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
            prob_given_true=e.prob_given_true,
            prob_given_false=e.prob_given_false,
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
            prob_given_true=e.prob_given_true,
            prob_given_false=e.prob_given_false,
            evidence=effective_evidence,
            decay_factor=e.decay_factor,
        )
        # Interpolate between previous posterior and entity_posterior using entity weight
        weight = e.type.weight
        posterior = posterior * (1 - weight) + entity_posterior * weight

    return posterior
