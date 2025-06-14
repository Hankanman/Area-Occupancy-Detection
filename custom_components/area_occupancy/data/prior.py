"""Prior probability calculations for Area Occupancy Detection."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, TypedDict

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
)
from ..utils import validate_datetime, validate_prior, validate_prob

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from .entity import Entity

_LOGGER = logging.getLogger(__name__)


class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str


@dataclass
class Prior:
    """Holds learned prior probability data for an entity or type.

    This class stores historically learned probabilities but does not update
    them based on real-time sensor data. All probability calculations should
    be handled by the Probability class using Bayesian inference.

    The Prior class is responsible for:
    - Storing learned historical probabilities
    - Calculating priors from historical sensor data
    - Providing stable baseline probabilities for Bayesian calculations
    """

    prior: float
    prob_given_true: float
    prob_given_false: float
    last_updated: datetime

    def __post_init__(self):
        """Validate properties after initialization."""
        self.prior = validate_prior(self.prior)
        self.prob_given_true = validate_prob(self.prob_given_true)
        self.prob_given_false = validate_prob(self.prob_given_false)
        self.last_updated = validate_datetime(self.last_updated)

    def to_dict(self) -> dict[str, Any]:
        """Convert prior to dictionary for storage."""
        return {
            "prior": self.prior,
            "prob_given_true": self.prob_given_true,
            "prob_given_false": self.prob_given_false,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prior":
        """Create prior from dictionary."""
        last_updated = validate_datetime(dt_util.parse_datetime(data["last_updated"]))

        return cls(
            prior=data["prior"],
            prob_given_true=data["prob_given_true"],
            prob_given_false=data["prob_given_false"],
            last_updated=last_updated,
        )


class PriorManager:
    """Manages prior probability calculations."""

    def __init__(self, coordinator: "AreaOccupancyCoordinator"):
        """Initialize the prior manager."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self._priors: dict[str, Prior] = {}

    @property
    def priors(self) -> dict[str, Prior]:
        """Get all stored priors."""
        return self._priors

    def get_prior(self, entity_id: str) -> Prior | None:
        """Get the prior for an entity."""
        return self._priors.get(entity_id)

    def update_prior(self, entity_id: str, prior: Prior) -> None:
        """Update the prior for an entity."""
        self._priors[entity_id] = prior

    def remove_prior(self, entity_id: str) -> None:
        """Remove the prior for an entity."""
        self._priors.pop(entity_id, None)

    def clear_priors(self) -> None:
        """Clear all stored priors."""
        self._priors.clear()

    async def update_all_entity_priors(self) -> int:
        """Update learned priors for all entities in the coordinator.

        Args:
            history_period: Number of days of history to analyze (defaults to config value)

        Returns:
            Number of entities successfully updated

        Raises:
            ValueError: If no primary occupancy sensor is configured

        """

        _LOGGER.info(
            "Updating learned priors for area %s: analyzing %d days of history",
            self.coordinator.config.name,
            self.coordinator.config.history.period,
        )

        # Update priors for all entities
        updated_count = 0
        for entity in self.coordinator.entities.entities.values():
            try:
                prior = await self.calculate(entity)

                # Update the entity's prior
                entity.prior = prior
                updated_count += 1

                _LOGGER.debug(
                    "Updated prior for %s: prior=%.3f, prob_true=%.3f, prob_false=%.3f",
                    entity.entity_id,
                    prior.prior,
                    prior.prob_given_true,
                    prior.prob_given_false,
                )

            except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
                _LOGGER.warning(
                    "Failed to update prior for %s: %s", entity.entity_id, err
                )
                continue

        # Learn and update entity type priors from all entities
        self.coordinator.entity_types.learn_from_entities(
            self.coordinator.entities.entities
        )

        _LOGGER.info(
            "Completed learned priors update for area %s: updated %d/%d entities",
            self.coordinator.config.name,
            updated_count,
            len(self.coordinator.entities.entities),
        )

        return updated_count

    async def calculate(
        self,
        entity: "Entity",
    ) -> "Prior":
        """Calculate learned priors for a given entity."""

        hass: HomeAssistant = self.coordinator.hass
        entity_id = entity.entity_id
        entity_type = entity.type
        if self.coordinator.config.sensors.primary_occupancy:
            primary_sensor = self.coordinator.config.sensors.primary_occupancy
        else:
            raise ValueError("No primary occupancy sensor configured")
        if self.coordinator.config.start_time:
            start_time = self.coordinator.config.start_time
        else:
            raise ValueError("No start time configured")
        if self.coordinator.config.end_time:
            end_time = self.coordinator.config.end_time

        if end_time <= start_time:
            _LOGGER.error("End time must be after start time")
            raise ValueError("End time must be after start time")

        _LOGGER.debug(
            "Calculating prior for entity: %s",
            entity_id,
        )

        # Use default priors as fallback values
        fallback_prior = Prior(
            prob_given_true=entity_type.prob_true,
            prob_given_false=entity_type.prob_false,
            prior=entity_type.prior,
            last_updated=validate_datetime(None),
        )

        # Check if we have a cached prior that's still valid
        cached_prior = self.get_prior(entity_id)
        if cached_prior and cached_prior.last_updated > start_time:
            _LOGGER.debug("Using cached prior for %s", entity_id)
            return cached_prior

        is_primary = entity_id == primary_sensor

        # For the primary sensor, we can only calculate the prior (occupancy rate),
        # but not meaningful conditional probabilities since P(sensor|sensor) doesn't make sense
        if is_primary:
            try:
                primary_states = await self._get_states_from_recorder(
                    hass, entity_id, start_time, end_time
                )
                if not primary_states:
                    _LOGGER.warning(
                        "No states found for primary sensor (%s). Using defaults",
                        entity_id,
                    )
                    return fallback_prior

                primary_state_objects = [
                    state for state in primary_states if isinstance(state, State)
                ]
                if not primary_state_objects:
                    return fallback_prior

                primary_intervals = await self._states_to_intervals(
                    primary_state_objects, start_time, end_time
                )

                # For primary sensor, just calculate the occupancy rate as prior
                occupied_duration = sum(
                    (interval["end"] - interval["start"]).total_seconds()
                    for interval in primary_intervals
                    if interval["state"] == STATE_ON
                )
                total_duration = sum(
                    (interval["end"] - interval["start"]).total_seconds()
                    for interval in primary_intervals
                )

                if total_duration == 0:
                    return fallback_prior

                learned_prior_value = max(
                    MIN_PROBABILITY,
                    min(occupied_duration / total_duration, MAX_PROBABILITY),
                )

                # Create learned prior with entity type defaults for conditional probabilities
                learned_prior = Prior(
                    prior=learned_prior_value,
                    prob_given_true=entity_type.prob_true,
                    prob_given_false=entity_type.prob_false,
                    last_updated=validate_datetime(None),
                )

                # Validate learned prior and log comparison
                _LOGGER.debug(
                    "Primary sensor prior calculation for %s: learned=%.3f, default=%.3f",
                    entity_id,
                    learned_prior.prior,
                    fallback_prior.prior,
                )

                # If learned prior is suspiciously low (< 0.05), prefer defaults
                # This prevents the system from getting stuck with unrealistic low priors
                if learned_prior.prior < MIN_PRIOR:
                    _LOGGER.warning(
                        "Learned prior %.3f for primary sensor %s is very low, using default %.3f instead",
                        learned_prior.prior,
                        entity_id,
                        fallback_prior.prior,
                    )
                    return fallback_prior

                self.update_prior(entity_id, learned_prior)

            except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
                _LOGGER.warning(
                    "Could not fetch states for primary sensor (%s): %s. Using defaults",
                    entity_id,
                    err,
                )
                return fallback_prior
            else:
                return learned_prior

        # For non-primary sensors, calculate conditional probabilities
        try:
            primary_states, entity_states = await asyncio.gather(
                self._get_states_from_recorder(
                    hass, primary_sensor, start_time, end_time
                ),
                self._get_states_from_recorder(hass, entity_id, start_time, end_time),
            )
        except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
            _LOGGER.warning(
                "Could not fetch states for prior calculation (%s): %s. Using defaults",
                entity_id,
                err,
            )
            return fallback_prior

        if not primary_states or not entity_states:
            _LOGGER.warning(
                "No states found for prior calculation (%s). Using defaults",
                entity_id,
            )
            return fallback_prior

        primary_state_objects = [
            state for state in primary_states if isinstance(state, State)
        ]
        entity_state_objects = [
            state for state in entity_states if isinstance(state, State)
        ]

        if not primary_state_objects or not entity_state_objects:
            _LOGGER.warning(
                "No valid states found for prior calculation (%s). Using defaults",
                entity_id,
            )
            return fallback_prior

        primary_intervals = await self._states_to_intervals(
            primary_state_objects, start_time, end_time
        )
        entity_intervals = await self._states_to_intervals(
            entity_state_objects, start_time, end_time
        )

        learned_prob_given_true = (
            self._calculate_conditional_probability_with_intervals(
                entity_id,
                entity_intervals,
                {primary_sensor: primary_intervals},
                STATE_ON,
                entity_type.active_states or [],
            )
        )
        learned_prob_given_false = (
            self._calculate_conditional_probability_with_intervals(
                entity_id,
                entity_intervals,
                {primary_sensor: primary_intervals},
                STATE_OFF,
                entity_type.active_states or [],
            )
        )

        learned_prior_value = self._calculate_prior_probability(primary_intervals)

        learned_prior = Prior(
            prior=learned_prior_value,
            prob_given_true=learned_prob_given_true,
            prob_given_false=learned_prob_given_false,
            last_updated=validate_datetime(None),
        )

        # Log the learned vs default values for debugging
        _LOGGER.debug(
            "Prior calculation for %s: learned=%.3f, default=%.3f, prob_true: learned=%.3f/default=%.3f, prob_false: learned=%.3f/default=%.3f",
            entity_id,
            learned_prior.prior,
            fallback_prior.prior,
            learned_prior.prob_given_true,
            fallback_prior.prob_given_true,
            learned_prior.prob_given_false,
            fallback_prior.prob_given_false,
        )

        # If learned prior is suspiciously low (< 0.05), prefer defaults
        # This prevents the system from getting stuck with unrealistic low priors
        if learned_prior.prior < 0.05:
            _LOGGER.warning(
                "Learned prior %.3f for %s is very low, using default %.3f instead",
                learned_prior.prior,
                entity_id,
                fallback_prior.prior,
            )
            return fallback_prior

        # Store the calculated prior
        self.update_prior(entity_id, learned_prior)

        return learned_prior

    async def _get_states_from_recorder(
        self,
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
        try:
            _LOGGER.debug(
                "Fetching states: %s [%s -> %s]",
                entity_id,
                start_time,
                end_time,
            )

            states = await get_instance(
                hass
            ).async_add_executor_job(
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
            # Propagate error to be handled by the coordinator
            raise HomeAssistantError(f"Recorder error for {entity_id}: {err}") from err
        else:
            return entity_states

    @staticmethod
    async def _states_to_intervals(
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

    @staticmethod
    def _calculate_conditional_probability_with_intervals(
        entity_id: str,
        entity_intervals: list[TimeInterval],
        motion_intervals_by_sensor: dict[str, list[TimeInterval]],
        motion_state_filter: str,  # Should be STATE_ON or STATE_OFF
        entity_active_states: list[str],
    ) -> float:
        """Calculate conditional probability using time intervals.

        Args:
            entity_id: Entity ID being analyzed
            entity_intervals: List of time intervals for the entity
            motion_intervals_by_sensor: Dict of motion sensor intervals
            motion_state_filter: State to filter motion intervals by
            entity_active_states: Set of active entity states

        Returns:
            Conditional probability value

        """
        # Get motion intervals for the primary sensor (first key in the dict)
        motion_intervals = next(iter(motion_intervals_by_sensor.values()), [])

        # Filter motion intervals by state
        filtered_motion_intervals = [
            interval
            for interval in motion_intervals
            if interval["state"] == motion_state_filter
        ]

        if not filtered_motion_intervals:
            return (
                DEFAULT_PROB_GIVEN_TRUE
                if motion_state_filter == STATE_ON
                else DEFAULT_PROB_GIVEN_FALSE
            )

        # Calculate total duration of filtered motion intervals
        total_motion_duration = sum(
            (interval["end"] - interval["start"]).total_seconds()
            for interval in filtered_motion_intervals
        )

        if total_motion_duration == 0:
            return (
                DEFAULT_PROB_GIVEN_TRUE
                if motion_state_filter == STATE_ON
                else DEFAULT_PROB_GIVEN_FALSE
            )

        # Calculate entity state durations during filtered motion intervals
        entity_active_duration = 0.0
        for motion_interval in filtered_motion_intervals:
            for entity_interval in entity_intervals:
                # Calculate overlap between intervals
                overlap_start = max(motion_interval["start"], entity_interval["start"])
                overlap_end = min(motion_interval["end"], entity_interval["end"])
                if overlap_end > overlap_start:
                    # Check if entity state is considered active
                    if entity_interval["state"] in set(entity_active_states):
                        entity_active_duration += (
                            overlap_end - overlap_start
                        ).total_seconds()

        # Calculate conditional probability
        probability = entity_active_duration / total_motion_duration
        return max(MIN_PROBABILITY, min(probability, MAX_PROBABILITY))

    @staticmethod
    def _calculate_prior_probability(
        primary_intervals: list[TimeInterval],
    ) -> float:
        """Calculate prior probability from historical occupancy rate.

        This calculates the simple historical prior probability (occupancy rate)
        from the primary sensor intervals. This is NOT a Bayesian calculation
        but rather a straightforward frequency calculation.

        Args:
            primary_intervals: List of time intervals for primary sensor

        Returns:
            Prior probability value (historical occupancy rate)

        """
        # Calculate total duration of occupied and unoccupied states
        occupied_duration = sum(
            (interval["end"] - interval["start"]).total_seconds()
            for interval in primary_intervals
            if interval["state"] == STATE_ON
        )
        total_duration = sum(
            (interval["end"] - interval["start"]).total_seconds()
            for interval in primary_intervals
        )

        if total_duration == 0:
            return 0.5  # Default to 0.5 if no data

        # Calculate P(area occupied) - this IS the prior
        p_occupied = occupied_duration / total_duration

        # The prior is simply the historical occupancy rate
        prior = p_occupied

        return max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
