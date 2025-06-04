"""Prior probability calculations for Area Occupancy Detection."""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, TypedDict

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.recorder import get_instance
from sqlalchemy.exc import SQLAlchemyError

from ..const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from ..utils import validate_datetime, validate_prior, validate_prob
from .entity_type import EntityType
from ..coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str

class PriorType(StrEnum):
    """Prior type."""

    ENTITY = "entity"
    ENTITY_TYPE = "entity_type"


@dataclass
class Prior:
    """Holds prior probability data for an entity or type, and calculates it."""

    prior: float
    prob_given_true: float
    prob_given_false: float
    last_updated: datetime
    type: PriorType

    def __post_init__(self):
        """Validate properties after initialization."""
        self.prior = validate_prior(self.prior)
        self.prob_given_true = validate_prob(self.prob_given_true)
        self.prob_given_false = validate_prob(self.prob_given_false)
        self.last_updated = validate_datetime(self.last_updated)


class PriorManager:
    """Manages prior probability calculations."""

    def __init__(self, coordinator: AreaOccupancyCoordinator):
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

    async def calculate(
        self,
        hass: HomeAssistant,
        entity_type: EntityType,
        prior_type: PriorType,
        primary_sensor: str,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> "Prior":
        """Calculate learned priors for a given entity."""
        if end_time <= start_time:
            _LOGGER.error("End time must be after start time")
            raise ValueError("End time must be after start time")

        _LOGGER.debug(
            "Calculating prior for entity: %s",
            entity_id,
        )

        # Use default priors as fallback values
        fallback_prior = Prior(
            prob_given_true=entity_type["prob_true"],
            prob_given_false=entity_type["prob_false"],
            prior=entity_type["prior"],
            last_updated=validate_datetime(None),
            type=prior_type,
        )

        # Check if we have a cached prior that's still valid
        cached_prior = self.get_prior(entity_id)
        if cached_prior and cached_prior.last_updated > start_time:
            _LOGGER.debug("Using cached prior for %s", entity_id)
            return cached_prior

        is_primary = entity_id == primary_sensor

        try:
            if is_primary:
                primary_states = await self._get_states_from_recorder(
                    hass, entity_id, start_time, end_time
                )
                entity_states = primary_states
            else:
                try:
                    primary_states, entity_states = await asyncio.gather(
                        self._get_states_from_recorder(
                            hass, primary_sensor, start_time, end_time
                        ),
                        self._get_states_from_recorder(
                            hass, entity_id, start_time, end_time
                        ),
                    )
                except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
                    _LOGGER.warning(
                        "Could not fetch states for prior calculation (%s): %s. Using defaults",
                        entity_id,
                        err,
                    )
                    return fallback_prior
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

        prob_given_true = self._calculate_conditional_probability_with_intervals(
            entity_id,
            entity_intervals,
            {primary_sensor: primary_intervals},
            STATE_ON,
            entity_type["active_states"],
        )
        prob_given_false = self._calculate_conditional_probability_with_intervals(
            entity_id,
            entity_intervals,
            {primary_sensor: primary_intervals},
            STATE_OFF,
            entity_type["active_states"],
        )

        prior = self._calculate_prior_probability(
            prob_given_true, prob_given_false, primary_intervals
        )

        calculated_prior = Prior(
            prior=prior,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            last_updated=validate_datetime(None),
            type=prior_type,
        )

        # Store the calculated prior
        self.update_prior(entity_id, calculated_prior)

        return calculated_prior

    async def _get_states_from_recorder(
        self,
        hass: HomeAssistant,
        entity_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[State | Dict[str, Any]] | None:
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
    ) -> List[TimeInterval]:
        """Convert state history to time intervals.

        Args:
            states: List of State objects
            start: Start time for analysis
            end: End time for analysis

        Returns:
            List of TimeInterval objects

        """
        intervals: List[TimeInterval] = []
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
        entity_intervals: List[TimeInterval],
        motion_intervals_by_sensor: Dict[str, List[TimeInterval]],
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
        # Get motion intervals for the primary sensor
        motion_intervals = motion_intervals_by_sensor.get(entity_id, [])

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
        entity_active_duration = 0
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
        prob_given_true: float,
        prob_given_false: float,
        primary_intervals: List[TimeInterval],
    ) -> float:
        """Calculate prior probability using Bayes' theorem.

        Args:
            prob_given_true: P(sensor active | area occupied)
            prob_given_false: P(sensor active | area not occupied)
            primary_intervals: List of time intervals for primary sensor

        Returns:
            Prior probability value

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

        # Calculate P(area occupied)
        p_occupied = occupied_duration / total_duration

        # Calculate prior using Bayes' theorem
        prior = (prob_given_true * p_occupied) / (
            (prob_given_true * p_occupied) + (prob_given_false * (1 - p_occupied))
        )

        return max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
