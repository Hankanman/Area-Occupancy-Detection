"""Prior probability calculations for Area Occupancy Detection."""

import asyncio
import logging

_LOGGER = logging.getLogger(__name__)
_LOGGER.debug("Starting imports for prior.py")

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
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
    MAX_PRIOR,
    MAX_PROBABILITY,
    MIN_PRIOR,
    MIN_PROBABILITY,
)

_LOGGER.debug("Imported area_occupancy constants")


class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str


@dataclass
class Prior:
    """Holds prior probability data for an entity or type, and calculates it."""

    prior: float = field(default=MIN_PROBABILITY)
    prob_given_true: float | None = field(default=None)
    prob_given_false: float | None = field(default=None)
    last_updated: str | None = field(default=None)

    def __post_init__(self):
        """Validate probabilities."""
        if not MIN_PRIOR <= self.prior <= MAX_PRIOR:
            raise ValueError(
                f"Prior must be between {MIN_PRIOR} and {MAX_PRIOR} got: {self.prior}"
            )
        if (
            self.prob_given_true is not None
            and not MIN_PROBABILITY <= self.prob_given_true <= MAX_PROBABILITY
        ):
            raise ValueError(
                f"prob_given_true must be between {MIN_PROBABILITY} and {MAX_PROBABILITY} got: {self.prob_given_true}"
            )
        if (
            self.prob_given_false is not None
            and not MIN_PROBABILITY <= self.prob_given_false <= MAX_PROBABILITY
        ):
            raise ValueError(
                f"prob_given_false must be between {MIN_PROBABILITY} and {MAX_PROBABILITY} got: {self.prob_given_false}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert PriorData to a dictionary."""
        data = {
            "prior": self.prior,
            "last_updated": self.last_updated,
        }
        if self.prob_given_true is not None:
            data["prob_given_true"] = self.prob_given_true
        if self.prob_given_false is not None:
            data["prob_given_false"] = self.prob_given_false
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prior | None":
        """Create PriorData from a dictionary."""
        prior = data.get("prior")
        if prior is None or float(prior) < MIN_PRIOR:
            return None  # Skip invalid/zero priors
        # Handle potential None values during deserialization
        p_true = data.get("prob_given_true")
        p_false = data.get("prob_given_false")
        last_updated = data.get("last_updated")

        return cls(
            prior=float(prior) if prior is not None else MIN_PROBABILITY,
            prob_given_true=float(p_true) if p_true is not None else None,
            prob_given_false=float(p_false) if p_false is not None else None,
            last_updated=str(last_updated) if last_updated is not None else None,
        )

    @classmethod
    async def calculate(
        cls,
        hass: HomeAssistant,
        default_prior: float,
        default_prob_given_true: float,
        default_prob_given_false: float,
        entity_active_states: set,
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
        fallback_prior: Prior = Prior(
            prob_given_true=default_prob_given_true,
            prob_given_false=default_prob_given_false,
            prior=default_prior,
        )

        is_primary = entity_id == primary_sensor

        try:
            if is_primary:
                primary_states = await cls._get_states_from_recorder(
                    hass, entity_id, start_time, end_time
                )
                entity_states = primary_states
            else:
                try:
                    primary_states, entity_states = await asyncio.gather(
                        cls._get_states_from_recorder(
                            hass, primary_sensor, start_time, end_time
                        ),
                        cls._get_states_from_recorder(
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

        primary_intervals = await cls._states_to_intervals(
            primary_state_objects, start_time, end_time
        )
        entity_intervals = await cls._states_to_intervals(
            entity_state_objects, start_time, end_time
        )

        prob_given_true = cls._calculate_conditional_probability_with_intervals(
            entity_id,
            entity_intervals,
            {primary_sensor: primary_intervals},
            STATE_ON,
            entity_active_states,
        )
        prob_given_false = cls._calculate_conditional_probability_with_intervals(
            entity_id,
            entity_intervals,
            {primary_sensor: primary_intervals},
            STATE_OFF,
            entity_active_states,
        )

        prior = cls._calculate_prior_probability(
            prob_given_true, prob_given_false, primary_intervals
        )

        return Prior(
            prior=prior,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            last_updated=datetime.utcnow().isoformat(),
        )

    @staticmethod
    async def _get_states_from_recorder(
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
        entity_active_states: set,
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
                    if entity_interval["state"] in entity_active_states:
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
