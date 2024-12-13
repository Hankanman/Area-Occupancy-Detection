"""Historical analysis for Area Occupancy Detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import (
    STATE_ON,
    STATE_PLAYING,
    STATE_PAUSED,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

from .probabilities import (
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
    MOTION_PROB_GIVEN_TRUE,
    MOTION_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MEDIA_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    APPLIANCE_PROB_GIVEN_FALSE,
)

_LOGGER = logging.getLogger(__name__)

VALID_ACTIVE_STATES = {STATE_ON, STATE_PLAYING, STATE_PAUSED}
CACHE_DURATION = timedelta(hours=6)
TIMESLOT_DURATION = timedelta(minutes=30)
SLOTS_PER_DAY = 48


class HistoricalAnalysis:
    """Handles historical data analysis."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the analysis."""
        self.hass = hass
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_cache_update: datetime | None = None

    def _needs_cache_update(self) -> bool:
        """Check if cache needs updating."""
        if not self._last_cache_update:
            return True
        return dt_util.utcnow() - self._last_cache_update > CACHE_DURATION

    async def _get_states_from_recorder(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> list[State] | None:
        """Get states from recorder."""
        try:

            def _get_states():
                return get_significant_states(
                    self.hass,
                    start_time,
                    end_time,
                    [entity_id],
                    minimal_response=False,  # Changed to get full state objects
                )

            recorder = get_instance(self.hass)
            states = await recorder.async_add_executor_job(_get_states)

            if not states:
                return None

            state_list = states.get(entity_id, [])

            # Validate each state object
            validated_states = []
            for state in state_list:
                if hasattr(state, "state") or (
                    isinstance(state, dict) and "state" in state
                ):
                    validated_states.append(state)
                else:
                    _LOGGER.warning(
                        "Invalid state object found for %s: %s", entity_id, state
                    )

            return validated_states if validated_states else None

        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.error("Error getting states for %s: %s", entity_id, err)
            return None

    def _calculate_active_duration(
        self, states: list[State], start_time: datetime, end_time: datetime
    ) -> float:
        """Calculate total duration in active states."""
        if not states:
            return 0.0

        total_active_time = timedelta()
        current_state = None
        last_change = start_time

        # Process each state change
        for state_obj in states:
            try:
                # Ensure we're working with a proper state object
                if hasattr(state_obj, "state"):
                    state = state_obj.state
                elif isinstance(state_obj, dict) and "state" in state_obj:
                    state = state_obj["state"]
                else:
                    _LOGGER.warning("Invalid state object encountered: %s", state_obj)
                    continue

                if state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                    continue

                current_time = (
                    state_obj.last_changed
                    if hasattr(state_obj, "last_changed")
                    else (
                        dt_util.parse_datetime(state_obj.get("last_changed", ""))
                        if isinstance(state_obj, dict)
                        else start_time
                    )
                )

                if not current_time:
                    _LOGGER.warning(
                        "Could not determine state change time for %s", state_obj
                    )
                    continue

                # If previous state was active, add its duration
                if current_state in VALID_ACTIVE_STATES:
                    duration = current_time - last_change
                    total_active_time += duration

                current_state = state
                last_change = current_time

            except Exception as err:  # pylint: disable=broad-except
                _LOGGER.error("Error processing state object: %s", err)
                continue

        # Handle final state up to end_time
        if current_state in VALID_ACTIVE_STATES:
            total_active_time += end_time - last_change

        return total_active_time.total_seconds()

    async def calculate_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> tuple[float, float]:
        """Calculate prior probabilities based on historical state."""
        # Get sensor type from entity_id to determine defaults
        if "motion" in entity_id:
            default_true, default_false = (
                MOTION_PROB_GIVEN_TRUE,
                MOTION_PROB_GIVEN_FALSE,
            )
            min_true, max_false = 0.15, 0.85  # Minimum thresholds for motion
        elif "media" in entity_id:
            default_true, default_false = MEDIA_PROB_GIVEN_TRUE, MEDIA_PROB_GIVEN_FALSE
            min_true, max_false = 0.10, 0.90  # Minimum thresholds for media
        elif "appliance" in entity_id:
            default_true, default_false = (
                APPLIANCE_PROB_GIVEN_TRUE,
                APPLIANCE_PROB_GIVEN_FALSE,
            )
            min_true, max_false = 0.05, 0.95  # Minimum thresholds for appliances
        else:
            default_true, default_false = (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
            )
            min_true, max_false = 0.01, 0.99  # Default minimum thresholds

        total_duration = (end_time - start_time).total_seconds()
        if total_duration <= 0:
            return default_true, default_false

        states = await self._get_states_from_recorder(entity_id, start_time, end_time)
        if not states:
            return default_true, default_false

        active_duration = min(
            self._calculate_active_duration(states, start_time, end_time),
            total_duration,
        )

        # Calculate probabilities with sensor-specific bounds
        prob_given_true = min(
            max(active_duration / total_duration, min_true), max_false
        )
        prob_given_false = 1 - prob_given_true

        return round(prob_given_true, 4), round(prob_given_false, 4)

    async def calculate_timeslots(
        self, entity_ids: list[str], history_period: int
    ) -> dict[str, Any]:
        """Calculate timeslot probabilities for entities."""
        if self._needs_cache_update():
            self._cache = {}
            self._last_cache_update = dt_util.utcnow()

        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_period)

        timeslots = {}

        # Generate time slots
        for hour in range(24):
            for minute in (0, 30):
                slot_key = f"{hour:02d}:{minute:02d}"

                # Skip if cached and valid
                if slot_key in self._cache:
                    timeslots[slot_key] = self._cache[slot_key]
                    continue

                slot_data = {
                    "entities": [],
                    "prob_given_true": 0.0,
                    "prob_given_false": 0.0,
                }

                # Process each entity
                for entity_id in entity_ids:
                    daily_probs = []
                    current = start_time

                    # Analyze each day in history period
                    while current < end_time:
                        slot_start = current.replace(
                            hour=hour, minute=minute, second=0, microsecond=0
                        )
                        slot_end = slot_start + TIMESLOT_DURATION

                        if slot_end > end_time:
                            break

                        prob_true, prob_false = await self.calculate_prior(
                            entity_id, slot_start, slot_end
                        )
                        daily_probs.append((prob_true, prob_false))
                        current += timedelta(days=1)

                    # Average probabilities across days
                    if daily_probs:
                        avg_prob_true = round(
                            sum(p[0] for p in daily_probs) / len(daily_probs), 4
                        )
                        avg_prob_false = round(
                            sum(p[1] for p in daily_probs) / len(daily_probs), 4
                        )

                        slot_data["entities"].append(
                            {
                                "id": entity_id,
                                "prob_given_true": avg_prob_true,
                                "prob_given_false": avg_prob_false,
                            }
                        )

                # Calculate combined probability for slot
                if slot_data["entities"]:
                    combined_true = 1.0
                    combined_false = 1.0
                    for entity in slot_data["entities"]:
                        combined_true *= entity["prob_given_true"]
                        combined_false *= entity["prob_given_false"]

                    slot_data["prob_given_true"] = round(combined_true, 4)
                    slot_data["prob_given_false"] = round(combined_false, 4)

                timeslots[slot_key] = slot_data
                self._cache[slot_key] = slot_data

        _LOGGER.debug("Historical analysis complete")
        return timeslots
