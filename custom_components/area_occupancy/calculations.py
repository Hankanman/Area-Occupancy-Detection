"""Probability calculations and historical analysis for Area Occupancy Detection."""

from __future__ import annotations

import logging
import asyncio
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
from homeassistant.exceptions import HomeAssistantError
from sqlalchemy.exc import SQLAlchemyError

from .types import (
    ProbabilityResult,
    SensorStates,
    DecayConfig,
    Timeslot,
)
from .probabilities import (
    MIN_PROBABILITY,
    MAX_PROBABILITY,
    MOTION_PROB_GIVEN_TRUE,
    MOTION_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MEDIA_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    APPLIANCE_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
)

_LOGGER = logging.getLogger(__name__)

VALID_ACTIVE_STATES = {STATE_ON, STATE_PLAYING, STATE_PAUSED}
CACHE_DURATION = timedelta(hours=6)
TIMESLOT_DURATION = timedelta(minutes=30)

# MODIFICATION START: Define minimum sample count for learned priors
MIN_SAMPLE_COUNT_FOR_PRIORS = 5
# MODIFICATION END


def update_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Perform a Bayesian update given a prior probability and evidence.

    numerator = P(E|H)*P(H)
    denominator = P(E|H)*P(H) + P(E|~H)*P(~H)

    This function updates the prior probability P(H) with new evidence E.
    """
    numerator = prob_given_true * prior
    denominator = numerator + prob_given_false * (1 - prior)
    if denominator == 0:
        return prior
    return numerator / denominator


class ProbabilityCalculator:
    """Handles probability calculations and historical analysis.

    This class calculates occupancy probability by:
    1. Starting from a prior (often derived from motion sensors).
    2. Incorporating additional sensors (media, appliances, environment) by updating
       the probability using learned priors or fallback defaults.
    3. Provides methods to compute learned priors from historical data and to fall back
       to timeslot-based aggregated probabilities if no direct learned priors are found.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        coordinator,
        motion_sensors: list[str],
        media_devices: list[str] | None = None,
        appliances: list[str] | None = None,
        illuminance_sensors: list[str] | None = None,
        humidity_sensors: list[str] | None = None,
        temperature_sensors: list[str] | None = None,
        decay_config: DecayConfig | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.motion_sensors = motion_sensors
        self.media_devices = media_devices or []
        self.appliances = appliances or []
        self.illuminance_sensors = illuminance_sensors or []
        self.humidity_sensors = humidity_sensors or []
        self.temperature_sensors = temperature_sensors or []
        self.decay_config = decay_config or DecayConfig()

        self.hass = hass
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_cache_update: datetime | None = None

    def _needs_cache_update(self) -> bool:
        return (
            not self._last_cache_update
            or dt_util.utcnow() - self._last_cache_update > CACHE_DURATION
        )

    async def _get_states_from_recorder(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> list[State] | None:
        """Fetch states history from recorder."""
        try:
            states = await get_instance(self.hass).async_add_executor_job(
                lambda: get_significant_states(
                    self.hass,
                    start_time,
                    end_time,
                    [entity_id],
                    minimal_response=False,
                )
            )
            return states.get(entity_id) if states else None
        except (HomeAssistantError, SQLAlchemyError) as err:
            _LOGGER.error("Error getting states for %s: %s", entity_id, err)
            return None

    def _calculate_active_duration(
        self, states: list[State], start_time: datetime, end_time: datetime
    ) -> float:
        """Calculate how long an entity was in an active state within the time period."""
        if not states:
            return 0.0

        total_active_time = timedelta()
        current_state = None
        last_change = start_time

        for state_obj in states:
            state = getattr(state_obj, "state", None) or state_obj.get("state")
            if not state or state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
                continue

            current_time = (
                getattr(state_obj, "last_changed", None)
                or dt_util.parse_datetime(state_obj.get("last_changed", ""))
                or start_time
            )

            if current_state in VALID_ACTIVE_STATES:
                total_active_time += current_time - last_change

            current_state = state
            last_change = current_time

        if current_state in VALID_ACTIVE_STATES:
            total_active_time += end_time - last_change

        return total_active_time.total_seconds()

    async def calculate_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> tuple[float, float]:
        """Calculate learned priors for a given entity based on historical data.

        We compare how often the entity's 'active' state correlated with motion being on.
        If insufficient data or no meaningful correlation found, default priors are returned.

        Only store learned priors if minimum sample count is met.
        """
        _LOGGER.info(
            "Calculating prior for %s from %s to %s",
            entity_id,
            start_time,
            end_time,
        )

        try:
            motion_states = {}
            for motion_sensor in self.motion_sensors:
                states = await self._get_states_from_recorder(
                    motion_sensor, start_time, end_time
                )
                if states:
                    motion_states[motion_sensor] = states

            if not motion_states:
                _LOGGER.debug(
                    "No motion states found, returning defaults for %s", entity_id
                )
                return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE

            entity_states = await self._get_states_from_recorder(
                entity_id, start_time, end_time
            )
            if not entity_states:
                _LOGGER.debug(
                    "No entity states found for %s, using defaults", entity_id
                )
                return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE

            total_time = 0.0
            active_with_motion = 0.0
            active_without_motion = 0.0
            sample_count = 0  # Count how many times we see active states

            # Simple approach: sum durations when active and check if motion was also active
            for state_obj in entity_states:
                duration = self._get_state_duration(state_obj)
                if duration <= 0:
                    continue
                total_time += duration
                motion_active = self._check_motion_active_during(
                    state_obj.last_updated, motion_states
                )

                if self._is_active_state(state_obj.state, entity_id):
                    sample_count += 1
                    if motion_active:
                        active_with_motion += duration
                    else:
                        active_without_motion += duration

            if total_time == 0:
                _LOGGER.info("No active time found for %s, using defaults", entity_id)
                return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE

            prob_given_true = (
                active_with_motion / total_time
                if total_time > 0
                else DEFAULT_PROB_GIVEN_TRUE
            )
            prob_given_false = (
                active_without_motion / total_time
                if total_time > 0
                else DEFAULT_PROB_GIVEN_FALSE
            )

            # Bound probabilities
            prob_given_true = max(min(prob_given_true, 0.99), 0.01)
            prob_given_false = max(min(prob_given_false, 0.99), 0.01)

            # MODIFICATION START: Only store learned priors if we have enough samples
            if sample_count >= MIN_SAMPLE_COUNT_FOR_PRIORS:
                self.coordinator.update_learned_priors(
                    entity_id, prob_given_true, prob_given_false
                )
                _LOGGER.info(
                    "Learned priors for %s: true=%.4f, false=%.4f based on %d samples",
                    entity_id,
                    prob_given_true,
                    prob_given_false,
                    sample_count,
                )
            else:
                _LOGGER.info(
                    "Not enough samples (%d) for %s, not storing learned priors, using defaults",
                    sample_count,
                    entity_id,
                )
                prob_given_true, prob_given_false = (
                    DEFAULT_PROB_GIVEN_TRUE,
                    DEFAULT_PROB_GIVEN_FALSE,
                )
            # MODIFICATION END

            return round(prob_given_true, 4), round(prob_given_false, 4)

        except (HomeAssistantError, SQLAlchemyError, ValueError, AttributeError) as err:
            _LOGGER.error("Error calculating priors for %s: %s", entity_id, err)
            return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE

    async def calculate_timeslots(
        self, entity_ids: list[str], history_period: int
    ) -> dict[str, Any]:
        """Calculate aggregated timeslot-based probabilities over a historical period.

        This can serve as a fallback if no learned priors are available for a sensor.
        Timeslot data is grouped in 30-minute increments over the specified history period.
        """
        if not self._needs_cache_update():
            return self._cache

        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_period)
        timeslots = {}

        _LOGGER.info(
            "Calculating timeslots for %s for the last %s days",
            entity_ids,
            history_period,
        )

        # Generate half-hour slots for a 24-hour period
        for hour in range(24):
            for minute in (0, 30):
                if hour % 4 == 0 and minute == 0:
                    await asyncio.sleep(0.1)  # Sleep briefly to avoid DB pressure

                slot_key = f"{hour:02d}:{minute:02d}"
                if slot_key in self._cache:
                    timeslots[slot_key] = self._cache[slot_key]
                    continue

                slot_data = await self._process_timeslot(
                    entity_ids, start_time, end_time, hour, minute
                )
                timeslots[slot_key] = self._cache[slot_key] = slot_data

        self._last_cache_update = dt_util.utcnow()
        return {"slots": timeslots, "last_updated": self._last_cache_update}

    async def _process_timeslot(
        self,
        entity_ids: list[str],
        start_time: datetime,
        end_time: datetime,
        hour: int,
        minute: int,
    ) -> dict[str, Any]:
        """Compute average priors for a given timeslot over the historical period."""
        slot_key = f"{hour:02d}:{minute:02d}"
        slot_data = {"entities": []}
        combined_true = 1.0
        combined_false = 1.0

        _LOGGER.info("Processing timeslot %s for %s", slot_key, entity_ids)

        # For each entity, we compute priors in daily slices aligned with the slot
        for entity_id in entity_ids:
            daily_probs = []
            current = start_time

            _LOGGER.debug("Processing entity %s for timeslot %s", entity_id, slot_key)

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

                combined_true *= avg_prob_true
                combined_false *= avg_prob_false

        if slot_data["entities"]:
            slot_data.update(
                {
                    "prob_given_true": round(combined_true, 4),
                    "prob_given_false": round(combined_false, 4),
                }
            )

        return slot_data

    async def calculate(
        self,
        sensor_states: SensorStates,
        motion_timestamps: dict[str, datetime],
        timeslot: Timeslot | None = None,
    ) -> ProbabilityResult:
        """Calculate the current occupancy probability by applying Bayesian updates.

        This method:
        1. Starts with a probability heavily influenced by motion sensors (if active).
        2. For each other sensor, attempts to get learned priors. If none, it tries timeslot data.
           If still none found, it falls back to default constants.
        3. Updates the current probability using each sensor's state as evidence.

        The final probability is bounded between MIN_PROBABILITY and MAX_PROBABILITY.
        """
        try:
            # Start from a motion-based prior
            motion_active = any(
                sensor_states.get(sensor_id, {}).get("state") == STATE_ON
                for sensor_id in self.motion_sensors
            )
            current_probability = (
                MOTION_PROB_GIVEN_TRUE
                if motion_active
                else (1 - MOTION_PROB_GIVEN_TRUE)
            )

            active_triggers = []
            sensor_probs = {}

            # For each sensor, we get priors from learned data, timeslots, or defaults
            for entity_id, state in sensor_states.items():
                if not state or not state.get("availability", False):
                    continue

                # MODIFICATION START: Detailed fallback logging
                p_true, p_false = self._get_sensor_priors_from_history(entity_id)
                if p_true is None or p_false is None:
                    _LOGGER.debug(
                        "No learned priors for %s, attempting timeslot fallback.",
                        entity_id,
                    )
                    p_true, p_false = self.get_timeslot_probabilities(
                        entity_id, timeslot
                    )
                    if p_true is None or p_false is None:
                        _LOGGER.debug(
                            "No timeslot data for %s, using default priors.", entity_id
                        )
                        p_true, p_false = self.get_sensor_priors(entity_id)
                # MODIFICATION END

                is_active = self._is_active_state(state["state"], entity_id)
                if is_active:
                    active_triggers.append(entity_id)
                    # Apply Bayesian update with the found priors
                    current_probability = update_probability(
                        current_probability, p_true, p_false
                    )
                sensor_probs[entity_id] = current_probability

            final_probability = max(
                MIN_PROBABILITY, min(current_probability, MAX_PROBABILITY)
            )

            return {
                "probability": final_probability,
                "prior_probability": 0.0,
                "active_triggers": active_triggers,
                "sensor_probabilities": sensor_probs,
                "device_states": {},
                "decay_status": {},
                "sensor_availability": {
                    k: v.get("availability", False) for k, v in sensor_states.items()
                },
                "is_occupied": final_probability
                >= self.coordinator.get_threshold_decimal(),
            }

        except (HomeAssistantError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error in probability calculation: %s", err)
            raise HomeAssistantError(
                "Failed to calculate occupancy probability"
            ) from err

    def get_sensor_priors(self, entity_id: str) -> tuple[float, float]:
        """Return default priors for a sensor based on its category."""
        if entity_id in self.motion_sensors:
            return MOTION_PROB_GIVEN_TRUE, MOTION_PROB_GIVEN_FALSE
        elif entity_id in self.media_devices:
            return MEDIA_PROB_GIVEN_TRUE, MEDIA_PROB_GIVEN_FALSE
        elif entity_id in self.appliances:
            return APPLIANCE_PROB_GIVEN_TRUE, APPLIANCE_PROB_GIVEN_FALSE
        return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE

    def _get_sensor_priors_from_history(
        self, entity_id: str
    ) -> tuple[float, float] | tuple[None, None]:
        """Attempt to retrieve learned priors for a sensor from stored coordinator data."""
        priors = self.coordinator.learned_priors.get(entity_id)
        if priors:
            return priors["prob_given_true"], priors["prob_given_false"]
        return None, None

    def get_timeslot_probabilities(
        self, entity_id: str, timeslot: Timeslot | None
    ) -> tuple[float, float] | tuple[None, None]:
        """Get probability from timeslot data if learned priors are unavailable.

        Timeslot probabilities are an aggregated fallback, representing how historically
        active this sensor was during similar times of day. If no timeslot data is found,
        return None, prompting a fallback to defaults.
        """
        if timeslot and "entities" in timeslot:
            entity_data = next(
                (e for e in timeslot["entities"] if e["id"] == entity_id),
                None,
            )
            if entity_data:
                return entity_data["prob_given_true"], entity_data["prob_given_false"]

        # If no timeslot data, return None to indicate we should fallback further
        return None, None

    def _get_state_duration(self, state: State) -> float:
        """Calculate how long a given state lasted."""
        try:
            last_changed = getattr(
                state, "last_changed", None
            ) or dt_util.parse_datetime(state.get("last_changed", ""))
            last_updated = getattr(
                state, "last_updated", None
            ) or dt_util.parse_datetime(state.get("last_updated", ""))
            if not last_changed or not last_updated:
                return 0.0
            return (last_updated - last_changed).total_seconds()
        except (AttributeError, ValueError):
            return 0.0

    def _check_motion_active_during(
        self, timestamp: datetime, motion_states: dict
    ) -> bool:
        """Check if motion was active at a given timestamp."""
        try:
            for states in motion_states.values():
                for state in states:
                    if (
                        state.state == STATE_ON
                        and state.last_changed <= timestamp <= state.last_updated
                    ):
                        return True
            return False
        except (AttributeError, TypeError):
            return False

    def _is_active_state(self, state: str, entity_id: str) -> bool:
        """Check if given state indicates 'activity' for the entity type."""
        if entity_id in self.motion_sensors:
            return state == STATE_ON
        elif entity_id in self.media_devices:
            return state in (STATE_PLAYING, STATE_PAUSED)
        elif entity_id in self.appliances:
            return state == STATE_ON
        return False
