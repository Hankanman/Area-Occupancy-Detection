"""Probability calculations and historical analysis for Area Occupancy Detection."""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Any, Optional
import collections

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_PLAYING,
    STATE_PAUSED,
    STATE_OPEN,
    STATE_CLOSED,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util
from homeassistant.exceptions import HomeAssistantError
from sqlalchemy.exc import SQLAlchemyError

from .types import (
    ProbabilityResult,
    DecayConfig,
)
from .probabilities import (
    MOTION_PROB_GIVEN_TRUE,
    MOTION_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MEDIA_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    APPLIANCE_PROB_GIVEN_FALSE,
    DOOR_PROB_GIVEN_TRUE,
    DOOR_PROB_GIVEN_FALSE,
    WINDOW_PROB_GIVEN_TRUE,
    WINDOW_PROB_GIVEN_FALSE,
    LIGHT_PROB_GIVEN_TRUE,
    LIGHT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    DEFAULT_PROB_GIVEN_FALSE,
    ENVIRONMENTAL_BASELINE_PERCENT,
    BASELINE_CACHE_TTL,
    DECAY_LAMBDA,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
    DEFAULT_PRIOR,
    MOTION_DEFAULT_PRIOR,
    MEDIA_DEFAULT_PRIOR,
    APPLIANCE_DEFAULT_PRIOR,
    DOOR_DEFAULT_PRIOR,
    WINDOW_DEFAULT_PRIOR,
    LIGHT_DEFAULT_PRIOR,
    ENVIRONMENTAL_PROB_GIVEN_TRUE,
    ENVIRONMENTAL_PROB_GIVEN_FALSE,
    ENVIRONMENTAL_DEFAULT_PRIOR,
)
from .const import (
    CACHE_DURATION,
)

_LOGGER = logging.getLogger(__name__)


def update_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Perform a Bayesian update."""
    _LOGGER.debug(
        "Updating probability with prior: %s, prob_given_true: %s, prob_given_false: %s",
        prior,
        prob_given_true,
        prob_given_false,
    )
    # Clamp input probabilities
    prior = max(MIN_PROBABILITY, min(prior, MAX_PROBABILITY))
    prob_given_true = max(MIN_PROBABILITY, min(prob_given_true, MAX_PROBABILITY))
    prob_given_false = max(MIN_PROBABILITY, min(prob_given_false, MAX_PROBABILITY))

    numerator = prob_given_true * prior
    denominator = numerator + prob_given_false * (1 - prior)
    if denominator == 0:
        return prior

    # Clamp result
    result = numerator / denominator
    return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))


def get_default_prior(entity_id: str, calc) -> float:
    """Return default prior based on entity category."""
    if entity_id in calc.motion_sensors:
        return MOTION_DEFAULT_PRIOR
    elif entity_id in calc.media_devices:
        return MEDIA_DEFAULT_PRIOR
    elif entity_id in calc.appliances:
        return APPLIANCE_DEFAULT_PRIOR
    elif entity_id in calc.door_sensors:
        return DOOR_DEFAULT_PRIOR
    elif entity_id in calc.window_sensors:
        return WINDOW_DEFAULT_PRIOR
    elif entity_id in calc.lights:
        return LIGHT_DEFAULT_PRIOR
    elif (
        entity_id in calc.illuminance_sensors
        or entity_id in calc.humidity_sensors
        or entity_id in calc.temperature_sensors
    ):
        return ENVIRONMENTAL_DEFAULT_PRIOR
    return DEFAULT_PRIOR


class ProbabilityCalculator:
    """Handles probability calculations and historical analysis."""

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
        door_sensors: list[str] | None = None,
        window_sensors: list[str] | None = None,
        lights: list[str] | None = None,
        decay_config: DecayConfig | None = None,
    ) -> None:
        _LOGGER.debug("Initializing ProbabilityCalculator")
        self.coordinator = coordinator
        self.motion_sensors = motion_sensors
        self.media_devices = media_devices or []
        self.appliances = appliances or []
        self.illuminance_sensors = illuminance_sensors or []
        self.humidity_sensors = humidity_sensors or []
        self.temperature_sensors = temperature_sensors or []
        self.door_sensors = door_sensors or []
        self.window_sensors = window_sensors or []
        self.lights = lights or []
        self.decay_config = decay_config or DecayConfig()

        self.hass = hass
        self._cache: dict[str, dict[str, Any]] = {}
        self._last_cache_update: datetime | None = None

        # Use an OrderedDict with a maximum size for the baseline cache
        self._baseline_cache = collections.OrderedDict()
        self._baseline_cache_max_size = 100  # Set an appropriate size limit

    def _needs_cache_update(self) -> bool:
        _LOGGER.debug("Checking if cache needs update")
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

    async def calculate_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> tuple[float, float, float]:
        """Calculate learned priors for a given entity."""
        _LOGGER.debug("Calculating prior for %s", entity_id)
        # Fetch motion sensor states
        motion_states = {}
        for motion_sensor in self.motion_sensors:
            ms = await self._get_states_from_recorder(
                motion_sensor, start_time, end_time
            )
            if ms:
                motion_states[motion_sensor] = ms

        if not motion_states:
            # No motion data, fallback to defaults
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                get_default_prior(entity_id, self),
            )

        # Fetch entity states
        entity_states = await self._get_states_from_recorder(
            entity_id, start_time, end_time
        )
        if not entity_states:
            # No entity data, fallback to defaults
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                get_default_prior(entity_id, self),
            )

        # Compute intervals for motion sensors once
        motion_intervals_by_sensor = {}
        for sensor_id, states in motion_states.items():
            intervals = self._states_to_intervals(states, start_time, end_time)
            motion_intervals_by_sensor[sensor_id] = intervals

        # Compute total durations for motion sensors from precomputed intervals
        motion_durations = self._compute_state_durations_from_intervals(
            motion_intervals_by_sensor
        )
        total_motion_active_time = motion_durations.get(STATE_ON, 0.0)
        total_motion_inactive_time = motion_durations.get(STATE_OFF, 0.0)
        total_motion_time = total_motion_active_time + total_motion_inactive_time

        if total_motion_time == 0:
            # No motion duration data, fallback to defaults
            return (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
                get_default_prior(entity_id, self),
            )

        # Calculate prior probability based on motion sensor and clamp it
        prior = max(
            MIN_PROBABILITY,
            min(total_motion_active_time / total_motion_time, MAX_PROBABILITY),
        )

        # Compute intervals for the entity once
        entity_intervals = self._states_to_intervals(
            entity_states, start_time, end_time
        )

        # Calculate conditional probabilities using precomputed intervals
        prob_given_true = self._calculate_conditional_probability_with_intervals(
            entity_id, entity_intervals, motion_intervals_by_sensor, STATE_ON
        )
        prob_given_false = self._calculate_conditional_probability_with_intervals(
            entity_id, entity_intervals, motion_intervals_by_sensor, STATE_OFF
        )

        # After computing the probabilities, update learned priors
        self.coordinator.update_learned_priors(
            entity_id,
            prob_given_true,
            prob_given_false,
            prior,
        )

        return prob_given_true, prob_given_false, prior

    async def _get_environmental_baseline(
        self,
        entity_id: str,
        entity_states: list[State],
        start: datetime,
        end: datetime,
    ) -> Optional[float]:
        """Get environmental baseline from cache or compute it if expired."""
        _LOGGER.debug("Getting environmental baseline for %s", entity_id)
        now = dt_util.utcnow()
        cached = self._baseline_cache.get(entity_id)
        if cached and (now - cached["timestamp"]).total_seconds() < BASELINE_CACHE_TTL:
            # Move the key to the end to indicate recent use
            self._baseline_cache.move_to_end(entity_id)
            return cached["upper_bound"]

        # Compute new baseline since cache is empty or expired
        env_upper_bound = self._compute_environmental_baseline(entity_states)
        if env_upper_bound is not None:
            # Before adding to cache, check size and evict oldest if necessary
            if len(self._baseline_cache) >= self._baseline_cache_max_size:
                self._baseline_cache.popitem(last=False)  # Remove oldest item

            self._baseline_cache[entity_id] = {
                "upper_bound": env_upper_bound,
                "timestamp": now,
            }
        return env_upper_bound

    def _compute_environmental_baseline(self, states: list[State]) -> Optional[float]:
        """Compute a baseline upper bound for environmental sensors."""
        _LOGGER.debug("Computing environmental baseline")
        values = []
        for s in states:
            try:
                val = float(s.state)
                values.append(val)
            except (ValueError, TypeError):
                continue

        if not values:
            return None

        min_val = min(values)
        max_val = max(values)
        mean_val = sum(values) / len(values)

        range_val = max_val - min_val
        if range_val <= 0:
            return mean_val

        threshold_increase = range_val * ENVIRONMENTAL_BASELINE_PERCENT
        upper_bound = mean_val + threshold_increase
        return upper_bound

    def _is_entity_active(
        self,
        entity_id: str,
        state_val: Any,
        env_upper_bound: Optional[float] = None,
        is_environmental: bool = False,
    ) -> bool:
        """Check if entity is active in the given interval."""
        if is_environmental:
            # Environmental active if value > env_upper_bound
            if env_upper_bound is None:
                return False
            try:
                val = float(state_val)
                return val > env_upper_bound
            except (ValueError, TypeError):
                return False
        # Non-environmental logic
        if entity_id in self.motion_sensors:
            return state_val == STATE_ON
        elif entity_id in self.media_devices:
            return state_val in (STATE_PLAYING, STATE_PAUSED)
        elif entity_id in self.appliances:
            return state_val == STATE_ON
        elif entity_id in self.door_sensors:
            return state_val in (STATE_ON, STATE_OPEN)
        elif entity_id in self.window_sensors:
            return state_val in (STATE_ON, STATE_OPEN)
        elif entity_id in self.lights:
            return state_val == STATE_ON
        return False

    def _states_to_intervals(
        self, states: list[State], start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime, Any]]:
        """Convert a list of states into intervals [(start, end, state), ...]."""
        _LOGGER.debug("Converting states to intervals")
        intervals = []
        sorted_states = sorted(states, key=lambda s: s.last_changed)
        current_start = start
        current_state = None

        for s in sorted_states:
            s_start = s.last_changed
            if s_start < start:
                s_start = start
            if current_state is None:
                current_state = s.state
                current_start = s_start
            else:
                if s_start > current_start:
                    intervals.append((current_start, s_start, current_state))
                current_state = s.state
                current_start = s_start

        if current_start < end:
            intervals.append((current_start, end, current_state))

        return intervals

    def _combine_motion_intervals(
        self, motion_states: dict[str, list[State]], start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime]]:
        """Combine all motion sensor ON intervals into a unified motion-active timeline."""
        _LOGGER.debug("Combining motion intervals")
        all_intervals = []

        for sensor_states in motion_states.values():
            sensor_intervals = self._states_to_intervals(sensor_states, start, end)
            on_intervals = [
                (st, en) for (st, en, val) in sensor_intervals if val == STATE_ON
            ]
            all_intervals.extend(on_intervals)

        all_intervals.sort(key=lambda x: x[0])
        merged = []
        for interval in all_intervals:
            if not merged:
                merged.append(interval)
            else:
                last = merged[-1]
                if interval[0] <= last[1]:
                    merged[-1] = (last[0], max(last[1], interval[1]))
                else:
                    merged.append(interval)

        return merged

    def _was_motion_active_during(
        self,
        start: datetime,
        end: datetime,
        motion_intervals: list[tuple[datetime, datetime]],
    ) -> bool:
        """Check if any motion interval overlaps with [start, end]."""
        _LOGGER.debug("Checking if motion intervals overlap with %s to %s", start, end)
        for m_start, m_end in motion_intervals:
            if m_start < end and m_end > start:
                return True
        return False

    def _perform_calculation_logic(
        self,
        sensor_states: dict[str, Any],
        current_probability: float,
        now: datetime,
        is_sync: bool = False,
    ) -> ProbabilityResult:
        """Core calculation logic shared between sync and async methods."""
        active_triggers = []
        sensor_probs = {}
        decay_status = {}

        # First pass - identify active triggers and update probabilities
        for entity_id, state in sensor_states.items():
            if not state or not state.get("availability", False):
                continue

            p_true, p_false = self._get_sensor_priors_from_history(entity_id)
            prior_val = self._get_sensor_prior(entity_id)
            if p_true is None or p_false is None:
                p_true, p_false = self.get_sensor_priors(entity_id)
                prior_val = get_default_prior(entity_id, self)

            is_active = self._is_active_now(entity_id, state["state"])
            if is_active:
                active_triggers.append(entity_id)
                current_probability = update_probability(prior_val, p_true, p_false)

            sensor_probs[entity_id] = current_probability

        # Apply decay only if:
        # 1. No active triggers
        # 2. Decay is enabled
        # 3. Minimum delay has passed since last trigger
        decay_factor = 1.0
        if not active_triggers:
            if is_sync:
                last_trigger = self.coordinator.get_last_positive_trigger_sync()
            else:
                last_trigger = self.coordinator.get_last_positive_trigger_sync()

            if last_trigger is not None:
                elapsed = (now - last_trigger).total_seconds()
                decay_window = self.coordinator.get_decay_window()
                decay_min_delay = self.coordinator.get_decay_min_delay()

                # Only start decay after min_delay has passed
                if elapsed > decay_min_delay and decay_window > 0:
                    # Calculate time since decay should have started
                    decay_time = elapsed - decay_min_delay
                    # Apply exponential decay based on time since decay started
                    decay_factor = math.exp(-DECAY_LAMBDA * (decay_time / decay_window))
                    current_probability *= decay_factor
                    current_probability = max(
                        MIN_PROBABILITY, min(current_probability, MAX_PROBABILITY)
                    )
                    decay_status["global_decay"] = round(1.0 - decay_factor, 4)
                else:
                    decay_status["global_decay"] = 0.0
            else:
                decay_status["global_decay"] = 0.0
        else:
            # Reset decay when there are active triggers
            decay_status["global_decay"] = 0.0
            if is_sync:
                self.coordinator.set_last_positive_trigger_sync(now)
            else:
                self.hass.async_create_task(
                    self.coordinator.set_last_positive_trigger(now)
                )

        final_probability = current_probability

        return {
            "probability": final_probability,
            "prior_probability": 0.0,
            "active_triggers": active_triggers,
            "sensor_probabilities": sensor_probs,
            "device_states": {},
            "decay_status": decay_status,
            "sensor_availability": {
                k: v.get("availability", False) for k, v in sensor_states.items()
            },
            "is_occupied": final_probability
            >= self.coordinator.get_threshold_decimal(),
        }

    async def calculate(
        self,
        sensor_states: dict[str, dict[str, Any]],
        motion_timestamps: dict[str, datetime],
    ) -> ProbabilityResult:
        """Asynchronous method to calculate occupancy probability."""
        _LOGGER.debug("Calculating occupancy probability asynchronously")
        try:
            # Determine prior
            if self.coordinator.data and "probability" in self.coordinator.data:
                previous_probability = self.coordinator.data["probability"]
            else:
                previous_probability = DEFAULT_PRIOR

            now = dt_util.utcnow()

            # Use the shared calculation logic
            result = self._perform_calculation_logic(
                sensor_states, previous_probability, now, is_sync=False
            )

            return result

        except (HomeAssistantError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error in probability calculation: %s", err)
            raise HomeAssistantError(
                "Failed to calculate occupancy probability"
            ) from err

    def calculate_sync(
        self,
        sensor_states: dict[str, Any],
        motion_timestamps: dict[str, datetime],
    ) -> ProbabilityResult:
        """Synchronous method to calculate occupancy probability."""
        _LOGGER.debug("Calculating occupancy probability synchronously")
        try:
            # Determine prior
            if self.coordinator.data and "probability" in self.coordinator.data:
                previous_probability = self.coordinator.data["probability"]
            else:
                previous_probability = DEFAULT_PRIOR

            now = dt_util.utcnow()

            # Use the shared calculation logic
            result = self._perform_calculation_logic(
                sensor_states, previous_probability, now, is_sync=True
            )

            return result

        except (HomeAssistantError, ValueError, AttributeError, KeyError) as err:
            _LOGGER.error("Error in synchronous probability calculation: %s", err)
            raise HomeAssistantError(
                "Failed to calculate occupancy probability synchronously"
            ) from err

    def get_sensor_priors(self, entity_id: str) -> tuple[float, float]:
        """Return default priors for a sensor based on its category."""
        _LOGGER.debug("Getting sensor priors for %s", entity_id)
        if entity_id in self.motion_sensors:
            return MOTION_PROB_GIVEN_TRUE, MOTION_PROB_GIVEN_FALSE
        elif entity_id in self.media_devices:
            return MEDIA_PROB_GIVEN_TRUE, MEDIA_PROB_GIVEN_FALSE
        elif entity_id in self.appliances:
            return APPLIANCE_PROB_GIVEN_TRUE, APPLIANCE_PROB_GIVEN_FALSE
        elif entity_id in self.door_sensors:
            return DOOR_PROB_GIVEN_TRUE, DOOR_PROB_GIVEN_FALSE
        elif entity_id in self.window_sensors:
            return WINDOW_PROB_GIVEN_TRUE, WINDOW_PROB_GIVEN_FALSE
        elif entity_id in self.lights:
            return LIGHT_PROB_GIVEN_TRUE, LIGHT_PROB_GIVEN_FALSE
        elif (
            entity_id in self.illuminance_sensors
            or entity_id in self.humidity_sensors
            or entity_id in self.temperature_sensors
        ):
            return ENVIRONMENTAL_PROB_GIVEN_TRUE, ENVIRONMENTAL_PROB_GIVEN_FALSE
        return DEFAULT_PROB_GIVEN_TRUE, DEFAULT_PROB_GIVEN_FALSE

    def _get_sensor_priors_from_history(
        self, entity_id: str
    ) -> tuple[float, float] | tuple[None, None]:
        """Retrieve learned priors for a sensor from coordinator data."""
        _LOGGER.debug("Getting sensor priors from history for %s", entity_id)
        priors = self.coordinator.learned_priors.get(entity_id)
        if priors:
            return priors["prob_given_true"], priors["prob_given_false"]
        else:
            # Use default priors if learned priors are not available
            return None, None

    def _get_sensor_prior(self, entity_id: str) -> float:
        _LOGGER.debug("Getting sensor prior for %s", entity_id)
        priors = self.coordinator.learned_priors.get(entity_id)
        if priors and "prior" in priors:
            return priors["prior"]
        return get_default_prior(entity_id, self)

    def _is_active_now(self, entity_id: str, state: str) -> bool:
        if entity_id in self.motion_sensors:
            return state == STATE_ON
        elif entity_id in self.media_devices:
            return state in (STATE_PLAYING, STATE_PAUSED)
        elif entity_id in self.appliances:
            return state == STATE_ON
        elif entity_id in self.door_sensors:
            return state in (STATE_OFF, STATE_CLOSED)
        elif entity_id in self.window_sensors:
            return state in (STATE_ON, STATE_OPEN)
        elif entity_id in self.lights:
            return state == STATE_ON
        return False

    def _compute_state_durations_from_intervals(
        self, intervals_dict: dict[str, list[tuple[datetime, datetime, Any]]]
    ) -> dict[str, float]:
        """Compute total durations for each state from precomputed intervals."""
        _LOGGER.debug("Computing state durations from intervals")
        durations = {}
        for intervals in intervals_dict.values():
            for interval_start, interval_end, state_val in intervals:
                duration = (interval_end - interval_start).total_seconds()
                durations[state_val] = durations.get(state_val, 0.0) + duration
        return durations

    def _calculate_conditional_probability_with_intervals(
        self,
        entity_id: str,
        entity_intervals: list[tuple[datetime, datetime, Any]],
        motion_intervals_by_sensor: dict[str, list[tuple[datetime, datetime, Any]]],
        motion_state_filter: str,
    ) -> float:
        """Calculate P(entity_active | motion_state) using precomputed intervals."""
        _LOGGER.debug("Calculating conditional probability for %s", entity_id)
        # Combine motion intervals for the specified motion state
        motion_intervals = []
        for intervals in motion_intervals_by_sensor.values():
            motion_intervals.extend(
                (start, end)
                for start, end, state in intervals
                if state == motion_state_filter
            )

        total_motion_duration = sum(
            (end - start).total_seconds() for start, end in motion_intervals
        )
        if total_motion_duration == 0:
            return (
                DEFAULT_PROB_GIVEN_TRUE
                if motion_state_filter == STATE_ON
                else DEFAULT_PROB_GIVEN_FALSE
            )

        # Get entity active intervals
        entity_active_intervals = [
            (start, end)
            for start, end, state in entity_intervals
            if self._is_entity_active(entity_id, state)
        ]

        # Calculate the overlap duration
        overlap_duration = 0.0
        for e_start, e_end in entity_active_intervals:
            for m_start, m_end in motion_intervals:
                overlap_start = max(e_start, m_start)
                overlap_end = min(e_end, m_end)
                if overlap_start < overlap_end:
                    overlap_duration += (overlap_end - overlap_start).total_seconds()

        # Clamp the final probability before returning
        result = overlap_duration / total_motion_duration
        return max(MIN_PROBABILITY, min(result, MAX_PROBABILITY))
