"""Probability calculations and historical analysis for Area Occupancy Detection."""

from __future__ import annotations

import math
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

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
    Timeslot,
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
    MIN_ACTIVE_DURATION_FOR_PRIORS,
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

_LOGGER = logging.getLogger(__name__)

VALID_ACTIVE_STATES = {STATE_ON, STATE_PLAYING, STATE_PAUSED, STATE_CLOSED}
CACHE_DURATION = timedelta(hours=6)
TIMESLOT_DURATION = timedelta(minutes=30)


def update_probability(
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
) -> float:
    """Perform a Bayesian update."""
    numerator = prob_given_true * prior
    denominator = numerator + prob_given_false * (1 - prior)
    if denominator == 0:
        return prior
    return numerator / denominator


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

        # Baseline cache: {entity_id: {"upper_bound": float, "timestamp": datetime}}
        self._baseline_cache: dict[str, dict[str, Any]] = {}

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

    async def calculate_prior(
        self, entity_id: str, start_time: datetime, end_time: datetime
    ) -> tuple[float, float, float]:
        """
        Calculate learned priors for a given entity.
        Returns (prob_given_true, prob_given_false, prior).
        If conditions not met, defaults are returned.
        """
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

        is_environmental = entity_id in (
            self.illuminance_sensors + self.humidity_sensors + self.temperature_sensors
        )

        # If environmental, get cached or computed baseline
        env_upper_bound = None
        if is_environmental:
            env_upper_bound = await self._get_environmental_baseline(
                entity_id, entity_states, start_time, end_time
            )

        intervals = self._states_to_intervals(entity_states, start_time, end_time)
        motion_intervals = self._combine_motion_intervals(
            motion_states, start_time, end_time
        )

        active_with_motion = 0.0
        active_without_motion = 0.0
        total_motion_active_time = 0.0
        total_motion_inactive_time = 0.0

        for interval_start, interval_end, state_val in intervals:
            interval_duration = (interval_end - interval_start).total_seconds()
            motion_active = self._was_motion_active_during(
                interval_start, interval_end, motion_intervals
            )

            # Determine if entity is active in this interval
            entity_active = self._is_entity_active(
                entity_id, state_val, env_upper_bound, is_environmental
            )

            # Accumulate
            if motion_active:
                total_motion_active_time += interval_duration
                if entity_active:
                    active_with_motion += interval_duration
            else:
                total_motion_inactive_time += interval_duration
                if entity_active:
                    active_without_motion += interval_duration

        # Compute probabilities
        prob_given_true = DEFAULT_PROB_GIVEN_TRUE
        prob_given_false = DEFAULT_PROB_GIVEN_FALSE

        if total_motion_active_time > 0:
            prob_given_true = max(
                min(active_with_motion / total_motion_active_time, 0.99), 0.01
            )
        if total_motion_inactive_time > 0:
            prob_given_false = max(
                min(active_without_motion / total_motion_inactive_time, 0.99), 0.01
            )

        total_active_time = active_with_motion + active_without_motion
        if total_active_time >= MIN_ACTIVE_DURATION_FOR_PRIORS:
            # We have learned priors
            prior = get_default_prior(entity_id, self)  # You can refine this if needed
            self.coordinator.update_learned_priors(
                entity_id, prob_given_true, prob_given_false, prior
            )
        else:
            # Not enough data, store defaults anyway
            prob_given_true, prob_given_false = (
                DEFAULT_PROB_GIVEN_TRUE,
                DEFAULT_PROB_GIVEN_FALSE,
            )
            prior = get_default_prior(entity_id, self)
            # Still store them so learned_priors is never empty
            self.coordinator.update_learned_priors(
                entity_id, prob_given_true, prob_given_false, prior
            )

        return round(prob_given_true, 4), round(prob_given_false, 4), prior

    async def _get_environmental_baseline(
        self,
        entity_id: str,
        entity_states: list[State],
        start: datetime,
        end: datetime,
    ) -> Optional[float]:
        """Get environmental baseline from cache or compute it if expired."""

        now = dt_util.utcnow()
        cached = self._baseline_cache.get(entity_id)
        if cached and (now - cached["timestamp"]).total_seconds() < BASELINE_CACHE_TTL:
            return cached["upper_bound"]

        # Compute new baseline since cache is empty or expired
        env_upper_bound = self._compute_environmental_baseline(entity_states)
        self._baseline_cache[entity_id] = {
            "upper_bound": env_upper_bound,
            "timestamp": now,
        }
        return env_upper_bound

    def _compute_environmental_baseline(self, states: list[State]) -> Optional[float]:
        """Compute a baseline upper bound for environmental sensors."""
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
        env_upper_bound: Optional[float],
        is_environmental: bool,
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
            # Doors "active"? This logic can vary, assume open means active?
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
        for m_start, m_end in motion_intervals:
            if m_start < end and m_end > start:
                return True
        return False

    async def calculate_timeslots(
        self, entity_ids: list[str], history_period: int
    ) -> dict[str, Any]:
        """Calculate timeslot-based probabilities (fallback logic)."""
        if not self._needs_cache_update():
            return self._cache

        end_time = dt_util.utcnow()
        start_time = end_time - timedelta(days=history_period)
        timeslots = {}

        for hour in range(24):
            for minute in (0, 30):
                slot_data = await self._process_timeslot(
                    entity_ids, start_time, end_time, hour, minute
                )
                time_key = f"{hour:02d}:{minute:02d}"
                timeslots[time_key] = self._cache[time_key] = slot_data

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
        slot_data = {"entities": []}
        combined_true = 1.0
        combined_false = 1.0

        for entity_id in entity_ids:
            daily_probs = []
            current = start_time

            while current < end_time:
                slot_start = current.replace(
                    hour=hour, minute=minute, second=0, microsecond=0
                )
                slot_end = slot_start + TIMESLOT_DURATION

                if slot_end > end_time:
                    break

                p_true, p_false, _ = await self.calculate_prior(
                    entity_id, slot_start, slot_end
                )
                daily_probs.append((p_true, p_false))
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
        sensor_states: dict[str, dict[str, Any]],
        motion_timestamps: dict[str, datetime],
        timeslot: Timeslot | None = None,
    ) -> ProbabilityResult:
        try:
            decay_min_delay = self.coordinator.get_decay_min_delay()

            # Determine prior
            if self.coordinator.data and "probability" in self.coordinator.data:
                previous_probability = self.coordinator.data["probability"]
            else:
                # No previous probability, use DEFAULT_PRIOR as a global starting prior
                previous_probability = DEFAULT_PRIOR

            current_probability = previous_probability
            active_triggers = []
            sensor_probs = {}

            for entity_id, state in sensor_states.items():
                if not state or not state.get("availability", False):
                    continue

                p_true, p_false = self._get_sensor_priors_from_history(entity_id)
                prior_val = self._get_sensor_prior(entity_id)
                if p_true is None or p_false is None:
                    p_true, p_false = self.get_timeslot_probabilities(
                        entity_id, timeslot
                    )
                    if p_true is None or p_false is None:
                        p_true, p_false = self.get_sensor_priors(entity_id)
                        prior_val = get_default_prior(entity_id, self)

                is_active = self._is_active_now(entity_id, state["state"])
                if is_active:
                    active_triggers.append(entity_id)
                    # Use Bayesian update: current_probability is our running posterior -> next prior
                    current_probability = update_probability(prior_val, p_true, p_false)

                sensor_probs[entity_id] = current_probability

            current_probability = max(
                MIN_PROBABILITY, min(current_probability, MAX_PROBABILITY)
            )

            now = dt_util.utcnow()
            decay_status = {}

            if active_triggers:
                self.coordinator.set_last_positive_trigger(now)
                decay_status["global_decay"] = 0.0
            else:
                last_trigger = self.coordinator.get_last_positive_trigger()
                if last_trigger is not None:
                    elapsed = (now - last_trigger).total_seconds()
                    decay_window = self.coordinator.get_decay_window()
                    if elapsed > decay_min_delay and elapsed > 0 and decay_window > 0:
                        decay_factor = math.exp(
                            -DECAY_LAMBDA * (elapsed / decay_window)
                        )
                        current_probability *= decay_factor
                        current_probability = max(
                            MIN_PROBABILITY, min(current_probability, MAX_PROBABILITY)
                        )
                        decay_status["global_decay"] = round(1.0 - decay_factor, 4)
                    else:
                        decay_status["global_decay"] = 0.0
                else:
                    decay_status["global_decay"] = 0.0

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
        """Attempt to retrieve learned priors for a sensor from coordinator data."""
        priors = self.coordinator.learned_priors.get(entity_id)
        if priors:
            return priors["prob_given_true"], priors["prob_given_false"]
        return None, None

    def get_timeslot_probabilities(
        self, entity_id: str, timeslot: Timeslot | None
    ) -> tuple[float, float] | tuple[None, None]:
        """Get probability from timeslot data if learned priors are unavailable."""
        if timeslot and "entities" in timeslot:
            entity_data = next(
                (e for e in timeslot["entities"] if e["id"] == entity_id),
                None,
            )
            if entity_data:
                return entity_data["prob_given_true"], entity_data["prob_given_false"]
        return None, None

    def _get_sensor_prior(self, entity_id: str) -> float:
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
