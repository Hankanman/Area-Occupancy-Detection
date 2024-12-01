"""Coordinator for Area Occupancy Detection."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import timedelta
from typing import Any

from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.const import STATE_ON, STATE_OFF
from homeassistant.util import dt as dt_util

from . import AreaConfig, DataProvider, ProbabilityResult, SensorState
from ..core.calculators.probability import ProbabilityCalculator
from ..core.calculators.pattern import PatternAnalyzer
from ..core.calculators.historical import HistoricalAnalyzer
from ..const import DOMAIN

_LOGGER = logging.getLogger(__name__)


class StateCache:
    """Efficient state caching with change detection."""

    def __init__(self, max_size: int = 100):
        """Initialize the cache."""
        self._states: dict[str, deque[SensorState]] = defaultdict(
            lambda: deque(maxlen=max_size)
        )
        self._last_result: dict[str, Any] = {}
        self._last_update = dt_util.utcnow()
        self._min_update_interval = timedelta(seconds=1)
        self._change_threshold = 0.1  # 10% change threshold

    def has_significant_changes(self, new_states: dict[str, SensorState]) -> bool:
        """Detect if updates warrant recalculation."""
        now = dt_util.utcnow()
        if now - self._last_update < self._min_update_interval:
            return False

        for entity_id, state in new_states.items():
            if self._is_significant_change(entity_id, state):
                return True

        return False

    def _is_significant_change(self, entity_id: str, state: SensorState) -> bool:
        """Check if state change is significant."""
        if entity_id not in self._states or not self._states[entity_id]:
            return True

        last_state = self._states[entity_id][-1]

        # Always consider motion sensor changes significant
        if "motion" in entity_id and state.state != last_state.state:
            return True

        # Check numeric sensors for significant changes
        try:
            if isinstance(state.state, (int, float)) and isinstance(
                last_state.state, (int, float)
            ):
                return (
                    abs(float(state.state) - float(last_state.state))
                    > self._change_threshold
                )
        except (ValueError, TypeError):
            _LOGGER.warning(
                "Failed to compare sensor states for %s: %s -> %s",
                entity_id,
                last_state.state,
                state.state,
            )
            return False

        # Check binary states
        if state.state in (STATE_ON, STATE_OFF):
            return state.state != last_state.state

        return False

    def update(self, states: dict[str, SensorState], result: ProbabilityResult) -> None:
        """Update cache with new states and result."""
        for entity_id, state in states.items():
            self._states[entity_id].append(state)
        self._last_result = result
        self._last_update = dt_util.utcnow()

    def get_last_result(self) -> ProbabilityResult:
        """Get the last calculated result."""
        return self._last_result


class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityResult]):
    """Coordinates updates and calculations for area occupancy detection."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        config: AreaConfig,
        data_provider: DataProvider,
    ) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=5),
        )

        self.entry_id = entry_id
        self.config = config
        self._data_provider = data_provider
        self._state_lock = asyncio.Lock()
        self._cache = StateCache()

        self._probability_calculator = ProbabilityCalculator(config)
        self._pattern_analyzer = PatternAnalyzer()
        self._historical_analyzer = HistoricalAnalyzer(hass, config)

        self._unsubscribe_handlers: list[callable] = []

    async def _async_update_data(self) -> ProbabilityResult:
        """Update data with optimized calculations."""
        async with self._state_lock:
            try:
                sensor_states = await self._data_provider.get_sensor_states()

                if not self._cache.has_significant_changes(sensor_states):
                    return self._cache.get_last_result()

                historical_data = await self._data_provider.get_historical_data()

                base_result = self._probability_calculator.calculate(sensor_states)
                pattern_adjustment = self._pattern_analyzer.calculate(sensor_states)
                historical_adjustment = await self._historical_analyzer.calculate(
                    sensor_states, historical_data
                )

                final_result = self._combine_results(
                    base_result, pattern_adjustment, historical_adjustment
                )

                self._cache.update(sensor_states, final_result)

                return final_result

            except Exception as err:
                _LOGGER.error(
                    "Error updating area occupancy data: %s", err, exc_info=True
                )
                raise

    def _combine_results(
        self,
        base: ProbabilityResult,
        pattern: float,
        historical: float,
    ) -> ProbabilityResult:
        """Combine different probability calculations."""
        pattern_weight = 0.2
        historical_weight = 0.1
        base_weight = 1.0 - (pattern_weight + historical_weight)

        final_probability = (
            base.probability * base_weight
            + pattern * pattern_weight
            + historical * historical_weight
        )

        return ProbabilityResult(
            probability=max(0.0, min(1.0, final_probability)),
            prior_probability=base.prior_probability,
            active_triggers=base.active_triggers,
            sensor_probabilities=base.sensor_probabilities,
            decay_status=base.decay_status,
            confidence_score=base.confidence_score,
            sensor_availability=base.sensor_availability,
            device_states=base.device_states,
            last_occupied=base.last_occupied,
            state_duration=base.state_duration,
            occupancy_rate=base.occupancy_rate,
            moving_average=base.moving_average,
            rate_of_change=base.rate_of_change,
            min_probability=base.min_probability,
            max_probability=base.max_probability,
        )

    async def async_setup(self) -> None:
        """Perform async setup tasks."""
        try:
            await self._historical_analyzer.async_initialize()
            self._setup_state_listeners()
            await self.async_config_entry_first_refresh()
        except Exception as err:
            _LOGGER.error("Failed to setup area occupancy coordinator: %s", err)
            await self.cleanup()
            raise

    async def cleanup(self) -> None:
        """Clean up resources."""
        _LOGGER.debug("Cleaning up AreaOccupancyCoordinator")
        for unsub in self._unsubscribe_handlers:
            unsub()
        self._unsubscribe_handlers.clear()

    @callback
    def _setup_state_listeners(self) -> None:
        """Set up state change listeners."""
        entities = (
            self.config.motion_sensors
            + self.config.media_devices
            + self.config.appliances
            + self.config.illuminance_sensors
            + self.config.humidity_sensors
            + self.config.temperature_sensors
        )

        @callback
        def async_state_changed(event) -> None:
            """Handle state changes."""
            self.async_set_updated_data(None)

        while self._unsubscribe_handlers:
            self._unsubscribe_handlers.pop()()

        self._unsubscribe_handlers.append(
            async_track_state_change_event(self.hass, entities, async_state_changed)
        )
