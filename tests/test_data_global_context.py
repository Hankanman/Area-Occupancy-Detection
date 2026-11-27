"""Tests for optional house-wide context signals."""

from datetime import datetime

from custom_components.area_occupancy.const import (
    CONF_GLOBAL_ACTIVITY_ENTITY,
    CONF_GLOBAL_POWER_ENTITY,
)
from custom_components.area_occupancy.data.global_context import GlobalContextProvider
from homeassistant.core import HomeAssistant


def test_missing_entities_are_ignored(hass: HomeAssistant, coordinator) -> None:
    """Unconfigured or missing entities must not make context unavailable."""
    coordinator.config_entry.options = {CONF_GLOBAL_POWER_ENTITY: "sensor.power"}
    context = GlobalContextProvider(hass, coordinator.integration_config).snapshot(
        datetime.now()
    )

    assert context.signals["power"].available is False
    assert context.activity_boost(away=False) == 0.0


async def test_numeric_activity_signal_is_active(
    hass: HomeAssistant, coordinator
) -> None:
    """A positive numeric activity signal adds the small opt-in boost."""
    hass.states.async_set("sensor.activity", "1")
    coordinator.config_entry.options = {CONF_GLOBAL_ACTIVITY_ENTITY: "sensor.activity"}
    context = GlobalContextProvider(hass, coordinator.integration_config).snapshot(
        datetime.now()
    )

    assert context.is_active("activity") is True
    assert context.activity_boost(away=False) > 0
    assert context.activity_boost(away=True) == 0.0
