"""Test Area Occupancy number platform."""

from unittest.mock import patch

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.area_occupancy.const import DOMAIN, CONF_THRESHOLD


async def test_threshold_number(hass: HomeAssistant, init_integration):
    """Test threshold number entity."""
    state = hass.states.get("number.area_occupancy_threshold")
    assert state is not None
    assert float(state.state) == 50  # Default threshold value

    # Test setting new value
    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data"
    ):
        await hass.services.async_call(
            Platform.NUMBER,
            "set_value",
            {"entity_id": "number.area_occupancy_threshold", "value": 0.7},
            blocking=True,
        )
        await hass.async_block_till_done()

    state = hass.states.get("number.area_occupancy_threshold")
    assert float(state.state) == 0.7

    # Verify coordinator was updated
    coordinator = hass.data[DOMAIN][init_integration.entry_id]["coordinator"]
    assert coordinator.options_config[CONF_THRESHOLD] == 0.7
