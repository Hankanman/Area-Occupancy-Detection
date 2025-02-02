"""Test Area Occupancy number platform."""

from unittest.mock import patch

from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from custom_components.area_occupancy.const import DOMAIN, CONF_THRESHOLD


async def test_threshold_number(hass: HomeAssistant, init_integration):
    """Test threshold number entity."""
    state = hass.states.get("number.test_area_occupancy_threshold")
    assert state is not None
    assert state.state == "50"
    assert state.attributes["min"] == 1.0
    assert state.attributes["max"] == 99.0
    assert state.attributes["step"] == 1.0
    assert state.attributes["unit_of_measurement"] == "%"

    # Test setting new value (as percentage)
    with patch(
        "custom_components.area_occupancy.coordinator.AreaOccupancyCoordinator._async_update_data"
    ):
        await hass.services.async_call(
            Platform.NUMBER,
            "set_value",
            {"entity_id": "number.test_area_occupancy_threshold", "value": 70.0},
            blocking=True,
        )
        await hass.async_block_till_done()

    state = hass.states.get("number.test_area_occupancy_threshold")
    assert float(state.state) == 70.0

    # Verify coordinator was updated with percentage value
    coordinator = hass.data[DOMAIN][init_integration.entry_id]["coordinator"]
    assert coordinator.config[CONF_THRESHOLD] == 70.0
