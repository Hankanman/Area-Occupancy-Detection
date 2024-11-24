"""Tests for Room Occupancy Detection integration initialization."""
from unittest.mock import patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.config_entries import ConfigEntryState

from custom_components.room_occupancy.const import DOMAIN

async def test_setup_entry(hass: HomeAssistant, mock_fully_setup_entry):
    """Test setting up an entry loads the necessary platforms."""
    assert len(hass.config_entries.async_entries(DOMAIN)) == 1
    
    entry = mock_fully_setup_entry
    assert entry.state == ConfigEntryState.LOADED
    
    # Verify the platforms were set up
    assert hass.states.get("sensor.test_room_occupancy_probability") is not None
    assert hass.states.get("binary_sensor.test_room_occupancy_status") is not None

async def test_unload_entry(hass: HomeAssistant, mock_fully_setup_entry):
    """Test unloading an entry."""
    entry = mock_fully_setup_entry
    
    # Unload the entry
    assert await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
    
    # Verify the entry was unloaded
    assert entry.state == ConfigEntryState.NOT_LOADED
    assert hass.states.get("sensor.test_room_occupancy_probability") is None
    assert hass.states.get("binary_sensor.test_room_occupancy_status") is None

async def test_setup_entry_fails_not_ready(hass: HomeAssistant, mock_config_entry):
    """Test setup entry fails if required entities are not ready."""
    # Don't set up any mock entities
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=mock_config_entry,
        entry_id="test_entry_id"
    )
    entry.add_to_hass(hass)
    
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    
    assert entry.state == ConfigEntryState.SETUP_RETRY

async def test_setup_entry_migration(hass: HomeAssistant, mock_config_entry):
    """Test migrating an entry from an old version."""
    old_config = dict(mock_config_entry)
    old_config.pop(CONF_DECAY_TYPE)  # Simulate old config without decay type
    
    entry = MockConfigEntry(
        domain=DOMAIN,
        data=old_config,
        entry_id="test_entry_id",
        version=1
    )
    entry.add_to_hass(hass)
    
    await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    
    # Verify migration added new fields with defaults
    assert entry.data.get(CONF_DECAY_TYPE) == "linear"
