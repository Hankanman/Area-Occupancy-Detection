"""Tests for the Probabilities class."""

import pytest

from custom_components.area_occupancy.exceptions import ConfigurationError
from custom_components.area_occupancy.probabilities import Probabilities
from custom_components.area_occupancy.types import EntityType, PriorData

# Minimal valid config for all types
VALID_CONFIG = {
    "motion_sensors": ["binary_sensor.motion1"],
    "media_devices": ["media_player.media1"],
    "appliances": ["switch.appliance1"],
    "door_sensors": ["binary_sensor.door1"],
    "window_sensors": ["binary_sensor.window1"],
    "lights": ["light.light1"],
    "illuminance_sensors": ["sensor.illuminance1"],
    "humidity_sensors": ["sensor.humidity1"],
    "temperature_sensors": ["sensor.temp1"],
    "weight_motion": 0.8,
    "weight_media": 0.7,
    "weight_appliance": 0.6,
    "weight_door": 0.5,
    "weight_window": 0.4,
    "weight_light": 0.3,
    "weight_environmental": 0.2,
    "media_active_states": ["playing", "paused"],
    "appliance_active_states": ["on", "standby"],
    "door_active_state": "closed",
    "window_active_state": "open",
}


def test_probabilities_init_valid():
    """Test that the Probabilities class initializes correctly with valid configuration."""
    p = Probabilities(VALID_CONFIG)
    assert p.sensor_weights["motion"] == 0.8
    assert p.entity_types["binary_sensor.motion1"] == EntityType.MOTION
    assert p.sensor_configs["motion"]["weight"] == 0.8


def test_probabilities_init_invalid_entity_id():
    """Test that the Probabilities class raises a ConfigurationError when an invalid entity ID is provided."""
    bad_config = dict(VALID_CONFIG)
    bad_config["motion_sensors"] = ["badid"]
    with pytest.raises(ConfigurationError):
        Probabilities(bad_config)


def test_probabilities_init_invalid_weight():
    """Test that the Probabilities class raises a ConfigurationError when an invalid weight is provided."""
    bad_config = dict(VALID_CONFIG)
    bad_config["weight_motion"] = 2.0
    with pytest.raises(ConfigurationError):
        Probabilities(bad_config)


def test_probabilities_init_nonlist_entities():
    """Test that the Probabilities class raises a ConfigurationError when a non-list entity is provided."""
    bad_config = dict(VALID_CONFIG)
    bad_config["motion_sensors"] = "notalist"
    with pytest.raises(ConfigurationError):
        Probabilities(bad_config)


def test_probabilities_translate_binary_sensor_active_state():
    """Test that the Probabilities class translates binary sensor active states correctly."""
    p = Probabilities(VALID_CONFIG)
    assert p._translate_binary_sensor_active_state("Open") == "on"
    assert p._translate_binary_sensor_active_state("Closed") == "off"
    assert p._translate_binary_sensor_active_state("foo") == "off"


def test_probabilities_build_sensor_configs_missing_key():
    """Test that the Probabilities class uses default weights when a key is missing."""
    bad_config = dict(VALID_CONFIG)
    del bad_config["weight_motion"]
    # Should use default, not error
    p = Probabilities(bad_config)
    assert p.sensor_weights["motion"] > 0


def test_probabilities_get_default_prior_valid():
    """Test that the Probabilities class returns a default prior for a valid entity ID."""
    p = Probabilities(VALID_CONFIG)
    eid = "binary_sensor.motion1"
    val = p.get_default_prior(eid)
    assert 0 < val < 1


def test_probabilities_get_default_prior_missing_entity():
    """Test that the Probabilities class raises a ValueError when an invalid entity ID is provided."""
    p = Probabilities(VALID_CONFIG)
    with pytest.raises(ValueError):
        p.get_default_prior("binary_sensor.unknown")


def test_probabilities_update_config_valid():
    """Test that the Probabilities class updates the configuration correctly."""
    p = Probabilities(VALID_CONFIG)
    new_config = dict(VALID_CONFIG)
    new_config["weight_motion"] = 0.5
    p.update_config(new_config)
    assert p.sensor_weights["motion"] == 0.5


def test_probabilities_update_config_invalid():
    """Test that the Probabilities class raises a ConfigurationError when an invalid weight is provided."""
    p = Probabilities(VALID_CONFIG)
    bad_config = dict(VALID_CONFIG)
    bad_config["weight_motion"] = -1
    with pytest.raises(ConfigurationError):
        p.update_config(bad_config)


def test_probabilities_get_sensor_config_valid():
    """Test that the Probabilities class returns a sensor configuration for a valid entity ID."""
    p = Probabilities(VALID_CONFIG)
    eid = "binary_sensor.motion1"
    cfg = p.get_sensor_config(eid)
    assert isinstance(cfg, dict)
    assert "weight" in cfg


def test_probabilities_get_sensor_config_missing_entity():
    """Test that the Probabilities class raises a ValueError when an invalid entity ID is provided."""
    p = Probabilities(VALID_CONFIG)
    with pytest.raises(ValueError):
        p.get_sensor_config("binary_sensor.unknown")


def test_probabilities_is_entity_active_true_false():
    """Test that the Probabilities class returns True for active states and False for inactive states."""
    p = Probabilities(VALID_CONFIG)
    eid = "binary_sensor.motion1"
    # Should be active for 'on', inactive for 'off', False for None
    assert p.is_entity_active(eid, "on") is True
    assert p.is_entity_active(eid, "off") is False
    assert p.is_entity_active(eid, None) is False


def test_probabilities_is_entity_active_missing_config():
    """Test that the Probabilities class raises a ValueError when an invalid entity ID is provided."""
    p = Probabilities(VALID_CONFIG)
    with pytest.raises(ValueError):
        p.is_entity_active("binary_sensor.unknown", "on")


def test_probabilities_get_initial_type_priors():
    """Test that the Probabilities class returns a dictionary of initial type priors."""
    p = Probabilities(VALID_CONFIG)
    priors = p.get_initial_type_priors()
    assert isinstance(priors, dict)
    for v in priors.values():
        assert isinstance(v, PriorData)


def test_probabilities_get_entity_type():
    """Test that the Probabilities class returns the correct entity type for a valid entity ID."""
    p = Probabilities(VALID_CONFIG)
    eid = "binary_sensor.motion1"
    assert p.get_entity_type(eid) == EntityType.MOTION
    assert p.get_entity_type("binary_sensor.unknown") is None
