"""Tests for the Area Occupancy Detection config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
import voluptuous as vol

from custom_components.area_occupancy.config_flow import (
    AreaOccupancyConfigFlow,
    AreaOccupancyOptionsFlow,
    BaseOccupancyFlow,
    _apply_purpose_based_decay_default,
    _build_area_description_placeholders,
    _create_action_selection_schema,
    _create_area_selector_schema,
    _entity_contains_keyword,
    _find_area_by_id,
    _find_area_by_sanitized_id,
    _flatten_sectioned_input,
    _get_area_summary_info,
    _get_include_entities,
    _get_purpose_display_name,
    _get_state_select_options,
    _handle_step_error,
    _is_weather_entity,
    _remove_area_from_list,
    _update_area_in_list,
    create_schema,
)
from custom_components.area_occupancy.const import (
    CONF_ACTION_CANCEL,
    CONF_ACTION_EDIT,
    CONF_ACTION_REMOVE,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_PROB_GIVEN_FALSE,
    CONF_MOTION_PROB_GIVEN_TRUE,
    CONF_MOTION_SENSORS,
    CONF_OPTION_PREFIX_AREA,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_PURPOSE,
    DOMAIN,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import AbortFlow, FlowResultType
from homeassistant.exceptions import HomeAssistantError
from tests.conftest import create_area_config, patch_create_schema_context


# ruff: noqa: SLF001, TID251, PLC0415
@pytest.mark.parametrize("expected_lingering_timers", [True])
class TestBaseOccupancyFlow:
    """Test BaseOccupancyFlow class."""

    @pytest.fixture
    def flow(self):
        """Create a BaseOccupancyFlow instance."""
        return BaseOccupancyFlow()

    @pytest.mark.parametrize(
        ("config_modification", "should_have_errors", "expected_error_key"),
        [
            ({}, False, None),  # basic_valid
            (
                {"decay_enabled": True, "decay_half_life": 0},
                False,
                None,
            ),  # decay_zero_valid
            ({"weight_motion": 0.0}, False, None),  # weight_min_valid
            ({"weight_motion": 1.0}, False, None),  # weight_max_valid
            (
                {CONF_AREA_ID: "nonexistent_area_id_12345"},
                True,
                "area_not_found",
            ),  # invalid_area_id
        ],
    )
    def test_validate_config_valid_scenarios(
        self,
        flow,
        config_flow_base_config,
        hass,
        config_modification,
        should_have_errors,
        expected_error_key,
    ):
        """Test validating various valid and invalid configuration scenarios."""
        test_config = {**config_flow_base_config, **config_modification}

        errors = flow._validate_config(test_config, hass)
        if should_have_errors:
            assert errors, f"Expected errors but got none for {config_modification}"
            assert expected_error_key in errors.values()
        else:
            assert not errors, f"Expected no errors but got {errors}"

    @pytest.mark.parametrize(
        ("invalid_config", "expected_error_key"),
        [
            (
                {"motion_sensors": []},
                "motion_required",
            ),
            (
                {"weight_motion": 1.5},
                "invalid_weight",
            ),
            (
                {"threshold": 150},
                "invalid_threshold",
            ),
            (
                {"threshold": 0},
                "invalid_threshold",
            ),
            (
                {"threshold": 101},
                "invalid_threshold",
            ),
            (
                {CONF_AREA_ID: ""},
                "area_required",
            ),
            (
                {"decay_enabled": True, "decay_half_life": -1},
                "invalid_decay_half_life",
            ),
            (
                {"decay_enabled": True, "decay_half_life": 5},
                "invalid_decay_half_life",
            ),
            (
                {"decay_enabled": True, "decay_half_life": 3601},
                "invalid_decay_half_life",
            ),
            (
                {CONF_PURPOSE: ""},
                "purpose_required",
            ),
            (
                {CONF_MEDIA_DEVICES: ["media_player.tv"], CONF_MEDIA_ACTIVE_STATES: []},
                "media_states_required",
            ),
            (
                {CONF_APPLIANCES: ["switch.light"], CONF_APPLIANCE_ACTIVE_STATES: []},
                "appliance_states_required",
            ),
            (
                {
                    CONF_DOOR_SENSORS: ["binary_sensor.door1"],
                    CONF_DOOR_ACTIVE_STATE: "",
                },
                "door_state_required",
            ),
            (
                {
                    CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
                    CONF_WINDOW_ACTIVE_STATE: "",
                },
                "window_state_required",
            ),
            (
                {
                    CONF_MOTION_PROB_GIVEN_TRUE: 0.5,
                    CONF_MOTION_PROB_GIVEN_FALSE: 0.6,
                },
                "prob_true_must_exceed_false",
            ),
            (
                {
                    CONF_MOTION_PROB_GIVEN_TRUE: 0.5,
                    CONF_MOTION_PROB_GIVEN_FALSE: 0.5,
                },
                "prob_true_must_exceed_false",
            ),
        ],
    )
    def test_validate_config_invalid_scenarios(
        self, flow, config_flow_base_config, invalid_config, expected_error_key, hass
    ):
        """Test various invalid configuration scenarios."""
        test_config = {**config_flow_base_config, **invalid_config}
        # Remove None values to test missing keys
        test_config = {k: v for k, v in test_config.items() if v is not None}

        errors = flow._validate_config(test_config, hass)
        assert errors, f"Expected errors but got none for {invalid_config}"
        assert expected_error_key in errors.values(), (
            f"Expected error key '{expected_error_key}' in errors {errors}"
        )


class TestHelperFunctions:
    """Test helper functions."""

    @pytest.mark.parametrize(
        "platform",
        ["door", "window", "media", "appliance", "unknown"],
    )
    def test_get_state_select_options(self, platform):
        """Test _get_state_select_options function for all platforms."""
        options = _get_state_select_options(platform)
        assert isinstance(options, list)
        assert len(options) > 0
        # Validate structure and content
        for option in options:
            assert "value" in option
            assert "label" in option
            assert isinstance(option["value"], str)
            assert isinstance(option["label"], str)
            assert len(option["value"]) > 0  # Values should not be empty
            assert len(option["label"]) > 0  # Labels should not be empty

    @pytest.mark.parametrize(
        ("purpose", "expected"),
        [
            ("social", None),  # Valid - check it's a non-empty string
            ("invalid_purpose", "Invalid Purpose"),  # Invalid - check exact fallback
        ],
    )
    def test_get_purpose_display_name(self, purpose, expected):
        """Test _get_purpose_display_name function."""
        result = _get_purpose_display_name(purpose)
        if expected is None:
            # Valid purpose - just check it's a non-empty string
            assert isinstance(result, str)
            assert len(result) > 0
        else:
            # Invalid purpose - check exact fallback
            assert result == expected

    @pytest.mark.parametrize(
        ("areas", "sanitized_id", "expected_id"),
        [
            (
                [{CONF_AREA_ID: "living_room", CONF_PURPOSE: "social"}],
                "living_room",
                "living_room",
            ),
            (
                [
                    {CONF_AREA_ID: "living_room", CONF_PURPOSE: "social"},
                    {CONF_AREA_ID: "kitchen", CONF_PURPOSE: "work"},
                ],
                "bedroom",
                None,
            ),
            ([], "living_room", None),
        ],
    )
    def test_find_area_by_sanitized_id(self, areas, sanitized_id, expected_id):
        """Test _find_area_by_sanitized_id function."""
        result = _find_area_by_sanitized_id(areas, sanitized_id)
        if expected_id is None:
            assert result is None
        else:
            assert result is not None
            assert result[CONF_AREA_ID] == expected_id

    def test_build_area_description_placeholders(self):
        """Test _build_area_description_placeholders function."""
        area_config = {
            CONF_AREA_ID: "living_room",
            CONF_PURPOSE: "social",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_DOOR_SENSORS: ["binary_sensor.door1"],
            CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
            CONF_APPLIANCES: ["switch.light"],
            CONF_THRESHOLD: 60.0,
        }

        placeholders = _build_area_description_placeholders(
            area_config, "living_room", hass=None
        )

        assert (
            placeholders["area_name"] == "living_room"
        )  # Uses area_id when hass is None
        assert placeholders["motion_count"] == "1"
        assert placeholders["media_count"] == "1"
        assert placeholders["door_count"] == "1"
        assert placeholders["window_count"] == "1"
        assert placeholders["appliance_count"] == "1"
        assert placeholders["threshold"] == "60.0"

    def test_get_area_summary_info(self):
        """Test _get_area_summary_info function."""
        area = {
            CONF_AREA_ID: "living_room",
            CONF_PURPOSE: "social",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_DOOR_SENSORS: ["binary_sensor.door1"],
            CONF_WINDOW_SENSORS: [],
            CONF_APPLIANCES: [],
            CONF_THRESHOLD: 60.0,
        }

        summary = _get_area_summary_info(area)
        assert isinstance(summary, str)
        assert "living_room" not in summary  # Area ID should not be in summary
        assert "60" in summary  # Threshold should be included
        assert "3" in summary  # Total sensors count

    @pytest.mark.parametrize(
        "areas",
        [
            (
                [
                    {
                        CONF_AREA_ID: "living_room",
                        CONF_PURPOSE: "social",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                        CONF_THRESHOLD: 60.0,
                    }
                ]
            ),
            ([]),
        ],
    )
    def test_create_area_selector_schema(self, areas):
        """Test _create_area_selector_schema function."""
        schema = _create_area_selector_schema(areas)
        assert isinstance(schema, vol.Schema)

        # Validate schema structure
        schema_dict = schema.schema
        assert "selected_option" in schema_dict

        # If areas provided, validate options match
        if areas and len(areas) > 0:
            # Get the selector config
            selector = schema_dict["selected_option"]
            # Schema uses vol.Required wrapper, so we need to access the selector
            # The actual validation happens when schema is used, but we can check structure
            assert selector is not None

    def test_create_action_selection_schema(self):
        """Test _create_action_selection_schema function."""
        schema = _create_action_selection_schema()
        assert isinstance(schema, vol.Schema)

        # Validate schema structure
        schema_dict = schema.schema
        assert "action" in schema_dict

        # Validate that schema can be used with expected action values
        valid_actions = [CONF_ACTION_EDIT, CONF_ACTION_REMOVE, CONF_ACTION_CANCEL]
        for action in valid_actions:
            # Should not raise when using valid action
            result = schema({"action": action})
            assert result["action"] == action

        # Invalid action should raise
        with pytest.raises(vol.Invalid):
            schema({"action": "invalid_action"})

    def test_entity_contains_keyword_in_entity_id(self, hass):
        """Test _entity_contains_keyword finds keyword in entity_id."""
        # Create a state
        hass.states.async_set("binary_sensor.test_window_sensor", "off", {})

        # Test that keyword is found in entity_id
        assert _entity_contains_keyword(
            hass, "binary_sensor.test_window_sensor", "window"
        )
        assert not _entity_contains_keyword(
            hass, "binary_sensor.test_window_sensor", "door"
        )

    def test_entity_contains_keyword_in_friendly_name(self, hass):
        """Test _entity_contains_keyword finds keyword in friendly name."""
        # Create a state with friendly name
        hass.states.async_set(
            "binary_sensor.test_sensor_1",
            "off",
            {"friendly_name": "Living Room Window"},
        )

        # Test that keyword is found in friendly name
        assert _entity_contains_keyword(hass, "binary_sensor.test_sensor_1", "window")
        assert _entity_contains_keyword(hass, "binary_sensor.test_sensor_1", "living")
        assert not _entity_contains_keyword(hass, "binary_sensor.test_sensor_1", "door")

    def test_entity_contains_keyword_case_insensitive(self, hass):
        """Test _entity_contains_keyword is case insensitive."""
        # Create a state with mixed case friendly name
        hass.states.async_set(
            "binary_sensor.test_sensor_2",
            "off",
            {"friendly_name": "Front DOOR Sensor"},
        )

        # Test case insensitivity
        assert _entity_contains_keyword(hass, "binary_sensor.test_sensor_2", "door")
        assert _entity_contains_keyword(hass, "binary_sensor.test_sensor_2", "DOOR")
        assert _entity_contains_keyword(hass, "binary_sensor.test_sensor_2", "Door")
        assert _entity_contains_keyword(hass, "binary_sensor.test_sensor_2", "front")

    def test_entity_contains_keyword_no_state(self, hass):
        """Test _entity_contains_keyword handles missing state gracefully."""
        # Test with entity that doesn't exist
        result = _entity_contains_keyword(hass, "binary_sensor.nonexistent", "window")
        assert not result

    def test_get_include_entities(self, hass, entity_registry):
        """Test getting include entities."""
        # Register entities
        entity_registry.async_get_or_create(
            "binary_sensor", "test", "door_1", original_device_class="door"
        )
        entity_registry.async_get_or_create(
            "binary_sensor", "test", "window_1", original_device_class="window"
        )
        entity_registry.async_get_or_create("switch", "test", "appliance_1")

        # Create states
        hass.states.async_set(
            "binary_sensor.test_door_1", "off", {"device_class": "door"}
        )
        hass.states.async_set(
            "binary_sensor.test_window_1", "off", {"device_class": "window"}
        )
        hass.states.async_set("switch.test_appliance_1", "off")

        result = _get_include_entities(hass)

        assert "door" in result
        assert "window" in result
        assert "appliance" in result
        assert "binary_sensor.test_door_1" in result["door"]
        assert "binary_sensor.test_window_1" in result["window"]
        assert "switch.test_appliance_1" in result["appliance"]

    def test_get_include_entities_window_by_original_device_class(
        self, hass, entity_registry
    ):
        """Test that window sensors are detected by original_device_class.

        This tests the fix for the issue where binary_sensor.window type devices
        with only original_device_class set (not device_class) and without
        'window' in the entity_id were not showing up in the window picker.
        """
        # Register a window sensor with only original_device_class set
        # and an entity_id that doesn't contain "window"
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "living_room_contact",  # No 'window' in name
            original_device_class="window",  # original_device_class is 'window'
        )

        # Create state without device_class attribute (simulating real sensor)
        hass.states.async_set("binary_sensor.test_living_room_contact", "off", {})

        result = _get_include_entities(hass)

        # The entity should appear in the window list because of original_device_class
        assert "window" in result
        assert "binary_sensor.test_living_room_contact" in result["window"]

    def test_get_include_entities_door_by_original_device_class(
        self, hass, entity_registry
    ):
        """Test that door sensors are detected by original_device_class.

        This tests the fix for the issue where binary_sensor.door type devices
        with only original_device_class set (not device_class) and without
        'door' in the entity_id were not showing up in the door picker.
        """
        # Register a door sensor with only original_device_class set
        # and an entity_id that doesn't contain "door"
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "front_entrance_contact",  # No 'door' in name
            original_device_class="door",  # original_device_class is 'door'
        )

        # Create state without device_class attribute (simulating real sensor)
        hass.states.async_set("binary_sensor.test_front_entrance_contact", "off", {})

        result = _get_include_entities(hass)

        # The entity should appear in the door list because of original_device_class
        assert "door" in result
        assert "binary_sensor.test_front_entrance_contact" in result["door"]

    def test_get_include_entities_window_by_friendly_name(self, hass, entity_registry):
        """Test that window sensors are detected by friendly name.

        This tests that entities with 'window' in their friendly name (user-visible name)
        are correctly detected as window sensors, even if the entity_id doesn't contain 'window'.
        """
        # Register a sensor with opening device class and an entity_id without 'window'
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "contact_sensor_1",  # No 'window' in entity_id
            original_device_class="opening",
        )

        # Create state with friendly name containing 'window'
        hass.states.async_set(
            "binary_sensor.test_contact_sensor_1",
            "off",
            {"friendly_name": "Living Room Window", "device_class": "opening"},
        )

        result = _get_include_entities(hass)

        # The entity should appear in the window list because of friendly name
        assert "window" in result
        assert "binary_sensor.test_contact_sensor_1" in result["window"]

    def test_get_include_entities_door_by_friendly_name(self, hass, entity_registry):
        """Test that door sensors are detected by friendly name.

        This tests that entities with 'door' in their friendly name (user-visible name)
        are correctly detected as door sensors, even if the entity_id doesn't contain 'door'.
        """
        # Register a sensor with opening device class and an entity_id without 'door'
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "contact_sensor_2",  # No 'door' in entity_id
            original_device_class="opening",
        )

        # Create state with friendly name containing 'door'
        hass.states.async_set(
            "binary_sensor.test_contact_sensor_2",
            "off",
            {"friendly_name": "Front Door Sensor", "device_class": "opening"},
        )

        result = _get_include_entities(hass)

        # The entity should appear in the door list because of friendly name
        assert "door" in result
        assert "binary_sensor.test_contact_sensor_2" in result["door"]

    def test_get_include_entities_prioritize_window_in_name_over_door(
        self, hass, entity_registry
    ):
        """Test that window keyword in name takes precedence over door classification.

        When an entity has 'window' in its friendly name, it should be categorized as
        a window sensor even if it has door-like device class.
        """
        # Register a sensor with door device class
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "contact_3",
            original_device_class="door",
        )

        # Create state with friendly name containing 'window'
        hass.states.async_set(
            "binary_sensor.test_contact_3",
            "off",
            {"friendly_name": "Bedroom Window Contact", "device_class": "door"},
        )

        result = _get_include_entities(hass)

        # The entity should appear in window list due to name, not door list
        assert "window" in result
        assert "binary_sensor.test_contact_3" in result["window"]
        # Should NOT be in door list
        assert "binary_sensor.test_contact_3" not in result.get("door", [])

    def test_get_include_entities_door_with_door_keyword_in_opening(
        self, hass, entity_registry
    ):
        """Test that entities with 'door' keyword and opening device class are detected as doors.

        This tests the fix for the issue where door entities were only showing up in the
        window dropdown. Entities with 'door' in their name/entity_id and device class
        'opening' should be categorized as door sensors.
        """
        # Register a sensor with opening device class and 'door' in entity_id
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "front_door_contact",  # Has 'door' in entity_id
            original_device_class="opening",
        )

        # Create state
        hass.states.async_set(
            "binary_sensor.test_front_door_contact",
            "off",
            {"friendly_name": "Front Door Contact", "device_class": "opening"},
        )

        result = _get_include_entities(hass)

        # The entity should appear in the door list
        assert "door" in result
        assert "binary_sensor.test_front_door_contact" in result["door"]
        # Should NOT be in window list
        assert "binary_sensor.test_front_door_contact" not in result.get("window", [])

    def test_get_include_entities_door_with_garage_door_class_and_door_keyword(
        self, hass, entity_registry
    ):
        """Test that garage door sensors with 'door' keyword are detected as doors.

        Entities with garage_door device class and 'door' in their name should be
        categorized as door sensors.
        """
        # Register a sensor with garage_door device class
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "garage_contact",  # No 'door' in entity_id
            original_device_class="garage_door",
        )

        # Create state with friendly name containing 'door'
        hass.states.async_set(
            "binary_sensor.test_garage_contact",
            "off",
            {"friendly_name": "Garage Door Sensor", "device_class": "garage_door"},
        )

        result = _get_include_entities(hass)

        # The entity should appear in the door list due to garage_door device class
        assert "door" in result
        assert "binary_sensor.test_garage_contact" in result["door"]
        # Should NOT be in window list
        assert "binary_sensor.test_garage_contact" not in result.get("window", [])

    def test_get_include_entities_door_with_both_keywords(self, hass, entity_registry):
        """Test that entities with both 'door' and 'window' keywords are doors."""
        # Register a sensor with opening device class and both keywords in entity_id
        entity_registry.async_get_or_create(
            "binary_sensor",
            "test",
            "door_window_contact",
            original_device_class="opening",
        )

        # Create state with friendly name containing both keywords
        hass.states.async_set(
            "binary_sensor.test_door_window_contact",
            "off",
            {"friendly_name": "Patio Door Window Sensor", "device_class": "opening"},
        )

        result = _get_include_entities(hass)

        # The entity should appear in the door list, not the window list
        assert "door" in result
        assert "binary_sensor.test_door_window_contact" in result["door"]
        assert "binary_sensor.test_door_window_contact" not in result.get("window", [])

    def test_is_weather_entity_by_platform(self):
        """Test that weather entities are detected by platform."""
        # Test known weather platforms
        assert _is_weather_entity("sensor.outdoor_temp", "weather") is True
        assert _is_weather_entity("sensor.temp", "openweathermap") is True
        assert _is_weather_entity("sensor.temp", "met") is True
        assert _is_weather_entity("sensor.temp", "accuweather") is True
        assert _is_weather_entity("sensor.temp", "dwd") is True
        assert _is_weather_entity("sensor.temp", "dwd_weather") is True

        # Test non-weather platforms
        assert _is_weather_entity("sensor.room_temp", "zha") is False
        assert _is_weather_entity("sensor.room_temp", "mqtt") is False
        assert _is_weather_entity("sensor.room_temp", "esphome") is False
        assert _is_weather_entity("sensor.bedroom_temp", "ecobee") is False

    def test_is_weather_entity_by_keyword(self):
        """Test that weather entities are detected by entity_id keywords."""
        # Test weather-related keywords in entity_id
        assert _is_weather_entity("sensor.weather_temperature", None) is True
        assert _is_weather_entity("sensor.forecast_humidity", None) is True

        # "outdoor" is intentionally NOT a keyword - too generic
        # Users may have legitimate outdoor sensors (porch, patio) they want to use
        assert _is_weather_entity("sensor.outdoor_pressure", None) is False
        assert (
            _is_weather_entity("sensor.ecobee_outdoor_temperature", "ecobee") is False
        )

        # Test non-weather entity_ids
        assert _is_weather_entity("sensor.living_room_temperature", None) is False
        assert _is_weather_entity("sensor.bedroom_humidity", None) is False
        assert (
            _is_weather_entity("sensor.ecobee_bedroom_temperature", "ecobee") is False
        )

    def test_get_include_entities_excludes_weather_sensors(self, hass, entity_registry):
        """Test that weather sensors are excluded from environmental entities."""
        # Register weather temperature sensor (should be excluded)
        entity_registry.async_get_or_create(
            "sensor",
            "weather",
            "outdoor_temp",
            original_device_class="temperature",
        )
        # Register room temperature sensor (should be included)
        entity_registry.async_get_or_create(
            "sensor",
            "zha",
            "living_room_temp",
            original_device_class="temperature",
        )
        # Register weather humidity sensor (should be excluded)
        entity_registry.async_get_or_create(
            "sensor",
            "openweathermap",
            "outdoor_humidity",
            original_device_class="humidity",
        )
        # Register room humidity sensor (should be included)
        entity_registry.async_get_or_create(
            "sensor",
            "mqtt",
            "bathroom_humidity",
            original_device_class="humidity",
        )

        result = _get_include_entities(hass)

        # Check that weather sensors are excluded
        assert "sensor.weather_outdoor_temp" not in result.get("temperature", [])
        assert "sensor.openweathermap_outdoor_humidity" not in result.get(
            "humidity", []
        )

        # Check that room sensors are included
        assert "sensor.zha_living_room_temp" in result["temperature"]
        assert "sensor.mqtt_bathroom_humidity" in result["humidity"]

    def test_get_include_entities_excludes_area_occupancy_motion(
        self, hass, entity_registry
    ):
        """Test that area occupancy sensors are excluded from motion list."""
        # Register an area_occupancy sensor (should be excluded)
        entity_registry.async_get_or_create(
            "binary_sensor",
            DOMAIN,
            "living_room_occupancy",
            original_device_class="occupancy",
        )
        # Register external motion sensor (should be included)
        entity_registry.async_get_or_create(
            "binary_sensor",
            "zha",
            "motion_sensor",
            original_device_class="motion",
        )
        # Register external occupancy sensor (should be included)
        entity_registry.async_get_or_create(
            "binary_sensor",
            "mqtt",
            "room_occupancy",
            original_device_class="occupancy",
        )
        # Register external presence sensor (should be included)
        entity_registry.async_get_or_create(
            "binary_sensor",
            "ble_monitor",
            "person_presence",
            original_device_class="presence",
        )

        result = _get_include_entities(hass)

        # Check that our own sensors are excluded
        assert f"binary_sensor.{DOMAIN}_living_room_occupancy" not in result["motion"]

        # Check that external sensors are included
        assert "binary_sensor.zha_motion_sensor" in result["motion"]
        assert "binary_sensor.mqtt_room_occupancy" in result["motion"]
        assert "binary_sensor.ble_monitor_person_presence" in result["motion"]

    @pytest.mark.parametrize(
        ("defaults", "is_options", "expected_name_present", "test_schema_validation"),
        [
            (None, False, True, False),  # defaults test
            (
                {
                    CONF_AREA_ID: "test_area",
                    CONF_MOTION_SENSORS: ["binary_sensor.motion_1"],
                },
                False,
                True,  # CONF_AREA_ID is always present in schema now
                True,
            ),  # with_defaults test
            (
                None,
                True,
                True,
                False,
            ),  # options_mode test - CONF_AREA_ID is always present
        ],
    )
    def test_create_schema(
        self,
        hass,
        entity_registry,
        defaults,
        is_options,
        expected_name_present,
        test_schema_validation,
    ):
        """Test creating schema with different configurations."""
        # Use real entity registry via fixture
        schema_dict = create_schema(hass, defaults, is_options)
        schema = vol.Schema(schema_dict)

        expected_sections = [
            "motion",
            "windows_and_doors",
            "media",
            "appliances",
            "environmental",
            "power",
            "wasp_in_box",
            "parameters",
        ]
        assert isinstance(schema_dict, dict)
        for section in expected_sections:
            assert section in schema_dict

        # Check if CONF_AREA_ID is present in schema_dict
        # Schema dict uses vol.Required/vol.Optional markers as keys, so we need to check the .schema attribute
        area_id_present = any(
            hasattr(key, "schema") and key.schema == CONF_AREA_ID for key in schema_dict
        )
        if expected_name_present:
            assert area_id_present, (
                "CONF_AREA_ID should be present in schema but was not found"
            )
        else:
            assert not area_id_present, (
                "CONF_AREA_ID should not be present in schema but was found"
            )

        if test_schema_validation:
            # Test schema instantiation
            # Note: purpose is a string, not a dict section
            data = schema(
                {
                    CONF_AREA_ID: "test_area",
                    "purpose": DEFAULT_PURPOSE,  # purpose is a string value, not a section
                    "motion": {},
                    "windows_and_doors": {},
                    "media": {},
                    "appliances": {},
                    "environmental": {},
                    "power": {},
                    "wasp_in_box": {},
                    "parameters": {},
                }
            )
            assert data[CONF_AREA_ID] == "test_area"


class TestAreaOccupancyConfigFlow:
    """Test AreaOccupancyConfigFlow class."""

    @pytest.mark.parametrize(
        ("areas", "user_input", "expected_step_id", "expected_type", "patch_type"),
        [
            ([], None, "area_basics", FlowResultType.FORM, None),  # auto-start wizard
            (
                [
                    {
                        CONF_AREA_ID: "living_room",
                        CONF_PURPOSE: "social",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                    }
                ],
                None,
                "user",
                FlowResultType.MENU,
                None,
            ),  # show menu
        ],
    )
    async def test_async_step_user_scenarios(
        self,
        hass: HomeAssistant,
        config_flow_flow,
        setup_area_registry: dict[str, str],
        areas,
        user_input,
        expected_step_id,
        expected_type,
        patch_type,
    ):
        """Test async_step_user with various scenarios."""
        # Replace hardcoded area IDs with actual area IDs from registry
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")
        for area in areas:
            if area.get(CONF_AREA_ID) == "living_room":
                area[CONF_AREA_ID] = living_room_area_id

        # Set up areas
        config_flow_flow._areas = areas

        if patch_type == "schema":
            with patch_create_schema_context():
                result = await config_flow_flow.async_step_user(user_input)
        elif patch_type == "unique_id":
            with (
                patch.object(
                    config_flow_flow, "async_set_unique_id", new_callable=AsyncMock
                ),
                patch.object(config_flow_flow, "_abort_if_unique_id_configured"),
            ):
                result = await config_flow_flow.async_step_user(user_input)
        else:
            result = await config_flow_flow.async_step_user(user_input)

        assert result.get("type") == expected_type
        if expected_step_id:
            assert result.get("step_id") == expected_step_id
        if expected_type == FlowResultType.CREATE_ENTRY:
            assert result.get("title") == "Area Occupancy Detection"
            assert CONF_AREAS in result.get("data", {})
        elif expected_step_id == "user" and expected_type == FlowResultType.FORM:
            assert "data_schema" in result
        elif expected_step_id == "user" and expected_type == FlowResultType.MENU:
            assert "menu_options" in result
        elif expected_step_id == "area_action":
            # _area_being_edited now stores area ID, not name
            assert config_flow_flow._area_being_edited == living_room_area_id

    @pytest.mark.parametrize(
        (
            "action",
            "expected_step_id",
            "expected_area_edited",
            "expected_area_to_remove",
            "needs_schema_mock",
        ),
        [
            (CONF_ACTION_EDIT, "area_basics", True, None, False),
            (CONF_ACTION_REMOVE, "remove_area", None, True, False),
            (CONF_ACTION_CANCEL, "user", None, None, False),
        ],
    )
    async def test_async_step_area_action_scenarios(
        self,
        config_flow_flow,
        config_flow_sample_area,
        setup_area_registry: dict[str, str],
        action,
        expected_step_id,
        expected_area_edited,
        expected_area_to_remove,
        needs_schema_mock,
    ):
        """Test async_step_area_action with different actions."""
        # Get actual area ID from sample area
        living_room_area_id = config_flow_sample_area[CONF_AREA_ID]
        config_flow_flow._areas = [config_flow_sample_area]
        # _area_being_edited now stores area ID, not name
        config_flow_flow._area_being_edited = living_room_area_id

        user_input = {"action": action}

        if needs_schema_mock:
            with patch_create_schema_context():
                result = await config_flow_flow.async_step_area_action(user_input)
        else:
            result = await config_flow_flow.async_step_area_action(user_input)

        if expected_step_id == "user":
            assert result.get("type") == FlowResultType.MENU
        else:
            assert result.get("type") == FlowResultType.FORM
        assert result.get("step_id") == expected_step_id
        if expected_area_edited:
            assert config_flow_flow._area_being_edited == living_room_area_id
        elif action == CONF_ACTION_CANCEL:
            assert config_flow_flow._area_being_edited is None
        if expected_area_to_remove:
            assert config_flow_flow._area_to_remove == living_room_area_id

    async def test_wizard_edit_mode_initializes_draft(
        self, config_flow_flow, setup_area_registry: dict[str, str]
    ):
        """Test that the wizard initializes draft with existing config in edit mode."""
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")
        area_config = create_area_config(
            name="Living Room",
            motion_sensors=["binary_sensor.motion1"],
        )
        area_config[CONF_AREA_ID] = living_room_area_id
        config_flow_flow._areas = [area_config]
        config_flow_flow._area_being_edited = living_room_area_id

        # Initialize wizard - should populate draft from existing config
        result = await config_flow_flow.async_step_area_config()

        assert result.get("step_id") == "area_basics"
        assert (
            config_flow_flow._area_config_draft.get(CONF_AREA_ID) == living_room_area_id
        )
        assert config_flow_flow._area_config_draft.get(CONF_MOTION_SENSORS) == [
            "binary_sensor.motion1"
        ]

    @pytest.mark.parametrize(
        (
            "area_being_edited",
            "area_to_remove",
            "step_method",
            "expected_step_id",
        ),
        [
            (None, None, "async_step_area_action", "user"),
            ("NonExistent", None, "async_step_area_action", "user"),
            (None, None, "async_step_remove_area", "user"),
        ],
        ids=["no_area", "area_not_found", "remove_no_area"],
    )
    async def test_config_flow_edge_cases(
        self,
        config_flow_flow,
        area_being_edited,
        area_to_remove,
        step_method,
        expected_step_id,
    ):
        """Test config flow edge cases."""
        config_flow_flow._areas = [create_area_config(name="Test")]
        config_flow_flow._area_being_edited = area_being_edited
        config_flow_flow._area_to_remove = area_to_remove

        with patch_create_schema_context():
            method = getattr(config_flow_flow, step_method)
            result = await method()
            if expected_step_id == "user":
                assert result.get("type") == FlowResultType.MENU
            else:
                assert result.get("type") == FlowResultType.FORM
            assert result.get("step_id") == expected_step_id

    @pytest.mark.parametrize(
        (
            "confirm",
            "expected_type",
            "expected_step_id",
            "has_error",
            "area_to_remove_cleared",
        ),
        [
            (False, FlowResultType.MENU, "user", False, True),  # cancel
            (True, FlowResultType.FORM, "remove_area", True, False),  # last_area_error
        ],
    )
    async def test_config_flow_remove_area_scenarios(
        self,
        config_flow_flow,
        setup_area_registry: dict[str, str],
        confirm,
        expected_type,
        expected_step_id,
        has_error,
        area_to_remove_cleared,
    ):
        """Test config flow remove area with various scenarios."""
        # Get actual area ID from registry
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")
        area_config = create_area_config(
            name="Living Room",
            motion_sensors=["binary_sensor.motion1"],
        )
        area_config[CONF_AREA_ID] = living_room_area_id
        config_flow_flow._areas = [area_config]
        # _area_to_remove now stores area ID, not name
        config_flow_flow._area_to_remove = living_room_area_id
        user_input = {"confirm": confirm}
        result = await config_flow_flow.async_step_remove_area(user_input)
        assert result.get("type") == expected_type
        assert result.get("step_id") == expected_step_id
        if has_error:
            assert "errors" in result
            assert "last area" in result["errors"]["base"].lower()
        if area_to_remove_cleared:
            assert config_flow_flow._area_to_remove is None


class TestConfigFlowIntegration:
    """Test config flow integration scenarios."""

    async def test_complete_config_flow(
        self,
        config_flow_flow,
        setup_area_registry: dict[str, str],
    ):
        """Test complete configuration flow through all wizard steps."""
        expected_area_id = setup_area_registry.get("Living Room", "living_room")

        # Step 1: Auto-starts wizard when no areas exist
        result1 = await config_flow_flow.async_step_user()
        assert result1.get("type") == FlowResultType.FORM
        assert result1.get("step_id") == "area_basics"

        # Step 2: Submit basics (area + purpose)
        result2 = await config_flow_flow.async_step_area_basics(
            {CONF_AREA_ID: expected_area_id, CONF_PURPOSE: "social"}
        )
        assert result2.get("type") == FlowResultType.FORM
        assert result2.get("step_id") == "area_motion"

        # Step 3: Submit motion sensors
        result3 = await config_flow_flow.async_step_area_motion(
            {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        )
        assert result3.get("type") == FlowResultType.FORM
        assert result3.get("step_id") == "area_sensors"

        # Step 4: Submit additional sensors (empty sections)
        result4 = await config_flow_flow.async_step_area_sensors(
            {
                "windows_and_doors": {},
                "media": {},
                "appliances": {},
                "environmental": {},
                "power": {},
            }
        )
        assert result4.get("type") == FlowResultType.FORM
        assert result4.get("step_id") == "area_behavior"

        # Step 5: Submit behavior parameters
        result5 = await config_flow_flow.async_step_area_behavior(
            {CONF_THRESHOLD: 60, CONF_DECAY_ENABLED: True, CONF_WASP_ENABLED: False}
        )
        assert result5.get("type") == FlowResultType.MENU
        assert result5.get("step_id") == "user"  # Returns to menu

        # Step 6: Finish setup
        with (
            patch.object(
                config_flow_flow, "async_set_unique_id", new_callable=AsyncMock
            ),
            patch.object(config_flow_flow, "_abort_if_unique_id_configured"),
        ):
            result6 = await config_flow_flow.async_step_finish_setup()

            assert result6.get("type") == FlowResultType.CREATE_ENTRY
            assert result6.get("title") == "Area Occupancy Detection"

            # Areas stored in CONF_AREAS list in data
            result_data = result6.get("data", {})
            assert CONF_AREAS in result_data
            areas_list = result_data[CONF_AREAS]
            assert len(areas_list) == 1
            area_data = areas_list[0]
            assert area_data.get(CONF_AREA_ID) == expected_area_id
            assert area_data.get(CONF_MOTION_SENSORS) == ["binary_sensor.motion1"]
            assert area_data.get(CONF_THRESHOLD) == 60

    async def test_config_flow_with_existing_entry(
        self, config_flow_flow, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ):
        """Test config flow when entry already exists."""
        hass.data = {}

        # Use actual area ID from registry
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")

        # When finish setup is selected, it should check for existing entry
        area_config = create_area_config(
            name="Living Room",
            motion_sensors=["binary_sensor.motion1"],
        )
        # Update to use actual area ID from registry
        area_config[CONF_AREA_ID] = living_room_area_id
        config_flow_flow._areas = [area_config]

        with (
            patch.object(
                config_flow_flow, "async_set_unique_id", new_callable=AsyncMock
            ),
            patch.object(
                config_flow_flow,
                "_abort_if_unique_id_configured",
                side_effect=AbortFlow("already_configured"),
            ),
            pytest.raises(AbortFlow, match="already_configured"),
        ):
            # AbortFlow should propagate, but it's caught and shown as error
            await config_flow_flow.async_step_finish_setup()

    async def test_config_flow_user_area_not_found(self, config_flow_flow):
        """Test config flow manage areas step when selected area is not found."""
        flow = config_flow_flow
        flow._areas = [create_area_config(name="Living Room")]

        user_input = {"selected_option": f"{CONF_OPTION_PREFIX_AREA}NonExistent"}
        result = await flow.async_step_manage_areas(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "manage_areas"
        assert "errors" in result
        assert "base" in result["errors"]

    @pytest.mark.parametrize(
        ("areas", "mock_validate_return", "expected_has_errors"),
        [
            ([], None, True),  # no_areas
            (
                [create_area_config(name="Living Room", motion_sensors=[])],
                None,
                True,
            ),  # validation_error (returns errors from real validate)
            (
                [create_area_config(name="Living Room")],
                {"base": "unknown"},
                True,
            ),  # forced_error
        ],
    )
    async def test_config_flow_user_finish_setup_errors(
        self, config_flow_flow, areas, mock_validate_return, expected_has_errors
    ):
        """Test config flow finish setup with various error scenarios."""
        flow = config_flow_flow
        flow._areas = areas

        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            if mock_validate_return is not None:
                with patch.object(
                    flow, "_validate_config", return_value=mock_validate_return
                ):
                    result = await flow.async_step_finish_setup()
            else:
                result = await flow.async_step_finish_setup()

            # If validation fails, it returns to user menu (or form if no areas)
            if not areas:
                assert result["type"] == FlowResultType.FORM
                assert result["step_id"] == "area_basics"
            else:
                assert result["type"] == FlowResultType.MENU
                assert result["step_id"] == "user"

    @pytest.mark.parametrize(
        (
            "flow_type",
            "step_method",
            "step_id",
            "area_being_edited",
            "area_to_remove",
            "expected_placeholders",
        ),
        [
            (
                "config",
                "async_step_area_action",
                "area_action",
                "Living Room",
                None,
                {},
            ),
            (
                "config",
                "async_step_remove_area",
                "remove_area",
                None,
                "Living Room",
                {"area_name": "Living Room"},
            ),
            # Note: options flow no longer has area_action/remove_area steps
            # (area management is in options flow)
        ],
    )
    async def test_flow_show_form(
        self,
        hass: HomeAssistant,
        config_flow_flow,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        setup_area_registry: dict[str, str],
        flow_type,
        step_method,
        step_id,
        area_being_edited,
        area_to_remove,
        expected_placeholders,
    ) -> None:
        """Test that flows show forms correctly when no user input."""
        if flow_type == "config":
            flow = config_flow_flow
            # Use actual area ID from registry
            living_room_area_id = setup_area_registry.get("Living Room", "living_room")
            area_config = create_area_config(name="Living Room")
            area_config[CONF_AREA_ID] = living_room_area_id
            flow._areas = [area_config]
        else:
            flow = config_flow_options_flow
            flow.config_entry = config_flow_mock_config_entry_with_areas

        # Convert area names to area IDs
        if area_being_edited:
            area_id = setup_area_registry.get(area_being_edited)
            if area_id:
                flow._area_being_edited = area_id
            else:
                flow._area_being_edited = (
                    area_being_edited  # Fallback if not in registry
                )
        else:
            flow._area_being_edited = area_being_edited

        if area_to_remove:
            area_id = setup_area_registry.get(area_to_remove)
            if area_id:
                flow._area_to_remove = area_id
            else:
                flow._area_to_remove = area_to_remove  # Fallback if not in registry
        else:
            flow._area_to_remove = area_to_remove

        method = getattr(flow, step_method)
        result = await method()

        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == step_id
        assert "data_schema" in result
        assert "description_placeholders" in result
        for key, value in expected_placeholders.items():
            assert result["description_placeholders"][key] == value

    async def test_error_recovery_in_config_flow(
        self, config_flow_flow, hass: HomeAssistant, setup_area_registry: dict[str, str]
    ):
        """Test error recovery in config flow via wizard steps."""
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")

        # Initialize wizard
        config_flow_flow._init_area_wizard()

        # Step 1: Submit basics
        result1 = await config_flow_flow.async_step_area_basics(
            {CONF_AREA_ID: living_room_area_id, CONF_PURPOSE: "social"}
        )
        assert result1.get("type") == FlowResultType.FORM
        assert result1.get("step_id") == "area_motion"

        # Step 2: Submit invalid motion (empty sensors) - should show error
        result2 = await config_flow_flow.async_step_area_motion(
            {CONF_MOTION_SENSORS: []}
        )
        assert result2.get("type") == FlowResultType.FORM
        assert result2.get("step_id") == "area_motion"
        assert "errors" in result2

        # Step 2 retry: Submit valid motion sensors
        result3 = await config_flow_flow.async_step_area_motion(
            {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
        )
        assert result3.get("type") == FlowResultType.FORM
        assert result3.get("step_id") == "area_sensors"

    async def test_schema_generation_with_entities(self, hass):
        """Test schema generation with available entities."""
        with patch(
            "custom_components.area_occupancy.config_flow._get_include_entities"
        ) as mock_get_entities:
            mock_get_entities.return_value = {
                "appliance": ["binary_sensor.motion1", "binary_sensor.door1"],
                "window": ["binary_sensor.window1"],
                "door": ["binary_sensor.door1"],
                "cover": ["cover.blinds1"],
                "temperature": ["sensor.temp1"],
                "humidity": ["sensor.humidity1"],
                "pressure": ["sensor.pressure1"],
                "air_quality": ["sensor.aqi1"],
                "pm25": ["sensor.pm25_1"],
                "pm10": ["sensor.pm10_1"],
                "motion": ["binary_sensor.motion1"],
            }
            schema_dict = create_schema(hass)
            assert isinstance(schema_dict, dict)
            assert len(schema_dict) > 0


class TestAreaOccupancyOptionsFlow:
    """Test AreaOccupancyOptionsFlow class."""

    async def test_options_flow_init_menu(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ) -> None:
        """Test options flow init returns menu."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        result = await flow.async_step_init()
        assert result["type"] == FlowResultType.MENU
        assert result["step_id"] == "init"
        assert "menu_options" in result
        assert "add_area" in result["menu_options"]
        assert "manage_areas" in result["menu_options"]
        assert "global_settings" in result["menu_options"]
        assert "manage_people" in result["menu_options"]

    async def test_options_flow_global_settings_save(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test that global settings are actually saved."""
        from custom_components.area_occupancy.const import (
            CONF_SLEEP_END,
            CONF_SLEEP_START,
        )

        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        # Set initial options
        flow.config_entry.options = {
            CONF_SLEEP_START: "22:00:00",
            CONF_SLEEP_END: "07:00:00",
        }

        # Update global settings
        user_input = {
            CONF_SLEEP_START: "23:00:00",
            CONF_SLEEP_END: "08:00:00",
        }

        result = await flow.async_step_global_settings(user_input)
        assert result["type"] == FlowResultType.CREATE_ENTRY

        # Verify settings were saved
        result_data = result["data"]
        assert result_data[CONF_SLEEP_START] == "23:00:00"
        assert result_data[CONF_SLEEP_END] == "08:00:00"


class TestHelperFunctionEdgeCases:
    """Test edge cases for helper functions."""

    @pytest.mark.parametrize(
        "areas",
        [
            ("not a list"),  # not_list
            (["not a dict", 123, None]),  # invalid_area_dict
            ([{CONF_PURPOSE: "social"}]),  # missing_name
            ([{CONF_AREA_ID: "", CONF_PURPOSE: "social"}]),  # empty_area_id
            ([{CONF_AREA_ID: "unknown", CONF_PURPOSE: "social"}]),  # unknown_area_id
        ],
    )
    def test_create_area_selector_schema_edge_cases(self, areas):
        """Test _create_area_selector_schema with various edge cases."""
        schema = _create_area_selector_schema(areas)
        assert isinstance(schema, vol.Schema)

        # Schema should handle edge cases gracefully
        # Invalid areas should be filtered out, resulting in empty options if all invalid
        schema_dict = schema.schema
        assert "selected_option" in schema_dict

        # If all areas are invalid, schema should still be valid but have no options
        # (This is tested by the fact that schema creation doesn't raise)

    def test_find_area_by_sanitized_id_unknown_area(self):
        """Test _find_area_by_sanitized_id when area ID is 'unknown'."""
        areas = [{CONF_AREA_ID: "unknown", CONF_PURPOSE: "social"}]
        result = _find_area_by_sanitized_id(areas, "unknown")
        assert result is not None  # Should find it
        assert result[CONF_AREA_ID] == "unknown"

    @pytest.mark.parametrize(
        ("area_being_edited", "should_have_errors", "expected_error_key"),
        [
            (None, True, "area_already_configured"),  # duplicate_raises
            ("test_area", False, None),  # same_area_editing_allowed
        ],
    )
    def test_validate_duplicate_area_id_scenarios(
        self, area_being_edited, should_have_errors, expected_error_key
    ):
        """Test _validate_duplicate_area_id with various scenarios."""
        flow = BaseOccupancyFlow()
        flattened_input = {CONF_AREA_ID: "test_area"}
        areas = [{CONF_AREA_ID: "test_area", CONF_PURPOSE: "social"}]

        errors = flow._validate_duplicate_area_id(
            flattened_input, areas, area_being_edited, None
        )
        if should_have_errors:
            assert errors
            assert expected_error_key in errors.values()
        else:
            assert not errors


class TestStaticMethods:
    """Test static methods."""

    def test_async_get_options_flow(self):
        """Test async_get_options_flow returns OptionsFlow instance."""
        mock_entry = Mock(spec=ConfigEntry)
        result = AreaOccupancyConfigFlow.async_get_options_flow(mock_entry)
        assert isinstance(result, AreaOccupancyOptionsFlow)


class TestNewHelperFunctions:
    """Test newly extracted helper functions."""

    @pytest.mark.parametrize(
        ("purpose", "expected_has_decay_half_life"),
        [
            ("social", True),  # with_purpose
            (None, False),  # no_purpose
        ],
    )
    def test_apply_purpose_based_decay_default(
        self, purpose, expected_has_decay_half_life
    ):
        """Test applying purpose-based decay default."""
        flattened_input = {CONF_PURPOSE: purpose} if purpose else {}
        _apply_purpose_based_decay_default(flattened_input, purpose)
        if expected_has_decay_half_life:
            assert CONF_DECAY_HALF_LIFE in flattened_input
        else:
            assert CONF_DECAY_HALF_LIFE not in flattened_input

    def test_flatten_sectioned_input(self):
        """Test flattening sectioned input."""
        user_input = {
            CONF_AREA_ID: "test_area",
            "motion": {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            },
            CONF_PURPOSE: "social",  # Purpose is now at root level
            "wasp_in_box": {CONF_WASP_ENABLED: True},
        }
        result = _flatten_sectioned_input(user_input)
        assert result[CONF_AREA_ID] == "test_area"
        assert result[CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]
        assert result[CONF_PURPOSE] == "social"
        assert result[CONF_WASP_ENABLED] is True

    @pytest.mark.parametrize(
        ("areas", "search_name", "expected_found", "expected_name"),
        [
            (
                [
                    {CONF_AREA_ID: "living_room", CONF_PURPOSE: "social"},
                    {CONF_AREA_ID: "kitchen", CONF_PURPOSE: "work"},
                ],
                "living_room",
                True,
                "living_room",
            ),  # found
            (
                [{CONF_AREA_ID: "living_room", CONF_PURPOSE: "social"}],
                "bedroom",
                False,
                None,
            ),  # not_found
        ],
    )
    def test_find_area_by_id(self, areas, search_name, expected_found, expected_name):
        """Test finding area by ID."""
        result = _find_area_by_id(areas, search_name)
        if expected_found:
            assert result is not None
            assert result[CONF_AREA_ID] == expected_name
        else:
            assert result is None

    @pytest.mark.parametrize(
        (
            "initial_areas",
            "updated_area",
            "old_name",
            "expected_count",
            "expected_purpose",
            "expected_name",
        ),
        [
            (
                [
                    {CONF_AREA_ID: "living_room", CONF_PURPOSE: "social"},
                    {CONF_AREA_ID: "kitchen", CONF_PURPOSE: "work"},
                ],
                {CONF_AREA_ID: "living_room", CONF_PURPOSE: "entertainment"},
                "living_room",
                2,
                "entertainment",
                None,
            ),  # update_existing
            (
                [{CONF_AREA_ID: "living_room", CONF_PURPOSE: "social"}],
                {CONF_AREA_ID: "kitchen", CONF_PURPOSE: "work"},
                None,
                2,
                None,
                "kitchen",
            ),  # add_new
        ],
    )
    def test_update_area_in_list(
        self,
        initial_areas,
        updated_area,
        old_name,
        expected_count,
        expected_purpose,
        expected_name,
    ):
        """Test updating or adding area in list."""
        result = _update_area_in_list(initial_areas.copy(), updated_area, old_name)
        assert len(result) == expected_count
        if expected_purpose:
            assert result[0][CONF_PURPOSE] == expected_purpose
        if expected_name:
            assert result[1][CONF_AREA_ID] == expected_name

    def test_remove_area_from_list(self):
        """Test removing an area from list."""
        areas = [
            {CONF_AREA_ID: "living_room", CONF_PURPOSE: "social"},
            {CONF_AREA_ID: "kitchen", CONF_PURPOSE: "work"},
        ]
        result = _remove_area_from_list(areas, "living_room")
        assert len(result) == 1
        assert result[0][CONF_AREA_ID] == "kitchen"

    @pytest.mark.parametrize(
        ("error_type", "error_message", "expected_result"),
        [
            (HomeAssistantError, "Test error", "Test error"),
            (vol.Invalid, "Validation error", "Validation error"),
            (ValueError, "Value error", "unknown"),
            (KeyError, "key", "unknown"),
            (TypeError, "Type error", "unknown"),
        ],
    )
    def test_handle_step_error(self, error_type, error_message, expected_result):
        """Test error handling for different exception types."""
        err = error_type(error_message)
        result = _handle_step_error(err)
        assert result == expected_result

        # Validate error messages are user-friendly (not empty, not technical jargon)
        assert len(result) > 0  # Error messages should not be empty
        if result != "unknown":
            # User-friendly errors should not contain Python traceback info
            assert "Traceback" not in result
            assert "File" not in result
            assert "line" not in result.lower()
            # Should be readable (no excessive technical details)
            assert len(result) < 500  # Reasonable length for user-facing errors
