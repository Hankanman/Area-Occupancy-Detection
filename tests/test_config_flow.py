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
    _find_area_by_id,
    _find_area_by_sanitized_id,
    _flatten_sectioned_input,
    _get_area_summary_info,
    _get_include_entities,
    _get_purpose_display_name,
    _get_state_select_options,
    _handle_step_error,
    _remove_area_from_list,
    _update_area_in_list,
    create_schema,
)
from custom_components.area_occupancy.const import (
    CONF_ACTION_ADD_AREA,
    CONF_ACTION_CANCEL,
    CONF_ACTION_EDIT,
    CONF_ACTION_REMOVE,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_AREAS,
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
from homeassistant.helpers import area_registry as ar
from tests.conftest import (
    create_area_config,
    create_user_input,
    patch_create_schema_context,
)


# ruff: noqa: SLF001, TID251, PLC0415
@pytest.mark.parametrize("expected_lingering_timers", [True])
class TestBaseOccupancyFlow:
    """Test BaseOccupancyFlow class."""

    @pytest.fixture
    def flow(self):
        """Create a BaseOccupancyFlow instance."""
        return BaseOccupancyFlow()

    @pytest.mark.parametrize(
        ("config_modification", "should_raise", "expected_error_match"),
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
                "no longer exists",
            ),  # invalid_area_id
        ],
    )
    def test_validate_config_valid_scenarios(
        self,
        flow,
        config_flow_base_config,
        hass,
        config_modification,
        should_raise,
        expected_error_match,
    ):
        """Test validating various valid and invalid configuration scenarios."""
        test_config = {**config_flow_base_config, **config_modification}

        if should_raise:
            with pytest.raises(vol.Invalid, match=expected_error_match):
                flow._validate_config(test_config, hass)
        else:
            flow._validate_config(test_config, hass)  # Should not raise any exception

    @pytest.mark.parametrize(
        ("invalid_config", "expected_error"),
        [
            (
                {"motion_sensors": []},
                "At least one motion sensor is required",
            ),
            (
                {"weight_motion": 1.5},
                "weight_motion must be between 0 and 1",
            ),
            (
                {"threshold": 150},
                "threshold",
            ),
            (
                {"threshold": 0},
                "Threshold must be between 1 and 100",
            ),
            (
                {"threshold": 101},
                "Threshold must be between 1 and 100",
            ),
            (
                {CONF_AREA_ID: ""},
                "Area selection is required",
            ),
            (
                {"decay_enabled": True, "decay_half_life": -1},
                "between 10 and 3600",
            ),
            (
                {"decay_enabled": True, "decay_half_life": 5},
                "between 10 and 3600",
            ),
            (
                {"decay_enabled": True, "decay_half_life": 3601},
                "between 10 and 3600",
            ),
            (
                {CONF_PURPOSE: ""},
                "Purpose is required",
            ),
            (
                {CONF_MEDIA_DEVICES: ["media_player.tv"], CONF_MEDIA_ACTIVE_STATES: []},
                "Media active states are required",
            ),
            (
                {CONF_APPLIANCES: ["switch.light"], CONF_APPLIANCE_ACTIVE_STATES: []},
                "Appliance active states are required",
            ),
            (
                {
                    CONF_DOOR_SENSORS: ["binary_sensor.door1"],
                    CONF_DOOR_ACTIVE_STATE: "",
                },
                "Door active state is required",
            ),
            (
                {
                    CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
                    CONF_WINDOW_ACTIVE_STATE: "",
                },
                "Window active state is required",
            ),
            (
                {
                    CONF_MOTION_PROB_GIVEN_TRUE: 0.5,
                    CONF_MOTION_PROB_GIVEN_FALSE: 0.6,
                },
                "Motion sensor P(Active | Occupied) must be greater than",
            ),
            (
                {
                    CONF_MOTION_PROB_GIVEN_TRUE: 0.5,
                    CONF_MOTION_PROB_GIVEN_FALSE: 0.5,
                },
                "Motion sensor P(Active | Occupied) must be greater than",
            ),
        ],
    )
    def test_validate_config_invalid_scenarios(
        self, flow, config_flow_base_config, invalid_config, expected_error, hass
    ):
        """Test various invalid configuration scenarios."""
        test_config = {**config_flow_base_config, **invalid_config}
        # Remove None values to test missing keys
        test_config = {k: v for k, v in test_config.items() if v is not None}

        with pytest.raises(vol.Invalid) as excinfo:
            flow._validate_config(test_config, hass)
        error_message = str(excinfo.value)
        assert expected_error.lower() in error_message.lower()

        # Validate error messages are user-friendly
        assert len(error_message) > 0  # Should not be empty
        assert len(error_message) < 500  # Reasonable length
        # Should not contain technical Python details
        assert "Traceback" not in error_message
        assert "File" not in error_message


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
            "energy",
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
                    "energy": {},
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
            ([], None, "area_config", FlowResultType.FORM, "schema"),  # auto-start
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
            (CONF_ACTION_EDIT, "area_config", True, None, True),
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

    async def test_async_step_area_config_preserves_name_when_editing(
        self, config_flow_flow, setup_area_registry: dict[str, str]
    ):
        """Test that area_id is preserved when editing an area."""
        # Get actual area ID from registry
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")
        area_config = create_area_config(
            name="Living Room",
            motion_sensors=["binary_sensor.motion1"],
        )
        area_config[CONF_AREA_ID] = living_room_area_id
        config_flow_flow._areas = [area_config]
        # _area_being_edited now stores area ID, not name
        config_flow_flow._area_being_edited = living_room_area_id

        # User submits form without area_id field (or with empty area_id)
        user_input = create_user_input(name="")  # Empty name - should be preserved
        del user_input[CONF_AREA_ID]  # Remove area_id to test preservation

        with (
            patch.object(config_flow_flow, "_validate_config") as mock_validate,
            patch_create_schema_context(),
        ):
            # Call async_step_area_config to trigger validation
            await config_flow_flow.async_step_area_config(user_input)

            # Should have preserved the area_id
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args[0][0]
            assert call_args[CONF_AREA_ID] == living_room_area_id

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
        config_flow_valid_user_input,
        setup_area_registry: dict[str, str],
    ):
        """Test complete configuration flow."""
        # Get actual area ID from user input
        expected_area_id = config_flow_valid_user_input[CONF_AREA_ID]

        # Step 1: Auto-starts area_config when no areas exist
        with patch_create_schema_context():
            result1 = await config_flow_flow.async_step_user()
            assert result1.get("type") == FlowResultType.FORM
            assert result1.get("step_id") == "area_config"

        # Step 2: Submit area config data
        with patch_create_schema_context():
            result2 = await config_flow_flow.async_step_area_config(
                config_flow_valid_user_input
            )
            assert result2.get("type") == FlowResultType.MENU
            assert result2.get("step_id") == "user"  # Returns to menu

        # Step 3: Finish setup
        with (
            patch.object(
                config_flow_flow, "async_set_unique_id", new_callable=AsyncMock
            ),
            patch.object(config_flow_flow, "_abort_if_unique_id_configured"),
        ):
            result3 = await config_flow_flow.async_step_finish_setup()

            assert result3.get("type") == FlowResultType.CREATE_ENTRY
            assert result3.get("title") == "Area Occupancy Detection"

            result_data = result3.get("data", {})
            # Data is now stored in CONF_AREAS list format
            areas = result_data.get(CONF_AREAS, [])
            assert len(areas) == 1
            area_data = areas[0]
            assert area_data.get(CONF_AREA_ID) == expected_area_id  # Area ID
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
        ("areas", "error_type", "expected_has_errors"),
        [
            ([], None, True),  # no_areas
            (
                [create_area_config(name="Living Room", motion_sensors=[])],
                None,
                True,
            ),  # validation_error
            (
                [create_area_config(name="Living Room")],
                KeyError,
                True,
            ),  # unexpected_error
        ],
    )
    async def test_config_flow_user_finish_setup_errors(
        self, config_flow_flow, areas, error_type, expected_has_errors
    ):
        """Test config flow finish setup with various error scenarios."""
        flow = config_flow_flow
        flow._areas = areas

        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            if error_type:
                with patch.object(
                    flow, "_validate_config", side_effect=error_type("test")
                ):
                    result = await flow.async_step_finish_setup()
            else:
                result = await flow.async_step_finish_setup()

            # If validation fails, it returns to user menu (or form if no areas)
            if not areas:
                assert result["type"] == FlowResultType.FORM
                assert result["step_id"] == "area_config"
            else:
                assert result["type"] == FlowResultType.MENU
                assert result["step_id"] == "user"
            # Note: errors are currently not shown in menu step
            # if expected_has_errors:
            #    assert "errors" in result

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
            (
                "options",
                "async_step_area_action",
                "area_action",
                "Living Room",
                None,
                {},
            ),
            (
                "options",
                "async_step_remove_area",
                "remove_area",
                None,
                "Living Room",
                {"area_name": "Living Room"},
            ),
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
    ):
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
        """Test error recovery in config flow."""
        # Use actual area ID from registry
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")

        # First attempt with invalid data in area_config
        invalid_input = create_user_input(
            name="Living Room",
            motion={CONF_MOTION_SENSORS: []},
        )
        # Update to use actual area ID from registry
        invalid_input[CONF_AREA_ID] = living_room_area_id

        with patch_create_schema_context():
            result1 = await config_flow_flow.async_step_area_config(invalid_input)
            assert result1.get("type") == FlowResultType.FORM
            assert "errors" in result1

        # Second attempt with valid data
        valid_input = create_user_input(name="Living Room")

        with patch_create_schema_context():
            result2 = await config_flow_flow.async_step_area_config(valid_input)
            assert result2.get("type") == FlowResultType.MENU
            assert result2.get("step_id") == "user"  # Returns to area selection

    async def test_schema_generation_with_entities(self, hass):
        """Test schema generation with available entities."""
        with patch(
            "custom_components.area_occupancy.config_flow._get_include_entities"
        ) as mock_get_entities:
            mock_get_entities.return_value = {
                "appliance": ["binary_sensor.motion1", "binary_sensor.door1"],
                "window": ["binary_sensor.window1"],
                "door": ["binary_sensor.door1"],
            }
            schema_dict = create_schema(hass)
            assert isinstance(schema_dict, dict)
            assert len(schema_dict) > 0


class TestAreaOccupancyOptionsFlow:
    """Test AreaOccupancyOptionsFlow class."""

    @pytest.mark.parametrize(
        ("config_entry_fixture", "expected_area_id"),
        [
            (
                "config_flow_mock_config_entry_with_areas",
                "living_room",
            ),  # Will be replaced with actual ID
        ],
    )
    def test_get_areas_from_config(
        self,
        hass: HomeAssistant,
        config_flow_options_flow,
        setup_area_registry: dict[str, str],
        config_entry_fixture,
        expected_area_id,
        request,
    ):
        """Test _get_areas_from_config with different config formats."""
        flow = config_flow_options_flow
        flow.config_entry = request.getfixturevalue(config_entry_fixture)
        areas = flow._get_areas_from_config()
        assert isinstance(areas, list)

        # Use actual area ID from registry for comparison
        if expected_area_id == "living_room":
            expected_area_id = setup_area_registry.get("Living Room", "living_room")

        # Should have at least one area for valid configs
        if expected_area_id:
            assert len(areas) >= 1
            assert areas[0][CONF_AREA_ID] == expected_area_id
        else:
            # Empty config should return empty list
            assert len(areas) == 0

    async def test_options_flow_init_with_device_id(
        self, config_flow_options_flow, hass, device_registry
    ):
        """Test options flow init when called from device registry."""
        flow = config_flow_options_flow

        # Add config entry to hass.config_entries so device registry can link to it
        hass.config_entries._entries[flow.config_entry.entry_id] = flow.config_entry

        # Create a device in the registry
        # Device identifier now uses area_id, not area name
        device_entry = device_registry.async_get_or_create(
            config_entry_id=flow.config_entry.entry_id,
            identifiers={(DOMAIN, "test_area")},  # Use area ID, not name
            name="Test Area",
        )

        # Update flow's device_id to match the created device
        flow._device_id = device_entry.id

        with patch_create_schema_context():
            result = await flow.async_step_init()
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "area_config"
            # _area_being_edited now stores area ID from device identifier
            assert flow._area_being_edited == "test_area"

    async def test_options_flow_init_device_not_found(
        self, config_flow_options_flow, device_registry
    ):
        """Test options flow init when device is not found."""
        flow = config_flow_options_flow
        flow._device_id = "non_existent_device_id"

        result = await flow.async_step_init()
        assert result["type"] == FlowResultType.MENU
        assert result["step_id"] == "init"

    async def test_options_flow_init_menu(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow init returns menu."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        result = await flow.async_step_init()
        assert result["type"] == FlowResultType.MENU
        assert result["step_id"] == "init"
        assert "menu_options" in result
        assert CONF_ACTION_ADD_AREA in result["menu_options"]

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

    async def test_options_flow_manage_areas_selection_error(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow manage areas when selected area is not found."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        user_input = {"selected_option": f"{CONF_OPTION_PREFIX_AREA}NonExistent"}
        result = await flow.async_step_manage_areas(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "manage_areas"
        assert "errors" in result
        assert "base" in result["errors"]

    async def test_options_flow_area_config_add_new_area(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config when adding new area."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = None  # Adding new area

        user_input = create_user_input(name="Kitchen")

        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            assert result["type"] == FlowResultType.CREATE_ENTRY
            # Verify area_id field was added to schema
            areas = result["data"][CONF_AREAS]
            assert len(areas) == 2  # Original + new
            assert any(area[CONF_AREA_ID] == "kitchen" for area in areas)

    async def test_options_flow_area_config_duplicate_area_id(
        self,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        setup_area_registry: dict[str, str],
    ):
        """Test that duplicate area ID detection works in actual flow."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = None  # Adding new area

        # Get existing area ID from config
        existing_areas = flow._get_areas_from_config()
        existing_area_id = existing_areas[0][CONF_AREA_ID]

        # Try to add a new area with the same area ID
        user_input = create_user_input(name="Living Room")  # Same area name
        user_input[CONF_AREA_ID] = existing_area_id  # Use same area ID

        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "area_config"
            assert "errors" in result
            assert "base" in result["errors"]
            assert "already configured" in result["errors"]["base"].lower()

    async def test_options_flow_area_config_change_area_id(
        self,
        hass: HomeAssistant,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        setup_area_registry: dict[str, str],
    ):
        """Test area ID change during edit (changing to different area)."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        # Get existing area ID from config
        existing_areas = flow._get_areas_from_config()
        existing_area_id = existing_areas[0][CONF_AREA_ID]
        flow._area_being_edited = existing_area_id

        # Use existing "Kitchen" area from fixture (already created by setup_area_registry)
        kitchen_area_id = setup_area_registry.get("Kitchen")
        assert kitchen_area_id is not None, "Kitchen area should exist from fixture"

        # Ensure we're changing to a different area
        assert kitchen_area_id != existing_area_id, (
            "Kitchen should be different from existing area"
        )

        # Change area ID to the Kitchen area
        user_input = create_user_input(name="Kitchen")
        user_input[CONF_AREA_ID] = kitchen_area_id

        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            # Should succeed - changing area ID means selecting a different area
            assert result["type"] == FlowResultType.CREATE_ENTRY
            areas = result["data"][CONF_AREAS]
            # Should have updated the area with new ID
            assert any(area[CONF_AREA_ID] == kitchen_area_id for area in areas)

    async def test_options_flow_area_config_no_old_area(
        self,
        hass: HomeAssistant,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        setup_area_registry: dict[str, str],
    ):
        """Test options flow area config when old area is not found."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "nonexistent_area_id"  # Non-existent area ID

        # Create "New Name" area in registry for the test
        area_reg = ar.async_get(hass)
        new_area = area_reg.async_create("New Name")
        new_area_id = new_area.id

        user_input = create_user_input(name="New Name")
        # Update user_input to use the actual area ID from registry
        user_input[CONF_AREA_ID] = new_area_id

        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            # Should succeed without migration since old area not found
            assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_options_flow_area_config_error_handling(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config error handling."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = None

        # Invalid input that will cause validation error
        user_input = create_user_input(name="", motion={CONF_MOTION_SENSORS: []})
        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "area_config"
            assert "errors" in result

    @pytest.mark.parametrize(
        ("action", "expected_step_id", "needs_schema_mock", "expected_type"),
        [
            (CONF_ACTION_EDIT, "area_config", True, FlowResultType.FORM),
            (CONF_ACTION_REMOVE, "remove_area", False, FlowResultType.FORM),
            (CONF_ACTION_CANCEL, "init", False, FlowResultType.MENU),
        ],
    )
    async def test_options_flow_area_action(
        self,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        action,
        expected_step_id,
        needs_schema_mock,
        expected_type,
    ):
        """Test options flow area action with different actions."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        # _area_being_edited now stores area ID, not name
        flow._area_being_edited = "living_room"

        user_input = {"action": action}
        if needs_schema_mock:
            with patch_create_schema_context():
                result = await flow.async_step_area_action(user_input)
        else:
            result = await flow.async_step_area_action(user_input)
        assert result["type"] == expected_type
        assert result["step_id"] == expected_step_id

    @pytest.mark.parametrize(
        ("area_being_edited", "setup_areas", "expected_type", "expected_step_id"),
        [
            (None, False, FlowResultType.MENU, "init"),  # no_area
            ("NonExistent", True, FlowResultType.FORM, "area_config"),  # area_not_found
        ],
    )
    async def test_options_flow_area_action_edge_cases(
        self,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        area_being_edited,
        setup_areas,
        expected_type,
        expected_step_id,
    ):
        """Test options flow area action edge cases."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = area_being_edited
        if setup_areas:
            # Ensure areas list exists
            flow._areas = flow._get_areas_from_config()

        result = await flow.async_step_area_action()
        assert result["type"] == expected_type
        assert result["step_id"] == expected_step_id

    @pytest.mark.parametrize(
        (
            "confirm",
            "area_to_remove",
            "add_second_area",
            "user_input_provided",
            "expected_type",
            "expected_step_id",
            "has_error",
            "has_placeholders",
        ),
        [
            (
                True,
                "Living Room",
                True,
                True,
                FlowResultType.CREATE_ENTRY,
                None,
                False,
                False,
            ),  # confirm
            (
                False,
                "Living Room",
                False,
                True,
                FlowResultType.MENU,
                "init",
                False,
                False,
            ),  # cancel
            (
                True,
                "Living Room",
                False,
                True,
                FlowResultType.FORM,
                "remove_area",
                True,
                False,
            ),  # last_area_error
            (
                None,
                None,
                False,
                False,
                FlowResultType.MENU,
                "init",
                False,
                False,
            ),  # no_area
            (
                None,
                "Living Room",
                False,
                False,
                FlowResultType.FORM,
                "remove_area",
                False,
                True,
            ),  # show_form
        ],
    )
    async def test_options_flow_remove_area(
        self,
        hass: HomeAssistant,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        setup_area_registry: dict[str, str],
        confirm,
        area_to_remove,
        add_second_area,
        user_input_provided,
        expected_type,
        expected_step_id,
        has_error,
        has_placeholders,
    ):
        """Test options flow remove area with various scenarios."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        # Convert area name to area_id if area_to_remove is provided
        if area_to_remove:
            # Get area_id from registry (area_to_remove is the area name like "Living Room")
            area_id = setup_area_registry.get(area_to_remove)
            if not area_id:
                # Fallback: convert name to area_id format
                area_id = area_to_remove.lower().replace(" ", "_")
            flow._area_to_remove = area_id
        else:
            flow._area_to_remove = area_to_remove

        if add_second_area:
            # Add another area so we can remove one
            kitchen_area_id = setup_area_registry.get("Kitchen", "kitchen")
            flow.config_entry.data[CONF_AREAS].append(
                create_area_config(
                    name="Kitchen", motion_sensors=["binary_sensor.kitchen_motion"]
                )
            )
            # Update the area_id to use the actual one from registry
            flow.config_entry.data[CONF_AREAS][-1][CONF_AREA_ID] = kitchen_area_id

        user_input = {"confirm": confirm} if user_input_provided else None

        with patch.object(hass.config_entries, "async_reload", new_callable=AsyncMock):
            result = await flow.async_step_remove_area(user_input)

        assert result["type"] == expected_type
        if expected_step_id:
            assert result["step_id"] == expected_step_id
        if has_error:
            assert "errors" in result
            assert "last area" in result["errors"]["base"].lower()
        if has_placeholders:
            assert "data_schema" in result
            assert "description_placeholders" in result
            assert result["description_placeholders"]["area_name"] == "Living Room"


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
        ("area_being_edited", "should_raise", "expected_error_match"),
        [
            (None, True, "already configured"),  # duplicate_raises
            ("test_area", False, None),  # same_area_editing_allowed
        ],
    )
    def test_validate_duplicate_area_id_scenarios(
        self, area_being_edited, should_raise, expected_error_match
    ):
        """Test _validate_duplicate_area_id with various scenarios."""
        flow = BaseOccupancyFlow()
        flattened_input = {CONF_AREA_ID: "test_area"}
        areas = [{CONF_AREA_ID: "test_area", CONF_PURPOSE: "social"}]

        if should_raise:
            with pytest.raises(vol.Invalid, match=expected_error_match):
                flow._validate_duplicate_area_id(
                    flattened_input, areas, area_being_edited, None
                )
        else:
            # Should not raise when editing the same area
            flow._validate_duplicate_area_id(
                flattened_input, areas, area_being_edited, None
            )


class TestStaticMethods:
    """Test static methods."""

    @pytest.mark.parametrize(
        ("method_name", "args", "expected_device_id"),
        [
            ("async_get_options_flow", (), None),
            ("async_get_device_options_flow", ("test_device_id",), "test_device_id"),
        ],
    )
    def test_static_methods(self, method_name, args, expected_device_id):
        """Test static methods return OptionsFlow instance."""
        mock_entry = Mock(spec=ConfigEntry)
        method = getattr(AreaOccupancyConfigFlow, method_name)
        result = method(mock_entry, *args)
        assert isinstance(result, AreaOccupancyOptionsFlow)
        if expected_device_id:
            assert result._device_id == expected_device_id

        # Validate that returned flow instance is usable
        assert result._area_being_edited is None
        assert result._area_to_remove is None
        if expected_device_id:
            assert result._device_id == expected_device_id
        else:
            assert result._device_id is None


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
