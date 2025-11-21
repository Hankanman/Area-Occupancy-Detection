"""Tests for the Area Occupancy Detection config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import voluptuous as vol

from custom_components.area_occupancy.config_flow import (
    AreaOccupancyConfigFlow,
    AreaOccupancyOptionsFlow,
    BaseOccupancyFlow,
    _apply_purpose_based_decay_default,
    _build_area_description_placeholders,
    _create_action_selection_schema,
    _create_area_selection_schema,
    _find_area_by_id,
    _find_area_by_sanitized_id,
    _flatten_sectioned_input,
    _get_area_summary_info,
    _get_default_decay_half_life,
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
    CONF_ACTION_FINISH_SETUP,
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
    CONF_MOTION_SENSORS,
    CONF_OPTION_PREFIX_AREA,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
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


# ruff: noqa: SLF001, TID251
@pytest.mark.parametrize("expected_lingering_timers", [True])
class TestBaseOccupancyFlow:
    """Test BaseOccupancyFlow class."""

    @pytest.fixture
    def flow(self):
        """Create a BaseOccupancyFlow instance."""
        return BaseOccupancyFlow()

    def test_validate_config_valid(self, flow, config_flow_base_config, hass):
        """Test validating a valid configuration."""
        flow._validate_config(
            config_flow_base_config, hass
        )  # Should not raise any exception

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
                {"decay_enabled": True, "decay_half_life": 0},
                "Decay half life must be between",
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
        assert expected_error.lower() in str(excinfo.value).lower()


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
        assert all("value" in option and "label" in option for option in options)

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
        ("areas", "is_initial"),
        [
            (
                [
                    {
                        CONF_AREA_ID: "living_room",
                        CONF_PURPOSE: "social",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                        CONF_THRESHOLD: 60.0,
                    }
                ],
                True,
            ),
            (
                [
                    {
                        CONF_AREA_ID: "living_room",
                        CONF_PURPOSE: "social",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                        CONF_THRESHOLD: 60.0,
                    }
                ],
                False,
            ),
            ([], True),
        ],
    )
    def test_create_area_selection_schema(self, areas, is_initial):
        """Test _create_area_selection_schema function."""
        schema = _create_area_selection_schema(areas, is_initial)
        assert isinstance(schema, vol.Schema)

    def test_create_action_selection_schema(self):
        """Test _create_action_selection_schema function."""
        schema = _create_action_selection_schema()
        assert isinstance(schema, vol.Schema)

        # Validate schema structure
        schema_dict = schema.schema
        assert "action" in schema_dict

    def test_get_include_entities(self, hass, mock_entity_registry):
        """Test getting include entities."""
        # Setup test entities
        entity_list = [
            Mock(
                entity_id="binary_sensor.door_1",
                domain="binary_sensor",
                device_class="door",
                original_device_class="door",
            ),
            Mock(
                entity_id="binary_sensor.window_1",
                domain="binary_sensor",
                device_class="window",
                original_device_class="window",
            ),
            Mock(
                entity_id="switch.appliance_1",
                domain="switch",
                device_class=None,
                original_device_class=None,
            ),
        ]

        # Setup mocks
        mock_entity_registry.entities = Mock()
        mock_entity_registry.entities.values = Mock(return_value=entity_list)

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=mock_entity_registry,
        ):
            # Since StateMachine attributes are read-only, we need to use monkeypatch
            # or create states directly in hass. Let's use monkeypatch approach.
            def mock_async_entity_ids(domain):
                """Mock async_entity_ids to return test entities."""
                domain_map = {
                    "binary_sensor": ["binary_sensor.door_1", "binary_sensor.window_1"],
                    "switch": ["switch.appliance_1"],
                    "fan": [],
                    "light": [],
                }
                return domain_map.get(domain, [])

            # Since StateMachine attributes are read-only, use object.__setattr__ to bypass
            # Create a mock states object
            mock_states = MagicMock()
            mock_states.async_entity_ids = MagicMock(side_effect=mock_async_entity_ids)
            mock_states.get = MagicMock(
                return_value=Mock(attributes={"device_class": None})
            )

            # Store original and replace using object.__setattr__ to bypass read-only protection
            original_states = hass.states
            object.__setattr__(hass, "states", mock_states)
            try:
                result = _get_include_entities(hass)
            finally:
                object.__setattr__(hass, "states", original_states)

        assert "door" in result
        assert "window" in result
        assert "appliance" in result
        assert "binary_sensor.door_1" in result["door"]
        assert "binary_sensor.window_1" in result["window"]
        assert "switch.appliance_1" in result["appliance"]

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
        config_flow_mock_entity_registry_for_schema,
        defaults,
        is_options,
        expected_name_present,
        test_schema_validation,
    ):
        """Test creating schema with different configurations."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=config_flow_mock_entity_registry_for_schema,
        ):
            schema_dict = create_schema(hass, defaults, is_options)
            schema = vol.Schema(schema_dict)

        expected_sections = [
            "motion",
            "windows_and_doors",
            "media",
            "appliances",
            "environmental",
            "wasp_in_box",
            "parameters",
        ]
        assert isinstance(schema_dict, dict)
        for section in expected_sections:
            assert section in schema_dict

        if expected_name_present:
            assert CONF_AREA_ID in schema_dict
        else:
            assert CONF_AREA_ID not in schema_dict

        if test_schema_validation:
            # Test schema instantiation
            data = schema(
                {
                    CONF_AREA_ID: "test_area",
                    "purpose": {},
                    "motion": {},
                    "windows_and_doors": {},
                    "media": {},
                    "appliances": {},
                    "environmental": {},
                    "wasp_in_box": {},
                    "parameters": {},
                }
            )
            assert data[CONF_AREA_ID] == "test_area"


class TestAreaOccupancyConfigFlow:
    """Test AreaOccupancyConfigFlow class."""

    def test_initialization(self):
        """Test ConfigFlow initialization."""
        flow = AreaOccupancyConfigFlow()
        assert flow.VERSION == 1
        assert flow.MINOR_VERSION == 1

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
                FlowResultType.FORM,
                None,
            ),  # show selection
            (
                [],
                {"selected_option": CONF_ACTION_ADD_AREA},
                "area_config",
                FlowResultType.FORM,
                "schema",
            ),  # add area
            (
                [
                    {
                        CONF_AREA_ID: "living_room",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                        CONF_PURPOSE: "social",
                    }
                ],
                {"selected_option": CONF_ACTION_FINISH_SETUP},
                None,
                FlowResultType.CREATE_ENTRY,
                "unique_id",
            ),  # finish
            (
                [
                    {
                        CONF_AREA_ID: "living_room",
                        CONF_PURPOSE: "social",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                    }
                ],
                {"selected_option": f"{CONF_OPTION_PREFIX_AREA}living_room"},
                "area_action",
                FlowResultType.FORM,
                None,
            ),  # select area
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

        # Update user_input if it references living_room
        if user_input and "selected_option" in user_input:
            selected_option = user_input["selected_option"]
            if selected_option and "living_room" in selected_option:
                # The option format is "area_<sanitized_id>"
                # Sanitize the actual area ID (replace spaces/slashes with underscores)
                sanitized_area_id = living_room_area_id.replace(" ", "_").replace(
                    "/", "_"
                )
                # Replace the sanitized "living_room" with the sanitized actual area ID
                user_input["selected_option"] = selected_option.replace(
                    "living_room", sanitized_area_id
                )

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
        elif expected_step_id == "user":
            assert "data_schema" in result
        elif expected_step_id == "area_action":
            # _area_being_edited now stores area ID, not name
            assert config_flow_flow._area_being_edited == "living_room"

    @pytest.mark.parametrize(
        (
            "action",
            "expected_step_id",
            "expected_area_edited",
            "expected_area_to_remove",
            "needs_schema_mock",
        ),
        [
            (CONF_ACTION_EDIT, "area_config", "living_room", None, True),
            (CONF_ACTION_REMOVE, "remove_area", None, "living_room", False),
            (CONF_ACTION_CANCEL, "user", None, None, False),
        ],
    )
    async def test_async_step_area_action_scenarios(
        self,
        config_flow_flow,
        config_flow_sample_area,
        action,
        expected_step_id,
        expected_area_edited,
        expected_area_to_remove,
        needs_schema_mock,
    ):
        """Test async_step_area_action with different actions."""
        config_flow_flow._areas = [config_flow_sample_area]
        # _area_being_edited now stores area ID, not name
        config_flow_flow._area_being_edited = "living_room"

        user_input = {"action": action}

        if needs_schema_mock:
            with patch_create_schema_context():
                result = await config_flow_flow.async_step_area_action(user_input)
        else:
            result = await config_flow_flow.async_step_area_action(user_input)

        assert result.get("type") == FlowResultType.FORM
        assert result.get("step_id") == expected_step_id
        if expected_area_edited:
            assert config_flow_flow._area_being_edited == expected_area_edited
        elif action == CONF_ACTION_CANCEL:
            assert config_flow_flow._area_being_edited is None
        if expected_area_to_remove:
            assert config_flow_flow._area_to_remove == expected_area_to_remove

    async def test_async_step_area_config_preserves_name_when_editing(
        self, config_flow_flow
    ):
        """Test that area_id is preserved when editing an area."""
        config_flow_flow._areas = [
            create_area_config(
                name="Living Room",
                motion_sensors=["binary_sensor.motion1"],
            )
        ]
        # _area_being_edited now stores area ID, not name
        config_flow_flow._area_being_edited = "living_room"

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
            assert call_args[CONF_AREA_ID] == "living_room"

    @pytest.mark.parametrize(
        (
            "scenario",
            "area_being_edited",
            "area_to_remove",
            "step_method",
            "expected_step_id",
        ),
        [
            ("no_area", None, None, "async_step_area_action", "user"),
            ("area_not_found", "NonExistent", None, "async_step_area_action", "user"),
            ("remove_no_area", None, None, "async_step_remove_area", "user"),
        ],
    )
    async def test_config_flow_edge_cases(
        self,
        config_flow_flow,
        scenario,
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
            assert result.get("type") == FlowResultType.FORM
            assert result.get("step_id") == expected_step_id

    async def test_config_flow_remove_area_cancel(self, config_flow_flow):
        """Test cancellation path."""
        config_flow_flow._areas = [
            create_area_config(
                name="Living Room",
                motion_sensors=["binary_sensor.motion1"],
            )
        ]
        # _area_to_remove now stores area ID, not name
        config_flow_flow._area_to_remove = "living_room"
        user_input = {"confirm": False}
        result = await config_flow_flow.async_step_remove_area(user_input)
        assert result.get("type") == FlowResultType.FORM
        assert result.get("step_id") == "user"
        assert config_flow_flow._area_to_remove is None

    async def test_config_flow_remove_area_last_area_error(self, config_flow_flow):
        """Test error when removing last area."""
        config_flow_flow._areas = [
            create_area_config(
                name="Living Room",
                motion_sensors=["binary_sensor.motion1"],
            )
        ]
        # _area_to_remove now stores area ID, not name
        config_flow_flow._area_to_remove = "living_room"
        user_input = {"confirm": True}
        result = await config_flow_flow.async_step_remove_area(user_input)
        assert result.get("type") == FlowResultType.FORM
        assert result.get("step_id") == "remove_area"
        assert "errors" in result
        assert "last area" in result["errors"]["base"].lower()


class TestConfigFlowIntegration:
    """Test config flow integration scenarios."""

    async def test_complete_config_flow(
        self, config_flow_flow, config_flow_valid_user_input
    ):
        """Test complete configuration flow."""
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
            assert result2.get("type") == FlowResultType.FORM
            assert result2.get("step_id") == "user"  # Returns to area selection

        # Step 3: Finish setup
        with (
            patch.object(
                config_flow_flow, "async_set_unique_id", new_callable=AsyncMock
            ),
            patch.object(config_flow_flow, "_abort_if_unique_id_configured"),
        ):
            finish_input = {"selected_option": CONF_ACTION_FINISH_SETUP}
            result3 = await config_flow_flow.async_step_user(finish_input)

            assert result3.get("type") == FlowResultType.CREATE_ENTRY
            assert result3.get("title") == "Area Occupancy Detection"

            result_data = result3.get("data", {})
            # Data is now stored in CONF_AREAS list format
            areas = result_data.get(CONF_AREAS, [])
            assert len(areas) == 1
            area_data = areas[0]
            assert area_data.get(CONF_AREA_ID) == "living_room"  # Area ID
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

        user_input = {"selected_option": CONF_ACTION_FINISH_SETUP}

        with (
            patch.object(
                config_flow_flow, "async_set_unique_id", new_callable=AsyncMock
            ),
            patch.object(
                config_flow_flow,
                "_abort_if_unique_id_configured",
                side_effect=AbortFlow("already_configured"),
            ),
        ):
            # AbortFlow should propagate, but it's caught and shown as error
            result = await config_flow_flow.async_step_user(user_input)

            # The flow catches AbortFlow and shows it as an error in the form
            assert result.get("type") == FlowResultType.FORM
            assert "errors" in result
            assert "already_configured" in result["errors"]["base"]

    async def test_config_flow_user_area_not_found(self, config_flow_flow):
        """Test config flow user step when selected area is not found."""
        flow = config_flow_flow
        flow._areas = [create_area_config(name="Living Room")]

        user_input = {"selected_option": f"{CONF_OPTION_PREFIX_AREA}NonExistent"}
        result = await flow.async_step_user(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"
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

        user_input = {"selected_option": CONF_ACTION_FINISH_SETUP}
        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            if error_type:
                with patch.object(
                    flow, "_validate_config", side_effect=error_type("test")
                ):
                    result = await flow.async_step_user(user_input)
            else:
                result = await flow.async_step_user(user_input)
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "user"
            if expected_has_errors:
                assert "errors" in result

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
            assert result2.get("type") == FlowResultType.FORM
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

    @pytest.mark.parametrize("platform", ["door", "window", "media", "appliance"])
    def test_state_options_generation(self, platform):
        """Test state options generation for different platforms."""
        options = _get_state_select_options(platform)
        assert isinstance(options, list)
        assert len(options) > 0

        for option in options:
            assert "value" in option
            assert "label" in option
            assert isinstance(option["value"], str)
            assert isinstance(option["label"], str)


class TestAreaOccupancyOptionsFlow:
    """Test AreaOccupancyOptionsFlow class."""

    def test_options_flow_init(self):
        """Test OptionsFlow initialization."""
        flow = AreaOccupancyOptionsFlow()
        assert flow._area_being_edited is None
        assert flow._area_to_remove is None
        assert flow._device_id is None

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
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

    async def test_options_flow_init_add_area(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow init add area action."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        user_input = {"selected_option": CONF_ACTION_ADD_AREA}
        with patch_create_schema_context():
            result = await flow.async_step_init(user_input)
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "area_config"

    async def test_options_flow_init_area_selection_error(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow init when selected area is not found."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas

        user_input = {"selected_option": f"{CONF_OPTION_PREFIX_AREA}NonExistent"}
        result = await flow.async_step_init(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"
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

    async def test_options_flow_area_config_migration(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config with migration.

        Note: Migration is no longer needed since area IDs are stable.
        This test verifies that area config works without migration.
        """
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        # Use sanitized name to avoid triggering migration due to sanitization
        # _area_being_edited now stores area ID, not name
        flow._area_being_edited = "living_room"

        user_input = create_user_input(name="Living_Room")

        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            # Should succeed without calling migration since name didn't change
            assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_options_flow_area_config_migration_on_rename(
        self,
        hass: HomeAssistant,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        setup_area_registry: dict[str, str],
    ):
        """Test options flow area config when area ID changes (different area selection).

        Note: With area IDs, changing the area ID means selecting a different area,
        not renaming. Migration is no longer needed since area IDs are stable.
        """

        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        # Use actual area ID from registry
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")
        flow._area_being_edited = living_room_area_id

        # Create "Living Room Renamed" area in registry
        area_reg = ar.async_get(hass)
        renamed_area = area_reg.async_create("Living Room Renamed")
        renamed_area_id = renamed_area.id

        user_input = create_user_input(name="Living Room Renamed")
        # Update user_input to use the actual area ID from registry
        user_input[CONF_AREA_ID] = renamed_area_id

        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            assert result["type"] == FlowResultType.CREATE_ENTRY
            # Migration is no longer called when area ID changes (it's a different area selection)

    @pytest.mark.parametrize(
        "error_type",
        [ValueError, HomeAssistantError, KeyError],
    )
    async def test_options_flow_area_config_migration_errors(
        self,
        hass: HomeAssistant,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        setup_area_registry: dict[str, str],
        error_type,
    ):
        """Test options flow area config when area ID changes.

        Note: Migration is no longer needed since area IDs are stable.
        This test verifies that area config works when changing area selection.
        """

        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        # Use actual area ID from registry
        living_room_area_id = setup_area_registry.get("Living Room", "living_room")
        flow._area_being_edited = living_room_area_id

        # Create "Living Room Renamed" area in registry
        area_reg = ar.async_get(hass)
        renamed_area = area_reg.async_create("Living Room Renamed")
        renamed_area_id = renamed_area.id

        user_input = create_user_input(name="Living Room Renamed")
        # Update user_input to use the actual area ID from registry
        user_input[CONF_AREA_ID] = renamed_area_id

        with patch_create_schema_context():
            result = await flow.async_step_area_config(user_input)
            # Should succeed without migration since migration is no longer used
            assert result["type"] == FlowResultType.CREATE_ENTRY

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
        ("action", "expected_step_id", "needs_schema_mock"),
        [
            (CONF_ACTION_EDIT, "area_config", True),
            (CONF_ACTION_REMOVE, "remove_area", False),
            (CONF_ACTION_CANCEL, "init", False),
        ],
    )
    async def test_options_flow_area_action(
        self,
        config_flow_options_flow,
        config_flow_mock_config_entry_with_areas,
        action,
        expected_step_id,
        needs_schema_mock,
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
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == expected_step_id

    async def test_options_flow_area_action_no_area(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area action when no area is selected."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = None

        result = await flow.async_step_area_action()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

    async def test_options_flow_area_action_area_not_found(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area action when area is not found."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "NonExistent"
        # Ensure areas list exists
        flow._areas = flow._get_areas_from_config()

        result = await flow.async_step_area_action()
        # When area is not found, async_step_area_action calls async_step_init()
        # But async_step_init checks _area_being_edited and redirects to area_config
        # So the actual behavior is that it goes to area_config
        # This is because async_step_init has logic to redirect if _area_being_edited is set
        assert result["type"] == FlowResultType.FORM
        # The current implementation redirects to area_config when _area_being_edited is set
        # even if the area wasn't found, because async_step_init checks _area_being_edited first
        assert result["step_id"] == "area_config"

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
                FlowResultType.FORM,
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
                FlowResultType.FORM,
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

    def test_get_default_decay_half_life_with_purpose(self):
        """Test _get_default_decay_half_life with valid purpose."""
        result = _get_default_decay_half_life("social")
        assert isinstance(result, float)
        assert result > 0

    def test_get_default_decay_half_life_invalid_purpose(self):
        """Test _get_default_decay_half_life with invalid purpose - should fallback."""
        result = _get_default_decay_half_life("invalid_purpose")
        assert isinstance(result, float)
        assert result > 0

    def test_get_default_decay_half_life_none(self):
        """Test _get_default_decay_half_life with None purpose."""
        result = _get_default_decay_half_life(None)
        assert isinstance(result, float)
        assert result > 0

    @pytest.mark.parametrize(
        ("areas", "is_initial"),
        [
            ("not a list", True),  # not_list
            (["not a dict", 123, None], True),  # invalid_area_dict
            ([{CONF_PURPOSE: "social"}], True),  # missing_name
            ([{CONF_AREA_ID: "", CONF_PURPOSE: "social"}], True),  # empty_area_id
            (
                [{CONF_AREA_ID: "unknown", CONF_PURPOSE: "social"}],
                True,
            ),  # unknown_area_id
        ],
    )
    def test_create_area_selection_schema_edge_cases(self, areas, is_initial):
        """Test _create_area_selection_schema with various edge cases."""
        schema = _create_area_selection_schema(areas, is_initial=is_initial)
        assert isinstance(schema, vol.Schema)

    def test_find_area_by_sanitized_id_unknown_area(self):
        """Test _find_area_by_sanitized_id when area ID is 'unknown'."""
        areas = [{CONF_AREA_ID: "unknown", CONF_PURPOSE: "social"}]
        result = _find_area_by_sanitized_id(areas, "unknown")
        assert result is not None  # Should find it
        assert result[CONF_AREA_ID] == "unknown"

    def test_validate_duplicate_area_id_raises(self):
        """Test _validate_duplicate_area_id raises vol.Invalid for duplicate."""
        flow = BaseOccupancyFlow()
        flattened_input = {CONF_AREA_ID: "test_area"}
        areas = [{CONF_AREA_ID: "test_area", CONF_PURPOSE: "social"}]
        with pytest.raises(vol.Invalid, match="already configured"):
            flow._validate_duplicate_area_id(flattened_input, areas, None, None)

    def test_validate_duplicate_area_id_same_area_editing(self):
        """Test _validate_duplicate_area_id allows same ID when editing same area."""
        flow = BaseOccupancyFlow()
        flattened_input = {CONF_AREA_ID: "test_area"}
        areas = [{CONF_AREA_ID: "test_area", CONF_PURPOSE: "social"}]
        # Should not raise when editing the same area
        flow._validate_duplicate_area_id(flattened_input, areas, "test_area", None)


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
    def test_find_area_by_name(self, areas, search_name, expected_found, expected_name):
        """Test finding area by name (legacy support)."""
        # Note: _find_area_by_name is for legacy support only
        # For new format areas with CONF_AREA_ID, use _find_area_by_id instead
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
