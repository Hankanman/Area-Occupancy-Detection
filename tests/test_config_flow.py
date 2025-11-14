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
    _ensure_primary_in_motion_sensors,
    _find_area_by_name,
    _find_area_by_sanitized_name,
    _flatten_sectioned_input,
    _get_area_summary_info,
    _get_default_decay_half_life,
    _get_include_entities,
    _get_purpose_display_name,
    _get_state_select_options,
    _handle_step_error,
    _remove_area_from_list,
    _sanitize_area_name_for_option,
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
    CONF_AREAS,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_OPTION_PREFIX_AREA,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DOMAIN,
)
from homeassistant.data_entry_flow import AbortFlow, FlowResultType
from tests.conftest import (
    create_area_config,
    create_user_input,
    patch_create_schema_context,
)


# ruff: noqa: SLF001, PLC0415, TID251
@pytest.mark.parametrize("expected_lingering_timers", [True])
class TestBaseOccupancyFlow:
    """Test BaseOccupancyFlow class."""

    @pytest.fixture
    def flow(self):
        """Create a BaseOccupancyFlow instance."""
        return BaseOccupancyFlow()

    def test_validate_config_valid(self, flow, config_flow_base_config):
        """Test validating a valid configuration."""
        flow._validate_config(config_flow_base_config)  # Should not raise any exception

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
                {"primary_occupancy_sensor": "binary_sensor.motion2"},
                "Primary occupancy sensor must be one of the selected motion sensors",
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
                {"name": ""},
                "Name is required",
            ),
            (
                {"decay_enabled": True, "decay_half_life": 0},
                "Decay half life must be between",
            ),
        ],
    )
    def test_validate_config_invalid_scenarios(
        self, flow, config_flow_base_config, invalid_config, expected_error
    ):
        """Test various invalid configuration scenarios."""
        test_config = {**config_flow_base_config, **invalid_config}

        with pytest.raises(vol.Invalid) as excinfo:
            flow._validate_config(test_config)
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
        ("input_name", "expected"),
        [
            ("Living Room", "Living_Room"),
            ("Living Room/Kitchen", "Living_Room_Kitchen"),
            ("Living_Room", "Living_Room"),
        ],
    )
    def test_sanitize_area_name_for_option(self, input_name, expected):
        """Test _sanitize_area_name_for_option function."""
        result = _sanitize_area_name_for_option(input_name)
        assert result == expected

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
        ("areas", "sanitized_name", "expected_name"),
        [
            (
                [{CONF_NAME: "Living Room", CONF_PURPOSE: "social"}],
                "Living_Room",
                "Living Room",
            ),
            (
                [
                    {CONF_NAME: "Living Room", CONF_PURPOSE: "social"},
                    {CONF_NAME: "Kitchen", CONF_PURPOSE: "work"},
                ],
                "Bedroom",
                None,
            ),
            ([], "Living_Room", None),
        ],
    )
    def test_find_area_by_sanitized_name(self, areas, sanitized_name, expected_name):
        """Test _find_area_by_sanitized_name function."""
        result = _find_area_by_sanitized_name(areas, sanitized_name)
        if expected_name is None:
            assert result is None
        else:
            assert result is not None
            assert result[CONF_NAME] == expected_name

    def test_build_area_description_placeholders(self):
        """Test _build_area_description_placeholders function."""
        area_config = {
            CONF_NAME: "Living Room",
            CONF_PURPOSE: "social",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_DOOR_SENSORS: ["binary_sensor.door1"],
            CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
            CONF_APPLIANCES: ["switch.light"],
            CONF_THRESHOLD: 60.0,
        }

        placeholders = _build_area_description_placeholders(area_config, "Living Room")

        assert placeholders["area_name"] == "Living Room"
        assert placeholders["motion_count"] == "1"
        assert placeholders["media_count"] == "1"
        assert placeholders["door_count"] == "1"
        assert placeholders["window_count"] == "1"
        assert placeholders["appliance_count"] == "1"
        assert placeholders["threshold"] == "60.0"

    def test_get_area_summary_info(self):
        """Test _get_area_summary_info function."""
        area = {
            CONF_NAME: "Living Room",
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
        assert "Living Room" not in summary  # Name should not be in summary
        assert "60" in summary  # Threshold should be included
        assert "3" in summary  # Total sensors count

    @pytest.mark.parametrize(
        ("areas", "is_initial"),
        [
            (
                [
                    {
                        CONF_NAME: "Living Room",
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
                        CONF_NAME: "Living Room",
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

    @pytest.fixture
    def mock_hass_for_schema(self, hass):
        """Set up hass for schema tests - real hass fixture provides states."""
        # Real hass fixture already has states, no mocking needed
        return hass

    @pytest.fixture
    def mock_entity_registry_for_schema(self):
        """Create mock entity registry for schema tests."""
        mock_registry = Mock()
        mock_registry.entities = {}
        return mock_registry

    @pytest.mark.parametrize(
        ("defaults", "is_options", "expected_name_present", "test_schema_validation"),
        [
            (None, False, True, False),  # defaults test
            (
                {
                    CONF_NAME: "Test Area",
                    CONF_MOTION_SENSORS: ["binary_sensor.motion_1"],
                    CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion_1",
                },
                False,
                True,
                True,
            ),  # with_defaults test
            (None, True, False, False),  # options_mode test
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
            "doors",
            "windows",
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
            assert CONF_NAME in schema_dict
        else:
            assert CONF_NAME not in schema_dict

        if test_schema_validation:
            # Test schema instantiation
            data = schema(
                {
                    CONF_NAME: "Test Area",
                    "purpose": {},
                    "motion": {},
                    "doors": {},
                    "windows": {},
                    "media": {},
                    "appliances": {},
                    "environmental": {},
                    "wasp_in_box": {},
                    "parameters": {},
                }
            )
            assert data[CONF_NAME] == "Test Area"


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
                        CONF_NAME: "Living Room",
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
                        CONF_NAME: "Living Room",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
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
                        CONF_NAME: "Living Room",
                        CONF_PURPOSE: "social",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                    }
                ],
                {"selected_option": f"{CONF_OPTION_PREFIX_AREA}Living_Room"},
                "area_action",
                FlowResultType.FORM,
                None,
            ),  # select area
        ],
    )
    async def test_async_step_user_scenarios(
        self,
        config_flow_flow,
        areas,
        user_input,
        expected_step_id,
        expected_type,
        patch_type,
    ):
        """Test async_step_user with various scenarios."""
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
            assert config_flow_flow._area_being_edited == "Living Room"

    @pytest.mark.parametrize(
        (
            "action",
            "expected_step_id",
            "expected_area_edited",
            "expected_area_to_remove",
            "needs_schema_mock",
        ),
        [
            (CONF_ACTION_EDIT, "area_config", "Living Room", None, True),
            (CONF_ACTION_REMOVE, "remove_area", None, "Living Room", False),
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
        config_flow_flow._area_being_edited = "Living Room"

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
        """Test that name is preserved when editing an area."""
        config_flow_flow._areas = [
            create_area_config(
                name="Living Room",
                motion_sensors=["binary_sensor.motion1"],
                primary_occupancy_sensor="binary_sensor.motion1",
            )
        ]
        config_flow_flow._area_being_edited = "Living Room"

        # User submits form without name field (or with empty name)
        user_input = create_user_input(name="")  # Empty name - should be preserved
        del user_input[CONF_NAME]  # Remove name to test preservation

        with (
            patch.object(config_flow_flow, "_validate_config") as mock_validate,
            patch_create_schema_context(),
        ):
            # Call async_step_area_config to trigger validation
            await config_flow_flow.async_step_area_config(user_input)

            # Should have preserved the name
            mock_validate.assert_called_once()
            call_args = mock_validate.call_args[0][0]
            assert call_args[CONF_NAME] == "Living Room"

    # Edge case tests
    async def test_config_flow_area_action_no_area(self, config_flow_flow):
        """Test when _area_being_edited is None."""
        config_flow_flow._areas = [
            create_area_config(name="Test")
        ]  # Has areas, so won't auto-start
        config_flow_flow._area_being_edited = None
        with patch_create_schema_context():
            result = await config_flow_flow.async_step_area_action()
            assert result.get("type") == FlowResultType.FORM
            assert result.get("step_id") == "user"

    async def test_config_flow_area_action_area_not_found(self, config_flow_flow):
        """Test when area not found in areas list."""
        config_flow_flow._areas = [
            create_area_config(name="Test")
        ]  # Has areas, so won't auto-start
        config_flow_flow._area_being_edited = "NonExistent"
        with patch_create_schema_context():
            result = await config_flow_flow.async_step_area_action()
            assert result.get("type") == FlowResultType.FORM
            assert result.get("step_id") == "user"

    async def test_config_flow_remove_area_no_area(self, config_flow_flow):
        """Test when _area_to_remove is None."""
        config_flow_flow._areas = [
            create_area_config(name="Test")
        ]  # Has areas, so won't auto-start
        config_flow_flow._area_to_remove = None
        with patch_create_schema_context():
            result = await config_flow_flow.async_step_remove_area()
            assert result.get("type") == FlowResultType.FORM
            assert result.get("step_id") == "user"

    async def test_config_flow_remove_area_cancel(self, config_flow_flow):
        """Test cancellation path."""
        config_flow_flow._areas = [
            create_area_config(
                name="Living Room",
                motion_sensors=["binary_sensor.motion1"],
            )
        ]
        config_flow_flow._area_to_remove = "Living Room"
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
        config_flow_flow._area_to_remove = "Living Room"
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
            assert area_data.get(CONF_NAME) == "Living_Room"  # Sanitized
            assert area_data.get(CONF_MOTION_SENSORS) == ["binary_sensor.motion1"]
            assert (
                area_data.get(CONF_PRIMARY_OCCUPANCY_SENSOR) == "binary_sensor.motion1"
            )
            assert area_data.get(CONF_THRESHOLD) == 60

    async def test_config_flow_with_existing_entry(self, config_flow_flow, hass):
        """Test config flow when entry already exists."""
        hass.data = {}

        # When finish setup is selected, it should check for existing entry
        config_flow_flow._areas = [
            create_area_config(
                name="Living Room",
                motion_sensors=["binary_sensor.motion1"],
                primary_occupancy_sensor="binary_sensor.motion1",
            )
        ]

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

    async def test_config_flow_user_finish_setup_no_areas(self, config_flow_flow):
        """Test config flow finish setup with no areas."""
        flow = config_flow_flow
        flow._areas = []

        user_input = {"selected_option": CONF_ACTION_FINISH_SETUP}
        result = await flow.async_step_user(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "user"
        assert "errors" in result
        assert "base" in result["errors"]

    async def test_config_flow_user_finish_setup_validation_error(
        self, config_flow_flow
    ):
        """Test config flow finish setup with validation error."""
        flow = config_flow_flow
        flow._areas = [create_area_config(name="Living Room", motion_sensors=[])]

        user_input = {"selected_option": CONF_ACTION_FINISH_SETUP}
        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            result = await flow.async_step_user(user_input)
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "user"
            assert "errors" in result

    async def test_config_flow_user_finish_setup_unexpected_error(
        self, config_flow_flow
    ):
        """Test config flow finish setup with unexpected error."""
        flow = config_flow_flow
        flow._areas = [create_area_config(name="Living Room")]

        user_input = {"selected_option": CONF_ACTION_FINISH_SETUP}
        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
            patch.object(flow, "_validate_config", side_effect=KeyError("test")),
        ):
            result = await flow.async_step_user(user_input)
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "user"
            assert "errors" in result

    async def test_config_flow_area_action_show_form(self, config_flow_flow):
        """Test config flow area action shows form when no user input."""
        flow = config_flow_flow
        flow._areas = [create_area_config(name="Living Room")]
        flow._area_being_edited = "Living Room"

        result = await flow.async_step_area_action()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "area_action"
        assert "data_schema" in result
        assert "description_placeholders" in result

    async def test_config_flow_remove_area_show_form(self, config_flow_flow):
        """Test config flow remove area shows confirmation form."""
        flow = config_flow_flow
        flow._areas = [create_area_config(name="Living Room")]
        flow._area_to_remove = "Living Room"

        result = await flow.async_step_remove_area()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "remove_area"
        assert "data_schema" in result
        assert "description_placeholders" in result
        assert result["description_placeholders"]["area_name"] == "Living Room"

    async def test_error_recovery_in_config_flow(self, config_flow_flow):
        """Test error recovery in config flow."""
        # First attempt with invalid data in area_config
        invalid_input = create_user_input(
            name="Living Room",
            motion={CONF_MOTION_SENSORS: [], CONF_PRIMARY_OCCUPANCY_SENSOR: ""},
        )

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

    def test_get_areas_from_config_multi_area_format(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test _get_areas_from_config with multi-area format."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        areas = flow._get_areas_from_config()
        assert isinstance(areas, list)
        assert len(areas) == 1
        assert areas[0][CONF_NAME] == "Living Room"

    def test_get_areas_from_config_legacy_format(
        self, config_flow_options_flow, config_flow_mock_config_entry_legacy
    ):
        """Test _get_areas_from_config with legacy format."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_legacy
        areas = flow._get_areas_from_config()
        assert isinstance(areas, list)
        assert len(areas) == 1
        assert areas[0][CONF_NAME] == "Legacy Area"

    def test_get_areas_from_config_legacy_no_name(
        self, config_flow_options_flow, config_flow_mock_config_entry_legacy_no_name
    ):
        """Test _get_areas_from_config with legacy format without name."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_legacy_no_name
        areas = flow._get_areas_from_config()
        assert isinstance(areas, list)
        assert len(areas) == 1
        assert areas[0][CONF_NAME] == "Area"

    async def test_options_flow_init_with_device_id(
        self, config_flow_options_flow, hass, device_registry
    ):
        """Test options flow init when called from device registry."""
        flow = config_flow_options_flow

        # Add config entry to hass.config_entries so device registry can link to it
        hass.config_entries._entries[flow.config_entry.entry_id] = flow.config_entry

        # Create a device in the registry
        device_entry = device_registry.async_get_or_create(
            config_entry_id=flow.config_entry.entry_id,
            identifiers={(DOMAIN, "Test Area")},
            name="Test Area",
        )

        # Update flow's device_id to match the created device
        flow._device_id = device_entry.id

        with patch_create_schema_context():
            result = await flow.async_step_init()
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "area_config"
            assert flow._area_being_edited == "Test Area"

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
            # Verify name field was added to schema
            areas = result["data"][CONF_AREAS]
            assert len(areas) == 2  # Original + new
            assert any(area[CONF_NAME] == "Kitchen" for area in areas)

    async def test_options_flow_area_config_migration(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config with migration."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        # Use sanitized name to avoid triggering migration due to sanitization
        flow._area_being_edited = "Living_Room"

        mock_migrate = AsyncMock()
        user_input = create_user_input(name="Living_Room")

        with patch_create_schema_context():
            result = await flow.async_step_area_config(
                user_input, migrate_fn=mock_migrate
            )
            # Should succeed without calling migration since name didn't change
            assert result["type"] == FlowResultType.CREATE_ENTRY
            mock_migrate.assert_not_called()

    async def test_options_flow_area_config_migration_on_rename(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config triggers migration on rename."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        mock_migrate = AsyncMock()
        user_input = create_user_input(name="Living Room Renamed")

        with patch_create_schema_context():
            result = await flow.async_step_area_config(
                user_input, migrate_fn=mock_migrate
            )
            assert result["type"] == FlowResultType.CREATE_ENTRY
            mock_migrate.assert_called_once()

    async def test_options_flow_area_config_migration_error(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config handles migration errors gracefully."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        mock_migrate = AsyncMock(side_effect=ValueError("Migration failed"))
        user_input = create_user_input(name="Living Room Renamed")

        with patch_create_schema_context():
            result = await flow.async_step_area_config(
                user_input, migrate_fn=mock_migrate
            )
            # Should still succeed even if migration fails
            assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_options_flow_area_config_migration_homeassistant_error(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config handles HomeAssistantError in migration."""
        from homeassistant.exceptions import HomeAssistantError

        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        mock_migrate = AsyncMock(side_effect=HomeAssistantError("Migration failed"))
        user_input = create_user_input(name="Living Room Renamed")

        with patch_create_schema_context():
            result = await flow.async_step_area_config(
                user_input, migrate_fn=mock_migrate
            )
            # Should still succeed even if migration fails
            assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_options_flow_area_config_migration_key_error(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config handles KeyError in migration."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        mock_migrate = AsyncMock(side_effect=KeyError("Migration failed"))
        user_input = create_user_input(name="Living Room Renamed")

        with patch_create_schema_context():
            result = await flow.async_step_area_config(
                user_input, migrate_fn=mock_migrate
            )
            # Should still succeed even if migration fails
            assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_options_flow_area_config_no_old_area(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area config when old area is not found."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "NonExistent"

        user_input = create_user_input(name="New Name")
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

    async def test_options_flow_area_action_edit(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area action edit."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        user_input = {"action": CONF_ACTION_EDIT}
        with patch_create_schema_context():
            result = await flow.async_step_area_action(user_input)
            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "area_config"

    async def test_options_flow_area_action_remove(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area action remove."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        user_input = {"action": CONF_ACTION_REMOVE}
        result = await flow.async_step_area_action(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "remove_area"

    async def test_options_flow_area_action_cancel(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area action cancel."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        user_input = {"action": CONF_ACTION_CANCEL}
        result = await flow.async_step_area_action(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

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

    async def test_options_flow_area_action_show_form(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow area action shows form when no user input."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_being_edited = "Living Room"

        result = await flow.async_step_area_action()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "area_action"
        assert "data_schema" in result
        assert "description_placeholders" in result

    async def test_options_flow_remove_area_confirm(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow remove area with confirmation."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_to_remove = "Living Room"

        # Add another area so we can remove one
        flow.config_entry.data[CONF_AREAS].append(
            create_area_config(
                name="Kitchen", motion_sensors=["binary_sensor.kitchen_motion"]
            )
        )

        user_input = {"confirm": True}
        result = await flow.async_step_remove_area(user_input)
        assert result["type"] == FlowResultType.CREATE_ENTRY

    async def test_options_flow_remove_area_cancel(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow remove area cancellation."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_to_remove = "Living Room"

        user_input = {"confirm": False}
        result = await flow.async_step_remove_area(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

    async def test_options_flow_remove_area_last_area_error(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow remove area prevents removing last area."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_to_remove = "Living Room"

        user_input = {"confirm": True}
        result = await flow.async_step_remove_area(user_input)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "remove_area"
        assert "errors" in result
        assert "last area" in result["errors"]["base"].lower()

    async def test_options_flow_remove_area_no_area(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow remove area when no area is set."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_to_remove = None

        result = await flow.async_step_remove_area()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

    async def test_options_flow_remove_area_show_form(
        self, config_flow_options_flow, config_flow_mock_config_entry_with_areas
    ):
        """Test options flow remove area shows confirmation form."""
        flow = config_flow_options_flow
        flow.config_entry = config_flow_mock_config_entry_with_areas
        flow._area_to_remove = "Living Room"

        result = await flow.async_step_remove_area()
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "remove_area"
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

    def test_create_area_selection_schema_not_list(self):
        """Test _create_area_selection_schema when areas is not a list."""
        schema = _create_area_selection_schema("not a list", is_initial=True)
        assert isinstance(schema, vol.Schema)

    def test_create_area_selection_schema_invalid_area_dict(self):
        """Test _create_area_selection_schema when area is not a dict."""
        areas = ["not a dict", 123, None]
        schema = _create_area_selection_schema(areas, is_initial=True)
        assert isinstance(schema, vol.Schema)

    def test_create_area_selection_schema_missing_name(self):
        """Test _create_area_selection_schema when area name is missing."""
        areas = [{CONF_PURPOSE: "social"}]  # No CONF_NAME
        schema = _create_area_selection_schema(areas, is_initial=True)
        assert isinstance(schema, vol.Schema)

    def test_create_area_selection_schema_empty_name(self):
        """Test _create_area_selection_schema when area name is empty."""
        areas = [{CONF_NAME: "", CONF_PURPOSE: "social"}]
        schema = _create_area_selection_schema(areas, is_initial=True)
        assert isinstance(schema, vol.Schema)

    def test_create_area_selection_schema_unknown_name(self):
        """Test _create_area_selection_schema when area name is 'Unknown'."""
        areas = [{CONF_NAME: "Unknown", CONF_PURPOSE: "social"}]
        schema = _create_area_selection_schema(areas, is_initial=True)
        assert isinstance(schema, vol.Schema)

    def test_find_area_by_sanitized_name_unknown_area(self):
        """Test _find_area_by_sanitized_name when area name is 'Unknown'."""
        areas = [{CONF_NAME: "Unknown", CONF_PURPOSE: "social"}]
        result = _find_area_by_sanitized_name(areas, "Unknown")
        assert result is None

    def test_find_area_by_sanitized_name_empty_name(self):
        """Test _find_area_by_sanitized_name when area name is empty."""
        areas = [{CONF_NAME: "", CONF_PURPOSE: "social"}]
        result = _find_area_by_sanitized_name(areas, "")
        assert result is None

    def test_validate_duplicate_name_raises(self):
        """Test _validate_duplicate_name_internal raises vol.Invalid for duplicate."""
        flow = BaseOccupancyFlow()
        flattened_input = {CONF_NAME: "Test Area"}
        areas = [{CONF_NAME: "Test Area", CONF_PURPOSE: "social"}]
        with pytest.raises(vol.Invalid, match="already exists"):
            flow._validate_duplicate_name_internal(flattened_input, areas)

    def test_validate_duplicate_name_same_area_editing(self):
        """Test _validate_duplicate_name_internal allows same name when editing same area."""
        flow = BaseOccupancyFlow()
        flattened_input = {CONF_NAME: "Test Area"}
        areas = [{CONF_NAME: "Test Area", CONF_PURPOSE: "social"}]
        # Should not raise when editing the same area
        flow._validate_duplicate_name_internal(
            flattened_input, areas, area_being_edited="Test Area"
        )

    def test_validate_config_name_sanitization_error(self):
        """Test _validate_config when validate_and_sanitize_area_name raises ValueError."""
        flow = BaseOccupancyFlow()
        with (
            patch(
                "custom_components.area_occupancy.config_flow.validate_and_sanitize_area_name",
                side_effect=ValueError("Invalid name"),
            ),
            pytest.raises(vol.Invalid, match="Invalid name"),
        ):
            flow._validate_config({CONF_NAME: "Invalid/Name"})

    def test_validate_config_empty_purpose(self):
        """Test _validate_config with empty purpose."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_PURPOSE: "",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }
        with pytest.raises(vol.Invalid, match="Purpose is required"):
            flow._validate_config(config)

    def test_validate_config_no_primary_sensor(self):
        """Test _validate_config with no primary sensor."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
        }
        with pytest.raises(
            vol.Invalid, match="primary occupancy sensor must be selected"
        ):
            flow._validate_config(config)

    def test_validate_config_media_devices_no_states(self):
        """Test _validate_config with media devices but no states."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_MEDIA_DEVICES: ["media_player.tv"],
            CONF_MEDIA_ACTIVE_STATES: [],
        }
        with pytest.raises(vol.Invalid, match="Media active states are required"):
            flow._validate_config(config)

    def test_validate_config_appliances_no_states(self):
        """Test _validate_config with appliances but no states."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_APPLIANCES: ["switch.light"],
            CONF_APPLIANCE_ACTIVE_STATES: [],
        }
        with pytest.raises(vol.Invalid, match="Appliance active states are required"):
            flow._validate_config(config)

    def test_validate_config_doors_no_state(self):
        """Test _validate_config with door sensors but no state."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_DOOR_SENSORS: ["binary_sensor.door1"],
            CONF_DOOR_ACTIVE_STATE: "",
        }
        with pytest.raises(vol.Invalid, match="Door active state is required"):
            flow._validate_config(config)

    def test_validate_config_windows_no_state(self):
        """Test _validate_config with window sensors but no state."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_WINDOW_SENSORS: ["binary_sensor.window1"],
            CONF_WINDOW_ACTIVE_STATE: "",
        }
        with pytest.raises(vol.Invalid, match="Window active state is required"):
            flow._validate_config(config)


class TestStaticMethods:
    """Test static methods."""

    def test_async_get_options_flow(self):
        """Test static method returns OptionsFlow instance."""
        from homeassistant.config_entries import ConfigEntry

        mock_entry = Mock(spec=ConfigEntry)
        result = AreaOccupancyConfigFlow.async_get_options_flow(mock_entry)
        assert isinstance(result, AreaOccupancyOptionsFlow)

    def test_async_get_device_options_flow(self):
        """Test static method sets device_id."""
        from homeassistant.config_entries import ConfigEntry

        mock_entry = Mock(spec=ConfigEntry)
        result = AreaOccupancyConfigFlow.async_get_device_options_flow(
            mock_entry, "test_device_id"
        )
        assert isinstance(result, AreaOccupancyOptionsFlow)
        assert result._device_id == "test_device_id"


class TestNewHelperFunctions:
    """Test newly extracted helper functions."""

    def test_ensure_primary_in_motion_sensors_adds_primary(self):
        """Test that primary sensor is auto-added to motion sensors."""
        user_input = {
            "motion": {
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.primary",
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            }
        }
        _ensure_primary_in_motion_sensors(user_input)
        assert "binary_sensor.primary" in user_input["motion"][CONF_MOTION_SENSORS]

    def test_ensure_primary_in_motion_sensors_already_present(self):
        """Test that primary sensor is not added if already present."""
        user_input = {
            "motion": {
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.primary",
                CONF_MOTION_SENSORS: ["binary_sensor.primary", "binary_sensor.motion1"],
            }
        }
        original_list = user_input["motion"][CONF_MOTION_SENSORS].copy()
        _ensure_primary_in_motion_sensors(user_input)
        assert user_input["motion"][CONF_MOTION_SENSORS] == original_list

    def test_ensure_primary_in_motion_sensors_no_primary(self):
        """Test that function handles missing primary sensor gracefully."""
        user_input = {
            "motion": {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            }
        }
        _ensure_primary_in_motion_sensors(user_input)
        assert user_input["motion"][CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]

    def test_apply_purpose_based_decay_default_with_purpose(self):
        """Test applying purpose-based decay default."""
        flattened_input = {CONF_PURPOSE: "social"}
        _apply_purpose_based_decay_default(flattened_input, "social")
        assert CONF_DECAY_HALF_LIFE in flattened_input

    def test_apply_purpose_based_decay_default_no_purpose(self):
        """Test that function returns early when purpose is None."""
        flattened_input = {}
        _apply_purpose_based_decay_default(flattened_input, None)
        assert CONF_DECAY_HALF_LIFE not in flattened_input

    def test_flatten_sectioned_input(self):
        """Test flattening sectioned input."""
        user_input = {
            CONF_NAME: "Test Area",
            "motion": {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            },
            "purpose": {CONF_PURPOSE: "social"},
            "wasp_in_box": {CONF_WASP_ENABLED: True},
        }
        result = _flatten_sectioned_input(user_input)
        assert result[CONF_NAME] == "Test Area"
        assert result[CONF_MOTION_SENSORS] == ["binary_sensor.motion1"]
        assert result[CONF_PURPOSE] == "social"
        assert result[CONF_WASP_ENABLED] is True

    def test_find_area_by_name_found(self):
        """Test finding area by name."""
        areas = [
            {CONF_NAME: "Living Room", CONF_PURPOSE: "social"},
            {CONF_NAME: "Kitchen", CONF_PURPOSE: "work"},
        ]
        result = _find_area_by_name(areas, "Living Room")
        assert result is not None
        assert result[CONF_NAME] == "Living Room"

    def test_find_area_by_name_not_found(self):
        """Test finding area by name when not found."""
        areas = [
            {CONF_NAME: "Living Room", CONF_PURPOSE: "social"},
        ]
        result = _find_area_by_name(areas, "Bedroom")
        assert result is None

    def test_update_area_in_list_update_existing(self):
        """Test updating an existing area in list."""
        areas = [
            {CONF_NAME: "Living Room", CONF_PURPOSE: "social"},
            {CONF_NAME: "Kitchen", CONF_PURPOSE: "work"},
        ]
        updated_area = {CONF_NAME: "Living Room", CONF_PURPOSE: "entertainment"}
        result = _update_area_in_list(areas, updated_area, "Living Room")
        assert len(result) == 2
        assert result[0][CONF_PURPOSE] == "entertainment"
        assert result[1][CONF_PURPOSE] == "work"

    def test_update_area_in_list_add_new(self):
        """Test adding a new area to list."""
        areas = [
            {CONF_NAME: "Living Room", CONF_PURPOSE: "social"},
        ]
        new_area = {CONF_NAME: "Kitchen", CONF_PURPOSE: "work"}
        result = _update_area_in_list(areas, new_area, None)
        assert len(result) == 2
        assert result[1][CONF_NAME] == "Kitchen"

    def test_remove_area_from_list(self):
        """Test removing an area from list."""
        areas = [
            {CONF_NAME: "Living Room", CONF_PURPOSE: "social"},
            {CONF_NAME: "Kitchen", CONF_PURPOSE: "work"},
        ]
        result = _remove_area_from_list(areas, "Living Room")
        assert len(result) == 1
        assert result[0][CONF_NAME] == "Kitchen"

    def test_handle_step_error_homeassistant_error(self):
        """Test error handling for HomeAssistantError."""
        from homeassistant.exceptions import HomeAssistantError

        err = HomeAssistantError("Test error")
        result = _handle_step_error(err)
        assert result == "Test error"

    def test_handle_step_error_vol_invalid(self):
        """Test error handling for vol.Invalid."""
        err = vol.Invalid("Validation error")
        result = _handle_step_error(err)
        assert result == "Validation error"

    def test_handle_step_error_value_error(self):
        """Test error handling for ValueError."""
        err = ValueError("Value error")
        result = _handle_step_error(err)
        assert result == "unknown"

    def test_handle_step_error_key_error(self):
        """Test error handling for KeyError."""
        err = KeyError("key")
        result = _handle_step_error(err)
        assert result == "unknown"

    def test_handle_step_error_type_error(self):
        """Test error handling for TypeError."""
        err = TypeError("Type error")
        result = _handle_step_error(err)
        assert result == "unknown"
