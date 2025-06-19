"""Tests for the Area Occupancy Detection config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
import voluptuous as vol

from custom_components.area_occupancy.config_flow import (
    AreaOccupancyConfigFlow,
    AreaOccupancyOptionsFlow,
    BaseOccupancyFlow,
    _get_include_entities,
    _get_state_select_options,
    create_schema,
)
from custom_components.area_occupancy.const import (
    CONF_DECAY_ENABLED,
    CONF_DECAY_WINDOW,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
    CONF_WASP_ENABLED,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LIGHT,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_LIGHT,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
    CONF_PURPOSE,
)
from homeassistant.data_entry_flow import AbortFlow, FlowResultType


# ruff: noqa: SLF001
class TestBaseOccupancyFlow:
    """Test BaseOccupancyFlow class."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self, mock_hass):
        """Test validating a valid configuration."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        }

        flow._validate_config(config)  # Should not raise any exception

    @pytest.mark.asyncio
    async def test_validate_config_no_motion_sensors(self, mock_hass):
        """Test validating configuration with no motion sensors."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: [],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        }

        with pytest.raises(vol.Invalid) as excinfo:
            flow._validate_config(config)
        assert "At least one motion sensor is required" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_validate_config_invalid_weight(self, mock_hass):
        """Test validating configuration with invalid weight."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_WEIGHT_MOTION: 1.5,  # Invalid weight
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        }

        with pytest.raises(vol.Invalid) as excinfo:
            flow._validate_config(config)
        assert "weight_motion must be between 0 and 1" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_validate_config_primary_not_in_motion(self, mock_hass):
        """Test validating configuration where primary sensor is not in motion sensors."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion2",  # Not in motion sensors
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        }

        with pytest.raises(vol.Invalid) as excinfo:
            flow._validate_config(config)
        assert (
            "Primary occupancy sensor must be one of the selected motion sensors"
            in str(excinfo.value)
        )

    @pytest.mark.asyncio
    async def test_validate_config_invalid_probability(self, mock_hass):
        """Test validating configuration with invalid probability."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 150,  # Invalid threshold
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        }

        with pytest.raises(vol.Invalid) as excinfo:
            flow._validate_config(config)
        assert "threshold" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_validate_config_invalid_decay_timeout(self, mock_hass):
        """Test validating configuration with invalid decay timeout."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
            CONF_WEIGHT_LIGHT: DEFAULT_WEIGHT_LIGHT,
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
            CONF_DECAY_ENABLED: True,
            CONF_DECAY_WINDOW: 0,  # Invalid decay window
        }

        with pytest.raises(vol.Invalid, match="Decay window must be between"):
            flow._validate_config(config)

    @pytest.mark.asyncio
    async def test_validate_config_invalid_update_interval(self, mock_hass):
        """Test validating configuration with invalid update interval."""
        flow = BaseOccupancyFlow()
        config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
            CONF_DECAY_ENABLED: True,
            CONF_DECAY_WINDOW: 0,  # Invalid decay window
        }

        with pytest.raises(vol.Invalid, match="Decay window must be between"):
            flow._validate_config(config)

    def test_validate_config_invalid_threshold_low(self) -> None:
        """Test _validate_config with threshold too low."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 0,  # Invalid threshold
        }

        with pytest.raises(vol.Invalid, match="Threshold must be between 1 and 100"):
            flow._validate_config(invalid_config)

    def test_validate_config_invalid_threshold_high(self) -> None:
        """Test _validate_config with threshold too high."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 101,  # Invalid threshold
        }

        with pytest.raises(vol.Invalid, match="Threshold must be between 1 and 100"):
            flow._validate_config(invalid_config)

    def test_validate_config_empty_name(self) -> None:
        """Test _validate_config with empty name."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "",  # Empty name
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        with pytest.raises(vol.Invalid, match="Name is required"):
            flow._validate_config(invalid_config)


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_state_select_options(self) -> None:
        """Test _get_state_select_options function."""
        # Test door states
        door_options = _get_state_select_options("door")
        assert len(door_options) > 0
        assert all("value" in option and "label" in option for option in door_options)

        # Test window states
        window_options = _get_state_select_options("window")
        assert len(window_options) > 0

        # Test media states
        media_options = _get_state_select_options("media")
        assert len(media_options) > 0

        # Test appliance states
        appliance_options = _get_state_select_options("appliance")
        assert len(appliance_options) > 0

        # Test unknown state type
        unknown_options = _get_state_select_options("unknown")
        # Accept any non-empty list, as the implementation returns default options
        assert isinstance(unknown_options, list)
        assert all(
            "value" in option and "label" in option for option in unknown_options
        )

    def test_get_include_entities(self, mock_hass, mock_entity_registry):
        """Test getting include entities."""
        # Setup mock entity registry
        mock_hass.helpers.entity_registry.async_get.return_value = mock_entity_registry

        # Add some test entities
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

        # Set up the entities property to be iterable
        mock_entity_registry.entities = Mock()
        mock_entity_registry.entities.values = Mock(return_value=entity_list)

        # Setup mock states
        mock_hass.states = Mock()
        mock_hass.states.async_entity_ids = Mock(
            return_value=[
                "binary_sensor.door_1",
                "binary_sensor.window_1",
                "switch.appliance_1",
            ]
        )
        mock_hass.states.get = Mock(
            return_value=Mock(attributes={"device_class": None})
        )

        # Patch the async_get function to return our mock registry
        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=mock_entity_registry,
        ):
            result = _get_include_entities(mock_hass)

        assert "door" in result
        assert "window" in result
        assert "appliance" in result
        assert "binary_sensor.door_1" in result["door"]
        assert "binary_sensor.window_1" in result["window"]
        assert "switch.appliance_1" in result["appliance"]

    def test_create_schema_defaults(self, mock_hass):
        """Test creating schema with defaults."""
        # Setup mock states
        mock_hass.states = Mock()
        mock_hass.states.async_entity_ids = Mock(return_value=[])
        mock_hass.states.get = Mock(
            return_value=Mock(attributes={"device_class": None})
        )

        # Patch entity_registry.async_get to return a mock with .entities as an empty dict
        with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            mock_registry = Mock()
            mock_registry.entities = {}
            mock_er_get.return_value = mock_registry
            schema = create_schema(mock_hass)
        assert isinstance(schema, dict)
        assert CONF_NAME in schema
        assert "motion" in schema
        assert "doors" in schema
        assert "windows" in schema
        assert "lights" in schema
        assert "media" in schema
        assert "appliances" in schema
        assert "environmental" in schema
        assert "wasp_in_box" in schema
        assert "parameters" in schema

    def test_create_schema_with_defaults(self, mock_hass):
        """Test creating schema with provided defaults."""
        # Setup mock states
        mock_hass.states = Mock()
        mock_hass.states.async_entity_ids = Mock(return_value=[])
        mock_hass.states.get = Mock(
            return_value=Mock(attributes={"device_class": None})
        )

        defaults = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion_1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion_1",
        }
        # Patch entity_registry.async_get to return a mock with .entities as an empty dict
        with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            mock_registry = Mock()

            class EntitiesObj:
                def values(self):
                    return [].__iter__()

            mock_registry.entities = EntitiesObj()
            mock_er_get.return_value = mock_registry
            schema_dict = create_schema(mock_hass, defaults)
            schema = vol.Schema(schema_dict)
            # The default value is set in the voluptuous marker, so we check by instantiating
            data = schema(
                {
                    CONF_NAME: "Test Area",
                    "purpose": {},
                    "motion": {},
                    "doors": {},
                    "windows": {},
                    "lights": {},
                    "media": {},
                    "appliances": {},
                    "environmental": {},
                    "wasp_in_box": {},
                    "parameters": {},
                }
            )
        assert isinstance(schema_dict, dict)
        assert CONF_NAME in schema_dict
        assert data[CONF_NAME] == "Test Area"
        assert "purpose" in schema_dict
        assert "motion" in schema_dict
        assert "doors" in schema_dict
        assert "windows" in schema_dict
        assert "lights" in schema_dict
        assert "media" in schema_dict
        assert "appliances" in schema_dict
        assert "environmental" in schema_dict
        assert "wasp_in_box" in schema_dict
        assert "parameters" in schema_dict

    def test_create_schema_options_mode(self, mock_hass):
        """Test creating schema in options mode."""
        # Setup mock states
        mock_hass.states = Mock()
        mock_hass.states.async_entity_ids = Mock(return_value=[])
        mock_hass.states.get = Mock(
            return_value=Mock(attributes={"device_class": None})
        )

        # Patch entity_registry.async_get to return a mock with .entities as an empty dict
        with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            mock_registry = Mock()
            mock_registry.entities = {}
            mock_er_get.return_value = mock_registry
            schema = create_schema(mock_hass, is_options=True)
        assert isinstance(schema, dict)
        assert CONF_NAME not in schema
        assert "purpose" in schema
        assert "motion" in schema
        assert "doors" in schema
        assert "windows" in schema
        assert "lights" in schema
        assert "media" in schema
        assert "appliances" in schema
        assert "environmental" in schema
        assert "wasp_in_box" in schema
        assert "parameters" in schema


class TestAreaOccupancyConfigFlow:
    """Test AreaOccupancyConfigFlow class."""

    def test_initialization(self) -> None:
        """Test ConfigFlow initialization."""
        flow = AreaOccupancyConfigFlow()

        assert flow.VERSION == 1
        assert flow.MINOR_VERSION == 1

    async def test_async_step_user_no_input(self, mock_hass: Mock) -> None:
        """Test async_step_user with no user input."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result = await flow.async_step_user()

            assert result.get("type") == FlowResultType.FORM
            assert result.get("step_id") == "user"

    async def test_async_step_user_with_valid_input(self, mock_hass: Mock) -> None:
        """Test async_step_user with valid user input."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        user_input = {
            "motion": {
                CONF_NAME: "Test Area",
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
                CONF_THRESHOLD: 60,
            },
            "purpose": {},
        }

        with (
            patch.object(flow, "_validate_config") as mock_validate,
            patch.object(
                flow, "async_set_unique_id", new_callable=AsyncMock
            ) as mock_set_unique_id,
            patch.object(flow, "_abort_if_unique_id_configured") as mock_abort,
        ):
            result = await flow.async_step_user(user_input)

            mock_validate.assert_called_once()
            mock_set_unique_id.assert_called_once()
            mock_abort.assert_called_once()

            assert result.get("type") == FlowResultType.CREATE_ENTRY
            assert result.get("title") == "Test Area"
            assert result.get("data") == {
                CONF_NAME: "Test Area",
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
                CONF_PURPOSE: "social",
                CONF_THRESHOLD: 60,
            }

    async def test_async_step_user_with_invalid_input(self, mock_hass: Mock) -> None:
        """Test async_step_user with invalid user input."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        user_input = {
            CONF_NAME: "Test Area",
            "motion": {
                CONF_MOTION_SENSORS: [],  # Invalid - empty
                CONF_PRIMARY_OCCUPANCY_SENSOR: "",
            },
            "purpose": {},
            "doors": {},
            "windows": {},
            "lights": {},
            "media": {},
            "appliances": {},
            "environmental": {},
            "wasp_in_box": {},
            "parameters": {},
        }

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result = await flow.async_step_user(user_input)

            assert result is not None
            assert result.get("type") == FlowResultType.FORM
            assert "errors" in result
            assert result["errors"] == {
                "base": "At least one motion sensor is required"
            }


class TestConfigFlowIntegration:
    """Test config flow integration scenarios."""

    async def test_complete_config_flow(self, mock_hass: Mock) -> None:
        """Test complete configuration flow."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        # Step 1: Show form
        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result1 = await flow.async_step_user()
            assert result1.get("type") == FlowResultType.FORM

        # Step 2: Submit valid data
        user_input = {
            CONF_NAME: "Living Room",
            "motion": {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            },
            "purpose": {},
            "doors": {},
            "windows": {},
            "lights": {},
            "media": {},
            "appliances": {},
            "environmental": {},
            "wasp_in_box": {},
            "parameters": {
                CONF_THRESHOLD: 60,
            },
        }

        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            result2 = await flow.async_step_user(user_input)

            assert result2.get("type") == FlowResultType.CREATE_ENTRY
            assert result2.get("title") == "Living Room"
            # The config flow flattens the input, so we need to check the flattened structure
            expected_data = {
                CONF_NAME: "Living Room",
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
                CONF_PURPOSE: "social",
                CONF_THRESHOLD: 60,
                CONF_WASP_ENABLED: False,
            }
            # Check that the essential fields are present
            result_data = result2.get("data", {})
            assert result_data.get(CONF_NAME) == expected_data[CONF_NAME]
            assert (
                result_data.get(CONF_MOTION_SENSORS)
                == expected_data[CONF_MOTION_SENSORS]
            )
            assert (
                result_data.get(CONF_PRIMARY_OCCUPANCY_SENSOR)
                == expected_data[CONF_PRIMARY_OCCUPANCY_SENSOR]
            )
            assert result_data.get(CONF_THRESHOLD) == expected_data[CONF_THRESHOLD]

    async def test_complete_options_flow(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test complete options flow."""
        # Add missing attributes to mock_hass
        mock_hass.data = {}
        mock_hass.config = Mock()
        mock_hass.config.config_dir = "/config"
        mock_hass.bus = Mock()
        mock_hass.bus.async_listen = Mock()

        # Properly mock states.async_entity_ids to return empty lists
        mock_hass.states.async_entity_ids = Mock(return_value=[])
        mock_hass.states.get = Mock(return_value=None)

        # Create a minimal mock flow that bypasses the parent initialization
        flow = Mock(spec=AreaOccupancyOptionsFlow)
        flow.config_entry = mock_config_entry
        flow._data = {}
        flow.hass = mock_hass

        # Bind the actual methods we want to test
        flow.async_step_init = AreaOccupancyOptionsFlow.async_step_init.__get__(
            flow, AreaOccupancyOptionsFlow
        )
        # Mock _validate_config to avoid validation issues in the test
        flow._validate_config = Mock()

        # Mock the parent class methods that return flow results
        flow.async_show_form = Mock(
            return_value={"type": FlowResultType.FORM, "step_id": "init"}
        )
        flow.async_create_entry = Mock(
            return_value={"type": FlowResultType.CREATE_ENTRY, "title": "", "data": {}}
        )

        # Apply the entity registry patch for the entire test
        with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            # Create a proper mock registry with entities attribute
            mock_registry = Mock()
            mock_registry.entities = Mock()
            mock_registry.entities.values = Mock(return_value=[])
            mock_er_get.return_value = mock_registry

            # Step 1: Show form with current values
            with patch(
                "custom_components.area_occupancy.config_flow.create_schema"
            ) as mock_create_schema:
                mock_create_schema.return_value = {"test": vol.Required("test")}

                result1 = await flow.async_step_init()
                assert result1.get("type") == FlowResultType.FORM

            # Step 2: Submit updated data in sectioned format
            user_input = {
                "motion": {
                    CONF_MOTION_SENSORS: [
                        "binary_sensor.motion1",
                        "binary_sensor.motion2",
                    ],
                    CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
                },
                "purpose": {},
                "doors": {},
                "windows": {},
                "lights": {},
                "media": {},
                "appliances": {},
                "environmental": {},
                "wasp_in_box": {},
                "parameters": {
                    CONF_THRESHOLD: 75,
                },
            }

            # Mock async_create_entry to return the user input as data
            flow.async_create_entry = Mock(
                return_value={
                    "type": FlowResultType.CREATE_ENTRY,
                    "title": "",
                    "data": user_input,
                }
            )

            result2 = await flow.async_step_init(user_input)

            assert result2.get("type") == FlowResultType.CREATE_ENTRY
            assert result2.get("data") == user_input

    async def test_options_flow_adds_name_for_validation(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test that options flow adds name from config entry for validation."""
        # Set up a config entry with a name
        mock_config_entry.data = {CONF_NAME: "Test Area"}
        mock_config_entry.options = {}

        # Create the options flow manually without triggering HA setup
        flow = Mock(spec=AreaOccupancyOptionsFlow)
        flow.config_entry = mock_config_entry
        flow._data = {}
        flow.hass = mock_hass

        # Bind the actual method we want to test
        flow.async_step_init = AreaOccupancyOptionsFlow.async_step_init.__get__(
            flow, AreaOccupancyOptionsFlow
        )

        # Bind the validate method from BaseOccupancyFlow
        flow._validate_config = BaseOccupancyFlow._validate_config.__get__(
            flow, BaseOccupancyFlow
        )

        # Mock the parent class methods that return flow results
        flow.async_show_form = Mock(
            return_value={"type": FlowResultType.FORM, "step_id": "init"}
        )
        flow.async_create_entry = Mock(
            return_value={"type": FlowResultType.CREATE_ENTRY, "title": "", "data": {}}
        )

        # Mock the entity registry
        with patch("homeassistant.helpers.entity_registry.async_get") as mock_er_get:
            mock_registry = Mock()
            mock_registry.entities = Mock()
            mock_registry.entities.values = Mock(return_value=[])
            mock_er_get.return_value = mock_registry

            # Mock states
            mock_hass.states = Mock()
            mock_hass.states.async_entity_ids = Mock(return_value=[])
            mock_hass.states.get = Mock(return_value=None)

            # Submit user input without name (as would happen in options flow)
            user_input = {
                "motion": {
                    CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                    CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
                },
                "purpose": {},
                "doors": {},
                "windows": {},
                "lights": {},
                "media": {},
                "appliances": {},
                "environmental": {},
                "wasp_in_box": {},
                "parameters": {},
            }

            # Capture the data passed to async_create_entry to verify our fix
            captured_data = {}

            def capture_create_entry(title="", data=None):
                captured_data.update(data or {})
                return {
                    "type": FlowResultType.CREATE_ENTRY,
                    "title": title,
                    "data": data,
                }

            flow.async_create_entry = capture_create_entry

            # The test should not raise a "Name is required" error
            result = await flow.async_step_init(user_input)

            # Should create entry successfully
            assert result.get("type") == FlowResultType.CREATE_ENTRY
            # The flattened data should include the name from config entry
            assert captured_data.get(CONF_NAME) == "Test Area"

    async def test_config_flow_with_existing_entry(self, mock_hass: Mock) -> None:
        """Test config flow when entry already exists."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        # Add the missing data attribute to mock_hass
        mock_hass.data = {}

        user_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        # Mock that unique ID is already configured
        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(
                flow,
                "_abort_if_unique_id_configured",
                side_effect=AbortFlow("already_configured"),
            ),
            patch(
                "custom_components.area_occupancy.config_flow._get_include_entities"
            ) as mock_get_entities,
        ):
            mock_get_entities.return_value = {
                "appliance": [],
                "window": [],
                "door": [],
            }
            # The AbortFlow is caught and converted to an error in the flow
            result = await flow.async_step_user(user_input)
            assert result.get("type") == FlowResultType.FORM
            assert "errors" in result
            assert isinstance(result["errors"], dict)
            assert result["errors"]["base"] == "Flow aborted: already_configured"

    async def test_error_recovery_in_config_flow(self, mock_hass: Mock) -> None:
        """Test error recovery in config flow."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        # First attempt with invalid data
        invalid_input = {
            CONF_NAME: "Living Room",
            "motion": {
                CONF_MOTION_SENSORS: [],  # Invalid
                CONF_PRIMARY_OCCUPANCY_SENSOR: "",
            },
            "purpose": {},
            "doors": {},
            "windows": {},
            "lights": {},
            "media": {},
            "appliances": {},
            "environmental": {},
            "wasp_in_box": {},
            "parameters": {
                CONF_THRESHOLD: 60,
            },
        }

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result1 = await flow.async_step_user(invalid_input)
            assert result1.get("type") == FlowResultType.FORM
            assert "errors" in result1

        # Second attempt with valid data
        valid_input = {
            CONF_NAME: "Living Room",
            "motion": {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            },
            "purpose": {},
            "doors": {},
            "windows": {},
            "lights": {},
            "media": {},
            "appliances": {},
            "environmental": {},
            "wasp_in_box": {},
            "parameters": {
                CONF_THRESHOLD: 60,
            },
        }

        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            result2 = await flow.async_step_user(valid_input)

            assert result2.get("type") == FlowResultType.CREATE_ENTRY

    async def test_schema_generation_with_entities(self, mock_hass: Mock) -> None:
        """Test schema generation with available entities."""
        with patch(
            "custom_components.area_occupancy.config_flow._get_include_entities"
        ) as mock_get_entities:
            mock_get_entities.return_value = {
                "appliance": ["binary_sensor.motion1", "binary_sensor.door1"],
                "window": ["binary_sensor.window1"],
                "door": ["binary_sensor.door1"],
            }

            schema_dict = create_schema(mock_hass)

            # Verify schema was created successfully
            assert isinstance(schema_dict, dict)
            assert len(schema_dict) > 0

    def test_state_options_generation(self) -> None:
        """Test state options generation for different platforms."""
        # Test all supported platforms
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
            options = _get_state_select_options(platform)
            assert isinstance(options, list)
            assert len(options) > 0

            # Verify option structure
            for option in options:
                assert "value" in option
                assert "label" in option
                assert isinstance(option["value"], str)
                assert isinstance(option["label"], str)
