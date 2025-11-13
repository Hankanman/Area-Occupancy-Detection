"""Tests for the Area Occupancy Detection config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest
import voluptuous as vol

from custom_components.area_occupancy.config_flow import (
    AreaOccupancyConfigFlow,
    BaseOccupancyFlow,
    _get_include_entities,
    _get_state_select_options,
    create_schema,
)
from custom_components.area_occupancy.const import (
    CONF_AREAS,
    CONF_DECAY_HALF_LIFE,
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_WINDOW,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_WINDOW,
)
from homeassistant.data_entry_flow import AbortFlow, FlowResultType


# ruff: noqa: SLF001
@pytest.mark.parametrize("expected_lingering_timers", [True])
class TestBaseOccupancyFlow:
    """Test BaseOccupancyFlow class."""

    @pytest.fixture
    def base_config(self):
        """Create a base valid configuration for testing."""
        return {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_WEIGHT_MOTION: DEFAULT_WEIGHT_MOTION,
            CONF_WEIGHT_MEDIA: DEFAULT_WEIGHT_MEDIA,
            CONF_WEIGHT_APPLIANCE: DEFAULT_WEIGHT_APPLIANCE,
            CONF_WEIGHT_DOOR: DEFAULT_WEIGHT_DOOR,
            CONF_WEIGHT_WINDOW: DEFAULT_WEIGHT_WINDOW,
            CONF_WEIGHT_ENVIRONMENTAL: DEFAULT_WEIGHT_ENVIRONMENTAL,
        }

    @pytest.fixture
    def flow(self):
        """Create a BaseOccupancyFlow instance."""
        return BaseOccupancyFlow()

    def test_validate_config_valid(self, flow, base_config):
        """Test validating a valid configuration."""
        flow._validate_config(base_config)  # Should not raise any exception

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
        self, flow, base_config, invalid_config, expected_error
    ):
        """Test various invalid configuration scenarios."""
        test_config = {**base_config, **invalid_config}

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

    def test_get_include_entities(self, mock_hass, mock_entity_registry):
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

    @pytest.fixture
    def mock_hass_for_schema(self, mock_hass):
        """Set up mock hass for schema tests."""
        mock_hass.states = Mock()
        mock_hass.states.async_entity_ids = Mock(return_value=[])
        mock_hass.states.get = Mock(
            return_value=Mock(attributes={"device_class": None})
        )
        return mock_hass

    @pytest.fixture
    def mock_entity_registry_for_schema(self):
        """Create mock entity registry for schema tests."""
        mock_registry = Mock()
        mock_registry.entities = {}
        return mock_registry

    def test_create_schema_defaults(
        self, mock_hass_for_schema, mock_entity_registry_for_schema
    ):
        """Test creating schema with defaults."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=mock_entity_registry_for_schema,
        ):
            schema = create_schema(mock_hass_for_schema)

        expected_sections = [
            CONF_NAME,
            "motion",
            "doors",
            "windows",
            "media",
            "appliances",
            "environmental",
            "wasp_in_box",
            "parameters",
        ]
        assert isinstance(schema, dict)
        for section in expected_sections:
            assert section in schema

    def test_create_schema_with_defaults(
        self, mock_hass_for_schema, mock_entity_registry_for_schema
    ):
        """Test creating schema with provided defaults."""
        defaults = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion_1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion_1",
        }

        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=mock_entity_registry_for_schema,
        ):
            schema_dict = create_schema(mock_hass_for_schema, defaults)
            schema = vol.Schema(schema_dict)

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

    def test_create_schema_options_mode(
        self, mock_hass_for_schema, mock_entity_registry_for_schema
    ):
        """Test creating schema in options mode."""
        with patch(
            "homeassistant.helpers.entity_registry.async_get",
            return_value=mock_entity_registry_for_schema,
        ):
            schema = create_schema(mock_hass_for_schema, is_options=True)

        assert isinstance(schema, dict)
        assert CONF_NAME not in schema  # Name should not be in options mode


class TestAreaOccupancyConfigFlow:
    """Test AreaOccupancyConfigFlow class."""

    @pytest.fixture
    def flow(self, mock_hass):
        """Create a config flow instance."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass
        return flow

    def test_initialization(self):
        """Test ConfigFlow initialization."""
        flow = AreaOccupancyConfigFlow()
        assert flow.VERSION == 1
        assert flow.MINOR_VERSION == 1

    async def test_async_step_user_no_input(self, flow):
        """Test async_step_user with no user input."""
        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}
            result = await flow.async_step_user()

            assert result.get("type") == FlowResultType.FORM
            assert result.get("step_id") == "user"

    async def test_async_step_user_with_valid_input(self, flow):
        """Test async_step_user with valid user input."""
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
            assert result.get("title") == "Area Occupancy Detection"

            expected_data = {
                CONF_AREAS: [
                    {
                        CONF_NAME: "Test Area",
                        CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                        CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
                        CONF_PURPOSE: "social",
                        CONF_THRESHOLD: 60,
                        CONF_DECAY_HALF_LIFE: 720.0,
                    }
                ]
            }
            assert result.get("data") == expected_data

    async def test_async_step_user_with_invalid_input(self, flow):
        """Test async_step_user with invalid user input."""
        user_input = {
            CONF_NAME: "Test Area",
            "motion": {
                CONF_MOTION_SENSORS: [],  # Invalid - empty
                CONF_PRIMARY_OCCUPANCY_SENSOR: "",
            },
            "purpose": {},
            "doors": {},
            "windows": {},
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

            assert result.get("type") == FlowResultType.FORM
            assert "errors" in result
            assert result["errors"] == {
                "base": "At least one motion sensor is required"
            }


class TestConfigFlowIntegration:
    """Test config flow integration scenarios."""

    @pytest.fixture
    def flow(self, mock_hass):
        """Create a config flow instance."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass
        return flow

    @pytest.fixture
    def valid_user_input(self):
        """Create valid user input for testing."""
        return {
            CONF_NAME: "Living Room",
            "motion": {
                CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
                CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            },
            "purpose": {},
            "doors": {},
            "windows": {},
            "media": {},
            "appliances": {},
            "environmental": {},
            "wasp_in_box": {},
            "parameters": {CONF_THRESHOLD: 60},
        }

    async def test_complete_config_flow(self, flow, valid_user_input):
        """Test complete configuration flow."""
        # Step 1: Show form
        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}
            result1 = await flow.async_step_user()
            assert result1.get("type") == FlowResultType.FORM

        # Step 2: Submit valid data
        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            result2 = await flow.async_step_user(valid_user_input)

            assert result2.get("type") == FlowResultType.CREATE_ENTRY
            assert result2.get("title") == "Area Occupancy Detection"

            result_data = result2.get("data", {})
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

    async def test_config_flow_with_existing_entry(self, flow, mock_hass):
        """Test config flow when entry already exists."""
        mock_hass.data = {}

        user_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

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
            mock_get_entities.return_value = {"appliance": [], "window": [], "door": []}
            result = await flow.async_step_user(user_input)

            assert result.get("type") == FlowResultType.FORM
            assert "errors" in result
            assert result["errors"]["base"] == "Flow aborted: already_configured"

    async def test_error_recovery_in_config_flow(self, flow):
        """Test error recovery in config flow."""
        # First attempt with invalid data
        invalid_input = {
            CONF_NAME: "Living Room",
            "motion": {CONF_MOTION_SENSORS: [], CONF_PRIMARY_OCCUPANCY_SENSOR: ""},
            "purpose": {},
            "doors": {},
            "windows": {},
            "media": {},
            "appliances": {},
            "environmental": {},
            "wasp_in_box": {},
            "parameters": {CONF_THRESHOLD: 60},
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
            "media": {},
            "appliances": {},
            "environmental": {},
            "wasp_in_box": {},
            "parameters": {CONF_THRESHOLD: 60},
        }

        with (
            patch.object(flow, "async_set_unique_id", new_callable=AsyncMock),
            patch.object(flow, "_abort_if_unique_id_configured"),
        ):
            result2 = await flow.async_step_user(valid_input)
            assert result2.get("type") == FlowResultType.CREATE_ENTRY

    async def test_schema_generation_with_entities(self, mock_hass):
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
            assert isinstance(schema_dict, dict)
            assert len(schema_dict) > 0

    def test_state_options_generation(self):
        """Test state options generation for different platforms."""
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
            options = _get_state_select_options(platform)
            assert isinstance(options, list)
            assert len(options) > 0

            for option in options:
                assert "value" in option
                assert "label" in option
                assert isinstance(option["value"], str)
                assert isinstance(option["label"], str)
