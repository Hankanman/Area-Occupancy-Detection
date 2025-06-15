"""Tests for config_flow module."""

from unittest.mock import Mock, patch

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
    CONF_MOTION_SENSORS,
    CONF_NAME,
    CONF_PRIMARY_OCCUPANCY_SENSOR,
    CONF_THRESHOLD,
)
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType


class TestBaseOccupancyFlow:
    """Test BaseOccupancyFlow class."""

    def test_validate_config_valid(self) -> None:
        """Test _validate_config with valid configuration."""
        flow = BaseOccupancyFlow()

        valid_config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        # Should not raise any exception
        flow._validate_config(valid_config)

    def test_validate_config_no_motion_sensors(self) -> None:
        """Test _validate_config with no motion sensors."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: [],
            CONF_THRESHOLD: 60,
        }

        with pytest.raises(
            vol.Invalid, match="At least one motion sensor must be selected"
        ):
            flow._validate_config(invalid_config)

    def test_validate_config_primary_not_in_motion(self) -> None:
        """Test _validate_config with primary sensor not in motion sensors."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion2",
            CONF_THRESHOLD: 60,
        }

        with pytest.raises(
            vol.Invalid,
            match="Primary occupancy sensor must be one of the selected motion sensors",
        ):
            flow._validate_config(invalid_config)

    def test_validate_config_invalid_threshold_low(self) -> None:
        """Test _validate_config with threshold too low."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 0,
        }

        with pytest.raises(vol.Invalid, match="Threshold must be between 1 and 99"):
            flow._validate_config(invalid_config)

    def test_validate_config_invalid_threshold_high(self) -> None:
        """Test _validate_config with threshold too high."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 100,
        }

        with pytest.raises(vol.Invalid, match="Threshold must be between 1 and 99"):
            flow._validate_config(invalid_config)

    def test_validate_config_empty_name(self) -> None:
        """Test _validate_config with empty name."""
        flow = BaseOccupancyFlow()

        invalid_config = {
            CONF_NAME: "",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        with pytest.raises(vol.Invalid, match="Name cannot be empty"):
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
        assert unknown_options == []

    @patch("homeassistant.helpers.entity_registry.async_get")
    def test_get_include_entities(self, mock_entity_registry: Mock) -> None:
        """Test _get_include_entities function."""
        mock_hass = Mock(spec=HomeAssistant)

        # Mock entity registry
        mock_registry = Mock()
        mock_registry.entities = {
            "binary_sensor.motion1": Mock(
                entity_id="binary_sensor.motion1", platform="template"
            ),
            "binary_sensor.motion2": Mock(
                entity_id="binary_sensor.motion2", platform="zha"
            ),
            "light.test_light": Mock(entity_id="light.test_light", platform="hue"),
            "media_player.tv": Mock(entity_id="media_player.tv", platform="cast"),
            "sensor.temperature": Mock(
                entity_id="sensor.temperature", platform="sensor"
            ),
        }
        mock_entity_registry.return_value = mock_registry

        # Mock states
        mock_hass.states.async_all.return_value = [
            Mock(entity_id="binary_sensor.motion1", domain="binary_sensor"),
            Mock(entity_id="binary_sensor.motion2", domain="binary_sensor"),
            Mock(entity_id="light.test_light", domain="light"),
            Mock(entity_id="media_player.tv", domain="media_player"),
            Mock(entity_id="sensor.temperature", domain="sensor"),
        ]

        result = _get_include_entities(mock_hass)

        # Check that entities are categorized correctly
        assert "binary_sensor" in result
        assert "light" in result
        assert "media_player" in result
        assert "sensor" in result

        assert "binary_sensor.motion1" in result["binary_sensor"]
        assert "binary_sensor.motion2" in result["binary_sensor"]
        assert "light.test_light" in result["light"]
        assert "media_player.tv" in result["media_player"]
        assert "sensor.temperature" in result["sensor"]

    def test_create_schema_defaults(self) -> None:
        """Test create_schema with default values."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_hass.states.async_all.return_value = []

        with patch(
            "custom_components.area_occupancy.config_flow._get_include_entities"
        ) as mock_get_entities:
            mock_get_entities.return_value = {
                "binary_sensor": ["binary_sensor.motion1"],
                "light": ["light.test_light"],
            }

            schema_dict = create_schema(mock_hass)

            assert isinstance(schema_dict, dict)
            assert len(schema_dict) > 0

    def test_create_schema_with_defaults(self) -> None:
        """Test create_schema with provided defaults."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_hass.states.async_all.return_value = []

        defaults = {
            CONF_NAME: "Test Area",
            CONF_THRESHOLD: 75,
        }

        with patch(
            "custom_components.area_occupancy.config_flow._get_include_entities"
        ) as mock_get_entities:
            mock_get_entities.return_value = {
                "binary_sensor": ["binary_sensor.motion1"],
            }

            schema_dict = create_schema(mock_hass, defaults)

            assert isinstance(schema_dict, dict)

    def test_create_schema_options_mode(self) -> None:
        """Test create_schema in options mode."""
        mock_hass = Mock(spec=HomeAssistant)
        mock_hass.states.async_all.return_value = []

        with patch(
            "custom_components.area_occupancy.config_flow._get_include_entities"
        ) as mock_get_entities:
            mock_get_entities.return_value = {
                "binary_sensor": ["binary_sensor.motion1"],
            }

            schema_dict = create_schema(mock_hass, is_options=True)

            assert isinstance(schema_dict, dict)


class TestAreaOccupancyConfigFlow:
    """Test AreaOccupancyConfigFlow class."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.states.async_all.return_value = [
            Mock(entity_id="binary_sensor.motion1", domain="binary_sensor"),
            Mock(entity_id="binary_sensor.motion2", domain="binary_sensor"),
        ]
        return hass

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

            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "user"

    async def test_async_step_user_with_valid_input(self, mock_hass: Mock) -> None:
        """Test async_step_user with valid user input."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        user_input = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        with patch.object(flow, "_validate_config") as mock_validate:
            with patch.object(flow, "async_set_unique_id") as mock_set_unique_id:
                with patch.object(flow, "_abort_if_unique_id_configured") as mock_abort:
                    result = await flow.async_step_user(user_input)

                    mock_validate.assert_called_once_with(user_input)
                    mock_set_unique_id.assert_called_once()
                    mock_abort.assert_called_once()

                    assert result["type"] == FlowResultType.CREATE_ENTRY
                    assert result["title"] == "Test Area"
                    assert result["data"] == user_input

    async def test_async_step_user_with_invalid_input(self, mock_hass: Mock) -> None:
        """Test async_step_user with invalid user input."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        user_input = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: [],  # Invalid - empty
            CONF_THRESHOLD: 60,
        }

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result = await flow.async_step_user(user_input)

            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "user"
            assert "errors" in result

    async def test_async_step_user_validation_error(self, mock_hass: Mock) -> None:
        """Test async_step_user with validation error."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        user_input = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion2",  # Not in motion sensors
            CONF_THRESHOLD: 60,
        }

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result = await flow.async_step_user(user_input)

            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "user"
            assert "errors" in result

    async def test_async_step_user_unexpected_error(self, mock_hass: Mock) -> None:
        """Test async_step_user with unexpected error."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        user_input = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        with patch.object(
            flow, "_validate_config", side_effect=Exception("Unexpected error")
        ):
            with patch(
                "custom_components.area_occupancy.config_flow.create_schema"
            ) as mock_create_schema:
                mock_create_schema.return_value = {"test": vol.Required("test")}

                result = await flow.async_step_user(user_input)

                assert result["type"] == FlowResultType.FORM
                assert result["step_id"] == "user"
                assert "errors" in result
                assert result["errors"]["base"] == "unknown"

    def test_async_get_options_flow(self) -> None:
        """Test async_get_options_flow static method."""
        mock_config_entry = Mock(spec=ConfigEntry)

        options_flow = AreaOccupancyConfigFlow.async_get_options_flow(mock_config_entry)

        assert isinstance(options_flow, AreaOccupancyOptionsFlow)
        assert options_flow.config_entry == mock_config_entry


class TestAreaOccupancyOptionsFlow:
    """Test AreaOccupancyOptionsFlow class."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)
        hass.states.async_all.return_value = [
            Mock(entity_id="binary_sensor.motion1", domain="binary_sensor"),
            Mock(entity_id="binary_sensor.motion2", domain="binary_sensor"),
        ]
        return hass

    @pytest.fixture
    def mock_config_entry(self) -> Mock:
        """Create a mock config entry."""
        entry = Mock(spec=ConfigEntry)
        entry.data = {
            CONF_NAME: "Test Area",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }
        entry.options = {}
        return entry

    def test_initialization(self, mock_config_entry: Mock) -> None:
        """Test OptionsFlow initialization."""
        flow = AreaOccupancyOptionsFlow(mock_config_entry)

        assert flow.config_entry == mock_config_entry

    async def test_async_step_init_no_input(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_step_init with no user input."""
        flow = AreaOccupancyOptionsFlow(mock_config_entry)
        flow.hass = mock_hass

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result = await flow.async_step_init()

            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "init"

    async def test_async_step_init_with_valid_input(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_step_init with valid user input."""
        flow = AreaOccupancyOptionsFlow(mock_config_entry)
        flow.hass = mock_hass

        user_input = {
            CONF_THRESHOLD: 75,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }

        with patch.object(flow, "_validate_config") as mock_validate:
            result = await flow.async_step_init(user_input)

            mock_validate.assert_called_once()

            assert result["type"] == FlowResultType.CREATE_ENTRY
            assert result["title"] == ""
            assert result["data"] == user_input

    async def test_async_step_init_with_invalid_input(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_step_init with invalid user input."""
        flow = AreaOccupancyOptionsFlow(mock_config_entry)
        flow.hass = mock_hass

        user_input = {
            CONF_THRESHOLD: 150,  # Invalid - too high
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result = await flow.async_step_init(user_input)

            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "init"
            assert "errors" in result

    async def test_async_step_init_validation_error(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_step_init with validation error."""
        flow = AreaOccupancyOptionsFlow(mock_config_entry)
        flow.hass = mock_hass

        user_input = {
            CONF_THRESHOLD: 75,
            CONF_MOTION_SENSORS: [],  # Invalid - empty
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result = await flow.async_step_init(user_input)

            assert result["type"] == FlowResultType.FORM
            assert result["step_id"] == "init"
            assert "errors" in result

    async def test_async_step_init_unexpected_error(
        self, mock_hass: Mock, mock_config_entry: Mock
    ) -> None:
        """Test async_step_init with unexpected error."""
        flow = AreaOccupancyOptionsFlow(mock_config_entry)
        flow.hass = mock_hass

        user_input = {
            CONF_THRESHOLD: 75,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }

        with patch.object(
            flow, "_validate_config", side_effect=Exception("Unexpected error")
        ):
            with patch(
                "custom_components.area_occupancy.config_flow.create_schema"
            ) as mock_create_schema:
                mock_create_schema.return_value = {"test": vol.Required("test")}

                result = await flow.async_step_init(user_input)

                assert result["type"] == FlowResultType.FORM
                assert result["step_id"] == "init"
                assert "errors" in result
                assert result["errors"]["base"] == "unknown"


class TestConfigFlowIntegration:
    """Test config flow integration scenarios."""

    @pytest.fixture
    def mock_hass(self) -> Mock:
        """Create a comprehensive mock Home Assistant instance."""
        hass = Mock(spec=HomeAssistant)

        # Mock states with various entity types
        hass.states.async_all.return_value = [
            Mock(entity_id="binary_sensor.motion1", domain="binary_sensor"),
            Mock(entity_id="binary_sensor.motion2", domain="binary_sensor"),
            Mock(entity_id="binary_sensor.door1", domain="binary_sensor"),
            Mock(entity_id="light.test_light", domain="light"),
            Mock(entity_id="media_player.tv", domain="media_player"),
            Mock(entity_id="sensor.temperature", domain="sensor"),
        ]

        return hass

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
            assert result1["type"] == FlowResultType.FORM

        # Step 2: Submit valid data
        user_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 65,
        }

        with patch.object(flow, "async_set_unique_id") as mock_set_unique_id:
            with patch.object(flow, "_abort_if_unique_id_configured") as mock_abort:
                result2 = await flow.async_step_user(user_input)

                assert result2["type"] == FlowResultType.CREATE_ENTRY
                assert result2["title"] == "Living Room"
                assert result2["data"] == user_input

    async def test_complete_options_flow(self, mock_hass: Mock) -> None:
        """Test complete options flow."""
        mock_config_entry = Mock(spec=ConfigEntry)
        mock_config_entry.data = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }
        mock_config_entry.options = {}

        flow = AreaOccupancyOptionsFlow(mock_config_entry)
        flow.hass = mock_hass

        # Step 1: Show form with current values
        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result1 = await flow.async_step_init()
            assert result1["type"] == FlowResultType.FORM

        # Step 2: Submit updated data
        user_input = {
            CONF_THRESHOLD: 75,
            CONF_MOTION_SENSORS: ["binary_sensor.motion1", "binary_sensor.motion2"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
        }

        result2 = await flow.async_step_init(user_input)

        assert result2["type"] == FlowResultType.CREATE_ENTRY
        assert result2["data"] == user_input

    async def test_config_flow_with_existing_entry(self, mock_hass: Mock) -> None:
        """Test config flow when entry already exists."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        user_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        # Mock that unique ID is already configured
        with patch.object(flow, "async_set_unique_id") as mock_set_unique_id:
            with patch.object(
                flow,
                "_abort_if_unique_id_configured",
                side_effect=config_entries.AbortFlow("already_configured"),
            ):
                with pytest.raises(config_entries.AbortFlow):
                    await flow.async_step_user(user_input)

    async def test_error_recovery_in_config_flow(self, mock_hass: Mock) -> None:
        """Test error recovery in config flow."""
        flow = AreaOccupancyConfigFlow()
        flow.hass = mock_hass

        # First attempt with invalid data
        invalid_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: [],  # Invalid
            CONF_THRESHOLD: 60,
        }

        with patch(
            "custom_components.area_occupancy.config_flow.create_schema"
        ) as mock_create_schema:
            mock_create_schema.return_value = {"test": vol.Required("test")}

            result1 = await flow.async_step_user(invalid_input)
            assert result1["type"] == FlowResultType.FORM
            assert "errors" in result1

        # Second attempt with valid data
        valid_input = {
            CONF_NAME: "Living Room",
            CONF_MOTION_SENSORS: ["binary_sensor.motion1"],
            CONF_PRIMARY_OCCUPANCY_SENSOR: "binary_sensor.motion1",
            CONF_THRESHOLD: 60,
        }

        with patch.object(flow, "async_set_unique_id") as mock_set_unique_id:
            with patch.object(flow, "_abort_if_unique_id_configured") as mock_abort:
                result2 = await flow.async_step_user(valid_input)

                assert result2["type"] == FlowResultType.CREATE_ENTRY

    async def test_schema_generation_with_entities(self, mock_hass: Mock) -> None:
        """Test schema generation with available entities."""
        with patch(
            "custom_components.area_occupancy.config_flow._get_include_entities"
        ) as mock_get_entities:
            mock_get_entities.return_value = {
                "binary_sensor": ["binary_sensor.motion1", "binary_sensor.door1"],
                "light": ["light.test_light"],
                "media_player": ["media_player.tv"],
                "sensor": ["sensor.temperature"],
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
