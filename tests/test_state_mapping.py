"""Tests for state_mapping module."""

import pytest

from custom_components.area_occupancy.const import (
    StateOption,
    get_default_state,
    get_friendly_state_name,
    get_state_icon,
    get_state_options,
)

# Constants for test data
KNOWN_PLATFORMS = ["door", "window", "media", "appliance"]
UNKNOWN_PLATFORM = "unknown_platform"


class TestStateOption:
    """Test StateOption dataclass."""

    def test_initialization(self) -> None:
        """Test StateOption initialization."""
        option = StateOption(value="on", name="On")

        assert option.value == "on"
        assert option.name == "On"
        assert option.icon is None

    def test_initialization_with_icon(self) -> None:
        """Test StateOption initialization with icon."""
        option = StateOption(value="on", name="On", icon="mdi:power")

        assert option.value == "on"
        assert option.name == "On"
        assert option.icon == "mdi:power"


class TestGetStateOptions:
    """Test get_state_options function."""

    @pytest.mark.parametrize("platform", KNOWN_PLATFORMS)
    def test_known_platforms(self, platform: str) -> None:
        """Test getting state options for known platforms."""
        result = get_state_options(platform)

        assert "options" in result
        assert "default" in result
        assert isinstance(result["options"], list)
        assert len(result["options"]) > 0

        # Check that all options are StateOption objects
        for option in result["options"]:
            assert hasattr(option, "value")
            assert hasattr(option, "name")

        # Verify default is in options
        option_values = [opt.value for opt in result["options"]]
        assert result["default"] in option_values

    def test_unknown_platform(self) -> None:
        """Test getting state options for unknown platform."""
        result = get_state_options(UNKNOWN_PLATFORM)

        # Should return motion states as default fallback
        assert "options" in result
        assert "default" in result
        assert len(result["options"]) == 2  # on/off states
        assert result["default"] == "on"


class TestGetFriendlyStateName:
    """Test get_friendly_state_name function."""

    @pytest.mark.parametrize(
        ("platform", "state"),
        [
            ("door", "closed"),
            ("media", "playing"),
        ],
    )
    def test_known_states(self, platform: str, state: str) -> None:
        """Test getting friendly names for known states."""
        name = get_friendly_state_name(platform, state)
        assert isinstance(name, str)
        assert len(name) > 0

    @pytest.mark.parametrize(
        ("platform", "state"),
        [
            ("door", "unknown_state"),
            (UNKNOWN_PLATFORM, "some_state"),
        ],
    )
    def test_unknown_states_return_original(self, platform: str, state: str) -> None:
        """Test that unknown states return the original state name."""
        name = get_friendly_state_name(platform, state)
        assert name == state

    def test_case_sensitivity(self) -> None:
        """Test that state name lookup is case sensitive."""
        name1 = get_friendly_state_name("door", "closed")
        name2 = get_friendly_state_name("door", "CLOSED")

        # If the mapping is case sensitive, these should be different
        if name1 != "CLOSED":
            assert name2 == "CLOSED"


class TestGetStateIcon:
    """Test get_state_icon function."""

    @pytest.mark.parametrize(
        ("platform", "state"),
        [
            ("door", "closed"),
            ("media", "playing"),
        ],
    )
    def test_known_states(self, platform: str, state: str) -> None:
        """Test getting icons for known states."""
        icon = get_state_icon(platform, state)
        # Icon might be None or a string, both are valid
        assert icon is None or isinstance(icon, str)

    @pytest.mark.parametrize(
        ("platform", "state"),
        [
            ("door", "unknown_state"),
            (UNKNOWN_PLATFORM, "some_state"),
        ],
    )
    def test_unknown_states_return_none(self, platform: str, state: str) -> None:
        """Test that unknown states return None for icons."""
        icon = get_state_icon(platform, state)
        assert icon is None

    def test_icon_format(self) -> None:
        """Test that returned icons have correct format."""
        for platform in KNOWN_PLATFORMS:
            options = get_state_options(platform)
            for option in options["options"]:
                if option.icon is not None:
                    assert isinstance(option.icon, str)
                    assert len(option.icon) > 0


class TestGetDefaultState:
    """Test get_default_state function."""

    @pytest.mark.parametrize("platform", KNOWN_PLATFORMS)
    def test_known_platforms(self, platform: str) -> None:
        """Test getting default state for known platforms."""
        default = get_default_state(platform)
        assert isinstance(default, str)
        assert len(default) > 0

        # Verify consistency with get_state_options
        options = get_state_options(platform)
        assert default == options["default"]
        assert default in [opt.value for opt in options["options"]]

    def test_unknown_platform_default(self) -> None:
        """Test getting default state for unknown platform."""
        default = get_default_state(UNKNOWN_PLATFORM)
        assert default == "on"  # Should return motion default


class TestStateMapping:
    """Test overall state mapping functionality."""

    def test_mapping_completeness(self) -> None:
        """Test that all platforms have complete mappings."""
        for platform in KNOWN_PLATFORMS:
            options = get_state_options(platform)
            assert len(options["options"]) > 0

            # Test each option
            for option in options["options"]:
                # Test friendly name lookup
                name = get_friendly_state_name(platform, option.value)
                assert isinstance(name, str)
                assert len(name) > 0

                # Test icon lookup (can be None)
                icon = get_state_icon(platform, option.value)
                assert icon is None or isinstance(icon, str)

            # Test default
            default = get_default_state(platform)
            assert isinstance(default, str)
            assert len(default) > 0

    def test_state_option_uniqueness(self) -> None:
        """Test that state option values are unique within each platform."""
        for platform in KNOWN_PLATFORMS:
            options = get_state_options(platform)
            values = [opt.value for opt in options["options"]]

            # Check for duplicates
            assert len(values) == len(set(values)), (
                f"Duplicate values found in {platform} options"
            )

    def test_state_option_names_exist(self) -> None:
        """Test that all state options have non-empty names."""
        for platform in KNOWN_PLATFORMS:
            options = get_state_options(platform)

            for option in options["options"]:
                assert option.name is not None
                assert len(option.name.strip()) > 0

    def test_return_type_consistency(self) -> None:
        """Test that return types are consistent across functions."""
        for platform in KNOWN_PLATFORMS:
            # get_state_options should return dict with specific structure
            options = get_state_options(platform)
            assert isinstance(options, dict)
            assert "options" in options
            assert "default" in options
            assert isinstance(options["options"], list)
            assert isinstance(options["default"], str)

            # get_default_state should return string
            default = get_default_state(platform)
            assert isinstance(default, str)

            # Test with a valid state
            if options["options"]:
                test_state = options["options"][0].value

                # get_friendly_state_name should return string
                name = get_friendly_state_name(platform, test_state)
                assert isinstance(name, str)

                # get_state_icon should return string or None
                icon = get_state_icon(platform, test_state)
                assert icon is None or isinstance(icon, str)
