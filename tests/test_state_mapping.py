"""Tests for state_mapping module."""


from custom_components.area_occupancy.state_mapping import (
    StateOption,
    get_default_state,
    get_friendly_state_name,
    get_state_icon,
    get_state_options,
)


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

    def test_door_states(self) -> None:
        """Test getting door state options."""
        result = get_state_options("door")

        assert "options" in result
        assert "default" in result
        assert isinstance(result["options"], list)
        assert len(result["options"]) > 0

        # Check that all options are StateOption objects
        for option in result["options"]:
            assert hasattr(option, "value")
            assert hasattr(option, "name")

    def test_window_states(self) -> None:
        """Test getting window state options."""
        result = get_state_options("window")

        assert "options" in result
        assert "default" in result
        assert isinstance(result["options"], list)
        assert len(result["options"]) > 0

    def test_media_states(self) -> None:
        """Test getting media state options."""
        result = get_state_options("media")

        assert "options" in result
        assert "default" in result
        assert isinstance(result["options"], list)
        assert len(result["options"]) > 0

    def test_appliance_states(self) -> None:
        """Test getting appliance state options."""
        result = get_state_options("appliance")

        assert "options" in result
        assert "default" in result
        assert isinstance(result["options"], list)
        assert len(result["options"]) > 0

    def test_unknown_platform(self) -> None:
        """Test getting state options for unknown platform."""
        result = get_state_options("unknown_platform")

        # Should return motion states as default fallback
        assert "options" in result
        assert "default" in result
        assert len(result["options"]) == 2  # on/off states
        assert result["default"] == "on"

    def test_all_known_platforms(self) -> None:
        """Test that all known platforms return valid state options."""
        known_platforms = ["door", "window", "media", "appliance"]

        for platform in known_platforms:
            result = get_state_options(platform)
            assert len(result["options"]) > 0
            assert result["default"] != ""

            # Verify default is in options
            option_values = [opt.value for opt in result["options"]]
            assert result["default"] in option_values


class TestGetFriendlyStateName:
    """Test get_friendly_state_name function."""

    def test_door_state_names(self) -> None:
        """Test getting friendly names for door states."""
        # Test known states
        name = get_friendly_state_name("door", "closed")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_media_state_names(self) -> None:
        """Test getting friendly names for media states."""
        name = get_friendly_state_name("media", "playing")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_unknown_state(self) -> None:
        """Test getting friendly name for unknown state."""
        name = get_friendly_state_name("door", "unknown_state")
        assert name == "unknown_state"  # Should return the original state

    def test_unknown_platform(self) -> None:
        """Test getting friendly name for unknown platform."""
        name = get_friendly_state_name("unknown_platform", "some_state")
        assert name == "some_state"  # Should return the original state

    def test_case_sensitivity(self) -> None:
        """Test that state name lookup is case sensitive."""
        # Test that exact case matching is required
        name1 = get_friendly_state_name("door", "closed")
        name2 = get_friendly_state_name("door", "CLOSED")

        # If the mapping is case sensitive, these should be different
        # (one should be the friendly name, the other the original)
        if name1 != "CLOSED":
            assert name2 == "CLOSED"


class TestGetStateIcon:
    """Test get_state_icon function."""

    def test_door_state_icons(self) -> None:
        """Test getting icons for door states."""
        icon = get_state_icon("door", "closed")
        # Icon might be None or a string, both are valid
        assert icon is None or isinstance(icon, str)

    def test_media_state_icons(self) -> None:
        """Test getting icons for media states."""
        icon = get_state_icon("media", "playing")
        assert icon is None or isinstance(icon, str)

    def test_unknown_state_icon(self) -> None:
        """Test getting icon for unknown state."""
        icon = get_state_icon("door", "unknown_state")
        assert icon is None

    def test_unknown_platform_icon(self) -> None:
        """Test getting icon for unknown platform."""
        icon = get_state_icon("unknown_platform", "some_state")
        assert icon is None

    def test_icon_format(self) -> None:
        """Test that returned icons have correct format."""
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
            options = get_state_options(platform)
            for option in options["options"]:
                if option.icon is not None:
                    # Icons should start with 'mdi:' typically
                    assert isinstance(option.icon, str)
                    assert len(option.icon) > 0


class TestGetDefaultState:
    """Test get_default_state function."""

    def test_door_default(self) -> None:
        """Test getting default state for door."""
        default = get_default_state("door")
        assert isinstance(default, str)
        assert len(default) > 0

    def test_window_default(self) -> None:
        """Test getting default state for window."""
        default = get_default_state("window")
        assert isinstance(default, str)
        assert len(default) > 0

    def test_media_default(self) -> None:
        """Test getting default state for media."""
        default = get_default_state("media")
        assert isinstance(default, str)
        assert len(default) > 0

    def test_appliance_default(self) -> None:
        """Test getting default state for appliance."""
        default = get_default_state("appliance")
        assert isinstance(default, str)
        assert len(default) > 0

    def test_unknown_platform_default(self) -> None:
        """Test getting default state for unknown platform."""
        default = get_default_state("unknown_platform")
        assert default == "on"  # Should return motion default

    def test_default_consistency(self) -> None:
        """Test that default states are consistent with options."""
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
            default = get_default_state(platform)
            options = get_state_options(platform)

            # Default should match the default from get_state_options
            assert default == options["default"]

            # Default should be in the list of option values
            option_values = [opt.value for opt in options["options"]]
            assert default in option_values


class TestStateMapping:
    """Test overall state mapping functionality."""

    def test_mapping_completeness(self) -> None:
        """Test that all platforms have complete mappings."""
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
            # Get options
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
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
            options = get_state_options(platform)
            values = [opt.value for opt in options["options"]]

            # Check for duplicates
            assert len(values) == len(set(values)), (
                f"Duplicate values found in {platform} options"
            )

    def test_state_option_names_exist(self) -> None:
        """Test that all state options have non-empty names."""
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
            options = get_state_options(platform)

            for option in options["options"]:
                assert option.name is not None
                assert len(option.name.strip()) > 0

    def test_return_type_consistency(self) -> None:
        """Test that return types are consistent across functions."""
        platforms = ["door", "window", "media", "appliance"]

        for platform in platforms:
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
