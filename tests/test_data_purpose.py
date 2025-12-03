"""Test the purpose data module."""

import pytest

from custom_components.area_occupancy.data.purpose import (
    PURPOSE_DEFINITIONS,
    AreaPurpose,
    Purpose,
    get_default_decay_half_life,
    get_purpose_options,
)

# Shared constant for all purpose enums
ALL_PURPOSES = list(AreaPurpose)


class TestPurpose:
    """Test Purpose class."""

    @pytest.mark.parametrize("purpose_enum", ALL_PURPOSES)
    def test_purpose_creation_from_enum(self, purpose_enum):
        """Test creating a Purpose instance from enum."""
        purpose = Purpose(purpose=purpose_enum)
        definition = PURPOSE_DEFINITIONS[purpose_enum]
        assert purpose.purpose == purpose_enum
        assert purpose.name == definition.name
        assert purpose.description == definition.description
        assert purpose.half_life == definition.half_life

    @pytest.mark.parametrize(
        ("purpose_string", "expected_enum"),
        [(purpose.value, purpose) for purpose in ALL_PURPOSES],
    )
    def test_purpose_creation_from_string(self, purpose_string, expected_enum):
        """Test creating a Purpose instance from string."""
        purpose = Purpose(purpose=purpose_string)
        definition = PURPOSE_DEFINITIONS[expected_enum]
        assert purpose.purpose == expected_enum
        assert purpose.name == definition.name
        assert purpose.description == definition.description
        assert purpose.half_life == definition.half_life

    @pytest.mark.parametrize("purpose_enum", ALL_PURPOSES)
    def test_get_purpose(self, purpose_enum):
        """Test getting specific purpose."""
        purpose = Purpose.get_purpose(purpose_enum)
        definition = PURPOSE_DEFINITIONS[purpose_enum]
        assert purpose.purpose == purpose_enum
        assert purpose.name == definition.name
        assert purpose.description == definition.description
        assert purpose.half_life == definition.half_life


class TestPurposeDefinitions:
    """Test purpose definitions."""

    def test_all_purposes_defined(self):
        """Test that all purposes have definitions."""
        for purpose_type in AreaPurpose:
            assert purpose_type in PURPOSE_DEFINITIONS

    @pytest.mark.parametrize("purpose_enum", ALL_PURPOSES)
    def test_purpose_half_lives(self, purpose_enum):
        """Test that purpose half-lives are defined and positive."""
        half_life = PURPOSE_DEFINITIONS[purpose_enum].half_life
        assert half_life > 0, f"Half-life for {purpose_enum} must be positive"
        assert isinstance(half_life, float), (
            f"Half-life for {purpose_enum} must be a float"
        )

    def test_get_purpose_options(self):
        """Test getting purpose options for UI."""
        options = get_purpose_options()
        assert len(options) == 9

        # Verify structure
        assert all("value" in option and "label" in option for option in options)

        # Verify all purposes are present and correct
        options_dict = {opt["value"]: opt["label"] for opt in options}
        for purpose_enum, definition in PURPOSE_DEFINITIONS.items():
            assert purpose_enum.value in options_dict
            assert options_dict[purpose_enum.value] == definition.name


class TestPurposeInitialization:
    """Test Purpose initialization."""

    def test_initialization_with_none_defaults_to_social(self):
        """Test Purpose initialization with None defaults to SOCIAL."""
        purpose = Purpose(purpose=None)
        definition = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL]
        assert purpose.purpose == AreaPurpose.SOCIAL
        assert purpose.name == definition.name
        assert purpose.description == definition.description
        assert purpose.half_life == definition.half_life

    @pytest.mark.parametrize(
        ("invalid_purpose", "expected_fallback"),
        [
            ("invalid", AreaPurpose.SOCIAL),
            ("", AreaPurpose.SOCIAL),
        ],
    )
    def test_initialization_with_invalid_purpose(
        self, invalid_purpose, expected_fallback
    ):
        """Test initialization with invalid purpose falls back to SOCIAL."""
        purpose = Purpose(purpose=invalid_purpose)
        definition = PURPOSE_DEFINITIONS[expected_fallback]
        assert purpose.purpose == expected_fallback
        assert purpose.name == definition.name
        assert purpose.description == definition.description
        assert purpose.half_life == definition.half_life

    @pytest.mark.parametrize(
        ("purpose_input", "expected_enum"),
        [
            (AreaPurpose.SOCIAL, AreaPurpose.SOCIAL),
            ("social", AreaPurpose.SOCIAL),
        ],
    )
    def test_direct_creation_with_parameters(self, purpose_input, expected_enum):
        """Test Purpose creation with direct creation parameters (enum and string)."""
        purpose = Purpose(
            purpose=purpose_input,
            _name="Test Name",
            _description="Test Description",
            _half_life=123.45,
        )
        assert purpose.purpose == expected_enum
        assert purpose.name == "Test Name"
        assert purpose.description == "Test Description"
        assert purpose.half_life == 123.45

    def test_direct_creation_with_none_purpose_raises_error(self):
        """Test that direct creation with None purpose raises ValueError."""
        with pytest.raises(ValueError, match="purpose must be provided"):
            Purpose(
                purpose=None,
                _name="Test Name",
                _description="Test Description",
                _half_life=123.45,
            )


class TestGetDefaultDecayHalfLife:
    """Test get_default_decay_half_life function."""

    @pytest.mark.parametrize(
        ("purpose_string", "expected_enum"),
        [(purpose.value, purpose) for purpose in ALL_PURPOSES],
    )
    def test_get_default_decay_half_life_with_valid_purpose(
        self, purpose_string, expected_enum
    ):
        """Test get_default_decay_half_life with valid purpose strings."""
        half_life = get_default_decay_half_life(purpose_string)
        expected_half_life = PURPOSE_DEFINITIONS[expected_enum].half_life
        assert half_life == expected_half_life
        assert isinstance(half_life, float)
        assert half_life > 0

    def test_get_default_decay_half_life_with_none(self):
        """Test get_default_decay_half_life with None uses DEFAULT_PURPOSE."""
        half_life = get_default_decay_half_life(None)
        expected_half_life = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life
        assert half_life == expected_half_life

    @pytest.mark.parametrize("invalid_purpose", ["invalid", "", "nonexistent"])
    def test_get_default_decay_half_life_with_invalid_purpose(self, invalid_purpose):
        """Test get_default_decay_half_life with invalid purpose falls back to default."""
        half_life = get_default_decay_half_life(invalid_purpose)
        expected_half_life = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life
        assert half_life == expected_half_life
