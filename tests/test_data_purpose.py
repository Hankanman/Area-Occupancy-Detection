"""Test the purpose data module."""

import pytest

from custom_components.area_occupancy.data.purpose import (
    PURPOSE_DEFINITIONS,
    AreaPurpose,
    Purpose,
    get_purpose_options,
)


class TestAreaPurpose:
    """Test AreaPurpose enum."""

    @pytest.mark.parametrize(
        ("purpose_enum", "expected_value"),
        [
            (AreaPurpose.PASSAGEWAY, "passageway"),
            (AreaPurpose.UTILITY, "utility"),
            (AreaPurpose.FOOD_PREP, "food_prep"),
            (AreaPurpose.EATING, "eating"),
            (AreaPurpose.WORKING, "working"),
            (AreaPurpose.SOCIAL, "social"),
            (AreaPurpose.RELAXING, "relaxing"),
            (AreaPurpose.SLEEPING, "sleeping"),
        ],
    )
    def test_area_purpose_values(self, purpose_enum, expected_value):
        """Test that AreaPurpose enum has correct values."""
        assert purpose_enum == expected_value


class TestPurpose:
    """Test Purpose class."""

    def test_purpose_creation_from_enum(self):
        """Test creating a Purpose instance from enum."""
        purpose = Purpose(purpose=AreaPurpose.SOCIAL)
        assert purpose.purpose == AreaPurpose.SOCIAL
        assert purpose.name == "Social"
        assert purpose.half_life == 720.0

    def test_purpose_creation_from_string(self):
        """Test creating a Purpose instance from string."""
        purpose = Purpose(purpose="social")
        assert purpose.purpose == AreaPurpose.SOCIAL
        assert purpose.name == "Social"
        assert purpose.half_life == 720.0


class TestPurposeDefinitions:
    """Test purpose definitions."""

    def test_all_purposes_defined(self):
        """Test that all purposes have definitions."""
        for purpose_type in AreaPurpose:
            assert purpose_type in PURPOSE_DEFINITIONS

    @pytest.mark.parametrize(
        ("purpose_enum", "expected_half_life"),
        [
            (AreaPurpose.PASSAGEWAY, 60.0),
            (AreaPurpose.UTILITY, 120.0),
            (AreaPurpose.FOOD_PREP, 300.0),
            (AreaPurpose.EATING, 600.0),
            (AreaPurpose.WORKING, 600.0),
            (AreaPurpose.SOCIAL, 720.0),
            (AreaPurpose.RELAXING, 900.0),
            (AreaPurpose.SLEEPING, 1800.0),
        ],
    )
    def test_purpose_half_lives(self, purpose_enum, expected_half_life):
        """Test that purpose half-lives match the expected values."""
        assert PURPOSE_DEFINITIONS[purpose_enum].half_life == expected_half_life

    def test_get_purpose_options(self):
        """Test getting purpose options for UI."""
        options = get_purpose_options()
        assert len(options) == 8
        assert all("value" in option and "label" in option for option in options)

        # Check specific options
        social_option = next(opt for opt in options if opt["value"] == "social")
        assert social_option["label"] == "Social"


class TestPurposeInitialization:
    """Test Purpose initialization."""

    def test_initialization_with_none_defaults_to_social(self):
        """Test Purpose initialization with None defaults to SOCIAL."""
        purpose = Purpose(purpose=None)
        assert purpose.purpose == AreaPurpose.SOCIAL
        assert purpose.half_life == 720.0

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
        assert purpose.purpose == expected_fallback

    def test_initialization_with_key_error_fallback(self):
        """Test initialization with purpose that causes KeyError."""
        # Mock the PURPOSE_DEFINITIONS to raise KeyError for a specific purpose
        with pytest.MonkeyPatch().context() as m:
            # Temporarily modify PURPOSE_DEFINITIONS to simulate missing key
            # but keep SOCIAL for the fallback
            original_definitions = PURPOSE_DEFINITIONS.copy()
            m.setattr(
                "custom_components.area_occupancy.data.purpose.PURPOSE_DEFINITIONS",
                {
                    k: v
                    for k, v in original_definitions.items()
                    if k != AreaPurpose.WORKING
                },
            )

            purpose = Purpose(purpose="working")
            # Should fall back to social since working is missing from definitions
            assert purpose.purpose == AreaPurpose.SOCIAL

    def test_get_purpose(self):
        """Test getting specific purpose."""
        purpose = Purpose.get_purpose(AreaPurpose.WORKING)
        assert purpose.purpose == AreaPurpose.WORKING
        assert purpose.half_life == 600.0

    def test_get_all_purposes(self):
        """Test getting all purposes."""
        purposes = Purpose.get_all_purposes()
        assert len(purposes) == 8
        assert AreaPurpose.SOCIAL in purposes

    def test_cleanup(self):
        """Test cleanup (no-op for compatibility)."""
        purpose = Purpose(purpose=AreaPurpose.SOCIAL)
        purpose.cleanup()  # Should not raise
