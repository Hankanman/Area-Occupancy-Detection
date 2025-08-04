"""Test the purpose data module."""

from unittest.mock import MagicMock

import pytest

from custom_components.area_occupancy.data.purpose import (
    PURPOSE_DEFINITIONS,
    AreaPurpose,
    Purpose,
    PurposeManager,
    get_purpose_options,
)


# ruff: noqa: SLF001
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
    """Test Purpose dataclass."""

    def test_purpose_creation(self):
        """Test creating a Purpose instance."""
        purpose = Purpose(
            purpose=AreaPurpose.SOCIAL,
            name="Social / Play",
            description="Living room area",
            half_life=720.0,
        )
        assert purpose.purpose == AreaPurpose.SOCIAL
        assert purpose.name == "Social / Play"
        assert purpose.description == "Living room area"
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


class TestPurposeManager:
    """Test PurposeManager."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock coordinator."""
        coordinator = MagicMock()
        coordinator.config.purpose = "social"
        return coordinator

    @pytest.fixture
    def purpose_manager(self, mock_coordinator):
        """Create a PurposeManager instance."""
        return PurposeManager(mock_coordinator)

    def test_initialization(self, purpose_manager):
        """Test PurposeManager initialization."""
        assert purpose_manager._current_purpose is None

    @pytest.mark.asyncio
    async def test_async_initialize_with_valid_purpose(self, purpose_manager):
        """Test initialization with valid purpose."""
        await purpose_manager.async_initialize()
        assert purpose_manager.current_purpose.purpose == AreaPurpose.SOCIAL
        assert purpose_manager.half_life == 720.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("invalid_purpose", "expected_fallback"),
        [
            ("invalid", AreaPurpose.SOCIAL),
            (None, AreaPurpose.SOCIAL),
        ],
    )
    async def test_async_initialize_with_invalid_purpose(
        self, mock_coordinator, invalid_purpose, expected_fallback
    ):
        """Test initialization with invalid or missing purpose."""
        mock_coordinator.config.purpose = invalid_purpose
        purpose_manager = PurposeManager(mock_coordinator)
        await purpose_manager.async_initialize()
        # Should fall back to social
        assert purpose_manager.current_purpose.purpose == expected_fallback

    @pytest.mark.asyncio
    async def test_async_initialize_with_key_error_fallback(self, mock_coordinator):
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

            mock_coordinator.config.purpose = "working"
            purpose_manager = PurposeManager(mock_coordinator)
            await purpose_manager.async_initialize()
            # Should fall back to social since working is missing from definitions
            assert purpose_manager.current_purpose.purpose == AreaPurpose.SOCIAL

    def test_get_purpose(self, purpose_manager):
        """Test getting specific purpose."""
        purpose = purpose_manager.get_purpose(AreaPurpose.WORKING)
        assert purpose.purpose == AreaPurpose.WORKING
        assert purpose.half_life == 600.0

    def test_get_all_purposes(self, purpose_manager):
        """Test getting all purposes."""
        purposes = purpose_manager.get_all_purposes()
        assert len(purposes) == 8
        assert AreaPurpose.SOCIAL in purposes

    def test_set_purpose(self, purpose_manager):
        """Test setting purpose."""
        purpose_manager.set_purpose(AreaPurpose.SLEEPING)
        assert purpose_manager.current_purpose.purpose == AreaPurpose.SLEEPING
        assert purpose_manager.half_life == 1800.0

    def test_cleanup(self, purpose_manager):
        """Test cleanup."""
        purpose_manager._current_purpose = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL]
        purpose_manager.cleanup()
        assert purpose_manager._current_purpose is None
