"""Test the purpose data module."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from custom_components.area_occupancy.data.purpose import (
    AreaPurpose,
    Purpose,
    PurposeManager,
    PURPOSE_DEFINITIONS,
    get_purpose_options,
)


class TestAreaPurpose:
    """Test AreaPurpose enum."""

    def test_area_purpose_values(self):
        """Test that AreaPurpose enum has correct values."""
        assert AreaPurpose.PASSAGEWAY == "passageway"
        assert AreaPurpose.UTILITY == "utility"
        assert AreaPurpose.FOOD_PREP == "food_prep"
        assert AreaPurpose.EATING == "eating"
        assert AreaPurpose.WORKING == "working"
        assert AreaPurpose.SOCIAL == "social"
        assert AreaPurpose.RELAXING == "relaxing"
        assert AreaPurpose.SLEEPING == "sleeping"


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

    def test_purpose_to_dict(self):
        """Test converting Purpose to dictionary."""
        purpose = Purpose(
            purpose=AreaPurpose.WORKING,
            name="Working / Studying",
            description="Home office",
            half_life=600.0,
        )
        result = purpose.to_dict()
        expected = {
            "purpose": "working",
            "name": "Working / Studying",
            "description": "Home office",
            "half_life": 600.0,
        }
        assert result == expected

    def test_purpose_from_dict(self):
        """Test creating Purpose from dictionary."""
        data = {
            "purpose": "sleeping",
            "name": "Sleeping / Resting",
            "description": "Bedroom",
            "half_life": 1800.0,
        }
        purpose = Purpose.from_dict(data)
        assert purpose.purpose == AreaPurpose.SLEEPING
        assert purpose.name == "Sleeping / Resting"
        assert purpose.description == "Bedroom"
        assert purpose.half_life == 1800.0


class TestPurposeDefinitions:
    """Test purpose definitions."""

    def test_all_purposes_defined(self):
        """Test that all purposes have definitions."""
        for purpose_type in AreaPurpose:
            assert purpose_type in PURPOSE_DEFINITIONS

    def test_purpose_half_lives(self):
        """Test that purpose half-lives match the expected values."""
        assert PURPOSE_DEFINITIONS[AreaPurpose.PASSAGEWAY].half_life == 45.0
        assert PURPOSE_DEFINITIONS[AreaPurpose.UTILITY].half_life == 90.0
        assert PURPOSE_DEFINITIONS[AreaPurpose.FOOD_PREP].half_life == 240.0
        assert PURPOSE_DEFINITIONS[AreaPurpose.EATING].half_life == 450.0
        assert PURPOSE_DEFINITIONS[AreaPurpose.WORKING].half_life == 600.0
        assert PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life == 720.0
        assert PURPOSE_DEFINITIONS[AreaPurpose.RELAXING].half_life == 900.0
        assert PURPOSE_DEFINITIONS[AreaPurpose.SLEEPING].half_life == 1800.0

    def test_get_purpose_options(self):
        """Test getting purpose options for UI."""
        options = get_purpose_options()
        assert len(options) == 8
        assert all("value" in option and "label" in option for option in options)
        
        # Check specific options
        social_option = next(opt for opt in options if opt["value"] == "social")
        assert social_option["label"] == "Social / Play (720s)"


class TestPurposeManager:
    """Test PurposeManager."""

    @pytest.fixture
    def mock_coordinator(self):
        """Create a mock coordinator."""
        coordinator = MagicMock()
        coordinator.config_manager.config.purpose = "social"
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
    async def test_async_initialize_with_invalid_purpose(self, mock_coordinator):
        """Test initialization with invalid purpose."""
        mock_coordinator.config_manager.config.purpose = "invalid"
        purpose_manager = PurposeManager(mock_coordinator)
        await purpose_manager.async_initialize()
        # Should fall back to social
        assert purpose_manager.current_purpose.purpose == AreaPurpose.SOCIAL

    @pytest.mark.asyncio
    async def test_async_initialize_with_no_purpose(self, mock_coordinator):
        """Test initialization with no purpose configured."""
        mock_coordinator.config_manager.config.purpose = None
        purpose_manager = PurposeManager(mock_coordinator)
        await purpose_manager.async_initialize()
        # Should default to social
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