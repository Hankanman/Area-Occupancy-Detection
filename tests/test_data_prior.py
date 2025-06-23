"""Tests for prior module."""

from unittest.mock import Mock

from custom_components.area_occupancy.data.prior import Prior


class TestPrior:
    """Test Prior class."""

    def test_current_value_property(self, mock_coordinator: Mock) -> None:
        """Test current_value property."""
        prior = Prior(mock_coordinator)

        # Test with value explicitly set
        prior.value = 0.35
        assert prior.current_value == 0.35

        # Test with None value - should return default
        prior.value = None
        assert prior.current_value == 0.1713  # DEFAULT_PRIOR from const.py
