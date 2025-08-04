"""Tests for utils module."""

from custom_components.area_occupancy.utils import format_float, format_percentage


class TestFormatFloat:
    """Test format_float function."""

    def test_formatting(self) -> None:
        """Test float formatting to 2 decimal places."""
        assert format_float(1.234567) == 1.23
        assert format_float(1.0) == 1.0
        assert format_float(0.999) == 1.0
        assert format_float(0.001) == 0.0


class TestFormatPercentage:
    """Test format_percentage function."""

    def test_percentage_formatting(self) -> None:
        """Test percentage formatting."""
        assert format_percentage(0.5) == "50.00%"
        assert format_percentage(0.123) == "12.30%"
        assert format_percentage(1.0) == "100.00%"
        assert format_percentage(0.0) == "0.00%"
