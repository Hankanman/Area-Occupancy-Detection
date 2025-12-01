"""Test constants for performance optimization features."""

import pytest

from custom_components.area_occupancy.const import (
    ALL_AREAS_IDENTIFIER,
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    RETENTION_DAYS,
    validate_and_sanitize_area_name,
)


class TestPerformanceConstants:
    """Test performance optimization constants."""

    def test_performance_constants_defined(self):
        """Test that performance constants are properly defined."""
        # Test DEFAULT_LOOKBACK_DAYS
        assert DEFAULT_LOOKBACK_DAYS == 60
        assert isinstance(DEFAULT_LOOKBACK_DAYS, int)
        assert DEFAULT_LOOKBACK_DAYS > 0

        # Test DEFAULT_CACHE_TTL_SECONDS
        assert DEFAULT_CACHE_TTL_SECONDS == 3600
        assert isinstance(DEFAULT_CACHE_TTL_SECONDS, int)
        assert DEFAULT_CACHE_TTL_SECONDS > 0

        # Test RETENTION_DAYS
        assert RETENTION_DAYS == 365
        assert isinstance(RETENTION_DAYS, int)
        assert RETENTION_DAYS > 0

    def test_constants_reasonable_values(self):
        """Test that constants have reasonable values."""
        # Lookback should be less than retention
        assert DEFAULT_LOOKBACK_DAYS < RETENTION_DAYS

        # Cache TTL should be reasonable (1 hour)
        assert DEFAULT_CACHE_TTL_SECONDS == 3600  # 1 hour

        # Lookback should be reasonable (2 months)
        assert DEFAULT_LOOKBACK_DAYS == 60  # 2 months

        # Retention should be reasonable (1 year)
        assert RETENTION_DAYS == 365  # 1 year

    def test_constants_importable(self):
        """Test that constants can be imported from const module."""

        # Should not raise ImportError
        assert DEFAULT_LOOKBACK_DAYS is not None
        assert DEFAULT_CACHE_TTL_SECONDS is not None
        assert RETENTION_DAYS is not None


class TestValidateAndSanitizeAreaName:
    """Test validate_and_sanitize_area_name function."""

    def test_valid_area_name(self):
        """Test that valid area names are accepted."""
        result = validate_and_sanitize_area_name("Living Room")
        assert result == "Living_Room"

    def test_empty_area_name(self):
        """Test that empty area names are rejected."""
        with pytest.raises(ValueError, match="Area name cannot be empty"):
            validate_and_sanitize_area_name("")
        with pytest.raises(ValueError, match="Area name cannot be empty"):
            validate_and_sanitize_area_name("   ")

    def test_exact_all_areas_identifier(self):
        """Test that exact ALL_AREAS_IDENTIFIER is rejected."""
        with pytest.raises(
            ValueError,
            match=f"Area name cannot be '{ALL_AREAS_IDENTIFIER}' as it conflicts with",
        ):
            validate_and_sanitize_area_name(ALL_AREAS_IDENTIFIER)

    def test_sanitized_all_areas_identifier(self):
        """Test that names that sanitize to ALL_AREAS_IDENTIFIER are rejected."""
        # "all areas" should sanitize to "all_areas"
        with pytest.raises(
            ValueError,
            match=f"Area name 'all areas' sanitizes to '{ALL_AREAS_IDENTIFIER}' which conflicts with",
        ):
            validate_and_sanitize_area_name("all areas")

        # "all  areas" with extra spaces should sanitize to "all_areas"
        with pytest.raises(
            ValueError,
            match=f"Area name 'all  areas' sanitizes to '{ALL_AREAS_IDENTIFIER}' which conflicts with",
        ):
            validate_and_sanitize_area_name("all  areas")

        # "all_areas" with underscores should match exactly (caught by first check)
        with pytest.raises(
            ValueError,
            match=f"Area name cannot be '{ALL_AREAS_IDENTIFIER}' as it conflicts with",
        ):
            validate_and_sanitize_area_name("all_areas")

    def test_sanitization_normalizes_whitespace(self):
        """Test that whitespace is normalized."""
        result = validate_and_sanitize_area_name("  Living  Room  ")
        assert result == "Living_Room"

    def test_sanitization_replaces_special_characters(self):
        """Test that special characters are replaced with underscores."""
        result = validate_and_sanitize_area_name("Living/Room")
        assert result == "Living_Room"

        result = validate_and_sanitize_area_name("Living@Room")
        assert result == "Living_Room"

    def test_sanitization_removes_leading_trailing_underscores(self):
        """Test that leading and trailing underscores are removed."""
        result = validate_and_sanitize_area_name("_Living_Room_")
        assert result == "Living_Room"

    def test_only_invalid_characters(self):
        """Test that names with only invalid characters are rejected."""
        with pytest.raises(
            ValueError, match="Area name contains only invalid characters"
        ):
            validate_and_sanitize_area_name("@@@")

    def test_sanitization_preserves_valid_names(self):
        """Test that valid names without special characters are preserved."""
        result = validate_and_sanitize_area_name("LivingRoom")
        assert result == "LivingRoom"

        result = validate_and_sanitize_area_name("Living_Room")
        assert result == "Living_Room"
