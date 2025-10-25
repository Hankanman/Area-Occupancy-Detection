"""Test constants for performance optimization features."""

from custom_components.area_occupancy.const import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    RETENTION_DAYS,
)


class TestPerformanceConstants:
    """Test performance optimization constants."""

    def test_performance_constants_defined(self):
        """Test that performance constants are properly defined."""
        # Test DEFAULT_LOOKBACK_DAYS
        assert DEFAULT_LOOKBACK_DAYS == 90
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

        # Lookback should be reasonable (3 months)
        assert DEFAULT_LOOKBACK_DAYS == 90  # 3 months

        # Retention should be reasonable (1 year)
        assert RETENTION_DAYS == 365  # 1 year

    def test_constants_importable(self):
        """Test that constants can be imported from const module."""

        # Should not raise ImportError
        assert DEFAULT_LOOKBACK_DAYS is not None
        assert DEFAULT_CACHE_TTL_SECONDS is not None
        assert RETENTION_DAYS is not None
