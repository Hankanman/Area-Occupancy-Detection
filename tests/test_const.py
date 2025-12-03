"""Test constants for performance optimization features.

This module tests performance-related constants that are actively used in the codebase.
Tests focus on validating important relationships and constraints rather than
redundant type checks or exact value assertions.
"""

import pytest

from custom_components.area_occupancy.const import (
    DEFAULT_CACHE_TTL_SECONDS,
    DEFAULT_LOOKBACK_DAYS,
    RETENTION_DAYS,
)


class TestPerformanceConstants:
    """Test performance optimization constants.

    These constants control data retention and analysis periods for the
    occupancy detection system. Tests validate important relationships
    and constraints to ensure reasonable configuration.
    """

    @pytest.mark.parametrize(
        ("constant_value", "expected_value", "constant_name", "usage_context"),
        [
            (
                DEFAULT_LOOKBACK_DAYS,
                60,
                "DEFAULT_LOOKBACK_DAYS",
                "used in data/analysis.py, db/core.py - 2 months reasonable for analysis",
            ),
            (
                RETENTION_DAYS,
                365,
                "RETENTION_DAYS",
                "used in db/operations.py, db/sync.py - 1 year reasonable retention period",
            ),
            (
                DEFAULT_CACHE_TTL_SECONDS,
                3600,
                "DEFAULT_CACHE_TTL_SECONDS",
                "currently unused, kept for future caching implementation - 1 hour reasonable cache TTL",
            ),
        ],
    )
    def test_performance_constant_values(
        self,
        constant_value: int,
        expected_value: int,
        constant_name: str,
        usage_context: str,
    ) -> None:
        """Test that performance constants have valid types and expected values.

        Args:
            constant_value: The actual constant value to test
            expected_value: The expected value for the constant
            constant_name: Name of the constant for error messages
            usage_context: Description of where/how the constant is used
        """
        # Type validation: must be integer
        assert isinstance(constant_value, int), (
            f"{constant_name} must be an integer, got {type(constant_value).__name__}"
        )

        # Positive value validation: must be greater than zero
        assert constant_value > 0, (
            f"{constant_name} must be positive, got {constant_value}"
        )

        # Exact value validation: must match expected value
        assert constant_value == expected_value, (
            f"{constant_name} expected {expected_value}, got {constant_value} ({usage_context})"
        )

    def test_performance_constants_relationship(self) -> None:
        """Test critical relationship constraint between performance constants.

        The lookback period must be less than the retention period to ensure
        we don't try to analyze more data than we retain. This is a critical
        constraint that prevents data access errors.
        """
        assert DEFAULT_LOOKBACK_DAYS < RETENTION_DAYS, (
            f"Lookback period ({DEFAULT_LOOKBACK_DAYS} days) must be less than "
            f"retention period ({RETENTION_DAYS} days) to prevent analyzing more data than retained"
        )
