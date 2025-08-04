"""Tests for utils module."""

import math
from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.utils import (
    FileLock,
    bayesian_probability,
    format_float,
    format_percentage,
)


# ruff: noqa: SLF001, PLC0415
class TestUtils:
    """Test utility functions."""

    def test_format_float(self) -> None:
        """Test float formatting to 2 decimal places."""
        # Basic formatting
        assert format_float(1.234567) == 1.23
        assert format_float(1.0) == 1.0
        assert format_float(0.999) == 1.0
        assert format_float(0.001) == 0.0

        # Edge cases
        assert format_float(0.0) == 0.0
        assert format_float(-1.234567) == -1.23
        assert format_float(999.999) == 1000.0

        # String conversion (format_float can handle strings)
        assert format_float("1.234567") == 1.23
        assert format_float("0") == 0.0

    def test_format_percentage(self) -> None:
        """Test percentage formatting."""
        # Basic percentage formatting
        assert format_percentage(0.5) == "50.00%"
        assert format_percentage(0.123) == "12.30%"
        assert format_percentage(1.0) == "100.00%"
        assert format_percentage(0.0) == "0.00%"

        # Edge cases
        assert format_percentage(0.999) == "99.90%"
        assert format_percentage(0.001) == "0.10%"
        assert format_percentage(1.5) == "150.00%"
        assert format_percentage(-0.1) == "-10.00%"

    def test_format_float_edge_cases(self) -> None:
        """Test format_float with edge cases."""
        # Very large numbers
        assert format_float(1234567.89) == 1234567.89

        # Very small numbers
        assert format_float(0.0001) == 0.0

        # Infinity and NaN - these are handled by float() conversion
        import math

        # float('inf') and float('nan') are valid inputs
        assert format_float(float("inf")) == float("inf")
        assert math.isnan(format_float(float("nan")))

    def test_format_percentage_edge_cases(self) -> None:
        """Test format_percentage with edge cases."""
        # Very large percentages
        assert format_percentage(10.0) == "1000.00%"

        # Very small percentages
        assert format_percentage(0.0001) == "0.01%"

        # Negative percentages
        assert format_percentage(-0.5) == "-50.00%"

        # Infinity and NaN - these are converted to strings

        assert format_percentage(float("inf")) == "inf%"
        assert format_percentage(float("nan")) == "nan%"


class TestFileLock:
    """Test FileLock class."""

    def test_file_lock_creation(self, tmp_path):
        """Test FileLock creation and basic functionality."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path, timeout=5)

        assert lock.lock_path == lock_path
        assert lock.timeout == 5
        assert lock._lock_fd is None

    def test_file_lock_context_manager(self, tmp_path):
        """Test FileLock as context manager."""
        lock_path = tmp_path / "test.lock"
        lock = FileLock(lock_path, timeout=5)

        with lock:
            # Lock file should exist
            assert lock_path.exists()
            # Lock file descriptor should be set
            assert lock._lock_fd is not None
            # Should contain PID
            with open(lock_path) as f:
                content = f.read()
                assert content.isdigit()

        # After context exit, lock file should be removed
        assert not lock_path.exists()
        assert lock._lock_fd is None

    def test_file_lock_timeout(self, tmp_path):
        """Test FileLock timeout behavior."""
        lock_path = tmp_path / "test.lock"

        # Create a lock file manually to simulate existing lock
        with open(lock_path, "w") as f:
            f.write("12345")

        lock = FileLock(lock_path, timeout=0.1)  # Very short timeout

        with pytest.raises(TimeoutError), lock:
            pass


class TestBayesianProbability:
    """Test bayesian_probability function."""

    def test_basic_bayesian_calculation(self):
        """Test basic Bayesian probability calculation."""
        # Create mock entities
        entity1 = Mock()
        entity1.evidence = True
        entity1.decay.decay_factor = 1.0
        entity1.decay.is_decaying = False
        entity1.prob_given_true = 0.8
        entity1.prob_given_false = 0.1
        entity1.weight = 0.5

        entity2 = Mock()
        entity2.evidence = False
        entity2.decay.decay_factor = 1.0
        entity2.decay.is_decaying = False
        entity2.prob_given_true = 0.7
        entity2.prob_given_false = 0.2
        entity2.weight = 0.3

        entities = {"entity1": entity1, "entity2": entity2}

        # Test with default priors
        result = bayesian_probability(entities)
        assert 0.0 <= result <= 1.0

        # Test with custom priors
        result = bayesian_probability(entities, area_prior=0.3, time_prior=0.7)
        assert 0.0 <= result <= 1.0

    def test_bayesian_with_decay(self):
        """Test Bayesian probability with decaying entities."""
        entity = Mock()
        entity.evidence = False  # No current evidence
        entity.decay.decay_factor = 0.5  # Decaying
        entity.decay.is_decaying = True
        entity.prob_given_true = 0.8
        entity.prob_given_false = 0.1
        entity.weight = 0.5

        entities = {"entity1": entity}

        result = bayesian_probability(entities)
        assert 0.0 <= result <= 1.0

    def test_bayesian_edge_cases(self):
        """Test Bayesian probability with edge cases."""
        # Empty entities
        result = bayesian_probability({})
        assert 0.0 <= result <= 1.0

        # Extreme priors (should be clamped)
        entity = Mock()
        entity.evidence = True
        entity.decay.decay_factor = 1.0
        entity.decay.is_decaying = False
        entity.prob_given_true = 0.8
        entity.prob_given_false = 0.1
        entity.weight = 0.5

        entities = {"entity1": entity}

        # Test with extreme priors
        result = bayesian_probability(entities, area_prior=0.0, time_prior=0.0)
        assert 0.0 <= result <= 1.0

        result = bayesian_probability(entities, area_prior=1.0, time_prior=1.0)
        assert 0.0 <= result <= 1.0

    def test_bayesian_numerical_stability(self):
        """Test Bayesian probability numerical stability with many entities."""
        entities = {}

        # Create many entities with varying probabilities
        for i in range(10):
            entity = Mock()
            entity.evidence = i % 2 == 0  # Alternate evidence
            entity.decay.decay_factor = 1.0
            entity.decay.is_decaying = False
            entity.prob_given_true = 0.8
            entity.prob_given_false = 0.1
            entity.weight = 0.1

            entities[f"entity_{i}"] = entity

        result = bayesian_probability(entities)
        assert 0.0 <= result <= 1.0
        assert not (math.isnan(result) or math.isinf(result))
