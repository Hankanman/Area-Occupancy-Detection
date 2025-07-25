"""Tests for data.likelihood module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from custom_components.area_occupancy.const import HA_RECORDER_DAYS
from custom_components.area_occupancy.data.likelihood import Likelihood
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
class TestLikelihood:
    """Test Likelihood class."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test Likelihood initialization."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.85,
        )

        assert likelihood.entity_id == "binary_sensor.motion"
        assert likelihood.active_states == ["on"]
        assert likelihood.default_prob_true == 0.8
        assert likelihood.default_prob_false == 0.1
        assert likelihood.weight == 0.85
        assert likelihood.coordinator == mock_coordinator
        assert likelihood.last_updated is None
        assert likelihood.active_ratio is None
        assert likelihood.inactive_ratio is None

    def test_prob_given_true_with_calculated_values(
        self, mock_coordinator: Mock
    ) -> None:
        """Test prob_given_true property with calculated values."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Test with no calculated values (should use defaults)
        assert likelihood.prob_given_true == 0.8

        # Test with calculated values
        likelihood.active_ratio = 0.6
        assert likelihood.prob_given_true == 0.6

    def test_prob_given_false_with_calculated_values(
        self, mock_coordinator: Mock
    ) -> None:
        """Test prob_given_false property with calculated values."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Test with no calculated values (should use defaults)
        assert likelihood.prob_given_false == 0.1

        # Test with calculated values
        likelihood.inactive_ratio = 0.05
        assert likelihood.prob_given_false == 0.05

    def test_raw_probability_properties(self, mock_coordinator: Mock) -> None:
        """Test raw probability properties."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Test with no calculated values (should return defaults)
        assert likelihood.prob_given_true_raw == 0.8
        assert likelihood.prob_given_false_raw == 0.1

        # Test with calculated values
        likelihood.active_ratio = 0.6
        likelihood.inactive_ratio = 0.05
        assert likelihood.prob_given_true_raw == 0.6
        assert likelihood.prob_given_false_raw == 0.05

    async def test_update_history_enabled_success(self, mock_coordinator: Mock) -> None:
        """Test update method when history is enabled and calculation succeeds."""

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Mock successful calculation
        with patch.object(
            likelihood, "calculate", new_callable=AsyncMock
        ) as mock_calculate:
            mock_calculate.return_value = (0.6, 0.05)

            prob_true, prob_false = await likelihood.update()

            mock_calculate.assert_called_once()
            assert likelihood.active_ratio == 0.6
            assert likelihood.inactive_ratio == 0.05
            assert likelihood.last_updated is not None
            assert prob_true == likelihood.prob_given_true
            assert prob_false == likelihood.prob_given_false

    async def test_update_history_enabled_exception(
        self, mock_coordinator: Mock
    ) -> None:
        """Test update method when calculation raises exception."""

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Mock calculation that raises exception
        with patch.object(
            likelihood, "calculate", new_callable=AsyncMock
        ) as mock_calculate:
            mock_calculate.side_effect = ValueError("Test error")

            prob_true, prob_false = await likelihood.update()

            # Should fall back to defaults
            assert likelihood.active_ratio == 0.8
            assert likelihood.inactive_ratio == 0.1
            assert likelihood.last_updated is not None

    async def test_update_recalculates_when_cache_stale(
        self, mock_coordinator: Mock
    ) -> None:
        """Test that update() recalculates when cache is stale."""

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Set stale cached values (older than cache TTL)
        likelihood.active_ratio = 0.6
        likelihood.inactive_ratio = 0.05
        likelihood.last_updated = dt_util.utcnow() - timedelta(hours=3)  # Stale

        # Mock successful calculation
        with patch.object(
            likelihood, "calculate", new_callable=AsyncMock
        ) as mock_calculate:
            mock_calculate.return_value = (0.7, 0.02)

            prob_true, prob_false = await likelihood.update()

            # Should call calculate since cache is stale
            mock_calculate.assert_called_once()

            # Should update stored values
            assert likelihood.active_ratio == 0.7
            assert likelihood.inactive_ratio == 0.02
            assert likelihood.last_updated is not None

    async def test_update_recalculates_when_no_cache(
        self, mock_coordinator: Mock
    ) -> None:
        """Test that update() recalculates when no cached values exist."""

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # No cached values (fresh instance)
        assert likelihood.active_ratio is None
        assert likelihood.inactive_ratio is None
        assert likelihood.last_updated is None

        # Mock successful calculation
        with patch.object(
            likelihood, "calculate", new_callable=AsyncMock
        ) as mock_calculate:
            mock_calculate.return_value = (0.75, 0.03)

            prob_true, prob_false = await likelihood.update()

            # Should call calculate since no cache exists
            mock_calculate.assert_called_once()

            # Should store calculated values
            assert likelihood.active_ratio == 0.75
            assert likelihood.inactive_ratio == 0.03
            assert likelihood.last_updated is not None

    def test_is_cache_valid_no_values(self, mock_coordinator: Mock) -> None:
        """Test _is_cache_valid() returns False when no values exist."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Fresh instance with no cached values
        assert not likelihood._is_cache_valid()

    def test_is_cache_valid_fresh_values(self, mock_coordinator: Mock) -> None:
        """Test _is_cache_valid() returns True for fresh values."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Set fresh cached values
        likelihood.active_ratio = 0.6
        likelihood.inactive_ratio = 0.05
        likelihood.last_updated = dt_util.utcnow()

        assert likelihood._is_cache_valid()

    def test_is_cache_valid_stale_values(self, mock_coordinator: Mock) -> None:
        """Test _is_cache_valid() returns False for stale values."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Set stale cached values (older than cache TTL)
        likelihood.active_ratio = 0.6
        likelihood.inactive_ratio = 0.05
        likelihood.last_updated = dt_util.utcnow() - timedelta(hours=3)

        assert not likelihood._is_cache_valid()

    def test_to_dict(self, mock_coordinator: Mock) -> None:
        """Test converting likelihood to dictionary."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Set some calculated values
        likelihood.active_ratio = 0.6
        likelihood.inactive_ratio = 0.05
        likelihood.last_updated = dt_util.utcnow()

        data = likelihood.to_dict()

        assert data["prob_given_true"] == 0.6  # Raw value, not weighted
        assert data["prob_given_false"] == 0.05  # Raw value, not weighted
        assert data["last_updated"] == likelihood.last_updated.isoformat()

    def test_to_dict_no_values(self, mock_coordinator: Mock) -> None:
        """Test converting likelihood to dictionary with no calculated values."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        data = likelihood.to_dict()

        assert data["prob_given_true"] == 0.8  # Default value
        assert data["prob_given_false"] == 0.1  # Default value
        assert data["last_updated"] is None

    def test_from_dict(self, mock_coordinator: Mock) -> None:
        """Test creating likelihood from dictionary."""
        now = dt_util.utcnow()
        data = {
            "prob_given_true": 0.6,
            "prob_given_false": 0.05,
            "last_updated": now.isoformat(),
        }

        likelihood = Likelihood.from_dict(
            data=data,
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        assert likelihood.entity_id == "binary_sensor.motion"
        assert likelihood.active_states == ["on"]
        assert likelihood.default_prob_true == 0.8
        assert likelihood.default_prob_false == 0.1
        assert likelihood.weight == 0.7
        assert likelihood.active_ratio == 0.6
        assert likelihood.inactive_ratio == 0.05
        assert likelihood.last_updated == now

    def test_from_dict_no_timestamp(self, mock_coordinator: Mock) -> None:
        """Test creating likelihood from dictionary with no timestamp."""
        data = {"prob_given_true": 0.6, "prob_given_false": 0.05, "last_updated": None}

        likelihood = Likelihood.from_dict(
            data=data,
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        assert likelihood.last_updated is None

    def test_roundtrip_serialization(self, mock_coordinator: Mock) -> None:
        """Test that serialization and deserialization preserve data."""
        original = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on", "open"],
            default_prob_true=0.75,
            default_prob_false=0.15,
            weight=0.6,
        )

        # Set some calculated values
        original.active_ratio = 0.55
        original.inactive_ratio = 0.08
        original.last_updated = dt_util.utcnow()

        # Serialize and deserialize
        data = original.to_dict()
        restored = Likelihood.from_dict(
            data=data,
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on", "open"],
            default_prob_true=0.75,
            default_prob_false=0.15,
            weight=0.6,
        )

        # Verify all values match
        assert restored.entity_id == original.entity_id
        assert restored.active_states == original.active_states
        assert restored.default_prob_true == original.default_prob_true
        assert restored.default_prob_false == original.default_prob_false
        assert restored.weight == original.weight
        assert restored.active_ratio == original.active_ratio
        assert restored.inactive_ratio == original.inactive_ratio
        assert restored.last_updated == original.last_updated


class TestLikelihoodEdgeCases:
    """Test edge cases and error conditions for Likelihood class."""

    def test_empty_active_states(self, mock_coordinator: Mock) -> None:
        """Test likelihood with empty active states."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=[],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        assert likelihood.active_states == []
        # Should still work with empty active states
        assert likelihood.prob_given_true > 0
        assert likelihood.prob_given_false > 0

    def test_extreme_weights(self, mock_coordinator: Mock) -> None:
        """Test likelihood with extreme weight values."""
        # Very low weight
        likelihood_low = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.001,
        )

        # Very high weight
        likelihood_high = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.999,
        )

        # Both should produce valid probabilities
        assert 0 < likelihood_low.prob_given_true < 1
        assert 0 < likelihood_low.prob_given_false < 1
        assert 0 < likelihood_high.prob_given_true < 1
        assert 0 < likelihood_high.prob_given_false < 1

    def test_extreme_default_probabilities(self, mock_coordinator: Mock) -> None:
        """Test likelihood with extreme default probability values."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.999,
            default_prob_false=0.001,
            weight=0.5,
        )

        # Should handle extreme values gracefully
        assert likelihood.prob_given_true <= 0.999
        assert likelihood.prob_given_false >= 0.001

    async def test_calculate_basic(self, mock_coordinator: Mock, freeze_time) -> None:
        """Test calculate() computes ratios from intervals and prior data."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=1.0,
        )

        base = freeze_time
        # Prior intervals representing occupancy
        prior_intervals = [
            {"start": base - timedelta(minutes=15), "end": base - timedelta(minutes=7)}
        ]
        mock_coordinator.prior.state_intervals = prior_intervals

        intervals = [
            {
                "state": "on",
                "start": base - timedelta(minutes=10),
                "end": base - timedelta(minutes=5),
            },
            {
                "state": "on",
                "start": base - timedelta(minutes=4),
                "end": base - timedelta(minutes=2),
            },
        ]
        # Mock the async method on the coordinator's sqlite_store
        mock_sqlite_store = Mock()
        mock_sqlite_store.get_historical_intervals = AsyncMock(return_value=intervals)
        mock_coordinator.sqlite_store = mock_sqlite_store

        active_ratio, inactive_ratio = await likelihood.calculate(
            history_period=HA_RECORDER_DAYS
        )

        assert active_ratio == pytest.approx(0.8, rel=1e-3)
        assert inactive_ratio == pytest.approx(0.1, rel=1e-3)

    def test_interval_overlap_helper(self, mock_coordinator: Mock, freeze_time) -> None:
        """Test the optimized interval overlap helper."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=1.0,
        )

        base = freeze_time
        prior = [
            {"start": base - timedelta(minutes=10), "end": base - timedelta(minutes=5)},
            {"start": base - timedelta(minutes=4), "end": base - timedelta(minutes=2)},
        ]
        sorted_prior = sorted(prior, key=lambda x: x["start"])

        overlapping = {
            "start": base - timedelta(minutes=3),
            "end": base - timedelta(minutes=1),
        }
        non_overlapping = {
            "start": base - timedelta(minutes=20),
            "end": base - timedelta(minutes=18),
        }

        assert likelihood._interval_overlaps_prior_optimized(overlapping, sorted_prior)
        assert not likelihood._interval_overlaps_prior_optimized(
            non_overlapping, sorted_prior
        )
