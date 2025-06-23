"""Tests for data.likelihood module."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

from custom_components.area_occupancy.data.likelihood import Likelihood
from homeassistant.core import State
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

    def test_apply_weight_to_probability_low_threshold(
        self, mock_coordinator: Mock
    ) -> None:
        """Test weight application when calculated probability is below threshold."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.5,
        )

        # Test with very low calculated probability (below 5% threshold)
        result = likelihood._apply_weight_to_probability(0.02, 0.8)
        # Should use default * weight = 0.8 * 0.5 = 0.4
        assert result == 0.4

        # Test with zero probability
        result = likelihood._apply_weight_to_probability(0.0, 0.6)
        # Should use default * weight = 0.6 * 0.5 = 0.3
        assert result == 0.3

    def test_apply_weight_to_probability_high_threshold(
        self, mock_coordinator: Mock
    ) -> None:
        """Test weight application when calculated probability is above threshold."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.6,
        )

        # Test with probability above 5% threshold
        # Formula: neutral + (prob - neutral) * weight = 0.5 + (0.8 - 0.5) * 0.6 = 0.5 + 0.18 = 0.68
        result = likelihood._apply_weight_to_probability(0.8, 0.2)
        assert abs(result - 0.68) < 0.001

        # Test with probability below neutral
        # Formula: 0.5 + (0.3 - 0.5) * 0.6 = 0.5 - 0.12 = 0.38
        result = likelihood._apply_weight_to_probability(0.3, 0.9)
        assert abs(result - 0.38) < 0.001

    def test_apply_weight_to_probability_clamping(self, mock_coordinator: Mock) -> None:
        """Test that weight application properly clamps values."""
        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=2.0,  # Very high weight
        )

        # Test upper clamping
        result = likelihood._apply_weight_to_probability(0.9, 0.8)
        assert result <= 0.999

        # Test lower clamping with very low default and weight
        likelihood.weight = 0.001
        result = likelihood._apply_weight_to_probability(0.01, 0.01)
        assert result >= 0.001

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
        assert likelihood.prob_given_true == likelihood._apply_weight_to_probability(
            0.8, 0.8
        )

        # Test with calculated values
        likelihood.active_ratio = 0.6
        assert likelihood.prob_given_true == likelihood._apply_weight_to_probability(
            0.6, 0.8
        )

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
        assert likelihood.prob_given_false == likelihood._apply_weight_to_probability(
            0.1, 0.1
        )

        # Test with calculated values
        likelihood.inactive_ratio = 0.05
        assert likelihood.prob_given_false == likelihood._apply_weight_to_probability(
            0.05, 0.1
        )

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

    async def test_update_history_disabled(self, mock_coordinator: Mock) -> None:
        """Test update method when history is disabled."""
        mock_coordinator.config.history.enabled = False

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        prob_true, prob_false = await likelihood.update()

        # Should return weighted default values
        assert prob_true == likelihood.prob_given_true
        assert prob_false == likelihood.prob_given_false

    async def test_update_history_enabled_success(self, mock_coordinator: Mock) -> None:
        """Test update method when history is enabled and calculation succeeds."""
        mock_coordinator.config.history.enabled = True

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
        mock_coordinator.config.history.enabled = True

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

    @patch("custom_components.area_occupancy.data.likelihood.get_states_from_recorder")
    @patch("custom_components.area_occupancy.data.likelihood.states_to_intervals")
    async def test_calculate_with_prior_intervals(
        self,
        mock_states_to_intervals: AsyncMock,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test calculate method with prior intervals and states."""
        # Set up mock coordinator with prior intervals
        mock_coordinator.prior.prior_intervals = [
            {
                "start": dt_util.utcnow() - timedelta(hours=2),
                "end": dt_util.utcnow() - timedelta(hours=1),
                "state": "on",
            }
        ]
        mock_coordinator.config.history.period = 1  # 1 day

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Mock states from recorder
        now = dt_util.utcnow()
        mock_states = [
            State(
                "binary_sensor.motion", "on", last_changed=now - timedelta(hours=1.5)
            ),
            State(
                "binary_sensor.motion", "off", last_changed=now - timedelta(hours=0.5)
            ),
        ]
        mock_get_states.return_value = mock_states

        # Mock intervals
        mock_intervals = [
            {
                "start": now - timedelta(hours=2),
                "end": now - timedelta(hours=1),
                "state": "on",
            },
            {
                "start": now - timedelta(hours=1),
                "end": now,
                "state": "off",
            },
        ]
        mock_states_to_intervals.return_value = mock_intervals

        active_ratio, inactive_ratio = await likelihood.calculate()

        # Verify calls
        mock_get_states.assert_called_once()
        mock_states_to_intervals.assert_called_once()

        # Should calculate based on overlap with prior intervals
        assert isinstance(active_ratio, float)
        assert isinstance(inactive_ratio, float)
        assert 0 <= active_ratio <= 1
        assert 0 <= inactive_ratio <= 1

    @patch("custom_components.area_occupancy.data.likelihood.get_states_from_recorder")
    async def test_calculate_no_states(
        self, mock_get_states: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test calculate method when no states are available."""
        mock_coordinator.prior.prior_intervals = []
        mock_coordinator.config.history.period = 1

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Mock no states
        mock_get_states.return_value = []

        active_ratio, inactive_ratio = await likelihood.calculate()

        # Should return defaults
        assert active_ratio == 0.8
        assert inactive_ratio == 0.1

    @patch("custom_components.area_occupancy.data.likelihood.get_states_from_recorder")
    async def test_calculate_no_prior_intervals(
        self, mock_get_states: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test calculate method when no prior intervals are available."""
        mock_coordinator.prior.prior_intervals = []
        mock_coordinator.config.history.period = 1

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        # Mock some states
        mock_get_states.return_value = [
            State("binary_sensor.motion", "on", last_changed=dt_util.utcnow()),
        ]

        active_ratio, inactive_ratio = await likelihood.calculate()

        # Should return defaults when no prior intervals
        assert active_ratio == 0.8
        assert inactive_ratio == 0.1

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
        data = {
            "prob_given_true": 0.6,
            "prob_given_false": 0.05,
            "last_updated": None,
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

    @patch("custom_components.area_occupancy.data.likelihood.get_states_from_recorder")
    @patch("custom_components.area_occupancy.data.likelihood.states_to_intervals")
    async def test_calculate_with_zero_time_periods(
        self,
        mock_states_to_intervals: AsyncMock,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test calculate method when occupied or not-occupied time is zero."""
        # Set up mock coordinator with no prior intervals (zero occupied time)
        mock_coordinator.prior.prior_intervals = []
        mock_coordinator.config.history.period = 1

        likelihood = Likelihood(
            coordinator=mock_coordinator,
            entity_id="binary_sensor.motion",
            active_states=["on"],
            default_prob_true=0.8,
            default_prob_false=0.1,
            weight=0.7,
        )

        mock_get_states.return_value = [
            State("binary_sensor.motion", "on", last_changed=dt_util.utcnow()),
        ]

        active_ratio, inactive_ratio = await likelihood.calculate()

        # Should fall back to defaults when time periods are invalid
        assert active_ratio == 0.8
        assert inactive_ratio == 0.1
