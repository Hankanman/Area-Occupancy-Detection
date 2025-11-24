"""Tests for likelihood calculation in entities (both numeric and binary)."""

import pytest

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity, EntityType
from custom_components.area_occupancy.data.entity_type import InputType
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
@pytest.fixture
def mock_numeric_entity():
    """Create a mock numeric entity for testing."""
    entity_type = EntityType(
        input_type=InputType.TEMPERATURE,
        weight=0.1,
        prob_given_true=0.5,
        prob_given_false=0.5,
        active_range=None,
    )
    decay = Decay(half_life=60.0)

    return Entity(
        entity_id="sensor.temp",
        type=entity_type,
        prob_given_true=0.5,
        prob_given_false=0.5,
        decay=decay,
        state_provider=lambda x: "20.0",
        last_updated=dt_util.utcnow(),
    )


@pytest.fixture
def mock_binary_entity():
    """Create a mock binary entity for testing."""
    entity_type = EntityType(
        input_type=InputType.MEDIA,
        weight=0.7,
        prob_given_true=0.5,
        prob_given_false=0.5,
        active_states=[STATE_ON],
    )
    decay = Decay(half_life=60.0)

    return Entity(
        entity_id="media_player.tv",
        type=entity_type,
        prob_given_true=0.5,
        prob_given_false=0.5,
        decay=decay,
        state_provider=lambda x: STATE_ON,
        last_updated=dt_util.utcnow(),
    )


class TestGaussianLikelihood:
    """Test Gaussian likelihood calculation."""

    def test_is_continuous_likelihood_property(self, mock_numeric_entity):
        """Test is_continuous_likelihood property."""
        # Initially false
        assert not mock_numeric_entity.is_continuous_likelihood

        # Set gaussian params
        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        # Now true
        assert mock_numeric_entity.is_continuous_likelihood

    def test_calculate_gaussian_density(self, mock_numeric_entity):
        """Test _calculate_gaussian_density method."""
        # Test peak density (at mean)
        # 1 / (sqrt(2*pi) * 1) ≈ 0.3989
        density = mock_numeric_entity._calculate_gaussian_density(20.0, 20.0, 1.0)
        assert abs(density - 0.3989) < 0.0001

        # Test 1 std dev away
        # 0.3989 * exp(-0.5 * 1^2) ≈ 0.2420
        density = mock_numeric_entity._calculate_gaussian_density(21.0, 20.0, 1.0)
        assert abs(density - 0.2420) < 0.0001

        # Test 2 std dev away
        # 0.3989 * exp(-0.5 * 2^2) ≈ 0.0540
        density = mock_numeric_entity._calculate_gaussian_density(22.0, 20.0, 1.0)
        assert abs(density - 0.0540) < 0.0001

        # Test small std dev (higher peak)
        # 1 / (sqrt(2*pi) * 0.1) ≈ 3.989
        density = mock_numeric_entity._calculate_gaussian_density(20.0, 20.0, 0.1)
        assert abs(density - 3.989) < 0.001

    def test_get_likelihoods_continuous_numeric(self, mock_numeric_entity):
        """Test get_likelihoods with continuous parameters for numeric sensor."""
        # Setup:
        # Occupied: Mean 22, Std 1
        # Unoccupied: Mean 20, Std 1
        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        # Case 1: Value = 22 (Occupied Mean)
        # P(val|Occ) should be peak (~0.3989)
        # P(val|Unocc) should be 2 std away (~0.054)
        mock_numeric_entity.state_provider = lambda x: "22.0"
        p_t, p_f = mock_numeric_entity.get_likelihoods()

        assert abs(p_t - 0.3989) < 0.001
        assert abs(p_f - 0.0540) < 0.001
        # Likelihood ratio > 1 (favors occupied)
        assert p_t > p_f

        # Case 2: Value = 20 (Unoccupied Mean)
        # P(val|Occ) should be 2 std away (~0.054)
        # P(val|Unocc) should be peak (~0.3989)
        mock_numeric_entity.state_provider = lambda x: "20.0"
        p_t, p_f = mock_numeric_entity.get_likelihoods()

        assert abs(p_t - 0.0540) < 0.001
        assert abs(p_f - 0.3989) < 0.001
        # Likelihood ratio < 1 (favors unoccupied)
        assert p_t < p_f

        # Case 3: Value = 21 (Middle)
        # Should be equal distance from both means (1 std)
        # Densities should be equal
        mock_numeric_entity.state_provider = lambda x: "21.0"
        p_t, p_f = mock_numeric_entity.get_likelihoods()

        assert abs(p_t - p_f) < 0.0001
        assert abs(p_t - 0.2420) < 0.001

    def test_get_likelihoods_continuous_binary(self, mock_binary_entity):
        """Test get_likelihoods with continuous parameters for binary sensor."""
        # Setup:
        # Binary sensors are converted to 0.0 (off) and 1.0 (on)
        # Let's say when occupied, it's mostly ON (mean ~0.9, std ~0.3)
        # When unoccupied, it's mostly OFF (mean ~0.1, std ~0.3)
        mock_binary_entity.learned_gaussian_params = {
            "mean_occupied": 0.9,
            "std_occupied": 0.3,
            "mean_unoccupied": 0.1,
            "std_unoccupied": 0.3,
        }

        # Case 1: State = ON (Value 1.0)
        # P(1.0|Occ) -> Gaussian(1.0, mean=0.9, std=0.3)
        #   diff = 0.1, exp(-0.5 * (0.1/0.3)^2) = exp(-0.055) ≈ 0.946
        #   coeff = 1 / (sqrt(2*pi) * 0.3) ≈ 1.33
        #   density ≈ 1.258
        # P(1.0|Unocc) -> Gaussian(1.0, mean=0.1, std=0.3)
        #   diff = 0.9, exp(-0.5 * (0.9/0.3)^2) = exp(-4.5) ≈ 0.011
        #   density ≈ 0.014
        mock_binary_entity.state_provider = lambda x: STATE_ON
        p_t, p_f = mock_binary_entity.get_likelihoods()

        assert p_t > 1.0  # Densities can be > 1
        assert p_f < 0.1
        assert p_t > p_f

        # Case 2: State = OFF (Value 0.0)
        # P(0.0|Occ) -> Gaussian(0.0, mean=0.9, std=0.3) -> Same as P(1.0|Unocc)
        # P(0.0|Unocc) -> Gaussian(0.0, mean=0.1, std=0.3) -> Same as P(1.0|Occ)
        mock_binary_entity.state_provider = lambda x: STATE_OFF
        p_t, p_f = mock_binary_entity.get_likelihoods()

        assert p_t < 0.1
        assert p_f > 1.0
        assert p_f > p_t

    def test_get_likelihoods_fallback(self, mock_numeric_entity):
        """Test get_likelihoods fallback behavior."""
        # No params -> returns static probabilities
        mock_numeric_entity.learned_gaussian_params = None
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        assert p_t == 0.5
        assert p_f == 0.5

        # With params but invalid state -> returns static probabilities
        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }
        mock_numeric_entity.state_provider = lambda x: "unavailable"
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        assert p_t == 0.5
        assert p_f == 0.5

    def test_update_correlation_populates_params(self, mock_numeric_entity):
        """Test update_correlation populates Gaussian params."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "occupancy_positive",
            "mean_value_when_occupied": 22.0,
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_occupied": 1.5,
            "std_dev_when_unoccupied": 1.2,
        }

        mock_numeric_entity.update_correlation(correlation_data)

        assert mock_numeric_entity.learned_gaussian_params is not None
        assert mock_numeric_entity.learned_gaussian_params["mean_occupied"] == 22.0
        assert mock_numeric_entity.learned_gaussian_params["std_occupied"] == 1.5
        assert mock_numeric_entity.learned_gaussian_params["mean_unoccupied"] == 20.0
        assert mock_numeric_entity.learned_gaussian_params["std_unoccupied"] == 1.2

        # Should also populate learned_active_range for UI
        assert mock_numeric_entity.learned_active_range is not None

    def test_update_correlation_missing_params(self, mock_numeric_entity):
        """Test update_correlation handles missing occupied stats."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "occupancy_positive",
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_unoccupied": 1.2,
            # Missing occupied stats
        }

        mock_numeric_entity.update_correlation(correlation_data)

        # Should NOT populate gaussian params
        assert mock_numeric_entity.learned_gaussian_params is None
        # Should still populate active range (open-ended)
        assert mock_numeric_entity.learned_active_range is not None
        assert mock_numeric_entity.learned_active_range[1] == float("inf")
        # Should update static likelihoods
        assert mock_numeric_entity.prob_given_true != 0.5
