"""Tests for continuous likelihood calculation in entities."""

import pytest

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity, EntityType
from custom_components.area_occupancy.data.entity_type import InputType
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001
@pytest.fixture
def mock_entity():
    """Create a mock entity for testing."""
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


class TestGaussianLikelihood:
    """Test Gaussian likelihood calculation."""

    def test_is_continuous_likelihood_property(self, mock_entity):
        """Test is_continuous_likelihood property."""
        # Initially false
        assert not mock_entity.is_continuous_likelihood

        # Set gaussian params
        mock_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        # Now true
        assert mock_entity.is_continuous_likelihood

    def test_calculate_gaussian_density(self, mock_entity):
        """Test _calculate_gaussian_density method."""
        # Test peak density (at mean)
        # 1 / (sqrt(2*pi) * 1) ≈ 0.3989
        density = mock_entity._calculate_gaussian_density(20.0, 20.0, 1.0)
        assert abs(density - 0.3989) < 0.0001

        # Test 1 std dev away
        # 0.3989 * exp(-0.5 * 1^2) ≈ 0.2420
        density = mock_entity._calculate_gaussian_density(21.0, 20.0, 1.0)
        assert abs(density - 0.2420) < 0.0001

        # Test 2 std dev away
        # 0.3989 * exp(-0.5 * 2^2) ≈ 0.0540
        density = mock_entity._calculate_gaussian_density(22.0, 20.0, 1.0)
        assert abs(density - 0.0540) < 0.0001

        # Test small std dev (higher peak)
        # 1 / (sqrt(2*pi) * 0.1) ≈ 3.989
        density = mock_entity._calculate_gaussian_density(20.0, 20.0, 0.1)
        assert abs(density - 3.989) < 0.001

    def test_get_likelihoods_continuous(self, mock_entity):
        """Test get_likelihoods with continuous parameters."""
        # Setup:
        # Occupied: Mean 22, Std 1
        # Unoccupied: Mean 20, Std 1
        mock_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }

        # Case 1: Value = 22 (Occupied Mean)
        # P(val|Occ) should be peak (~0.4)
        # P(val|Unocc) should be 2 std away (~0.054)
        mock_entity.state_provider = lambda x: "22.0"
        p_t, p_f = mock_entity.get_likelihoods()

        assert abs(p_t - 0.3989) < 0.001
        assert abs(p_f - 0.0540) < 0.001
        # Likelihood ratio > 1 (favors occupied)
        assert p_t > p_f

        # Case 2: Value = 20 (Unoccupied Mean)
        # P(val|Occ) should be 2 std away (~0.054)
        # P(val|Unocc) should be peak (~0.4)
        mock_entity.state_provider = lambda x: "20.0"
        p_t, p_f = mock_entity.get_likelihoods()

        assert abs(p_t - 0.0540) < 0.001
        assert abs(p_f - 0.3989) < 0.001
        # Likelihood ratio < 1 (favors unoccupied)
        assert p_t < p_f

        # Case 3: Value = 21 (Middle)
        # Should be equal distance from both means (1 std)
        # Densities should be equal
        mock_entity.state_provider = lambda x: "21.0"
        p_t, p_f = mock_entity.get_likelihoods()

        assert abs(p_t - p_f) < 0.0001
        assert abs(p_t - 0.2420) < 0.001

    def test_get_likelihoods_fallback(self, mock_entity):
        """Test get_likelihoods fallback behavior."""
        # No params -> returns static probabilities
        mock_entity.learned_gaussian_params = None
        p_t, p_f = mock_entity.get_likelihoods()
        assert p_t == 0.5
        assert p_f == 0.5

        # With params but invalid state -> returns static probabilities
        mock_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }
        mock_entity.state_provider = lambda x: "unavailable"
        p_t, p_f = mock_entity.get_likelihoods()
        assert p_t == 0.5
        assert p_f == 0.5

    def test_update_correlation_populates_params(self, mock_entity):
        """Test update_correlation populates Gaussian params."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "occupancy_positive",
            "mean_value_when_occupied": 22.0,
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_occupied": 1.5,
            "std_dev_when_unoccupied": 1.2,
        }

        mock_entity.update_correlation(correlation_data)

        assert mock_entity.learned_gaussian_params is not None
        assert mock_entity.learned_gaussian_params["mean_occupied"] == 22.0
        assert mock_entity.learned_gaussian_params["std_occupied"] == 1.5
        assert mock_entity.learned_gaussian_params["mean_unoccupied"] == 20.0
        assert mock_entity.learned_gaussian_params["std_unoccupied"] == 1.2

        # Should also populate learned_active_range for UI
        assert mock_entity.learned_active_range is not None

    def test_update_correlation_missing_params(self, mock_entity):
        """Test update_correlation handles missing occupied stats."""
        correlation_data = {
            "confidence": 0.8,
            "correlation_type": "occupancy_positive",
            "mean_value_when_unoccupied": 20.0,
            "std_dev_when_unoccupied": 1.2,
            # Missing occupied stats
        }

        mock_entity.update_correlation(correlation_data)

        # Should NOT populate gaussian params
        assert mock_entity.learned_gaussian_params is None
        # Should still populate active range (open-ended)
        assert mock_entity.learned_active_range is not None
        assert mock_entity.learned_active_range[1] == float("inf")
        # Should update static likelihoods
        assert mock_entity.prob_given_true != 0.5
