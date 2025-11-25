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

    def test_get_likelihoods_binary_sensor_static(self, mock_binary_entity):
        """Test get_likelihoods for binary sensor using static probabilities."""
        # Binary sensors now use static probabilities, not Gaussian PDF
        # Set up learned probabilities from duration-based analysis
        mock_binary_entity.prob_given_true = 0.8
        mock_binary_entity.prob_given_false = 0.1
        mock_binary_entity.analysis_error = None  # Analysis succeeded
        mock_binary_entity.learned_gaussian_params = (
            None  # Binary sensors don't use Gaussian
        )

        # Should return static probabilities regardless of state
        mock_binary_entity.state_provider = lambda x: STATE_ON
        p_t, p_f = mock_binary_entity.get_likelihoods()
        assert p_t == 0.8
        assert p_f == 0.1

        mock_binary_entity.state_provider = lambda x: STATE_OFF
        p_t, p_f = mock_binary_entity.get_likelihoods()
        assert p_t == 0.8  # Still same static values
        assert p_f == 0.1

        # If not analyzed yet, should use EntityType defaults
        mock_binary_entity.analysis_error = "not_analyzed"
        p_t, p_f = mock_binary_entity.get_likelihoods()
        assert p_t == mock_binary_entity.type.prob_given_true
        assert p_f == mock_binary_entity.type.prob_given_false

    def test_get_likelihoods_fallback(self, mock_numeric_entity):
        """Test get_likelihoods fallback behavior uses EntityType defaults."""
        # No params -> returns EntityType defaults (not stored prob_given_true/false)
        mock_numeric_entity.learned_gaussian_params = None
        # Change stored values to verify we use EntityType defaults
        mock_numeric_entity.prob_given_true = 0.9
        mock_numeric_entity.prob_given_false = 0.1
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        # Should use EntityType defaults (0.5, 0.5), not stored values
        assert p_t == 0.5
        assert p_f == 0.5

        # With params but invalid state -> uses representative value (average of means)
        mock_numeric_entity.learned_gaussian_params = {
            "mean_occupied": 22.0,
            "std_occupied": 1.0,
            "mean_unoccupied": 20.0,
            "std_unoccupied": 1.0,
        }
        mock_numeric_entity.state_provider = lambda x: "unavailable"
        p_t, p_f = mock_numeric_entity.get_likelihoods()
        # Should use representative value (average of means = 21.0) to calculate probabilities
        # This will give non-zero probabilities based on Gaussian PDF
        assert p_t > 0.0
        assert p_f > 0.0
        assert p_t != 0.5  # Should be calculated, not default
        assert p_f != 0.5  # Should be calculated, not default

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
        # Should NOT update stored prob_given_true/false (no fallback)

    def test_get_likelihoods_motion_sensor_uses_configured_values(self):
        """Test that motion sensors always use configured prob_given_true/false."""
        entity_type = EntityType(
            input_type=InputType.MOTION,
            weight=0.85,
            prob_given_true=0.95,  # EntityType default
            prob_given_false=0.02,  # EntityType default
            active_states=[STATE_ON],
        )
        decay = Decay(half_life=60.0)

        # Create motion sensor with different configured values
        motion_entity = Entity(
            entity_id="binary_sensor.motion",
            type=entity_type,
            prob_given_true=0.9,  # Configured value (different from EntityType)
            prob_given_false=0.05,  # Configured value (different from EntityType)
            decay=decay,
            state_provider=lambda x: STATE_ON,
            last_updated=dt_util.utcnow(),
        )

        # Even with Gaussian params, motion sensors should use configured values
        motion_entity.learned_gaussian_params = {
            "mean_occupied": 0.9,
            "std_occupied": 0.3,
            "mean_unoccupied": 0.1,
            "std_unoccupied": 0.3,
        }

        p_t, p_f = motion_entity.get_likelihoods()
        # Should use configured values, not Gaussian params or EntityType defaults
        assert p_t == 0.9
        assert p_f == 0.05
