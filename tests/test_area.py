"""Tests for Area class methods."""

from unittest.mock import Mock, PropertyMock, patch

from custom_components.area_occupancy.area.area import Area
from custom_components.area_occupancy.const import (
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.entity_type import InputType


# ruff: noqa: SLF001
class TestAreaMethods:
    """Test Area class methods."""

    def test_device_info(self, default_area: Area) -> None:
        """Test device_info method."""
        device_info = default_area.device_info()

        assert device_info is not None
        # Device identifier should use area_id (stable even if area is renamed)
        # Fallback to area_name for legacy compatibility
        expected_identifier = default_area.config.area_id or default_area.area_name
        assert device_info["identifiers"] == {(DOMAIN, expected_identifier)}
        assert device_info["name"] == default_area.config.name
        assert device_info["manufacturer"] == DEVICE_MANUFACTURER
        assert device_info["model"] == DEVICE_MODEL
        assert device_info["sw_version"] == DEVICE_SW_VERSION

    def test_probability_with_entities(self, default_area: Area) -> None:
        """Test probability method with entities."""
        # Set up mock entities with PropertyMock for weight property
        mock_entity1 = Mock()
        mock_entity1.evidence = True
        mock_entity1.prob_given_true = 0.8
        mock_entity1.prob_given_false = 0.2
        mock_entity1.type = Mock(weight=0.85)
        mock_entity1.decay = Mock(decay_factor=1.0)
        type(mock_entity1).weight = PropertyMock(return_value=0.85)

        mock_entity2 = Mock()
        mock_entity2.evidence = False
        mock_entity2.prob_given_true = 0.7
        mock_entity2.prob_given_false = 0.3
        mock_entity2.type = Mock(weight=0.7)
        mock_entity2.decay = Mock(decay_factor=1.0)
        type(mock_entity2).weight = PropertyMock(return_value=0.7)

        # Set entities via _entities private attribute
        default_area.entities._entities = {
            "binary_sensor.motion1": mock_entity1,
            "media_player.tv": mock_entity2,
        }

        # Set prior - need to set both global_prior and ensure time_prior is None
        default_area.prior.global_prior = 0.3
        default_area.prior._cached_time_prior = None

        prob = default_area.probability()
        assert 0.0 <= prob <= 1.0
        assert (
            prob > MIN_PROBABILITY
        )  # Should be higher than minimum with active entity

    def test_probability_no_entities(self, default_area: Area) -> None:
        """Test probability method with no entities."""
        default_area.entities._entities = {}
        prob = default_area.probability()
        assert prob == MIN_PROBABILITY

    def test_type_probabilities(self, default_area: Area) -> None:
        """Test type_probabilities method."""
        # Set up mock entities by type with PropertyMock for weight property
        mock_motion = Mock()
        mock_motion.evidence = True
        mock_motion.prob_given_true = 0.8
        mock_motion.prob_given_false = 0.2
        mock_motion.type = Mock(input_type=InputType.MOTION, weight=0.85)
        mock_motion.decay = Mock(decay_factor=1.0)
        type(mock_motion).weight = PropertyMock(return_value=0.85)

        mock_media = Mock()
        mock_media.evidence = False
        mock_media.prob_given_true = 0.7
        mock_media.prob_given_false = 0.3
        mock_media.type = Mock(input_type=InputType.MEDIA, weight=0.7)
        mock_media.decay = Mock(decay_factor=1.0)
        type(mock_media).weight = PropertyMock(return_value=0.7)

        default_area.entities._entities = {
            "binary_sensor.motion": mock_motion,
            "media_player.tv": mock_media,
        }
        default_area.prior.global_prior = 0.3
        default_area.prior._cached_time_prior = None

        type_probs = default_area.type_probabilities()

        assert isinstance(type_probs, dict)
        assert InputType.MOTION in type_probs
        assert InputType.MEDIA in type_probs
        assert 0.0 <= type_probs[InputType.MOTION] <= 1.0
        assert 0.0 <= type_probs[InputType.MEDIA] <= 1.0

    def test_type_probabilities_no_entities(self, default_area: Area) -> None:
        """Test type_probabilities method with no entities."""
        default_area.entities._entities = {}
        type_probs = default_area.type_probabilities()
        assert type_probs == {}

    def test_area_prior(self, default_area: Area) -> None:
        """Test area_prior method."""
        # Set global_prior and ensure time_prior is None to get predictable result
        default_area.prior.global_prior = 0.35
        default_area.prior._cached_time_prior = None
        # area_prior() returns prior.value which applies PRIOR_FACTOR (1.05)
        # and clamps to bounds, so we check it's approximately correct
        prior_value = default_area.area_prior()
        assert 0.0 <= prior_value <= 1.0
        # Should be close to 0.35 * 1.05 = 0.3675, but clamped to MAX_PRIOR
        assert prior_value >= 0.35

    def test_decay_with_entities(self, default_area: Area) -> None:
        """Test decay method with entities."""
        mock_entity1 = Mock()
        mock_entity1.decay = Mock(decay_factor=0.8)

        mock_entity2 = Mock()
        mock_entity2.decay = Mock(decay_factor=0.6)

        # Set entities via _entities private attribute
        default_area.entities._entities = {
            "binary_sensor.motion1": mock_entity1,
            "media_player.tv": mock_entity2,
        }

        decay = default_area.decay()
        assert decay == 0.7  # (0.8 + 0.6) / 2

    def test_decay_no_entities(self, default_area: Area) -> None:
        """Test decay method with no entities."""
        default_area.entities._entities = {}
        decay = default_area.decay()
        assert decay == 1.0

    def test_occupied_true(self, default_area: Area) -> None:
        """Test occupied method returns True when probability >= threshold."""
        # Set threshold
        default_area.config.threshold = 0.5

        # Mock probability to return 0.6 (above threshold)
        with patch.object(default_area, "probability", return_value=0.6):
            assert default_area.occupied() is True

    def test_occupied_false(self, default_area: Area) -> None:
        """Test occupied method returns False when probability < threshold."""
        # Set threshold
        default_area.config.threshold = 0.5

        # Mock probability to return 0.4 (below threshold)
        with patch.object(default_area, "probability", return_value=0.4):
            assert default_area.occupied() is False

    def test_occupied_at_threshold(self, default_area: Area) -> None:
        """Test occupied method returns True when probability equals threshold."""
        # Set threshold
        default_area.config.threshold = 0.5

        # Mock probability to return 0.5 (at threshold)
        with patch.object(default_area, "probability", return_value=0.5):
            assert default_area.occupied() is True

    def test_threshold(self, default_area: Area) -> None:
        """Test threshold method."""
        default_area.config.threshold = 0.65
        assert default_area.threshold() == 0.65

    def test_threshold_default(self, default_area: Area) -> None:
        """Test threshold method returns default value."""
        # Default threshold should be around 0.5 (50%)
        threshold = default_area.threshold()
        assert 0.0 <= threshold <= 1.0


class TestAreaMethodsIntegration:
    """Integration tests for Area methods with real coordinator."""

    def test_probability_integration(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test probability method with real coordinator."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)

        prob = area.probability()
        assert 0.0 <= prob <= 1.0

    def test_occupied_integration(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test occupied method with real coordinator."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)

        occupied = area.occupied()
        assert isinstance(occupied, bool)

    def test_device_info_integration(
        self, coordinator_with_areas: AreaOccupancyCoordinator
    ) -> None:
        """Test device_info method with real coordinator."""
        area_name = coordinator_with_areas.get_area_names()[0]
        area = coordinator_with_areas.get_area_or_default(area_name)

        device_info = area.device_info()
        assert device_info is not None
        # Device identifier should use area_id (stable even if area is renamed)
        # Fallback to area_name for legacy compatibility
        expected_identifier = area.config.area_id or area_name
        assert device_info["identifiers"] == {(DOMAIN, expected_identifier)}
        assert device_info["name"] == area.config.name
