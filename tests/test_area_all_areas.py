"""Tests for AllAreas aggregation class."""

from unittest.mock import MagicMock, patch

from custom_components.area_occupancy.area.all_areas import AllAreas
from custom_components.area_occupancy.const import ALL_AREAS_IDENTIFIER, MIN_PROBABILITY


class TestAllAreas:
    """Test AllAreas aggregation class."""

    def test_device_info(self, coordinator_with_areas) -> None:
        """Test device_info method."""
        all_areas = AllAreas(coordinator_with_areas)
        device_info = all_areas.device_info()

        assert device_info is not None
        assert device_info["identifiers"] == {("area_occupancy", ALL_AREAS_IDENTIFIER)}
        assert device_info["name"] == "All Areas"

    def test_probability_average(self, coordinator_with_areas) -> None:
        """Test probability aggregation averages across all areas."""
        all_areas = AllAreas(coordinator_with_areas)

        area1 = MagicMock()
        area1.probability.return_value = 0.3
        area2 = MagicMock()
        area2.probability.return_value = 0.7
        with patch.dict(
            coordinator_with_areas.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            prob = all_areas.probability()
            # Average of 0.3 and 0.7 = 0.5
            assert prob == 0.5

    def test_occupied_any_area(self, coordinator_with_areas) -> None:
        """Test occupied returns True if ANY area is occupied."""
        all_areas = AllAreas(coordinator_with_areas)

        area1 = MagicMock()
        area1.occupied.return_value = False
        area2 = MagicMock()
        area2.occupied.return_value = True
        with patch.dict(
            coordinator_with_areas.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            assert all_areas.occupied() is True

    def test_occupied_no_areas_occupied(self, coordinator_with_areas) -> None:
        """Test occupied returns False if no areas are occupied."""
        all_areas = AllAreas(coordinator_with_areas)

        area1 = MagicMock()
        area1.occupied.return_value = False
        area2 = MagicMock()
        area2.occupied.return_value = False
        with patch.dict(
            coordinator_with_areas.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            assert all_areas.occupied() is False

    def test_area_prior_average(self, coordinator_with_areas) -> None:
        """Test area_prior aggregation averages across all areas."""
        all_areas = AllAreas(coordinator_with_areas)

        area1 = MagicMock()
        area1.area_prior.return_value = 0.2
        area2 = MagicMock()
        area2.area_prior.return_value = 0.8
        with patch.dict(
            coordinator_with_areas.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            prior = all_areas.area_prior()
            # Average of 0.2 and 0.8 = 0.5
            assert prior == 0.5

    def test_decay_average(self, coordinator_with_areas) -> None:
        """Test decay aggregation averages across all areas."""
        all_areas = AllAreas(coordinator_with_areas)

        area1 = MagicMock()
        area1.decay.return_value = 0.4
        area2 = MagicMock()
        area2.decay.return_value = 0.6
        with patch.dict(
            coordinator_with_areas.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            decay = all_areas.decay()
            # Average of 0.4 and 0.6 = 0.5
            assert decay == 0.5

    def test_probability_clamps_to_bounds(self, coordinator_with_areas) -> None:
        """Test probability clamps to valid bounds."""
        all_areas = AllAreas(coordinator_with_areas)

        area1 = MagicMock()
        area1.probability.return_value = -0.5
        area2 = MagicMock()
        area2.probability.return_value = 1.5
        with patch.dict(
            coordinator_with_areas.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            prob = all_areas.probability()
            # Should clamp to [MIN_PROBABILITY, 1.0]
            assert MIN_PROBABILITY <= prob <= 1.0

    def test_probability_empty_areas(self, coordinator_with_areas) -> None:
        """Test probability returns safe default when no areas exist."""
        all_areas = AllAreas(coordinator_with_areas)

        with patch.dict(coordinator_with_areas.areas, {}, clear=True):
            prob = all_areas.probability()
            # Should return MIN_PROBABILITY as safe default
            assert prob == MIN_PROBABILITY

    def test_area_prior_empty_areas(self, coordinator_with_areas) -> None:
        """Test area_prior returns safe default when no areas exist."""
        all_areas = AllAreas(coordinator_with_areas)

        with patch.dict(coordinator_with_areas.areas, {}, clear=True):
            prior = all_areas.area_prior()
            # Should return MIN_PROBABILITY as safe default
            assert prior == MIN_PROBABILITY

    def test_decay_empty_areas(self, coordinator_with_areas) -> None:
        """Test decay returns safe default when no areas exist."""
        all_areas = AllAreas(coordinator_with_areas)

        with patch.dict(coordinator_with_areas.areas, {}, clear=True):
            decay = all_areas.decay()
            # Should return 1.0 as safe default (no decay = full probability)
            assert decay == 1.0
