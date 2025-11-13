"""Tests for AllAreas aggregation class."""

from unittest.mock import patch

from custom_components.area_occupancy.area.all_areas import AllAreas
from custom_components.area_occupancy.const import ALL_AREAS_IDENTIFIER, MIN_PROBABILITY


# ruff: noqa: PLC0415
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

        # Mock get_area_names to return multiple areas, and probability to return known values
        with (
            patch.object(
                coordinator_with_areas,
                "get_area_names",
                return_value=["Area1", "Area2"],
            ),
            patch.object(coordinator_with_areas, "probability", side_effect=[0.3, 0.7]),
        ):
            prob = all_areas.probability()
            # Average of 0.3 and 0.7 = 0.5
            assert prob == 0.5

    def test_probability_no_areas(self, mock_hass, mock_config_entry) -> None:
        """Test probability with no areas returns MIN_PROBABILITY."""
        from custom_components.area_occupancy.coordinator import (
            AreaOccupancyCoordinator,
        )

        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
        all_areas = AllAreas(coordinator)
        prob = all_areas.probability()
        assert prob == MIN_PROBABILITY

    def test_occupied_any_area(self, coordinator_with_areas) -> None:
        """Test occupied returns True if ANY area is occupied."""
        all_areas = AllAreas(coordinator_with_areas)

        # Mock get_area_names to return multiple areas, and occupied to return True for at least one
        with (
            patch.object(
                coordinator_with_areas,
                "get_area_names",
                return_value=["Area1", "Area2"],
            ),
            patch.object(coordinator_with_areas, "occupied", side_effect=[False, True]),
        ):
            assert all_areas.occupied() is True

    def test_occupied_no_areas_occupied(self, coordinator_with_areas) -> None:
        """Test occupied returns False if no areas are occupied."""
        all_areas = AllAreas(coordinator_with_areas)

        # Mock coordinator.occupied to return False for all areas
        with patch.object(coordinator_with_areas, "occupied", return_value=False):
            assert all_areas.occupied() is False

    def test_occupied_no_areas(self, mock_hass, mock_config_entry) -> None:
        """Test occupied with no areas returns False."""
        from custom_components.area_occupancy.coordinator import (
            AreaOccupancyCoordinator,
        )

        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
        all_areas = AllAreas(coordinator)
        assert all_areas.occupied() is False

    def test_area_prior_average(self, coordinator_with_areas) -> None:
        """Test area_prior aggregation averages across all areas."""
        all_areas = AllAreas(coordinator_with_areas)

        # Mock get_area_names to return multiple areas, and area_prior to return known values
        with (
            patch.object(
                coordinator_with_areas,
                "get_area_names",
                return_value=["Area1", "Area2"],
            ),
            patch.object(coordinator_with_areas, "area_prior", side_effect=[0.2, 0.8]),
        ):
            prior = all_areas.area_prior()
            # Average of 0.2 and 0.8 = 0.5
            assert prior == 0.5

    def test_area_prior_no_areas(self, mock_hass, mock_config_entry) -> None:
        """Test area_prior with no areas returns MIN_PROBABILITY."""
        from custom_components.area_occupancy.coordinator import (
            AreaOccupancyCoordinator,
        )

        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
        all_areas = AllAreas(coordinator)
        prior = all_areas.area_prior()
        assert prior == MIN_PROBABILITY

    def test_decay_average(self, coordinator_with_areas) -> None:
        """Test decay aggregation averages across all areas."""
        all_areas = AllAreas(coordinator_with_areas)

        # Mock get_area_names to return multiple areas, and decay to return known values
        with (
            patch.object(
                coordinator_with_areas,
                "get_area_names",
                return_value=["Area1", "Area2"],
            ),
            patch.object(coordinator_with_areas, "decay", side_effect=[0.4, 0.6]),
        ):
            decay = all_areas.decay()
            # Average of 0.4 and 0.6 = 0.5
            assert decay == 0.5

    def test_decay_no_areas(self, mock_hass, mock_config_entry) -> None:
        """Test decay with no areas returns 1.0."""
        from custom_components.area_occupancy.coordinator import (
            AreaOccupancyCoordinator,
        )

        coordinator = AreaOccupancyCoordinator(mock_hass, mock_config_entry)
        all_areas = AllAreas(coordinator)
        decay = all_areas.decay()
        assert decay == 1.0

    def test_probability_clamps_to_bounds(self, coordinator_with_areas) -> None:
        """Test probability clamps to valid bounds."""
        all_areas = AllAreas(coordinator_with_areas)

        # Mock get_area_names to return multiple areas, and test with values that would average outside bounds
        with (
            patch.object(
                coordinator_with_areas,
                "get_area_names",
                return_value=["Area1", "Area2"],
            ),
            patch.object(
                coordinator_with_areas, "probability", side_effect=[-0.5, 1.5]
            ),
        ):
            prob = all_areas.probability()
            # Should clamp to [MIN_PROBABILITY, 1.0]
            assert MIN_PROBABILITY <= prob <= 1.0
