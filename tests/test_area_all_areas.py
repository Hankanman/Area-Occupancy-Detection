"""Tests for AllAreas aggregation class."""

from unittest.mock import MagicMock, patch

import pytest

from custom_components.area_occupancy.area.all_areas import AllAreas
from custom_components.area_occupancy.const import (
    ALL_AREAS_IDENTIFIER,
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    MIN_PROBABILITY,
)


class TestAllAreas:
    """Test AllAreas aggregation class."""

    def test_device_info(self, coordinator) -> None:
        """Test device_info method returns complete device information."""
        all_areas = AllAreas(coordinator)
        device_info = all_areas.device_info()

        assert device_info is not None
        assert device_info["identifiers"] == {("area_occupancy", ALL_AREAS_IDENTIFIER)}
        assert device_info["name"] == "All Areas"
        assert device_info["manufacturer"] == DEVICE_MANUFACTURER
        assert device_info["model"] == DEVICE_MODEL
        assert device_info["sw_version"] == DEVICE_SW_VERSION

    @pytest.mark.parametrize(
        ("method_name", "value1", "value2", "expected_average"),
        [
            ("probability", 0.3, 0.7, 0.5),
            ("area_prior", 0.2, 0.8, 0.5),
            ("decay", 0.4, 0.6, 0.5),
        ],
    )
    def test_average_calculation(
        self,
        coordinator,
        method_name: str,
        value1: float,
        value2: float,
        expected_average: float,
    ) -> None:
        """Test aggregation methods average values across all areas."""
        all_areas = AllAreas(coordinator)

        area1 = MagicMock()
        setattr(area1, method_name, MagicMock(return_value=value1))
        area2 = MagicMock()
        setattr(area2, method_name, MagicMock(return_value=value2))
        with patch.dict(
            coordinator.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            result = getattr(all_areas, method_name)()
            assert result == expected_average

    def test_occupied_any_area(self, coordinator) -> None:
        """Test occupied returns True if ANY area is occupied."""
        all_areas = AllAreas(coordinator)

        area1 = MagicMock()
        area1.occupied.return_value = False
        area2 = MagicMock()
        area2.occupied.return_value = True
        with patch.dict(
            coordinator.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            assert all_areas.occupied() is True

    def test_occupied_no_areas_occupied(self, coordinator) -> None:
        """Test occupied returns False if no areas are occupied."""
        all_areas = AllAreas(coordinator)

        area1 = MagicMock()
        area1.occupied.return_value = False
        area2 = MagicMock()
        area2.occupied.return_value = False
        with patch.dict(
            coordinator.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            assert all_areas.occupied() is False

    @pytest.mark.parametrize(
        ("method_name", "min_bound", "max_bound"),
        [
            ("probability", MIN_PROBABILITY, 1.0),
            ("area_prior", MIN_PROBABILITY, 1.0),
            ("decay", 0.0, 1.0),
        ],
    )
    def test_clamps_to_bounds(
        self,
        coordinator,
        method_name: str,
        min_bound: float,
        max_bound: float,
    ) -> None:
        """Test methods clamp out-of-bounds values correctly."""
        all_areas = AllAreas(coordinator)

        # Test clamping to min_bound (average below minimum)
        area1 = MagicMock()
        setattr(area1, method_name, MagicMock(return_value=-0.5))
        area2 = MagicMock()
        setattr(area2, method_name, MagicMock(return_value=-0.3))
        with patch.dict(
            coordinator.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            result = getattr(all_areas, method_name)()
            # Average of -0.5 and -0.3 = -0.4, should clamp to min_bound
            assert result == min_bound

        # Test clamping to max_bound (average above maximum)
        area3 = MagicMock()
        setattr(area3, method_name, MagicMock(return_value=1.5))
        area4 = MagicMock()
        setattr(area4, method_name, MagicMock(return_value=1.2))
        with patch.dict(
            coordinator.areas,
            {"Area3": area3, "Area4": area4},
            clear=True,
        ):
            result = getattr(all_areas, method_name)()
            # Average of 1.5 and 1.2 = 1.35, should clamp to max_bound
            assert result == max_bound

    @pytest.mark.parametrize(
        ("method_name", "expected_default"),
        [
            ("probability", MIN_PROBABILITY),
            ("area_prior", MIN_PROBABILITY),
            ("decay", 1.0),
        ],
    )
    def test_empty_areas(
        self, coordinator, method_name: str, expected_default: float
    ) -> None:
        """Test methods return safe defaults when no areas exist."""
        all_areas = AllAreas(coordinator)

        with patch.dict(coordinator.areas, {}, clear=True):
            result = getattr(all_areas, method_name)()
            assert result == expected_default

    @pytest.mark.parametrize(
        ("method_name", "single_value", "expected_result"),
        [
            ("probability", 0.7, 0.7),
            ("area_prior", 0.6, 0.6),
            ("decay", 0.8, 0.8),
        ],
    )
    def test_single_area(
        self,
        coordinator,
        method_name: str,
        single_value: float,
        expected_result: float,
    ) -> None:
        """Test methods with only one area return that area's value."""
        all_areas = AllAreas(coordinator)

        area1 = MagicMock()
        setattr(area1, method_name, MagicMock(return_value=single_value))
        with patch.dict(
            coordinator.areas,
            {"Area1": area1},
            clear=True,
        ):
            result = getattr(all_areas, method_name)()
            # With single area, should return that area's value
            assert result == expected_result

    def test_occupied_empty_areas(self, coordinator) -> None:
        """Test occupied returns False when no areas exist."""
        all_areas = AllAreas(coordinator)

        with patch.dict(coordinator.areas, {}, clear=True):
            # any() on empty iterable returns False
            assert all_areas.occupied() is False

    @pytest.mark.parametrize(
        ("method_name", "boundary_value", "expected_result"),
        [
            ("probability", MIN_PROBABILITY, MIN_PROBABILITY),
            ("probability", 1.0, 1.0),
            ("area_prior", MIN_PROBABILITY, MIN_PROBABILITY),
            ("area_prior", 1.0, 1.0),
            ("decay", 0.0, 0.0),
            ("decay", 1.0, 1.0),
        ],
    )
    def test_boundary_values(
        self,
        coordinator,
        method_name: str,
        boundary_value: float,
        expected_result: float,
    ) -> None:
        """Test methods with boundary values return expected results."""
        all_areas = AllAreas(coordinator)

        # Test with all areas at boundary value
        area1 = MagicMock()
        setattr(area1, method_name, MagicMock(return_value=boundary_value))
        area2 = MagicMock()
        setattr(area2, method_name, MagicMock(return_value=boundary_value))
        with patch.dict(
            coordinator.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            result = getattr(all_areas, method_name)()
            assert result == expected_result

    @pytest.mark.parametrize(
        ("method_name", "value1", "value2", "expected_average"),
        [
            ("probability", MIN_PROBABILITY, 1.0, (MIN_PROBABILITY + 1.0) / 2.0),
            ("area_prior", MIN_PROBABILITY, 1.0, (MIN_PROBABILITY + 1.0) / 2.0),
            ("decay", 0.0, 1.0, 0.5),
        ],
    )
    def test_boundary_values_mixed(
        self,
        coordinator,
        method_name: str,
        value1: float,
        value2: float,
        expected_average: float,
    ) -> None:
        """Test methods with mixed boundary values return average."""
        all_areas = AllAreas(coordinator)

        area1 = MagicMock()
        setattr(area1, method_name, MagicMock(return_value=value1))
        area2 = MagicMock()
        setattr(area2, method_name, MagicMock(return_value=value2))
        with patch.dict(
            coordinator.areas,
            {"Area1": area1, "Area2": area2},
            clear=True,
        ):
            result = getattr(all_areas, method_name)()
            assert result == expected_average
