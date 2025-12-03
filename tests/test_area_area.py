"""Tests for Area class methods."""

from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest

from custom_components.area_occupancy.area.area import Area
from custom_components.area_occupancy.const import (
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator


# ruff: noqa: SLF001
class TestAreaMethods:
    """Test Area class methods."""

    def test_device_info(self, default_area: Area) -> None:
        """Test device_info method with area_id set."""
        device_info = default_area.device_info()

        assert device_info is not None
        # Device identifier should use area_id when available
        assert device_info["identifiers"] == {(DOMAIN, default_area.config.area_id)}
        assert device_info["name"] == default_area.config.name
        assert device_info["manufacturer"] == DEVICE_MANUFACTURER
        assert device_info["model"] == DEVICE_MODEL
        assert device_info["sw_version"] == DEVICE_SW_VERSION

    def test_device_info_property(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Test device_info property structure."""
        # device_info is now accessed directly from Area
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        device_info = area.device_info()

        assert "identifiers" in device_info
        assert "name" in device_info
        assert "manufacturer" in device_info
        assert "model" in device_info
        assert isinstance(device_info["identifiers"], set)
        assert isinstance(device_info["name"], str)

    def test_device_info_with_real_constants(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test device_info property with actual constant values."""
        # device_info is now accessed directly from Area
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        device_info = area.device_info()

        assert device_info.get("manufacturer") == DEVICE_MANUFACTURER
        assert device_info.get("model") == DEVICE_MODEL
        assert device_info.get("sw_version") == DEVICE_SW_VERSION

        identifiers = device_info.get("identifiers")
        assert identifiers is not None
        assert isinstance(identifiers, set)
        # In multi-area architecture, device_info uses area_id as identifier (stable even if area is renamed)
        expected_identifier = (DOMAIN, area.config.area_id)
        assert expected_identifier in identifiers, (
            f"Expected {expected_identifier} in {identifiers}"
        )

    def test_device_info_area_id_fallback(self, default_area: Area) -> None:
        """Test device_info method falls back to area_name when area_id is None."""
        # Set area_id to None to test fallback behavior
        original_area_id = default_area.config.area_id
        default_area.config.area_id = None

        device_info = default_area.device_info()

        assert device_info is not None
        # Device identifier should fallback to area_name when area_id is None
        # Note: Current implementation may not have this fallback - this test documents expected behavior
        expected_identifier = default_area.config.area_id or default_area.area_name
        assert device_info["identifiers"] == {(DOMAIN, expected_identifier)}
        assert device_info["name"] == default_area.config.name

        # Restore original area_id
        default_area.config.area_id = original_area_id

    def test_probability_with_entities(self, default_area: Area) -> None:
        """Test probability method with entities verifies meaningful calculation."""
        # Set prior
        default_area.prior.global_prior = 0.3
        default_area.prior._cached_time_prior = None

        # Test 1: With active motion sensor, probability should increase from prior
        mock_active_entity = Mock()
        mock_active_entity.evidence = True
        mock_active_entity.prob_given_true = 0.9
        mock_active_entity.prob_given_false = 0.1
        mock_active_entity.decay = Mock(decay_factor=1.0, is_decaying=False)
        mock_active_entity.decay_factor = 1.0
        type(mock_active_entity).weight = PropertyMock(return_value=0.85)

        default_area.entities._entities = {"binary_sensor.motion": mock_active_entity}
        prob_with_active = default_area.probability()
        assert prob_with_active > default_area.prior.value, (
            f"Active entity should increase probability from prior "
            f"({default_area.prior.value}) to {prob_with_active}"
        )

        # Test 2: With inactive entity, probability should decrease from prior
        mock_inactive_entity = Mock()
        mock_inactive_entity.evidence = False
        mock_inactive_entity.prob_given_true = 0.7
        mock_inactive_entity.prob_given_false = 0.3
        mock_inactive_entity.decay = Mock(decay_factor=1.0, is_decaying=False)
        mock_inactive_entity.decay_factor = 1.0
        type(mock_inactive_entity).weight = PropertyMock(return_value=0.7)

        default_area.entities._entities = {"media_player.tv": mock_inactive_entity}
        prob_with_inactive = default_area.probability()
        assert prob_with_inactive < default_area.prior.value, (
            f"Inactive entity should decrease probability from prior "
            f"({default_area.prior.value}) to {prob_with_inactive}"
        )

        # Test 3: Verify probability is in valid range
        assert 0.0 <= prob_with_active <= 1.0
        assert 0.0 <= prob_with_inactive <= 1.0

    def test_probability_no_entities(self, default_area: Area) -> None:
        """Test probability method with no entities."""
        default_area.entities._entities = {}
        prob = default_area.probability()
        assert prob == MIN_PROBABILITY

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

    @pytest.mark.parametrize(
        "prior_value", [0.45, 0.75], ids=["prior_0.45", "prior_0.75"]
    )
    def test_area_prior(self, default_area: Area, prior_value: float) -> None:
        """Test area_prior method returns prior.value."""
        # Set a known prior value
        default_area.prior.global_prior = prior_value
        default_area.prior._cached_time_prior = None

        # Verify area_prior returns prior.value
        area_prior = default_area.area_prior()
        expected_prior = default_area.prior.value
        assert area_prior == expected_prior

    @pytest.mark.asyncio
    async def test_run_prior_analysis(self, default_area: Area) -> None:
        """Test run_prior_analysis calls start_prior_analysis with correct parameters."""
        with patch(
            "custom_components.area_occupancy.area.area.start_prior_analysis",
            new_callable=AsyncMock,
        ) as mock_start_prior:
            await default_area.run_prior_analysis()

            # Verify start_prior_analysis was called with correct parameters
            mock_start_prior.assert_called_once_with(
                default_area.coordinator,
                default_area.area_name,
                default_area.prior,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "prior_is_none", [False, True], ids=["with_prior", "no_prior"]
    )
    async def test_async_cleanup(self, default_area: Area, prior_is_none: bool) -> None:
        """Test async_cleanup method cleans up all resources."""
        # Set prior to None if testing that case
        if prior_is_none:
            default_area._prior = None

        # Mock the cleanup methods
        with (
            patch.object(
                default_area.entities, "cleanup", new_callable=AsyncMock
            ) as mock_entities_cleanup,
            patch.object(default_area.purpose, "cleanup") as mock_purpose_cleanup,
        ):
            # Only mock prior.clear_cache if prior is not None
            if not prior_is_none:
                with patch.object(
                    default_area.prior, "clear_cache"
                ) as mock_clear_cache:
                    await default_area.async_cleanup()

                    # Verify all cleanup methods were called
                    mock_clear_cache.assert_called_once()
                    mock_entities_cleanup.assert_called_once()
                    mock_purpose_cleanup.assert_called_once()
            else:
                await default_area.async_cleanup()

                # Verify entities and purpose cleanup were called
                mock_entities_cleanup.assert_called_once()
                mock_purpose_cleanup.assert_called_once()

                # Prior cleanup should not be called since _prior is None
                # (no exception should be raised)

    def test_probability_real_calculation(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test probability() with real EntityManager and entities."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Get real probability calculation
        prob = area.probability()

        # Verify probability is in valid range
        assert 0.0 <= prob <= 1.0

        # Verify probability uses real entities from EntityManager
        # If entities exist, probability should be calculated from them
        if area.entities.entities:
            # With real entities, probability should be calculated (not just MIN_PROBABILITY)
            # unless all entities are inactive and prior is very low
            assert prob >= MIN_PROBABILITY

        # Verify probability changes when entities change state
        # (This tests that probability() actually uses the entities)
        if area.entities.entities:
            # Store original probability
            original_prob = prob

            # Change entity states and verify probability recalculates
            # Note: We can't easily change real entity states, but we can verify
            # that probability() is called and returns a value based on current state
            new_prob = area.probability()
            # Probability should be consistent for same state
            assert new_prob == original_prob

    def test_occupied_real_calculation(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test occupied() with real probability() calls."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        # Get real probability and threshold
        prob = area.probability()
        threshold = area.config.threshold

        # Verify occupied() returns correct value based on real probability
        occupied = area.occupied()

        # Verify occupied matches threshold comparison
        if prob >= threshold:
            assert occupied is True, (
                f"occupied() should be True when probability ({prob}) >= threshold ({threshold})"
            )
        else:
            assert occupied is False, (
                f"occupied() should be False when probability ({prob}) < threshold ({threshold})"
            )

        # Test threshold boundary: set threshold to current probability
        original_threshold = area.config.threshold
        area.config.threshold = prob
        occupied_at_threshold = area.occupied()
        # At threshold, occupied should be True (>= comparison)
        assert occupied_at_threshold is True

        # Restore original threshold
        area.config.threshold = original_threshold

    @pytest.mark.parametrize(
        (
            "entity_config",
            "prior_value",
            "threshold_value",
            "use_calculated_threshold",
            "expected_occupied",
            "description",
        ),
        [
            # Test case: probability >= threshold (True)
            (
                {
                    "evidence": True,
                    "prob_given_true": 0.9,
                    "prob_given_false": 0.1,
                    "weight": 0.85,
                    "entity_id": "binary_sensor.motion",
                },
                0.5,
                0.5,
                False,
                True,
                "probability_above_threshold",
            ),
            # Test case: probability < threshold (False)
            (
                {
                    "evidence": False,
                    "prob_given_true": 0.7,
                    "prob_given_false": 0.3,
                    "weight": 0.3,
                    "entity_id": "media_player.tv",
                },
                0.2,
                0.5,
                False,
                False,
                "probability_below_threshold",
            ),
            # Test case: probability == threshold (True)
            (
                {
                    "evidence": True,
                    "prob_given_true": 0.8,
                    "prob_given_false": 0.2,
                    "weight": 0.85,
                    "entity_id": "binary_sensor.motion",
                },
                0.5,
                0.5,
                True,  # Set threshold to calculated probability
                True,
                "probability_at_threshold",
            ),
        ],
    )
    def test_occupied(
        self,
        default_area: Area,
        entity_config: dict,
        prior_value: float,
        threshold_value: float,
        use_calculated_threshold: bool,
        expected_occupied: bool,
        description: str,
    ) -> None:
        """Test occupied method with different probability/threshold scenarios."""
        # Set up entity
        mock_entity = Mock()
        mock_entity.evidence = entity_config["evidence"]
        mock_entity.prob_given_true = entity_config["prob_given_true"]
        mock_entity.prob_given_false = entity_config["prob_given_false"]
        mock_entity.decay = Mock(decay_factor=1.0, is_decaying=False)
        mock_entity.decay_factor = 1.0
        type(mock_entity).weight = PropertyMock(return_value=entity_config["weight"])

        default_area.entities._entities = {entity_config["entity_id"]: mock_entity}
        default_area.prior.global_prior = prior_value
        default_area.prior._cached_time_prior = None

        # Set threshold
        if use_calculated_threshold:
            # Calculate probability first, then set threshold to match
            prob = default_area.probability()
            default_area.config.threshold = prob
        else:
            default_area.config.threshold = threshold_value
            prob = default_area.probability()

        # Verify occupied returns expected value
        assert default_area.occupied() == expected_occupied, (
            f"occupied() should return {expected_occupied} for scenario: {description} "
            f"(probability={prob}, threshold={default_area.config.threshold})"
        )
