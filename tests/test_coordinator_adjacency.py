"""Tests for the coordinator-level adjacent-areas wiring (Phase 4).

Covers the runtime path that connects the pure math in
``data/adjacency.py`` to the coordinator tick: lagged-probability
snapshots, per-tick boost/modifier caches, application of the boost in
``Area.probability()``, decay-modifier propagation to entity decay, and
trajectory bookkeeping.
"""

from datetime import timedelta
from unittest.mock import patch

import pytest

from custom_components.area_occupancy.const import (
    ADJACENCY_BOOST_GAIN,
    ADJACENCY_DECAY_MODIFIER_GAIN,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.adjacency import (
    BoostContribution,
    apply_logit_boost,
)
from custom_components.area_occupancy.db.transitions import (
    LEVEL_1HOP_HOUR_OF_WEEK,
    TransitionLookupResult,
)
from custom_components.area_occupancy.time_utils import to_local
from custom_components.area_occupancy.utils import logit
from homeassistant.util import dt as dt_util

# ruff: noqa: SLF001


def _lookup_stub(probability: float):
    """Return a lookup function that always yields the given probability."""

    def lookup(db, entry_id, *, from_area, mid_area, to_area, hour_of_week):
        return TransitionLookupResult(
            probability=probability,
            level=LEVEL_1HOP_HOUR_OF_WEEK,
            observed_count=9.0,
            total_count=10.0,
        )

    return lookup


class TestLaggedProbabilities:
    """The tick must read last-tick state, not the in-progress recompute."""

    async def test_update_snapshots_previous_tick(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that lagged_probabilities reflects the pre-update data."""
        area_name = coordinator.get_area_names()[0]
        coordinator.data = {
            area_name: {"probability": 0.42, "occupied": True},
            "phantom_area": {"probability": 0.9, "occupied": True},
        }

        await coordinator.update()

        assert coordinator.lagged_probabilities == {
            area_name: 0.42,
            "phantom_area": 0.9,
        }

    async def test_first_tick_has_empty_lagged_state(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that the first tick (no previous data) yields an empty snapshot."""
        coordinator.data = None

        await coordinator.update()

        assert coordinator.lagged_probabilities == {}


class TestAdjacencyBoostWiring:
    """Boost computation and application in the tick."""

    async def test_boost_computed_when_trajectory_exists(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that a recent adjacent end event produces a fired boost."""
        area_name = coordinator.get_area_names()[0]
        now = dt_util.utcnow()
        # Another area just became unoccupied → 1-hop trajectory
        coordinator._trajectory_tracker.observe(
            "hallway", was_occupied=True, is_occupied=False, now=now
        )

        with (
            patch(
                "custom_components.area_occupancy.coordinator.build_adjacency_index",
                return_value={area_name: {"hallway"}, "hallway": {area_name}},
            ),
            patch(
                "custom_components.area_occupancy.coordinator.lookup_transition_probability",
                new=_lookup_stub(0.9),
            ),
        ):
            await coordinator.update()

        boost = coordinator.adjacency_boost_for(area_name)
        assert boost is not None
        assert boost.fired
        assert boost.trajectory_prev == "hallway"
        assert boost.raw_probability == 0.9
        assert boost.fallback_level == LEVEL_1HOP_HOUR_OF_WEEK
        assert boost.logit_contribution == pytest.approx(
            ADJACENCY_BOOST_GAIN * logit(0.9)
        )

    async def test_no_trajectory_no_boost(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that without recent adjacent activity no boost is cached."""
        area_name = coordinator.get_area_names()[0]

        await coordinator.update()

        assert coordinator.adjacency_boost_for(area_name) is None

    async def test_area_probability_applies_cached_boost(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that Area.probability() bends toward a cached boost."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)

        coordinator._adjacency_boosts = {}
        base = area.probability()

        boost = BoostContribution(fired=True, logit_contribution=2.0)
        coordinator._adjacency_boosts = {area_name: boost}
        boosted = area.probability()

        assert boosted > base
        assert boosted == pytest.approx(apply_logit_boost(base, boost))


class TestDecayModifierWiring:
    """Decay-modifier computation and propagation to entities."""

    async def test_modifier_propagates_to_entity_decay(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test that silent neighbours stretch every entity's half-life."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)

        # Neighbour is fully silent (no lagged data → 0.0) and the
        # household always exits through it (P=0.8):
        # silence = (1 - 0) * 0.8 = 0.8 → modifier = 1 + 0.75 * 0.8 = 1.6
        with (
            patch(
                "custom_components.area_occupancy.coordinator.build_adjacency_index",
                return_value={area_name: {"hallway"}, "hallway": {area_name}},
            ),
            patch(
                "custom_components.area_occupancy.coordinator.lookup_transition_probability",
                new=_lookup_stub(0.8),
            ),
        ):
            await coordinator_with_sensors.update()

        expected = 1.0 + ADJACENCY_DECAY_MODIFIER_GAIN * 0.8
        modifier = coordinator_with_sensors.adjacency_decay_modifier_for(area_name)
        assert modifier is not None
        assert modifier.fired
        assert modifier.decay_modifier == pytest.approx(expected)

        assert area.entities.entities, "fixture should provide entities"
        for entity in area.entities.entities.values():
            assert entity.decay.modifier_factor == pytest.approx(expected)

    async def test_active_neighbour_produces_no_stretch(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test that an occupied neighbour contributes no silence."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)
        coordinator_with_sensors.data = {
            area_name: {"probability": 0.5, "occupied": False},
            "hallway": {"probability": 1.0, "occupied": True},
        }

        with (
            patch(
                "custom_components.area_occupancy.coordinator.build_adjacency_index",
                return_value={area_name: {"hallway"}, "hallway": {area_name}},
            ),
            patch(
                "custom_components.area_occupancy.coordinator.lookup_transition_probability",
                new=_lookup_stub(0.8),
            ),
        ):
            await coordinator_with_sensors.update()

        modifier = coordinator_with_sensors.adjacency_decay_modifier_for(area_name)
        assert modifier is not None
        # silence = (1 - 1.0) * 0.8 = 0 → modifier stays 1.0
        assert modifier.decay_modifier == pytest.approx(1.0)
        for entity in area.entities.entities.values():
            assert entity.decay.modifier_factor == pytest.approx(1.0)

    async def test_no_neighbours_leaves_decay_untouched(
        self, coordinator_with_sensors: AreaOccupancyCoordinator
    ) -> None:
        """Test that without configured adjacency nothing is modified."""
        area_name = coordinator_with_sensors.get_area_names()[0]
        area = coordinator_with_sensors.get_area(area_name)

        await coordinator_with_sensors.update()

        assert coordinator_with_sensors.adjacency_decay_modifier_for(area_name) is None
        for entity in area.entities.entities.values():
            assert entity.decay.modifier_factor == pytest.approx(1.0)


class TestTrajectoryBookkeeping:
    """Trajectory observation and hour-of-week bucketing."""

    async def test_end_edge_recorded_during_update(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that an occupied→clear edge lands in the tracker."""
        area_name = coordinator.get_area_names()[0]
        # Previous tick: occupied. Current tick will compute a low
        # probability (no active evidence in the bare fixture), so the
        # area produces an end edge.
        coordinator.data = {area_name: {"probability": 0.99, "occupied": True}}

        await coordinator.update()

        recorded_areas = [
            name for name, _ in coordinator._trajectory_tracker.snapshot()
        ]
        assert area_name in recorded_areas

    def test_trajectory_hour_of_week_uses_local_time(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test the 0-167 bucket derives from local weekday and hour."""
        now = dt_util.utcnow().replace(microsecond=0)
        local = to_local(now)
        expected = local.weekday() * 24 + local.hour

        trajectory = coordinator.trajectory_for("anything", now=now)
        assert trajectory.hour_of_week == expected

    def test_trajectory_excludes_target_area(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that the target's own end events don't feed its trajectory."""
        now = dt_util.utcnow()
        tracker = coordinator._trajectory_tracker
        tracker.observe("bedroom", was_occupied=True, is_occupied=False, now=now)
        tracker.observe(
            "hallway",
            was_occupied=True,
            is_occupied=False,
            now=now + timedelta(seconds=1),
        )

        trajectory = coordinator.trajectory_for(
            "hallway", now=now + timedelta(seconds=2)
        )
        assert trajectory.prev_area == "bedroom"
        assert trajectory.prev_prev_area is None
