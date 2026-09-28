"""Tests for the coordinator-level adjacent-areas wiring (Phase 4).

Covers the runtime path that connects the pure math in
``data/adjacency.py`` to the coordinator tick: lagged-probability
snapshots, per-tick boost/modifier caches, application of the boost in
``Area.probability()``, decay-modifier propagation to entity decay, and
trajectory bookkeeping.
"""

from datetime import timedelta
from itertools import pairwise
from typing import Any
from unittest.mock import patch

import pytest
import sqlalchemy as sa

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
    LEVEL_2HOP_HOUR_OF_WEEK,
)
from custom_components.area_occupancy.time_utils import to_local
from custom_components.area_occupancy.utils import logit
from homeassistant.util import dt as dt_util
from tests.conftest import create_test_area

# ruff: noqa: SLF001

# A motion sensor configured on the ``coordinator`` fixture's area.
FIXTURE_MOTION = "binary_sensor.motion_sensor_1"


def _motion_on(coordinator: AreaOccupancyCoordinator, area_name: str) -> None:
    """Turn a motion sensor on and run the evidence gate, as the listener does."""
    coordinator.hass.states.async_set(FIXTURE_MOTION, "on")
    coordinator.get_area(area_name).entities.get_entity(
        FIXTURE_MOTION
    ).has_new_evidence()


def _seed_adjacency(
    coordinator: AreaOccupancyCoordinator,
    adjacency: dict[str, set[str]],
    counts: dict[tuple[str, str], dict[str, float]],
) -> None:
    """Seed adjacency rows and identical transition counts in every hour.

    ``counts`` maps ``(from_area, mid_area)`` to ``{to_area: count}``.
    Filling all 168 hour-of-week buckets makes the exact-hour fallback
    levels fire whatever hour the test runs in.
    """
    db = coordinator.db
    with db.get_session() as session:
        for area_name, neighbours in adjacency.items():
            for neighbour in neighbours:
                session.add(
                    db.AreaRelationships(
                        entry_id=coordinator.entry_id,
                        area_name=area_name,
                        related_area_name=neighbour,
                        relationship_type="adjacent",
                        influence_weight=0.3,
                    )
                )
        for (from_area, mid_area), to_counts in counts.items():
            for hour in range(168):
                for to_area, count in to_counts.items():
                    session.add(
                        db.AreaTransitions(
                            entry_id=coordinator.entry_id,
                            from_area=from_area,
                            mid_area=mid_area,
                            to_area=to_area,
                            hour_of_week=hour,
                            count=count,
                        )
                    )
        session.commit()


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

    async def _seed_hallway_exits(
        self, coordinator: AreaOccupancyCoordinator, area_name: str
    ) -> None:
        # 9 of 10 exits from the hallway went to this area → P = 0.9 at
        # the 1-hop exact-hour level.
        _seed_adjacency(
            coordinator,
            {area_name: {"hallway"}, "hallway": {area_name}},
            {("hallway", ""): {area_name: 9.0, "kitchen": 1.0}},
        )
        await coordinator.async_load_adjacency_snapshot()

    async def test_boost_computed_when_area_entered_after_departure(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that leaving the hallway, then this area's motion, boosts it."""
        area_name = coordinator.get_area_names()[0]
        await self._seed_hallway_exits(coordinator, area_name)
        now = dt_util.utcnow()
        # The hallway's sensors just went quiet → 1-hop trajectory
        coordinator._trajectory_tracker.observe(
            "hallway", was_present=True, is_present=False, now=now
        )
        _motion_on(coordinator, area_name)

        await coordinator.update()

        boost = coordinator.adjacency_boost_for(area_name)
        assert boost is not None
        assert boost.fired
        assert boost.trajectory_prev == "hallway"
        assert boost.raw_probability == pytest.approx(0.9)
        assert boost.fallback_level == LEVEL_1HOP_HOUR_OF_WEEK
        assert boost.observed_count == pytest.approx(9.0)
        assert boost.total_count == pytest.approx(10.0)
        assert boost.logit_contribution == pytest.approx(
            ADJACENCY_BOOST_GAIN * logit(0.9)
        )

    async def test_no_boost_until_the_area_is_entered(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that a departure alone doesn't boost the likely next area.

        Nothing has confirmed anyone went there, and boosting an area
        that is still decaying from earlier is what lifted it back over
        the threshold in an empty house.
        """
        area_name = coordinator.get_area_names()[0]
        await self._seed_hallway_exits(coordinator, area_name)
        coordinator._trajectory_tracker.observe(
            "hallway", was_present=True, is_present=False, now=dt_util.utcnow()
        )

        await coordinator.update()

        assert coordinator.adjacency_boost_for(area_name) is None

    async def test_no_boost_when_sensor_was_already_active(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that evidence from before the departure isn't an arrival."""
        area_name = coordinator.get_area_names()[0]
        await self._seed_hallway_exits(coordinator, area_name)
        _motion_on(coordinator, area_name)
        coordinator._trajectory_tracker.observe(
            "hallway",
            was_present=True,
            is_present=False,
            now=dt_util.utcnow() + timedelta(seconds=1),
        )

        await coordinator.update()

        assert coordinator.adjacency_boost_for(area_name) is None

    async def test_no_boost_when_arrival_is_outside_transition_window(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that motion 2 minutes after the departure isn't that move.

        The learner only records a transition when the next area starts
        within ``ADJACENCY_TRANSITION_WINDOW_S`` (60 s) of the previous
        one ending; the departure is still inside the 300 s trajectory
        window.
        """
        area_name = coordinator.get_area_names()[0]
        await self._seed_hallway_exits(coordinator, area_name)
        coordinator._trajectory_tracker.observe(
            "hallway",
            was_present=True,
            is_present=False,
            now=dt_util.utcnow() - timedelta(seconds=120),
        )
        _motion_on(coordinator, area_name)

        await coordinator.update()

        assert coordinator.adjacency_boost_for(area_name) is None

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

        # Neighbour is fully silent (no lagged data → 0.0) and 8 of 10
        # exits from this area go through it (P=0.8):
        # silence = (1 - 0) * 0.8 = 0.8 → modifier = 1 + 0.75 * 0.8 = 1.6
        _seed_adjacency(
            coordinator_with_sensors,
            {area_name: {"hallway"}, "hallway": {area_name}},
            {(area_name, ""): {"hallway": 8.0, "kitchen": 2.0}},
        )
        await coordinator_with_sensors.async_load_adjacency_snapshot()

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

        _seed_adjacency(
            coordinator_with_sensors,
            {area_name: {"hallway"}, "hallway": {area_name}},
            {(area_name, ""): {"hallway": 8.0, "kitchen": 2.0}},
        )
        await coordinator_with_sensors.async_load_adjacency_snapshot()

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

    async def test_departure_recorded_when_sensors_go_quiet(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that the area's motion going off lands in the tracker."""
        area_name = coordinator.get_area_names()[0]
        _motion_on(coordinator, area_name)
        await coordinator.update()
        assert coordinator._trajectory_tracker.snapshot() == []

        coordinator.hass.states.async_set(FIXTURE_MOTION, "off")
        coordinator.get_area(area_name).entities.get_entity(
            FIXTURE_MOTION
        ).has_new_evidence()
        await coordinator.update()

        assert [name for name, _ in coordinator._trajectory_tracker.snapshot()] == [
            area_name
        ]

    async def test_probability_drop_is_not_a_departure(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that falling below the threshold doesn't record a departure.

        The previous tick was occupied and this one computes a low
        probability, but no sensor changed. A decay tail running out is
        not someone leaving; the learner records ends of sensor activity.
        """
        area_name = coordinator.get_area_names()[0]
        coordinator.data = {area_name: {"probability": 0.99, "occupied": True}}

        result = await coordinator.update()

        assert result[area_name]["occupied"] is False
        assert coordinator._trajectory_tracker.snapshot() == []

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
        tracker.observe("bedroom", was_present=True, is_present=False, now=now)
        tracker.observe(
            "hallway",
            was_present=True,
            is_present=False,
            now=now + timedelta(seconds=1),
        )

        trajectory = coordinator.trajectory_for(
            "hallway", now=now + timedelta(seconds=2)
        )
        assert trajectory.prev_area == "bedroom"
        assert trajectory.prev_prev_area is None


class TestAdjacencySnapshot:
    """The refresh path reads adjacency data from memory, never SQLite."""

    async def test_update_issues_no_sql_and_no_executor_jobs(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test a refresh stays off the database and the executor pool.

        ``update`` runs on every sensor state change. It used to open a
        SQLite connection per fallback level per adjacency lookup inside
        an executor job, which delayed occupancy updates by seconds on
        slow hosts. With adjacency configured, a 2-hop trajectory (so
        every lookup level can be walked) and a learned prior, the whole
        refresh must now complete without a single SQL statement.
        """
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        _seed_adjacency(
            coordinator,
            {area_name: {"hallway"}, "hallway": {area_name, "kitchen"}},
            {("kitchen", "hallway"): {area_name: 6.0, "bedroom": 4.0}},
        )
        await coordinator.async_load_adjacency_snapshot()
        # A learned global prior makes probability() consult the time-prior
        # cache; load_data warms it, as it does at setup.
        area.prior.set_global_prior(0.3)
        await coordinator.db.load_data()

        now = dt_util.utcnow()
        tracker = coordinator._trajectory_tracker
        tracker.observe(
            "kitchen",
            was_present=True,
            is_present=False,
            now=now - timedelta(seconds=2),
        )
        tracker.observe(
            "hallway",
            was_present=True,
            is_present=False,
            now=now - timedelta(seconds=1),
        )
        # This area's motion fires after the hallway departure, so the
        # boost (and every lookup level behind it) is computed.
        _motion_on(coordinator, area_name)

        statements: list[str] = []

        def _record(_conn, _cursor, statement, *_args) -> None:
            statements.append(statement)

        engine = coordinator.db.engine
        sa.event.listen(engine, "before_cursor_execute", _record)
        try:
            with patch.object(
                coordinator.hass,
                "async_add_executor_job",
                side_effect=AssertionError("refresh used the executor"),
            ):
                await coordinator.update()
        finally:
            sa.event.remove(engine, "before_cursor_execute", _record)

        assert statements == []
        boost = coordinator.adjacency_boost_for(area_name)
        assert boost is not None
        assert boost.trajectory_prev_prev == "kitchen"
        assert boost.fallback_level == LEVEL_2HOP_HOUR_OF_WEEK
        assert boost.raw_probability == pytest.approx(0.6)

    async def test_failed_reload_keeps_previous_snapshot(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test a database error on reload doesn't drop learned transitions."""
        area_name = coordinator.get_area_names()[0]
        _seed_adjacency(
            coordinator,
            {area_name: {"hallway"}, "hallway": {area_name}},
            {("hallway", ""): {area_name: 9.0, "kitchen": 1.0}},
        )
        await coordinator.async_load_adjacency_snapshot()
        loaded = coordinator._adjacency_snapshot
        assert loaded.adjacency_index == {
            area_name: {"hallway"},
            "hallway": {area_name},
        }

        # load_adjacency_snapshot returns None on SQLAlchemyError.
        with patch(
            "custom_components.area_occupancy.coordinator.load_adjacency_snapshot",
            return_value=None,
        ):
            await coordinator.async_load_adjacency_snapshot()

        assert coordinator._adjacency_snapshot is loaded


class TestHouse:
    """A small house driven through the real tick, as the decay timer does.

    Bedroom – Hallway – Kitchen in a line, one motion sensor each, the
    default purpose (social), a 50 % threshold, a learned prior of 0.1,
    and learned 1-hop transitions between neighbours (the learner only
    records adjacent pairs).
    """

    AREAS = ("Bedroom", "Hallway", "Kitchen")
    ADJACENCY = {
        "Bedroom": {"Hallway"},
        "Hallway": {"Bedroom", "Kitchen"},
        "Kitchen": {"Hallway"},
    }
    COUNTS = {
        ("Bedroom", ""): {"Hallway": 10.0},
        ("Hallway", ""): {"Kitchen": 8.0, "Bedroom": 2.0},
        ("Kitchen", ""): {"Hallway": 10.0},
    }

    @staticmethod
    def _motion(name: str) -> str:
        return f"binary_sensor.{name.lower()}_motion"

    async def _house(
        self, coordinator: AreaOccupancyCoordinator
    ) -> AreaOccupancyCoordinator:
        """Turn the shared fixture coordinator into the three-area house."""
        # Drop the fixture's own area: its dozen unset sensors aren't part
        # of the house.
        coordinator.areas.clear()
        for name in self.AREAS:
            area = create_test_area(
                coordinator,
                area_name=name,
                entity_ids=[self._motion(name)],
                threshold=0.5,
            )
            area.prior.set_global_prior(0.1)
            area.prior._cached_time_priors = {}
            coordinator.hass.states.async_set(self._motion(name), "off")
        _seed_adjacency(coordinator, self.ADJACENCY, self.COUNTS)
        await coordinator.async_load_adjacency_snapshot()
        return coordinator

    async def _run(
        self,
        house: AreaOccupancyCoordinator,
        events: dict[int, list[tuple[str, str]]],
        until: int,
        on_tick: Any = None,
    ) -> list[tuple[int, dict[str, tuple[float, bool]]]]:
        """Apply sensor events and tick every 10 s; return each tick's state."""
        clock = {"now": dt_util.utcnow().replace(microsecond=0)}
        history = []
        with patch("homeassistant.util.dt.utcnow", side_effect=lambda: clock["now"]):
            house.data = await house.update()
            for t in range(0, until + 1, 10):
                for name, state in events.get(t, []):
                    house.hass.states.async_set(self._motion(name), state)
                    house.get_area(name).entities.get_entity(
                        self._motion(name)
                    ).has_new_evidence()
                for area in house.areas.values():
                    area.tick_decay()
                house.data = await house.update()
                if on_tick is not None:
                    on_tick(t, house)
                history.append(
                    (
                        t,
                        {
                            name: (entry["probability"], entry["occupied"])
                            for name, entry in house.data.items()
                        },
                    )
                )
                clock["now"] += timedelta(seconds=10)
        return history

    # Someone walks Bedroom → Hallway → Kitchen and leaves the house from
    # the Kitchen at t = 600 s.
    WALK = {
        0: [("Bedroom", "on")],
        120: [("Bedroom", "off"), ("Hallway", "on")],
        150: [("Hallway", "off"), ("Kitchen", "on")],
        600: [("Kitchen", "off")],
    }

    async def test_entering_the_likely_next_area_boosts_it(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that leaving the Hallway and entering the Kitchen boosts it.

        The boost is 0.5 × logit(P(Kitchen | Hallway) = 0.8), applied on the
        tick the Kitchen's motion fires. When departures were only recorded
        once an area's probability fell below the threshold, the Hallway
        hadn't "ended" yet at that point, so the Kitchen got nothing.
        """
        house = await self._house(coordinator)
        boosts: dict[int, BoostContribution | None] = {}

        def _record_kitchen_boost(t: int, coordinator: AreaOccupancyCoordinator):
            boosts[t] = coordinator.adjacency_boost_for("Kitchen")

        await self._run(house, self.WALK, until=600, on_tick=_record_kitchen_boost)

        assert boosts[140] is None
        assert boosts[150] is not None
        assert boosts[150].trajectory_prev == "Hallway"
        assert boosts[150].logit_contribution == pytest.approx(
            ADJACENCY_BOOST_GAIN * logit(0.8)
        )
        # Once the Kitchen's motion stops, the boost stops with it.
        assert boosts[600] is None

    async def test_empty_house_only_decays(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that after everyone leaves, no probability rises or switches on.

        Each area's probability dropping below the threshold used to count
        as a departure and boost its neighbours for five minutes (up to
        0.5 × logit(0.99) = +2.3 logits here), pushing areas that were
        still decaying back over the threshold.
        """
        house = await self._house(coordinator)

        history = await self._run(house, self.WALK, until=600 + 2400)

        for name in self.AREAS:
            assert any(state[name][1] for t, state in history if t < 600)
        after = [state for t, state in history if t >= 600]
        for name in self.AREAS:
            probabilities = [state[name][0] for state in after]
            occupied = [state[name][1] for state in after]
            rises = [
                (before, now)
                for before, now in pairwise(probabilities)
                if now > before + 1e-9
            ]
            assert rises == [], f"{name} rose after the house emptied: {rises[:3]}"
            assert not any(now and not before for before, now in pairwise(occupied)), (
                f"{name} switched back on after the house emptied"
            )
            assert occupied[-1] is False

    async def test_departure_elsewhere_leaves_an_occupied_area_alone(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Test that the Bedroom emptying doesn't switch the Kitchen off.

        One person stays in the Kitchen while the Bedroom's motion stops at
        t = 60 s. P(Kitchen | Bedroom) is a learned 0, since the Bedroom's
        only neighbour is the Hallway, and that used to add
        0.5 × logit(0.01) = -2.3 logits to the Kitchen for five minutes
        once the Bedroom decayed below the threshold: its active motion
        sensor then read about 0.23.
        """
        house = await self._house(coordinator)

        history = await self._run(
            house,
            {0: [("Kitchen", "on"), ("Bedroom", "on")], 60: [("Bedroom", "off")]},
            until=1200,
        )

        kitchen = [state["Kitchen"] for t, state in history]
        assert all(occupied for _, occupied in kitchen)
        probabilities = [probability for probability, _ in kitchen]
        assert max(probabilities) - min(probabilities) < 1e-9
