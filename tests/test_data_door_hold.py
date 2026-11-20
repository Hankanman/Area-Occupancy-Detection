"""Tests for the closed-door hold (#558)."""

from datetime import UTC, datetime, timedelta

import pytest

from custom_components.area_occupancy.config_flow import _closed_door_hold_default
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.door_hold import (
    DoorHold,
    DoorHoldState,
    hold_logit,
)
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.utils import logit, sigmoid

T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=UTC)
WINDOW = 300.0
MAX_HOLD = 3600.0
HALF_LIFE = 450.0


def _observe(
    hold: DoorHold,
    seconds: float,
    *,
    doors_open: bool,
    motion_on: bool,
    motion_window: float = WINDOW,
    presence_on: bool = False,
) -> None:
    hold.observe(
        T0 + timedelta(seconds=seconds),
        doors_open=doors_open,
        motion_on=motion_on,
        motion_window=motion_window,
        max_hold=MAX_HOLD,
        fade_half_life=HALF_LIFE,
        presence_on=presence_on,
    )


# Each sequence: (name, steps, trace). A step is (seconds, door_open, motion_on)
# or (seconds, door_open, motion_on, presence_on); the trace is one character per
# step, ``H`` where the area is held after that observation and ``-`` where not.
# Motion window 90 s (the area's motion timeout). A closed door says nothing
# about who is behind it, so the hold arms on an arrival or a stronger signal
# and prefers a lost hold to a false one.
DOOR_SEQUENCES = [
    (
        "leave: PIR still on after the door shuts does not arm",
        [
            (0, 0, 1),
            (10, 1, 1),
            (20, 0, 1),
            (30, 0, 1),
            (40, 0, 1),
            (50, 0, 1),
            (60, 0, 0),
            (70, 0, 0),
        ],
        "H-------",
    ),
    (
        "arrive: motion starts while the door is open, then it shuts",
        [
            (0, 0, 0),
            (10, 1, 0),
            (20, 1, 1),
            (30, 0, 1),
            (40, 0, 1),
            (50, 0, 0),
            (200, 0, 0),
        ],
        "---HHHH",
    ),
    (
        "arrive and sit still after shutting the door",
        [(0, 0, 0), (10, 1, 0), (20, 1, 1), (25, 0, 1), (30, 0, 0), (200, 0, 0)],
        "---HHH",
    ),
    (
        "short peek from inside, then still: a lost hold, not a false one",
        [(0, 0, 1), (10, 1, 1), (12, 0, 1), (20, 0, 0), (200, 0, 0)],
        "H----",
    ),
    (
        "longer reopen from inside, then still: lost as well",
        [(0, 0, 1), (10, 1, 1), (20, 0, 1), (30, 0, 0), (200, 0, 0)],
        "H----",
    ),
    (
        "quick exit with the PIR trailing: not held",
        [
            (0, 0, 1),
            (10, 1, 1),
            (12, 0, 1),
            (22, 0, 1),
            (32, 0, 1),
            (42, 0, 0),
            (52, 0, 0),
        ],
        "H------",
    ),
    (
        "a stronger signal (mmWave, media...) confirms: held",
        [
            (0, 0, 1),
            (10, 1, 1),
            (20, 0, 1, 1),
            (30, 0, 1, 1),
            (40, 0, 0, 1),
            (200, 0, 0, 1),
        ],
        "H-HHHH",
    ),
    (
        "door cycles after the PIR lost everyone: nothing to hold",
        [(0, 0, 1), (5, 0, 0), (35, 1, 0), (45, 0, 0), (55, 0, 0)],
        "HH---",
    ),
    (
        "door open at start-up, walk in and shut it behind: held",
        [(0, 1, 0), (10, 1, 1), (20, 0, 1), (30, 0, 1), (60, 0, 0), (200, 0, 0)],
        "--HHHH",
    ),
    (
        "ambiguous close, PIR clears, then fires again inside: held",
        [
            (0, 0, 1),
            (10, 1, 1),
            (20, 0, 1),
            (30, 0, 1),
            (60, 0, 0),
            (100, 0, 0),
            (120, 0, 1),
            (130, 0, 0),
        ],
        "H-----HH",
    ),
    (
        "motion started more than a timeout before the door shuts: not an arrival",
        [(0, 0, 0), (10, 1, 0), (20, 1, 1), (30, 1, 0), (200, 0, 0), (210, 0, 0)],
        "------",
    ),
]


class TestDoorCycles:
    """The hold tells arrivals from departures using only what it can see."""

    @pytest.mark.parametrize(
        ("steps", "trace"),
        [pytest.param(steps, trace, id=name) for name, steps, trace in DOOR_SEQUENCES],
    )
    def test_sequence(self, steps: list[tuple], trace: str) -> None:
        hold = DoorHold()
        seen = ""
        for seconds, door_open, motion, *rest in steps:
            _observe(
                hold,
                seconds,
                doors_open=bool(door_open),
                motion_on=bool(motion),
                motion_window=90.0,
                presence_on=bool(rest[0]) if rest else False,
            )
            seen += "H" if hold.held else "-"
        assert seen == trace

    def test_motion_window_is_the_areas_timeout(self) -> None:
        """Motion that began 46 s before the door shut is an arrival at 90 s only."""
        for window, held in ((90.0, True), (45.0, False)):
            hold = DoorHold()
            _observe(hold, 0, doors_open=True, motion_on=False, motion_window=window)
            _observe(hold, 10, doors_open=True, motion_on=True, motion_window=window)
            _observe(hold, 56, doors_open=False, motion_on=False, motion_window=window)
            assert hold.held is held


def _floor(hold: DoorHold, seconds: float, *, prior: float = 0.028) -> float | None:
    return hold.floor_logit(
        T0 + timedelta(seconds=seconds),
        bias=logit(prior),
        threshold=0.5,
        fade_half_life=HALF_LIFE,
    )


class TestArming:
    """Basic arming and release."""

    def test_motion_behind_closed_door_holds_at_75(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=True)
        _observe(hold, 600, doors_open=False, motion_on=False)
        assert hold.held
        assert sigmoid(_floor(hold, 600)) == pytest.approx(0.75)

    def test_door_closed_soon_after_motion_holds(self) -> None:
        """Went in, motion lost them, then shut the door: held."""
        hold = DoorHold()
        _observe(hold, 0, doors_open=True, motion_on=True)
        _observe(hold, 20, doors_open=True, motion_on=False)
        _observe(hold, 200, doors_open=False, motion_on=False)
        assert hold.held

    def test_door_closed_long_after_motion_does_not_hold(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=True, motion_on=True)
        _observe(hold, 20, doors_open=True, motion_on=False)
        _observe(hold, 20 + WINDOW + 1, doors_open=False, motion_on=False)
        assert not hold.held
        assert _floor(hold, 400) is None

    def test_closed_door_without_motion_does_not_hold(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=False)
        _observe(hold, 10, doors_open=False, motion_on=False)
        assert not hold.held

    def test_first_observation_closed_is_not_a_closing(self) -> None:
        """After a restart with no state, a shut door isn't 'just closed'."""
        hold = DoorHold(DoorHoldState())
        _observe(hold, 10, doors_open=False, motion_on=False)
        assert not hold.held

    def test_opening_releases(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=True)
        _observe(hold, 300, doors_open=True, motion_on=False)
        assert not hold.held
        assert _floor(hold, 300) is None


class TestExpiry:
    def test_hold_counts_from_the_last_motion(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=True)
        _observe(hold, 1800, doors_open=False, motion_on=True)
        _observe(hold, 1800 + MAX_HOLD - 1, doors_open=False, motion_on=False)
        assert hold.held

    def test_fades_over_the_half_life_after_max_hold(self) -> None:
        """Hand-computed: prior 2.8%, half-life 450 s.

        b = logit(0.028) = -3.547151, hold = logit(0.75) = 1.098612.
        One half-life into the fade: b + (hold - b) / 2 = -1.224270,
        sigmoid = 0.227186.
        """
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=True)
        _observe(hold, MAX_HOLD + 100, doors_open=False, motion_on=False)
        assert not hold.held
        # The fade runs from when the hold ran out, not from this tick.
        assert hold.state.fade_from == T0 + timedelta(seconds=MAX_HOLD)
        floor = _floor(hold, MAX_HOLD + HALF_LIFE)
        assert floor == pytest.approx(-1.224270, abs=1e-6)
        assert sigmoid(floor) == pytest.approx(0.227186, abs=1e-6)

    def test_fade_is_dropped_after_six_half_lives(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=True)
        _observe(hold, MAX_HOLD + 1, doors_open=False, motion_on=False)
        _observe(hold, MAX_HOLD + 6 * HALF_LIFE, doors_open=False, motion_on=False)
        assert hold.state.fade_from is None
        assert _floor(hold, MAX_HOLD + 6 * HALF_LIFE) is None

    def test_zero_max_hold_never_expires(self) -> None:
        """Wasp in Box's maximum duration of 0 means no limit."""
        hold = DoorHold()
        hold.observe(
            T0,
            doors_open=False,
            motion_on=True,
            motion_window=WINDOW,
            max_hold=0,
            fade_half_life=HALF_LIFE,
        )
        hold.observe(
            T0 + timedelta(days=1),
            doors_open=False,
            motion_on=False,
            motion_window=WINDOW,
            max_hold=0,
            fade_half_life=HALF_LIFE,
        )
        assert hold.held

    def test_motion_during_fade_rearms(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=True)
        _observe(hold, MAX_HOLD + 1, doors_open=False, motion_on=False)
        _observe(hold, MAX_HOLD + 60, doors_open=False, motion_on=True)
        assert hold.held
        assert hold.state.fade_from is None


class TestFloorLevel:
    def test_threshold_above_75_is_cleared(self) -> None:
        """Hand-computed: threshold 0.8 -> logit(0.8) + 0.05 = 1.436294."""
        assert hold_logit(0.8) == pytest.approx(1.436294, abs=1e-6)
        assert sigmoid(hold_logit(0.8)) > 0.8

    def test_threshold_below_75_holds_at_75(self) -> None:
        assert hold_logit(0.5) == pytest.approx(logit(0.75))


class TestState:
    def test_round_trip(self) -> None:
        hold = DoorHold()
        _observe(hold, 0, doors_open=False, motion_on=True)
        assert DoorHoldState.from_dict(hold.state.to_dict()) == hold.state

    def test_malformed_falls_back_to_empty(self) -> None:
        assert DoorHoldState.from_dict({"held_from": "nonsense"}) == DoorHoldState()


class TestAreaIntegration:
    """The hold floors the published probability; it never creates evidence."""

    async def test_held_area_reads_occupied(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.closed_door_hold = True
        coordinator._door_holds[area_name] = DoorHold(  # noqa: SLF001
            DoorHoldState(held_from=datetime.now(UTC), doors_open=False)
        )
        assert area.probability() >= 0.75
        assert area.occupied()

    async def test_setting_off_ignores_a_hold(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.closed_door_hold = False
        before = area.probability()
        coordinator._door_holds[area_name] = DoorHold(  # noqa: SLF001
            DoorHoldState(held_from=datetime.now(UTC), doors_open=False)
        )
        assert area.probability() == before

    async def test_area_without_doors_keeps_no_hold(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.closed_door_hold = True
        has_doors = any(
            e.type.input_type == InputType.DOOR for e in area.entities.entities.values()
        )
        coordinator._observe_door_hold(area_name, area, datetime.now(UTC))  # noqa: SLF001
        assert (coordinator.door_hold_for(area_name) is not None) == has_doors

    async def test_hold_is_not_ground_truth(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """A held tick is labelled by the sensors alone (#558 criterion 5)."""
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        now = datetime.now(UTC)
        area.config.closed_door_hold = False
        label_off = coordinator._label_ground_truth(area_name, area, now)  # noqa: SLF001
        area.config.closed_door_hold = True
        coordinator._door_holds[area_name] = DoorHold(  # noqa: SLF001
            DoorHoldState(held_from=now, doors_open=False)
        )
        label_on = coordinator._label_ground_truth(  # noqa: SLF001
            area_name, area, now + timedelta(seconds=1)
        )
        assert label_on == label_off


class TestSettingDefaults:
    """Existing areas follow Wasp in Box; new bathrooms start with it on."""

    @pytest.mark.parametrize(
        ("area", "expected"),
        [
            ({"closed_door_hold": False, "wasp_enabled": True}, False),
            ({"closed_door_hold": True, "purpose": "social"}, True),
            ({"wasp_enabled": True, "purpose": "social"}, True),
            ({"wasp_enabled": False, "purpose": "bathroom"}, False),
            ({"purpose": "bathroom"}, True),
            ({"purpose": "social"}, False),
        ],
    )
    def test_form_default(self, area: dict, expected: bool) -> None:
        assert _closed_door_hold_default(area) is expected

    async def test_loader_follows_wasp_when_unset(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area = coordinator.get_area(coordinator.get_area_names()[0])
        config = area.config
        config._load_config({"wasp_enabled": True, "purpose": "social"})  # noqa: SLF001
        assert config.closed_door_hold is True
        config._load_config({"purpose": "bathroom"})  # noqa: SLF001
        assert config.closed_door_hold is False
        config._load_config({"purpose": "bathroom", "closed_door_hold": True})  # noqa: SLF001
        assert config.closed_door_hold is True
