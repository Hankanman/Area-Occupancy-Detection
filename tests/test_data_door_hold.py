"""Tests for the conservative closed-door hold (#558)."""

from datetime import UTC, datetime, timedelta

from custom_components.area_occupancy.config_flow import _closed_door_hold_default
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.door_hold import DoorHold, DoorHoldState

T0 = datetime(2026, 10, 6, 12, 0, 0, tzinfo=UTC)
LONG_TIMEOUT = 65.0
MAX_HOLD = 3600.0
HALF_LIFE = 450.0


def observe(
    hold: DoorHold,
    seconds: float,
    *,
    doors_open: bool | None,
    motion_on: bool | None,
    trusted_presence_on: bool = False,
    occupancy_button_pressed: bool = False,
    long_motion_timeout: float = LONG_TIMEOUT,
    max_hold: float = MAX_HOLD,
) -> None:
    hold.observe(
        T0 + timedelta(seconds=seconds),
        doors_open=doors_open,
        motion_on=motion_on,
        trusted_presence_on=trusted_presence_on,
        occupancy_button_pressed=occupancy_button_pressed,
        long_motion_timeout=long_motion_timeout,
        max_hold=max_hold,
        fade_half_life=HALF_LIFE,
    )


class TestArming:
    def test_closed_door_alone_never_arms(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=False)
        observe(hold, 10, doors_open=False, motion_on=False)
        assert not hold.held

    def test_trusted_presence_arms_when_closed(self) -> None:
        hold = DoorHold()
        observe(
            hold,
            0,
            doors_open=False,
            motion_on=False,
            trusted_presence_on=True,
        )
        assert hold.held

    def test_button_arms_when_closed(self) -> None:
        hold = DoorHold()
        observe(
            hold,
            0,
            doors_open=False,
            motion_on=False,
            occupancy_button_pressed=True,
        )
        assert hold.held

    def test_untrusted_signals_do_not_arm(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=False)
        observe(hold, 10, doors_open=False, motion_on=False)
        assert not hold.held

    def test_long_motion_before_timeout_does_not_arm(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=True)
        observe(hold, LONG_TIMEOUT - 1, doors_open=False, motion_on=True)
        assert not hold.held

    def test_long_motion_exactly_at_timeout_arms(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=True)
        observe(hold, LONG_TIMEOUT, doors_open=False, motion_on=True)
        assert hold.held

    def test_long_motion_after_timeout_arms(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=True)
        observe(hold, LONG_TIMEOUT + 1, doors_open=False, motion_on=True)
        assert hold.held

    def test_motion_interruption_resets_timer(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=True)
        observe(hold, 60, doors_open=False, motion_on=False)
        observe(hold, 120, doors_open=False, motion_on=True)
        observe(hold, 184, doors_open=False, motion_on=True)
        assert not hold.held
        observe(hold, 185, doors_open=False, motion_on=True)
        assert hold.held

    def test_unknown_motion_does_not_arm(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=None)
        assert not hold.held

    def test_unknown_door_does_not_arm(self) -> None:
        hold = DoorHold()
        observe(
            hold,
            0,
            doors_open=None,
            motion_on=True,
            trusted_presence_on=True,
        )
        assert not hold.held


class TestReleaseAndExpiry:
    def test_opening_releases_hold_fade_and_motion_timer(self) -> None:
        hold = DoorHold()
        observe(
            hold,
            0,
            doors_open=False,
            motion_on=True,
            trusted_presence_on=True,
        )
        hold.observe(
            T0 + timedelta(seconds=10),
            doors_open=True,
            motion_on=True,
            long_motion_timeout=LONG_TIMEOUT,
            max_hold=MAX_HOLD,
            fade_half_life=HALF_LIFE,
        )
        assert not hold.held
        assert hold.state.fade_from is None
        assert hold.state.closed_motion_started_at is None

    def test_max_hold_expiries_and_fades(self) -> None:
        hold = DoorHold()
        observe(
            hold,
            0,
            doors_open=False,
            motion_on=False,
            trusted_presence_on=True,
        )
        observe(
            hold,
            MAX_HOLD,
            doors_open=False,
            motion_on=False,
            trusted_presence_on=False,
        )
        assert not hold.held
        assert hold.state.fade_from == T0 + timedelta(seconds=MAX_HOLD)

    def test_zero_max_hold_never_expires(self) -> None:
        hold = DoorHold()
        observe(
            hold,
            0,
            doors_open=False,
            motion_on=False,
            trusted_presence_on=True,
            max_hold=0,
        )
        observe(
            hold,
            86400,
            doors_open=False,
            motion_on=False,
            max_hold=0,
        )
        assert hold.held

    def test_fade_ends_after_six_half_lives(self) -> None:
        hold = DoorHold()
        observe(
            hold,
            0,
            doors_open=False,
            motion_on=False,
            trusted_presence_on=True,
        )
        observe(
            hold,
            MAX_HOLD + 6 * HALF_LIFE,
            doors_open=False,
            motion_on=False,
        )
        assert hold.state.fade_from is None


class TestState:
    def test_round_trip(self) -> None:
        hold = DoorHold()
        observe(hold, 0, doors_open=False, motion_on=False, trusted_presence_on=True)
        assert DoorHoldState.from_dict(hold.state.to_dict()) == hold.state

    def test_malformed_payload_is_empty(self) -> None:
        assert (
            DoorHoldState.from_dict({"last_evidence_at": "nonsense"}) == DoorHoldState()
        )

    def test_legacy_held_from_is_not_trusted_as_new_evidence(self) -> None:
        state = DoorHoldState.from_dict({"held_from": T0.isoformat()})
        assert state.last_evidence_at is None


class TestAreaIntegration:
    async def test_setting_off_ignores_hold(
        self,
        coordinator: AreaOccupancyCoordinator,
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        area.config.closed_door_hold = False
        before = area.probability()
        coordinator._door_holds[area_name] = DoorHold(  # noqa: SLF001
            DoorHoldState(last_evidence_at=datetime.now(UTC), doors_open=False)
        )
        assert area.probability() == before


class TestSettingDefaults:
    def test_feature_is_disabled_by_default(self) -> None:
        assert _closed_door_hold_default({}) is False
        assert _closed_door_hold_default({"purpose": "bathroom"}) is False

    def test_explicit_setting_is_preserved(self) -> None:
        assert _closed_door_hold_default({"closed_door_hold": True}) is True
