"""The motion timeout holds PIR sensors in full before they decay."""

from datetime import UTC, datetime, timedelta

from freezegun import freeze_time
import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity_type import InputType

T0 = datetime(2026, 10, 22, 12, 0, 0, tzinfo=UTC)


class TestDecayDelay:
    def test_counts_in_full_until_the_delay_ends(self) -> None:
        """Hall: 90 s timeout, 45 s half-life.

        Factor 1.0 at 89 s, 0.5 at 90 + 45 = 135 s: the hall stays
        occupied ~2 1/4 minutes after the PIR turns off, not 45 s.
        """
        with freeze_time(T0):
            decay = Decay(half_life=45.0)
            decay.start_decay(delay=90)
        with freeze_time(T0 + timedelta(seconds=89)):
            assert decay.decay_factor == 1.0
        with freeze_time(T0 + timedelta(seconds=135)):
            assert decay.decay_factor == pytest.approx(0.5)

    def test_no_delay_decays_at_once(self) -> None:
        with freeze_time(T0):
            decay = Decay(half_life=45.0)
            decay.start_decay()
        with freeze_time(T0 + timedelta(seconds=45)):
            assert decay.decay_factor == pytest.approx(0.5)


class TestCoordinatorSetsTheDelay:
    async def test_pir_gets_the_timeout_and_presence_does_not(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        motion = [
            e
            for e in area.entities.entities.values()
            if e.type.input_type == InputType.MOTION
        ]
        assert motion, "fixture area needs a motion sensor"
        area.config.sensors.motion_timeout = 90
        now = datetime.now(UTC)

        # Too little history to judge: treated as PIR.
        coordinator._label_ground_truth(area_name, area, now)  # noqa: SLF001
        assert all(e.decay_delay == 90.0 for e in motion)

        # Classified as a presence sensor (holds "on"): no timeout.
        labeler = coordinator._labelers[area_name]  # noqa: SLF001
        for e in motion:
            labeler.state.pulse[e.entity_id] = False
        coordinator._label_ground_truth(area_name, area, now)  # noqa: SLF001
        assert all(e.decay_delay == 0.0 for e in motion)

    async def test_turning_off_starts_the_delayed_decay(
        self, coordinator: AreaOccupancyCoordinator
    ) -> None:
        area_name = coordinator.get_area_names()[0]
        area = coordinator.get_area(area_name)
        entity = next(
            e
            for e in area.entities.entities.values()
            if e.type.input_type == InputType.MOTION
        )
        entity.decay_delay = 90.0
        entity.decay.stop_decay()
        entity.previous_evidence = True
        entity.stuck_since = None
        with freeze_time(T0):
            coordinator.hass.states.async_set(entity.entity_id, "off")
            entity.has_new_evidence()
            assert entity.decay.is_decaying
            assert entity.decay.decay_start == T0 + timedelta(seconds=90)
