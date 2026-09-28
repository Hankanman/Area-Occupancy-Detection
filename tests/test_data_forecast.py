"""Tests for the read-side forecast helpers (data/forecast.py)."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from custom_components.area_occupancy.const import (
    MAX_PRIOR,
    MIN_PRIOR,
    PRIOR_WARMUP_MIN_SPAN_HOURS,
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.analysis import (
    compute_slot_priors,
    compute_zone_priors,
)
from custom_components.area_occupancy.data.forecast import (
    build_aggregate_time_priors,
    build_area_time_priors,
    conditioned_forecast,
    forecast_prior,
    persistence_tau_slots,
    shrink_slot_prior,
    slots_ahead_of,
)
from custom_components.area_occupancy.data.prior import (
    DEFAULT_SLOT_MINUTES,
    PRIOR_FACTOR,
)
from custom_components.area_occupancy.data.types import ZonePriors
from custom_components.area_occupancy.time_utils import to_local
from custom_components.area_occupancy.utils import combine_priors

# ruff: noqa: SLF001


def test_forecast_prior_without_global_returns_clamped_slot():
    """With no learned global prior, the slot's own value (clamped) is used."""
    assert forecast_prior(None, 0.6, prior_factor=PRIOR_FACTOR) == pytest.approx(
        max(MIN_PRIOR, min(MAX_PRIOR, 0.6))
    )


def test_forecast_prior_combines_global_and_slot():
    """A learned global prior is combined with the slot's time prior."""
    expected = max(MIN_PRIOR, min(MAX_PRIOR, combine_priors(0.5, 0.6) * PRIOR_FACTOR))
    assert forecast_prior(0.5, 0.6, prior_factor=PRIOR_FACTOR) == pytest.approx(
        expected
    )


def test_forecast_prior_within_bounds():
    """The forecast is always clamped to [MIN_PRIOR, MAX_PRIOR]."""
    result = forecast_prior(0.99, 0.9, prior_factor=PRIOR_FACTOR)
    assert MIN_PRIOR <= result <= MAX_PRIOR


def test_forecast_prior_respects_prior_factor():
    """prior_factor scales the combined value before clamping."""
    # factor 0 → adjusted 0 → clamped up to MIN_PRIOR
    assert forecast_prior(0.5, 0.5, prior_factor=0.0) == pytest.approx(MIN_PRIOR)


def test_build_area_time_priors_structure(coordinator: AreaOccupancyCoordinator):
    """build_area_time_priors emits area_id, global_prior, slot_minutes, slots."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    area.prior.global_prior = 0.5
    area.prior._cached_time_priors = {(0, 0): 0.4, (2, 10): 0.6}
    # Seed the points map too: an unset one makes the builder reload the grid
    # from the database, replacing the values seeded above.
    area.prior._cached_time_prior_points = {(0, 0): 1, (2, 10): 1}

    data = build_area_time_priors(area, DEFAULT_SLOT_MINUTES)

    assert data["area_id"] == area.config.area_id
    assert data["global_prior"] == 0.5
    assert data["slot_minutes"] == DEFAULT_SLOT_MINUTES
    assert set(data["slots"]) == {"0,0", "2,10"}
    # The baseline is the evidence-free series, so it is the seeded slot's
    # forecast whatever the clock says (``slots`` blends in live evidence for
    # the slots just ahead of now).
    assert data["slots_baseline"]["2,10"] == round(
        forecast_prior(0.5, 0.6, prior_factor=PRIOR_FACTOR, weeks=1), 4
    )


def _member(
    area_id: str,
    prior_map: dict[tuple[int, int], float],
    points_map: dict[tuple[int, int], int] | None = None,
    probability: float = 0.5,
):
    """A minimal Area-like stand-in for aggregation tests.

    ``points_map`` defaults to "every known slot has one week of data" so tests
    that don't care about sample counts stay readable.
    """
    points = points_map if points_map is not None else dict.fromkeys(prior_map, 1)
    prior = SimpleNamespace(
        all_time_priors=lambda _m=prior_map: dict(_m),
        all_time_prior_points=lambda _p=points: dict(_p),
        prior_for=lambda d, s, _m=prior_map: _m.get((d, s), 0.0),
        # Anchor "now" far from the tested slots so the evidence weight vanishes
        # and aggregation maths can be asserted against the plain baseline.
        day_of_week=3,
        time_slot=12,
    )
    return SimpleNamespace(
        prior=prior,
        config=SimpleNamespace(area_id=area_id, threshold=0.5),
        purpose=SimpleNamespace(half_life=600.0),
        probability=lambda _p=probability: _p,
    )


def _aggregate(members, **kwargs):
    return build_aggregate_time_priors(
        members,
        DEFAULT_SLOT_MINUTES,
        "all_areas",
        "All Areas",
        prior_factor=PRIOR_FACTOR,
        **kwargs,
    )


def test_build_aggregate_takes_the_highest_member_without_zone_priors():
    """Before the first analysis, each slot is its busiest member (#557).

    An average says a room busy at 08:00 and an idle one make a half-busy
    zone; "anyone in the zone" is at least as likely as its busiest room.
    """
    m1 = _member("a", {(0, 8): 0.8, (1, 9): 0.2})
    m2 = _member("b", {(0, 8): 0.4, (1, 9): 0.6})

    res = _aggregate([m1, m2])

    assert res["area_id"] == "all_areas"
    assert res["name"] == "All Areas"
    assert res["aggregate"] is True
    assert set(res["members"]) == {"a", "b"}
    assert res["slots_baseline"]["0,8"] == pytest.approx(0.8)
    assert res["slots_baseline"]["1,9"] == pytest.approx(0.6)
    assert res["slots"]["0,8"] == pytest.approx(0.8)
    assert res["slots_raw"]["0,8"] == pytest.approx(0.8)


def test_build_aggregate_unions_member_slots():
    """Slots present in any member are aggregated."""
    res = _aggregate([_member("a", {(0, 8): 0.9}), _member("b", {(2, 3): 0.5})])

    assert set(res["slots"]) == {"0,8", "2,3"}
    assert res["slots_baseline"]["0,8"] == pytest.approx(0.9)


def test_build_aggregate_empty_returns_none():
    """No members → nothing to aggregate."""
    assert _aggregate([]) is None


def test_build_aggregate_result_within_bounds():
    """Aggregated values stay within [MIN_PRIOR, MAX_PRIOR]."""
    res = _aggregate([_member("a", {(0, 0): 0.99})])

    assert MIN_PRIOR <= res["slots"]["0,0"] <= MAX_PRIOR


def test_build_aggregate_data_points_take_member_minimum():
    """Without zone priors, a zone is only as well-learned as its weakest room."""
    m1 = _member("a", {(0, 8): 0.8}, {(0, 8): 4})
    m2 = _member("b", {(0, 8): 0.4}, {(0, 8): 1})

    assert _aggregate([m1, m2])["data_points"]["0,8"] == 1


def test_build_aggregate_uses_the_zone_priors():
    """With empirical zone priors the habit is the zone's own (#557).

    Three weeks at 0.6 shrink to (3 * 0.6 + 2 * 0.4) / 5 = 0.52, and
    sigmoid(0.6 * logit(0.4) + 0.4 * logit(0.52)) = sigmoid(-0.2113) = 0.44738;
    raw and data points are the zone's.
    """
    empirical = ZonePriors(
        global_prior=0.4,
        time_priors={(0, 8): 0.6},
        data_points={(0, 8): 3},
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    members = [_member("a", {(0, 8): 0.3}), _member("b", {(0, 8): 0.2})]

    res = _aggregate(members, empirical=empirical)

    assert res["slots_baseline"]["0,8"] == pytest.approx(0.4474, abs=1e-4)
    assert res["slots_raw"]["0,8"] == pytest.approx(0.6)
    assert res["data_points"]["0,8"] == 3
    # Quiet members cannot pull the live series below the zone habit.
    assert res["slots"]["0,8"] == pytest.approx(0.4474, abs=1e-4)


def test_build_aggregate_live_follows_an_occupied_member():
    """A room occupied right now lifts its zone's current slot to at least its own."""
    busy = _member("a", {(3, 12): 0.1}, probability=0.98)
    quiet = _member("b", {(3, 12): 0.1}, probability=0.05)

    res = _aggregate([busy, quiet])

    # "now" is (3, 12) for these stand-ins: slot 0 is the member's posterior.
    assert res["slots"]["3,12"] == pytest.approx(0.98, abs=1e-4)


class TestComputeZonePriors:
    """Zone priors come from the union of the rooms' occupied history (#557)."""

    START = datetime(2026, 1, 5, tzinfo=UTC)
    END = START + timedelta(hours=48)

    def _hour(self, h: float) -> datetime:
        return self.START + timedelta(hours=h)

    def test_union_not_mean(self) -> None:
        """Rooms busy 10-12 and 11-13 make a zone busy 10-13.

        Union 3 h over 48 h = 0.0625; the mean of the rooms (2/48 each) was
        0.0417, and so would be either room alone.
        """
        zone = compute_zone_priors(
            [
                [(self._hour(10), self._hour(12))],
                [(self._hour(11), self._hour(13))],
            ],
            self.START,
            self.END,
        )

        assert zone is not None
        assert zone.global_prior == pytest.approx(3 / 48)
        # Each of the three hours is fully occupied once in the window, so
        # each clamps to the time-prior ceiling; every other slot is empty.
        assert sorted(zone.time_priors.values()).count(TIME_PRIOR_MAX_BOUND) == 3
        assert len(zone.time_priors) == 48

    def test_too_short_a_window_is_skipped(self) -> None:
        """Below the prior warm-up span there is nothing trustworthy to report."""
        assert (
            compute_zone_priors(
                [[(self._hour(1), self._hour(2))]],
                self.START,
                self._hour(PRIOR_WARMUP_MIN_SPAN_HOURS - 1),
            )
            is None
        )


def test_zone_prior_reads_the_empirical_priors(coordinator: AreaOccupancyCoordinator):
    """The All Areas prior uses the zone priors for the current slot.

    Three weeks at 0.6 shrink to 0.52 against the 0.4 global: 0.44738.
    """
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    all_areas = coordinator.get_all_areas()
    slot = (area.prior.day_of_week, area.prior.time_slot)
    all_areas.empirical = ZonePriors(
        global_prior=0.4,
        time_priors={slot: 0.6},
        data_points={slot: 3},
        computed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert all_areas.area_prior() == pytest.approx(0.4474, abs=1e-4)


def test_persistence_tau_follows_purpose_ordering():
    """Tau ranks purposes the way their half-lives do, within sane bounds."""
    passageway = persistence_tau_slots(45.0, DEFAULT_SLOT_MINUTES)
    working = persistence_tau_slots(600.0, DEFAULT_SLOT_MINUTES)
    sleeping = persistence_tau_slots(1200.0, DEFAULT_SLOT_MINUTES)

    assert passageway < working < sleeping
    # Raw half-lives are evidence-decay seconds, far too short to be used as
    # persistence directly; the remap floors them instead of collapsing to ~0.
    assert passageway == pytest.approx(0.5)
    assert working == pytest.approx(1.0)
    assert sleeping == pytest.approx(2.0)


def test_conditioned_forecast_anchors_on_now_and_relaxes_to_baseline():
    """Slot 0 is the posterior; distant slots are the untouched baseline."""
    posterior, baseline = 0.9848, 0.3512

    assert conditioned_forecast(posterior, baseline, 0, 1.0) == pytest.approx(
        posterior, abs=1e-6
    )
    # The next slot is lifted well above the baseline but not to the posterior.
    nxt = conditioned_forecast(posterior, baseline, 1, 1.0)
    assert baseline < nxt < posterior
    assert nxt == pytest.approx(0.759, abs=0.002)
    # Far ahead, the evidence has no say left.
    assert conditioned_forecast(posterior, baseline, 40, 1.0) == pytest.approx(
        baseline, abs=1e-6
    )


def test_conditioned_forecast_suppresses_when_area_just_emptied():
    """Evidence cuts both ways: empty now means less likely than the habit."""
    baseline = 0.3512
    nxt = conditioned_forecast(0.03, baseline, 1, 1.0)

    assert nxt < baseline
    assert nxt == pytest.approx(0.159, abs=0.002)


def test_slots_ahead_wraps_the_week():
    """A slot earlier in the week is that slot next week, never negative."""
    assert slots_ahead_of(3, 12, 3, 12, 24) == 0
    assert slots_ahead_of(3, 13, 3, 12, 24) == 1
    assert slots_ahead_of(3, 11, 3, 12, 24) == 167
    assert slots_ahead_of(0, 0, 6, 23, 24) == 1


class TestSparseSlotPriors:
    """Thin slots and the hour in progress cannot flip the live prior.

    Live report on 2026.9.2: one week of history per slot (the 2026.9.1
    database reset) put slots on the 0.9 / 0.03 bounds, and the :57
    analysis counted the hour still in progress. The Kitchen prior went
    0.228 -> 0.537 at 17:57 and back at 18:00; the Lounge went occupied
    at 18:00 on its prior alone, with its only sensor unavailable.
    """

    def test_shrink_pulls_one_week_a_third_of_the_way(self) -> None:
        """(1 * 0.9 + 2 * 0.22801) / 3 = 0.45201."""
        assert shrink_slot_prior(0.9, 1, 0.22801) == pytest.approx(0.452007, abs=1e-6)

    def test_shrink_fades_with_data(self) -> None:
        """Eight weeks keep 80% of the slot: (8 * 0.9 + 2 * 0.3) / 10 = 0.78."""
        assert shrink_slot_prior(0.9, 8, 0.3) == pytest.approx(0.78)

    def test_shrink_leaves_unlearned_and_unknown_alone(self) -> None:
        """No weeks, or no global prior yet: the slot is used as it is."""
        assert shrink_slot_prior(0.9, 0, 0.3) == 0.9
        assert shrink_slot_prior(0.9, 3, None) == 0.9

    @pytest.mark.parametrize(
        ("global_prior", "old", "new"),
        [
            # Kitchen and Lounge values from the live report. ``old`` is what
            # 2026.9.2 showed; ``new`` is combine(g, (0.9 + 2g) / 3).
            (0.22801, 0.53672, 0.30815),
            (0.26118, 0.56340, 0.33954),
        ],
    )
    def test_live_prior_stays_below_threshold_on_one_saturated_week(
        self, coordinator: AreaOccupancyCoordinator, global_prior, old, new
    ) -> None:
        """A single week at the 0.9 bound no longer carries the room over 0.5."""
        area = coordinator.get_area(coordinator.get_area_names()[0])
        prior = area.prior
        slot = (prior.day_of_week, prior.time_slot)
        prior.global_prior = global_prior
        prior._cached_time_priors = {slot: TIME_PRIOR_MAX_BOUND}
        prior._cached_time_prior_points = {slot: 1}

        assert combine_priors(global_prior, TIME_PRIOR_MAX_BOUND) == pytest.approx(
            old, abs=1e-5
        )
        assert prior.value == pytest.approx(new, abs=1e-5)
        assert prior.value < area.config.threshold

    def test_the_hour_in_progress_is_not_learned(self) -> None:
        """Analysis at 12:57 with the room occupied 12:00-12:57 leaves slot 12 alone.

        Before, slot 12 got 57/57 occupied and sat on the 0.9 bound.
        """
        start = datetime(2026, 1, 5, tzinfo=UTC)
        now = start + timedelta(hours=36, minutes=57)
        busy = (start + timedelta(hours=36), now)
        slot_12 = (to_local(busy[0]).weekday(), to_local(busy[0]).hour)

        priors, points = compute_slot_priors([busy], start, now)

        assert slot_12 not in priors
        assert slot_12 not in points
        # The completed hour before it is still learned (and empty).
        prev = to_local(busy[0] - timedelta(hours=1))
        assert priors[(prev.weekday(), prev.hour)] == TIME_PRIOR_MIN_BOUND
