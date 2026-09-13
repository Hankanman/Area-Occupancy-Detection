"""Synthetic learned history for a throwaway instance.

An instance with no history is only half an instance: priors sit at the
default, likelihoods are the type defaults, and nothing that depends on
learning can be looked at. This module fabricates the history instead of
waiting days for it.

Occupancy is generated as a two-state Markov chain per area whose stationary
occupancy tracks the profile's routine and whose dwell times come from
``mean_visit_minutes``, then every sensor is generated as a second chain
conditioned on that occupancy. The declared ``p_active_occupied`` / ``p_active_empty`` shape the data but
are not what comes out of it: a sensor with a two-minute dwell time stays on
for a while after the room empties, so the realised correlation always drifts
from the target. The generator therefore measures what it produced, exactly,
and that measurement is the ground truth -- it is what the seeded likelihoods
are written from and what ``harness verify`` holds the integration's own
correlation analysis against.

Rows are written straight into the integration's SQLite database, the same
``intervals`` table the recorder sync fills, so everything downstream --
occupied-interval cache, priors, correlations, transitions -- runs on it
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import exp
from pathlib import Path
import random
from typing import Any
from zoneinfo import ZoneInfo

import sqlalchemy as sa

from custom_components.area_occupancy.const import DB_NAME, DB_SCHEMA_VERSION
from custom_components.area_occupancy.data.entity_type import DEFAULT_TYPES
from custom_components.area_occupancy.db.schema import (
    Areas,
    Base,
    Entities,
    GlobalPriors,
    Intervals,
    Metadata,
    NumericSamples,
    Priors,
)
from custom_components.area_occupancy.time_utils import to_db_utc
from homeassistant.util import dt as dt_util

from .profiles import CHANNELS, AreaSpec, ChannelSpec, Profile, area_entities

#: Resolution the Markov chains are stepped at.
STEP = timedelta(minutes=1)

#: How often numeric channels are sampled.
SAMPLE_INTERVAL = timedelta(minutes=10)

#: Priors are bucketed into hourly slots, matching ``prior.DEFAULT_SLOT_MINUTES``.
SLOTS_PER_DAY = 24


@dataclass(frozen=True, slots=True)
class Run:
    """One uninterrupted stretch of a two-state chain.

    Attributes:
        active: Whether the chain was in its active state.
        start: Start of the stretch, aware UTC.
        end: End of the stretch, aware UTC.
    """

    active: bool
    start: datetime
    end: datetime

    @property
    def seconds(self) -> float:
        """Length of the stretch in seconds."""
        return (self.end - self.start).total_seconds()


@dataclass(slots=True)
class SensorHistory:
    """Generated history for one mock sensor.

    Attributes:
        entity_id: The sensor's entity id.
        channel: Channel name it was generated from.
        runs: Its state runs, for binary and media channels.
        samples: ``(timestamp, value)`` readings, for numeric channels.
        active_occupied_seconds: Seconds active while the area was occupied.
        occupied_seconds: Seconds the area was occupied over the window.
        active_empty_seconds: Seconds active while the area was empty.
        empty_seconds: Seconds the area was empty over the window.
    """

    entity_id: str
    channel: str
    runs: list[Run] = field(default_factory=list)
    samples: list[tuple[datetime, float]] = field(default_factory=list)
    active_occupied_seconds: float = 0.0
    occupied_seconds: float = 0.0
    active_empty_seconds: float = 0.0
    empty_seconds: float = 0.0

    @property
    def observed_prob_given_true(self) -> float:
        """Measured P(active | occupied) in the generated history."""
        if self.occupied_seconds <= 0:
            return 0.0
        return self.active_occupied_seconds / self.occupied_seconds

    @property
    def observed_prob_given_false(self) -> float:
        """Measured P(active | empty) in the generated history."""
        if self.empty_seconds <= 0:
            return 0.0
        return self.active_empty_seconds / self.empty_seconds


@dataclass(slots=True)
class AreaHistory:
    """Generated history for one area.

    Attributes:
        area: The area spec it came from.
        area_name: The area's display name, which is how the database keys it.
        start: Start of the generated window, aware UTC.
        end: End of the generated window, aware UTC.
        occupied: Occupied stretches.
        sensors: Per-sensor history.
        slot_occupied_seconds: Occupied seconds per ``(weekday, hour)`` slot.
        slot_total_seconds: Total seconds per ``(weekday, hour)`` slot.
    """

    area: AreaSpec
    area_name: str
    start: datetime
    end: datetime
    occupied: list[Run] = field(default_factory=list)
    sensors: list[SensorHistory] = field(default_factory=list)
    slot_occupied_seconds: dict[tuple[int, int], float] = field(default_factory=dict)
    slot_total_seconds: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def occupied_seconds(self) -> float:
        """Total occupied seconds over the window."""
        return sum(run.seconds for run in self.occupied)

    @property
    def total_seconds(self) -> float:
        """Length of the generated window in seconds."""
        return (self.end - self.start).total_seconds()

    @property
    def global_prior(self) -> float:
        """Fraction of the window the area was occupied."""
        if self.total_seconds <= 0:
            return 0.0
        return self.occupied_seconds / self.total_seconds


def _transition_probabilities(
    target: float, mean_active_minutes: float, step_minutes: float
) -> tuple[float, float]:
    """Solve per-step transition probabilities for a two-state chain.

    The chain is parameterised by where it should settle and how long an
    active stretch lasts: the leave rate follows from the mean duration, and
    the enter rate is whatever makes the stationary fraction equal ``target``.

    Args:
        target: Stationary probability of being active, in [0, 1].
        mean_active_minutes: Mean length of an active stretch.
        step_minutes: Chain step size in minutes.

    Returns:
        Tuple of ``(p_enter, p_leave)`` per step.
    """
    if target <= 0.0:
        return (0.0, 1.0)
    if target >= 1.0:
        return (1.0, 0.0)
    leave_rate = 1.0 / max(mean_active_minutes, step_minutes)
    enter_rate = leave_rate * target / (1.0 - target)
    return (
        1.0 - exp(-enter_rate * step_minutes),
        1.0 - exp(-leave_rate * step_minutes),
    )


def _walk(
    start: datetime,
    end: datetime,
    target_at: Callable[[datetime], float],
    mean_active_minutes: float,
    rng: random.Random,
) -> Iterator[Run]:
    """Step a two-state chain across a window, yielding merged runs.

    Args:
        start: Window start, aware UTC.
        end: Window end, aware UTC.
        target_at: Stationary active probability at a given moment.
        mean_active_minutes: Mean length of an active stretch.
        rng: Seeded random source.

    Yields:
        Consecutive runs covering the whole window, alternating state.
    """
    step_minutes = STEP.total_seconds() / 60.0
    moment = start
    active = rng.random() < target_at(start)
    run_start = start

    while moment < end:
        enter, leave = _transition_probabilities(
            target_at(moment), mean_active_minutes, step_minutes
        )
        flips = rng.random() < (leave if active else enter)
        moment += STEP
        if flips:
            yield Run(active=active, start=run_start, end=min(moment, end))
            active = not active
            run_start = moment

    if run_start < end:
        yield Run(active=active, start=run_start, end=end)


def _occupancy_target(area: AreaSpec, tz: ZoneInfo) -> Callable[[datetime], float]:
    """Build the area's routine lookup in local time.

    Args:
        area: The area whose routine to read.
        tz: Instance time zone. Routines are a local-time notion, and so are
            the prior slots they end up in.

    Returns:
        A callable from UTC moment to occupancy fraction.
    """

    def target(moment: datetime) -> float:
        local = moment.astimezone(tz)
        return area.occupancy.fraction_at(local.hour, local.weekday())

    return target


def _occupied_at(occupied: Sequence[Run]) -> Callable[[datetime], bool]:
    """Build a fast membership test over occupied stretches.

    Args:
        occupied: Occupied runs in ascending order.

    Returns:
        A callable answering whether a moment falls inside one of them. It
        assumes it is called with non-decreasing moments, which every caller
        here does, and walks the list once.
    """
    cursor = 0

    def is_occupied(moment: datetime) -> bool:
        nonlocal cursor
        while cursor < len(occupied) and occupied[cursor].end <= moment:
            cursor += 1
        return cursor < len(occupied) and occupied[cursor].start <= moment

    return is_occupied


def _accumulate_slots(history: AreaHistory, tz: ZoneInfo) -> None:
    """Fill in per-slot occupied and total seconds for prior seeding.

    Args:
        history: The area history to annotate, with ``occupied`` already set.
        tz: Instance time zone, since slots are local.
    """
    is_occupied = _occupied_at(history.occupied)
    moment = history.start
    seconds = STEP.total_seconds()
    while moment < history.end:
        local = moment.astimezone(tz)
        slot = (local.weekday(), local.hour)
        history.slot_total_seconds[slot] = (
            history.slot_total_seconds.get(slot, 0.0) + seconds
        )
        if is_occupied(moment):
            history.slot_occupied_seconds[slot] = (
                history.slot_occupied_seconds.get(slot, 0.0) + seconds
            )
        moment += STEP


def _generate_sensor(
    entity_id: str,
    channel: str,
    spec: ChannelSpec,
    history: AreaHistory,
    rng: random.Random,
) -> SensorHistory:
    """Generate one sensor's history conditioned on the area's occupancy.

    Args:
        entity_id: The sensor's entity id.
        channel: Channel name.
        spec: The channel's behaviour.
        history: The area history, with occupancy already generated.
        rng: Seeded random source.

    Returns:
        The sensor's generated history, including the correlation actually
        realised, which will differ slightly from the declared one.
    """
    sensor = SensorHistory(entity_id=entity_id, channel=channel)
    is_occupied = _occupied_at(history.occupied)

    if spec.is_numeric:
        model = spec.numeric
        assert model is not None
        moment = history.start
        while moment < history.end:
            occupied = is_occupied(moment)
            mean = model.occupied_mean if occupied else model.empty_mean
            value = round(rng.gauss(mean, model.sd), model.decimals)
            sensor.samples.append((moment, value))
            moment += SAMPLE_INTERVAL
        return sensor

    def target(moment: datetime) -> float:
        return spec.p_active_occupied if is_occupied(moment) else spec.p_active_empty

    # The occupancy test only moves forward, so the two consumers of it need
    # their own cursor; rebuild it for the measurement pass below.
    sensor.runs = list(
        _walk(history.start, history.end, target, spec.mean_active_minutes, rng)
    )

    measure = _occupied_at(history.occupied)
    seconds = STEP.total_seconds()
    for run in sensor.runs:
        moment = run.start
        while moment < run.end:
            if measure(moment):
                sensor.occupied_seconds += seconds
                if run.active:
                    sensor.active_occupied_seconds += seconds
            else:
                sensor.empty_seconds += seconds
                if run.active:
                    sensor.active_empty_seconds += seconds
            moment += STEP

    return sensor


def generate(
    profile: Profile,
    *,
    days: int,
    time_zone: str,
    seed: int = 1234,
    end: datetime | None = None,
) -> dict[str, AreaHistory]:
    """Generate history for every area in a profile.

    Args:
        profile: The profile to generate for.
        days: How many days of history to synthesise, ending now.
        time_zone: Instance time zone, used for routines and prior slots.
        seed: Random seed, so a given profile and seed always produce the
            same history.
        end: End of the window, defaulting to now.

    Returns:
        Mapping of area slug to its generated history.
    """
    tz = ZoneInfo(time_zone)
    window_end = end or dt_util.utcnow()
    window_start = window_end - timedelta(days=days)

    histories: dict[str, AreaHistory] = {}
    for area in profile.areas:
        # Seed per area so adding an area to a profile does not reshuffle the
        # history of the ones before it.
        rng = random.Random(f"{seed}:{area.slug}")
        history = AreaHistory(
            area=area,
            area_name=area.name,
            start=window_start,
            end=window_end,
        )
        history.occupied = [
            run
            for run in _walk(
                window_start,
                window_end,
                _occupancy_target(area, tz),
                area.occupancy.mean_visit_minutes,
                rng,
            )
            if run.active
        ]
        _accumulate_slots(history, tz)

        for channel, entities in area_entities(area).items():
            spec = CHANNELS[channel]
            for entity in entities:
                history.sensors.append(
                    _generate_sensor(entity, channel, spec, history, rng)
                )

        histories[area.slug] = history
    return histories


def _state_for(spec: ChannelSpec, active: bool) -> str:
    """State string to record for a run.

    Args:
        spec: The channel being recorded.
        active: Whether the run is the active state.

    Returns:
        The state the integration will see in the intervals table.
    """
    return spec.active_states[0] if active else spec.idle_state


def _chunked(
    rows: list[dict[str, Any]], size: int = 5000
) -> Iterator[list[dict[str, Any]]]:
    """Split rows into batches for executemany inserts.

    Args:
        rows: Rows to split.
        size: Maximum batch size.

    Yields:
        Successive batches.
    """
    for index in range(0, len(rows), size):
        yield rows[index : index + size]


def write(
    config_dir: Path,
    profile: Profile,
    histories: dict[str, AreaHistory],
    *,
    entry_id: str,
    with_priors: bool = True,
) -> dict[str, int]:
    """Write generated history into the integration's database.

    The database is created from the integration's own schema and stamped
    with ``DB_SCHEMA_VERSION``, so the integration adopts it on startup
    instead of deciding the schema is stale and recreating it.

    Args:
        config_dir: The instance's configuration directory.
        profile: The profile being seeded.
        histories: Generated history, keyed by area slug.
        entry_id: The config entry id the rows belong to.
        with_priors: Also write the priors the generated history implies, so
            an instance looks trained without waiting for an analysis run.
            The pipeline recomputes them from the same intervals.

    Returns:
        Row counts per table, for reporting.
    """
    db_path = config_dir / ".storage" / DB_NAME
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sa.create_engine(f"sqlite:///{db_path}")

    now = to_db_utc(dt_util.utcnow())
    counts: dict[str, int] = {}

    try:
        Base.metadata.create_all(engine)

        area_rows: list[dict[str, Any]] = []
        entity_rows: list[dict[str, Any]] = []
        interval_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        prior_rows: list[dict[str, Any]] = []
        global_prior_rows: list[dict[str, Any]] = []

        for area in profile.areas:
            history = histories[area.slug]
            area_rows.append(
                {
                    "entry_id": entry_id,
                    "area_name": history.area_name,
                    "area_id": area.slug,
                    "purpose": area.purpose,
                    # Stored as a fraction, matching AreaConfig.threshold.
                    "threshold": area.threshold / 100.0,
                    "adjacent_areas": [
                        slug
                        for slug in area.adjacent
                        if slug in {other.slug for other in profile.areas}
                    ],
                    "created_at": now,
                    "updated_at": now,
                }
            )

            for sensor in history.sensors:
                spec = CHANNELS[sensor.channel]
                defaults = DEFAULT_TYPES[spec.input_type]
                entity_rows.append(
                    {
                        "entry_id": entry_id,
                        "area_name": history.area_name,
                        "entity_id": sensor.entity_id,
                        "entity_type": spec.input_type.value,
                        "weight": float(defaults["weight"]),
                        # The correlation the history was actually generated
                        # with, which is what analysis should rediscover.
                        "prob_given_true": (
                            sensor.observed_prob_given_true
                            if not spec.is_numeric
                            else float(defaults["prob_given_true"])
                        ),
                        "prob_given_false": (
                            sensor.observed_prob_given_false
                            if not spec.is_numeric
                            else float(defaults["prob_given_false"])
                        ),
                        "is_shared": False,
                        "shared_with_areas": None,
                        "last_updated": now,
                        "created_at": now,
                        "is_decaying": False,
                        "decay_start": None,
                        "evidence": False,
                    }
                )

                interval_rows.extend(
                    {
                        "entry_id": entry_id,
                        "area_name": history.area_name,
                        "entity_id": sensor.entity_id,
                        "state": _state_for(spec, run.active),
                        "start_time": to_db_utc(run.start),
                        "end_time": to_db_utc(run.end),
                        "duration_seconds": run.seconds,
                        "aggregation_level": "raw",
                        "created_at": now,
                    }
                    for run in sensor.runs
                )

                for timestamp, value in sensor.samples:
                    sample_rows.append(
                        {
                            "entry_id": entry_id,
                            "area_name": history.area_name,
                            "entity_id": sensor.entity_id,
                            "timestamp": to_db_utc(timestamp),
                            "value": value,
                            "unit_of_measurement": spec.unit,
                            "state": str(value),
                            "created_at": now,
                        }
                    )

            if not with_priors:
                continue

            global_prior_rows.append(
                {
                    "entry_id": entry_id,
                    "area_name": history.area_name,
                    "prior_value": history.global_prior,
                    "calculation_date": now,
                    "data_period_start": to_db_utc(history.start),
                    "data_period_end": to_db_utc(history.end),
                    "total_occupied_seconds": history.occupied_seconds,
                    "total_period_seconds": history.total_seconds,
                    "interval_count": len(history.occupied),
                    "confidence": 0.9,
                    "calculation_method": "harness_seed",
                    "underlying_data_hash": None,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            for weekday in range(7):
                for slot in range(SLOTS_PER_DAY):
                    total = history.slot_total_seconds.get((weekday, slot), 0.0)
                    if total <= 0:
                        continue
                    occupied = history.slot_occupied_seconds.get((weekday, slot), 0.0)
                    prior_rows.append(
                        {
                            "entry_id": entry_id,
                            "area_name": history.area_name,
                            "day_of_week": weekday,
                            "time_slot": slot,
                            "prior_value": occupied / total,
                            "data_points": int(total // 60),
                            "confidence": 0.9,
                            "last_calculation_date": now,
                            "sample_period_start": to_db_utc(history.start),
                            "sample_period_end": to_db_utc(history.end),
                            "calculation_method": "harness_seed",
                            "last_updated": now,
                        }
                    )

        with engine.begin() as connection:
            connection.execute(sa.delete(Metadata).where(Metadata.key == "db_version"))
            connection.execute(
                sa.insert(Metadata),
                [{"key": "db_version", "value": str(DB_SCHEMA_VERSION)}],
            )
            for table, rows in (
                (Areas, area_rows),
                (Entities, entity_rows),
                (Intervals, interval_rows),
                (NumericSamples, sample_rows),
                (Priors, prior_rows),
                (GlobalPriors, global_prior_rows),
            ):
                counts[table.__tablename__] = len(rows)
                for batch in _chunked(rows):
                    connection.execute(sa.insert(table), batch)
    finally:
        engine.dispose()

    return counts
