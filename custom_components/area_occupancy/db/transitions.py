"""Transition learning between adjacent areas.

Phase 3 of the adjacent-areas feature (discussion #431). Walks each area's
occupied-interval timeline once per analysis cycle and counts area-to-area
transitions, bucketed by hour-of-week (0..167, weekday × 24 + hour). Two
chain depths are recorded per transition into the same ``AreaTransitions``
table:

* 1-hop ``X → Y``: stored with ``mid_area=""``.
* 2-hop ``W → X → Y``: stored with ``mid_area=X``, only when the prior
  ``W → X`` transition itself fits within the trajectory window.

Only transitions where every link is a configured-adjacent pair are
counted, so the table stays sparse and irrelevant pairs (rooms that aren't
physically connected in the user's installation) don't pollute the model.

Counts are exponentially decayed each cycle by
``ADJACENCY_RECENCY_HALF_LIFE_DAYS`` so the model adapts to changing
household patterns. Transitions older than the last full pipeline run are
not re-counted (we track ``last_observed_end_time`` per entry in the
``Metadata`` table); only the new tail since the last run is observed.

This module does **not** read from ``AreaTransitions`` — that's the
Phase 4 lookup path. Diagnostics consumers can however inspect what was
recorded via :func:`summarize_transitions_for_diagnostics`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import (
    ADJACENCY_N_CHAIN,
    ADJACENCY_N_HOUR,
    ADJACENCY_N_PAIR,
    ADJACENCY_N_SPECIFIC,
    ADJACENCY_RECENCY_HALF_LIFE_DAYS,
    ADJACENCY_TRAJECTORY_WINDOW_S,
    ADJACENCY_TRANSITION_WINDOW_S,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MOTION_TIMEOUT,
)
from ..time_utils import from_db_utc, to_local
from .queries import get_occupied_intervals
from .relationships import DEFAULT_INFLUENCE_WEIGHTS

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)

# Metadata key prefix used to remember how far we've consumed each entry's
# transition timeline. The actual key is suffixed with the entry_id to
# scope the watermark per integration entry. Value is ISO-8601 UTC.
_LAST_OBSERVED_KEY_PREFIX = "adjacency_last_observed_end_time"


def _watermark_key(entry_id: str) -> str:
    """Build the per-entry watermark key for the global Metadata table."""
    return f"{_LAST_OBSERVED_KEY_PREFIX}__{entry_id}"


@dataclass(frozen=True)
class _AreaEvent:
    """An area's interval boundary used during transition detection."""

    timestamp: datetime  # UTC-aware
    area_name: str
    is_start: bool  # True for interval start, False for interval end


@dataclass(frozen=True)
class _RecentArea:
    """A recently-ended area kept in the trajectory rolling window."""

    area_name: str
    end_time: datetime  # UTC-aware


def _hour_of_week(ts: datetime) -> int:
    """Return ``weekday × 24 + hour`` (0..167) in the configured local timezone.

    Mirrors the time-prior bucketing scheme so morning study→hall→bathroom
    learns separately from evening study→hall→bedroom.
    """
    local = to_local(ts)
    return local.weekday() * 24 + local.hour


def _is_pair_adjacent(
    adjacency_index: dict[str, set[str]], from_area: str, to_area: str
) -> bool:
    """True if ``to_area`` is in ``from_area``'s configured adjacents."""
    if from_area == to_area:
        return False
    return to_area in adjacency_index.get(from_area, set())


def _build_adjacency_index(db: AreaOccupancyDB, entry_id: str) -> dict[str, set[str]]:
    """Read the ``AreaRelationships`` rows once per cycle.

    Returns ``{area_name: {neighbour_area_name, ...}}`` for the given entry.
    Only ``relationship_type == "adjacent"`` rows participate.
    """
    try:
        with db.get_session() as session:
            rows = (
                session.query(db.AreaRelationships)
                .filter(
                    db.AreaRelationships.entry_id == entry_id,
                    db.AreaRelationships.relationship_type == "adjacent",
                )
                .all()
            )
            index: dict[str, set[str]] = {}
            for row in rows:
                index.setdefault(row.area_name, set()).add(row.related_area_name)
            return index
    except SQLAlchemyError:
        _LOGGER.exception("Error reading adjacency index")
        return {}


def _collect_events_for_areas(
    db: AreaOccupancyDB,
    entry_id: str,
    area_names: list[str],
    lookback_days: int,
    since: datetime | None,
) -> list[_AreaEvent]:
    """Build a sorted timeline of (start, end) boundaries across all areas.

    Each occupied interval contributes one start and one end event. Events
    older than ``since`` are dropped — we only learn from new tail data
    each cycle so the same transition isn't counted twice.
    """
    events: list[_AreaEvent] = []
    for area_name in area_names:
        intervals = get_occupied_intervals(
            db,
            entry_id,
            area_name,
            lookback_days=lookback_days,
            motion_timeout_seconds=DEFAULT_MOTION_TIMEOUT,
        )
        for start, end in intervals:
            start_aware = from_db_utc(start)
            end_aware = from_db_utc(end)
            # Only the END time gates inclusion — an interval that ends
            # after `since` may have started before it but is still new.
            if since is not None and end_aware <= since:
                continue
            events.append(_AreaEvent(start_aware, area_name, is_start=True))
            events.append(_AreaEvent(end_aware, area_name, is_start=False))
    # End events sort BEFORE start events at the same timestamp, so a
    # zero-gap "X ended at t, Y started at t" is detected: X's end
    # populates recent_ends first, then Y's start sees it and emits the
    # transition. Reversing this would silently miss zero-gap moves.
    events.sort(key=lambda e: (e.timestamp, 0 if not e.is_start else 1))
    return events


def _detect_transitions(
    events: list[_AreaEvent],
    adjacency_index: dict[str, set[str]],
) -> list[tuple[str, str, str, int]]:
    """Walk the timeline and yield (from, mid, to, hour_of_week) tuples.

    1-hop transitions emit ``(X, "", Y, hour)``; 2-hop transitions emit
    ``(W, X, Y, hour)`` in addition. Only chains where every link is in
    the configured adjacency index are recorded.

    The trajectory is a small rolling deque of ``_RecentArea`` entries
    pruned by ``ADJACENCY_TRAJECTORY_WINDOW_S``; we look at the most
    recent two for the 1-hop and 2-hop checks.
    """
    transition_window = timedelta(seconds=ADJACENCY_TRANSITION_WINDOW_S)
    trajectory_window = timedelta(seconds=ADJACENCY_TRAJECTORY_WINDOW_S)

    recent_ends: deque[_RecentArea] = deque(maxlen=8)
    transitions: list[tuple[str, str, str, int]] = []

    for event in events:
        if event.is_start:
            # Prune anything outside the trajectory window first.
            while recent_ends and (
                event.timestamp - recent_ends[0].end_time > trajectory_window
            ):
                recent_ends.popleft()
            if not recent_ends:
                continue

            # Most recent end: candidate for the 1-hop "X → Y" link.
            x_end = recent_ends[-1]
            within_x = event.timestamp - x_end.end_time <= transition_window
            if not within_x:
                continue
            if not _is_pair_adjacent(adjacency_index, x_end.area_name, event.area_name):
                continue

            hour = _hour_of_week(event.timestamp)
            transitions.append((x_end.area_name, "", event.area_name, hour))

            # 2-hop: the END before X must also be within trajectory window
            # of X's end, AND the (W → X) pair must itself be adjacent.
            if len(recent_ends) >= 2:
                w_end = recent_ends[-2]
                within_w = x_end.end_time - w_end.end_time <= trajectory_window
                if within_w and _is_pair_adjacent(
                    adjacency_index, w_end.area_name, x_end.area_name
                ):
                    transitions.append(
                        (w_end.area_name, x_end.area_name, event.area_name, hour)
                    )
        # Append the end event; deque auto-prunes the oldest entry
        # once we exceed the cap. The chronological window prune is
        # done above when the next start arrives.
        elif not recent_ends or recent_ends[-1].area_name != event.area_name:
            recent_ends.append(
                _RecentArea(area_name=event.area_name, end_time=event.timestamp)
            )
        else:
            # Same area ending again (overlapping intervals were
            # merged upstream, but defensive) — refresh end_time.
            recent_ends[-1] = _RecentArea(
                area_name=event.area_name, end_time=event.timestamp
            )

    return transitions


def _apply_recency_decay_in_session(
    session: Any,
    db: AreaOccupancyDB,
    entry_id: str,
    hours_since_last_run: float,
) -> None:
    """Apply recency decay inside an existing session.

    Multiplies every count by ``0.5 ^ (hours / (24 * half_life_days))``.
    Done before adding new observations so the new tail is full-weight
    relative to the aged history.

    No commit here — the caller controls transaction boundaries so the
    decay, upsert, and watermark advance can fail or succeed atomically.
    """
    if hours_since_last_run <= 0:
        return
    factor = 0.5 ** (hours_since_last_run / (24 * ADJACENCY_RECENCY_HALF_LIFE_DAYS))
    if factor >= 1.0:
        return  # No-op
    (
        session.query(db.AreaTransitions)
        .filter(db.AreaTransitions.entry_id == entry_id)
        .update(
            {db.AreaTransitions.count: db.AreaTransitions.count * factor},
            synchronize_session=False,
        )
    )


def _upsert_transition_counts_in_session(
    session: Any,
    db: AreaOccupancyDB,
    entry_id: str,
    transitions: list[tuple[str, str, str, int]],
) -> None:
    """Upsert observed transition counts inside an existing session."""
    if not transitions:
        return
    # Group by chain+hour so we issue at most one row per unique key.
    grouped: dict[tuple[str, str, str, int], int] = {}
    for chain in transitions:
        grouped[chain] = grouped.get(chain, 0) + 1

    existing_rows = (
        session.query(db.AreaTransitions)
        .filter(db.AreaTransitions.entry_id == entry_id)
        .all()
    )
    existing_by_key = {
        (r.from_area, r.mid_area, r.to_area, r.hour_of_week): r for r in existing_rows
    }

    for (from_area, mid_area, to_area, hour), increment in grouped.items():
        row = existing_by_key.get((from_area, mid_area, to_area, hour))
        if row is not None:
            row.count = (row.count or 0.0) + float(increment)
        else:
            session.add(
                db.AreaTransitions(
                    entry_id=entry_id,
                    from_area=from_area,
                    mid_area=mid_area,
                    to_area=to_area,
                    hour_of_week=hour,
                    count=float(increment),
                )
            )


def _set_metadata_in_session(
    session: Any, db: AreaOccupancyDB, key: str, value: str
) -> None:
    """Upsert a Metadata row inside an existing session (no commit)."""
    row = session.query(db.Metadata).filter_by(key=key).first()
    if row:
        row.value = value
    else:
        session.add(db.Metadata(key=key, value=value))


# Standalone wrappers for read-only metadata access and for tests that
# don't need transactional bundling. Errors propagate so callers can
# decide whether to skip the cycle or fail the analysis step.


def _apply_recency_decay(
    db: AreaOccupancyDB, entry_id: str, hours_since_last_run: float
) -> None:
    """Standalone recency decay (commits its own session). Used by tests."""
    with db.get_session() as session:
        _apply_recency_decay_in_session(session, db, entry_id, hours_since_last_run)
        session.commit()


def _upsert_transition_counts(
    db: AreaOccupancyDB,
    entry_id: str,
    transitions: list[tuple[str, str, str, int]],
) -> None:
    """Standalone upsert (commits its own session). Used by tests."""
    if not transitions:
        return
    with db.get_session() as session:
        _upsert_transition_counts_in_session(session, db, entry_id, transitions)
        session.commit()


def _get_metadata(db: AreaOccupancyDB, key: str) -> str | None:
    """Read a Metadata value. Returns None on missing row.

    SQLAlchemyError is allowed to propagate so the caller can decide
    whether the failure should skip the cycle (preventing double-counting
    on a lost watermark read).
    """
    with db.get_session() as session:
        row = session.query(db.Metadata).filter_by(key=key).first()
        return row.value if row else None


def record_transitions_for_entry(
    db: AreaOccupancyDB,
    entry_id: str,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one cycle of transition detection for an entry.

    Reads phase (separate sessions, idempotent):
      1. Read the prior watermark from ``Metadata``.
      2. Read the configured adjacency index.
      3. Collect interval boundaries since the watermark (separate
         per-area read sessions inside ``get_occupied_intervals``).
      4. Walk the timeline, emitting 1-hop and 2-hop chain observations.

    Write phase (single transaction):
      5. Apply exponential recency decay to existing counts.
      6. Upsert new counts.
      7. Advance the watermark.

    Bundling 5–7 into one transaction means a failure mid-write rolls
    everything back, so the next cycle re-walks the same window without
    double-counting (the watermark didn't move). If the read phase
    fails the cycle is skipped without side effects.

    Returns a small summary dict for logging/diagnostics. SQL errors
    propagate after a non-empty summary is logged, so the analysis
    pipeline records the step as FAILED and the next cycle retries.
    """
    now = now or dt_util.utcnow()
    summary: dict[str, Any] = {
        "transitions_recorded": 0,
        "events_walked": 0,
        "areas_considered": 0,
        "adjacency_pairs": 0,
    }

    try:
        last_observed_iso = _get_metadata(db, _watermark_key(entry_id))
    except SQLAlchemyError:
        _LOGGER.exception(
            "Skipping transition learning for entry %s: watermark read failed",
            entry_id,
        )
        return summary

    since: datetime | None = None
    if last_observed_iso:
        try:
            since = dt_util.parse_datetime(last_observed_iso)
        except (ValueError, TypeError):
            since = None

    if since is not None:
        hours_since_last = (now - since).total_seconds() / 3600
    else:
        # First run: don't decay (we have no aged data yet) and look back
        # over the full window.
        hours_since_last = 0.0

    adjacency_index = _build_adjacency_index(db, entry_id)
    summary["adjacency_pairs"] = sum(len(v) for v in adjacency_index.values())

    if not adjacency_index:
        # No configured adjacencies → nothing to learn. Still advance the
        # watermark so the next cycle starts from `now`. One small
        # transaction; failure here is logged but doesn't crash the
        # analysis step.
        try:
            with db.get_session() as session:
                _set_metadata_in_session(
                    session, db, _watermark_key(entry_id), now.isoformat()
                )
                session.commit()
        except SQLAlchemyError:
            _LOGGER.exception(
                "Failed to advance watermark for entry %s (no adjacency)",
                entry_id,
            )
        return summary

    area_names = sorted(adjacency_index.keys())
    summary["areas_considered"] = len(area_names)

    events = _collect_events_for_areas(db, entry_id, area_names, lookback_days, since)
    summary["events_walked"] = len(events)

    transitions = _detect_transitions(events, adjacency_index)
    summary["transitions_recorded"] = len(transitions)

    # Atomic write: decay + upsert + watermark in one transaction.
    try:
        with db.get_session() as session:
            _apply_recency_decay_in_session(session, db, entry_id, hours_since_last)
            _upsert_transition_counts_in_session(session, db, entry_id, transitions)
            _set_metadata_in_session(
                session, db, _watermark_key(entry_id), now.isoformat()
            )
            session.commit()
    except SQLAlchemyError:
        _LOGGER.exception(
            "Atomic transition write failed for entry %s — rolling back; "
            "next cycle will retry from the last successful watermark",
            entry_id,
        )
        # Reset the summary's recorded count so the caller's diagnostics
        # don't claim writes that were rolled back.
        summary["transitions_recorded"] = 0
        return summary

    _LOGGER.debug(
        "Transition learning for entry %s: %d transitions recorded across "
        "%d events from %d areas",
        entry_id,
        summary["transitions_recorded"],
        summary["events_walked"],
        summary["areas_considered"],
    )
    return summary


# ─── lookup: six-level smoothing fallback ────────────────────────────


# Public level labels — surfaced in diagnostics so users can see which
# fallback level decided their probability.
LEVEL_2HOP_HOUR_OF_WEEK = "2hop_hour_of_week"
LEVEL_2HOP_HOUR_OF_DAY = "2hop_hour_of_day"
LEVEL_2HOP_UNBUCKETED = "2hop_unbucketed"
LEVEL_1HOP_HOUR_OF_WEEK = "1hop_hour_of_week"
LEVEL_1HOP_UNBUCKETED = "1hop_unbucketed"
LEVEL_STATIC_DEFAULT = "static_default"


@dataclass(frozen=True)
class TransitionLookupResult:
    """Result of a transition probability lookup.

    ``probability`` is in [0, 1]. ``level`` is one of the ``LEVEL_*``
    constants describing which fallback layer fired. ``observed_count``
    is the count of the matching ``to_area`` observations at that
    level (0 for the static default). ``total_count`` is the total
    of the normalisation denominator at the level (count of all
    observed destinations from the same source).
    """

    probability: float
    level: str
    observed_count: float
    total_count: float


def _query_pair_counts(
    db: AreaOccupancyDB,
    entry_id: str,
    from_area: str,
    mid_area: str,
    *,
    hour_of_week: int | None = None,
    hour_of_day: int | None = None,
) -> dict[str, float]:
    """Sum counts grouped by ``to_area`` for the requested chain scope.

    ``hour_of_week`` filters to a specific (weekday, hour) bucket.
    ``hour_of_day`` (0..23) collapses across weekdays at a given hour.
    Pass neither to aggregate over the entire week.
    """
    try:
        with db.get_session() as session:
            q = session.query(
                db.AreaTransitions.to_area,
                db.AreaTransitions.count,
                db.AreaTransitions.hour_of_week,
            ).filter(
                db.AreaTransitions.entry_id == entry_id,
                db.AreaTransitions.from_area == from_area,
                db.AreaTransitions.mid_area == mid_area,
            )
            if hour_of_week is not None:
                q = q.filter(db.AreaTransitions.hour_of_week == hour_of_week)
            rows = q.all()
    except SQLAlchemyError:
        _LOGGER.exception("Error querying transition counts")
        return {}

    # SQLite has no integer modulo on a column expression that's clean
    # to filter on, so do hour_of_day collapsing in Python — the row
    # count is bounded by neighbours × hours so this is fine.
    sums: dict[str, float] = {}
    for to_area, count, hour in rows:
        if hour_of_day is not None and (hour % 24) != hour_of_day:
            continue
        sums[to_area] = sums.get(to_area, 0.0) + float(count or 0.0)
    return sums


def _try_level(
    sums: dict[str, float], to_area: str, threshold: int
) -> tuple[float, float, float] | None:
    """Return (probability, observed, total) if ``total`` ≥ threshold.

    The threshold is on the *total* observations at this level — once
    we trust the level, an unobserved destination is a real zero, not
    a "no data". Returns None to signal fall-through.
    """
    total = sum(sums.values())
    if total < threshold:
        return None
    observed = sums.get(to_area, 0.0)
    return (observed / total if total > 0 else 0.0, observed, total)


def lookup_transition_probability(
    db: AreaOccupancyDB,
    entry_id: str,
    *,
    from_area: str,
    mid_area: str,
    to_area: str,
    hour_of_week: int,
    static_default: float | None = None,
) -> TransitionLookupResult:
    """Look up ``P(to_area | from_area, mid_area, hour_of_week)`` with fallback.

    Walks six levels of progressively-wider scope until one has enough
    observations to trust:

    1. Specific 2-hop chain at the exact hour-of-week
    2. Specific 2-hop chain at the same hour-of-day (collapsed weekdays)
    3. Specific 2-hop chain un-bucketed
    4. Equivalent 1-hop chain at the exact hour-of-week
    5. Equivalent 1-hop chain un-bucketed
    6. Static default (``DEFAULT_INFLUENCE_WEIGHTS["adjacent"]`` unless
       overridden)

    Caller should normally pass ``mid_area`` populated for the 2-hop
    case. If only a 1-hop trajectory is available (no W known yet), pass
    ``mid_area=""`` to skip levels 1-3 and start at level 4.

    The thresholds come from ``ADJACENCY_N_*`` constants in ``const.py``.
    """
    if static_default is None:
        static_default = DEFAULT_INFLUENCE_WEIGHTS["adjacent"]
    hour_of_day = hour_of_week % 24

    # When the caller already knows there's no 2-hop trajectory, skip
    # straight to the 1-hop levels so we don't waste queries.
    if mid_area:
        # Level 1: specific chain at exact (weekday, hour)
        sums = _query_pair_counts(
            db, entry_id, from_area, mid_area, hour_of_week=hour_of_week
        )
        result = _try_level(sums, to_area, ADJACENCY_N_SPECIFIC)
        if result is not None:
            prob, observed, total = result
            return TransitionLookupResult(
                prob, LEVEL_2HOP_HOUR_OF_WEEK, observed, total
            )

        # Level 2: specific chain collapsed across weekdays
        sums = _query_pair_counts(
            db, entry_id, from_area, mid_area, hour_of_day=hour_of_day
        )
        result = _try_level(sums, to_area, ADJACENCY_N_HOUR)
        if result is not None:
            prob, observed, total = result
            return TransitionLookupResult(prob, LEVEL_2HOP_HOUR_OF_DAY, observed, total)

        # Level 3: specific chain un-bucketed
        sums = _query_pair_counts(db, entry_id, from_area, mid_area)
        result = _try_level(sums, to_area, ADJACENCY_N_CHAIN)
        if result is not None:
            prob, observed, total = result
            return TransitionLookupResult(prob, LEVEL_2HOP_UNBUCKETED, observed, total)

    # Level 4: 1-hop equivalent at exact hour-of-week
    sums = _query_pair_counts(db, entry_id, from_area, "", hour_of_week=hour_of_week)
    result = _try_level(sums, to_area, ADJACENCY_N_SPECIFIC)
    if result is not None:
        prob, observed, total = result
        return TransitionLookupResult(prob, LEVEL_1HOP_HOUR_OF_WEEK, observed, total)

    # Level 5: 1-hop equivalent un-bucketed
    sums = _query_pair_counts(db, entry_id, from_area, "")
    result = _try_level(sums, to_area, ADJACENCY_N_PAIR)
    if result is not None:
        prob, observed, total = result
        return TransitionLookupResult(prob, LEVEL_1HOP_UNBUCKETED, observed, total)

    # Level 6: static default. observed/total stay 0 to signal "no data".
    return TransitionLookupResult(static_default, LEVEL_STATIC_DEFAULT, 0.0, 0.0)


def summarize_transitions_for_diagnostics(
    db: AreaOccupancyDB, entry_id: str
) -> dict[str, Any]:
    """Lightweight summary for the diagnostics export.

    Returned shape::

        {"row_count": int, "1_hop_count": int, "2_hop_count": int,
         "total_observations": float, "last_observed": str | None}
    """
    out: dict[str, Any] = {
        "row_count": 0,
        "1_hop_count": 0,
        "2_hop_count": 0,
        "total_observations": 0.0,
        "last_observed": None,
    }
    # Watermark read is independent of the count query — handle each
    # separately so a single DB hiccup degrades the diagnostic gracefully
    # instead of returning nothing. ``_get_metadata`` propagates errors
    # by design (so the writer path can skip a cycle on read failure),
    # but the diagnostics consumer should tolerate them.
    try:
        out["last_observed"] = _get_metadata(db, _watermark_key(entry_id))
    except SQLAlchemyError:
        _LOGGER.exception("Error reading transition watermark for diagnostics")
    try:
        with db.get_session() as session:
            rows = (
                session.query(db.AreaTransitions)
                .filter(db.AreaTransitions.entry_id == entry_id)
                .all()
            )
            out["row_count"] = len(rows)
            for row in rows:
                if row.mid_area:
                    out["2_hop_count"] += 1
                else:
                    out["1_hop_count"] += 1
                out["total_observations"] += float(row.count or 0.0)
    except SQLAlchemyError:
        _LOGGER.exception("Error summarising transitions")
    return out
