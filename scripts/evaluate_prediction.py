#!/usr/bin/env python3
"""Replay historical occupancy events to evaluate next-area predictions (#502).

This script reads a copy of the Home Assistant area_occupancy database and
checks whether the integration's saved transition model would have ranked
the area that was actually entered among its most likely destinations.
It uses the production adjacency fallback rather than reimplementing it.

Usage:
    python scripts/evaluate_prediction.py --time-zone Europe/Berlin \
        --db-path /path/to/area_occupancy.db --days 30

The database copy must contain exactly one configuration entry in the areas
table, a populated occupied_intervals_cache, and learned adjacency data.
Run the integration's analysis before copying the database if the cache
needs to be populated. Use a copy, not the live database. Set --time-zone
to the source installation's IANA time zone so the replay queries the
same local hour-of-week buckets used to learn the transitions.

Replay method:
    Only occupancy intervals that start inside the evaluation window
    count as entry events. Earlier exits are retained as context if
    they fall within one trajectory window before the cutoff.

    For each entry, the script looks backward for up to two distinct
    areas whose occupancy ended within the trajectory window. It skips
    prior occupancy of the target area. It then queries the production
    adjacency snapshot using that trajectory and the entry's local
    hour-of-week bucket, and ranks the returned destinations.

Reading the report:
    Trajectory coverage is the share of all evaluated entries with
    usable prior trajectory context. Prediction coverage is the share
    for which the lookup actually returns at least one destination.

    Top-1 accuracy and the top-2 hit rate are calculated only for entries
    with predictions. The report also shows correct top-1 predictions
    across all entries, so low prediction coverage is visible even when
    accuracy on the predicted subset is high.

    Exit-to-entry gap percentiles describe observed gaps for entries
    with predictions. They do not measure advance warning or validate
    a claim that an area will be occupied within a given time horizon.

Limitations:
    This is an exploratory replay, not an independent test or a release
    gate. The saved transition counts may have been learned from some
    of the same events being evaluated, which can inflate accuracy.
    To assess live predictive performance, train on an earlier period
    and score on a separate, later period without using future events.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# ruff: noqa: T201, E402
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from custom_components.area_occupancy.const import ADJACENCY_TRAJECTORY_WINDOW_S
from custom_components.area_occupancy.db.schema import (
    AreaRelationships,
    AreaTransitions,
)
from custom_components.area_occupancy.db.transitions import load_adjacency_snapshot
from custom_components.area_occupancy.time_utils import to_local
from homeassistant.util import dt as dt_util


class _DbStub:
    """Provide the session and models used by the production snapshot loader."""

    AreaTransitions = AreaTransitions
    AreaRelationships = AreaRelationships

    def __init__(self, database_uri: str) -> None:
        self._engine = create_engine(
            "sqlite://", creator=lambda: sqlite3.connect(database_uri, uri=True)
        )
        self._session_factory = sessionmaker(bind=self._engine)

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        with self._session_factory() as session:
            yield session

    def close(self) -> None:
        """Release the database connections used for the snapshot."""
        self._engine.dispose()


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _entry_id(connection: sqlite3.Connection) -> str:
    """Reject multi-entry copies instead of choosing an arbitrary entry."""
    rows = connection.execute("SELECT DISTINCT entry_id FROM areas LIMIT 2").fetchall()
    if len(rows) != 1 or not rows[0][0]:
        msg = "Expected exactly one entry_id in the areas table."
        raise ValueError(msg)
    return str(rows[0][0])


def _load_occupancy_events(
    connection: sqlite3.Connection,
    entry_id: str,
    cutoff: datetime,
    until: datetime,
) -> tuple[list[tuple[datetime, str]], list[tuple[datetime, str]]]:
    """Return in-window starts and exits, including trajectory warm-up exits."""
    warmup = cutoff - timedelta(seconds=ADJACENCY_TRAJECTORY_WINDOW_S)
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(occupied_intervals_cache)")
    }
    sql = "SELECT area_name, start_time, end_time FROM occupied_intervals_cache"
    if "entry_id" in columns:
        rows = connection.execute(sql + " WHERE entry_id = ?", (entry_id,))
    else:
        # Older caches are keyed by area rather than by configuration entry.
        rows = connection.execute(
            sql + " WHERE area_name IN (SELECT area_name FROM areas WHERE entry_id = ?)",
            (entry_id,),
        )

    entries: list[tuple[datetime, str]] = []
    ends: list[tuple[datetime, str]] = []
    # Compare parsed UTC timestamps, not potentially mixed SQLite text formats.
    for area, start_raw, end_raw in rows:
        start, end = _parse_dt(start_raw), _parse_dt(end_raw)
        if start >= end:
            continue
        if cutoff <= start <= until:
            entries.append((start, area))
        if warmup <= end <= until:
            ends.append((end, area))
    return sorted(entries), sorted(ends)


def _trajectory_before(
    entry_time: datetime,
    target_area: str,
    ends: list[tuple[datetime, str]],
    end_times: list[datetime],
) -> tuple[str | None, str | None, datetime | None]:
    """Return the two most recent distinct other areas and the latest exit."""
    window_start = entry_time - timedelta(seconds=ADJACENCY_TRAJECTORY_WINDOW_S)
    index = bisect_right(end_times, entry_time) - 1
    prev: str | None = None
    prev_end: datetime | None = None
    while index >= 0:
        end, area = ends[index]
        if end < window_start:
            break
        if area != target_area:
            if prev is None:
                prev, prev_end = area, end
            elif area != prev:
                return prev, area, prev_end
        index -= 1
    return prev, None, prev_end


def main() -> int:
    """Replay entry events and score next-area rankings."""
    parser = argparse.ArgumentParser(
        description="Replay historical occupancy entries."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("config/.storage/area_occupancy.db"),
        help="Path to a COPY of area_occupancy.db (never the live file)",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--time-zone", required=True, help="Source IANA time zone")
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")
    try:
        dt_util.set_default_time_zone(ZoneInfo(args.time_zone))
    except (ValueError, ZoneInfoNotFoundError):
        parser.error(f"Invalid IANA time zone: {args.time_zone}")
    if not args.db_path.is_file():
        print(f"Database file not found: {args.db_path}", file=sys.stderr)
        return 1
    database_uri = args.db_path.resolve().as_uri() + "?mode=ro"
    until = datetime.now(tz=UTC)
    try:
        cutoff = until - timedelta(days=args.days)
        with closing(sqlite3.connect(database_uri, uri=True)) as connection:
            entry_id = _entry_id(connection)
            entries, ends = _load_occupancy_events(connection, entry_id, cutoff, until)
    except (sqlite3.Error, TypeError, ValueError, OverflowError) as exc:
        print(f"Could not read occupancy history: {exc}", file=sys.stderr)
        return 1
    if not entries:
        print("No cached entry events in the evaluation window.", file=sys.stderr)
        return 1

    with closing(_DbStub(database_uri)) as db:
        snapshot = load_adjacency_snapshot(db, entry_id)
    if snapshot is None:
        print("Failed to load the adjacency snapshot.", file=sys.stderr)
        return 1

    end_times = [end for end, _ in ends]
    with_trajectory = predicted = top1 = top2 = 0
    by_level: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_area: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    gaps: list[float] = []

    for area, start in entries:
        prev, prev_prev, prev_end = _trajectory_before(start, area, ends, end_times)
        if prev is None or prev_end is None:
            continue
        with_trajectory += 1
        local = to_local(start)
        dist = snapshot.lookup_distribution(
            from_area=prev if prev_prev is None else prev_prev,
            mid_area="" if prev_prev is None else prev,
            hour_of_week=local.weekday() * 24 + local.hour,
        )
        ranked = sorted(
            dist.probabilities.items(), key=lambda item: item[1], reverse=True
        )
        if not ranked:
            continue
        predicted += 1
        gaps.append((start - prev_end).total_seconds())
        hit1 = int(ranked[0][0] == area)
        top1 += hit1
        top2 += int(any(dest == area for dest, _ in ranked[:2]))
        by_level[dist.level][0] += hit1
        by_level[dist.level][1] += 1
        by_area[area][0] += hit1
        by_area[area][1] += 1

    total = len(entries)
    print("Exploratory replay: transition counts may include evaluation events.")
    print(f"Entry events: {total}")
    print(
        f"Trajectory coverage: {with_trajectory}/{total} = "
        f"{with_trajectory / total:.3f}"
    )
    print(f"Prediction coverage: {predicted}/{total} = {predicted / total:.3f}")
    if not predicted:
        print("No predictions returned. Accuracy is undefined.")
        return 0
    print(f"Top-1 accuracy on predictions: {top1}/{predicted} = {top1 / predicted:.3f}")
    print(f"Top-2 hit rate on predictions: {top2}/{predicted} = {top2 / predicted:.3f}")
    print(
        f"Correct top-1 predictions across all entries: {top1}/{total} = "
        f"{top1 / total:.3f}"
    )
    for label, scores in (("fallback level", by_level), ("target area", by_area)):
        print(f"\nBy {label} (top-1, predictions only):")
        for key, (hits, count) in sorted(scores.items()):
            print(f"  {key}: {hits}/{count} = {hits / count:.3f}")

    gaps.sort()
    # Preserve the previous empirical percentile convention.
    percentiles = (
        gaps[min(len(gaps) - 1, int(p * len(gaps)))] for p in (0.5, 0.9, 0.99)
    )
    p50, p90, p99 = percentiles
    print(
        f"\nObserved exit-to-entry gaps (seconds): "
        f"p50={p50:.0f} p90={p90:.0f} p99={p99:.0f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
