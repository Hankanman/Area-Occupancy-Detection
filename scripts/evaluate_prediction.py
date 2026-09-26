#!/usr/bin/env python3
"""Offline evaluation for predictive occupancy (#502 step 1 — the go/no-go gate).

Replays area-entry events from a copy of ``area_occupancy.db`` and asks:
for each entry, would the trailing trajectory + hour-of-week bucket —
fed through the REAL six-level transition fallback
(``lookup_transition_distribution``, reused, not reimplemented) — have
predicted the area a person entered next? Reports top-1/top-2 accuracy
per area and per fallback level, plus the transition-gap distribution
that tells us what a "P(occupied within N minutes)" horizon could
honestly promise.

Per the issue's fence: if no operating point clears the bar
(precision ≥ 0.7 at usable recall), the prediction sensor never gets
built and this report is the record of why.

Usage:
    python scripts/evaluate_prediction.py [--db-path PATH] [--days DAYS]

Requirements on the source install: adjacency configured (the
``area_transitions`` table populated) and a warm
``occupied_intervals_cache`` (run ``run_analysis`` before copying).

Method notes:

* Entry events and trailing trajectories are reconstructed from the
  occupied-intervals cache with the same semantics as the live
  ``TrajectoryTracker``: an area joins the trajectory when its
  occupancy ENDS, only ends within the 300 s window
  (``ADJACENCY_TRAJECTORY_WINDOW_S``) count, the target area's own
  prior occupancy is skipped, and the most recent end is ``prev``.
* Evaluation is leave-nothing-out: the transition table being queried
  was itself learned from (a decayed version of) the same history, so
  accuracy reads optimistic. Good enough for a go/no-go; a shipped
  sensor would be scored live by the #499 harness.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys

# ruff: noqa: T201, E402
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from custom_components.area_occupancy.const import ADJACENCY_TRAJECTORY_WINDOW_S
from custom_components.area_occupancy.db.schema import AreaTransitions
from custom_components.area_occupancy.db.transitions import (
    LEVEL_STATIC_DEFAULT,
    lookup_transition_distribution,
)
from custom_components.area_occupancy.time_utils import to_local


class _DbStub:
    """The minimal surface ``lookup_transition_distribution`` reads.

    A real ``AreaOccupancyDB`` needs a live coordinator; the lookup only
    touches ``get_session()`` and the ``AreaTransitions`` model.
    """

    AreaTransitions = AreaTransitions

    def __init__(self, db_path: Path) -> None:
        self._engine = create_engine(f"sqlite:///file:{db_path}?mode=ro&uri=true")
        self._session_factory = sessionmaker(bind=self._engine)

    @contextmanager
    def get_session(self):
        session = self._session_factory()
        try:
            yield session
        finally:
            session.close()


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _load_occupancy_events(
    db: _DbStub, cutoff: datetime
) -> list[tuple[str, datetime, datetime]]:
    """Return (area, start, end) spans from the occupied-intervals cache."""
    with db.get_session() as session:
        rows = session.execute(
            text(
                "SELECT area_name, start_time, end_time FROM occupied_intervals_cache "
                "WHERE end_time >= :cutoff ORDER BY start_time"
            ),
            {"cutoff": cutoff.replace(tzinfo=None).isoformat(sep=" ")},
        ).fetchall()
    return [(area, _parse_dt(s), _parse_dt(e)) for area, s, e in rows]


def _entry_id(db: _DbStub) -> str | None:
    with db.get_session() as session:
        row = session.execute(text("SELECT DISTINCT entry_id FROM areas")).fetchone()
    return row[0] if row else None


def _trajectory_before(
    entry_time: datetime,
    target_area: str,
    ends: list[tuple[datetime, str]],
) -> tuple[str | None, str | None]:
    """Return (prev, prev_prev) mirroring TrajectoryTracker semantics."""
    window_start = entry_time - timedelta(seconds=ADJACENCY_TRAJECTORY_WINDOW_S)
    recent = [
        area
        for end, area in reversed(ends)
        if window_start <= end <= entry_time and area != target_area
    ]
    deduped: list[str] = []
    for area in recent:
        if area not in deduped:
            deduped.append(area)
        if len(deduped) == 2:
            break
    prev = deduped[0] if deduped else None
    prev_prev = deduped[1] if len(deduped) > 1 else None
    return prev, prev_prev


def main() -> int:
    """Replay entry events and score next-area predictions."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--db-path",
        default="config/.storage/area_occupancy.db",
        help="Path to a COPY of area_occupancy.db (never the live file)",
    )
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    db = _DbStub(db_path)
    entry_id = _entry_id(db)
    if entry_id is None:
        print("No areas table / entry_id found", file=sys.stderr)
        return 1

    cutoff = datetime.now(tz=UTC) - timedelta(days=args.days)
    spans = _load_occupancy_events(db, cutoff)
    if not spans:
        print(
            "occupied_intervals_cache is empty for the window — run "
            "run_analysis on the source install before copying the DB.",
            file=sys.stderr,
        )
        return 1

    ends = sorted((end, area) for area, _start, end in spans)

    total = 0
    with_trajectory = 0
    top1 = 0
    top2 = 0
    by_level: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # level -> [hits, n]
    by_area: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    gaps: list[float] = []

    for area, start, _end in spans:
        total += 1
        prev, prev_prev = _trajectory_before(start, area, ends)
        if prev is None:
            continue
        with_trajectory += 1

        # Gap from the trajectory's most recent exit to this entry.
        window_start = start - timedelta(seconds=ADJACENCY_TRAJECTORY_WINDOW_S)
        prev_end = max(
            (end for end, a in ends if a == prev and window_start <= end <= start),
            default=None,
        )
        if prev_end is not None:
            gaps.append((start - prev_end).total_seconds())

        local = to_local(start)
        hour_of_week = local.weekday() * 24 + local.hour
        dist = lookup_transition_distribution(
            db,
            entry_id,
            from_area=prev_prev or "",
            mid_area=prev if prev_prev else "",
            hour_of_week=hour_of_week,
        )
        if dist.level == LEVEL_STATIC_DEFAULT and prev_prev:
            # 2-hop walk found nothing at all; retry as pure 1-hop.
            dist = lookup_transition_distribution(
                db, entry_id, from_area=prev, mid_area="", hour_of_week=hour_of_week
            )

        ranked = sorted(
            dist.probabilities.items(), key=lambda item: item[1], reverse=True
        )
        predictions = [dest for dest, _p in ranked[:2]]
        hit1 = bool(predictions) and predictions[0] == area
        hit2 = area in predictions
        top1 += hit1
        top2 += hit2
        by_level[dist.level][0] += hit1
        by_level[dist.level][1] += 1
        by_area[area][0] += hit1
        by_area[area][1] += 1

    print(f"Entry events: {total}; with usable trajectory: {with_trajectory}")
    if not with_trajectory:
        print("Nothing predictable — no trajectories within the window.")
        return 0
    print(
        f"Top-1 accuracy: {top1 / with_trajectory:.3f}  "
        f"Top-2 accuracy: {top2 / with_trajectory:.3f}  "
        f"(recall base: {with_trajectory / total:.3f} of entries had a trajectory)"
    )
    print("\nBy fallback level:")
    for level, (hits, n) in sorted(by_level.items()):
        print(f"  {level}: {hits}/{n} = {hits / n:.3f}")
    print("\nBy target area (top-1):")
    for area, (hits, n) in sorted(by_area.items()):
        print(f"  {area}: {hits}/{n} = {hits / n:.3f}")
    if gaps:
        gaps.sort()
        pct = lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]  # noqa: E731
        print(
            f"\nExit->entry gap seconds (what a 'within N minutes' horizon "
            f"could promise): p50={pct(0.5):.0f} p90={pct(0.9):.0f} "
            f"p99={pct(0.99):.0f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
