#!/usr/bin/env python3
"""Recompute the global prior by hand and diff it against the stored value.

The #483 quiet-tail-inflation detector: rebuilds the prior from raw motion
intervals and compares both the buggy and fixed formulas to what is stored.

Usage:
    python prior_quiet_tail_diff.py <db_path> <area_name> [--now ISO8601_UTC]

Example:
    python prior_quiet_tail_diff.py config/.storage/area_occupancy.db "Kitchen"
    python prior_quiet_tail_diff.py config/.storage/area_occupancy.db "Kitchen" --now 2026-07-06T12:00:00

Background (re-verified 2026-07-06 against data/analysis.py at main HEAD
17b71d2, post-merge):

Production computes, in PriorAnalyzer.calculate_and_update_prior()
(custom_components/area_occupancy/data/analysis.py):

    occupied_duration = sum(end - start for each occupied interval)
    actual_period_duration = actual_period_end - first_interval_start
    global_prior = clamp(occupied_duration / actual_period_duration, 0.01, 0.99)

SETTLED HISTORY: before PR #491 (merged 2026-07-06), `actual_period_end` was
`last_interval_end` whenever the area had been quiet for more than 1 hour,
which silently dropped the known-unoccupied quiet tail from the denominator
and inflated the prior every hourly recalculation that ran during a quiet
stretch (overnight, weekends, vacations) -- this was issue #483 (auto-closed
by #491's merge), and it could pin a prior at the 0.99 ceiling for an area
with a true occupancy rate of ~30%. PR #491's fix makes `actual_period_end`
always `now`, and that fix is on `main` as of 2026-07-06 -- it has not yet
reached a tagged release (integration version is still 2026.5.17), so the
"buggy vs. fixed" comparison below still matters for any database running an
older release.

This script recomputes BOTH formulas from the raw `intervals` table and
diffs them against whatever is actually stored in `global_priors`, so you
can tell which behavior a live database is exhibiting without reading logs.

Approximation notice: this script uses motion "on" intervals only (the
dominant ground-truth signal). Production's query
(db/queries.py::get_occupied_intervals) also merges in media/sleep presence
intervals and extends motion intervals by `motion_timeout_seconds`. For
areas with no media/sleep sensors configured, motion-only replicates
production almost exactly; for areas that do use media/sleep for ground
truth, treat this script's occupied_duration as a slight underestimate, not
an exact match -- the direction and rough magnitude of any quiet-tail
inflation are still valid.

Interpretation guide:
  - `stored` ~= `buggy_formula` AND `buggy_formula` >> `fixed_formula`
    -> this database is exhibiting the #483 pattern. Check the integration
    version (`aod-change-control`) -- if it predates the #491 fix (merged to
    `main` 2026-07-06, not yet in a tagged release as of this writing),
    upgrading once it ships resolves it; the stored prior will self-correct
    on the next hourly analysis cycle after upgrade (no manual DB surgery
    needed).
  - `stored` ~= `fixed_formula` -> already running the fixed logic, or the
    area has no long quiet gaps in its interval history so both formulas
    agree.
  - Large diff between `buggy_formula` and `fixed_formula` but `stored` is
    close to neither -> something else changed the stored value (manual
    override, a different code path, or the DB predates both formulas);
    don't assume #483 without checking `calculation_date` recency.
"""
# ruff: noqa: T201, D103  # CLI diagnostic script: print IS the output

from __future__ import annotations

import datetime as dt
import sqlite3
import sys


def _parse_iso(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def _merge_intervals(
    intervals: list[tuple[dt.datetime, dt.datetime]],
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Merge overlapping/touching (start, end) intervals, sorted by start."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def diff_prior(db_path: str, area_name: str, now: dt.datetime | None = None) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if now is None:
        now = dt.datetime.now(dt.UTC).replace(tzinfo=None)

    # Motion "on" intervals for this area (naive UTC, matches DB storage --
    # see schema.py's _utcnow_db(): "naive UTC for SQLite persistence").
    cur.execute(
        """
        SELECT i.start_time, i.end_time
        FROM intervals i
        JOIN entities e
          ON i.entity_id = e.entity_id AND i.area_name = e.area_name
        WHERE i.area_name = ?
          AND e.entity_type = 'motion'
          AND i.state = 'on'
        ORDER BY i.start_time
        """,
        (area_name,),
    )
    rows = cur.fetchall()
    if not rows:
        print(
            f"No motion intervals found for area '{area_name}'. Nothing to recompute."
        )
        conn.close()
        return

    raw_intervals = [
        (_parse_iso(r["start_time"]), _parse_iso(r["end_time"])) for r in rows
    ]
    merged = _merge_intervals(raw_intervals)

    first_start = merged[0][0]
    last_end = max(end for _, end in merged)
    occupied_duration = sum((end - start).total_seconds() for start, end in merged)

    # Pre-#491 (buggy): truncate at last_interval_end if quiet > 1h.
    if (now - last_end).total_seconds() > 3600:
        buggy_period_end = last_end
    else:
        buggy_period_end = now
    buggy_period_duration = (buggy_period_end - first_start).total_seconds()
    buggy_prior = (
        max(0.01, min(0.99, occupied_duration / buggy_period_duration))
        if buggy_period_duration > 0
        else None
    )

    # Post-#491 (fixed): period always ends at now.
    fixed_period_duration = (now - first_start).total_seconds()
    fixed_prior = (
        max(0.01, min(0.99, occupied_duration / fixed_period_duration))
        if fixed_period_duration > 0
        else None
    )

    print(f"=== Prior recompute for area '{area_name}' ===")
    print(f"  now                 = {now.isoformat()}")
    print(f"  first_interval_start= {first_start.isoformat()}")
    print(f"  last_interval_end   = {last_end.isoformat()}")
    print(
        f"  quiet tail          = {(now - last_end).total_seconds() / 3600:.1f}h since last motion"
    )
    print(
        f"  occupied_duration   = {occupied_duration / 3600:.2f}h ({len(merged)} merged motion intervals)"
    )
    print()
    print(
        f"  buggy formula  (pre-#491, truncates quiet tail): period={buggy_period_duration / 86400:.2f}d -> prior={buggy_prior:.4f}"
        if buggy_prior is not None
        else "  buggy formula: invalid period"
    )
    print(
        f"  fixed formula  (#491, period always to now):     period={fixed_period_duration / 86400:.2f}d -> prior={fixed_prior:.4f}"
        if fixed_prior is not None
        else "  fixed formula: invalid period"
    )

    cur.execute(
        "SELECT prior_value, calculation_date FROM global_priors WHERE area_name = ?",
        (area_name,),
    )
    stored = cur.fetchone()
    print()
    if stored is None:
        print("  stored global_priors row: NONE (never calculated)")
    else:
        stored_val = stored["prior_value"]
        print(
            f"  stored prior_value  = {stored_val:.4f}  (calculated {stored['calculation_date']})"
        )
        # Relative gap, not absolute: priors for quiet areas are often small
        # (0.05-0.15), where even a "small" absolute 0.02-0.03 gap is a large
        # percentage inflation. Flag anything >= 15% relative inflation.
        relative_gap = (
            (buggy_prior - fixed_prior) / buggy_prior
            if buggy_prior is not None and fixed_prior is not None and buggy_prior > 0
            else 0.0
        )
        if (
            buggy_prior is not None
            and abs(stored_val - buggy_prior) < 0.01
            and (fixed_prior is not None and relative_gap > 0.15)
        ):
            print(
                "  -> DIAGNOSIS: stored value matches the pre-#491 buggy formula and "
                "differs meaningfully from the fixed formula. This area is likely "
                "exhibiting the #483 quiet-tail inflation pattern."
            )
        elif fixed_prior is not None and abs(stored_val - fixed_prior) < 0.01:
            print("  -> DIAGNOSIS: stored value matches the fixed (#491) formula.")
        else:
            print(
                "  -> DIAGNOSIS: stored value doesn't cleanly match either formula "
                "-- check calculation_date recency before assuming #483."
            )

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    db_path = sys.argv[1]
    area = sys.argv[2]
    now_arg = None
    if len(sys.argv) > 4 and sys.argv[3] == "--now":
        now_arg = _parse_iso(sys.argv[4])
    diff_prior(db_path, area, now_arg)
