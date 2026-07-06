#!/usr/bin/env python3
"""Dump learned priors (global + 168 time-of-week buckets) for one area.

Usage:
    python dump_priors.py <db_path> <area_name>

Example:
    python dump_priors.py config/.storage/area_occupancy.db "Living Room"

Reads two tables directly with sqlite3 (no Home Assistant import required):
  - global_priors  : one row per area, the long-run occupied-fraction prior
  - priors         : up to 168 rows per area (day_of_week 0-6 x time_slot 0-23),
                     the learned time-of-week prior

Schema reference (verified against custom_components/area_occupancy/db/schema.py
as of 2026-07-06, integration version 2026.5.17):
  - GlobalPriors: area_name, prior_value, calculation_date, data_period_start,
    data_period_end, total_occupied_seconds, total_period_seconds,
    interval_count, confidence, calculation_method
  - Priors: area_name, day_of_week, time_slot, prior_value, data_points,
    confidence, last_calculation_date, calculation_method

Interpretation guide:
  - `global_prior` near 0.5 with very few `interval_count` -> not enough
    history yet, this is close to the MIN_PRIOR/MAX_PRIOR-clamped default,
    not a learned value. Cross-check `total_period_seconds` against how long
    the area has actually been configured.
  - `global_prior` pinned at or very near 0.99 -> suspect the #483
    quiet-tail inflation bug family; run prior_quiet_tail_diff.py next.
  - Time-prior coverage (`buckets_populated` below) less than 168 means some
    day-of-week x hour combinations have never been analyzed (usually areas
    younger than ~1-2 weeks, since the pipeline needs enough interval data
    per bucket, not just wall-clock time).
  - A time_prior bucket wildly different from its neighbours (e.g. one hour
    at 0.9 surrounded by hours at 0.1) with a low `data_points` count is
    likely a small-sample artifact, not a real behavioral pattern -- check
    `confidence` for that row.
"""
# ruff: noqa: T201, D103  # CLI diagnostic script: print IS the output

from __future__ import annotations

import sqlite3
import sys


def dump_priors(db_path: str, area_name: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    print(f"=== Global prior for area '{area_name}' ===")
    cur.execute(
        """
        SELECT prior_value, calculation_date, data_period_start, data_period_end,
               total_occupied_seconds, total_period_seconds, interval_count,
               confidence, calculation_method
        FROM global_priors
        WHERE area_name = ?
        """,
        (area_name,),
    )
    row = cur.fetchone()
    if row is None:
        print("  No global_priors row found for this area (never calculated yet).")
    else:
        occupied_h = (row["total_occupied_seconds"] or 0) / 3600
        period_d = (row["total_period_seconds"] or 0) / 86400
        print(f"  prior_value            = {row['prior_value']:.4f}")
        print(f"  calculation_date       = {row['calculation_date']}")
        print(
            f"  data_period             = {row['data_period_start']} -> {row['data_period_end']}"
        )
        print(
            f"  total_occupied_seconds  = {row['total_occupied_seconds']} ({occupied_h:.1f}h)"
        )
        print(
            f"  total_period_seconds    = {row['total_period_seconds']} ({period_d:.1f}d)"
        )
        print(f"  interval_count          = {row['interval_count']}")
        print(f"  confidence              = {row['confidence']}")
        print(f"  calculation_method      = {row['calculation_method']}")
        if period_d > 0:
            recomputed = max(
                0.01,
                min(
                    0.99,
                    (row["total_occupied_seconds"] or 0)
                    / (row["total_period_seconds"] or 1),
                ),
            )
            print(
                f"  sanity check: occupied/period = {recomputed:.4f} (should equal prior_value above)"
            )

    print()
    print(f"=== Time-of-week priors for area '{area_name}' ===")
    cur.execute(
        """
        SELECT day_of_week, time_slot, prior_value, data_points, confidence,
               calculation_method
        FROM priors
        WHERE area_name = ?
        ORDER BY day_of_week, time_slot
        """,
        (area_name,),
    )
    rows = cur.fetchall()
    if not rows:
        print(
            "  No priors rows found for this area (time-of-week learning hasn't run yet)."
        )
        conn.close()
        return

    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"  buckets_populated = {len(rows)} / 168")
    print(
        f"  {'day':>4} {'slot':>4}  {'prior':>6}  {'data_pts':>8}  {'confidence':>10}"
    )
    for r in rows:
        dow_label = (
            dow_names[r["day_of_week"]]
            if 0 <= r["day_of_week"] <= 6
            else str(r["day_of_week"])
        )
        conf = "None" if r["confidence"] is None else f"{r['confidence']:.2f}"
        print(
            f"  {dow_label:>4} {r['time_slot']:>4}  {r['prior_value']:>6.3f}  "
            f"{r['data_points']:>8}  {conf:>10}"
        )

    values = [r["prior_value"] for r in rows]
    print()
    print(
        f"  min={min(values):.3f}  max={max(values):.3f}  avg={sum(values) / len(values):.3f}"
    )

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    dump_priors(sys.argv[1], sys.argv[2])
