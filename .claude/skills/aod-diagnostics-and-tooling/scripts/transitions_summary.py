#!/usr/bin/env python3
"""Summarize AreaTransitions learning state (adjacency / transition learning).

Usage:
    python transitions_summary.py <db_path> [--top N]

Example:
    python transitions_summary.py config/.storage/area_occupancy.db --top 10

Requires the `area_transitions` table, which only exists once the
adjacent-areas feature (PR #454) is on the running branch/release --
merged into `main` 2026-07-06 (HEAD 17b71d2). Running this script against
a database from a release that predates the merge (e.g. the 2026.5.17
release) will just report "table not found".

Schema reference (custom_components/area_occupancy/db/schema.py::AreaTransitions,
verified 2026-07-06 against main HEAD 17b71d2): from_area, mid_area, to_area, hour_of_week (0-167),
count (float, recency-decayed), smoothed_prob, updated_at. A row is a "chain":
2-hop when mid_area is non-empty (mid_area -> from_area -> to_area was the
observed path), 1-hop when mid_area == '' (empty-string sentinel, not NULL,
so the UniqueConstraint on chain+hour actually enforces uniqueness in SQLite).

Minimum-observation thresholds this script checks each chain against
(const.py, "First-pass values; tune from real data" -- unvalidated on real
homes as of 2026-07-06):
  ADJACENCY_N_SPECIFIC = 5   (2-hop, exact hour-of-week bucket)
  ADJACENCY_N_HOUR     = 20  (2-hop, day-collapsed hour-of-day)
  ADJACENCY_N_CHAIN    = 50  (2-hop, unbucketed)
  ADJACENCY_N_PAIR     = 20  (1-hop, unbucketed)
  ADJACENCY_RECENCY_HALF_LIFE_DAYS = 30 (count decays by half every 30 days
    of analysis cycles without a fresh observation -- a chain with a recent
    `updated_at` but a low `count` has genuinely low traffic, not just old data)

Interpretation guide:
  - `total_count` for a chain below ADJACENCY_N_PAIR (20, for 1-hop) or
    ADJACENCY_N_CHAIN (50, for 2-hop unbucketed) means the adjacency lookup
    for that chain is still falling back toward LEVEL_STATIC_DEFAULT (the
    hand-configured influence_weight in area_relationships) -- learning
    hasn't taken over yet for that pair.
  - `hour_buckets` close to 168 with reasonable `total_count` per bucket
    means the chain has enough data to use the finest-grained lookup level
    (LEVEL_2HOP_HOUR_OF_WEEK / LEVEL_1HOP_HOUR_OF_WEEK).
  - `days_since_update` much larger than ADJACENCY_RECENCY_HALF_LIFE_DAYS
    (30) means this chain's `count` has decayed substantially since the last
    real observation and its influence on today's boost/decay-modifier math
    is now small even if the raw historical count once looked large.
  - Compare `top chains by total_count` against your actual configured
    `adjacent_areas` (see the `areas.adjacent_areas` column or the
    diagnostics export's per-area config) -- a chain that doesn't correspond
    to any configured adjacency is stale data from a since-removed pairing.
"""
# ruff: noqa: T201, D103  # CLI diagnostic script: print IS the output

from __future__ import annotations

import datetime as dt
import sqlite3
import sys

ADJACENCY_N_SPECIFIC = 5
ADJACENCY_N_HOUR = 20
ADJACENCY_N_CHAIN = 50
ADJACENCY_N_PAIR = 20
ADJACENCY_RECENCY_HALF_LIFE_DAYS = 30


def summarize(db_path: str, top_n: int = 10) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute("SELECT COUNT(*) AS n FROM area_transitions")
    except sqlite3.OperationalError as err:
        print(f"area_transitions table not found ({err}).")
        print(
            "This DB predates the adjacent-areas feature (PR #454, merged "
            "2026-07-06). Verify the running build includes it with: gh pr view 454"
        )
        conn.close()
        return

    total_rows = cur.fetchone()["n"]
    print("=== AreaTransitions summary ===")
    print(f"  total rows (chain x hour-of-week buckets) = {total_rows}")
    if total_rows == 0:
        print(
            "  No transitions recorded yet -- adjacency is falling back to static config-weight for every pair."
        )
        conn.close()
        return

    cur.execute(
        """
        SELECT from_area, mid_area, to_area,
               SUM(count) AS total_count,
               COUNT(*) AS hour_buckets,
               MAX(updated_at) AS most_recent
        FROM area_transitions
        GROUP BY from_area, mid_area, to_area
        ORDER BY total_count DESC
        """
    )
    chains = cur.fetchall()
    print(f"  distinct chains = {len(chains)}")

    n_1hop = sum(1 for c in chains if c["mid_area"] == "")
    n_2hop = len(chains) - n_1hop
    print(f"  1-hop chains (mid_area='') = {n_1hop}   2-hop chains = {n_2hop}")

    now = dt.datetime.now(dt.UTC).replace(tzinfo=None)
    print()
    print(f"  Top {min(top_n, len(chains))} chains by total observation count:")
    header = f"  {'chain':<45} {'hops':>4} {'total_count':>11} {'hour_buckets':>12} {'days_since_upd':>15} {'level_cleared':>20}"
    print(header)
    for c in chains[:top_n]:
        chain_label = (
            f"{c['from_area']} -> {c['to_area']}"
            if c["mid_area"] == ""
            else f"{c['mid_area']} -> {c['from_area']} -> {c['to_area']}"
        )
        hops = "1hop" if c["mid_area"] == "" else "2hop"
        try:
            updated = dt.datetime.fromisoformat(c["most_recent"])
            days_since = (now - updated).total_seconds() / 86400
        except TypeError, ValueError:
            days_since = float("nan")

        total_count = c["total_count"] or 0.0
        if hops == "2hop":
            if total_count >= ADJACENCY_N_SPECIFIC:
                level = "2hop_hour_of_week*"
            elif total_count >= ADJACENCY_N_HOUR:
                level = "2hop_hour_of_day"
            elif total_count >= ADJACENCY_N_CHAIN:
                level = "2hop_unbucketed"
            else:
                level = "static_default"
        elif total_count >= ADJACENCY_N_SPECIFIC:
            level = "1hop_hour_of_week*"
        elif total_count >= ADJACENCY_N_PAIR:
            level = "1hop_unbucketed"
        else:
            level = "static_default"

        print(
            f"  {chain_label:<45} {hops:>4} {total_count:>11.1f} {c['hour_buckets']:>12} "
            f"{days_since:>15.1f} {level:>20}"
        )
    print(
        "  (*'_hour_of_week' level requires meeting the threshold at the SPECIFIC "
        "hour-of-week bucket being queried, not just in aggregate -- this column "
        "shows the best case, i.e. the chain's total across all 168 buckets. A "
        "chain can clear the aggregate threshold while still falling back to a "
        "coarser level for hour-of-week buckets with little traffic.)"
    )

    stale_cutoff = ADJACENCY_RECENCY_HALF_LIFE_DAYS * 2
    stale = [
        c
        for c in chains
        if c["most_recent"]
        and (now - dt.datetime.fromisoformat(c["most_recent"])).total_seconds() / 86400
        > stale_cutoff
    ]
    if stale:
        print()
        print(
            f"  {len(stale)} chain(s) not updated in > {stale_cutoff} days "
            f"(2x the {ADJACENCY_RECENCY_HALF_LIFE_DAYS}-day recency half-life) -- "
            "their learned influence has decayed substantially:"
        )
        for c in stale:
            chain_label = (
                f"{c['from_area']} -> {c['to_area']}"
                if c["mid_area"] == ""
                else f"{c['mid_area']} -> {c['from_area']} -> {c['to_area']}"
            )
            print(f"    {chain_label}  (last updated {c['most_recent']})")

    conn.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    db_path = sys.argv[1]
    top = 10
    if "--top" in sys.argv:
        idx = sys.argv.index("--top")
        top = int(sys.argv[idx + 1])
    summarize(db_path, top)
