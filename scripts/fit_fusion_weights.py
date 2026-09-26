#!/usr/bin/env python3
"""Fit per-area fusion weights offline from a database copy (#501 step 1).

Reconstructs (evidence, occupied) training rows from a copy of
``area_occupancy.db``, fits per-entity logistic weights with plain
gradient descent (numpy, L2-anchored toward the shipped type defaults),
and reports the calibration of the replayed probability stream under
the default vs the fitted weights — the go/no-go evidence #501's first
step asks for, produced without shipping any live code.

Usage:
    python scripts/fit_fusion_weights.py [--db-path PATH] [--days DAYS]
        [--step-seconds N] [--epochs N] [--lr F] [--l2 F]

KNOWN LIMITATIONS (read before trusting the numbers):

* **0/1 evidence only.** Raw ``intervals`` store on/off state spans; the
  live pipeline's decaying 0..1 evidence middle is not reconstructable
  offline, so the replay under-represents decay periods. Online shadow
  learning (``data/fusion.py``) does not share this limitation.
* **No correlations.** The learned correlation multipliers are omitted
  (they change over time and are not versioned in the DB); features are
  ``active × prob_given_true × strength_multiplier``.
* **Ground truth = occupied_intervals_cache.** Run ``run_analysis`` on
  the source install before copying the DB so the cache is warm.
* **Binary types only.** Numeric/environmental evidence lives in
  ``numeric_samples`` with runtime-learned bounds; reconstructing it
  faithfully is out of scope for this first pass.
* **MOTION/SLEEP excluded** from fitting (self-labelling — the ground
  truth is derived from them), same rule as the shadow learner.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
import sys

# ruff: noqa: T201, E402
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

from custom_components.area_occupancy.data.entity_type import DEFAULT_TYPES, InputType
from custom_components.area_occupancy.data.metrics import (
    TickSample,
    compute_accuracy_metrics,
)
from custom_components.area_occupancy.utils import logit

EXCLUDED_TYPES = {"motion", "sleep"}


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _load_area_names(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("SELECT area_name FROM areas")]


def _load_occupied(
    conn: sqlite3.Connection, area: str, cutoff: datetime
) -> list[tuple[datetime, datetime]]:
    rows = conn.execute(
        "SELECT start_time, end_time FROM occupied_intervals_cache "
        "WHERE area_name = ? AND end_time >= ? ORDER BY start_time",
        (area, cutoff.replace(tzinfo=None).isoformat(sep=" ")),
    ).fetchall()
    return [(_parse_dt(s), _parse_dt(e)) for s, e in rows]


def _load_entities(conn: sqlite3.Connection, area: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for entity_id, entity_type, pgt in conn.execute(
        "SELECT entity_id, entity_type, prob_given_true FROM entities "
        "WHERE area_name = ?",
        (area,),
    ):
        if entity_type in EXCLUDED_TYPES:
            continue
        try:
            defaults = DEFAULT_TYPES[InputType(entity_type)]
        except (KeyError, ValueError):
            continue
        if not defaults.get("active_states"):
            continue  # numeric types: see limitations header
        out[entity_id] = {
            "active_states": set(defaults["active_states"]),
            "pgt": float(pgt),
            "strength_multiplier": float(defaults.get("strength_multiplier", 2.0)),
            "default_weight": float(defaults["weight"]),
        }
    return out


def _load_active_spans(
    conn: sqlite3.Connection,
    entity_id: str,
    active_states: set[str],
    cutoff: datetime,
) -> list[tuple[datetime, datetime]]:
    # The f-string only interpolates a generated "?,?,..." placeholder
    # list; every value goes through sqlite3 parameter binding.
    placeholders = ",".join("?" for _ in active_states)
    query = (
        "SELECT start_time, end_time FROM intervals "  # noqa: S608
        "WHERE entity_id = ? AND aggregation_level = 'raw' "
        f"AND state IN ({placeholders}) AND end_time >= ? ORDER BY start_time"
    )
    rows = conn.execute(
        query,
        (entity_id, *active_states, cutoff.replace(tzinfo=None).isoformat(sep=" ")),
    ).fetchall()
    return [(_parse_dt(s), _parse_dt(e)) for s, e in rows]


def _covered(ts: datetime, spans: list[tuple[datetime, datetime]]) -> bool:
    # Linear scan is fine at this scale; spans are per-entity per-window.
    return any(start <= ts < end for start, end in spans)


def _fit_area(
    conn: sqlite3.Connection,
    area: str,
    *,
    days: int,
    step_seconds: int,
    epochs: int,
    lr: float,
    l2: float,
) -> None:
    now = datetime.now(tz=UTC)
    cutoff = now - timedelta(days=days)
    occupied = _load_occupied(conn, area, cutoff)
    entities = _load_entities(conn, area)
    if not occupied or not entities:
        print(f"[{area}] skipped: occupied={len(occupied)} entities={len(entities)}")
        return
    spans = {
        eid: _load_active_spans(conn, eid, meta["active_states"], cutoff)
        for eid, meta in entities.items()
    }

    entity_ids = sorted(entities)
    scale = np.array(
        [entities[e]["pgt"] * entities[e]["strength_multiplier"] for e in entity_ids]
    )
    defaults = np.array([entities[e]["default_weight"] for e in entity_ids])

    # Sample the window on a fixed grid.
    times: list[datetime] = []
    t = cutoff
    while t < now:
        times.append(t)
        t += timedelta(seconds=step_seconds)
    x = (
        np.array(
            [
                [1.0 if _covered(ts, spans[e]) else 0.0 for e in entity_ids]
                for ts in times
            ]
        )
        * scale
    )  # feature = active × pgt × strength_multiplier
    y = np.array([1.0 if _covered(ts, occupied) else 0.0 for ts in times])
    base_rate = float(y.mean())
    bias = logit(min(max(base_rate, 0.01), 0.99))

    def replay(weights: np.ndarray) -> list[TickSample]:
        z = bias + x @ weights
        p = 1.0 / (1.0 + np.exp(-z))
        return [
            TickSample(timestamp=ts, probability=float(pi), occupied=bool(pi >= 0.5))
            for ts, pi in zip(times, p, strict=True)
        ]

    # Batch gradient descent on logistic loss + L2 toward defaults.
    w = defaults.copy()
    n = len(times)
    for _ in range(epochs):
        p = 1.0 / (1.0 + np.exp(-(bias + x @ w)))
        grad = (x.T @ (p - y)) / n + l2 * (w - defaults)
        w = np.clip(w - lr * grad, 0.0, 0.99)

    before = compute_accuracy_metrics(replay(defaults), occupied)
    after = compute_accuracy_metrics(replay(w), occupied)
    print(f"\n[{area}] samples={n} occupancy_rate={base_rate:.3f}")
    print(
        f"  default weights: ece={before.expected_calibration_error:.4f} "
        f"agreement={before.agreement:.3f}"
    )
    print(
        f"  fitted weights:  ece={after.expected_calibration_error:.4f} "
        f"agreement={after.agreement:.3f}"
    )
    for i, eid in enumerate(entity_ids):
        marker = " *" if abs(w[i] - defaults[i]) > 0.05 else ""
        print(f"    {eid}: default={defaults[i]:.3f} fitted={w[i]:.3f}{marker}")


def main() -> int:
    """Parse arguments and fit each area in the database copy."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--db-path",
        default="config/.storage/area_occupancy.db",
        help="Path to a COPY of area_occupancy.db (never the live file)",
    )
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--step-seconds", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for area in _load_area_names(conn):
            _fit_area(
                conn,
                area,
                days=args.days,
                step_seconds=args.step_seconds,
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
            )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
