"""Shadow-mode accuracy metrics for the trust-score epic (#499).

Pure math over two inputs the coordinator already produces:

* **Tick samples** — the per-refresh ``(timestamp, probability, occupied)``
  triples the coordinator records for each area (10s decay-timer cadence).
* **Occupied intervals** — the motion-confirmed ground truth the prior
  learning already uses (``db.get_occupied_intervals``).

Two questions get answered per area:

1. **Calibration** — when the engine says 70%, is the area actually
   occupied about 70% of those times? Reported as per-bin reliability
   rows plus the expected calibration error (ECE), the count-weighted
   mean of |mean predicted − observed rate| across bins.
2. **Decision stability** — how often does the occupancy binary flip
   compared to the ground truth, and how much of the time does the
   decision disagree with the truth (false-on / false-off rates)?

This module deliberately has no coordinator or DB dependencies —
callers gather inputs and pass them in, mirroring ``data/adjacency.py``,
so the math stays testable without SQLite or Home Assistant.

Shadow-mode contract: nothing here feeds back into probability,
thresholds, or decay. Metrics are computed, cached, logged, and exported
in diagnostics only. Promotion to a user-facing sensor or auto-threshold
routes through the change-control gates (see issue #499's phases).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

DEFAULT_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class TickSample:
    """One coordinator refresh observation for an area."""

    timestamp: datetime
    probability: float
    occupied: bool


@dataclass
class CalibrationBin:
    """Reliability row for one probability band."""

    lower: float
    upper: float
    count: int = 0
    mean_probability: float = 0.0
    observed_rate: float = 0.0


@dataclass
class AccuracyMetrics:
    """Per-area shadow accuracy snapshot over an evaluation window."""

    sample_count: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    # Calibration
    expected_calibration_error: float | None = None
    bins: list[CalibrationBin] = field(default_factory=list)
    # Decision stability (sample-weighted at the fixed tick cadence)
    decision_transitions: int = 0
    truth_transitions: int = 0
    false_on_rate: float | None = None  # P(decision on | truth off)
    false_off_rate: float | None = None  # P(decision off | truth on)
    agreement: float | None = None  # fraction of samples where decision == truth


def _is_occupied_at(ts: datetime, intervals: list[tuple[datetime, datetime]]) -> bool:
    """Return whether ``ts`` falls inside any occupied interval.

    Intervals are half-open ``[start, end)`` so a tick landing exactly on
    an interval boundary is attributed to only one side.
    """
    return any(start <= ts < end for start, end in intervals)


def compute_accuracy_metrics(
    samples: list[TickSample],
    occupied_intervals: list[tuple[datetime, datetime]],
    *,
    bins: int = DEFAULT_CALIBRATION_BINS,
) -> AccuracyMetrics:
    """Score an area's probability stream against motion-confirmed truth.

    Args:
        samples: Tick observations ordered oldest-first. Samples outside
            the intervals' coverage are still scored (truth = not
            occupied), matching how the prior denominator treats
            quiet time (#483 lesson: unobserved-quiet counts).
        occupied_intervals: Motion-confirmed ``(start, end)`` ground
            truth, same source the prior learning uses.
        bins: Number of equal-width probability bands for calibration.

    Returns:
        ``AccuracyMetrics``; with no samples, a zeroed snapshot whose
        rate fields stay ``None`` (unknown, not zero).
    """
    out = AccuracyMetrics()
    if not samples:
        return out

    out.sample_count = len(samples)
    out.window_start = samples[0].timestamp
    out.window_end = samples[-1].timestamp

    bin_count = [0] * bins
    bin_prob_sum = [0.0] * bins
    bin_hits = [0] * bins

    true_on = true_off = false_on = false_off = 0
    prev_decision: bool | None = None
    prev_truth: bool | None = None

    for sample in samples:
        truth = _is_occupied_at(sample.timestamp, occupied_intervals)

        # Calibration accumulation
        p = min(max(sample.probability, 0.0), 1.0)
        idx = min(int(p * bins), bins - 1)
        bin_count[idx] += 1
        bin_prob_sum[idx] += p
        if truth:
            bin_hits[idx] += 1

        # Confusion accumulation
        if sample.occupied and truth:
            true_on += 1
        elif sample.occupied and not truth:
            false_on += 1
        elif not sample.occupied and truth:
            false_off += 1
        else:
            true_off += 1

        # Flip counting
        if prev_decision is not None and sample.occupied != prev_decision:
            out.decision_transitions += 1
        if prev_truth is not None and truth != prev_truth:
            out.truth_transitions += 1
        prev_decision = sample.occupied
        prev_truth = truth

    # Reliability rows + ECE
    ece = 0.0
    for i in range(bins):
        row = CalibrationBin(lower=i / bins, upper=(i + 1) / bins)
        row.count = bin_count[i]
        if row.count:
            row.mean_probability = bin_prob_sum[i] / row.count
            row.observed_rate = bin_hits[i] / row.count
            ece += (row.count / out.sample_count) * abs(
                row.mean_probability - row.observed_rate
            )
        out.bins.append(row)
    out.expected_calibration_error = ece

    truth_off_total = true_off + false_on
    truth_on_total = true_on + false_off
    if truth_off_total:
        out.false_on_rate = false_on / truth_off_total
    if truth_on_total:
        out.false_off_rate = false_off / truth_on_total
    out.agreement = (true_on + true_off) / out.sample_count

    return out


def metrics_to_diagnostics(metrics: AccuracyMetrics) -> dict:
    """Flatten an ``AccuracyMetrics`` into a JSON-safe diagnostics block."""
    return {
        "shadow_mode": True,
        "sample_count": metrics.sample_count,
        "window_start": metrics.window_start.isoformat()
        if metrics.window_start
        else None,
        "window_end": metrics.window_end.isoformat() if metrics.window_end else None,
        "expected_calibration_error": metrics.expected_calibration_error,
        "agreement": metrics.agreement,
        "false_on_rate": metrics.false_on_rate,
        "false_off_rate": metrics.false_off_rate,
        "decision_transitions": metrics.decision_transitions,
        "truth_transitions": metrics.truth_transitions,
        "calibration_bins": [
            {
                "band": f"{b.lower:.1f}-{b.upper:.1f}",
                "count": b.count,
                "mean_probability": round(b.mean_probability, 4),
                "observed_rate": round(b.observed_rate, 4),
            }
            for b in metrics.bins
            if b.count
        ],
    }
