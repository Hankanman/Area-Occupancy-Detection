"""Shadow-mode accuracy metrics for the trust-score epic (#499).

Pure math over two inputs the coordinator already produces:

* **Tick samples** — the per-refresh ``(timestamp, probability, occupied)``
  triples the coordinator records for each area, at roughly the 10s
  decay-timer cadence plus extra ticks on evidence-triggered refreshes.
  Every metric below is time-weighted (see ``_tick_weights``) so that
  irregular cadence doesn't bias the result.
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

Each window is first reduced to an ``AccuracyTally``: the seconds behind
every rate. Tallies add, so an hour's ticks fold into the day's tally and
thirty days pool into one honest monthly figure, which daily rates can't.

Shadow-mode contract: nothing here feeds back into probability,
thresholds, or decay. Metrics are computed, cached, logged, and exported
in diagnostics only. Promotion to a user-facing sensor or auto-threshold
routes through the change-control gates (see issue #499's phases).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

DEFAULT_CALIBRATION_BINS = 10


@dataclass(frozen=True)
class TickSample:
    """One coordinator refresh observation for an area."""

    timestamp: datetime
    probability: float
    occupied: bool
    # The live ground-truth label at this tick (data.ground_truth). When
    # set it is the truth this sample is scored against; ``None`` falls
    # back to the occupied intervals passed in.
    truth: bool | None = None
    # A sensor is decaying and none is active: the area is fading out after
    # its motion timeout. With the label off, such a tick is the configured
    # hold, neither right nor wrong, and is left unscored.
    fading: bool = False


@dataclass
class CalibrationBin:
    """Reliability row for one probability band."""

    lower: float
    upper: float
    count: int = 0
    mean_probability: float = 0.0
    observed_rate: float = 0.0
    # Time-weighted seconds this band represents (see _tick_weights).
    # ``count`` stays a raw sample count for diagnostics readability;
    # weight is what the reliability stats and suggest_threshold use.
    weight: float = 0.0


@dataclass
class AccuracyMetrics:
    """Per-area shadow accuracy snapshot over an evaluation window."""

    sample_count: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    # Calibration
    expected_calibration_error: float | None = None
    bins: list[CalibrationBin] = field(default_factory=list)
    # Decision stability (time-weighted — see _tick_weights)
    decision_transitions: int = 0
    truth_transitions: int = 0
    false_on_rate: float | None = None  # P(decision on | truth off)
    false_off_rate: float | None = None  # P(decision off | truth on)
    agreement: float | None = None  # fraction of samples where decision == truth
    # Seconds of each truth class needed before a rate (or a threshold
    # trading the two off) is reported; set by compute_accuracy_metrics.
    min_class_seconds: float = 0.0
    # Time-weighted share of labelled ticks where the live ground truth
    # matched the database's occupied history (None without live labels).
    truth_agreement_with_db: float | None = None
    # Seconds left unscored because the area was fading with the label off.
    fade_seconds: float = 0.0


def _is_occupied_at(ts: datetime, intervals: list[tuple[datetime, datetime]]) -> bool:
    """Return whether ``ts`` falls inside any occupied interval.

    Intervals are half-open ``[start, end)`` so a tick landing exactly on
    an interval boundary is attributed to only one side.
    """
    return any(start <= ts < end for start, end in intervals)


def _tick_weights(samples: list[TickSample]) -> list[float]:
    """Return each sample's duration weight (seconds it represents).

    Ticks do not land at a fixed cadence — the coordinator also ticks on
    every evidence-triggered refresh, on top of the decay timer — so a
    busy period (frequent evidence changes) contributes disproportionately
    many samples relative to the wall-clock time it spans. Weighting each
    sample by the gap to the next tick (piecewise-constant, matching
    ``OnlinePriorEstimator.observe``'s assumption) makes every metric
    below a time-weighted average instead of a sample-weighted one.

    The last sample has no "next tick" to measure a gap to; it reuses the
    preceding gap as an estimate. Out-of-order or duplicate timestamps
    clamp to a zero gap. If every gap is zero (e.g. a single sample, or
    all samples share one timestamp), falls back to uniform weight-1 per
    sample so callers never divide by zero.
    """
    n = len(samples)
    if n == 1:
        return [1.0]
    weights = [0.0] * n
    for i in range(n - 1):
        gap = (samples[i + 1].timestamp - samples[i].timestamp).total_seconds()
        weights[i] = max(gap, 0.0)
    weights[-1] = weights[-2]
    if not any(weights):
        return [1.0] * n
    return weights


@dataclass
class AccuracyTally:
    """The seconds behind an area's accuracy rates over some stretch of time.

    Rates can't be combined (a day's 90% and another's 60% say nothing
    about the pair without the time behind each); these seconds can, so a
    tally is what the daily history keeps and what a 30-day figure pools.
    """

    samples: int = 0
    true_on: float = 0.0
    true_off: float = 0.0
    false_on: float = 0.0
    false_off: float = 0.0
    # Label off while the area faded out: the configured hold, unscored.
    fade: float = 0.0
    decision_transitions: int = 0
    truth_transitions: int = 0
    # Seconds with a live label, and those where it matched the database.
    labelled: float = 0.0
    labelled_agree: float = 0.0
    # Per probability band: ticks, seconds, Σ probability·seconds, and the
    # seconds the label was occupied.
    bin_count: list[int] = field(default_factory=lambda: [0] * DEFAULT_CALIBRATION_BINS)
    bin_weight: list[float] = field(
        default_factory=lambda: [0.0] * DEFAULT_CALIBRATION_BINS
    )
    bin_prob: list[float] = field(
        default_factory=lambda: [0.0] * DEFAULT_CALIBRATION_BINS
    )
    bin_hit: list[float] = field(
        default_factory=lambda: [0.0] * DEFAULT_CALIBRATION_BINS
    )

    _SCALARS = (
        "samples",
        "true_on",
        "true_off",
        "false_on",
        "false_off",
        "fade",
        "decision_transitions",
        "truth_transitions",
        "labelled",
        "labelled_agree",
    )
    _BINS = ("bin_count", "bin_weight", "bin_prob", "bin_hit")

    def add(self, other: AccuracyTally) -> None:
        """Fold another tally into this one."""
        for name in self._SCALARS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in self._BINS:
            mine, theirs = getattr(self, name), getattr(other, name)
            for i, value in enumerate(theirs):
                mine[i] += value

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form for the HA store."""
        out: dict[str, Any] = {name: getattr(self, name) for name in self._SCALARS}
        for name in self._BINS:
            out[name] = list(getattr(self, name))
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccuracyTally:
        """Rebuild from ``to_dict`` output; unknown or bad fields are dropped."""
        tally = cls()
        for name in cls._SCALARS:
            value = data.get(name)
            if isinstance(value, int | float):
                setattr(tally, name, type(getattr(tally, name))(value))
        for name in cls._BINS:
            values = data.get(name)
            if isinstance(values, list) and len(values) == DEFAULT_CALIBRATION_BINS:
                kind = type(getattr(tally, name)[0])
                setattr(tally, name, [kind(v) for v in values])
        return tally


def tally_by_day(
    samples: list[TickSample],
    occupied_intervals: list[tuple[datetime, datetime]],
    day_of: Callable[[datetime], str],
    *,
    score_last: bool = True,
) -> dict[str, AccuracyTally]:
    """Score ticks against ground truth, one tally per day.

    Each tick is weighted by the seconds to the next one (see
    ``_tick_weights``). With ``score_last`` off, the last tick only closes
    the one before it and is not scored itself, so consecutive batches
    split at a shared tick neither overlap nor leave a gap.

    Args:
        samples: Tick observations ordered oldest-first.
        occupied_intervals: The database's occupied ``(start, end)`` spans:
            the truth for ticks without a live label, and the parity check
            for ticks with one.
        day_of: The day a tick belongs to (the caller's local date).
        score_last: Score the last tick too, with the preceding gap as its
            weight, for a one-off window.

    Returns:
        Tallies keyed by day.
    """
    tallies: dict[str, AccuracyTally] = {}
    scored = samples if score_last else samples[:-1]
    if not scored:
        return tallies
    weights = _tick_weights(samples)[: len(scored)]
    prev: dict[str, tuple[bool, bool]] = {}
    for sample, weight in zip(scored, weights, strict=True):
        day = day_of(sample.timestamp)
        tally = tallies.setdefault(day, AccuracyTally())
        tally.samples += 1
        db_truth = _is_occupied_at(sample.timestamp, occupied_intervals)
        if sample.truth is None:
            truth = db_truth
        else:
            truth = sample.truth
            # Parity of the live label with the database's history, the
            # evidence that retiring the database keeps the same truth.
            tally.labelled += weight
            if truth == db_truth:
                tally.labelled_agree += weight

        # Flip counting: events, not time.
        if day in prev:
            last_decision, last_truth = prev[day]
            tally.decision_transitions += sample.occupied != last_decision
            tally.truth_transitions += truth != last_truth
        prev[day] = (sample.occupied, truth)

        if sample.fading and not truth:
            tally.fade += weight
            continue

        p = min(max(sample.probability, 0.0), 1.0)
        idx = min(int(p * DEFAULT_CALIBRATION_BINS), DEFAULT_CALIBRATION_BINS - 1)
        tally.bin_count[idx] += 1
        tally.bin_weight[idx] += weight
        tally.bin_prob[idx] += p * weight
        if truth:
            tally.bin_hit[idx] += weight

        if sample.occupied and truth:
            tally.true_on += weight
        elif sample.occupied:
            tally.false_on += weight
        elif truth:
            tally.false_off += weight
        else:
            tally.true_off += weight
    return tallies


def metrics_from_tally(
    tally: AccuracyTally,
    *,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    min_class_seconds: float = 0.0,
) -> AccuracyMetrics:
    """The calibration and decision rates a tally adds up to.

    A false-on (false-off) rate is only reported when the tally holds at
    least ``min_class_seconds`` of empty (occupied) time; the analysis
    passes ``ACCURACY_MIN_CLASS_SECONDS``.
    """
    out = AccuracyMetrics(
        sample_count=tally.samples,
        window_start=window_start,
        window_end=window_end,
        decision_transitions=tally.decision_transitions,
        truth_transitions=tally.truth_transitions,
        min_class_seconds=min_class_seconds,
        fade_seconds=tally.fade,
    )
    total_weight = sum(tally.bin_weight)
    if not tally.samples:
        return out
    if tally.labelled:
        out.truth_agreement_with_db = tally.labelled_agree / tally.labelled
    if not total_weight:
        return out

    # Reliability rows + ECE. ``count`` stays a raw tick count for
    # readability; the stats use seconds, so a burst of evidence ticks
    # doesn't outweigh a quiet decay tick spanning the same time.
    ece = 0.0
    bins = len(tally.bin_weight)
    for i in range(bins):
        row = CalibrationBin(lower=i / bins, upper=(i + 1) / bins)
        row.count = tally.bin_count[i]
        row.weight = tally.bin_weight[i]
        if row.count and row.weight:
            row.mean_probability = tally.bin_prob[i] / row.weight
            row.observed_rate = tally.bin_hit[i] / row.weight
            ece += (row.weight / total_weight) * abs(
                row.mean_probability - row.observed_rate
            )
        out.bins.append(row)
    out.expected_calibration_error = ece

    truth_off_total = tally.true_off + tally.false_on
    truth_on_total = tally.true_on + tally.false_off
    # A rate over a few minutes of one truth class is noise: a bedroom
    # occupied all night has seconds of "empty", and two mis-scored ticks
    # there read as a 100% false-on rate.
    if truth_off_total and truth_off_total >= min_class_seconds:
        out.false_on_rate = tally.false_on / truth_off_total
    if truth_on_total and truth_on_total >= min_class_seconds:
        out.false_off_rate = tally.false_off / truth_on_total
    out.agreement = (tally.true_on + tally.true_off) / total_weight
    return out


def compute_accuracy_metrics(
    samples: list[TickSample],
    occupied_intervals: list[tuple[datetime, datetime]],
    *,
    min_class_seconds: float = 0.0,
) -> AccuracyMetrics:
    """Score an area's probability stream against ground truth.

    Args:
        samples: Tick observations ordered oldest-first. Samples outside
            the intervals' coverage are still scored (truth = not
            occupied), matching how the prior denominator treats quiet
            time (#483 lesson: unobserved-quiet counts).
        occupied_intervals: Motion-confirmed ``(start, end)`` ground
            truth, same source the prior learning uses.
        min_class_seconds: Seconds of empty (occupied) time below which
            the false-on (false-off) rate is left unreported.

    Returns:
        ``AccuracyMetrics``; with no samples, a zeroed snapshot whose rate
        fields stay ``None`` (unknown, not zero).
    """
    tally = AccuracyTally()
    for day_tally in tally_by_day(samples, occupied_intervals, lambda _ts: "").values():
        tally.add(day_tally)
    return metrics_from_tally(
        tally,
        window_start=samples[0].timestamp if samples else None,
        window_end=samples[-1].timestamp if samples else None,
        min_class_seconds=min_class_seconds,
    )


# Below this many tick samples in the window, suggest_threshold returns
# None — a handful of ticks says nothing about the false-on/false-off
# tradeoff. At the ~10s decay-timer cadence this is under 20 minutes of
# observation, a deliberately low floor: the gate protects against
# near-empty windows (fresh restart, brand-new area), not against noisy
# ones — noise shows up as the suggestion moving between analysis runs,
# which is exactly what the #499 "stable for a release cycle" promotion
# gate watches before auto-threshold may ever consume this value.
SUGGEST_THRESHOLD_MIN_SAMPLES = 100

# Seconds of each truth class (occupied, empty) a window needs before its
# false-on / false-off rate, or a threshold trading them off, is reported.
ACCURACY_MIN_CLASS_SECONDS = 3600.0


def suggest_threshold(metrics: AccuracyMetrics) -> float | None:
    """Return the decision threshold the calibration data argues for.

    Scans every bin edge as a candidate threshold and picks the one
    minimizing ``false_on_rate + false_off_rate`` (equal weighting —
    treating a phantom "occupied" exactly as bad as a missed person),
    computed from the time-weighted calibration bins: a bin at or above
    the candidate decides "on" for ``weight`` seconds of which
    ``weight × observed_rate`` were truly occupied.

    Shadow-mode contract: nothing in the decision path reads this value.
    It is exposed read-only (diagnostics + the Accuracy sensor attribute)
    so the data for the #499 auto-threshold decision accumulates in the
    open; promotion to actually driving the threshold routes through the
    change-control gates.

    Returns:
        The candidate threshold (0-1 domain, a bin edge) with the lowest
        combined error, or ``None`` when the window is too small
        (< ``SUGGEST_THRESHOLD_MIN_SAMPLES``), the bins are empty, or the
        window never observed one of the two truth classes (a threshold
        tradeoff needs both). Ties resolve to the lowest such threshold.
    """
    if metrics.sample_count < SUGGEST_THRESHOLD_MIN_SAMPLES:
        return None
    populated = [b for b in metrics.bins if b.weight > 0]
    if not populated:
        return None

    truth_on_total = sum(b.weight * b.observed_rate for b in populated)
    truth_off_total = sum(b.weight * (1.0 - b.observed_rate) for b in populated)
    # A class too small to rate (see compute_accuracy_metrics'
    # min_class_seconds) can't anchor a tradeoff either.
    floor = max(metrics.min_class_seconds, 0.0)
    if (
        truth_on_total <= 0
        or truth_off_total <= 0
        or truth_on_total < floor
        or truth_off_total < floor
    ):
        return None

    best_threshold: float | None = None
    best_cost = float("inf")
    # Candidate thresholds are the interior bin edges (for 10 bins:
    # 0.1 .. 0.9). The exterior edges (0.0 / 1.0) would mean "always on" /
    # "always off", which is not a threshold suggestion.
    for edge_bin in metrics.bins[1:]:
        threshold = edge_bin.lower
        false_on = sum(
            b.weight * (1.0 - b.observed_rate)
            for b in populated
            if b.lower >= threshold
        )
        false_off = sum(
            b.weight * b.observed_rate for b in populated if b.lower < threshold
        )
        cost = false_on / truth_off_total + false_off / truth_on_total
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
    return best_threshold


def accuracy_summary(metrics: AccuracyMetrics) -> dict:
    """The headline numbers of a snapshot, for the persisted daily history."""
    return {
        "samples": metrics.sample_count,
        "agreement": metrics.agreement,
        "expected_calibration_error": metrics.expected_calibration_error,
        "false_on_rate": metrics.false_on_rate,
        "false_off_rate": metrics.false_off_rate,
        "suggested_threshold": suggest_threshold(metrics),
        "truth_agreement_with_db": metrics.truth_agreement_with_db,
        "fade_seconds": round(metrics.fade_seconds),
    }


def metrics_to_diagnostics(metrics: AccuracyMetrics) -> dict:
    """Flatten an ``AccuracyMetrics`` into a JSON-safe diagnostics block."""
    return {
        # Still shadow: the Accuracy sensor (2026.9.1) exposes these values
        # read-only, but nothing in the decision path (probability,
        # threshold, decay) consumes them. The flag flips only when
        # auto-threshold actually engages, per #499's phase gates.
        "shadow_mode": True,
        "sample_count": metrics.sample_count,
        "window_start": metrics.window_start.isoformat()
        if metrics.window_start
        else None,
        "window_end": metrics.window_end.isoformat() if metrics.window_end else None,
        "expected_calibration_error": metrics.expected_calibration_error,
        "suggested_threshold": suggest_threshold(metrics),
        "truth_agreement_with_db": metrics.truth_agreement_with_db,
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
