"""Shadow-mode learned sensor fusion (#501, phase 1).

The live pipeline's evidence combination in ``utils.sigmoid_probability``
is, structurally, a logistic model with hand-tuned coefficients:

    p = sigmoid(logit(prior) + Σ_i effective_weight_i · x_i)

where ``x_i = evidence_i · correlation_i · prob_given_true_i ·
strength_multiplier_i`` — every factor except ``effective_weight`` is
already learned or derived per home. This module learns the remaining
coefficients per entity via plain online logistic-regression gradient
steps against the same motion-confirmed ground truth the accuracy
metrics (#499) score against, batched once per analysis cycle over the
coordinator's tick window.

Shadow-mode contract: learned weights are computed, persisted (HA
storage helper, lifecycle mirroring ``online_prior.py``), and exported
in diagnostics with ``engaged: false``. They are **never read by the
probability path** — promotion routes through #499's calibration
comparison, per #501's phases. That comparison is :meth:`FusionLearner.score`:
each hour, before training, the new ticks are scored twice against the
ground truth, once with the live probability and once with the learned
weights swapped into it, and the two are kept as daily tallies of the
#499 metrics (calibration error, agreement, false-on/false-off rates).

Safety rails, per the issue:

* **No negative weights this phase.** The live pipeline's evidence is
  structurally one-sided (``combined_probability`` documents its
  reliance on non-negative contributions), so learned weights clamp to
  ``[0, MAX_WEIGHT]``. A weight learning toward 0 is this phase's
  answer to an uninformative or confounded sensor — the negative-
  evidence generalization is a deliberate later decision, not a tuning
  detail.
* **MOTION and SLEEP are excluded** from learning: the ground-truth
  labels are derived from motion ∪ media ∪ sleep evidence, so those
  types are partially self-labelling (correlation analysis already
  excludes them for the same reason). The caller enforces this by
  leaving those entities out of ``FusionTick.features``. MEDIA stays
  learnable but shares the caveat — its learned weight will read
  optimistically high and must be judged accordingly at promotion time.
* **L2 anchor toward the live defaults**: with little data the learned
  weight stays near ``effective_weight``; a cold start is exactly
  today's behavior.
* **Minimum-sample gate** before a learned weight is even *reported*
  (``FUSION_MIN_SAMPLES`` ticks observed for the area); each weight
  additionally carries ``reliable`` once its own entity has that many.

Like ``metrics.py`` and ``adjacency.py``, this module has no
coordinator, HA, or DB dependencies — callers gather inputs and pass
them in, so the math stays testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import pairwise
import math
from typing import Any, cast

from ..const import FUSION_L2, FUSION_LEARNING_RATE, FUSION_MIN_SAMPLES, MAX_WEIGHT
from ..time_utils import ensure_utc_datetime, to_local
from ..utils import clamp_probability, sigmoid
from .metrics import DEFAULT_CALIBRATION_BINS, _is_occupied_at
from .online_prior import MAX_TICK_GAP_SECONDS

# Days of calibration tallies kept, and how many of the latest are pooled
# into the diagnostics comparison.
SCORE_HISTORY_DAYS = 45
SCORE_WINDOW_DAYS = 30
# Seconds of each truth class before a false-on/false-off rate is reported
# (as metrics.ACCURACY_MIN_CLASS_SECONDS).
SCORE_MIN_CLASS_SECONDS = 3600.0

_BIN_FIELDS = ("bin_weight", "bin_probability", "bin_hits")
_SUM_FIELDS = ("true_on", "false_on", "false_off", "true_off")


# --- storage parsing helpers -------------------------------------------------


def _as_float(value: Any, *, nonnegative: bool = False) -> float | None:
    """Return ``value`` as a finite float (optionally >= 0), else ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (nonnegative and number < 0):
        return None
    return number


def _as_count(value: Any) -> int | None:
    """Return ``value`` as a non-negative int, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _as_dict(value: Any) -> dict[Any, Any]:
    return cast(dict[Any, Any], value) if isinstance(value, dict) else {}


def _parse_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return ensure_utc_datetime(datetime.fromisoformat(raw))
    except ValueError:
        return None


def _is_iso_day(raw: Any) -> bool:
    if not isinstance(raw, str):
        return False
    try:
        date.fromisoformat(raw)
    except ValueError:
        return False
    return True


# --- calibration tallies -----------------------------------------------------


@dataclass
class CalibrationTally:
    """Time-weighted calibration and decision sums, mergeable across days.

    The same quantities ``metrics.compute_accuracy_metrics`` reports, kept
    as sums so hourly batches add up into exact daily and monthly figures.
    """

    bin_weight: list[float] = field(
        default_factory=lambda: [0.0] * DEFAULT_CALIBRATION_BINS
    )
    bin_probability: list[float] = field(
        default_factory=lambda: [0.0] * DEFAULT_CALIBRATION_BINS
    )
    bin_hits: list[float] = field(
        default_factory=lambda: [0.0] * DEFAULT_CALIBRATION_BINS
    )
    true_on: float = 0.0
    false_on: float = 0.0
    false_off: float = 0.0
    true_off: float = 0.0

    def add(self, probability: float, *, on: bool, truth: bool, seconds: float) -> None:
        """Credit ``seconds`` of one prediction against the truth."""
        bins = len(self.bin_weight)
        p = min(max(probability, 0.0), 1.0)
        idx = min(int(p * bins), bins - 1)
        self.bin_weight[idx] += seconds
        self.bin_probability[idx] += p * seconds
        if truth:
            self.bin_hits[idx] += seconds
        if on and truth:
            self.true_on += seconds
        elif on:
            self.false_on += seconds
        elif truth:
            self.false_off += seconds
        else:
            self.true_off += seconds

    def merge(self, other: CalibrationTally) -> None:
        """Add another tally's sums into this one."""
        for mine, theirs in (
            (self.bin_weight, other.bin_weight),
            (self.bin_probability, other.bin_probability),
            (self.bin_hits, other.bin_hits),
        ):
            for i, value in enumerate(theirs):
                mine[i] += value
        self.true_on += other.true_on
        self.false_on += other.false_on
        self.false_off += other.false_off
        self.true_off += other.true_off

    def summary(self) -> dict[str, Any]:
        """ECE, agreement and the two error rates (None where unknown)."""
        total = sum(self.bin_weight)
        if total <= 0:
            return {"seconds": 0.0}
        ece = sum(
            abs(prob - hits)
            for prob, hits in zip(self.bin_probability, self.bin_hits, strict=True)
        )
        truth_off = self.true_off + self.false_on
        truth_on = self.true_on + self.false_off
        return {
            "seconds": round(total, 1),
            # Σ_bins (w/W)·|Σpw/w − Σhw/w| = Σ_bins |Σpw − Σhw| / W
            "expected_calibration_error": round(ece / total, 4),
            "agreement": round((self.true_on + self.true_off) / total, 4),
            "false_on_rate": round(self.false_on / truth_off, 4)
            if truth_off >= SCORE_MIN_CLASS_SECONDS
            else None,
            "false_off_rate": round(self.false_off / truth_on, 4)
            if truth_on >= SCORE_MIN_CLASS_SECONDS
            else None,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize (JSON-safe; lists are copied, not shared)."""
        return {
            "bin_weight": list(self.bin_weight),
            "bin_probability": list(self.bin_probability),
            "bin_hits": list(self.bin_hits),
            "true_on": self.true_on,
            "false_on": self.false_on,
            "false_off": self.false_off,
            "true_off": self.true_off,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CalibrationTally:
        """Restore; raises ``ValueError`` on malformed data."""
        if not isinstance(data, dict):
            raise ValueError("tally")
        payload = cast(dict[str, Any], data)
        bins: dict[str, list[float]] = {}
        for name in _BIN_FIELDS:
            raw = payload.get(name)
            if not isinstance(raw, list):
                raise ValueError(name)
            values = [_as_float(v, nonnegative=True) for v in cast(list[Any], raw)]
            if len(values) != DEFAULT_CALIBRATION_BINS or None in values:
                raise ValueError(name)
            bins[name] = [v for v in values if v is not None]
        sums: dict[str, float] = {}
        for name in _SUM_FIELDS:
            value = _as_float(payload.get(name, 0.0), nonnegative=True)
            if value is None:
                raise ValueError(name)
            sums[name] = value
        return cls(
            bin_weight=bins["bin_weight"],
            bin_probability=bins["bin_probability"],
            bin_hits=bins["bin_hits"],
            true_on=sums["true_on"],
            false_on=sums["false_on"],
            false_off=sums["false_off"],
            true_off=sums["true_off"],
        )


# --- training rows and state -------------------------------------------------


@dataclass(frozen=True)
class FusionTick:
    """One coordinator refresh's training row for an area.

    ``bias`` is the logit of the prior the live pipeline used at that
    tick; ``features`` maps entity_id to its ``x_i`` product (evidence ·
    correlation · prob_given_true · strength_multiplier), sparse —
    entities with zero evidence are simply absent.
    """

    timestamp: datetime
    bias: float
    features: dict[str, float]
    # Live ground-truth label (data.ground_truth); None falls back to the
    # occupied intervals passed to ``update``.
    truth: bool | None = None
    # For scoring (see ``FusionLearner.score``): the area's live probability
    # at this tick, and its logit minus the learnable terms
    # ``Σ effective_weight_i · x_i``, i.e. everything the learned weights
    # don't touch (prior, motion and sleep, activity and adjacency boosts).
    probability: float | None = None
    fixed_logit: float | None = None


def _tick_truth(
    tick: FusionTick, occupied_intervals: list[tuple[datetime, datetime]]
) -> bool:
    if tick.truth is not None:
        return tick.truth
    return _is_occupied_at(tick.timestamp, occupied_intervals)


@dataclass
class FusionState:
    """Serializable learned-weight state for one area."""

    weights: dict[str, float] = field(default_factory=dict)
    samples: int = 0
    # Ticks each entity contributed a nonzero feature to (its own sample
    # count; ``samples`` counts the area's ticks).
    entity_samples: dict[str, int] = field(default_factory=dict)
    # Fingerprint (type, active states) each weight was learned under.
    signatures: dict[str, str] = field(default_factory=dict)
    # Timestamp of the newest tick already trained on. The analysis passes
    # its whole 24h window every hour, so without this each tick would be
    # stepped ~24 times and ``samples`` would pass the reporting gate early.
    trained_through: datetime | None = None
    # Local date -> {"live": tally, "learned": tally}: how the live
    # probability and the one with learned weights scored that day.
    score_days: dict[str, dict[str, CalibrationTally]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "weights": dict(self.weights),
            "samples": self.samples,
            "entity_samples": dict(self.entity_samples),
            "signatures": dict(self.signatures),
            "trained_through": (
                self.trained_through.isoformat() if self.trained_through else None
            ),
            "score_days": {
                day: {model: t.to_dict() for model, t in tallies.items()}
                for day, tallies in self.score_days.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Any) -> FusionState:
        """Restore from storage; malformed parts are dropped individually.

        A single bad value must not discard every learned weight (and
        restart the sample gate), so each field and entry is validated on
        its own.
        """
        payload = _as_dict(data)

        weights: dict[str, float] = {}
        for k, v in _as_dict(payload.get("weights")).items():
            w = _as_float(v)
            if isinstance(k, str) and w is not None:
                weights[k] = min(max(w, 0.0), MAX_WEIGHT)

        entity_samples: dict[str, int] = {}
        for k, v in _as_dict(payload.get("entity_samples")).items():
            n = _as_count(v)
            if isinstance(k, str) and n is not None:
                entity_samples[k] = n

        signatures = {
            k: v
            for k, v in _as_dict(payload.get("signatures")).items()
            if isinstance(k, str) and isinstance(v, str)
        }

        score_days: dict[str, dict[str, CalibrationTally]] = {}
        for day, tallies in _as_dict(payload.get("score_days")).items():
            if not _is_iso_day(day):
                continue
            restored: dict[str, CalibrationTally] = {}
            for model, raw_tally in _as_dict(tallies).items():
                if not isinstance(model, str):
                    continue
                try:
                    restored[model] = CalibrationTally.from_dict(raw_tally)
                except ValueError:
                    continue
            if restored:
                score_days[cast(str, day)] = restored

        return cls(
            weights=weights,
            samples=_as_count(payload.get("samples")) or 0,
            entity_samples=entity_samples,
            signatures=signatures,
            trained_through=_parse_datetime(payload.get("trained_through")),
            score_days=score_days,
        )


# --- learner -----------------------------------------------------------------


class FusionLearner:
    """Online logistic-weight learner for one area."""

    def __init__(self, state: FusionState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state: FusionState = state if state is not None else FusionState()

    def _new_ticks(self, ticks: list[FusionTick]) -> list[FusionTick]:
        """Ticks after ``trained_through``, oldest first, timestamps in UTC."""
        normalized = [
            tick
            if tick.timestamp.tzinfo is not None
            else _with_timestamp(tick, ensure_utc_datetime(tick.timestamp))
            for tick in ticks
        ]
        cutoff = self.state.trained_through
        if cutoff is not None:
            normalized = [t for t in normalized if t.timestamp > cutoff]
        return sorted(normalized, key=lambda t: t.timestamp)

    def update(
        self,
        ticks: list[FusionTick],
        occupied_intervals: list[tuple[datetime, datetime]],
        defaults: dict[str, float],
        *,
        learning_rate: float = FUSION_LEARNING_RATE,
        l2: float = FUSION_L2,
    ) -> int:
        """Run one gradient pass over a batch of ticks.

        For each tick, the label is the motion-confirmed ground truth at
        its timestamp and the prediction is
        ``sigmoid(bias + Σ w_i · x_i)`` with each unseen entity's weight
        initialized at its live default (cold start ≡ today's model).
        The logistic-loss gradient for each present feature is
        ``(ŷ − y) · x_i`` plus the L2 pull ``l2 · (w_i − default_i)``,
        and the stepped weight clamps to ``[0, MAX_WEIGHT]``.

        Entities absent from a tick's features (zero evidence) receive
        no update from it: their loss gradient is zero there, and
        applying the L2 pull only alongside real gradient signal keeps
        an entity that stops appearing anchored where its evidence left
        it rather than silently relaxing back to the default.

        Args:
            ticks: Training rows, any order; already-trained ones are skipped
                and the rest are stepped oldest first.
            occupied_intervals: Motion-confirmed ``(start, end)`` ground
                truth — the same source #499's metrics score against.
            defaults: entity_id -> the live pipeline's current
                ``effective_weight``, used for cold-start initialization
                and as the L2 anchor.
            learning_rate: SGD step size (``FUSION_LEARNING_RATE``).
            l2: Anchor strength toward ``defaults`` (``FUSION_L2``).

        Ticks at or before ``state.trained_through`` are skipped, so a tick
        contributes one gradient step however many overlapping windows it
        is offered in.

        Returns:
            The number of new ticks consumed.
        """
        ticks = self._new_ticks(ticks)
        if not ticks:
            return 0
        state = self.state
        weights = state.weights
        for tick in ticks:
            features = {
                entity_id: x
                for entity_id, x in tick.features.items()
                if math.isfinite(x)
            }
            y = 1.0 if _tick_truth(tick, occupied_intervals) else 0.0
            z = tick.bias
            for entity_id, x in features.items():
                z += weights.get(entity_id, defaults.get(entity_id, 0.0)) * x
            error = sigmoid(z) - y
            for entity_id, x in features.items():
                default = defaults.get(entity_id, 0.0)
                w = weights.get(entity_id, default)
                w -= learning_rate * (error * x + l2 * (w - default))
                weights[entity_id] = min(max(w, 0.0), MAX_WEIGHT)
                state.entity_samples[entity_id] = (
                    state.entity_samples.get(entity_id, 0) + 1
                )
        state.samples += len(ticks)
        state.trained_through = ticks[-1].timestamp
        return len(ticks)

    def score(
        self,
        ticks: list[FusionTick],
        occupied_intervals: list[tuple[datetime, datetime]],
        defaults: dict[str, float],
        threshold: float,
    ) -> int:
        """Score the learned weights against the live probability.

        Call before :meth:`update` with the same batch: each new tick is
        then predicted by weights that have not yet trained on it, so the
        score is out of sample. The learned probability swaps the learned
        weights into the live logit and holds everything else fixed:
        ``sigmoid(fixed_logit + Σ w_i · x_i)``; with every weight at its
        default it is the live probability. Both are tallied, time-weighted
        (gap to the next tick, capped at ``MAX_TICK_GAP_SECONDS``), into
        the tick's local day, deciding "occupied" at ``threshold``.

        Nothing is scored below ``FUSION_MIN_SAMPLES``: until then the
        weights are not reported, let alone candidates for promotion.

        Returns:
            The number of ticks actually tallied.
        """
        if self.state.samples < FUSION_MIN_SAMPLES:
            return 0
        scorable = [
            t
            for t in self._new_ticks(ticks)
            if t.probability is not None and t.fixed_logit is not None
        ]
        if len(scorable) < 2:
            return 0
        gaps = [
            min(
                max((b.timestamp - a.timestamp).total_seconds(), 0.0),
                MAX_TICK_GAP_SECONDS,
            )
            for a, b in pairwise(scorable)
        ]
        gaps.append(gaps[-1])
        weights = self.state.weights
        scored = 0
        for tick, seconds in zip(scorable, gaps, strict=True):
            # Both were filtered above; the checker can't follow that.
            probability = tick.probability
            z = tick.fixed_logit
            if seconds <= 0 or probability is None or z is None:
                continue
            truth = _tick_truth(tick, occupied_intervals)
            for entity_id, x in tick.features.items():
                if math.isfinite(x):
                    z += weights.get(entity_id, defaults.get(entity_id, 0.0)) * x
            learned = clamp_probability(sigmoid(z))
            day = to_local(tick.timestamp).date().isoformat()
            tallies = self.state.score_days.setdefault(
                day, {"live": CalibrationTally(), "learned": CalibrationTally()}
            )
            tallies["live"].add(
                probability, on=probability >= threshold, truth=truth, seconds=seconds
            )
            tallies["learned"].add(
                learned, on=learned >= threshold, truth=truth, seconds=seconds
            )
            scored += 1
        for day in sorted(self.state.score_days)[:-SCORE_HISTORY_DAYS]:
            del self.state.score_days[day]
        return scored

    def calibration(self) -> dict[str, Any] | None:
        """Live vs learned scores, pooled over the latest days (diagnostics)."""
        days = sorted(self.state.score_days)[-SCORE_WINDOW_DAYS:]
        if not days:
            return None
        pooled = {"live": CalibrationTally(), "learned": CalibrationTally()}
        daily: list[dict[str, Any]] = []
        for day in days:
            tallies = self.state.score_days[day]
            row: dict[str, Any] = {"date": day}
            for model, tally in pooled.items():
                if model in tallies:
                    tally.merge(tallies[model])
                    row[f"{model}_ece"] = (
                        tallies[model].summary().get("expected_calibration_error")
                    )
            daily.append(row)
        return {
            "days": len(days),
            "live": pooled["live"].summary(),
            "learned": pooled["learned"].summary(),
            "daily": daily,
        }

    def sync_entities(self, signatures: dict[str, str]) -> list[str]:
        """Forget entities that left the area or changed meaning.

        A removed sensor's weight is dead weight in the export; a sensor
        whose active states changed learned what the old states meant.
        Either way its weight and sample count go, and a changed one
        starts again from its live default. An entity with no stored
        fingerprint (state saved before fingerprints) adopts the current one.

        Note: ``score_days`` are not rewritten — days already tallied keep
        the scores the old weight produced until they age out of the window.

        Args:
            signatures: entity_id -> fingerprint for the area's current
                entities.

        Returns:
            The entity ids whose learning was discarded.
        """
        state = self.state
        dropped = sorted(
            entity_id
            for entity_id in set(state.weights) | set(state.entity_samples)
            if entity_id not in signatures
            or state.signatures.get(entity_id, signatures[entity_id])
            != signatures[entity_id]
        )
        for entity_id in dropped:
            state.weights.pop(entity_id, None)
            state.entity_samples.pop(entity_id, None)
        state.signatures = {
            entity_id: signatures[entity_id] for entity_id in state.weights
        }
        return dropped

    def snapshot(self, defaults: dict[str, float]) -> dict[str, Any]:
        """Return the JSON-safe diagnostics block for this area.

        Below the sample gate only the counters are reported — a weight
        learned from an hour of data is noise wearing a number's
        clothes. Above it, each weight's ``reliable`` flag says whether
        that entity has passed the same gate on its own (a sensor added
        or reset later has not).
        """
        block: dict[str, Any] = {
            "shadow_mode": True,
            "engaged": False,
            "samples": self.state.samples,
            "min_samples": FUSION_MIN_SAMPLES,
        }
        if self.state.samples >= FUSION_MIN_SAMPLES:
            weights: dict[str, dict[str, Any]] = {}
            for entity_id, w in sorted(self.state.weights.items()):
                entity_samples = self.state.entity_samples.get(entity_id, 0)
                weights[entity_id] = {
                    "learned_weight": round(w, 4),
                    # The live pipeline's weight for it: configured weight
                    # x information gain (0 when its learned likelihoods
                    # carry no information), the anchor learning starts from.
                    "effective_weight": round(defaults.get(entity_id, 0.0), 4),
                    "samples": entity_samples,
                    "reliable": entity_samples >= FUSION_MIN_SAMPLES,
                }
            block["weights"] = weights
        if (calibration := self.calibration()) is not None:
            block["calibration"] = calibration
        return block


def _with_timestamp(tick: FusionTick, timestamp: datetime) -> FusionTick:
    return FusionTick(
        timestamp=timestamp,
        bias=tick.bias,
        features=tick.features,
        truth=tick.truth,
        probability=tick.probability,
        fixed_logit=tick.fixed_logit,
    )
