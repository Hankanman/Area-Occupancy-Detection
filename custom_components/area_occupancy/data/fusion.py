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
  excludes them for the same reason). MEDIA stays learnable but shares
  the caveat — its learned weight will read optimistically high and
  must be judged accordingly at promotion time.
* **L2 anchor toward the live defaults**: with little data the learned
  weight stays near ``effective_weight``; a cold start is exactly
  today's behavior.
* **Minimum-sample gate** before a learned weight is even *reported*
  (``FUSION_MIN_SAMPLES`` ticks observed for the area).

Like ``metrics.py`` and ``adjacency.py``, this module has no
coordinator, HA, or DB dependencies — callers gather inputs and pass
them in, so the math stays testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise

from ..const import FUSION_L2, FUSION_LEARNING_RATE, FUSION_MIN_SAMPLES, MAX_WEIGHT
from ..time_utils import to_local
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
        for name in ("bin_weight", "bin_probability", "bin_hits"):
            mine = getattr(self, name)
            for i, value in enumerate(getattr(other, name)):
                mine[i] += value
        self.true_on += other.true_on
        self.false_on += other.false_on
        self.false_off += other.false_off
        self.true_off += other.true_off

    def summary(self) -> dict:
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

    def to_dict(self) -> dict:
        """Serialize (JSON-safe)."""
        return vars(self).copy()

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationTally:
        """Restore; raises on malformed data (callers fall back to empty)."""
        tally = cls(**{k: data[k] for k in data if k in cls.__dataclass_fields__})
        for name in ("bin_weight", "bin_probability", "bin_hits"):
            values = [float(v) for v in getattr(tally, name)]
            if len(values) != DEFAULT_CALIBRATION_BINS:
                raise ValueError(name)
            setattr(tally, name, values)
        return tally


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

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> FusionState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            return cls(
                weights={
                    str(k): float(v) for k, v in (data.get("weights") or {}).items()
                },
                samples=int(data.get("samples", 0)),
                entity_samples={
                    str(k): int(v)
                    for k, v in (data.get("entity_samples") or {}).items()
                },
                signatures={
                    str(k): str(v) for k, v in (data.get("signatures") or {}).items()
                },
                trained_through=(
                    datetime.fromisoformat(raw)
                    if (raw := data.get("trained_through"))
                    else None
                ),
                score_days={
                    str(day): {
                        str(model): CalibrationTally.from_dict(t)
                        for model, t in tallies.items()
                    }
                    for day, tallies in (data.get("score_days") or {}).items()
                },
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            return cls()


class FusionLearner:
    """Online logistic-weight learner for one area."""

    def __init__(self, state: FusionState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or FusionState()

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
            ticks: Training rows, any order; already-trained ones are skipped.
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
        if self.state.trained_through is not None:
            ticks = [t for t in ticks if t.timestamp > self.state.trained_through]
        if not ticks:
            return 0
        weights = self.state.weights
        for tick in ticks:
            occupied = (
                tick.truth
                if tick.truth is not None
                else _is_occupied_at(tick.timestamp, occupied_intervals)
            )
            y = 1.0 if occupied else 0.0
            z = tick.bias
            for entity_id, x in tick.features.items():
                w = weights.get(entity_id, defaults.get(entity_id, 0.0))
                z += w * x
            prediction = sigmoid(z)
            error = prediction - y
            for entity_id, x in tick.features.items():
                default = defaults.get(entity_id, 0.0)
                w = weights.get(entity_id, default)
                gradient = error * x + l2 * (w - default)
                w -= learning_rate * gradient
                weights[entity_id] = min(max(w, 0.0), MAX_WEIGHT)
            for entity_id in tick.features:
                self.state.entity_samples[entity_id] = (
                    self.state.entity_samples.get(entity_id, 0) + 1
                )
        self.state.samples += len(ticks)
        self.state.trained_through = max(t.timestamp for t in ticks)
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
            The number of ticks scored.
        """
        if self.state.samples < FUSION_MIN_SAMPLES:
            return 0
        ticks = sorted(
            (
                t
                for t in ticks
                if t.probability is not None
                and t.fixed_logit is not None
                and (
                    self.state.trained_through is None
                    or t.timestamp > self.state.trained_through
                )
            ),
            key=lambda t: t.timestamp,
        )
        if len(ticks) < 2:
            return 0
        gaps = [
            min(
                max((b.timestamp - a.timestamp).total_seconds(), 0.0),
                MAX_TICK_GAP_SECONDS,
            )
            for a, b in pairwise(ticks)
        ]
        gaps.append(gaps[-1])
        weights = self.state.weights
        for tick, seconds in zip(ticks, gaps, strict=True):
            if seconds <= 0:
                continue
            truth = (
                tick.truth
                if tick.truth is not None
                else _is_occupied_at(tick.timestamp, occupied_intervals)
            )
            z = tick.fixed_logit
            for entity_id, x in tick.features.items():
                z += weights.get(entity_id, defaults.get(entity_id, 0.0)) * x
            learned = clamp_probability(sigmoid(z))
            day = to_local(tick.timestamp).date().isoformat()
            tallies = self.state.score_days.setdefault(
                day, {"live": CalibrationTally(), "learned": CalibrationTally()}
            )
            tallies["live"].add(
                tick.probability,
                on=tick.probability >= threshold,
                truth=truth,
                seconds=seconds,
            )
            tallies["learned"].add(
                learned, on=learned >= threshold, truth=truth, seconds=seconds
            )
        for day in sorted(self.state.score_days)[:-SCORE_HISTORY_DAYS]:
            del self.state.score_days[day]
        return len(ticks)

    def calibration(self) -> dict | None:
        """Live vs learned scores, pooled over the latest days (diagnostics)."""
        days = sorted(self.state.score_days)[-SCORE_WINDOW_DAYS:]
        if not days:
            return None
        pooled = {"live": CalibrationTally(), "learned": CalibrationTally()}
        daily = []
        for day in days:
            tallies = self.state.score_days[day]
            row = {"date": day}
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

        Args:
            signatures: entity_id -> fingerprint for the area's current
                entities.

        Returns:
            The entity ids whose learning was discarded.
        """
        state = self.state
        dropped = [
            entity_id
            for entity_id in set(state.weights) | set(state.entity_samples)
            if entity_id not in signatures
            or state.signatures.get(entity_id, signatures[entity_id])
            != signatures[entity_id]
        ]
        for entity_id in dropped:
            state.weights.pop(entity_id, None)
            state.entity_samples.pop(entity_id, None)
            state.signatures.pop(entity_id, None)
        state.signatures = {
            entity_id: signatures[entity_id] for entity_id in state.weights
        }
        return sorted(dropped)

    def snapshot(self, defaults: dict[str, float]) -> dict:
        """Return the JSON-safe diagnostics block for this area.

        Below the sample gate only the counters are reported — a weight
        learned from an hour of data is noise wearing a number's
        clothes.
        """
        block: dict = {
            "shadow_mode": True,
            "engaged": False,
            "samples": self.state.samples,
            "min_samples": FUSION_MIN_SAMPLES,
        }
        if self.state.samples >= FUSION_MIN_SAMPLES:
            block["weights"] = {
                entity_id: {
                    "learned_weight": round(w, 4),
                    # The live pipeline's weight for it: configured weight
                    # x information gain (0 when its learned likelihoods
                    # carry no information), the anchor learning starts from.
                    "effective_weight": round(defaults.get(entity_id, 0.0), 4),
                    "samples": self.state.entity_samples.get(entity_id, 0),
                }
                for entity_id, w in sorted(self.state.weights.items())
            }
        if (calibration := self.calibration()) is not None:
            block["calibration"] = calibration
        return block
