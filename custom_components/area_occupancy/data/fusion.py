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
comparison, per #501's phases.

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

from ..const import FUSION_L2, FUSION_LEARNING_RATE, FUSION_MIN_SAMPLES, MAX_WEIGHT
from ..utils import sigmoid
from .metrics import _is_occupied_at


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


@dataclass
class FusionState:
    """Serializable learned-weight state for one area."""

    weights: dict[str, float] = field(default_factory=dict)
    samples: int = 0
    # Timestamp of the newest tick already trained on. The analysis passes
    # its whole 24h window every hour, so without this each tick would be
    # stepped ~24 times and ``samples`` would pass the reporting gate early.
    trained_through: datetime | None = None

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "weights": dict(self.weights),
            "samples": self.samples,
            "trained_through": (
                self.trained_through.isoformat() if self.trained_through else None
            ),
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
                trained_through=(
                    datetime.fromisoformat(raw)
                    if (raw := data.get("trained_through"))
                    else None
                ),
            )
        except (AttributeError, TypeError, ValueError):
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
            y = 1.0 if _is_occupied_at(tick.timestamp, occupied_intervals) else 0.0
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
        self.state.samples += len(ticks)
        self.state.trained_through = max(t.timestamp for t in ticks)
        return len(ticks)

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
                    "default_weight": round(defaults.get(entity_id, 0.0), 4),
                }
                for entity_id, w in sorted(self.state.weights.items())
            }
        return block
