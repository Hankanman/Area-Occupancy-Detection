"""Shadow-mode sensor likelihoods without the database (#603, phase A).

Today each sensor's learned likelihoods come from the hourly correlation
analysis replaying 30 days of intervals out of SQLite:

* a binary sensor's ``p(active | occupied)`` and ``p(active | empty)`` as
  ratios of durations, clamped to ``[0.05, 0.95]``;
* a numeric sensor's mean and standard deviation of its readings while
  occupied and while empty (one sample per recorded value change), and the
  Pearson correlation of its value with occupancy.

Every one of these is a ratio of sums, so it can be kept as running sums
updated tick by tick against the live ground-truth label
(:mod:`.ground_truth`) and persisted as a few numbers per sensor. To stand
in for the database's 30-day window the sums forget exponentially, with a
half-life of ``LIKELIHOOD_HALF_LIFE_DAYS``.

Shadow-mode contract: computed, persisted and compared daily with the
database's values (diagnostics ``likelihood_shadow``); never read by the
probability path until #603's switch-over.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math

from ..time_utils import ensure_utc_datetime
from .online_prior import MAX_TICK_GAP_SECONDS

# Forgetting half-life standing in for the database's 30-day window.
LIKELIHOOD_HALF_LIFE_DAYS = 15.0
# The database's clamp on binary likelihoods.
BINARY_MIN = 0.05
BINARY_MAX = 0.95
# Time of each class (occupied, empty) before a binary sensor's
# likelihoods are reported.
BINARY_MIN_CLASS_SECONDS = 3600.0
# Value-change samples of each class before numeric stats are reported.
NUMERIC_MIN_SAMPLES = 10


def _decay_factor(elapsed_seconds: float) -> float:
    half_life = LIKELIHOOD_HALF_LIFE_DAYS * 86400.0
    return 0.5 ** (max(elapsed_seconds, 0.0) / half_life)


@dataclass
class BinaryStats:
    """Decayed duration sums for one binary sensor."""

    occupied: float = 0.0
    empty: float = 0.0
    active_occupied: float = 0.0
    active_empty: float = 0.0

    def add(self, seconds: float, *, active: bool, occupied: bool) -> None:
        """Credit ``seconds`` of the given state."""
        if occupied:
            self.occupied += seconds
            if active:
                self.active_occupied += seconds
        else:
            self.empty += seconds
            if active:
                self.active_empty += seconds

    def decay(self, factor: float) -> None:
        """Forget: scale every sum by ``factor``."""
        self.occupied *= factor
        self.empty *= factor
        self.active_occupied *= factor
        self.active_empty *= factor

    def likelihoods(self) -> tuple[float, float] | None:
        """``(p_true, p_false)`` as the database computes them, or None.

        None until each class has ``BINARY_MIN_CLASS_SECONDS``, and when the
        sensor was never active while occupied (the database then falls
        back to type defaults).
        """
        if self.occupied < BINARY_MIN_CLASS_SECONDS or self.empty < (
            BINARY_MIN_CLASS_SECONDS
        ):
            return None
        if self.active_occupied <= 0.0:
            return None
        p_true = self.active_occupied / self.occupied
        p_false = self.active_empty / self.empty
        return (
            min(max(p_true, BINARY_MIN), BINARY_MAX),
            min(max(p_false, BINARY_MIN), BINARY_MAX),
        )


@dataclass
class NumericStats:
    """Decayed sums for one numeric sensor, per value-change sample."""

    # Per class: weight, Σx, Σx² (mean and variance from these).
    w_occupied: float = 0.0
    sx_occupied: float = 0.0
    sxx_occupied: float = 0.0
    w_empty: float = 0.0
    sx_empty: float = 0.0
    sxx_empty: float = 0.0
    # Pearson of value against the 0/1 label: Σxy (y is the label).
    sxy: float = 0.0
    last_value: float | None = None

    def add(self, value: float, *, occupied: bool) -> None:
        """Add one value-change sample with its label."""
        if occupied:
            self.w_occupied += 1.0
            self.sx_occupied += value
            self.sxx_occupied += value * value
            self.sxy += value
        else:
            self.w_empty += 1.0
            self.sx_empty += value
            self.sxx_empty += value * value

    def decay(self, factor: float) -> None:
        """Forget: scale every sum by ``factor``."""
        for name in (
            "w_occupied",
            "sx_occupied",
            "sxx_occupied",
            "w_empty",
            "sx_empty",
            "sxx_empty",
            "sxy",
        ):
            setattr(self, name, getattr(self, name) * factor)

    @staticmethod
    def _mean_std(w: float, sx: float, sxx: float) -> tuple[float, float]:
        mean = sx / w
        return mean, math.sqrt(max(sxx / w - mean * mean, 0.0))

    def stats(self) -> dict[str, float] | None:
        """Means, population standard deviations and correlation, or None.

        Matches the database: ``np.mean``/``np.std`` per class, and
        Pearson between value and occupancy over all samples.
        """
        if self.w_occupied < NUMERIC_MIN_SAMPLES or self.w_empty < NUMERIC_MIN_SAMPLES:
            return None
        mean_occ, std_occ = self._mean_std(
            self.w_occupied, self.sx_occupied, self.sxx_occupied
        )
        mean_emp, std_emp = self._mean_std(self.w_empty, self.sx_empty, self.sxx_empty)
        n = self.w_occupied + self.w_empty
        sx = self.sx_occupied + self.sx_empty
        sxx = self.sxx_occupied + self.sxx_empty
        sy = self.w_occupied  # Σy with y ∈ {0, 1}
        cov = self.sxy / n - (sx / n) * (sy / n)
        var_x = sxx / n - (sx / n) ** 2
        var_y = sy / n - (sy / n) ** 2
        correlation = cov / math.sqrt(var_x * var_y) if var_x > 0 and var_y > 0 else 0.0
        return {
            "mean_occupied": mean_occ,
            "std_occupied": std_occ,
            "mean_unoccupied": mean_emp,
            "std_unoccupied": std_emp,
            "correlation": correlation,
        }


@dataclass
class LikelihoodShadowState:
    """Per-area shadow likelihood sums (persisted)."""

    binary: dict[str, BinaryStats] = field(default_factory=dict)
    numeric: dict[str, NumericStats] = field(default_factory=dict)
    last_tick: datetime | None = None
    last_label: bool = False
    last_active: dict[str, bool] = field(default_factory=dict)
    # Daily worst-case gap to the database's values, oldest first.
    diff_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "binary": {k: vars(v) for k, v in self.binary.items()},
            "numeric": {k: vars(v) for k, v in self.numeric.items()},
            "last_tick": self.last_tick.isoformat() if self.last_tick else None,
            "last_label": self.last_label,
            "last_active": dict(self.last_active),
            "diff_history": list(self.diff_history),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LikelihoodShadowState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            return cls(
                binary={
                    str(k): BinaryStats(**v)
                    for k, v in (data.get("binary") or {}).items()
                },
                numeric={
                    str(k): NumericStats(**v)
                    for k, v in (data.get("numeric") or {}).items()
                },
                last_tick=(
                    ensure_utc_datetime(datetime.fromisoformat(raw))
                    if (raw := data.get("last_tick"))
                    else None
                ),
                last_label=bool(data.get("last_label", False)),
                last_active={
                    str(k): bool(v) for k, v in (data.get("last_active") or {}).items()
                },
                diff_history=[
                    dict(entry)
                    for entry in (data.get("diff_history") or [])
                    if isinstance(entry, dict)
                ],
            )
        except (AttributeError, TypeError, ValueError):
            return cls()


class LikelihoodShadow:
    """Learns one area's sensor likelihoods tick by tick."""

    def __init__(self, state: LikelihoodShadowState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or LikelihoodShadowState()

    def observe(
        self,
        *,
        now: datetime,
        label: bool,
        binary: dict[str, bool],
        numeric: dict[str, float],
    ) -> None:
        """Record one tick.

        The time since the previous tick is credited to the previous
        tick's states and label (gaps over ``MAX_TICK_GAP_SECONDS`` count
        nothing). Numeric sensors add a sample only when their value
        changed, as the database records one row per change.

        Args:
            now: The tick time.
            label: The live ground-truth label.
            binary: entity_id -> active, for the area's binary sensors.
            numeric: entity_id -> current value, for its numeric sensors.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        if state.last_tick is not None:
            gap = (now - state.last_tick).total_seconds()
            if gap > 0:
                factor = _decay_factor(gap)
                for stats in state.binary.values():
                    stats.decay(factor)
                for stats in state.numeric.values():
                    stats.decay(factor)
            if 0 < gap <= MAX_TICK_GAP_SECONDS:
                for entity_id, active in state.last_active.items():
                    state.binary.setdefault(entity_id, BinaryStats()).add(
                        gap, active=active, occupied=state.last_label
                    )
        for entity_id, value in numeric.items():
            stats = state.numeric.setdefault(entity_id, NumericStats())
            if stats.last_value != value:
                stats.add(value, occupied=label)
                stats.last_value = value
        state.last_tick = now
        state.last_label = label
        state.last_active = dict(binary)

    def forget_others(self, entity_ids: set[str]) -> None:
        """Drop sensors no longer in the area."""
        for table in (self.state.binary, self.state.numeric):
            for entity_id in [e for e in table if e not in entity_ids]:
                del table[entity_id]

    def compare(self, db_values: dict[str, dict[str, float]]) -> dict:
        """Compare the shadow likelihoods with the database's.

        Args:
            db_values: entity_id -> the database-learned values, with keys
                ``p_true``/``p_false`` (binary) or ``correlation`` and the
                Gaussian fields (numeric).

        Returns:
            Per-sensor shadow and database values, and ``max_diff``: the
            largest gap over binary probabilities and numeric correlations
            (None when nothing is comparable yet).
        """
        sensors: dict[str, dict] = {}
        diffs: list[float] = []
        for entity_id, stats in self.state.binary.items():
            shadow = stats.likelihoods()
            db = db_values.get(entity_id, {})
            row: dict = {
                "shadow": [round(v, 4) for v in shadow] if shadow else None,
                "db": [db.get("p_true"), db.get("p_false")] if "p_true" in db else None,
            }
            if shadow and "p_true" in db and "p_false" in db:
                gap = max(abs(shadow[0] - db["p_true"]), abs(shadow[1] - db["p_false"]))
                row["diff"] = round(gap, 4)
                diffs.append(gap)
            sensors[entity_id] = row
        for entity_id, stats in self.state.numeric.items():
            shadow_stats = stats.stats()
            db = db_values.get(entity_id, {})
            row = {
                "shadow": {k: round(v, 4) for k, v in shadow_stats.items()}
                if shadow_stats
                else None,
                "db": db or None,
            }
            if shadow_stats and "correlation" in db:
                gap = abs(shadow_stats["correlation"] - db["correlation"])
                row["diff"] = round(gap, 4)
                diffs.append(gap)
            sensors[entity_id] = row
        return {"sensors": sensors, "max_diff": max(diffs) if diffs else None}

    def record_divergence(self, day: str, max_diff: float | None) -> None:
        """Fold today's worst gap into the daily history (kept 90 days)."""
        if max_diff is None:
            return
        history = self.state.diff_history
        if history and history[-1].get("date") == day:
            history[-1]["max_diff"] = max(history[-1]["max_diff"], max_diff)
        else:
            history.append({"date": day, "max_diff": max_diff})
        del history[:-90]

    def days_within_tolerance(self, tolerance: float) -> int:
        """Consecutive most-recent days whose worst gap stayed in tolerance."""
        streak = 0
        for entry in reversed(self.state.diff_history):
            if entry.get("max_diff", math.inf) > tolerance:
                break
            streak += 1
        return streak
