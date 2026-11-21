"""Shadow-mode bounded evidence for auxiliary and environmental sensors.

The shadow model never engages occupancy decisions. Per sensor it keeps a
robust baseline (EMA of mean and absolute deviation, learned only while the
area is confidently unoccupied), scores positive level deviations (``z``) and
short-term rises (``j``), and maintains a decaying contribution ``E_s``. The
area value ``E_aux`` is the capped sum of all sensor contributions and is
exposed for diagnostics only.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
import math

from ..time_utils import ensure_utc_datetime
from .entity_type import InputType

AUXILIARY_SHADOW_STORE_VERSION = 1
AUXILIARY_E_MAX_DEFAULT = 1.5
MAX_TICK_GAP_SECONDS = 900.0
"""Upper bound for the elapsed time used in a single baseline EMA step."""
MISSING_RETENTION_SECONDS = 7 * 86400.0
"""Sensors without a valid value for this long (and ~zero evidence) are pruned."""

# The baseline tracks an EMA of |x - mu|, i.e. the *mean* absolute deviation.
# For Gaussian noise E|x - mu| = sigma * sqrt(2 / pi), hence sigma = MAD * sqrt(pi/2).
# (1.4826 would be the factor for the *median* absolute deviation.)
_MAD_TO_SIGMA = math.sqrt(math.pi / 2.0)
_PRUNE_EPSILON = 1e-6
_MIN_TAU_S = 1.0
_MIN_SCALE = 1e-9
_PRECISION = 6
_WET_ROOM_PURPOSES = frozenset({"bathroom", "food_prep"})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _opt_float(value: object) -> float | None:
    """Return a finite float or ``None`` (rejects NaN/inf and garbage)."""
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _float(value: object, default: float) -> float:
    result = _opt_float(value)
    return default if result is None else result


def _int(value: object, default: int = 0) -> int:
    result = _opt_float(value)
    return default if result is None else int(result)


def _parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = (
            value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        )
        return ensure_utc_datetime(parsed)
    except (TypeError, ValueError):
        return None


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _elapsed_s(since: datetime | None, now: datetime) -> float:
    if since is None:
        return 0.0
    return max((now - since).total_seconds(), 0.0)


def _decay_factor(dt: float, tau_s: float) -> float:
    return math.exp(-dt / max(tau_s, _MIN_TAU_S)) if dt > 0 else 1.0


def _r(value: float) -> float:
    return round(value, _PRECISION)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuxiliarySensorParams:
    """Tunable per-sensor parameters for the bounded shadow model."""

    role_weight: float = 0.5
    tau_baseline_s: float = 21600.0
    sigma_min_s: float = 1.0
    z_max: float = 4.0
    max_jump_dt_s: float = 900.0
    jump_scale: float = 1.5
    tau_decay_s: float = 600.0
    w_base: float = 0.20
    level_weight: float = 0.6
    jump_weight: float = 0.4
    outlier_multiplier: float = 6.0
    fresh_for_s: float = 300.0
    stale_after_s: float = 900.0
    stuck_quality: float = 0.15
    repeated_quality: float = 0.50
    repeat_delta: float = 0.01
    repeated_limit: int = 20
    outlier_quality: float = 0.10
    outlier_confirm_samples: int = 2
    baseline_clip_k: float = 3.0
    rebaseline_after_s: float = 7200.0
    rebaseline_band_k: float = 1.5

    def weights(self) -> tuple[float, float]:
        """Return normalized, non-negative level and jump weights."""
        level = max(self.level_weight, 0.0)
        jump = max(self.jump_weight, 0.0)
        total = level + jump
        if total <= 0.0:
            return 0.0, 0.0
        return level / total, jump / total


@dataclass(frozen=True)
class AuxiliaryReading:
    """One valid or invalid numeric reading for the shadow model."""

    value: float
    input_type: InputType
    confidently_unoccupied: bool
    stuck: bool = False
    timestamp: datetime | None = None
    lux_suppressed: bool = False


@dataclass
class AuxiliarySensorState:
    """Persisted baseline and decaying contribution for one sensor."""

    input_type: str | None = None
    mu: float | None = None
    mad: float = 0.0
    sigma: float | None = None
    previous_value: float | None = None
    last_value: float | None = None
    repeated_count: int = 0
    contribution: float = 0.0
    last_tick: datetime | None = None
    last_value_at: datetime | None = None
    last_reading_ts: datetime | None = None
    outlier_count: int = 0
    relearn_since: datetime | None = None
    relearn_anchor: float | None = None
    last_snapshot: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize the state for JSON-backed Home Assistant storage."""
        return {
            "input_type": self.input_type,
            "mu": self.mu,
            "mad": self.mad,
            "sigma": self.sigma,
            "previous_value": self.previous_value,
            "last_value": self.last_value,
            "repeated_count": self.repeated_count,
            "contribution": self.contribution,
            "last_tick": _iso(self.last_tick),
            "last_value_at": _iso(self.last_value_at),
            "last_reading_ts": _iso(self.last_reading_ts),
            "outlier_count": self.outlier_count,
            "relearn_since": _iso(self.relearn_since),
            "relearn_anchor": self.relearn_anchor,
            "last_snapshot": dict(self.last_snapshot),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AuxiliarySensorState:
        """Restore one sensor state field by field.

        Invalid or missing fields fall back to their defaults instead of
        discarding the whole learned baseline.
        """
        if not isinstance(data, Mapping):
            return cls()
        raw_type = data.get("input_type")
        sigma = _opt_float(data.get("sigma"))
        snapshot = data.get("last_snapshot")
        return cls(
            input_type=str(raw_type) if raw_type is not None else None,
            mu=_opt_float(data.get("mu")),
            mad=max(_float(data.get("mad"), 0.0), 0.0),
            sigma=None if sigma is None else max(sigma, 0.0),
            previous_value=_opt_float(data.get("previous_value")),
            last_value=_opt_float(data.get("last_value")),
            repeated_count=max(_int(data.get("repeated_count")), 0),
            contribution=max(_float(data.get("contribution"), 0.0), 0.0),
            last_tick=_parse_datetime(data.get("last_tick")),
            last_value_at=_parse_datetime(data.get("last_value_at")),
            last_reading_ts=_parse_datetime(data.get("last_reading_ts")),
            outlier_count=max(_int(data.get("outlier_count")), 0),
            relearn_since=_parse_datetime(data.get("relearn_since")),
            relearn_anchor=_opt_float(data.get("relearn_anchor")),
            last_snapshot=dict(snapshot) if isinstance(snapshot, Mapping) else {},
        )


@dataclass
class AuxiliaryShadowState:
    """Persisted per-area auxiliary shadow state."""

    sensors: dict[str, AuxiliarySensorState] = field(default_factory=dict)
    e_aux: float = 0.0
    last_tick: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize for the HA storage helper."""
        return {
            "sensors": {key: value.to_dict() for key, value in self.sensors.items()},
            "e_aux": self.e_aux,
            "last_tick": _iso(self.last_tick),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> AuxiliaryShadowState:
        """Restore an area state, tolerating malformed payloads."""
        if not isinstance(data, Mapping):
            return cls()
        raw_sensors = data.get("sensors")
        sensors: dict[str, AuxiliarySensorState] = {}
        if isinstance(raw_sensors, Mapping):
            sensors = {
                str(key): AuxiliarySensorState.from_dict(value)
                for key, value in raw_sensors.items()
                if isinstance(value, Mapping)
            }
        return cls(
            sensors=sensors,
            e_aux=max(_float(data.get("e_aux"), 0.0), 0.0),
            last_tick=_parse_datetime(data.get("last_tick")),
        )


# ---------------------------------------------------------------------------
# Default profiles
# ---------------------------------------------------------------------------

_HUMIDITY_PARAMS = AuxiliarySensorParams(
    role_weight=0.70,
    tau_baseline_s=21600.0,
    sigma_min_s=2.0,
    z_max=4.0,
    max_jump_dt_s=900.0,
    jump_scale=1.5,
    tau_decay_s=900.0,
    w_base=0.22,
)
_HUMIDITY_WET_ROOM_PARAMS = replace(_HUMIDITY_PARAMS, role_weight=0.85)

_VOC_PARAMS = AuxiliarySensorParams(
    role_weight=0.65,
    tau_baseline_s=21600.0,
    sigma_min_s=10.0,
    z_max=4.0,
    max_jump_dt_s=900.0,
    jump_scale=1.5,
    tau_decay_s=900.0,
    w_base=0.20,
)

_AIR_QUALITY_PARAMS = AuxiliarySensorParams(
    role_weight=0.60,
    tau_baseline_s=21600.0,
    sigma_min_s=5.0,
    z_max=4.0,
    max_jump_dt_s=900.0,
    jump_scale=1.5,
    tau_decay_s=900.0,
    w_base=0.18,
)

_DEFAULT_PROFILES: dict[InputType, AuxiliarySensorParams] = {
    InputType.CO2: AuxiliarySensorParams(
        role_weight=0.95,
        tau_baseline_s=21600.0,
        sigma_min_s=50.0,
        z_max=4.0,
        max_jump_dt_s=900.0,
        jump_scale=1.5,
        tau_decay_s=900.0,
        w_base=0.28,
        level_weight=0.6,
        jump_weight=0.4,
    ),
    InputType.SOUND_PRESSURE: AuxiliarySensorParams(
        role_weight=0.95,
        tau_baseline_s=10800.0,
        sigma_min_s=2.0,
        z_max=4.0,
        max_jump_dt_s=300.0,
        jump_scale=1.25,
        tau_decay_s=300.0,
        w_base=0.30,
        level_weight=0.4,
        jump_weight=0.6,
    ),
    InputType.POWER: AuxiliarySensorParams(
        role_weight=0.90,
        tau_baseline_s=21600.0,
        sigma_min_s=0.5,
        z_max=5.0,
        max_jump_dt_s=900.0,
        jump_scale=1.5,
        tau_decay_s=600.0,
        w_base=0.30,
        level_weight=0.5,
        jump_weight=0.5,
    ),
    InputType.HUMIDITY: _HUMIDITY_PARAMS,
    InputType.VOC: _VOC_PARAMS,
    InputType.PM25: replace(_VOC_PARAMS, sigma_min_s=2.0),
    InputType.PM10: AuxiliarySensorParams(
        role_weight=0.55,
        tau_baseline_s=21600.0,
        sigma_min_s=5.0,
        z_max=4.0,
        max_jump_dt_s=900.0,
        jump_scale=1.5,
        tau_decay_s=900.0,
        w_base=0.18,
    ),
    InputType.ILLUMINANCE: AuxiliarySensorParams(
        role_weight=0.50,
        tau_baseline_s=21600.0,
        sigma_min_s=5.0,
        z_max=4.0,
        max_jump_dt_s=600.0,
        jump_scale=2.0,
        tau_decay_s=300.0,
        w_base=0.20,
        level_weight=0.0,
        jump_weight=1.0,
    ),
    InputType.CO: AuxiliarySensorParams(
        role_weight=0.75,
        tau_baseline_s=21600.0,
        sigma_min_s=1.0,
        z_max=4.0,
        max_jump_dt_s=900.0,
        jump_scale=1.5,
        tau_decay_s=900.0,
        w_base=0.22,
    ),
    InputType.TEMPERATURE: AuxiliarySensorParams(
        role_weight=0.35,
        tau_baseline_s=43200.0,
        sigma_min_s=0.5,
        z_max=4.0,
        max_jump_dt_s=900.0,
        jump_scale=2.0,
        tau_decay_s=1200.0,
        w_base=0.12,
    ),
    InputType.PRESSURE: AuxiliarySensorParams(
        role_weight=0.25,
        tau_baseline_s=43200.0,
        sigma_min_s=1.0,
        z_max=4.0,
        max_jump_dt_s=1800.0,
        jump_scale=2.0,
        tau_decay_s=1800.0,
        w_base=0.10,
    ),
    InputType.AIR_QUALITY: _AIR_QUALITY_PARAMS,
    InputType.ENVIRONMENTAL: _AIR_QUALITY_PARAMS,
}
_FALLBACK_PARAMS = AuxiliarySensorParams()


def default_auxiliary_params(
    input_type: InputType,
    purpose: str | None = None,
) -> AuxiliarySensorParams:
    """Return a conservative role/profile for an auxiliary sensor type.

    Profiles are immutable module-level constants, so no objects are
    allocated per tick.
    """
    if (
        input_type == InputType.HUMIDITY
        and (purpose or "").lower() in _WET_ROOM_PURPOSES
    ):
        return _HUMIDITY_WET_ROOM_PARAMS
    return _DEFAULT_PROFILES.get(input_type, _FALLBACK_PARAMS)


# ---------------------------------------------------------------------------
# Shadow model
# ---------------------------------------------------------------------------


class AuxiliaryShadow:
    """Compute bounded, decaying auxiliary evidence for one area.

    Each sensor has an independent bounded contribution. The contributions
    are summed and then capped once for the whole area.
    """

    def __init__(
        self,
        state: AuxiliaryShadowState | None = None,
        *,
        e_aux_max: float = AUXILIARY_E_MAX_DEFAULT,
        params: dict[str, AuxiliarySensorParams] | None = None,
        purpose: str | None = None,
    ) -> None:
        """Initialize from persisted state or empty state."""
        self.state = state or AuxiliaryShadowState()
        self.e_aux_max = max(_float(e_aux_max, AUXILIARY_E_MAX_DEFAULT), 0.0)
        self.params = dict(params or {})
        self.purpose = purpose

    def _params_for(
        self, sensor_id: str, input_type: InputType
    ) -> AuxiliarySensorParams:
        return self.params.get(sensor_id) or default_auxiliary_params(
            input_type, self.purpose
        )

    # -- public API ---------------------------------------------------------

    def observe(
        self,
        now: datetime,
        *,
        readings: dict[str, AuxiliaryReading],
        confidently_unoccupied_by_area: bool | None = None,
    ) -> None:
        """Update baselines, quality and decaying contributions for all readings.

        Sensors without a valid (finite numeric) reading in this tick are
        treated as missing: their contribution keeps decaying and still counts
        towards ``E_aux``.
        """
        now = ensure_utc_datetime(now)
        observed: set[str] = set()

        for sensor_id, reading in readings.items():
            value = _opt_float(reading.value)
            if value is None:
                continue
            self._observe_sensor(
                sensor_id, reading, value, now, confidently_unoccupied_by_area
            )
            observed.add(sensor_id)

        for sensor_id, sensor_state in self.state.sensors.items():
            if sensor_id not in observed:
                self._decay_missing(sensor_id, sensor_state, now)

        self._prune_expired(now, observed)
        self.state.e_aux = _clip(self._aggregate(), 0.0, self.e_aux_max)
        self.state.last_tick = now

    def prune(self, active_sensor_ids: Iterable[str]) -> list[str]:
        """Drop persisted state of sensors no longer configured for the area."""
        active = set(active_sensor_ids)
        removed = [key for key in self.state.sensors if key not in active]
        for key in removed:
            del self.state.sensors[key]
        if removed:
            self.state.e_aux = _clip(self._aggregate(), 0.0, self.e_aux_max)
        return removed

    def snapshot(self) -> dict[str, object]:
        """Return diagnostics-safe shadow values and suppression reasons."""
        sensors: dict[str, object] = {}
        for sensor_id, state in self.state.sensors.items():
            if not state.last_snapshot:
                continue
            entry = dict(state.last_snapshot)
            reasons = entry.get("reasons")
            if isinstance(reasons, list):
                entry["reasons"] = list(reasons)
            sensors[sensor_id] = entry
        return {
            "shadow_mode": True,
            "engaged": False,
            "E_aux": _r(self.state.e_aux),
            "E_aux_max": self.e_aux_max,
            "area_cap": self.e_aux_max,
            "sensors": sensors,
        }

    # -- per-sensor update --------------------------------------------------

    def _observe_sensor(
        self,
        sensor_id: str,
        reading: AuxiliaryReading,
        value: float,
        now: datetime,
        area_unoccupied: bool | None,
    ) -> None:
        """Process one valid reading."""
        reasons: list[str] = []
        state = self.state.sensors.setdefault(sensor_id, AuxiliarySensorState())
        type_key = reading.input_type.value
        if state.input_type is not None and state.input_type != type_key:
            # A baseline learned in other units/physics is meaningless.
            state = AuxiliarySensorState()
            self.state.sensors[sensor_id] = state
            reasons.append("input_type_changed_reset")
        state.input_type = type_key
        params = self._params_for(sensor_id, reading.input_type)

        dt = _elapsed_s(state.last_tick, now)
        # Jumps are measured against the last *valid* value, so a sensor that
        # returns after an outage does not produce a spurious jump.
        value_dt = _elapsed_s(state.last_value_at or state.last_tick, now)
        unoccupied = bool(
            area_unoccupied
            if area_unoccupied is not None
            else reading.confidently_unoccupied
        )
        reading_ts = (
            ensure_utc_datetime(reading.timestamp)
            if reading.timestamp is not None
            else None
        )
        is_new_sample = reading_ts is None or reading_ts != state.last_reading_ts

        # Stale values describe the past and must not teach the baseline.
        freshness = self._freshness(reading_ts, now, params, reasons)
        usable = freshness > 0.0

        # Baseline initialisation -----------------------------------------
        initialized = False
        if state.mu is None and unoccupied and usable:
            self._reset_baseline(state, params, value)
            initialized = True
            reasons.append("baseline_initialized")

        scale = self._scale(state, params)
        outlier = (
            state.mu is not None
            and abs(value - state.mu) > params.outlier_multiplier * scale
        )

        # Outlier bookkeeping ----------------------------------------------
        relearned = False
        if not outlier:
            state.outlier_count = 0
            state.relearn_since = None
            state.relearn_anchor = None
        elif usable and is_new_sample:
            state.outlier_count += 1
            if unoccupied:
                relearned = self._track_relearn(state, params, value, scale, now)
        if outlier and not unoccupied:
            # Relearning requires an uninterrupted empty-room period.
            state.relearn_since = None
            state.relearn_anchor = None
        if relearned:
            self._reset_baseline(state, params, value)
            outlier = False
            reasons.append("baseline_relearned")
            scale = self._scale(state, params)

        # Robust baseline EMA ----------------------------------------------
        if (
            state.mu is not None
            and unoccupied
            and usable
            and not outlier
            and not initialized
            and not relearned
            and dt > 0
        ):
            self._update_baseline(state, params, value, dt, scale)
            reasons.append("baseline_ema_updated")
            scale = self._scale(state, params)

        # Level and jump scores --------------------------------------------
        if state.mu is None:
            z = 0.0
            reasons.append("baseline_missing")
        else:
            z = _clip((value - state.mu) / scale, 0.0, params.z_max)

        jump = 0.0
        if state.previous_value is not None and 0 < value_dt <= params.max_jump_dt_s:
            jump_scale = max(scale * params.jump_scale, _MIN_SCALE)
            jump = _clip((value - state.previous_value) / jump_scale, 0.0, 1.0)

        # Quality ----------------------------------------------------------
        repeated = self._track_repeats(state, params, value, reading_ts)

        validity = 1.0
        if reading.stuck:
            validity *= params.stuck_quality
            reasons.append("stuck_active")
        if repeated:
            validity *= params.repeated_quality
            reasons.append("repeated_stuck")
        if outlier:
            # A glitch is an *abrupt* excursion. A deviation that persists over
            # several samples, or that was reached by a gradual rise, is a real
            # physical change and is not dampened.
            abrupt = (
                state.previous_value is None
                or abs(value - state.previous_value)
                > 0.5 * params.outlier_multiplier * scale
            )
            if not abrupt or state.outlier_count >= max(
                params.outlier_confirm_samples, 1
            ):
                reasons.append("outlier_confirmed")
            else:
                validity *= _clip(params.outlier_quality, 0.0, 1.0)
                reasons.append("distribution_outlier")

        lux_suppressed = (
            reading.input_type == InputType.ILLUMINANCE and reading.lux_suppressed
        )
        if lux_suppressed:
            z = 0.0
            jump = 0.0
            reasons.append("lux_automation_suppressed")

        role_weight = _clip(params.role_weight, 0.0, 1.0)
        quality = _clip(freshness * validity * role_weight, 0.0, 1.0)
        level_weight, jump_weight = params.weights()

        # Contribution: peak hold with exponential decay --------------------
        decayed = state.contribution * _decay_factor(dt, params.tau_decay_s)
        candidate = (
            max(params.w_base, 0.0) * quality * (level_weight * z + jump_weight * jump)
        )
        state.contribution = max(decayed, candidate, 0.0)

        state.previous_value = value
        state.last_tick = now
        state.last_value_at = now
        state.last_snapshot = {
            "mu": _r(state.mu) if state.mu is not None else None,
            "sigma": _r(scale),
            "mad": _r(state.mad),
            "z": _r(z),
            "j": _r(jump),
            "quality": _r(quality),
            "role": role_weight,
            "freshness": _r(freshness),
            "validity": _r(validity),
            "baseline_updated": "baseline_ema_updated" in reasons,
            "outlier": outlier,
            "stuck": reading.stuck,
            "repeated_stuck": repeated,
            "lux_suppressed": lux_suppressed,
            "E_s": _r(state.contribution),
            "reasons": reasons,
        }

    def _decay_missing(
        self, sensor_id: str, state: AuxiliarySensorState, now: datetime
    ) -> None:
        """Decay a sensor that has no valid reading in this tick."""
        if state.last_tick is None:
            return
        dt = _elapsed_s(state.last_tick, now)
        if dt > 0:
            params = self._params_for(sensor_id, self._stored_input_type(state))
            state.contribution *= _decay_factor(dt, params.tau_decay_s)
            state.last_tick = now
            state.last_snapshot["E_s"] = _r(state.contribution)
            state.last_snapshot["reasons"] = ["reading_missing_decay"]

    def _prune_expired(self, now: datetime, observed: set[str]) -> None:
        """Forget sensors that have been silent far beyond any decay horizon."""
        expired = [
            key
            for key, state in self.state.sensors.items()
            if key not in observed
            and state.last_value_at is not None
            and _elapsed_s(state.last_value_at, now) > MISSING_RETENTION_SECONDS
            and state.contribution < _PRUNE_EPSILON
        ]
        for key in expired:
            del self.state.sensors[key]

    def _aggregate(self) -> float:
        """Sum each sensor contribution before applying the area cap."""
        return sum(
            max(state.contribution, 0.0) for state in self.state.sensors.values()
        )

    # -- building blocks ----------------------------------------------------

    @staticmethod
    def _scale(state: AuxiliarySensorState, params: AuxiliarySensorParams) -> float:
        return max(state.sigma or params.sigma_min_s, params.sigma_min_s, _MIN_SCALE)

    @staticmethod
    def _reset_baseline(
        state: AuxiliarySensorState, params: AuxiliarySensorParams, value: float
    ) -> None:
        state.mu = value
        state.mad = 0.0
        state.sigma = params.sigma_min_s
        state.outlier_count = 0
        state.relearn_since = None
        state.relearn_anchor = None

    @staticmethod
    def _track_relearn(
        state: AuxiliarySensorState,
        params: AuxiliarySensorParams,
        value: float,
        scale: float,
        now: datetime,
    ) -> bool:
        """Return True once an outlier level has been *stable* long enough.

        The run restarts whenever the value leaves a band around its anchor.
        This separates a genuine level shift (sensor moved, recalibrated,
        baseline learned on stale air) from a slow post-occupancy decay,
        which keeps drifting and therefore never qualifies.
        """
        band = max(params.rebaseline_band_k, 0.0) * scale
        if (
            state.relearn_since is None
            or state.relearn_anchor is None
            or abs(value - state.relearn_anchor) > band
        ):
            state.relearn_since = now
            state.relearn_anchor = value
            return False
        return _elapsed_s(state.relearn_since, now) >= params.rebaseline_after_s

    @staticmethod
    def _update_baseline(
        state: AuxiliarySensorState,
        params: AuxiliarySensorParams,
        value: float,
        dt: float,
        scale: float,
    ) -> None:
        """Robust EMA of mean and mean absolute deviation.

        ``alpha = 1 - exp(-dt / tau)`` is the exact discretisation of a
        continuous-time low-pass for irregular ticks; the step is capped after
        long gaps. The residual is Huber-clipped to ``baseline_clip_k * sigma``
        so residual occupancy effects cannot drag the baseline upwards.
        """
        if state.mu is None:
            return
        step = min(dt, MAX_TICK_GAP_SECONDS)
        alpha = 1.0 - math.exp(-step / max(params.tau_baseline_s, _MIN_TAU_S))
        limit = max(params.baseline_clip_k, 0.0) * scale
        residual = value - state.mu
        if limit > 0:
            residual = _clip(residual, -limit, limit)
        state.mu += alpha * residual
        state.mad = (1.0 - alpha) * state.mad + alpha * abs(residual)
        state.sigma = max(_MAD_TO_SIGMA * state.mad, params.sigma_min_s)

    @staticmethod
    def _freshness(
        reading_ts: datetime | None,
        now: datetime,
        params: AuxiliarySensorParams,
        reasons: list[str],
    ) -> float:
        if reading_ts is None:
            return 1.0
        age = _elapsed_s(reading_ts, now)
        if age > params.stale_after_s:
            reasons.append("stale")
            return 0.0
        if age > params.fresh_for_s:
            reasons.append("freshness_damped")
            span = max(params.stale_after_s - params.fresh_for_s, 1.0)
            return _clip(1.0 - (age - params.fresh_for_s) / span, 0.0, 1.0)
        return 1.0

    @staticmethod
    def _track_repeats(
        state: AuxiliarySensorState,
        params: AuxiliarySensorParams,
        value: float,
        reading_ts: datetime | None,
    ) -> bool:
        """Count identical consecutive samples.

        A sample with an unchanged source timestamp is the same HA state seen
        again in a later tick and is not counted as a new repeat.
        """
        is_new_sample = reading_ts is None or reading_ts != state.last_reading_ts
        if is_new_sample and state.last_value is not None:
            if abs(value - state.last_value) <= params.repeat_delta:
                state.repeated_count += 1
            else:
                state.repeated_count = 0
        state.last_value = value
        state.last_reading_ts = reading_ts
        return state.repeated_count >= params.repeated_limit

    @staticmethod
    def _stored_input_type(state: AuxiliarySensorState) -> InputType:
        try:
            return InputType(state.input_type)
        except (TypeError, ValueError):
            return InputType.ENVIRONMENTAL
