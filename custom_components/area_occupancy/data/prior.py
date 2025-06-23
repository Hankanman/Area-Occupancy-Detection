"""Area baseline prior (P(room occupied) *before* current evidence).

The class learns from recent recorder history, but also falls back to a
defensive default when data are sparse or sensors are being re-configured.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timedelta
import logging

from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util

from ..const import DEFAULT_PRIOR, MAX_PRIOR, MIN_PRIOR

_LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
class Prior:  # exported name must stay identical
    """Compute and cache the baseline probability for an Area entity."""

    CACHE_TTL = timedelta(hours=2)  # recalc at most this often

    # Heuristic thresholds (can be placed in config if needed)
    MOTION_QUORUM_RATIO = 0.60  # ≥60 % of motions must agree
    MIN_SAMPLE_SECONDS = 6_000  # ~1 h-40 m of data per sensor

    def __init__(
        self,
        hass: HomeAssistant,
        sensor_entity_ids: Sequence[str],
        recorder_days: int = 7,
    ) -> None:
        self._hass = hass
        self._sensor_ids = set(sensor_entity_ids)
        self._days = recorder_days

        self._cached_value: float | None = None
        self._cached_at: datetime | None = None
        self._cached_sensor_hash: int | None = None  # detects roster changes

    # --------------------------------------------------------------------- #
    @property
    def current_value(self) -> float:
        """Return the current cached prior value, or default if not yet calculated."""
        return self._cached_value if self._cached_value is not None else DEFAULT_PRIOR

    # --------------------------------------------------------------------- #
    async def update(self) -> float:
        """Return a baseline prior, re-computing if the cache is stale."""
        if self._is_cache_valid():
            return self._cached_value  # type: ignore[return-value]

        try:
            value = await self._calculate_prior()
        except Exception:  # pragma: no cover
            _LOGGER.exception(
                "Prior calculation failed, using default %.2f", DEFAULT_PRIOR
            )
            value = DEFAULT_PRIOR

        # Clip to sane range
        value = max(MIN_PRIOR, min(value, MAX_PRIOR))

        # Cache
        self._cached_value = value
        self._cached_at = dt_util.utcnow()
        self._cached_sensor_hash = hash(frozenset(self._sensor_ids))
        _LOGGER.debug("Calculated new area prior %.3f", value)
        return value

    # ------------------------------------------------------------------ #
    def _is_cache_valid(self) -> bool:
        if self._cached_value is None or self._cached_at is None:
            return False
        if (dt_util.utcnow() - self._cached_at) > self.CACHE_TTL:
            return False
        # in case sensors were added/removed
        return self._cached_sensor_hash == hash(frozenset(self._sensor_ids))

    # ------------------------------------------------------------------ #
    async def _calculate_prior(self) -> float:
        """Layered heuristic—a true Bayesian prior if sufficient data exist."""

        # 1. Consensus of motion sensors ----------------------------------
        motion_ids = [eid for eid in self._sensor_ids if ".motion_" in eid]
        if len(motion_ids) >= 2:
            ratio = await self._motion_quorum_ratio(motion_ids)
            if ratio is not None and ratio >= self.MOTION_QUORUM_RATIO:
                _LOGGER.debug("Using motion-quorum prior %.3f", ratio)
                return ratio

        # 2. Weighted confidence fusion -----------------------------------
        weighted = await self._confidence_weighted_prior()
        if weighted is not None:
            _LOGGER.debug("Using confidence-weighted prior %.3f", weighted)
            return weighted

        # 3. Time-of-day pattern ------------------------------------------
        tod = self._time_of_day_prior()
        if tod is not None:
            _LOGGER.debug("Using time-pattern prior %.3f", tod)
            return tod

        # 4. Default fallback ---------------------------------------------
        _LOGGER.debug("Falling back to default prior %.2f", DEFAULT_PRIOR)
        return DEFAULT_PRIOR

    # ------------------------------------------------------------------ #
    async def _motion_quorum_ratio(self, motion_ids: Sequence[str]) -> float | None:
        """Return fraction of seconds in which ≥60 % of motions were ON."""
        history = await self._async_get_history(motion_ids)
        if not history or any(
            len(counter) < self.MIN_SAMPLE_SECONDS for counter in history.values()
        ):
            return None

        # tally per-second OR across all sensors
        combined: Counter[int] = Counter()
        for states in history.values():
            combined.update(states)  # Counter of {timestamp:1}

        total_secs = len(combined)
        secs_on = sum(
            1
            for ts, cnt in combined.items()
            if cnt / len(motion_ids) >= self.MOTION_QUORUM_RATIO
        )
        return secs_on / total_secs if total_secs else None

    # ------------------------------------------------------------------ #
    async def _confidence_weighted_prior(self) -> float | None:
        """Uses per-sensor accuracy/confidence attributes if available."""
        reg = er.async_get(self._hass)
        acc_vals = []
        for eid in self._sensor_ids:
            ent = reg.async_get(eid)
            if ent and ent.capabilities and (acc := ent.capabilities.get("accuracy")):
                # accuracy assumed to be expressed as 0-1
                acc_vals.append(float(acc))
        if acc_vals:
            return sum(acc_vals) / len(acc_vals)
        return None

    # ------------------------------------------------------------------ #
    def _time_of_day_prior(self) -> float | None:
        """Simple heuristic: occupied during daytime, empty at night."""
        now = dt_util.now()
        hour = now.hour
        if 7 <= hour <= 22:
            return 0.35  # typical daytime baseline
        return 0.10  # night baseline

    # ------------------------------------------------------------------ #
    async def _async_get_history(
        self, entity_ids: Sequence[str]
    ) -> dict[str, Counter[int]]:
        """Return per-entity Counter of ON seconds over the look-back window.

        Simplified; relies on recorder.history async API.
        """
        from homeassistant.components.recorder.history import get_significant_states

        start = dt_util.utcnow() - timedelta(days=self._days)
        hist = get_significant_states(
            self._hass, start_time=start, entity_ids=list(entity_ids),
            minimal_response=False  # Ensure we get State objects
        )

        result: dict[str, Counter[int]] = {}
        for eid, states in hist.items():
            c = Counter()
            last_ts = None
            for s in states:
                if s.state == "on":  # type: ignore[union-attr]
                    last_ts = s.last_changed.timestamp()  # type: ignore[union-attr]
                elif last_ts:
                    for sec in range(int(last_ts), int(s.last_changed.timestamp())):  # type: ignore[union-attr]
                        c[sec] = 1
                    last_ts = None
            result[eid] = c
        return result
