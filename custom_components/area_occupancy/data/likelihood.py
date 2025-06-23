"""Per-sensor likelihoods P(E|H) and P(E|¬H).

Computes *overlap* of each sensor’s active intervals with the area’s
ground-truth occupied intervals, giving informative likelihoods that
differ between H and ¬H.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import MAX_PROBABILITY, MIN_PROBABILITY

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from .entity import Entity

_LOGGER = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return seconds of intersection between [a_start, a_end] and [b_start, b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


# ─────────────────────────────────────────────────────────────────────────────
class Likelihood:
    """Learn conditional probabilities for a single binary sensor."""

    def __init__(
        self,
        hass: HomeAssistant | None = None,
        sensor_entity_id: str | None = None,
        occupied_intervals: list[tuple[float, float]] | None = None,
        window_days: int = 7,
        prob_given_true: float = 0.5,
        prob_given_false: float = 0.5,
        last_updated: datetime | None = None,
    ) -> None:
        self._hass = hass
        self._eid = sensor_entity_id
        self._occ_ints = occupied_intervals or []  # list of (start_ts, end_ts)
        self._days = window_days

        self.prob_given_true: float = prob_given_true
        self.prob_given_false: float = prob_given_false
        self.last_updated: datetime = last_updated or dt_util.utcnow()

    # ------------------------------------------------------------------ #
    async def async_recompute(self) -> None:
        """Fetch sensor history and compute new likelihoods."""
        active_ints = await self._async_get_active_intervals()

        # Totals
        secs_H = sum(end - start for start, end in self._occ_ints)
        secs_not_H = (self._days * 86400) - secs_H

        if secs_H <= 0 or secs_not_H <= 0:
            _LOGGER.warning(
                "%s: insufficient occupancy history; keeping defaults", self._eid
            )
            return

        # Overlaps
        active_and_H = sum(
            _overlap(a0, a1, h0, h1)
            for a0, a1 in active_ints
            for h0, h1 in self._occ_ints
        )
        active_and_not_H = sum((a1 - a0) for a0, a1 in active_ints) - active_and_H

        # Conditional probabilities
        p_e_given_h = active_and_H / secs_H
        p_e_given_not_h = active_and_not_H / secs_not_H

        self.prob_given_true = max(MIN_PROBABILITY, min(p_e_given_h, MAX_PROBABILITY))
        self.prob_given_false = max(
            MIN_PROBABILITY, min(p_e_given_not_h, MAX_PROBABILITY)
        )
        _LOGGER.debug(
            "%s likelihoods updated: P(E|H)=%.3f  P(E|¬H)=%.3f",
            self._eid,
            self.prob_given_true,
            self.prob_given_false,
        )

    # ------------------------------------------------------------------ #
    async def _async_get_active_intervals(self) -> list[tuple[float, float]]:
        """Return list of (start_ts, end_ts) where sensor was ON."""
        if not self._hass or not self._eid:
            return []
            
        from homeassistant.components.recorder.history import get_significant_states

        start = dt_util.utcnow() - timedelta(days=self._days)
        hist = get_significant_states(
            self._hass, start_time=start, entity_ids=[self._eid],
            minimal_response=False  # Ensure we get State objects
        )
        states = hist.get(self._eid, [])

        intervals: list[tuple[float, float]] = []
        t_on: float | None = None
        for s in states:
            ts = s.last_changed.timestamp()  # type: ignore[union-attr]
            if s.state == "on":  # type: ignore[union-attr]
                if t_on is None:
                    t_on = ts
            elif t_on is not None:
                intervals.append((t_on, ts))
                t_on = None

        # If still ON at end of window
        if t_on:
            intervals.append((t_on, dt_util.utcnow().timestamp()))
        return intervals

    # ------------------------------------------------------------------ #
    async def update(
        self, 
        coordinator: "AreaOccupancyCoordinator", 
        entity: "Entity", 
        history_period: int | None = None
    ) -> None:
        """Update likelihood probabilities for this entity."""
        # For now, this is a placeholder that updates the timestamp
        # Full likelihood learning would be implemented here
        self.last_updated = dt_util.utcnow()
        _LOGGER.debug(
            "Updated likelihood for entity %s (placeholder implementation)",
            entity.entity_id,
        )

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Convert likelihood to dictionary for storage."""
        return {
            "prob_given_true": self.prob_given_true,
            "prob_given_false": self.prob_given_false,
            "last_updated": self.last_updated.isoformat(),
        }

    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Likelihood":
        """Create likelihood from dictionary."""
        return cls(
            prob_given_true=data["prob_given_true"],
            prob_given_false=data["prob_given_false"],
            last_updated=datetime.fromisoformat(data["last_updated"]),
        )
