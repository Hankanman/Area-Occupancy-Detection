"""House-wide context signals used by the occupancy coordinator."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from homeassistant.const import STATE_ON, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant

from ..const import GLOBAL_CONTEXT_BOOST

if TYPE_CHECKING:
    from .config import IntegrationConfig


@dataclass(frozen=True)
class GlobalSignal:
    """Snapshot of one optional house-wide entity."""

    role: str
    entity_id: str
    state: str
    value: float | None
    available: bool


@dataclass(frozen=True)
class GlobalContext:
    """Current house-wide context, with missing signals represented explicitly."""

    captured_at: datetime
    signals: dict[str, GlobalSignal]

    def is_active(self, role: str) -> bool:
        """Return whether a configured binary or numeric signal is active."""
        signal = self.signals.get(role)
        if signal is None or not signal.available:
            return False
        if signal.value is not None:
            return signal.value > 0
        return signal.state == STATE_ON

    def activity_boost(self, away: bool) -> float:
        """Return a small opt-in boost for explicit household activity signals."""
        if away:
            return 0.0
        return (
            GLOBAL_CONTEXT_BOOST
            if self.is_active("guest_mode") or self.is_active("activity")
            else 0.0
        )


class GlobalContextProvider:
    """Read optional house-wide entities without making them required."""

    def __init__(self, hass: HomeAssistant, config: IntegrationConfig) -> None:
        """Initialize the provider."""
        self.hass = hass
        self.config = config

    def snapshot(self, captured_at: datetime) -> GlobalContext:
        """Return the current state of all configured context entities."""
        signals: dict[str, GlobalSignal] = {}
        configured = self.config.global_context_entities
        for role, entity_id in configured.items():
            state = self.hass.states.get(entity_id)
            if state is None:
                signals[role] = GlobalSignal(
                    role, entity_id, STATE_UNKNOWN, None, False
                )
                continue
            available = state.state not in (STATE_UNKNOWN, STATE_UNAVAILABLE)
            value = None
            if available:
                with contextlib.suppress(TypeError, ValueError):
                    value = float(state.state)
            signals[role] = GlobalSignal(role, entity_id, state.state, value, available)
        return GlobalContext(captured_at, signals)
