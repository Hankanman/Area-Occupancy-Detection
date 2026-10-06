"""Closed-door hold: a room with one way in stays occupied while it is shut (#558).

Motion sensors lose people who keep still: someone in the bath or shower
looks like an empty room once the PIR stops seeing them. If the room has
one way in and its door has stayed shut since someone was seen inside,
they can't have left. This is what Wasp in Box did, as a virtual motion
sensor; here it is a floor on the area's probability instead, so it isn't
learned from as if it were real motion and isn't capped by a weight.

The rules are Wasp in Box's, so areas migrated from it behave as before:

* **Arm** while every door is closed and motion is on, or when the doors
  close within the motion window of motion (someone went in and shut the
  door behind them).
* **Release** as soon as any door opens.
* **Expire** ``max_hold`` after the last motion seen behind the closed
  door (0: never, as Wasp in Box's maximum duration); the floor then fades towards the prior over the area's half-life
  instead of snapping off.

While held the probability is floored at ``GROUND_TRUTH_ACTIVE_FLOOR``
(75%, the motion floor), or just above the area's threshold if that is
higher, so a held room always reads as occupied.

Whether a room has one way in is a fact about the house, so it is a
per-area setting (``closed_door_hold``), not something learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..const import GROUND_TRUTH_ACTIVE_FLOOR
from ..time_utils import ensure_utc_datetime
from ..utils import clamp_probability, logit

# Logits above the threshold the hold sits at when the threshold is above
# the 75% floor (so floating point can't land it a hair under).
HOLD_THRESHOLD_MARGIN = 0.05
# Half-lives after which a fading hold is dropped (2^-6: under 2% left).
FADE_HALF_LIVES = 6


def hold_logit(threshold: float) -> float:
    """The logit a held room is floored at."""
    return max(
        logit(GROUND_TRUTH_ACTIVE_FLOOR),
        logit(clamp_probability(threshold)) + HOLD_THRESHOLD_MARGIN,
    )


@dataclass
class DoorHoldState:
    """One area's hold state (persisted, so a restart keeps a hold)."""

    # Set while held: the last motion seen behind the closed door (or the
    # arming time), which ``max_hold`` counts from.
    held_from: datetime | None = None
    # Set while fading after ``max_hold``: when the fade began.
    fade_from: datetime | None = None
    last_motion: datetime | None = None
    # The doors' state at the previous observation (None: not seen yet).
    doors_open: bool | None = None

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "held_from": self.held_from.isoformat() if self.held_from else None,
            "fade_from": self.fade_from.isoformat() if self.fade_from else None,
            "last_motion": self.last_motion.isoformat() if self.last_motion else None,
            "doors_open": self.doors_open,
        }

    @classmethod
    def from_dict(cls, data: dict) -> DoorHoldState:
        """Restore from storage; malformed payloads fall back to empty."""

        def _ts(key: str) -> datetime | None:
            raw = data.get(key)
            return ensure_utc_datetime(datetime.fromisoformat(raw)) if raw else None

        try:
            doors_open = data.get("doors_open")
            return cls(
                held_from=_ts("held_from"),
                fade_from=_ts("fade_from"),
                last_motion=_ts("last_motion"),
                doors_open=None if doors_open is None else bool(doors_open),
            )
        except (AttributeError, TypeError, ValueError):
            return cls()


class DoorHold:
    """The closed-door hold for one area."""

    def __init__(self, state: DoorHoldState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or DoorHoldState()

    @property
    def held(self) -> bool:
        """Whether the hold is on (not counting a fade)."""
        return self.state.held_from is not None

    def observe(
        self,
        now: datetime,
        *,
        doors_open: bool,
        motion_on: bool,
        motion_window: float,
        max_hold: float,
        fade_half_life: float,
    ) -> None:
        """Update the hold from the area's doors and motion at ``now``.

        Args:
            now: The observation time.
            doors_open: Whether any of the area's doors is open.
            motion_on: Whether any of its motion sensors is active.
            motion_window: Seconds after motion that closing the doors
                still arms the hold.
            max_hold: Seconds after the last motion the hold lasts; 0 for
                no limit (as Wasp in Box's maximum duration).
            fade_half_life: Half-life of the fade after ``max_hold``.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        just_closed = state.doors_open is True and not doors_open
        state.doors_open = doors_open
        if motion_on:
            state.last_motion = now
        if doors_open:
            state.held_from = None
            state.fade_from = None
            return
        recent_motion = (
            state.last_motion is not None
            and (now - state.last_motion).total_seconds() <= motion_window
        )
        if motion_on or (just_closed and recent_motion):
            state.held_from = now
            state.fade_from = None
        elif (
            state.held_from is not None
            and max_hold > 0
            and (now - state.held_from).total_seconds() >= max_hold
        ):
            # Fade from the moment the hold ran out, not from this tick.
            state.fade_from = state.held_from + timedelta(seconds=max_hold)
            state.held_from = None
        if (
            state.fade_from is not None
            and (now - state.fade_from).total_seconds()
            >= FADE_HALF_LIVES * fade_half_life
        ):
            state.fade_from = None

    def floor_logit(
        self,
        now: datetime,
        *,
        bias: float,
        threshold: float,
        fade_half_life: float,
    ) -> float | None:
        """The logit the area's probability is floored at, or None.

        Held: :func:`hold_logit`. Fading: that level decaying towards
        ``bias`` (the prior's logit) with ``fade_half_life``, as evidence
        decays. Otherwise None.
        """
        state = self.state
        target = hold_logit(threshold)
        if state.held_from is not None:
            return target
        if state.fade_from is None:
            return None
        elapsed = max((ensure_utc_datetime(now) - state.fade_from).total_seconds(), 0)
        if target <= bias:
            return None
        return bias + (target - bias) * 0.5 ** (elapsed / max(fade_half_life, 1.0))
