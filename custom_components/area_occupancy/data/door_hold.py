"""Closed-door hold: a room with one way in stays occupied while it is shut (#558).

Motion sensors lose people who keep still: someone in the bath or shower
looks like an empty room once the PIR stops seeing them. If the room has
one way in and its door has stayed shut since someone was seen inside,
they can't have left. This is what Wasp in Box did, as a virtual motion
sensor; here it is a floor on the area's probability instead, so it isn't
learned from as if it were real motion and isn't capped by a weight.

A closed door says nothing about *who* is behind it, so the hold only arms
on evidence that someone is there, and prefers a lost hold to a false one:

* **Arm** when the door closes after motion started *while it was open*
  (someone came in and shut it behind them: the PIR only sees them once they
  are inside), or while a stronger current signal confirms someone is in the
  room (an mmWave presence sensor, media, an appliance, power draw...). Motion
  while the door is already shut arms it as well.
* **Do not arm** when the PIR was already on before the door opened and is
  merely still on after it closes: that is as likely to be someone leaving,
  trailing off over the sensor's off-delay. It arms later only if the PIR
  clears and then fires again (someone is moving about inside) or a stronger
  signal confirms.
* **Release** as soon as any door opens.
* **Expire** ``max_hold`` after the last motion seen behind the closed
  door (0: never, as Wasp in Box's maximum duration); the floor then fades
  towards the prior over the area's half-life instead of snapping off.

``motion_window`` is the area's motion timeout: how recent the motion that
started while the door was open must be when the door closes.

While held the probability is floored at ``GROUND_TRUTH_ACTIVE_FLOOR``
(75%, the motion floor), or just above the area's threshold if that is
higher, so a held room always reads as occupied.

Whether a room has one way in is a fact about the house, so it is a
per-area setting (``closed_door_hold``), not something learned.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from typing import Any

from ..const import DEFAULT_WASP_MAX_DURATION, GROUND_TRUTH_ACTIVE_FLOOR
from ..time_utils import ensure_utc_datetime
from ..utils import clamp_probability, logit
from .entity_type import InputType

# Input types that, while active, say more about someone being in the room
# right now than a PIR that has gone quiet: they can confirm a hold on their
# own. Environmental readings (humidity, temperature, CO2...) are left out:
# they lag, so they can't tell someone leaving a steamy room from someone
# staying in it. An mmWave presence sensor is a motion-type entity with the
# ``presence`` device class and is added alongside these by the coordinator.
DOOR_HOLD_CONFIRMING_TYPES: frozenset[InputType] = frozenset(
    {
        InputType.MEDIA,
        InputType.APPLIANCE,
        InputType.POWER,
        InputType.WIFI_CLIENTS,
        InputType.BUTTON,
        InputType.CUSTOM_BINARY,
        InputType.CUSTOM_NUMERIC,
    }
)
# Logits above the threshold the hold sits at when the threshold is above
# the 75% floor (so floating point can't land it a hair under).
HOLD_THRESHOLD_MARGIN = 0.05
# Half-lives after which a fading hold is dropped (2^-6: under 2% left).
FADE_HALF_LIVES = 6


def _safe_half_life(value: float) -> float:
    """The fade half-life, at least a second; a non-finite value fades fast."""
    return max(value, 1.0) if math.isfinite(value) else 1.0


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
    # The doors' state at the previous observation (None: not seen yet).
    doors_open: bool | None = None
    # Whether motion was on at the previous observation (for rising edges).
    motion_on_previous: bool | None = None
    # When motion last started while a door was open (cleared on opening and
    # on closing): someone arriving through the door.
    motion_rising_at: datetime | None = None
    # After a door closed with the PIR already on and no arrival seen:
    # ``"trailing"`` while it is still on, ``"cleared"`` once it has gone off
    # (fresh motion then arms). None otherwise.
    awaiting: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "held_from": self.held_from.isoformat() if self.held_from else None,
            "fade_from": self.fade_from.isoformat() if self.fade_from else None,
            "doors_open": self.doors_open,
            "motion_on_previous": self.motion_on_previous,
            "motion_rising_at": (
                self.motion_rising_at.isoformat() if self.motion_rising_at else None
            ),
            "awaiting": self.awaiting,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DoorHoldState:
        """Restore from storage; malformed payloads fall back to empty."""

        def _ts(key: str) -> datetime | None:
            raw = data.get(key)
            return ensure_utc_datetime(datetime.fromisoformat(raw)) if raw else None

        try:
            doors_open = data.get("doors_open")
            motion_on = data.get("motion_on_previous")
            awaiting = data.get("awaiting")
            return cls(
                held_from=_ts("held_from"),
                fade_from=_ts("fade_from"),
                doors_open=None if doors_open is None else bool(doors_open),
                motion_on_previous=None if motion_on is None else bool(motion_on),
                motion_rising_at=_ts("motion_rising_at"),
                awaiting=awaiting if awaiting in ("trailing", "cleared") else None,
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
        doors_open: bool | None,
        motion_on: bool,
        motion_window: float,
        max_hold: float,
        fade_half_life: float,
        presence_on: bool = False,
    ) -> None:
        """Update the hold from the area's doors and motion at ``now``.

        Args:
            now: The observation time.
            doors_open: Whether any of the area's doors is open; None when
                no door is known to be open and at least one reads unknown
                or unavailable. An unreadable door arms nothing and releases
                nothing: a hold already on carries on until it expires.
            motion_on: Whether any of its motion sensors is active.
            motion_window: The area's motion timeout in seconds: motion that
                started while the door was open counts as an arrival if it
                began within this long of the door closing.
            max_hold: Seconds after the last motion the hold lasts; 0 for
                no limit (as Wasp in Box's maximum duration). A negative or
                non-finite value is replaced by the default, never read as
                "no limit".
            fade_half_life: Half-life of the fade after ``max_hold``.
            presence_on: Whether a stronger current signal than the PIR (an
                mmWave presence sensor, media, an appliance, power...)
                confirms someone is in the area.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        if not math.isfinite(motion_window):
            motion_window = 0.0
        if not math.isfinite(max_hold) or max_hold < 0:
            max_hold = float(DEFAULT_WASP_MAX_DURATION)
        fade_half_life = _safe_half_life(fade_half_life)
        # A timestamp after "now" (a restored state, or the clock moving back)
        # must not stretch a hold: count from now instead.
        for name in ("held_from", "fade_from", "motion_rising_at"):
            stamp = getattr(state, name)
            if stamp is not None and stamp > now:
                setattr(state, name, now)
        if doors_open is None:
            # Nothing known about the doors: keep what we have. The pending
            # arrival is dropped (we can't tell it from a departure now) but
            # ``awaiting`` stays, so a trailing PIR can't arm once they read
            # closed again.
            state.motion_on_previous = motion_on
            state.motion_rising_at = None
            self._expire(now, max_hold, fade_half_life)
            return
        just_closed = state.doors_open is True and not doors_open
        just_opened = state.doors_open is not True and doors_open
        motion_rising = motion_on and state.motion_on_previous is not True
        state.doors_open = doors_open
        state.motion_on_previous = motion_on
        if just_opened:
            state.motion_rising_at = None
            state.awaiting = None
        if doors_open:
            if motion_rising:
                state.motion_rising_at = now
            state.held_from = None
            state.fade_from = None
            return
        arm = False
        if just_closed:
            arrived = (
                state.motion_rising_at is not None
                and (now - state.motion_rising_at).total_seconds() <= motion_window
            )
            state.motion_rising_at = None
            if arrived or presence_on:
                arm = True
                state.awaiting = None
            elif motion_on:
                state.awaiting = "trailing"
        elif state.awaiting == "trailing":
            if presence_on:
                arm = True
            elif not motion_on:
                state.awaiting = "cleared"
        elif state.awaiting == "cleared":
            arm = presence_on or motion_rising
        elif motion_on:
            arm = True
        if arm:
            state.held_from = now
            state.fade_from = None
            state.awaiting = None
        self._expire(now, max_hold, fade_half_life)

    def _expire(self, now: datetime, max_hold: float, fade_half_life: float) -> None:
        """End a hold that ran past ``max_hold`` and drop a finished fade."""
        state = self.state
        if (
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
        return bias + (target - bias) * 0.5 ** (
            elapsed / _safe_half_life(fade_half_life)
        )
