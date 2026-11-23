"""Conservative closed-door occupancy hold (#558).

A closed-door hold keeps the probability of a room at an occupied floor when
all configured doors are known to be closed. It exists for rooms with a single
way in, especially bathrooms, where a PIR or motion sensor can lose a person
who remains stationary after a door has shut.

The implementation deliberately does not treat a closed door, a door-closing
event, or suspected arrival as occupancy evidence. A hold may be activated
only by one of these explicit conditions:

* A trusted presence sensor is active, for example an mmWave sensor with
  Home Assistant's ``presence`` device class.
* A button explicitly assigned to this room is pressed.
* A reliable motion sensor remains continuously active for at least
  ``long_motion_timeout`` seconds.

The motion timeout is independent of the area's normal motion timeout. It
includes the PIR hardware off-delay plus a configurable tolerance. The default
PIR tolerance is 20 seconds, and motion-only arming can be disabled by setting
``long_motion_timeout`` to zero.

Other sensor types, including media players, appliances, power sensors,
Wi-Fi client sensors, custom binary sensors, and custom numeric sensors, never
confirm the hold. A motion sensor does not become trusted merely because it
reports a normal motion state: only an explicitly trusted presence sensor or
the continuous-motion rule may activate it. Stuck, stale, unknown,
unavailable, missing, or otherwise unreliable sensor states do not activate a
new hold.

Opening any configured door immediately releases the active hold, clears an
existing fade, and resets any pending continuous-motion timer. Unknown door
states also stop long-motion detection, while an existing hold is allowed to
continue only until its configured ``max_hold`` expires. A value of zero for
``max_hold`` disables automatic expiry. Once expiry occurs, the probability
floor fades toward the area's prior over its configured decay half-life and is
fully removed after six half-lives.

The hold is a probability floor, not ground-truth evidence. It is therefore
not stored as motion evidence, and it does not feed learned occupancy intervals
or sensor correlation. Hold state is persisted separately so an integration
restart can restore an active hold or its fade. Invalid timestamps, malformed
payloads, and unsupported legacy states fall back to an empty or conservative
state rather than creating a false hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..const import GROUND_TRUTH_ACTIVE_FLOOR
from ..time_utils import ensure_utc_datetime
from ..utils import clamp_probability, logit

HOLD_THRESHOLD_MARGIN = 0.05
FADE_HALF_LIVES = 6
MIN_LONG_MOTION_TIMEOUT = 1.0
MIN_FADE_HALF_LIFE = 1.0


def hold_logit(threshold: float) -> float:
    """Return the minimum logit for an actively held area."""

    return max(
        logit(GROUND_TRUTH_ACTIVE_FLOOR),
        logit(clamp_probability(threshold)) + HOLD_THRESHOLD_MARGIN,
    )


@dataclass
class DoorHoldState:
    """Persistent state for one area's closed-door hold."""

    last_evidence_at: datetime | None = None
    fade_from: datetime | None = None
    doors_open: bool | None = None
    motion_on_previous: bool | None = None
    closed_motion_started_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state into a JSON-safe dictionary."""

        return {
            "last_evidence_at": (
                self.last_evidence_at.isoformat()
                if self.last_evidence_at is not None
                else None
            ),
            "fade_from": (
                self.fade_from.isoformat() if self.fade_from is not None else None
            ),
            "doors_open": self.doors_open,
            "motion_on_previous": self.motion_on_previous,
            "closed_motion_started_at": (
                self.closed_motion_started_at.isoformat()
                if self.closed_motion_started_at is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DoorHoldState:
        """Restore state, migrating legacy held_from payloads defensively."""

        if not isinstance(data, dict):
            return cls()

        def timestamp(key: str) -> datetime | None:
            raw = data.get(key)
            if raw is None:
                return None
            if not isinstance(raw, str):
                raise TypeError(f"{key} must be an ISO datetime string")
            return ensure_utc_datetime(datetime.fromisoformat(raw))

        try:
            doors_open = data.get("doors_open")
            motion_on_previous = data.get("motion_on_previous")
            if doors_open not in (True, False, None):
                doors_open = None
            if motion_on_previous not in (True, False, None):
                motion_on_previous = None

            last_evidence_at = timestamp("last_evidence_at")

            return cls(
                last_evidence_at=last_evidence_at,
                fade_from=timestamp("fade_from"),
                doors_open=doors_open,
                motion_on_previous=motion_on_previous,
                closed_motion_started_at=timestamp("closed_motion_started_at"),
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return cls()


class DoorHold:
    """Conservative closed-door hold for one area."""

    def __init__(self, state: DoorHoldState | None = None) -> None:
        """Initialize from persisted state or an empty state."""

        self.state = state or DoorHoldState()

    @property
    def held(self) -> bool:
        """Return whether the active hold is enabled."""

        return self.state.last_evidence_at is not None

    def release(self) -> None:
        """Release the hold, fade, and pending long-motion detection."""

        self.state.last_evidence_at = None
        self.state.fade_from = None
        self.state.closed_motion_started_at = None

    def observe(
        self,
        now: datetime,
        *,
        doors_open: bool | None,
        motion_on: bool | None,
        long_motion_timeout: float,
        max_hold: float,
        fade_half_life: float,
        trusted_presence_on: bool = False,
        occupancy_button_pressed: bool = False,
    ) -> None:
        """Update the hold from reliable aggregate sensor observations."""

        try:
            now = ensure_utc_datetime(now)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return

        state = self.state
        try:
            long_motion_timeout = max(
                float(long_motion_timeout), MIN_LONG_MOTION_TIMEOUT
            )
        except (TypeError, ValueError, OverflowError):
            long_motion_timeout = MIN_LONG_MOTION_TIMEOUT
        try:
            max_hold = max(float(max_hold), 0.0)
        except (TypeError, ValueError, OverflowError):
            max_hold = 0.0
        try:
            fade_half_life = max(float(fade_half_life), MIN_FADE_HALF_LIFE)
        except (TypeError, ValueError, OverflowError):
            fade_half_life = MIN_FADE_HALF_LIFE

        state.doors_open = doors_open
        state.motion_on_previous = motion_on

        if doors_open is True:
            self.release()
            return

        if doors_open is None:
            state.closed_motion_started_at = None
            self._expire_if_needed(
                now=now,
                max_hold=max_hold,
                fade_half_life=fade_half_life,
            )
            return

        if motion_on is True:
            if state.closed_motion_started_at is None:
                state.closed_motion_started_at = now
            motion_duration = (now - state.closed_motion_started_at).total_seconds()
            long_motion_confirmed = motion_duration >= long_motion_timeout
        else:
            state.closed_motion_started_at = None
            long_motion_confirmed = False

        if (
            trusted_presence_on is True
            or occupancy_button_pressed is True
            or long_motion_confirmed
        ):
            state.last_evidence_at = now
            state.fade_from = None

        self._expire_if_needed(
            now=now,
            max_hold=max_hold,
            fade_half_life=fade_half_life,
        )

    def _expire_if_needed(
        self,
        *,
        now: datetime,
        max_hold: float,
        fade_half_life: float,
    ) -> None:
        """Expire an active hold and remove a completed fade."""

        state = self.state
        if state.last_evidence_at is not None and max_hold > 0:
            elapsed = (now - state.last_evidence_at).total_seconds()
            if elapsed >= max_hold:
                state.fade_from = state.last_evidence_at + timedelta(seconds=max_hold)
                state.last_evidence_at = None

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
        """Return the active or fading probability floor as a logit."""

        state = self.state
        target = hold_logit(threshold)

        if state.last_evidence_at is not None:
            return target
        if state.fade_from is None:
            return None

        try:
            fade_half_life = max(float(fade_half_life), MIN_FADE_HALF_LIFE)
            now = ensure_utc_datetime(now)
            elapsed = max(
                (now - state.fade_from).total_seconds(),
                0.0,
            )
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None

        if target <= bias:
            return None

        return bias + (target - bias) * (0.5 ** (elapsed / fade_half_life))
