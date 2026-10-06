"""What counts as "occupied" when learning: the ground-truth rule.

Every learner (priors, likelihoods, transitions, the shadow metrics) is
trained against the same definition: an area is occupied while one of its
motion, media or sleep sensors is active, plus, for PIR-style motion
sensors only, the area's motion timeout after each activation.

The timeout exists for PIR sensors: they pulse "on" for a few seconds per
detection and go quiet while someone sits still, so without it a person
reading in a chair looks like a string of separate visits with empty gaps.
mmWave and other presence sensors hold "on" while someone is there, so
extending them would count time after the person left. Which kind a sensor
is comes from its own history: PIR "on" stretches are short (typically
10-90 s), presence sensors hold for minutes. A sensor without enough
history to judge is treated as PIR, the case the timeout was built for.

The rule is shared by the database path (``db.queries``, applied after the
fact) and :class:`LiveLabeler` (applied as each moment happens), so both
always give the same answer: the parity that lets the database be retired.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics

from ..time_utils import ensure_utc_datetime

# A motion sensor whose median "on" stretch is shorter than this is
# PIR-like and gets the area's motion timeout when learning.
PULSE_MAX_MEDIAN_SECONDS = 120.0
# Hysteresis around it: a sensor already classified keeps its class until
# its median leaves this band, so one near the line (a PIR with a ~110 s
# cooldown) can't flip week to week and swing its room's prior with it.
PULSE_BAND_LOW_SECONDS = 90.0
PULSE_BAND_HIGH_SECONDS = 150.0
# Activations needed before a sensor's history decides; fewer than this and
# it is treated as PIR-like.
PULSE_MIN_ACTIVATIONS = 10


def is_pulse_like(values: Iterable[float], previous: bool | None = None) -> bool:
    """Whether a motion sensor's activations look like PIR pulses.

    Args:
        values: Lengths in seconds of its "on" stretches.
        previous: Its current classification, if it has one; inside the
            hysteresis band it is kept.

    Returns:
        True to extend the sensor by the motion timeout.
    """
    values = [v for v in values if v > 0]
    if len(values) < PULSE_MIN_ACTIVATIONS:
        return True if previous is None else previous
    median = statistics.median(values)
    if previous is None:
        return median < PULSE_MAX_MEDIAN_SECONDS
    if median < PULSE_BAND_LOW_SECONDS:
        return True
    if median > PULSE_BAND_HIGH_SECONDS:
        return False
    return previous


def pulse_like_sensors(
    durations: Mapping[str, Iterable[float]],
    previous: Mapping[str, bool] | None = None,
) -> set[str]:
    """The motion sensors whose activations look like PIR pulses.

    Args:
        durations: entity_id -> lengths in seconds of its "on" stretches.
        previous: entity_id -> current classification, for hysteresis.

    Returns:
        The entity ids to extend by the motion timeout.
    """
    previous = previous or {}
    return {
        entity_id
        for entity_id, values in durations.items()
        if is_pulse_like(values, previous.get(entity_id))
    }


def extend_pulses(
    motion: Mapping[str, list[tuple[datetime, datetime]]],
    pulse_ids: set[str],
    timeout: timedelta,
) -> list[tuple[datetime, datetime]]:
    """Motion "on" stretches, PIR-like ones extended by ``timeout``.

    Args:
        motion: entity_id -> its "on" stretches.
        pulse_ids: Sensors to extend (see :func:`pulse_like_sensors`).
        timeout: The area's motion timeout.

    Returns:
        All the stretches (unmerged).
    """
    return [
        (start, end + timeout if entity_id in pulse_ids else end)
        for entity_id, spans in motion.items()
        for start, end in spans
    ]


# Recent activations kept per motion sensor for the PIR-or-presence call.
ACTIVATION_HISTORY = 50


@dataclass
class LabelerState:
    """Per-area live labeller state (persisted)."""

    on_since: dict[str, datetime] = field(default_factory=dict)
    last_off: dict[str, datetime] = field(default_factory=dict)
    durations: dict[str, list[float]] = field(default_factory=dict)
    # Current PIR-like classification per sensor (sticky, see the band).
    pulse: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for the HA storage helper (JSON-safe)."""
        return {
            "on_since": {k: v.isoformat() for k, v in self.on_since.items()},
            "last_off": {k: v.isoformat() for k, v in self.last_off.items()},
            "durations": {k: list(v) for k, v in self.durations.items()},
            "pulse": dict(self.pulse),
        }

    @classmethod
    def from_dict(cls, data: dict) -> LabelerState:
        """Restore from storage; malformed payloads fall back to empty."""
        try:
            return cls(
                on_since={
                    str(k): ensure_utc_datetime(datetime.fromisoformat(v))
                    for k, v in (data.get("on_since") or {}).items()
                },
                last_off={
                    str(k): ensure_utc_datetime(datetime.fromisoformat(v))
                    for k, v in (data.get("last_off") or {}).items()
                },
                durations={
                    str(k): [float(x) for x in v][-ACTIVATION_HISTORY:]
                    for k, v in (data.get("durations") or {}).items()
                },
                pulse={str(k): bool(v) for k, v in (data.get("pulse") or {}).items()},
            )
        except (AttributeError, TypeError, ValueError):
            return cls()


class LiveLabeler:
    """Labels each moment occupied or not, live, by the ground-truth rule.

    The same definition the database path builds after the fact: a
    ground-truth sensor active now, or a PIR-like motion sensor that went
    off less than the area's motion timeout ago. Because the timeout only
    ever extends forward from an activation, it can be applied as it
    happens, so learners need no stored history to know the label.
    """

    def __init__(self, state: LabelerState | None = None) -> None:
        """Initialize from persisted state (or empty)."""
        self.state = state or LabelerState()

    def observe(
        self,
        *,
        now: datetime,
        motion: Mapping[str, bool],
        other_present: bool,
        timeout_seconds: float,
    ) -> bool:
        """Record one tick and return whether the area is occupied.

        Args:
            now: The tick time.
            motion: entity_id -> whether that motion sensor is active.
            other_present: Whether a media or sleep sensor is active.
            timeout_seconds: The area's motion timeout.

        Returns:
            The ground-truth label for this moment.
        """
        now = ensure_utc_datetime(now)
        state = self.state
        for entity_id, active in motion.items():
            if active and entity_id not in state.on_since:
                state.on_since[entity_id] = now
            elif not active and entity_id in state.on_since:
                started = state.on_since.pop(entity_id)
                history = state.durations.setdefault(entity_id, [])
                history.append((now - started).total_seconds())
                del history[:-ACTIVATION_HISTORY]
                state.last_off[entity_id] = now
                state.pulse[entity_id] = is_pulse_like(
                    history, state.pulse.get(entity_id)
                )
        if other_present or any(motion.values()):
            return True
        window = timedelta(seconds=timeout_seconds)
        return any(
            state.pulse.get(entity_id, True)
            and (off := state.last_off.get(entity_id)) is not None
            and now - off < window
            for entity_id in motion
        )
