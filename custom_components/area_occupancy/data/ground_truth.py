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

Pure functions only, shared by the database path (``db.queries``) and the
live labeller, so both always apply the same rule.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
import statistics

# A motion sensor whose median "on" stretch is shorter than this is
# PIR-like and gets the area's motion timeout when learning.
PULSE_MAX_MEDIAN_SECONDS = 120.0
# Activations needed before a sensor's history decides; fewer than this and
# it is treated as PIR-like.
PULSE_MIN_ACTIVATIONS = 10


def pulse_like_sensors(durations: Mapping[str, Iterable[float]]) -> set[str]:
    """The motion sensors whose activations look like PIR pulses.

    Args:
        durations: entity_id -> lengths in seconds of its "on" stretches.

    Returns:
        The entity ids to extend by the motion timeout.
    """
    pulse: set[str] = set()
    for entity_id, values in durations.items():
        values = [v for v in values if v > 0]
        if (
            len(values) < PULSE_MIN_ACTIVATIONS
            or statistics.median(values) < PULSE_MAX_MEDIAN_SECONDS
        ):
            pulse.add(entity_id)
    return pulse


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
