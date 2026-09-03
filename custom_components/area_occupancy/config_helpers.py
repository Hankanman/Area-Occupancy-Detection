"""Pure configuration helpers shared by every writer of area configuration.

Nothing in this module touches ``hass``. The config flow, the options flow,
the threshold ``number`` entity and any future writer (a websocket API, a
service call) validate and transform area configuration through these
functions so that the rules live in exactly one place.

Validation functions return translation keys (the ``strings.json`` error
ids), never user-facing text, so callers can surface them in a form or turn
them into an exception as they see fit.
"""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from .const import (
    CONF_ADJACENT_AREAS,
    CONF_APPLIANCE_ACTIVE_STATES,
    CONF_APPLIANCES,
    CONF_AREA_ID,
    CONF_COVER_ACTIVE_STATES,
    CONF_COVER_SENSORS,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_DOOR_ACTIVE_STATE,
    CONF_DOOR_SENSORS,
    CONF_LOCK_ACTIVE_STATE,
    CONF_LOCK_SENSORS,
    CONF_MEDIA_ACTIVE_STATES,
    CONF_MEDIA_DEVICES,
    CONF_MOTION_PROB_GIVEN_FALSE,
    CONF_MOTION_PROB_GIVEN_TRUE,
    CONF_MOTION_SENSORS,
    CONF_PERSON_CONFIDENCE_THRESHOLD,
    CONF_PERSON_DEVICE_TRACKER,
    CONF_PERSON_ENTITY,
    CONF_PERSON_SLEEP_AREA,
    CONF_PERSON_SLEEP_SENSORS,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_COVER,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LOCK,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_POWER,
    CONF_WEIGHT_WIFI_CLIENTS,
    CONF_WEIGHT_WINDOW,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_APPLIANCE_ACTIVE_STATES,
    DEFAULT_COVER_ACTIVE_STATES,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DEFAULT_DOOR_ACTIVE_STATE,
    DEFAULT_LOCK_ACTIVE_STATE,
    DEFAULT_MEDIA_ACTIVE_STATES,
    DEFAULT_MOTION_PROB_GIVEN_FALSE,
    DEFAULT_MOTION_PROB_GIVEN_TRUE,
    DEFAULT_PURPOSE,
    DEFAULT_SLEEP_CONFIDENCE_THRESHOLD,
    DEFAULT_WEIGHT_APPLIANCE,
    DEFAULT_WEIGHT_COVER,
    DEFAULT_WEIGHT_DOOR,
    DEFAULT_WEIGHT_ENVIRONMENTAL,
    DEFAULT_WEIGHT_LOCK,
    DEFAULT_WEIGHT_MEDIA,
    DEFAULT_WEIGHT_MOTION,
    DEFAULT_WEIGHT_POWER,
    DEFAULT_WEIGHT_WIFI_CLIENTS,
    DEFAULT_WEIGHT_WINDOW,
    DEFAULT_WINDOW_ACTIVE_STATE,
)
from .data.purpose import Purpose

# ── Shared bounds ────────────────────────────────────────────────────
# One definition for the UI selectors, the server-side validator and the
# threshold number entity. Change them here, nowhere else.
WEIGHT_STEP = 0.05
WEIGHT_MIN = 0
WEIGHT_MAX = 1

THRESHOLD_STEP = 1
THRESHOLD_MIN = 1
THRESHOLD_MAX = 100

# Explicit decay half-life bounds in seconds. ``0`` is the sentinel for "use
# the purpose default" and is always accepted (see ``apply_purpose_based_
# decay_default``).
DECAY_HALF_LIFE_MIN = 10
DECAY_HALF_LIFE_MAX = 3600

# Every per-type weight key with its default, in one place so the validator
# and any future per-type UI iterate the same list.
WEIGHT_KEYS: tuple[tuple[str, float], ...] = (
    (CONF_WEIGHT_MOTION, DEFAULT_WEIGHT_MOTION),
    (CONF_WEIGHT_MEDIA, DEFAULT_WEIGHT_MEDIA),
    (CONF_WEIGHT_APPLIANCE, DEFAULT_WEIGHT_APPLIANCE),
    (CONF_WEIGHT_DOOR, DEFAULT_WEIGHT_DOOR),
    (CONF_WEIGHT_LOCK, DEFAULT_WEIGHT_LOCK),
    (CONF_WEIGHT_WINDOW, DEFAULT_WEIGHT_WINDOW),
    (CONF_WEIGHT_COVER, DEFAULT_WEIGHT_COVER),
    (CONF_WEIGHT_ENVIRONMENTAL, DEFAULT_WEIGHT_ENVIRONMENTAL),
    (CONF_WEIGHT_POWER, DEFAULT_WEIGHT_POWER),
    (CONF_WEIGHT_WIFI_CLIENTS, DEFAULT_WEIGHT_WIFI_CLIENTS),
)

# (entities key, active-state key, default active state, error key) for every
# sensor channel whose entities need an active-state selection.
_STATE_REQUIREMENTS: tuple[tuple[str, str, Any, str], ...] = (
    (
        CONF_MEDIA_DEVICES,
        CONF_MEDIA_ACTIVE_STATES,
        DEFAULT_MEDIA_ACTIVE_STATES,
        "media_states_required",
    ),
    (
        CONF_APPLIANCES,
        CONF_APPLIANCE_ACTIVE_STATES,
        DEFAULT_APPLIANCE_ACTIVE_STATES,
        "appliance_states_required",
    ),
    (
        CONF_DOOR_SENSORS,
        CONF_DOOR_ACTIVE_STATE,
        DEFAULT_DOOR_ACTIVE_STATE,
        "door_state_required",
    ),
    (
        CONF_LOCK_SENSORS,
        CONF_LOCK_ACTIVE_STATE,
        DEFAULT_LOCK_ACTIVE_STATE,
        "lock_state_required",
    ),
    (
        CONF_WINDOW_SENSORS,
        CONF_WINDOW_ACTIVE_STATE,
        DEFAULT_WINDOW_ACTIVE_STATE,
        "window_state_required",
    ),
    (
        CONF_COVER_SENSORS,
        CONF_COVER_ACTIVE_STATES,
        DEFAULT_COVER_ACTIVE_STATES,
        "cover_states_required",
    ),
)


# ── Duration conversion ──────────────────────────────────────────────


def seconds_to_duration(seconds: float) -> dict[str, int]:
    """Convert seconds to the duration dict a ``DurationSelector`` expects.

    Args:
        seconds: Duration in seconds

    Returns:
        Dictionary with days, hours, minutes, seconds keys.
    """
    total = int(seconds)
    return {
        "days": total // 86400,
        "hours": (total % 86400) // 3600,
        "minutes": (total % 3600) // 60,
        "seconds": total % 60,
    }


def duration_to_seconds(duration: dict[str, int] | float) -> int:
    """Convert a duration dict or raw number to seconds.

    Args:
        duration: Duration dict with days/hours/minutes/seconds keys, or raw seconds.

    Returns:
        Total seconds as integer
    """
    if isinstance(duration, (int, float)):
        return int(duration)
    return (
        duration.get("days", 0) * 86400
        + duration.get("hours", 0) * 3600
        + duration.get("minutes", 0) * 60
        + duration.get("seconds", 0)
    )


# ── Field-level validation ───────────────────────────────────────────


def validate_threshold(value: Any) -> str | None:
    """Validate an occupancy threshold expressed as a percentage.

    Returns:
        ``None`` when valid, otherwise the ``strings.json`` error key.
    """
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or value < THRESHOLD_MIN
        or value > THRESHOLD_MAX
    ):
        return "invalid_threshold"
    return None


def validate_decay_half_life(value: Any) -> str | None:
    """Validate a decay half-life in seconds.

    ``0`` means "use the purpose default" and is always valid; any other
    value must fall inside ``[DECAY_HALF_LIFE_MIN, DECAY_HALF_LIFE_MAX]``.

    Returns:
        ``None`` when valid, otherwise the ``strings.json`` error key.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "invalid_decay_half_life"
    if value != 0 and (value < DECAY_HALF_LIFE_MIN or value > DECAY_HALF_LIFE_MAX):
        return "invalid_decay_half_life"
    return None


def validate_area_config(data: dict[str, Any]) -> dict[str, str]:
    """Validate a flat area configuration dict.

    Covers everything that can be checked without Home Assistant: purpose,
    motion sensors and likelihoods, threshold, per-channel active states,
    weights and decay. Existence of the HA area itself needs the registry
    and is checked by the caller (see ``BaseOccupancyFlow._validate_config``).

    Args:
        data: Flat (un-sectioned) area configuration.

    Returns:
        Mapping of field key (or ``"base"``) to ``strings.json`` error key.
        Empty when the configuration is valid.
    """
    errors: dict[str, str] = {}

    if not data.get(CONF_AREA_ID):
        errors[CONF_AREA_ID] = "area_required"

    if not data.get(CONF_PURPOSE, DEFAULT_PURPOSE):
        errors[CONF_PURPOSE] = "purpose_required"

    if not data.get(CONF_MOTION_SENSORS, []):
        errors.setdefault("base", "motion_required")

    prob_given_true = data.get(
        CONF_MOTION_PROB_GIVEN_TRUE, DEFAULT_MOTION_PROB_GIVEN_TRUE
    )
    prob_given_false = data.get(
        CONF_MOTION_PROB_GIVEN_FALSE, DEFAULT_MOTION_PROB_GIVEN_FALSE
    )
    if prob_given_true <= prob_given_false:
        errors.setdefault("base", "prob_true_must_exceed_false")

    threshold = data.get(CONF_THRESHOLD)
    if threshold is not None and (error := validate_threshold(threshold)):
        errors[CONF_THRESHOLD] = error

    errors.update(
        {
            entities_key: error_key
            for entities_key, state_key, default_state, error_key in _STATE_REQUIREMENTS
            if data.get(entities_key, []) and not data.get(state_key, default_state)
        }
    )

    for key, default in WEIGHT_KEYS:
        if not WEIGHT_MIN <= data.get(key, default) <= WEIGHT_MAX:
            errors[key] = "invalid_weight"
            break

    if data.get(CONF_DECAY_ENABLED, DEFAULT_DECAY_ENABLED):
        half_life = data.get(CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE)
        if error := validate_decay_half_life(half_life):
            errors[CONF_DECAY_HALF_LIFE] = error

    return errors


def validate_person_input(user_input: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize person configuration input.

    Args:
        user_input: Raw user input from the person config form

    Returns:
        Validated person data dict

    Raises:
        vol.Invalid: If required fields are missing or empty
    """
    person_entity = user_input.get(CONF_PERSON_ENTITY, "")
    sleep_sensors = user_input.get(CONF_PERSON_SLEEP_SENSORS, [])
    sleep_area = user_input.get(CONF_PERSON_SLEEP_AREA, "")

    if not person_entity:
        raise vol.Invalid("person_entity_required")
    if not sleep_sensors:
        raise vol.Invalid("sleep_sensor_required")
    if not sleep_area:
        raise vol.Invalid("sleep_area_required")

    raw_threshold = user_input.get(
        CONF_PERSON_CONFIDENCE_THRESHOLD, DEFAULT_SLEEP_CONFIDENCE_THRESHOLD
    )
    try:
        threshold = int(raw_threshold)
    except (ValueError, TypeError) as err:
        raise vol.Invalid("confidence_not_number") from err
    threshold = max(1, min(100, threshold))

    result = {
        CONF_PERSON_ENTITY: person_entity,
        CONF_PERSON_SLEEP_SENSORS: sleep_sensors,
        CONF_PERSON_SLEEP_AREA: sleep_area,
        CONF_PERSON_CONFIDENCE_THRESHOLD: threshold,
    }

    device_tracker = user_input.get(CONF_PERSON_DEVICE_TRACKER, "")
    if device_tracker:
        result[CONF_PERSON_DEVICE_TRACKER] = device_tracker

    return result


# ── Normalisation ────────────────────────────────────────────────────


def apply_purpose_based_decay_default(
    flattened_input: dict[str, Any], purpose: str | None
) -> None:
    """Normalise the decay half-life against the selected purpose.

    If decay half-life is not set or matches the *selected* purpose's default,
    store ``0`` ("use purpose value") so the area follows the purpose across
    later purpose changes. Any other custom value is preserved (#439/#440).
    Modifies ``flattened_input`` in place.

    Args:
        flattened_input: Flattened configuration dictionary
        purpose: Selected purpose value
    """
    if not purpose:
        return

    user_set_decay = flattened_input.get(CONF_DECAY_HALF_LIFE)
    if user_set_decay is None or Purpose.is_purpose_half_life(user_set_decay, purpose):
        flattened_input[CONF_DECAY_HALF_LIFE] = 0


# ── Area list transforms ─────────────────────────────────────────────


def find_area_by_id(areas: list[dict[str, Any]], area_id: str) -> dict[str, Any] | None:
    """Find an area by ID in a list of areas.

    Args:
        areas: List of area configuration dictionaries
        area_id: Area ID to find

    Returns:
        Area configuration dictionary if found, None otherwise
    """
    for area in areas:
        if area.get(CONF_AREA_ID) == area_id:
            return area
    return None


def normalize_adjacent_areas(value: Any) -> list[str]:
    """Coerce a ``CONF_ADJACENT_AREAS`` value to a clean list of area_id strings.

    The persistence layer normally writes a list, but config storage is JSON
    and a hand-edited file (or an old import) can supply other shapes. The
    mirror/strip helpers do set ops over the values, so a stray string would
    be iterated character-by-character and silently corrupt the data. This
    helper folds every shape into a ``list[str]``:

    - ``None`` → ``[]``
    - empty string → ``[]``
    - non-empty string → ``[value]`` (treated as a single area_id, not a
      sequence of characters)
    - list / tuple / set → list of non-empty stringified items (drops falsy
      entries like ``""`` or ``None``)
    - anything else → ``[str(value)]`` (best-effort preservation; the helper
      never raises, so an unexpected scalar is kept as a single id rather
      than silently dropped)
    """
    if value is None:
        return []
    if isinstance(value, str):
        # A bare string is a single area_id, not a sequence to iterate.
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v]
    # Best-effort: keep the value as a single entry rather than crashing.
    return [str(value)]


def apply_symmetric_adjacency(
    areas: list[dict[str, Any]], updated_area: dict[str, Any]
) -> list[dict[str, Any]]:
    """Mirror an area's adjacency edits across the paired areas.

    The adjacency UI is per-area (a flat multi-select of neighbours), but
    the underlying relation is mutual. When the user saves area A with
    adjacents ``[B, C]``:

    * Add A to B's and C's adjacents (if not already there).
    * Remove A from any other area X that previously listed A but
      isn't in A's new list.

    Returns a new list; does not mutate inputs.
    """
    target_area_id = updated_area.get(CONF_AREA_ID)
    if not target_area_id:
        return areas

    target_adjacents = set(
        normalize_adjacent_areas(updated_area.get(CONF_ADJACENT_AREAS))
    )
    # Defensive: the UI excludes self from the multi-select, but a
    # hand-edited storage file or imported config could carry a stray
    # self-reference. Drop it before any set ops so downstream callers
    # never see an area listed as adjacent to itself.
    target_adjacents.discard(target_area_id)

    result: list[dict[str, Any]] = []
    sanitized_target_adjacents = sorted(target_adjacents)
    for area in areas:
        area_id = area.get(CONF_AREA_ID)
        # The target row was substituted in by the caller; rewrite its
        # adjacents field to the normalised+self-stripped value so any
        # malformed input (non-list, self-link) doesn't survive a save.
        if area_id == target_area_id:
            cleaned_target = dict(area)
            cleaned_target[CONF_ADJACENT_AREAS] = list(sanitized_target_adjacents)
            result.append(cleaned_target)
            continue
        if not area_id:
            result.append(area)
            continue

        current_adjacents = set(normalize_adjacent_areas(area.get(CONF_ADJACENT_AREAS)))
        # Same defensive guard for the partner row.
        current_adjacents.discard(area_id)

        if area_id in target_adjacents:
            new_adjacents = current_adjacents | {target_area_id}
        else:
            new_adjacents = current_adjacents - {target_area_id}

        if new_adjacents != current_adjacents:
            mirrored = dict(area)
            mirrored[CONF_ADJACENT_AREAS] = sorted(new_adjacents)
            result.append(mirrored)
        else:
            result.append(area)
    return result


def strip_adjacency_references(
    areas: list[dict[str, Any]], removed_area_id: str
) -> list[dict[str, Any]]:
    """Remove a deleted area_id from every other area's adjacents list."""
    if not removed_area_id:
        return areas
    result: list[dict[str, Any]] = []
    for area in areas:
        normalized = normalize_adjacent_areas(area.get(CONF_ADJACENT_AREAS))
        if removed_area_id in normalized:
            cleaned = dict(area)
            cleaned[CONF_ADJACENT_AREAS] = [
                a for a in normalized if a != removed_area_id
            ]
            result.append(cleaned)
        else:
            result.append(area)
    return result


def update_area_in_list(
    areas: list[dict[str, Any]],
    updated_area: dict[str, Any],
    area_id: str | None,
) -> list[dict[str, Any]]:
    """Update or add an area in a list of areas.

    After the update or add, mirrors any adjacency changes across the
    other areas (adjacency is mutual; the UI is per-area).

    Args:
        areas: List of area configuration dictionaries
        updated_area: Updated area configuration
        area_id: Area ID being updated (None for new area)

    Returns:
        Updated list of areas
    """
    updated_areas = []
    area_updated = False
    for area in areas:
        if area_id and area.get(CONF_AREA_ID) == area_id:
            updated_areas.append(updated_area)
            area_updated = True
        else:
            updated_areas.append(area)

    if not area_updated:
        updated_areas.append(updated_area)

    return apply_symmetric_adjacency(updated_areas, updated_area)


def remove_area_from_list(
    areas: list[dict[str, Any]], area_id: str
) -> list[dict[str, Any]]:
    """Remove an area from a list of areas.

    Also strips the removed area_id from every surviving area's
    adjacents list so we don't leave dangling references.

    Args:
        areas: List of area configuration dictionaries
        area_id: Area ID to remove

    Returns:
        Updated list of areas with specified area removed
    """
    surviving = [area for area in areas if area.get(CONF_AREA_ID) != area_id]
    return strip_adjacency_references(surviving, area_id)
