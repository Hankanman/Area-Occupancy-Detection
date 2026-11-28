"""Activity detection for Area Occupancy Detection.

Detects what activity is happening in an area based on sensor signals
and the area's configured purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import logging
import math
from typing import TYPE_CHECKING

from ..const import (
    ACTIVITY_BOOST_HIGH,
    ACTIVITY_BOOST_MILD,
    ACTIVITY_BOOST_MODERATE,
    ACTIVITY_BOOST_STRONG,
)
from .entity_type import InputType
from .purpose import AreaPurpose

if TYPE_CHECKING:
    from ..area.area import Area
    from .entity import Entity

_LOGGER = logging.getLogger(__name__)

# Confidence values are reported rounded to this many decimals. Candidates
# are also compared at this resolution so that float noise (e.g. 0.1 + 0.2
# vs. 0.3) cannot bypass the documented tie-breakers.
_CONFIDENCE_DECIMALS = 4

# Means closer than this are treated as indistinguishable.
_MIN_MEAN_SPAN = 1e-9

# Environmental readings are only used when the learned means are at least
# this many average standard deviations apart.
_MIN_SEPARATION_IN_STD = 0.5


class ActivityId(StrEnum):
    """Detectable activities."""

    BATHING = "bathing"
    COOKING = "cooking"
    EATING = "eating"
    IDLE = "idle"
    LISTENING_TO_MUSIC = "listening_to_music"
    SHOWERING = "showering"
    SLEEPING = "sleeping"
    UNOCCUPIED = "unoccupied"
    WATCHING_TV = "watching_tv"
    WORKING = "working"


class EnvironmentalCondition(StrEnum):
    """Direction an environmental reading must move to count as evidence.

    ``StrEnum`` members compare equal to their plain string values, so code
    or tests that still pass ``"elevated"`` / ``"suppressed"`` keep working.
    """

    ELEVATED = "elevated"
    SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class Indicator:
    """A single sensor signal that indicates an activity.

    Attributes:
        input_type: Sensor type this indicator reads.
        weight: Contribution to the activity's confidence at full strength.
        require_active: Kept for API compatibility. Binary indicators
            currently score an entity identically either way (active → 1.0,
            decaying → decay factor). Environmental indicators ignore it.
        environmental_condition: When set, the indicator is scored from the
            entity's numeric state against its learned Gaussian means
            instead of its binary evidence.
        ha_device_classes: Optional allow-list of Home Assistant device
            classes for binary indicators.
    """

    input_type: InputType
    weight: float
    require_active: bool = True
    environmental_condition: str | None = None
    ha_device_classes: frozenset[str] | None = None


@dataclass(frozen=True)
class ActivityDefinition:
    """An activity with its indicators and constraints."""

    activity_id: ActivityId
    indicators: tuple[Indicator, ...]
    min_match_weight: float = 0.3
    purposes: frozenset[AreaPurpose] = frozenset()
    occupancy_boost: float = 0.0


@dataclass
class DetectedActivity:
    """Result of activity detection."""

    activity_id: ActivityId
    confidence: float
    matching_indicators: list[str] = field(default_factory=list)
    occupancy_boost: float = 0.0


# --- Activity Definitions ---

_ELEVATED = EnvironmentalCondition.ELEVATED
_SUPPRESSED = EnvironmentalCondition.SUPPRESSED

ACTIVITY_DEFINITIONS: tuple[ActivityDefinition, ...] = (
    ActivityDefinition(
        activity_id=ActivityId.SHOWERING,
        indicators=(
            Indicator(
                InputType.HUMIDITY,
                0.5,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.TEMPERATURE,
                0.2,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(InputType.MOTION, 0.15),
            Indicator(InputType.DOOR, 0.15),
        ),
        purposes=frozenset({AreaPurpose.BATHROOM}),
        occupancy_boost=ACTIVITY_BOOST_HIGH,
    ),
    ActivityDefinition(
        activity_id=ActivityId.BATHING,
        indicators=(
            Indicator(
                InputType.HUMIDITY,
                0.4,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(InputType.DOOR, 0.3),
            Indicator(
                InputType.TEMPERATURE,
                0.2,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(InputType.MOTION, 0.1),
        ),
        min_match_weight=0.3,
        purposes=frozenset({AreaPurpose.BATHROOM}),
        occupancy_boost=ACTIVITY_BOOST_HIGH,
    ),
    ActivityDefinition(
        activity_id=ActivityId.COOKING,
        indicators=(
            Indicator(InputType.APPLIANCE, 0.35),
            Indicator(
                InputType.TEMPERATURE,
                0.2,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.HUMIDITY,
                0.15,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.CO2,
                0.1,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.VOC,
                0.1,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(InputType.MOTION, 0.1),
        ),
        purposes=frozenset({AreaPurpose.FOOD_PREP}),
        occupancy_boost=ACTIVITY_BOOST_MODERATE,
    ),
    ActivityDefinition(
        activity_id=ActivityId.WATCHING_TV,
        indicators=(
            Indicator(
                InputType.MEDIA,
                0.6,
                ha_device_classes=frozenset({"tv", "receiver"}),
            ),
            Indicator(
                InputType.ILLUMINANCE,
                0.15,
                require_active=False,
                environmental_condition=_SUPPRESSED,
            ),
            Indicator(InputType.MOTION, 0.1),
            Indicator(
                InputType.SOUND_PRESSURE,
                0.15,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
        ),
        purposes=frozenset(
            {AreaPurpose.SOCIAL, AreaPurpose.RELAXING, AreaPurpose.SLEEPING}
        ),
        occupancy_boost=ACTIVITY_BOOST_STRONG,
    ),
    ActivityDefinition(
        activity_id=ActivityId.LISTENING_TO_MUSIC,
        indicators=(
            Indicator(
                InputType.MEDIA,
                0.5,
                ha_device_classes=frozenset({"speaker", "receiver"}),
            ),
            Indicator(
                InputType.SOUND_PRESSURE,
                0.3,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(InputType.MOTION, 0.2),
        ),
        purposes=frozenset(
            {AreaPurpose.SOCIAL, AreaPurpose.RELAXING, AreaPurpose.WORKING}
        ),
        occupancy_boost=ACTIVITY_BOOST_MILD,
    ),
    ActivityDefinition(
        activity_id=ActivityId.WORKING,
        indicators=(
            Indicator(InputType.APPLIANCE, 0.4),
            Indicator(InputType.POWER, 0.25),
            Indicator(InputType.MOTION, 0.15),
            Indicator(
                InputType.CO2,
                0.1,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.ILLUMINANCE,
                0.1,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
        ),
        purposes=frozenset({AreaPurpose.WORKING}),
        occupancy_boost=ACTIVITY_BOOST_MODERATE,
    ),
    ActivityDefinition(
        activity_id=ActivityId.SLEEPING,
        indicators=(
            Indicator(InputType.SLEEP, 0.5),
            Indicator(
                InputType.ILLUMINANCE,
                0.2,
                require_active=False,
                environmental_condition=_SUPPRESSED,
            ),
            Indicator(
                InputType.CO2,
                0.15,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.SOUND_PRESSURE,
                0.15,
                require_active=False,
                environmental_condition=_SUPPRESSED,
            ),
        ),
        purposes=frozenset({AreaPurpose.SLEEPING}),
        occupancy_boost=ACTIVITY_BOOST_HIGH,
    ),
    ActivityDefinition(
        activity_id=ActivityId.EATING,
        indicators=(
            Indicator(InputType.MOTION, 0.3),
            Indicator(
                InputType.ILLUMINANCE,
                0.25,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.CO2,
                0.2,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(
                InputType.TEMPERATURE,
                0.15,
                require_active=False,
                environmental_condition=_ELEVATED,
            ),
            Indicator(InputType.MEDIA, 0.1),
        ),
        purposes=frozenset({AreaPurpose.EATING}),
        occupancy_boost=ACTIVITY_BOOST_MILD,
    ),
)


# --- Scoring helpers ---


def _environmental_signal_strength(
    value: float,
    mean_occupied: float,
    mean_unoccupied: float,
    condition: str | None,
) -> float:
    """Compute 0-1 signal strength for an environmental reading.

    Uses learned Gaussian means to determine how far the current value
    has moved from unoccupied toward occupied (or away, for suppressed).

    Returns 0.0 when means are too close to distinguish or when any input
    is not a finite number.
    """
    if not (
        math.isfinite(value)
        and math.isfinite(mean_occupied)
        and math.isfinite(mean_unoccupied)
    ):
        # max(0, min(1, nan)) would evaluate to 1.0, i.e. full strength.
        return 0.0

    span = mean_occupied - mean_unoccupied
    if abs(span) < _MIN_MEAN_SPAN:
        return 0.0

    # NOTE: "elevated" measures movement toward the occupied mean (sign
    # follows ``span``); "suppressed" measures a drop below the unoccupied
    # mean regardless of ``span``'s sign. Behaviour intentionally unchanged.
    if condition == EnvironmentalCondition.ELEVATED:
        position = (value - mean_unoccupied) / span
    elif condition == EnvironmentalCondition.SUPPRESSED:
        position = (mean_unoccupied - value) / abs(span)
    else:
        return 0.0

    return max(0.0, min(1.0, position))


def _update_best(
    best_strength: float,
    matched_ids: list[str],
    entity_id: str,
    strength: float,
) -> float:
    """Track the strongest match; equal non-zero strengths are all kept."""
    if strength > best_strength:
        matched_ids.clear()
        matched_ids.append(entity_id)
        return strength
    if strength == best_strength and strength > 0:
        matched_ids.append(entity_id)
    return best_strength


def _score_indicator(
    indicator: Indicator,
    area: Area,
    *,
    entity_cache: dict[InputType, dict[str, Entity]] | None = None,
) -> tuple[float, list[str]]:
    """Score a single indicator against the area's entities.

    Args:
        indicator: The indicator to score.
        area: The area whose entities are inspected.
        entity_cache: Optional per-call cache of entity lookups by input
            type, so repeated indicator types (e.g. MOTION) are only
            resolved once per detection run.

    Returns:
        Tuple of (weighted_score, list_of_matching_entity_ids).
        weighted_score is indicator.weight * match_strength.
        Returns (-1.0, []) if no sensors of the required type exist.
    """
    if entity_cache is None:
        entities = area.entities.get_entities_by_input_type(indicator.input_type)
    else:
        if indicator.input_type not in entity_cache:
            entity_cache[indicator.input_type] = (
                area.entities.get_entities_by_input_type(indicator.input_type)
            )
        entities = entity_cache[indicator.input_type]

    if not entities:
        # No sensor of this type in the area — exclude from scoring.
        return (-1.0, [])

    if indicator.environmental_condition is not None:
        return _score_environmental_indicator(indicator, entities)

    return _score_binary_indicator(indicator, area, entities)


def _effective_device_class(indicator: Indicator, entity: Entity) -> str | None:
    """Return the device class the entity is behaving as for this indicator.

    A "speaker"-class media_player that is actually relaying TV audio
    (see Entity.media_is_tv_source) is treated as "tv" for MEDIA
    indicators, so WATCHING_TV vs LISTENING_TO_MUSIC scoring reflects
    what's really happening rather than the device_class HA reports for
    the hardware.

    A media_player with no device class (Apple TV and some TVs set none) is
    treated as "tv" too. Matching neither activity left the choice to motion,
    which weighs more in LISTENING_TO_MUSIC, so a playing Apple TV read as
    music (#696). A speaker still needs ``device_class: speaker``.
    """
    device_class = entity.ha_device_class
    if indicator.input_type != InputType.MEDIA:
        return device_class
    if device_class is None or (
        device_class == "speaker" and entity.media_is_tv_source
    ):
        return "tv"
    return device_class


def _score_binary_indicator(
    indicator: Indicator,
    area: Area,
    entities: dict[str, Entity],
) -> tuple[float, list[str]]:
    """Score a binary (active/inactive) indicator.

    When all entities are filtered out by device class, this returns
    (0.0, []) — not (-1.0, []) — because the area *has* sensors of this
    type, they just aren't the right kind. With total-weight normalization
    the outcome is identical, but semantically "no matching device" ≠
    "no sensor."
    """
    best_strength = 0.0
    matched_ids: list[str] = []

    for entity_id, entity in entities.items():
        if (
            indicator.ha_device_classes is not None
            and _effective_device_class(indicator, entity)
            not in indicator.ha_device_classes
        ):
            continue

        if getattr(entity, "is_stuck", False) is True:
            continue  # Flagged stuck active: not evidence of an activity

        # Active → full strength; decaying → residual decay factor.
        # (Previously split by ``require_active`` with identical branches.)
        if entity.evidence is True:
            strength = 1.0
        elif entity.decay.is_decaying:
            strength = entity.decay_factor
        else:
            continue

        best_strength = _update_best(best_strength, matched_ids, entity_id, strength)

    return (indicator.weight * best_strength, matched_ids)


def _score_environmental_indicator(
    indicator: Indicator,
    entities: dict[str, Entity],
) -> tuple[float, list[str]]:
    """Score an environmental (Gaussian-based) indicator."""
    best_strength = 0.0
    matched_ids: list[str] = []

    for entity_id, entity in entities.items():
        params = getattr(entity, "learned_gaussian_params", None)
        if params is None:
            continue

        state = entity.state
        if state is None:
            continue

        try:
            val = float(state)
        except (ValueError, TypeError):
            continue  # e.g. "unknown" / "unavailable"
        if not math.isfinite(val):
            continue  # "nan" / "inf" states are not usable readings

        # Skip entities where means are not statistically distinguishable
        span = abs(params.mean_occupied - params.mean_unoccupied)
        avg_std = (params.std_occupied + params.std_unoccupied) / 2
        if avg_std > 0 and span < avg_std * _MIN_SEPARATION_IN_STD:
            continue

        strength = _environmental_signal_strength(
            val,
            params.mean_occupied,
            params.mean_unoccupied,
            indicator.environmental_condition,
        )

        best_strength = _update_best(best_strength, matched_ids, entity_id, strength)

    return (indicator.weight * best_strength, matched_ids)


# --- Detection ---


def detect_activity(
    area: Area,
    *,
    base_probability: float | None = None,
    is_occupied: bool | None = None,
) -> DetectedActivity:
    """Detect the most likely activity in an area.

    Args:
        area: The area to detect activity in.
        base_probability: If provided, use this instead of calling area.probability().
            Used by Area.probability() to break circular dependency.
        is_occupied: If provided, use this instead of calling area.occupied().
            Used by Area.probability() to break circular dependency.

    Algorithm:
    1. If unoccupied, return Unoccupied.
    2. Filter definitions to matching purposes (empty = any).
    3. Score each candidate's indicators.
    4. Normalize by total definition weight (sum of all indicator weights).
       Missing sensors naturally reduce the maximum achievable confidence.
    5. Discard below min_match_weight.
    6. Highest confidence (at reported precision) wins; ties broken by
       purpose-specificity, then by matched_weight (more actual evidence).
       If none match, return Idle.
    """
    prob = base_probability if base_probability is not None else area.probability()
    if is_occupied is not None:
        occupied = is_occupied
    elif base_probability is not None:
        occupied = prob >= area.config.threshold
    else:
        occupied = area.occupied()

    if not occupied:
        return DetectedActivity(
            activity_id=ActivityId.UNOCCUPIED,
            confidence=round(1.0 - prob, _CONFIDENCE_DECIMALS),
        )

    purpose = area.purpose.purpose
    entity_cache: dict[InputType, dict[str, Entity]] = {}

    best: DetectedActivity | None = None
    best_matched_weight = 0.0
    best_is_specific = False

    for defn in ACTIVITY_DEFINITIONS:
        # Filter by purpose (empty purposes = any).
        if defn.purposes and purpose not in defn.purposes:
            continue

        # Normalize by total definition weight (always ~1.0), not just
        # the weight of sensors present. Missing sensors naturally reduce
        # the maximum achievable confidence.
        total_weight = sum(ind.weight for ind in defn.indicators)
        if total_weight <= 0:
            continue

        matched_weight = 0.0
        all_matched_ids: list[str] = []

        for indicator in defn.indicators:
            score, matched_ids = _score_indicator(
                indicator, area, entity_cache=entity_cache
            )
            if score < 0:
                # No sensor of this type — scores 0, but total weight
                # still includes it (no inflation from missing sensors).
                continue
            matched_weight += score
            all_matched_ids.extend(matched_ids)

        # Raw matched weight must meet the minimum threshold to prevent
        # single-sensor false positives after normalization.
        if matched_weight < defn.min_match_weight:
            continue

        confidence = matched_weight / total_weight
        if confidence < defn.min_match_weight:
            continue

        # Compare at reported precision. Previously the raw value was
        # compared with the already-rounded ``best.confidence``, so exact
        # ties were often missed and the tie-breakers below never applied.
        rounded_confidence = round(confidence, _CONFIDENCE_DECIMALS)
        is_specific = bool(defn.purposes)
        is_tie = best is not None and rounded_confidence == best.confidence
        if (
            best is None
            or rounded_confidence > best.confidence
            or (is_tie and is_specific and not best_is_specific)
            or (
                is_tie
                and is_specific == best_is_specific
                and round(matched_weight, _CONFIDENCE_DECIMALS)
                > round(best_matched_weight, _CONFIDENCE_DECIMALS)
            )
        ):
            best = DetectedActivity(
                activity_id=defn.activity_id,
                confidence=rounded_confidence,
                # An entity can match several indicators; list it once.
                matching_indicators=list(dict.fromkeys(all_matched_ids)),
                occupancy_boost=defn.occupancy_boost,
            )
            best_matched_weight = matched_weight
            best_is_specific = is_specific

    if best is not None:
        return best

    # No specific activity detected — area is occupied but idle.
    return DetectedActivity(
        activity_id=ActivityId.IDLE,
        confidence=round(prob, _CONFIDENCE_DECIMALS),
    )
