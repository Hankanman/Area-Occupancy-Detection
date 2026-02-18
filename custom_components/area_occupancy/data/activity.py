"""Activity detection for Area Occupancy Detection.

Detects what activity is happening in an area based on sensor signals
and the area's configured purpose.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass, field
from enum import StrEnum
import logging
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


@dataclass(frozen=True)
class Indicator:
    """A single sensor signal that indicates an activity."""

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

ACTIVITY_DEFINITIONS: tuple[ActivityDefinition, ...] = (
    ActivityDefinition(
        activity_id=ActivityId.SHOWERING,
        indicators=(
            Indicator(
                InputType.HUMIDITY,
                0.5,
                require_active=False,
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.TEMPERATURE,
                0.2,
                require_active=False,
                environmental_condition="elevated",
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
                environmental_condition="elevated",
            ),
            Indicator(InputType.DOOR, 0.3),
            Indicator(
                InputType.TEMPERATURE,
                0.2,
                require_active=False,
                environmental_condition="elevated",
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
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.HUMIDITY,
                0.15,
                require_active=False,
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.CO2,
                0.1,
                require_active=False,
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.VOC,
                0.1,
                require_active=False,
                environmental_condition="elevated",
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
                environmental_condition="suppressed",
            ),
            Indicator(InputType.MOTION, 0.1),
            Indicator(
                InputType.SOUND_PRESSURE,
                0.15,
                require_active=False,
                environmental_condition="elevated",
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
                environmental_condition="elevated",
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
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.ILLUMINANCE,
                0.1,
                require_active=False,
                environmental_condition="elevated",
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
                environmental_condition="suppressed",
            ),
            Indicator(
                InputType.CO2,
                0.15,
                require_active=False,
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.SOUND_PRESSURE,
                0.15,
                require_active=False,
                environmental_condition="suppressed",
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
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.CO2,
                0.2,
                require_active=False,
                environmental_condition="elevated",
            ),
            Indicator(
                InputType.TEMPERATURE,
                0.15,
                require_active=False,
                environmental_condition="elevated",
            ),
            Indicator(InputType.MEDIA, 0.1),
        ),
        purposes=frozenset({AreaPurpose.EATING}),
        occupancy_boost=ACTIVITY_BOOST_MILD,
    ),
)


def _environmental_signal_strength(
    value: float,
    mean_occupied: float,
    mean_unoccupied: float,
    condition: str,
) -> float:
    """Compute 0-1 signal strength for an environmental reading.

    Uses learned Gaussian means to determine how far the current value
    has moved from unoccupied toward occupied (or away, for suppressed).

    Returns 0.0 when means are too close to distinguish.
    """
    span = mean_occupied - mean_unoccupied
    if abs(span) < 1e-9:
        return 0.0

    if condition == "elevated":
        position = (value - mean_unoccupied) / span
    elif condition == "suppressed":
        position = (mean_unoccupied - value) / abs(span)
    else:
        return 0.0

    return max(0.0, min(1.0, position))


def _score_indicator(indicator: Indicator, area: Area) -> tuple[float, list[str]]:
    """Score a single indicator against the area's entities.

    Returns:
        Tuple of (weighted_score, list_of_matching_entity_ids).
        weighted_score is indicator.weight * match_strength.
        Returns (-1.0, []) if no sensors of the required type exist.
    """
    entities = area.entities.get_entities_by_input_type(indicator.input_type)
    if not entities:
        # No sensor of this type in the area — exclude from scoring.
        return (-1.0, [])

    if indicator.environmental_condition is not None:
        return _score_environmental_indicator(indicator, entities)

    return _score_binary_indicator(indicator, area, entities)


def _score_binary_indicator(
    indicator: Indicator,
    area: Area,
    entities: dict[str, Entity],
) -> tuple[float, list[str]]:
    """Score a binary (active/inactive) indicator."""
    best_strength = 0.0
    matched_ids: list[str] = []

    for entity_id, entity in entities.items():
        # Skip entities that don't match required device_class.
        # When all entities are filtered out, this returns (0.0, []) — not
        # (-1.0, []) — because the area *has* sensors of this type, they just
        # aren't the right kind. With total-weight normalization the outcome
        # is identical, but semantically "no matching device" ≠ "no sensor."
        if indicator.ha_device_classes is not None:
            if entity.ha_device_class not in indicator.ha_device_classes:
                continue

        if indicator.require_active:
            if entity.evidence is True:
                strength = 1.0
            elif entity.decay.is_decaying:
                strength = entity.decay_factor
            else:
                continue
        # Non-active-required binary: just check available.
        elif entity.evidence is True:
            strength = 1.0
        elif entity.decay.is_decaying:
            strength = entity.decay_factor
        else:
            continue

        if strength > best_strength:
            best_strength = strength
            matched_ids = [entity_id]
        elif strength == best_strength and strength > 0:
            matched_ids.append(entity_id)

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

        val: float | None = None
        with suppress(ValueError, TypeError):
            val = float(state)
        if val is None:
            continue

        # Skip entities where means are not statistically distinguishable
        span = abs(params.mean_occupied - params.mean_unoccupied)
        avg_std = (params.std_occupied + params.std_unoccupied) / 2
        if avg_std > 0 and span < avg_std * 0.5:
            continue

        strength = _environmental_signal_strength(
            val,
            params.mean_occupied,
            params.mean_unoccupied,
            indicator.environmental_condition,
        )

        if strength > best_strength:
            best_strength = strength
            matched_ids = [entity_id]
        elif strength == best_strength and strength > 0:
            matched_ids.append(entity_id)

    return (indicator.weight * best_strength, matched_ids)


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
    6. Highest confidence wins; ties broken by purpose-specificity,
       then by matched_weight (more actual evidence). If none match,
       return Idle.
    """
    prob = base_probability if base_probability is not None else area.probability()
    occupied = (
        is_occupied
        if is_occupied is not None
        else (
            prob >= area.config.threshold
            if base_probability is not None
            else area.occupied()
        )
    )

    if not occupied:
        return DetectedActivity(
            activity_id=ActivityId.UNOCCUPIED,
            confidence=round(1.0 - prob, 4),
        )

    purpose = area.purpose.purpose

    best: DetectedActivity | None = None
    best_matched_weight = 0.0
    best_is_specific = False

    for defn in ACTIVITY_DEFINITIONS:
        # Filter by purpose (empty purposes = any).
        if defn.purposes and purpose not in defn.purposes:
            continue

        matched_weight = 0.0
        all_matched_ids: list[str] = []

        for indicator in defn.indicators:
            score, matched_ids = _score_indicator(indicator, area)
            if score < 0:
                # No sensor of this type — scores 0, but total weight
                # still includes it (no inflation from missing sensors).
                continue
            matched_weight += score
            all_matched_ids.extend(matched_ids)

        # Normalize by total definition weight (always ~1.0), not just
        # the weight of sensors present. Missing sensors naturally reduce
        # the maximum achievable confidence.
        total_weight = sum(ind.weight for ind in defn.indicators)
        if total_weight <= 0:
            continue

        # Raw matched weight must meet the minimum threshold to prevent
        # single-sensor false positives after normalization.
        if matched_weight < defn.min_match_weight:
            continue

        confidence = matched_weight / total_weight
        if confidence < defn.min_match_weight:
            continue

        # Prefer higher confidence; break ties by purpose-specificity,
        # then by more actual evidence (matched_weight).
        is_specific = bool(defn.purposes)
        if (
            best is None
            or confidence > best.confidence
            or (confidence == best.confidence and is_specific and not best_is_specific)
            or (
                confidence == best.confidence
                and is_specific == best_is_specific
                and matched_weight > best_matched_weight
            )
        ):
            best = DetectedActivity(
                activity_id=defn.activity_id,
                confidence=round(confidence, 4),
                matching_indicators=all_matched_ids,
                occupancy_boost=defn.occupancy_boost,
            )
            best_matched_weight = matched_weight
            best_is_specific = is_specific

    if best is not None:
        return best

    # No specific activity detected — area is occupied but idle.
    return DetectedActivity(
        activity_id=ActivityId.IDLE,
        confidence=round(prob, 4),
    )
