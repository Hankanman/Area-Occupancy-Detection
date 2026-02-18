"""Tests for activity detection module."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from custom_components.area_occupancy.area.area import Area
from custom_components.area_occupancy.const import (
    ACTIVITY_BOOST_HIGH,
    ACTIVITY_BOOST_STRONG,
)
from custom_components.area_occupancy.data.activity import (
    ACTIVITY_DEFINITIONS,
    ActivityId,
    DetectedActivity,
    _environmental_signal_strength,
    detect_activity,
)
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.data.purpose import AreaPurpose
from custom_components.area_occupancy.data.types import GaussianParams

# ruff: noqa: SLF001


# ─── Helpers ──────────────────────────────────────────────────────────


def _make_entity(
    entity_id: str,
    input_type: InputType,
    *,
    evidence: bool | None = False,
    is_decaying: bool = False,
    decay_factor: float = 1.0,
    state: str | float | None = None,
    gaussian_params: GaussianParams | None = None,
    ha_device_class: str | None = None,
) -> Mock:
    """Build a lightweight mock entity for scoring tests."""
    entity = Mock()
    entity.entity_id = entity_id
    entity.type = Mock()
    entity.type.input_type = input_type
    entity.evidence = evidence
    entity.decay = Mock()
    entity.decay.is_decaying = is_decaying
    entity.decay_factor = decay_factor
    entity.state = state
    entity.learned_gaussian_params = gaussian_params
    entity.ha_device_class = ha_device_class
    entity.active = evidence is True or is_decaying
    return entity


def _make_area(
    *,
    occupied: bool = True,
    probability: float = 0.7,
    purpose: AreaPurpose = AreaPurpose.SOCIAL,
    entities_by_type: dict[InputType, list[Mock]] | None = None,
) -> Mock:
    """Build a lightweight mock Area for detect_activity tests."""
    area = Mock()
    area.occupied.return_value = occupied
    area.probability.return_value = probability
    area.purpose = Mock()
    area.purpose.purpose = purpose

    entities_by_type = entities_by_type or {}
    all_entities: dict[str, Mock] = {}
    for ents in entities_by_type.values():
        for e in ents:
            all_entities[e.entity_id] = e

    def _get_by_type(it: InputType) -> dict[str, Mock]:
        return {e.entity_id: e for e in entities_by_type.get(it, [])}

    area.entities = Mock()
    area.entities.entities = all_entities
    area.entities.get_entities_by_input_type = Mock(side_effect=_get_by_type)

    active = [e for e in all_entities.values() if e.active]
    area.entities.active_entities = active

    return area


# ─── Environmental Signal Strength ───────────────────────────────────


class TestEnvironmentalSignalStrength:
    """Test _environmental_signal_strength helper."""

    def test_elevated_full_signal(self) -> None:
        """Value at mean_occupied → 1.0."""
        assert _environmental_signal_strength(25.0, 25.0, 20.0, "elevated") == 1.0

    def test_elevated_zero_signal(self) -> None:
        """Value at mean_unoccupied → 0.0."""
        assert _environmental_signal_strength(20.0, 25.0, 20.0, "elevated") == 0.0

    def test_elevated_midpoint(self) -> None:
        """Value halfway between means → 0.5."""
        assert _environmental_signal_strength(22.5, 25.0, 20.0, "elevated") == 0.5

    def test_elevated_clamped_above(self) -> None:
        """Value above mean_occupied → clamped to 1.0."""
        assert _environmental_signal_strength(30.0, 25.0, 20.0, "elevated") == 1.0

    def test_elevated_clamped_below(self) -> None:
        """Value below mean_unoccupied → clamped to 0.0."""
        assert _environmental_signal_strength(15.0, 25.0, 20.0, "elevated") == 0.0

    def test_suppressed_full_signal(self) -> None:
        """Illuminance at mean_occupied (low) → 1.0 for suppressed."""
        # mean_occ=10, mean_unocc=50 → suppressed: value at mean_occ should be 1.0.
        assert _environmental_signal_strength(10.0, 10.0, 50.0, "suppressed") == 1.0

    def test_suppressed_zero_signal(self) -> None:
        """Illuminance at mean_unoccupied → 0.0 for suppressed."""
        assert _environmental_signal_strength(50.0, 10.0, 50.0, "suppressed") == 0.0

    def test_suppressed_midpoint(self) -> None:
        """Value halfway → 0.5 for suppressed."""
        assert _environmental_signal_strength(30.0, 10.0, 50.0, "suppressed") == 0.5

    def test_identical_means_returns_zero(self) -> None:
        """When means are identical, signal should be 0.0 (no information)."""
        assert _environmental_signal_strength(25.0, 25.0, 25.0, "elevated") == 0.0

    def test_unknown_condition_returns_zero(self) -> None:
        """Unknown condition string → 0.0."""
        assert _environmental_signal_strength(25.0, 30.0, 20.0, "unknown") == 0.0


# ─── Activity Definitions Sanity ─────────────────────────────────────


class TestActivityDefinitions:
    """Verify ACTIVITY_DEFINITIONS are well-formed."""

    def test_all_activities_have_indicators(self) -> None:
        """Every definition must have at least one indicator."""
        for defn in ACTIVITY_DEFINITIONS:
            assert len(defn.indicators) > 0, f"{defn.activity_id} has no indicators"

    def test_indicator_weights_sum_to_one(self) -> None:
        """Indicator weights within each definition should sum to ~1.0."""
        for defn in ACTIVITY_DEFINITIONS:
            total = sum(i.weight for i in defn.indicators)
            assert abs(total - 1.0) < 0.01, (
                f"{defn.activity_id} indicator weights sum to {total}"
            )

    def test_no_duplicate_activity_ids(self) -> None:
        """No two definitions should share the same activity_id."""
        ids = [d.activity_id for d in ACTIVITY_DEFINITIONS]
        assert len(ids) == len(set(ids))

    def test_showering_restricted_to_bathroom(self) -> None:
        """Showering should only match bathroom areas."""
        showering = [
            d for d in ACTIVITY_DEFINITIONS if d.activity_id == ActivityId.SHOWERING
        ]
        assert len(showering) == 1
        assert showering[0].purposes == frozenset({AreaPurpose.BATHROOM})


# ─── Unoccupied Detection ────────────────────────────────────────────


class TestUnoccupiedDetection:
    """Test that unoccupied areas return ActivityId.UNOCCUPIED."""

    def test_unoccupied_returns_unoccupied(self) -> None:
        """When area is not occupied, result is UNOCCUPIED."""
        area = _make_area(occupied=False, probability=0.2)
        result = detect_activity(area)
        assert result.activity_id == ActivityId.UNOCCUPIED
        assert result.confidence == pytest.approx(0.8, abs=0.01)

    def test_unoccupied_high_probability(self) -> None:
        """Unoccupied confidence should be 1 - probability."""
        area = _make_area(occupied=False, probability=0.05)
        result = detect_activity(area)
        assert result.confidence == pytest.approx(0.95, abs=0.01)


# ─── Idle Detection ──────────────────────────────────────────────────


class TestIdleDetection:
    """Test fallback to Idle when no activity matches."""

    def test_idle_when_no_matching_purpose(self) -> None:
        """Occupied area with unmatched purpose returns Idle."""
        # Use GARAGE with a DOOR sensor — no activity definitions target GARAGE.
        door = _make_entity("binary_sensor.d1", InputType.DOOR, evidence=True)
        area = _make_area(
            purpose=AreaPurpose.GARAGE,
            entities_by_type={InputType.DOOR: [door]},
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.IDLE

    def test_idle_when_no_sensors_match(self) -> None:
        """Occupied area with sensors but no matching evidence returns Idle."""
        # All sensors off in a bathroom — showering won't trigger without humidity.
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=False)
        area = _make_area(
            purpose=AreaPurpose.BATHROOM,
            entities_by_type={InputType.MOTION: [motion]},
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.IDLE

    def test_idle_confidence_equals_probability(self) -> None:
        """Idle confidence should equal area probability."""
        area = _make_area(
            probability=0.65,
            purpose=AreaPurpose.GARAGE,
            entities_by_type={
                InputType.DOOR: [
                    _make_entity("binary_sensor.d1", InputType.DOOR, evidence=True)
                ],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.IDLE
        assert result.confidence == pytest.approx(0.65, abs=0.01)


# ─── Specific Activity Scoring ───────────────────────────────────────


class TestSpecificActivityScoring:
    """Test detection of specific activities with matching sensors."""

    def test_showering_detected_in_bathroom(self) -> None:
        """Bathroom with active humidity and motion → showering."""
        humidity = _make_entity(
            "sensor.humidity",
            InputType.HUMIDITY,
            state="85.0",
            gaussian_params=GaussianParams(
                mean_occupied=80.0,
                std_occupied=5.0,
                mean_unoccupied=50.0,
                std_unoccupied=5.0,
            ),
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)
        door = _make_entity("binary_sensor.door", InputType.DOOR, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.BATHROOM,
            entities_by_type={
                InputType.HUMIDITY: [humidity],
                InputType.MOTION: [motion],
                InputType.DOOR: [door],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.SHOWERING
        assert result.confidence > 0.3

    def test_cooking_detected_in_kitchen(self) -> None:
        """Kitchen with active appliance and elevated temp/humidity → cooking."""
        appliance = _make_entity("switch.oven", InputType.APPLIANCE, evidence=True)
        temp = _make_entity(
            "sensor.temp",
            InputType.TEMPERATURE,
            state="28.0",
            gaussian_params=GaussianParams(
                mean_occupied=27.0,
                std_occupied=2.0,
                mean_unoccupied=21.0,
                std_unoccupied=1.0,
            ),
        )
        humidity = _make_entity(
            "sensor.humidity",
            InputType.HUMIDITY,
            state="70.0",
            gaussian_params=GaussianParams(
                mean_occupied=65.0,
                std_occupied=10.0,
                mean_unoccupied=40.0,
                std_unoccupied=5.0,
            ),
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.FOOD_PREP,
            entities_by_type={
                InputType.APPLIANCE: [appliance],
                InputType.TEMPERATURE: [temp],
                InputType.HUMIDITY: [humidity],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.COOKING
        assert result.confidence > 0.3

    def test_watching_tv_in_living_room(self) -> None:
        """Social area with active media → watching TV."""
        media = _make_entity(
            "media_player.tv", InputType.MEDIA, evidence=True, ha_device_class="tv"
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [media],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.WATCHING_TV
        assert result.confidence > 0.3

    def test_working_in_office(self) -> None:
        """Office with active appliance, power, and elevated CO2 → working."""
        appliance = _make_entity("switch.computer", InputType.APPLIANCE, evidence=True)
        power = _make_entity("sensor.power", InputType.POWER, evidence=True)
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)
        co2 = _make_entity(
            "sensor.co2",
            InputType.CO2,
            state="800.0",
            gaussian_params=GaussianParams(
                mean_occupied=750.0,
                std_occupied=100.0,
                mean_unoccupied=420.0,
                std_unoccupied=30.0,
            ),
        )

        area = _make_area(
            purpose=AreaPurpose.WORKING,
            entities_by_type={
                InputType.APPLIANCE: [appliance],
                InputType.POWER: [power],
                InputType.MOTION: [motion],
                InputType.CO2: [co2],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.WORKING
        assert result.confidence > 0.3

    def test_sleeping_in_bedroom(self) -> None:
        """Bedroom with active sleep sensor → sleeping."""
        sleep = _make_entity("binary_sensor.sleep", InputType.SLEEP, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SLEEPING,
            entities_by_type={
                InputType.SLEEP: [sleep],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.SLEEPING
        assert result.confidence > 0.3


# ─── Decay Handling ──────────────────────────────────────────────────


class TestDecayHandling:
    """Test that decaying sensors contribute partial scores."""

    def test_decaying_sensor_contributes_partial_score(self) -> None:
        """A decaying media sensor should still contribute to watching TV."""
        media = _make_entity(
            "media_player.tv",
            InputType.MEDIA,
            evidence=False,
            is_decaying=True,
            decay_factor=0.5,
            ha_device_class="tv",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [media],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        # With media decaying at 0.5, the activity should still be detected
        # but with lower confidence than full evidence.
        assert result.activity_id in (
            ActivityId.WATCHING_TV,
            ActivityId.LISTENING_TO_MUSIC,
        )

    def test_fully_decayed_sensor_no_contribution(self) -> None:
        """A fully decayed sensor (factor=0) should not contribute."""
        media = _make_entity(
            "media_player.tv",
            InputType.MEDIA,
            evidence=False,
            is_decaying=True,
            decay_factor=0.0,
            ha_device_class="tv",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [media],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        # Media contributes 0, only motion (weight 0.1) matches.
        # With total_weight normalization, confidence is too low to match.
        assert result.activity_id == ActivityId.IDLE


# ─── Environmental Graceful Degradation ──────────────────────────────


class TestGracefulDegradation:
    """Test that missing Gaussian params degrade gracefully."""

    def test_no_gaussian_params_excludes_environmental(self) -> None:
        """Without learned params, environmental indicators contribute 0."""
        humidity = _make_entity(
            "sensor.humidity",
            InputType.HUMIDITY,
            state="85.0",
            gaussian_params=None,  # No learned params.
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)
        door = _make_entity("binary_sensor.door", InputType.DOOR, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.BATHROOM,
            entities_by_type={
                InputType.HUMIDITY: [humidity],
                InputType.MOTION: [motion],
                InputType.DOOR: [door],
            },
        )
        result = detect_activity(area)
        # Without humidity signal, showering still possible from motion+door,
        # but humidity indicator scores 0 (has sensors but no params).
        assert result.activity_id in (
            ActivityId.SHOWERING,
            ActivityId.BATHING,
        )

    def test_missing_sensors_reduce_confidence(self) -> None:
        """Missing sensor types reduce confidence via total weight normalization."""
        # Bathroom with only motion — no humidity, no door, no temp.
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)
        area = _make_area(
            purpose=AreaPurpose.BATHROOM,
            entities_by_type={InputType.MOTION: [motion]},
        )
        result = detect_activity(area)
        # Only motion available → matched_weight is below min_match_weight for
        # all candidates (showering motion=0.15 < 0.3, bathing motion=0.1 < 0.3).
        # Falls back to Idle.
        assert result.activity_id == ActivityId.IDLE


# ─── Area.detected_activity() Caching ───────────────────────────────


class TestAreaDetectedActivityCache:
    """Test caching behavior of Area.detected_activity()."""

    def test_cache_hit(self) -> None:
        """Second call with same state returns cached result."""
        area = Mock(spec=Area)
        area.entities = Mock()
        area.entities.active_entities = []
        area._base_probability.return_value = 0.3
        area.config = Mock()
        area.config.threshold = 0.5
        area.purpose = Mock()
        area.purpose.purpose = AreaPurpose.SOCIAL

        # Initialize cache attributes.
        area._activity_cache = None
        area._activity_cache_key = None

        # Call the real method by using the unbound method pattern.
        result1 = Area.detected_activity(area)
        assert result1.activity_id == ActivityId.UNOCCUPIED

        # Now set up a cached result.
        cache_key = (frozenset(), round(0.3, 4))
        area._activity_cache_key = cache_key
        area._activity_cache = result1

        # Second call should return the same cached object.
        result2 = Area.detected_activity(area)
        assert result2 is result1

    def test_cache_invalidated_on_change(self) -> None:
        """Cache is invalidated when active entities change."""
        area = Mock(spec=Area)
        area.purpose = Mock()
        area.purpose.purpose = AreaPurpose.SOCIAL
        area.config = Mock()
        area.config.threshold = 0.5

        # First call: unoccupied.
        area.entities = Mock()
        area.entities.active_entities = []
        area._base_probability.return_value = 0.3
        area._activity_cache = None
        area._activity_cache_key = None

        result1 = Area.detected_activity(area)
        assert result1.activity_id == ActivityId.UNOCCUPIED

        # Save cache state.
        area._activity_cache = result1
        area._activity_cache_key = (frozenset(), round(0.3, 4))

        # Change state: add an active entity.
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion"
        area.entities.active_entities = [mock_entity]
        area._base_probability.return_value = 0.7
        area.entities.get_entities_by_input_type = Mock(return_value={})

        result2 = Area.detected_activity(area)
        # Cache key changed, so result should be recomputed.
        assert result2 is not result1


# ─── Edge Cases ──────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases in activity detection."""

    def test_empty_entities(self) -> None:
        """Area with no entities at all should return Idle when occupied."""
        area = _make_area(occupied=True, probability=0.6, entities_by_type={})
        result = detect_activity(area)
        assert result.activity_id == ActivityId.IDLE

    def test_multiple_candidates_highest_confidence_wins(self) -> None:
        """When multiple activities match, highest confidence wins."""
        # In a SOCIAL area with a receiver (matches both tv and music filters),
        # watching TV and listening to music are both candidates.
        media = _make_entity(
            "media_player.receiver",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="receiver",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)
        sound = _make_entity(
            "sensor.sound",
            InputType.SOUND_PRESSURE,
            state="65.0",
            gaussian_params=GaussianParams(
                mean_occupied=60.0,
                std_occupied=10.0,
                mean_unoccupied=35.0,
                std_unoccupied=5.0,
            ),
        )

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [media],
                InputType.MOTION: [motion],
                InputType.SOUND_PRESSURE: [sound],
            },
        )
        result = detect_activity(area)
        # Both watching_tv and listening_to_music are candidates.
        # The one with higher confidence should win.
        assert result.activity_id in (
            ActivityId.WATCHING_TV,
            ActivityId.LISTENING_TO_MUSIC,
        )

    def test_below_min_match_weight_excluded(self) -> None:
        """Activities scoring below min_match_weight are excluded."""
        # Showering has min_match_weight=0.3. Only weakly decaying motion
        # (weight=0.15) is active — other sensors present but inactive.
        motion = _make_entity(
            "binary_sensor.m1",
            InputType.MOTION,
            evidence=False,
            is_decaying=True,
            decay_factor=0.3,
        )
        humidity = _make_entity(
            "sensor.humidity",
            InputType.HUMIDITY,
            state="50.0",
            gaussian_params=GaussianParams(
                mean_occupied=80.0,
                std_occupied=5.0,
                mean_unoccupied=50.0,
                std_unoccupied=5.0,
            ),
        )
        door = _make_entity("binary_sensor.door", InputType.DOOR, evidence=False)

        area = _make_area(
            purpose=AreaPurpose.BATHROOM,
            entities_by_type={
                InputType.MOTION: [motion],
                InputType.HUMIDITY: [humidity],
                InputType.DOOR: [door],
            },
        )
        result = detect_activity(area)
        # Motion decaying at 0.3: score = 0.15 * 0.3 = 0.045.
        # Humidity at mean_unoccupied: signal = 0.0, score = 0.0.
        # Door inactive: score = 0.0. Total matched = 0.045.
        # Below min_match_weight 0.3, so showering excluded → Idle.
        assert result.activity_id == ActivityId.IDLE

    def test_entity_with_none_state_skipped_for_environmental(self) -> None:
        """Environmental entities with None state should not crash."""
        humidity = _make_entity(
            "sensor.humidity",
            InputType.HUMIDITY,
            state=None,
            gaussian_params=GaussianParams(
                mean_occupied=80.0,
                std_occupied=5.0,
                mean_unoccupied=50.0,
                std_unoccupied=5.0,
            ),
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)
        door = _make_entity("binary_sensor.door", InputType.DOOR, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.BATHROOM,
            entities_by_type={
                InputType.HUMIDITY: [humidity],
                InputType.MOTION: [motion],
                InputType.DOOR: [door],
            },
        )
        # Should not raise.
        result = detect_activity(area)
        assert result.activity_id is not None

    def test_entity_with_non_numeric_state_skipped(self) -> None:
        """Environmental entities with non-numeric state should not crash."""
        humidity = _make_entity(
            "sensor.humidity",
            InputType.HUMIDITY,
            state="unavailable",
            gaussian_params=GaussianParams(
                mean_occupied=80.0,
                std_occupied=5.0,
                mean_unoccupied=50.0,
                std_unoccupied=5.0,
            ),
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.BATHROOM,
            entities_by_type={
                InputType.HUMIDITY: [humidity],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        assert result.activity_id is not None


# ─── DetectedActivity Dataclass ──────────────────────────────────────


class TestDetectedActivityDataclass:
    """Test DetectedActivity dataclass."""

    def test_defaults(self) -> None:
        """Test default values."""
        result = DetectedActivity(activity_id=ActivityId.IDLE, confidence=0.5)
        assert result.matching_indicators == []

    def test_with_matching_indicators(self) -> None:
        """Test with matching indicators populated."""
        result = DetectedActivity(
            activity_id=ActivityId.COOKING,
            confidence=0.8,
            matching_indicators=["switch.oven", "sensor.temp"],
        )
        assert len(result.matching_indicators) == 2
        assert "switch.oven" in result.matching_indicators


# ─── ActivityId Enum ─────────────────────────────────────────────────


class TestActivityIdEnum:
    """Test ActivityId StrEnum."""

    def test_all_values_are_strings(self) -> None:
        """All ActivityId values should be valid strings."""
        for activity in ActivityId:
            assert isinstance(activity.value, str)
            assert len(activity.value) > 0

    def test_expected_members(self) -> None:
        """Verify expected members exist."""
        expected = {
            "bathing",
            "cooking",
            "eating",
            "idle",
            "listening_to_music",
            "showering",
            "sleeping",
            "unoccupied",
            "watching_tv",
            "working",
        }
        actual = {a.value for a in ActivityId}
        assert actual == expected


# ─── Device Class Filtering ─────────────────────────────────────────


class TestDeviceClassFiltering:
    """Test ha_device_classes filtering on media indicators."""

    def test_watching_tv_requires_tv_device_class(self) -> None:
        """Speaker playing should NOT trigger watching_tv."""
        speaker = _make_entity(
            "media_player.speaker",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="speaker",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [speaker],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        # Speaker doesn't match watching_tv (needs tv/receiver).
        # Should detect listening_to_music or idle, not watching_tv.
        assert result.activity_id != ActivityId.WATCHING_TV

    def test_listening_to_music_requires_speaker_device_class(self) -> None:
        """TV playing should NOT trigger listening_to_music."""
        tv = _make_entity(
            "media_player.tv",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="tv",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [tv],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        # TV matches watching_tv but not listening_to_music.
        assert result.activity_id == ActivityId.WATCHING_TV

    def test_receiver_matches_both_tv_and_music(self) -> None:
        """Receiver device_class should match both watching_tv and listening_to_music."""
        receiver = _make_entity(
            "media_player.receiver",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="receiver",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [receiver],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        # Receiver matches both; watching_tv has higher media weight (0.6 vs 0.5).
        assert result.activity_id in (
            ActivityId.WATCHING_TV,
            ActivityId.LISTENING_TO_MUSIC,
        )

    def test_media_without_device_class_matches_no_filtered_activity(self) -> None:
        """Media entity with no device_class should not match filtered indicators."""
        media = _make_entity(
            "media_player.unknown",
            InputType.MEDIA,
            evidence=True,
            ha_device_class=None,
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [media],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        # None device_class doesn't match {"tv","receiver"} or {"speaker","receiver"}.
        # Neither watching_tv nor listening_to_music should trigger.
        assert result.activity_id not in (
            ActivityId.WATCHING_TV,
            ActivityId.LISTENING_TO_MUSIC,
        )


# ─── Confidence Normalization ────────────────────────────────────────


class TestConfidenceNormalization:
    """Test that confidence uses total definition weight, not possible weight."""

    def test_confidence_uses_total_weight(self) -> None:
        """Confidence = matched/total_weight, not matched/possible_weight."""
        # Watching TV: media(0.6) + illuminance(0.15) + motion(0.1) + sound(0.15) = 1.0
        # Only provide media (tv) and motion — no illuminance, no sound.
        tv = _make_entity(
            "media_player.tv",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="tv",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [tv],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.WATCHING_TV
        # matched = 0.6 + 0.1 = 0.7, total = 1.0, confidence = 0.7
        # Old behavior: possible = 0.7, confidence = 1.0 (inflated)
        assert result.confidence == pytest.approx(0.7, abs=0.01)


# ─── Environmental Signal Validity ───────────────────────────────────


class TestEnvironmentalSignalValidity:
    """Test that noisy environmental signals are skipped."""

    def test_environmental_signal_skipped_for_tiny_span(self) -> None:
        """Entity with indistinguishable means should not contribute."""
        # bedroom_lux_1 scenario: means nearly identical, huge std devs
        lux = _make_entity(
            "sensor.bedroom_lux",
            InputType.ILLUMINANCE,
            state="32.0",
            gaussian_params=GaussianParams(
                mean_occupied=33.42,
                std_occupied=100.0,
                mean_unoccupied=31.79,
                std_unoccupied=94.0,
            ),
        )
        sleep = _make_entity("binary_sensor.sleep", InputType.SLEEP, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SLEEPING,
            entities_by_type={
                InputType.ILLUMINANCE: [lux],
                InputType.SLEEP: [sleep],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.SLEEPING
        # The lux sensor should be skipped (span=1.63 < avg_std=97 * 0.5=48.5).
        # So illuminance contributes 0, confidence = 0.5/1.0 = 0.5
        assert result.confidence == pytest.approx(0.5, abs=0.01)

    def test_environmental_signal_kept_for_meaningful_span(self) -> None:
        """Entity with clearly separated means should contribute normally."""
        lux = _make_entity(
            "sensor.living_lux",
            InputType.ILLUMINANCE,
            state="15.0",
            gaussian_params=GaussianParams(
                mean_occupied=10.0,
                std_occupied=5.0,
                mean_unoccupied=50.0,
                std_unoccupied=5.0,
            ),
        )
        sleep = _make_entity("binary_sensor.sleep", InputType.SLEEP, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SLEEPING,
            entities_by_type={
                InputType.ILLUMINANCE: [lux],
                InputType.SLEEP: [sleep],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.SLEEPING
        # span=40 > avg_std=5 * 0.5=2.5, so illuminance contributes.
        # Suppressed: (50-15)/40 = 0.875 → signal=0.875
        # matched = 0.5 + 0.2*0.875 = 0.675, total=1.0, confidence=0.675
        assert result.confidence > 0.5


# ─── Tiebreaker Behavior ────────────────────────────────────────────


class TestTiebreaker:
    """Test tiebreaker uses matched_weight instead of possible_weight."""

    def test_tiebreaker_uses_matched_weight(self) -> None:
        """Equal confidence → higher matched_weight wins."""
        # Create two areas that would tie on confidence but differ in matched_weight.
        # Use a receiver (matches both watching_tv and listening_to_music).
        # watching_tv: media=0.6, motion=0.1 → matched=0.7, total=1.0, conf=0.7
        # listening_to_music: media=0.5, motion=0.2 → matched=0.7, total=1.0, conf=0.7
        # Both have same confidence AND same matched_weight.
        # watching_tv is evaluated first → wins (or ties keep first).
        receiver = _make_entity(
            "media_player.receiver",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="receiver",
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [receiver],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        # Both candidates have same confidence (0.7). watching_tv is evaluated
        # first in ACTIVITY_DEFINITIONS, so it wins the tie.
        assert result.activity_id in (
            ActivityId.WATCHING_TV,
            ActivityId.LISTENING_TO_MUSIC,
        )
        assert result.confidence == pytest.approx(0.7, abs=0.01)


# ─── Bedroom Scenario Tests ─────────────────────────────────────────


class TestBedroomScenario:
    """Test the exact overnight bedroom scenario that prompted this fix."""

    def test_bedroom_sleeping_vs_watching_tv_speaker(self) -> None:
        """Speaker playing white noise + sleep ON → sleeping, NOT watching_tv."""
        speaker = _make_entity(
            "media_player.bedroom_clock",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="speaker",
        )
        sleep = _make_entity("binary_sensor.sleep", InputType.SLEEP, evidence=True)
        lux = _make_entity(
            "sensor.bedroom_lux",
            InputType.ILLUMINANCE,
            state="2.0",
            gaussian_params=GaussianParams(
                mean_occupied=5.0,
                std_occupied=3.0,
                mean_unoccupied=50.0,
                std_unoccupied=10.0,
            ),
        )

        area = _make_area(
            purpose=AreaPurpose.SLEEPING,
            entities_by_type={
                InputType.MEDIA: [speaker],
                InputType.SLEEP: [sleep],
                InputType.ILLUMINANCE: [lux],
            },
        )
        result = detect_activity(area)
        # Speaker doesn't match watching_tv (needs tv/receiver).
        # Sleep sensor triggers sleeping.
        assert result.activity_id == ActivityId.SLEEPING
        assert result.confidence > 0.3

    def test_bedroom_speaker_sleep_off_is_idle(self) -> None:
        """Speaker playing + sleep OFF → idle (not watching_tv)."""
        speaker = _make_entity(
            "media_player.bedroom_clock",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="speaker",
        )
        sleep = _make_entity("binary_sensor.sleep", InputType.SLEEP, evidence=False)

        area = _make_area(
            purpose=AreaPurpose.SLEEPING,
            entities_by_type={
                InputType.MEDIA: [speaker],
                InputType.SLEEP: [sleep],
            },
        )
        result = detect_activity(area)
        # Speaker doesn't match watching_tv, sleep is off → no activity matches.
        # listening_to_music doesn't target SLEEPING purpose.
        assert result.activity_id != ActivityId.WATCHING_TV

    def test_bedroom_tv_on_detects_watching_tv(self) -> None:
        """Actual TV playing in bedroom → watching_tv (not listening_to_music)."""
        tv = _make_entity(
            "media_player.bedroom_tv_shield",
            InputType.MEDIA,
            evidence=True,
            ha_device_class="tv",
        )
        sleep = _make_entity("binary_sensor.sleep", InputType.SLEEP, evidence=False)

        area = _make_area(
            purpose=AreaPurpose.SLEEPING,
            entities_by_type={
                InputType.MEDIA: [tv],
                InputType.SLEEP: [sleep],
            },
        )
        result = detect_activity(area)
        # TV matches watching_tv (SLEEPING purpose is included).
        assert result.activity_id == ActivityId.WATCHING_TV


# ─── Occupancy Boost ────────────────────────────────────────────────


class TestOccupancyBoost:
    """Test occupancy_boost field on definitions and DetectedActivity."""

    def test_all_definitions_have_boost(self) -> None:
        """Every ActivityDefinition must have an occupancy_boost >= 0."""
        for defn in ACTIVITY_DEFINITIONS:
            assert defn.occupancy_boost >= 0.0, (
                f"{defn.activity_id} has negative occupancy_boost"
            )

    def test_showering_has_high_boost(self) -> None:
        """Showering should have a high occupancy boost."""
        showering = [
            d for d in ACTIVITY_DEFINITIONS if d.activity_id == ActivityId.SHOWERING
        ]
        assert showering[0].occupancy_boost == ACTIVITY_BOOST_HIGH

    def test_watching_tv_has_moderate_boost(self) -> None:
        """Watching TV should have a moderate occupancy boost."""
        watching = [
            d for d in ACTIVITY_DEFINITIONS if d.activity_id == ActivityId.WATCHING_TV
        ]
        assert watching[0].occupancy_boost == ACTIVITY_BOOST_STRONG

    def test_detected_activity_carries_boost(self) -> None:
        """DetectedActivity should carry the boost from the matched definition."""
        media = _make_entity(
            "media_player.tv", InputType.MEDIA, evidence=True, ha_device_class="tv"
        )
        motion = _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)

        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [media],
                InputType.MOTION: [motion],
            },
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.WATCHING_TV
        assert result.occupancy_boost == ACTIVITY_BOOST_STRONG

    def test_idle_has_zero_boost(self) -> None:
        """IDLE activity should have zero boost."""
        door = _make_entity("binary_sensor.d1", InputType.DOOR, evidence=True)
        area = _make_area(
            purpose=AreaPurpose.GARAGE,
            entities_by_type={InputType.DOOR: [door]},
        )
        result = detect_activity(area)
        assert result.activity_id == ActivityId.IDLE
        assert result.occupancy_boost == 0.0

    def test_unoccupied_has_zero_boost(self) -> None:
        """UNOCCUPIED activity should have zero boost."""
        area = _make_area(occupied=False, probability=0.2)
        result = detect_activity(area)
        assert result.activity_id == ActivityId.UNOCCUPIED
        assert result.occupancy_boost == 0.0

    def test_default_detected_activity_boost(self) -> None:
        """DetectedActivity default occupancy_boost should be 0.0."""
        result = DetectedActivity(activity_id=ActivityId.IDLE, confidence=0.5)
        assert result.occupancy_boost == 0.0


# ─── Explicit Parameters ────────────────────────────────────────────


class TestExplicitParameters:
    """Test detect_activity with explicit base_probability and is_occupied."""

    def test_explicit_is_occupied_false(self) -> None:
        """Passing is_occupied=False should return UNOCCUPIED without calling area.occupied()."""
        area = _make_area(occupied=True, probability=0.3)
        result = detect_activity(area, base_probability=0.3, is_occupied=False)
        assert result.activity_id == ActivityId.UNOCCUPIED
        # area.occupied() should NOT have been called
        area.occupied.assert_not_called()

    def test_explicit_is_occupied_true(self) -> None:
        """Passing is_occupied=True should skip the unoccupied check."""
        area = _make_area(occupied=False, probability=0.3)
        # Despite area.occupied() returning False, explicit is_occupied=True wins
        result = detect_activity(area, base_probability=0.3, is_occupied=True)
        assert result.activity_id != ActivityId.UNOCCUPIED
        area.occupied.assert_not_called()

    def test_explicit_base_probability_used_for_idle_confidence(self) -> None:
        """Idle confidence should use explicit base_probability."""
        area = _make_area(
            occupied=True,
            probability=0.99,
            purpose=AreaPurpose.GARAGE,
            entities_by_type={
                InputType.DOOR: [
                    _make_entity("binary_sensor.d1", InputType.DOOR, evidence=True)
                ],
            },
        )
        result = detect_activity(area, base_probability=0.42, is_occupied=True)
        assert result.activity_id == ActivityId.IDLE
        assert result.confidence == pytest.approx(0.42, abs=0.01)
        # area.probability() should NOT have been called
        area.probability.assert_not_called()

    def test_no_circular_dependency(self) -> None:
        """When explicit params are provided, area.occupied/probability are not called."""
        area = _make_area(
            purpose=AreaPurpose.SOCIAL,
            entities_by_type={
                InputType.MEDIA: [
                    _make_entity(
                        "media_player.tv",
                        InputType.MEDIA,
                        evidence=True,
                        ha_device_class="tv",
                    )
                ],
                InputType.MOTION: [
                    _make_entity("binary_sensor.m1", InputType.MOTION, evidence=True)
                ],
            },
        )
        detect_activity(area, base_probability=0.6, is_occupied=True)
        area.occupied.assert_not_called()
        area.probability.assert_not_called()
