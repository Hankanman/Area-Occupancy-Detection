"""Tests for utils module."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.utils import (
    _extract_entity_keywords,
    _fuzzy_match_binary_sensor,
    _fuzzy_match_media_player,
    _fuzzy_match_sensor,
    _validate_and_correct_device_class,
    bayesian_probability,
    detect_device_class,
    detect_entity_type_from_device_class,
    format_float,
    get_device_class_with_fallback,
    get_entity_type_description,
    overall_probability,
    states_to_intervals,
    validate_datetime,
    validate_decay_factor,
    validate_prior,
    validate_prob,
    validate_weight,
)
from homeassistant.core import State
from homeassistant.util import dt as dt_util

if TYPE_CHECKING:
    from custom_components.area_occupancy.data.entity import Entity


# ruff: noqa: PLC0415
class TestValidateProb:
    """Test validate_prob function."""

    def test_valid_probabilities(self) -> None:
        """Test validation of valid probability values."""
        assert validate_prob(0.0) == 0.001  # Minimum enforced to avoid division by zero
        assert validate_prob(1.0) == 1.0
        assert validate_prob(0.5) == 0.5
        assert validate_prob(0.001) == 0.001
        assert validate_prob(0.999) == 0.999

    def test_clamp_invalid_probabilities(self) -> None:
        """Test clamping of invalid probability values."""
        assert validate_prob(-0.1) == 0.001  # Minimum enforced
        assert validate_prob(1.1) == 1.0
        assert validate_prob(-999) == 0.001  # Minimum enforced
        assert validate_prob(999) == 1.0

    def test_type_conversion(self) -> None:
        """Test type conversion to float."""
        assert validate_prob(1) == 1.0
        assert validate_prob(0) == 0.001  # Minimum enforced


class TestValidatePrior:
    """Test validate_prior function."""

    def test_valid_priors(self) -> None:
        """Test validation of valid prior values."""
        assert validate_prior(0.001) == 0.001
        assert validate_prior(0.5) == 0.5
        assert validate_prior(0.999) == 0.999

    def test_clamp_invalid_priors(self) -> None:
        """Test clamping of invalid prior values."""
        assert validate_prior(0.0) == 0.000001  # Minimum enforced
        assert validate_prior(-0.1) == 0.000001  # Minimum enforced
        assert validate_prior(1.0) == 0.999999
        assert validate_prior(1.1) == 0.999999


class TestValidateWeight:
    """Test validate_weight function."""

    def test_valid_weights(self) -> None:
        """Test validation of valid weight values."""
        assert validate_weight(0.01) == 0.01
        assert validate_weight(0.5) == 0.5
        assert validate_weight(0.99) == 0.99

    def test_clamp_invalid_weights(self) -> None:
        """Test clamping of invalid weight values."""
        assert validate_weight(0.0) == 0.01
        assert validate_weight(-0.1) == 0.01
        assert validate_weight(1.0) == 0.99
        assert validate_weight(1.1) == 0.99


class TestValidateDecayFactor:
    """Test validate_decay_factor function."""

    def test_valid_decay_factors(self) -> None:
        """Test validation of valid decay factor values."""
        assert validate_decay_factor(0.0) == 0.0
        assert validate_decay_factor(0.5) == 0.5
        assert validate_decay_factor(1.0) == 1.0

    def test_clamp_invalid_decay_factors(self) -> None:
        """Test clamping of invalid decay factor values."""
        assert validate_decay_factor(-0.1) == 0.0
        assert validate_decay_factor(1.1) == 1.0


class TestValidateDatetime:
    """Test validate_datetime function."""

    def test_valid_datetime(self) -> None:
        """Test validation of valid datetime objects."""
        now = dt_util.utcnow()
        assert validate_datetime(now) == now

    def test_none_datetime(self) -> None:
        """Test handling of None datetime."""
        result = validate_datetime(None)
        assert isinstance(result, datetime)
        # Should be recent (within last minute)
        assert (dt_util.utcnow() - result).total_seconds() < 60


class TestFormatFloat:
    """Test format_float function."""

    def test_formatting(self) -> None:
        """Test float formatting to 2 decimal places."""
        assert format_float(1.234567) == 1.23
        assert format_float(1.0) == 1.0
        assert format_float(0.999) == 1.0
        assert format_float(0.001) == 0.0


class TestBayesianProbability:
    """Test bayesian_probability function."""

    def test_active_state_calculation(self) -> None:
        """Test Bayesian calculation when sensor is active."""
        # P(occupied | sensor active) = P(sensor active | occupied) * P(occupied) / P(sensor active)
        prior = 0.3
        prob_given_true = 0.8
        prob_given_false = 0.1

        # Expected: (0.8 * 0.3) / (0.8 * 0.3 + 0.1 * 0.7) = 0.24 / 0.31 ≈ 0.774
        result = bayesian_probability(
            prior=prior,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            evidence=True,
            decay_factor=1.0,
        )
        assert abs(result - 0.774) < 0.001

    def test_inactive_state_calculation(self) -> None:
        """Test Bayesian calculation when sensor is inactive."""
        prior = 0.3
        prob_given_true = 0.8
        prob_given_false = 0.1

        # For inactive state: P(occupied | inactive)
        # P(inactive|occupied) = 1 - prob_given_true = 1 - 0.8 = 0.2
        # P(inactive|empty) = 1 - prob_given_false = 1 - 0.1 = 0.9
        # numerator = P(inactive|occupied) * P(occupied) = 0.2 * 0.3 = 0.06
        # denominator = (0.2 * 0.3) + (0.9 * 0.7) = 0.06 + 0.63 = 0.69
        # result = 0.06 / 0.69 ≈ 0.087
        result = bayesian_probability(
            prior=prior,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            evidence=False,
            decay_factor=1.0,
        )
        # Correct calculation:
        # P(OFF|occupied) = 0.2, P(OFF|unoccupied) = 0.9
        # P(evidence) = 0.2 * 0.3 + 0.9 * 0.7 = 0.69
        # P(occupied|OFF) = (0.2 * 0.3) / 0.69 ≈ 0.087
        assert abs(result - 0.087) < 0.01

    def test_edge_cases(self) -> None:
        """Test edge cases for Bayesian calculation."""
        # Test with extreme values - function should clamp to valid range
        result = bayesian_probability(
            prior=0.0001,
            prob_given_true=0.99,
            prob_given_false=0.01,
            evidence=True,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

        result = bayesian_probability(
            prior=0.9999,
            prob_given_true=0.99,
            prob_given_false=0.01,
            evidence=False,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

    def test_validation_of_inputs(self) -> None:
        """Test that invalid inputs are handled properly."""
        # The function should handle out-of-bounds inputs gracefully
        result = bayesian_probability(
            prior=-0.1,
            prob_given_true=1.1,
            prob_given_false=-0.1,
            evidence=True,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

        result = bayesian_probability(
            prior=1.1,
            prob_given_true=-0.1,
            prob_given_false=1.1,
            evidence=False,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0

    def test_bayesian_probability_fractional_weight(self) -> None:
        """Test Bayesian probability with fractional weight (now handled in likelihood)."""
        # Weight is now applied in likelihood calculation, not in bayesian_probability
        result = bayesian_probability(
            prior=0.5,
            prob_given_true=0.8,
            prob_given_false=0.2,
            evidence=True,
            decay_factor=1.0,
        )
        assert 0.0 <= result <= 1.0
        # Just test that it returns a valid probability
        assert 0.5 <= result <= 1.0

    def test_bayesian_probability_fractional_decay(self) -> None:
        """Test Bayesian probability with fractional decay factor."""
        result = bayesian_probability(
            prior=0.5,
            prob_given_true=0.8,
            prob_given_false=0.2,
            evidence=False,
            decay_factor=0.5,
        )
        assert 0.0 <= result <= 1.0


class TestOverallProbability:
    """Test overall_probability function."""

    def test_single_entity(self) -> None:
        """Test probability calculation with a single entity."""
        # Create a mock entity with known probabilities
        mock_entity = Mock()
        mock_entity.evidence = True
        mock_entity.decay.is_decaying = False
        mock_entity.type.weight = 1.0
        mock_entity.likelihood.prob_given_true = 0.8
        mock_entity.likelihood.prob_given_false = 0.1

        entities = {"test_entity": cast("Entity", mock_entity)}
        prior = 0.3
        expected = bayesian_probability(
            prior=prior,
            prob_given_true=0.8,
            prob_given_false=0.1,
            evidence=True,
            decay_factor=1.0,
        )

        result = overall_probability(entities, prior)
        assert abs(result - expected) < 0.001

    def test_multiple_entities(self) -> None:
        """Test probability calculation with multiple entities."""
        # Create mock entities with different probabilities
        mock_entity1 = Mock()
        mock_entity1.evidence = True
        mock_entity1.decay.is_decaying = False
        mock_entity1.type.weight = 0.8
        mock_entity1.likelihood.prob_given_true = 0.8
        mock_entity1.likelihood.prob_given_false = 0.1

        mock_entity2 = Mock()
        mock_entity2.evidence = False
        mock_entity2.decay.is_decaying = False
        mock_entity2.type.weight = 0.6
        mock_entity2.likelihood.prob_given_true = 0.7
        mock_entity2.likelihood.prob_given_false = 0.2

        entities = {
            "entity1": cast("Entity", mock_entity1),
            "entity2": cast("Entity", mock_entity2),
        }
        prior = 0.3

        expected = bayesian_probability(
            prior=prior,
            prob_given_true=0.8,
            prob_given_false=0.1,
            evidence=True,
            decay_factor=1.0,
        )

        result = overall_probability(entities, prior)
        assert abs(result - expected) < 0.001

    def test_decaying_entity(self) -> None:
        """Test probability calculation with a decaying entity."""
        mock_entity = Mock()
        mock_entity.evidence = False
        mock_entity.decay.is_decaying = True
        mock_entity.decay.decay_factor = 0.5  # Half decay
        mock_entity.type.weight = 1.0
        mock_entity.likelihood.prob_given_true = 0.8
        mock_entity.likelihood.prob_given_false = 0.1

        entities = {"test_entity": cast("Entity", mock_entity)}
        prior = 0.3
        expected = bayesian_probability(
            prior=prior,
            prob_given_true=0.8,
            prob_given_false=0.1,
            evidence=True,
            decay_factor=0.5,
        )

        result = overall_probability(entities, prior)
        assert abs(result - expected) < 0.001

    def test_no_entities(self) -> None:
        """Test probability calculation with no entities."""
        entities = {}
        prior = 0.3

        # With no entities, should return the prior unchanged
        result = overall_probability(entities, prior)
        assert result == prior

    def test_inactive_sensor_ignored(self) -> None:
        """Ensure inactive sensors do not influence the result."""
        mock_entity = Mock()
        mock_entity.evidence = False
        mock_entity.decay.is_decaying = False
        mock_entity.type.weight = 1.0
        mock_entity.likelihood.prob_given_true = 0.8
        mock_entity.likelihood.prob_given_false = 0.1

        entities = {"sensor": cast("Entity", mock_entity)}
        prior = 0.3

        result = overall_probability(entities, prior)
        assert result == prior

    def test_mixed_states(self) -> None:
        """Test probability calculation with mixed entity states."""
        # Create entities with different states
        mock_active = Mock()
        mock_active.evidence = True
        mock_active.decay.is_decaying = False
        mock_active.type.weight = 1.0
        mock_active.likelihood.prob_given_true = 0.8
        mock_active.likelihood.prob_given_false = 0.1

        mock_inactive = Mock()
        mock_inactive.evidence = False
        mock_inactive.decay.is_decaying = False
        mock_inactive.type.weight = 1.0
        mock_inactive.likelihood.prob_given_true = 0.8
        mock_inactive.likelihood.prob_given_false = 0.1

        mock_decaying = Mock()
        mock_decaying.evidence = False
        mock_decaying.decay.is_decaying = True
        mock_decaying.decay.decay_factor = 0.5
        mock_decaying.type.weight = 1.0
        mock_decaying.likelihood.prob_given_true = 0.8
        mock_decaying.likelihood.prob_given_false = 0.1

        entities = {
            "active": cast("Entity", mock_active),
            "inactive": cast("Entity", mock_inactive),
            "decaying": cast("Entity", mock_decaying),
        }
        prior = 0.3

        result = overall_probability(entities, prior)
        assert 0.0 <= result <= 1.0


class TestStatesToIntervals:
    """Test the states_to_intervals helper."""

    @pytest.mark.asyncio
    async def test_intervals_cover_full_range(self) -> None:
        """Intervals should span start to end even if first change is later."""
        start = dt_util.utcnow() - timedelta(minutes=30)
        end = dt_util.utcnow()

        states = [
            State(
                "binary_sensor.test", "off", last_changed=start - timedelta(minutes=5)
            ),
            State(
                "binary_sensor.test", "on", last_changed=start + timedelta(minutes=10)
            ),
            State(
                "binary_sensor.test", "off", last_changed=start + timedelta(minutes=20)
            ),
        ]

        intervals = await states_to_intervals(states, start, end)

        assert intervals[0]["start"] == start
        assert intervals[-1]["end"] == end
        assert intervals[0]["state"] == "off"


class TestExtractEntityKeywords:
    """Test _extract_entity_keywords function."""

    def test_basic_extraction(self) -> None:
        """Test basic keyword extraction from entity IDs."""
        assert _extract_entity_keywords("binary_sensor.motion_sensor") == [
            "motion",
            "sensor",
        ]
        assert _extract_entity_keywords("sensor.temperature_living_room") == [
            "temperature",
            "living",
            "room",
        ]

    def test_preserve_alphanumeric(self) -> None:
        """Test that alphanumeric combinations are preserved."""
        assert _extract_entity_keywords("sensor.co2_level") == ["co2", "level"]
        assert _extract_entity_keywords("sensor.pm25_sensor") == ["pm25", "sensor"]
        assert _extract_entity_keywords("binary_sensor.window_left_contact") == [
            "window",
            "left",
            "contact",
        ]

    def test_filter_short_words(self) -> None:
        """Test that short words and numbers are filtered."""
        assert _extract_entity_keywords("sensor.a_b_temperature_1_2") == ["temperature"]

    def test_handle_special_characters(self) -> None:
        """Test handling of various separators."""
        assert _extract_entity_keywords("sensor.temp-humidity_test") == [
            "temp",
            "humidity",
            "test",
        ]


class TestValidateAndCorrectDeviceClass:
    """Test _validate_and_correct_device_class function."""

    @patch("custom_components.area_occupancy.utils.BinarySensorDeviceClass")
    def test_door_to_window_correction(self, mock_binary_class) -> None:
        """Test correction of door device class to window when entity contains 'window'."""
        # Mock the enum values
        mock_binary_class.DOOR.value = "door"
        mock_binary_class.WINDOW.value = "window"

        # Mock _get_valid_device_classes_for_domain to return both classes
        with patch(
            "custom_components.area_occupancy.utils._get_valid_device_classes_for_domain",
            return_value={"door", "window"},
        ):
            result = _validate_and_correct_device_class(
                "binary_sensor.window_left_contact", "door", "binary_sensor"
            )
            assert result == "window"

    @patch("custom_components.area_occupancy.utils.BinarySensorDeviceClass")
    def test_window_to_door_correction(self, mock_binary_class) -> None:
        """Test correction of window device class to door when entity contains 'door'."""
        mock_binary_class.DOOR.value = "door"
        mock_binary_class.WINDOW.value = "window"

        with patch(
            "custom_components.area_occupancy.utils._get_valid_device_classes_for_domain",
            return_value={"door", "window"},
        ):
            result = _validate_and_correct_device_class(
                "binary_sensor.front_door_contact", "window", "binary_sensor"
            )
            assert result == "door"

    def test_invalid_device_class(self) -> None:
        """Test handling of invalid device class."""
        with patch(
            "custom_components.area_occupancy.utils._get_valid_device_classes_for_domain",
            return_value=set(),
        ):
            result = _validate_and_correct_device_class(
                "binary_sensor.test", "invalid", "binary_sensor"
            )
            assert result is None


class TestFuzzyMatchBinarySensor:
    """Test _fuzzy_match_binary_sensor function."""

    @patch("custom_components.area_occupancy.utils.BinarySensorDeviceClass")
    def test_exact_matches(self, mock_binary_class) -> None:
        """Test exact keyword matches for binary sensors."""
        mock_binary_class.MOTION.value = "motion"
        mock_binary_class.DOOR.value = "door"
        mock_binary_class.WINDOW.value = "window"

        # Test exact matches
        assert _fuzzy_match_binary_sensor(["motion", "sensor"], "test") == "motion"
        assert _fuzzy_match_binary_sensor(["door", "contact"], "test") == "door"
        assert _fuzzy_match_binary_sensor(["window", "sensor"], "test") == "window"

    @patch("custom_components.area_occupancy.utils.BinarySensorDeviceClass")
    def test_fuzzy_matches(self, mock_binary_class) -> None:
        """Test fuzzy keyword matches for binary sensors."""
        mock_binary_class.MOTION.value = "motion"
        mock_binary_class.MOISTURE.value = "moisture"
        mock_binary_class.CONNECTIVITY.value = "connectivity"

        # Test fuzzy matches
        assert _fuzzy_match_binary_sensor(["pir"], "test") == "motion"
        assert _fuzzy_match_binary_sensor(["leak"], "test") == "moisture"
        assert _fuzzy_match_binary_sensor(["online"], "test") == "connectivity"

    @patch("custom_components.area_occupancy.utils.BinarySensorDeviceClass")
    def test_contact_sensor_disambiguation(self, mock_binary_class) -> None:
        """Test contact sensor disambiguation based on context."""
        mock_binary_class.WINDOW.value = "window"
        mock_binary_class.DOOR.value = "door"

        # Window context
        assert _fuzzy_match_binary_sensor(["contact", "window"], "test") == "window"
        # Door context
        assert _fuzzy_match_binary_sensor(["contact", "door"], "test") == "door"
        # Default to door
        assert _fuzzy_match_binary_sensor(["contact"], "test") == "door"

    def test_no_enum_available(self) -> None:
        """Test behavior when BinarySensorDeviceClass is not available."""
        with patch(
            "custom_components.area_occupancy.utils.BinarySensorDeviceClass", None
        ):
            result = _fuzzy_match_binary_sensor(["motion"], "test")
            assert result is None


class TestFuzzyMatchSensor:
    """Test _fuzzy_match_sensor function."""

    @patch("custom_components.area_occupancy.utils.SensorDeviceClass")
    def test_unit_based_matching(self, mock_sensor_class) -> None:
        """Test unit-based device class matching for sensors."""
        mock_sensor_class.TEMPERATURE.value = "temperature"
        mock_sensor_class.HUMIDITY.value = "humidity"
        mock_sensor_class.ILLUMINANCE.value = "illuminance"

        # Temperature units
        assert _fuzzy_match_sensor([], "test", "°C") == "temperature"
        assert _fuzzy_match_sensor([], "test", "°F") == "temperature"

        # Illuminance units
        assert _fuzzy_match_sensor([], "test", "lx") == "illuminance"

    @patch("custom_components.area_occupancy.utils.SensorDeviceClass")
    def test_keyword_based_matching(self, mock_sensor_class) -> None:
        """Test keyword-based device class matching for sensors."""
        mock_sensor_class.CO2.value = "co2"
        mock_sensor_class.PM25.value = "pm25"
        mock_sensor_class.POWER.value = "power"

        # Keyword matches
        assert _fuzzy_match_sensor(["co2"], "test", None) == "co2"
        assert _fuzzy_match_sensor(["pm25"], "test", None) == "pm25"
        assert _fuzzy_match_sensor(["power"], "test", None) == "power"

    @patch("custom_components.area_occupancy.utils.SensorDeviceClass")
    def test_context_based_matching(self, mock_sensor_class) -> None:
        """Test context-based disambiguation for ambiguous units."""
        mock_sensor_class.HUMIDITY.value = "humidity"
        mock_sensor_class.BATTERY.value = "battery"
        mock_sensor_class.CO2.value = "co2"

        # Percentage with context
        assert _fuzzy_match_sensor(["humidity"], "test", "%") == "humidity"
        assert _fuzzy_match_sensor(["battery"], "test", "%") == "battery"

        # PPM with context
        assert _fuzzy_match_sensor(["co2"], "test", "ppm") == "co2"

    def test_no_enum_available(self) -> None:
        """Test behavior when SensorDeviceClass is not available."""
        with patch("custom_components.area_occupancy.utils.SensorDeviceClass", None):
            result = _fuzzy_match_sensor(["temperature"], "test", "°C")
            assert result is None


class TestFuzzyMatchMediaPlayer:
    """Test _fuzzy_match_media_player function."""

    @patch("custom_components.area_occupancy.utils.MediaPlayerDeviceClass")
    def test_exact_matches(self, mock_media_class) -> None:
        """Test exact keyword matches for media players."""
        mock_media_class.TV.value = "tv"
        mock_media_class.SPEAKER.value = "speaker"
        mock_media_class.RECEIVER.value = "receiver"

        # Test exact matches
        assert _fuzzy_match_media_player(["tv"], "test") == "tv"
        assert _fuzzy_match_media_player(["speaker"], "test") == "speaker"
        assert _fuzzy_match_media_player(["receiver"], "test") == "receiver"

    @patch("custom_components.area_occupancy.utils.MediaPlayerDeviceClass")
    def test_brand_recognition(self, mock_media_class) -> None:
        """Test brand-based device class recognition."""
        mock_media_class.TV.value = "tv"
        mock_media_class.SPEAKER.value = "speaker"
        mock_media_class.RECEIVER.value = "receiver"

        # TV brands
        assert _fuzzy_match_media_player(["samsung"], "test") == "tv"
        assert _fuzzy_match_media_player(["roku"], "test") == "tv"

        # Speaker brands
        assert _fuzzy_match_media_player(["sonos"], "test") == "speaker"
        assert _fuzzy_match_media_player(["echo"], "test") == "speaker"

        # Receiver brands
        assert _fuzzy_match_media_player(["denon"], "test") == "receiver"
        assert _fuzzy_match_media_player(["yamaha"], "test") == "receiver"

    @patch("custom_components.area_occupancy.utils.MediaPlayerDeviceClass")
    def test_compound_detection(self, mock_media_class) -> None:
        """Test compound word detection for media players."""
        mock_media_class.TV.value = "tv"
        mock_media_class.SPEAKER.value = "speaker"

        # Apple TV needs both words
        assert _fuzzy_match_media_player(["apple", "tv"], "test") == "tv"
        assert (
            _fuzzy_match_media_player(["apple"], "test") == "tv"
        )  # Still detected via fuzzy

        # Google Home
        assert _fuzzy_match_media_player(["google", "home"], "test") == "speaker"

    @patch("custom_components.area_occupancy.utils.MediaPlayerDeviceClass")
    def test_streaming_service_fallback(self, mock_media_class) -> None:
        """Test streaming service fallback to speaker."""
        mock_media_class.SPEAKER.value = "speaker"

        # Streaming services default to speaker
        assert _fuzzy_match_media_player(["spotify"], "test") == "speaker"
        assert _fuzzy_match_media_player(["pandora"], "test") == "speaker"

    def test_no_enum_available(self) -> None:
        """Test behavior when MediaPlayerDeviceClass is not available."""
        with patch(
            "custom_components.area_occupancy.utils.MediaPlayerDeviceClass", None
        ):
            result = _fuzzy_match_media_player(["tv"], "test")
            assert result is None


class TestDetectDeviceClass:
    """Test detect_device_class function."""

    @patch("custom_components.area_occupancy.utils.get_device_class")
    @patch("custom_components.area_occupancy.utils.get_unit_of_measurement")
    def test_validation_and_correction(self, mock_unit, mock_device_class) -> None:
        """Test device class validation and correction."""
        mock_hass = Mock()
        mock_device_class.return_value = "door"
        mock_unit.return_value = None

        with patch(
            "custom_components.area_occupancy.utils._validate_and_correct_device_class",
            return_value="window",
        ):
            result = detect_device_class(
                mock_hass, "binary_sensor.window_contact", "door", None
            )
            assert result == "window"

    @patch("custom_components.area_occupancy.utils.get_device_class")
    @patch("custom_components.area_occupancy.utils.get_unit_of_measurement")
    def test_fuzzy_matching_fallback(self, mock_unit, mock_device_class) -> None:
        """Test fuzzy matching when validation fails."""
        mock_hass = Mock()
        mock_device_class.return_value = None
        mock_unit.return_value = None

        with (
            patch(
                "custom_components.area_occupancy.utils._validate_and_correct_device_class",
                return_value=None,
            ),
            patch(
                "custom_components.area_occupancy.utils._fuzzy_match_device_class",
                return_value="motion",
            ),
        ):
            result = detect_device_class(
                mock_hass, "binary_sensor.motion_sensor", None, None
            )
            assert result == "motion"


class TestGetDeviceClassWithFallback:
    """Test get_device_class_with_fallback function."""

    @patch("custom_components.area_occupancy.utils.get_device_class")
    @patch("custom_components.area_occupancy.utils.get_unit_of_measurement")
    @patch("custom_components.area_occupancy.utils.detect_device_class")
    def test_intelligent_detection_priority(
        self, mock_detect, mock_unit, mock_device_class
    ) -> None:
        """Test that intelligent detection takes priority."""
        mock_hass = Mock()
        mock_device_class.return_value = "door"
        mock_unit.return_value = None
        mock_detect.return_value = "window"

        result = get_device_class_with_fallback(
            mock_hass, "binary_sensor.window_contact"
        )
        assert result == "window"

    @patch("custom_components.area_occupancy.utils.get_device_class")
    @patch("custom_components.area_occupancy.utils.get_unit_of_measurement")
    @patch("custom_components.area_occupancy.utils.detect_device_class")
    def test_fallback_to_ha_class(
        self, mock_detect, mock_unit, mock_device_class
    ) -> None:
        """Test fallback to HA device class when detection fails."""
        mock_hass = Mock()
        mock_device_class.return_value = "motion"
        mock_unit.return_value = None
        mock_detect.return_value = None

        result = get_device_class_with_fallback(mock_hass, "binary_sensor.test")
        assert result == "motion"

    @patch("custom_components.area_occupancy.utils.get_device_class")
    @patch("custom_components.area_occupancy.utils.get_unit_of_measurement")
    @patch("custom_components.area_occupancy.utils.detect_device_class")
    def test_unknown_fallback(self, mock_detect, mock_unit, mock_device_class) -> None:
        """Test fallback to 'Unknown' when all detection fails."""
        mock_hass = Mock()
        mock_device_class.return_value = None
        mock_unit.return_value = None
        mock_detect.return_value = None

        result = get_device_class_with_fallback(mock_hass, "binary_sensor.test")
        assert result == "Unknown"


class TestDetectEntityTypeFromDeviceClass:
    """Test detect_entity_type_from_device_class function."""

    def test_motion_mapping(self) -> None:
        """Test motion sensor device class mappings."""
        from custom_components.area_occupancy.data.entity_type import InputType

        assert detect_entity_type_from_device_class("motion") == InputType.MOTION
        assert detect_entity_type_from_device_class("occupancy") == InputType.MOTION
        assert detect_entity_type_from_device_class("presence") == InputType.MOTION

    def test_door_window_mapping(self) -> None:
        """Test door and window device class mappings."""
        from custom_components.area_occupancy.data.entity_type import InputType

        assert detect_entity_type_from_device_class("door") == InputType.DOOR
        assert detect_entity_type_from_device_class("window") == InputType.WINDOW

    def test_environmental_mapping(self) -> None:
        """Test environmental sensor device class mappings."""
        from custom_components.area_occupancy.data.entity_type import InputType

        assert (
            detect_entity_type_from_device_class("temperature")
            == InputType.ENVIRONMENTAL
        )
        assert (
            detect_entity_type_from_device_class("humidity") == InputType.ENVIRONMENTAL
        )
        assert detect_entity_type_from_device_class("co2") == InputType.ENVIRONMENTAL

    def test_media_mapping(self) -> None:
        """Test media player device class mappings."""
        from custom_components.area_occupancy.data.entity_type import InputType

        assert detect_entity_type_from_device_class("tv") == InputType.MEDIA
        assert detect_entity_type_from_device_class("speaker") == InputType.MEDIA
        assert detect_entity_type_from_device_class("receiver") == InputType.MEDIA

    def test_appliance_mapping(self) -> None:
        """Test appliance device class mappings."""
        from custom_components.area_occupancy.data.entity_type import InputType

        assert detect_entity_type_from_device_class("power") == InputType.APPLIANCE
        assert detect_entity_type_from_device_class("energy") == InputType.APPLIANCE
        assert detect_entity_type_from_device_class("running") == InputType.APPLIANCE

    def test_none_handling(self) -> None:
        """Test handling of None and unknown device classes."""
        assert detect_entity_type_from_device_class(None) is None
        assert detect_entity_type_from_device_class("unknown") is None
        assert detect_entity_type_from_device_class("invalid") is None


class TestGetEntityTypeDescription:
    """Test get_entity_type_description function."""

    @patch("custom_components.area_occupancy.utils.get_device_class")
    @patch("custom_components.area_occupancy.utils.get_unit_of_measurement")
    @patch("custom_components.area_occupancy.utils.get_device_class_with_fallback")
    @patch(
        "custom_components.area_occupancy.utils.detect_entity_type_from_device_class"
    )
    def test_complete_description(
        self, mock_detect_input, mock_fallback, mock_unit, mock_device_class
    ) -> None:
        """Test complete entity type description generation."""
        from custom_components.area_occupancy.data.entity_type import InputType

        mock_hass = Mock()
        mock_state = Mock()
        mock_state.name = "Motion Sensor"
        mock_state.state = "on"
        mock_hass.states.get.return_value = mock_state

        mock_device_class.return_value = "motion"
        mock_unit.return_value = None
        mock_fallback.return_value = "motion"
        mock_detect_input.return_value = InputType.MOTION

        result = get_entity_type_description(mock_hass, "binary_sensor.motion_sensor")

        assert result["entity_id"] == "binary_sensor.motion_sensor"
        assert result["domain"] == "binary_sensor"
        assert result["friendly_name"] == "Motion Sensor"
        assert result["ha_device_class"] == "motion"
        assert result["fallback_device_class"] == "motion"
        assert result["detected_input_type"] == "motion"
        assert result["current_state"] == "on"
        assert result["available"] is True

    @patch("custom_components.area_occupancy.utils.get_device_class")
    @patch("custom_components.area_occupancy.utils.get_unit_of_measurement")
    @patch("custom_components.area_occupancy.utils.get_device_class_with_fallback")
    @patch(
        "custom_components.area_occupancy.utils.detect_entity_type_from_device_class"
    )
    def test_unavailable_entity(
        self, mock_detect_input, mock_fallback, mock_unit, mock_device_class
    ) -> None:
        """Test description for unavailable entity."""
        mock_hass = Mock()
        mock_hass.states.get.return_value = None

        mock_device_class.return_value = None
        mock_unit.return_value = None
        mock_fallback.return_value = "Unknown"
        mock_detect_input.return_value = None

        result = get_entity_type_description(mock_hass, "binary_sensor.missing")

        assert result["friendly_name"] == "Unknown"
        assert result["current_state"] == "Unknown"
        assert result["available"] is False
