"""Tests for the hass-free configuration helpers.

The adjacency list transforms are exercised end-to-end in
``test_config_flow.py``; this module covers the validators and the pieces
that are new or were previously only reachable through the flow.
"""

from __future__ import annotations

import pytest
import voluptuous as vol

from custom_components.area_occupancy.config_helpers import (
    DECAY_HALF_LIFE_MAX,
    DECAY_HALF_LIFE_MIN,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    WEIGHT_KEYS,
    WEIGHT_MAX,
    apply_purpose_based_decay_default,
    duration_to_seconds,
    seconds_to_duration,
    validate_area_config,
    validate_decay_half_life,
    validate_person_input,
    validate_threshold,
)
from custom_components.area_occupancy.const import (
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
    CONF_WEIGHT_MOTION,
    CONF_WINDOW_ACTIVE_STATE,
    CONF_WINDOW_SENSORS,
    DEFAULT_WEIGHT_MOTION,
)
from custom_components.area_occupancy.data.purpose import (
    PURPOSE_DEFINITIONS,
    AreaPurpose,
)


def _valid_area() -> dict:
    """Return the smallest configuration that passes validation."""
    return {
        CONF_AREA_ID: "living_room",
        CONF_PURPOSE: "social",
        CONF_MOTION_SENSORS: ["binary_sensor.motion"],
    }


class TestDurationConversion:
    """seconds_to_duration / duration_to_seconds."""

    @pytest.mark.parametrize("seconds", [0, 59, 60, 3599, 3600, 90061])
    def test_round_trip(self, seconds: int) -> None:
        assert duration_to_seconds(seconds_to_duration(seconds)) == seconds

    def test_seconds_to_duration_splits_units(self) -> None:
        assert seconds_to_duration(90061) == {
            "days": 1,
            "hours": 1,
            "minutes": 1,
            "seconds": 1,
        }

    def test_duration_to_seconds_accepts_raw_numbers(self) -> None:
        assert duration_to_seconds(42) == 42
        assert duration_to_seconds(42.9) == 42

    def test_duration_to_seconds_tolerates_missing_keys(self) -> None:
        assert duration_to_seconds({"minutes": 2}) == 120


class TestValidateThreshold:
    """validate_threshold is the single rule for every threshold writer."""

    @pytest.mark.parametrize("value", [THRESHOLD_MIN, 50, 99.5, THRESHOLD_MAX])
    def test_accepts_in_range(self, value: float) -> None:
        assert validate_threshold(value) is None

    @pytest.mark.parametrize(
        "value",
        [THRESHOLD_MIN - 0.1, 0, -5, THRESHOLD_MAX + 0.1, "50", None, True],
    )
    def test_rejects_out_of_range_and_non_numeric(self, value: object) -> None:
        assert validate_threshold(value) == "invalid_threshold"


class TestValidateDecayHalfLife:
    """validate_decay_half_life honours the purpose-default sentinel."""

    @pytest.mark.parametrize(
        "value", [0, DECAY_HALF_LIFE_MIN, 600, DECAY_HALF_LIFE_MAX]
    )
    def test_accepts_sentinel_and_range(self, value: int) -> None:
        assert validate_decay_half_life(value) is None

    @pytest.mark.parametrize(
        "value",
        [DECAY_HALF_LIFE_MIN - 1, DECAY_HALF_LIFE_MAX + 1, -1, "600", None, False],
    )
    def test_rejects_out_of_range_and_non_numeric(self, value: object) -> None:
        assert validate_decay_half_life(value) == "invalid_decay_half_life"


class TestValidateAreaConfig:
    """validate_area_config covers every hass-free rule."""

    def test_minimal_valid_config_has_no_errors(self) -> None:
        assert validate_area_config(_valid_area()) == {}

    def test_missing_area_id(self) -> None:
        config = _valid_area()
        config[CONF_AREA_ID] = ""
        assert validate_area_config(config)[CONF_AREA_ID] == "area_required"

    def test_missing_purpose(self) -> None:
        config = _valid_area()
        config[CONF_PURPOSE] = ""
        assert validate_area_config(config)[CONF_PURPOSE] == "purpose_required"

    def test_motion_sensor_required(self) -> None:
        config = _valid_area()
        config[CONF_MOTION_SENSORS] = []
        assert validate_area_config(config)["base"] == "motion_required"

    def test_motion_likelihood_ordering(self) -> None:
        config = _valid_area()
        config[CONF_MOTION_PROB_GIVEN_TRUE] = 0.2
        config[CONF_MOTION_PROB_GIVEN_FALSE] = 0.3
        assert validate_area_config(config)["base"] == "prob_true_must_exceed_false"

    def test_motion_required_wins_over_likelihood_error_on_base(self) -> None:
        config = _valid_area()
        config[CONF_MOTION_SENSORS] = []
        config[CONF_MOTION_PROB_GIVEN_TRUE] = 0.2
        config[CONF_MOTION_PROB_GIVEN_FALSE] = 0.3
        assert validate_area_config(config)["base"] == "motion_required"

    @pytest.mark.parametrize("threshold", [0, 101, "50"])
    def test_invalid_threshold(self, threshold: object) -> None:
        config = _valid_area()
        config[CONF_THRESHOLD] = threshold
        assert validate_area_config(config)[CONF_THRESHOLD] == "invalid_threshold"

    def test_threshold_absent_is_fine(self) -> None:
        config = _valid_area()
        config.pop(CONF_THRESHOLD, None)
        assert CONF_THRESHOLD not in validate_area_config(config)

    @pytest.mark.parametrize(
        ("entities_key", "state_key", "error"),
        [
            (CONF_MEDIA_DEVICES, CONF_MEDIA_ACTIVE_STATES, "media_states_required"),
            (
                CONF_APPLIANCES,
                CONF_APPLIANCE_ACTIVE_STATES,
                "appliance_states_required",
            ),
            (CONF_DOOR_SENSORS, CONF_DOOR_ACTIVE_STATE, "door_state_required"),
            (CONF_LOCK_SENSORS, CONF_LOCK_ACTIVE_STATE, "lock_state_required"),
            (CONF_WINDOW_SENSORS, CONF_WINDOW_ACTIVE_STATE, "window_state_required"),
            (CONF_COVER_SENSORS, CONF_COVER_ACTIVE_STATES, "cover_states_required"),
        ],
    )
    def test_entities_without_active_state(
        self, entities_key: str, state_key: str, error: str
    ) -> None:
        config = _valid_area()
        config[entities_key] = ["some.entity"]
        config[state_key] = []
        assert validate_area_config(config)[entities_key] == error

    def test_entities_with_default_active_state_are_fine(self) -> None:
        config = _valid_area()
        config[CONF_DOOR_SENSORS] = ["binary_sensor.door"]
        assert validate_area_config(config) == {}

    @pytest.mark.parametrize("key", [key for key, _ in WEIGHT_KEYS])
    def test_weight_out_of_range(self, key: str) -> None:
        config = _valid_area()
        config[key] = WEIGHT_MAX + 0.5
        assert validate_area_config(config)[key] == "invalid_weight"

    def test_only_first_bad_weight_is_reported(self) -> None:
        config = _valid_area()
        for key, _ in WEIGHT_KEYS:
            config[key] = -1
        errors = validate_area_config(config)
        assert [k for k in errors if k.startswith("weight")] == [CONF_WEIGHT_MOTION]

    def test_weight_defaults_are_valid(self) -> None:
        config = _valid_area()
        config[CONF_WEIGHT_MOTION] = DEFAULT_WEIGHT_MOTION
        assert validate_area_config(config) == {}

    def test_decay_half_life_checked_only_when_decay_enabled(self) -> None:
        config = _valid_area()
        config[CONF_DECAY_ENABLED] = True
        config[CONF_DECAY_HALF_LIFE] = DECAY_HALF_LIFE_MAX + 1
        assert (
            validate_area_config(config)[CONF_DECAY_HALF_LIFE]
            == "invalid_decay_half_life"
        )

        config[CONF_DECAY_ENABLED] = False
        assert CONF_DECAY_HALF_LIFE not in validate_area_config(config)

    def test_decay_sentinel_accepted(self) -> None:
        config = _valid_area()
        config[CONF_DECAY_ENABLED] = True
        config[CONF_DECAY_HALF_LIFE] = 0
        assert validate_area_config(config) == {}


class TestApplyPurposeBasedDecayDefault:
    """The #439/#440 normalisation rule, scoped to the selected purpose."""

    def test_no_purpose_leaves_input_alone(self) -> None:
        data = {CONF_DECAY_HALF_LIFE: 600}
        apply_purpose_based_decay_default(data, None)
        assert data[CONF_DECAY_HALF_LIFE] == 600

    def test_missing_half_life_becomes_sentinel(self) -> None:
        data: dict = {}
        apply_purpose_based_decay_default(data, AreaPurpose.SOCIAL.value)
        assert data[CONF_DECAY_HALF_LIFE] == 0

    def test_selected_purpose_default_becomes_sentinel(self) -> None:
        purpose = AreaPurpose.WORKING
        data = {CONF_DECAY_HALF_LIFE: PURPOSE_DEFINITIONS[purpose].half_life}
        apply_purpose_based_decay_default(data, purpose.value)
        assert data[CONF_DECAY_HALF_LIFE] == 0

    def test_other_purposes_default_is_preserved(self) -> None:
        # A value that happens to equal ANOTHER purpose's default must stay.
        working = PURPOSE_DEFINITIONS[AreaPurpose.WORKING].half_life
        social = PURPOSE_DEFINITIONS[AreaPurpose.SOCIAL].half_life
        assert working != social
        data = {CONF_DECAY_HALF_LIFE: working}
        apply_purpose_based_decay_default(data, AreaPurpose.SOCIAL.value)
        assert data[CONF_DECAY_HALF_LIFE] == working


class TestValidatePersonInput:
    """validate_person_input normalises and clamps."""

    def _person(self) -> dict:
        return {
            CONF_PERSON_ENTITY: "person.seb",
            CONF_PERSON_SLEEP_SENSORS: ["sensor.sleep"],
            CONF_PERSON_SLEEP_AREA: "bedroom",
        }

    def test_valid_minimal(self) -> None:
        result = validate_person_input(self._person())
        assert result[CONF_PERSON_ENTITY] == "person.seb"
        assert CONF_PERSON_DEVICE_TRACKER not in result
        assert isinstance(result[CONF_PERSON_CONFIDENCE_THRESHOLD], int)

    def test_device_tracker_kept_when_set(self) -> None:
        data = self._person()
        data[CONF_PERSON_DEVICE_TRACKER] = "device_tracker.phone"
        result = validate_person_input(data)
        assert result[CONF_PERSON_DEVICE_TRACKER] == "device_tracker.phone"

    def test_confidence_is_clamped_and_coerced(self) -> None:
        data = self._person()
        data[CONF_PERSON_CONFIDENCE_THRESHOLD] = "250"
        assert validate_person_input(data)[CONF_PERSON_CONFIDENCE_THRESHOLD] == 100
        data[CONF_PERSON_CONFIDENCE_THRESHOLD] = -3
        assert validate_person_input(data)[CONF_PERSON_CONFIDENCE_THRESHOLD] == 1

    @pytest.mark.parametrize(
        ("missing", "error"),
        [
            (CONF_PERSON_ENTITY, "person_entity_required"),
            (CONF_PERSON_SLEEP_SENSORS, "sleep_sensor_required"),
            (CONF_PERSON_SLEEP_AREA, "sleep_area_required"),
        ],
    )
    def test_required_fields(self, missing: str, error: str) -> None:
        data = self._person()
        data[missing] = "" if missing != CONF_PERSON_SLEEP_SENSORS else []
        with pytest.raises(vol.Invalid, match=error):
            validate_person_input(data)

    def test_non_numeric_confidence(self) -> None:
        data = self._person()
        data[CONF_PERSON_CONFIDENCE_THRESHOLD] = "lots"
        with pytest.raises(vol.Invalid, match="confidence_not_number"):
            validate_person_input(data)
