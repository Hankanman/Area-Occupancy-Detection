"""Tests for the types module."""

import pytest

from custom_components.area_occupancy.types import (
    EntityType,
    InstanceData,
    LoadedInstanceData,
    PriorData,
    PriorState,
    ProbabilityState,
    SensorCalculation,
    SensorInputs,
    StoredData,
    TypeAggregate,
)


# --- EntityType ---
def test_entity_type_enum():
    """Test that the EntityType enum values are correct."""
    assert EntityType.MOTION == "motion"
    assert EntityType.MEDIA == "media"
    assert EntityType.APPLIANCE == "appliance"
    assert EntityType.DOOR == "door"
    assert EntityType.WINDOW == "window"
    assert EntityType.LIGHT == "light"
    assert EntityType.ENVIRONMENTAL == "environmental"


# --- SensorCalculation ---
def test_sensor_calculation_empty():
    """Test that the SensorCalculation class returns an empty calculation."""
    sc = SensorCalculation.empty()
    assert sc.weighted_probability == 0.0
    assert sc.is_active is False
    assert sc.details["probability"] == 0.0


# --- ProbabilityState ---
def test_probability_state_valid_and_update():
    """Test that the ProbabilityState class updates correctly."""
    ps = ProbabilityState()
    assert ps.probability == 0.0
    ps.update(probability=0.5, threshold=0.7, is_occupied=True)
    assert ps.probability == 0.5
    assert ps.threshold == 0.7
    assert ps.is_occupied is True
    # active_triggers property
    ps.sensor_probabilities = {
        "eid": {"probability": 0.1, "weight": 1.0, "weighted_probability": 0.1}
    }
    assert ps.active_triggers == ["eid"]
    # to_dict/from_dict roundtrip
    d = ps.to_dict()
    ps2 = ProbabilityState.from_dict(d)
    assert ps2.probability == ps.probability
    assert ps2.threshold == ps.threshold
    assert ps2.is_occupied == ps.is_occupied


@pytest.mark.parametrize(
    "field,value",
    [
        ("probability", -0.1),
        ("probability", 1.1),
        ("previous_probability", -0.1),
        ("threshold", 1.1),
        ("prior_probability", -0.1),
        ("decay_status", 101),
    ],
)
def test_probability_state_invalid(field, value):
    """Test that the ProbabilityState class raises a ValueError for invalid values."""
    kwargs = {field: value}
    with pytest.raises(ValueError):
        ProbabilityState(**kwargs)


# --- PriorData ---
def test_prior_data_valid_and_to_from_dict():
    """Test that the PriorData class can be converted to a dictionary and back."""
    pd = PriorData(
        prior=0.5, prob_given_true=0.6, prob_given_false=0.2, last_updated="now"
    )
    d = pd.to_dict()
    pd2 = PriorData.from_dict(d)
    assert pd2.prior == 0.5
    assert pd2.prob_given_true == 0.6
    assert pd2.prob_given_false == 0.2
    assert pd2.last_updated == "now"


def test_prior_data_invalid():
    """Test that the PriorData class raises a ValueError for invalid values."""
    with pytest.raises(ValueError):
        PriorData(prior=-0.1)
    with pytest.raises(ValueError):
        PriorData(prior=0.5, prob_given_true=2.0)
    with pytest.raises(ValueError):
        PriorData(prior=0.5, prob_given_false=-1.0)


# --- PriorState ---
def test_prior_state_valid_and_update():
    """Test that the PriorState class updates correctly."""
    ps = PriorState()
    ps.update(overall_prior=0.5, motion_prior=0.6)
    assert ps.overall_prior == 0.5
    assert ps.motion_prior == 0.6
    # update_entity_prior
    ps.update_entity_prior("eid", 0.5, 0.2, 0.3, "ts")
    assert "eid" in ps.entity_priors
    # update_type_prior
    ps.update_type_prior("motion", 0.4, "ts", 0.5, 0.2)
    assert "motion" in ps.type_priors
    # to_dict/from_dict roundtrip
    d = ps.to_dict()
    ps2 = PriorState.from_dict(d)
    assert ps2.overall_prior == ps.overall_prior
    assert ps2.motion_prior == ps.motion_prior


@pytest.mark.parametrize(
    "field,value",
    [
        ("overall_prior", -0.1),
        ("motion_prior", 1.1),
        ("media_prior", -0.1),
    ],
)
def test_prior_state_invalid(field, value):
    """Test that the PriorState class raises a ValueError for invalid values."""
    kwargs = {field: value}
    with pytest.raises(ValueError):
        PriorState(**kwargs)


def test_prior_state_update_entity_prior_invalid():
    """Test that the PriorState class raises a ValueError for invalid values."""
    ps = PriorState()
    with pytest.raises(ValueError):
        ps.update_entity_prior("eid", 2.0, 0.2, 0.3, "ts")
    with pytest.raises(ValueError):
        ps.update_entity_prior("eid", 0.5, -1.0, 0.3, "ts")
    with pytest.raises(ValueError):
        ps.update_entity_prior("eid", 0.5, 0.2, -0.1, "ts")


def test_prior_state_update_type_prior_unknown_type():
    """Test that the PriorState class logs a warning when an unknown type is updated."""
    ps = PriorState()
    # Should not raise, just log warning
    ps.update_type_prior("unknown", 0.5, "ts")
    assert "unknown" in ps.type_priors


# --- SensorInputs ---
def test_sensor_inputs_is_valid_entity_id():
    """Test that the SensorInputs class can validate entity IDs."""
    assert SensorInputs.is_valid_entity_id("binary_sensor.foo") is True
    assert SensorInputs.is_valid_entity_id("badid") is False
    assert SensorInputs.is_valid_entity_id(str(123)) is False


def test_sensor_inputs_validate_entity():
    """Test that the SensorInputs class can validate entity IDs."""
    config = {"foo": "binary_sensor.bar"}
    assert SensorInputs.validate_entity("foo", config) == "binary_sensor.bar"
    with pytest.raises(ValueError):
        SensorInputs.validate_entity("foo", {"foo": "badid"})


def test_sensor_inputs_validate_entity_list():
    """Test that the SensorInputs class can validate entity lists."""
    config = {"foo": ["binary_sensor.bar", "binary_sensor.baz"]}
    assert SensorInputs.validate_entity_list("foo", config) == [
        "binary_sensor.bar",
        "binary_sensor.baz",
    ]
    with pytest.raises(TypeError):
        SensorInputs.validate_entity_list("foo", {"foo": "notalist"})
    with pytest.raises(ValueError):
        SensorInputs.validate_entity_list("foo", {"foo": ["badid"]})


def test_sensor_inputs_from_config_and_get_all_sensors():
    """Test that the SensorInputs class can be initialized from a configuration and return all sensors."""
    config = {
        "motion_sensors": ["binary_sensor.m1"],
        "primary_occupancy_sensor": "binary_sensor.m1",
        "media_devices": ["media_player.m1"],
        "appliances": ["switch.a1"],
        "illuminance_sensors": ["sensor.i1"],
        "humidity_sensors": ["sensor.h1"],
        "temperature_sensors": ["sensor.t1"],
        "door_sensors": ["binary_sensor.d1"],
        "window_sensors": ["binary_sensor.w1"],
        "lights": ["light.l1"],
    }
    si = SensorInputs.from_config(config)
    all_sensors = si.get_all_sensors()
    for eid in [
        "binary_sensor.m1",
        "media_player.m1",
        "switch.a1",
        "sensor.i1",
        "sensor.h1",
        "sensor.t1",
        "binary_sensor.d1",
        "binary_sensor.w1",
        "light.l1",
    ]:
        assert eid in all_sensors
    # primary_sensor should be included only once
    assert all_sensors.count("binary_sensor.m1") == 1


# --- TypeAggregate ---
def test_type_aggregate():
    """Test that the TypeAggregate class can be initialized and updated."""
    ta = TypeAggregate()
    ta.priors.append(0.5)
    ta.p_true.append(0.6)
    ta.p_false.append(0.2)
    ta.count += 1
    assert ta.count == 1
    assert ta.priors == [0.5]


# --- TypedDicts basic instantiation ---
def test_typed_dicts_instantiation():
    """Test that the TypedDicts can be instantiated."""
    instance_data = InstanceData(name="area", prior_state=None, last_updated="now")
    stored_data = StoredData(instances={"id": instance_data})
    loaded = LoadedInstanceData(name="area", prior_state=None, last_updated="now")
    assert instance_data["name"] == "area"
    assert "id" in stored_data["instances"]
    assert loaded.name == "area"
