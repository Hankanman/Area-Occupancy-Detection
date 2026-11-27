"""Tests for bounded auxiliary and environmental shadow evidence."""

from datetime import UTC, datetime, timedelta

from custom_components.area_occupancy.data.auxiliary_shadow import (
    AuxiliaryReading,
    AuxiliarySensorParams,
    AuxiliaryShadow,
    AuxiliaryShadowState,
)
from custom_components.area_occupancy.data.entity_type import InputType


def reading(
    value: float,
    input_type: InputType = InputType.CO2,
    *,
    unoccupied: bool = True,
    stuck: bool = False,
    lux_suppressed: bool = False,
) -> AuxiliaryReading:
    """Build a numeric shadow reading."""
    return AuxiliaryReading(
        value=value,
        input_type=input_type,
        confidently_unoccupied=unoccupied,
        stuck=stuck,
        lux_suppressed=lux_suppressed,
    )


def test_baseline_updates_only_during_unoccupied_periods() -> None:
    shadow = AuxiliaryShadow()
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    shadow.observe(
        start,
        readings={"co2": reading(400)},
    )
    assert shadow.state.sensors["co2"].mu == 400

    shadow.observe(
        start + timedelta(seconds=60),
        readings={"co2": reading(1000, unoccupied=False)},
    )
    assert shadow.state.sensors["co2"].mu == 400

    shadow.observe(
        start + timedelta(seconds=120),
        readings={"co2": reading(420)},
    )
    assert 400 <= shadow.state.sensors["co2"].mu <= 420


def test_ventilation_drop_is_never_negative_evidence() -> None:
    shadow = AuxiliaryShadow()
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    shadow.observe(start, readings={"co2": reading(1000)})
    shadow.observe(
        start + timedelta(seconds=60),
        readings={"co2": reading(500)},
    )

    snapshot = shadow.snapshot()["sensors"]["co2"]
    assert snapshot["z"] == 0
    assert snapshot["j"] == 0
    assert shadow.state.e_aux >= 0


def test_stuck_and_distribution_outlier_damp_quality_and_baseline() -> None:
    shadow = AuxiliaryShadow()
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)
    shadow.observe(start, readings={"co2": reading(400)})

    shadow.observe(
        start + timedelta(seconds=60),
        readings={"co2": reading(5000, stuck=True)},
    )
    sensor = shadow.state.sensors["co2"]

    assert sensor.last_snapshot["stuck"] is True
    assert sensor.last_snapshot["outlier"] is True
    assert sensor.last_snapshot["quality"] < 0.1
    assert sensor.mu == 400


def test_lux_jump_is_suppressed_during_lighting_change() -> None:
    shadow = AuxiliaryShadow()
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    shadow.observe(
        start,
        readings={"lux": reading(100, InputType.ILLUMINANCE)},
    )
    shadow.observe(
        start + timedelta(seconds=10),
        readings={
            "lux": reading(
                500,
                InputType.ILLUMINANCE,
                lux_suppressed=True,
            )
        },
    )

    sensor = shadow.state.sensors["lux"].last_snapshot
    assert sensor["lux_suppressed"] is True
    assert sensor["j"] == 0
    assert "lux_automation_suppressed" in sensor["reasons"]


def test_exponential_decay_and_missing_reading_decay() -> None:
    shadow = AuxiliaryShadow(
        params={
            "co2": AuxiliarySensorParams(
                role_weight=1,
                sigma_min_s=1,
                tau_baseline_s=100,
                max_jump_dt_s=600,
                tau_decay_s=10,
                w_base=1,
                level_weight=1,
                jump_weight=0,
            )
        }
    )
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    shadow.observe(start, readings={"co2": reading(100)})
    shadow.observe(
        start + timedelta(seconds=10),
        readings={"co2": reading(110)},
    )
    first = shadow.state.sensors["co2"].contribution
    assert first > 0

    shadow.observe(start + timedelta(seconds=20), readings={})
    decayed = shadow.state.sensors["co2"].contribution
    assert decayed == 0 or abs(decayed - first * 0.3678794412) < 0.01


def test_area_level_evidence_is_capped() -> None:
    params = AuxiliarySensorParams(
        role_weight=1,
        sigma_min_s=1,
        tau_baseline_s=100,
        tau_decay_s=100,
        w_base=1,
        level_weight=1,
        jump_weight=0,
    )
    shadow = AuxiliaryShadow(e_aux_max=0.5, params={"a": params, "b": params})
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    shadow.observe(
        start,
        readings={
            "a": reading(100, InputType.CO2),
            "b": reading(100, InputType.CO2),
        },
    )
    shadow.observe(
        start + timedelta(seconds=10),
        readings={
            "a": reading(200, InputType.CO2),
            "b": reading(200, InputType.CO2),
        },
    )

    assert shadow.state.e_aux <= 0.5
    assert shadow.snapshot()["E_aux"] <= 0.5


def test_repeated_value_is_damped_as_stuck_active() -> None:
    params = AuxiliarySensorParams(
        role_weight=1,
        sigma_min_s=1,
        tau_decay_s=100,
        w_base=1,
        repeated_limit=3,
        repeat_delta=0.001,
    )
    shadow = AuxiliaryShadow(params={"co2": params})
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    for seconds in range(4):
        shadow.observe(
            start + timedelta(seconds=seconds),
            readings={"co2": reading(400)},
        )

    snapshot = shadow.snapshot()["sensors"]["co2"]
    assert snapshot["repeated_stuck"] is True
    assert "repeated_stuck" in snapshot["reasons"]
    assert snapshot["validity"] == 0.5


def test_stale_reading_has_zero_freshness_and_no_evidence() -> None:
    params = AuxiliarySensorParams(
        role_weight=1,
        sigma_min_s=1,
        fresh_for_s=60,
        stale_after_s=120,
        w_base=1,
    )
    shadow = AuxiliaryShadow(params={"co2": params})
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    stale = AuxiliaryReading(
        value=500,
        input_type=InputType.CO2,
        confidently_unoccupied=True,
        timestamp=start - timedelta(seconds=180),
    )
    shadow.observe(start, readings={"co2": stale})
    snapshot = shadow.snapshot()["sensors"]["co2"]

    assert snapshot["freshness"] == 0
    assert "stale" in snapshot["reasons"]
    assert snapshot["E_s"] == 0


def test_sensor_contributions_are_summed_not_deduplicated_by_type() -> None:
    params = AuxiliarySensorParams(
        role_weight=1,
        sigma_min_s=1,
        tau_decay_s=100,
        w_base=1,
        level_weight=1,
        jump_weight=0,
    )
    shadow = AuxiliaryShadow(e_aux_max=10, params={"a": params, "b": params})
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)

    shadow.observe(
        start,
        readings={
            "a": reading(100, InputType.CO2),
            "b": reading(100, InputType.CO2),
        },
    )
    shadow.observe(
        start + timedelta(seconds=10),
        readings={
            "a": reading(200, InputType.CO2),
            "b": reading(200, InputType.CO2),
        },
    )

    sensors = shadow.snapshot()["sensors"]
    assert sensors["a"]["E_s"] > 0
    assert sensors["b"]["E_s"] > 0
    assert shadow.state.e_aux >= sensors["a"]["E_s"] + sensors["b"]["E_s"] - 1e-6


def test_shadow_state_round_trips_and_rejects_nan() -> None:
    shadow = AuxiliaryShadow()
    start = datetime(2026, 11, 21, 12, tzinfo=UTC)
    shadow.observe(start, readings={"co2": reading(400)})

    restored_state = AuxiliaryShadowState.from_dict(shadow.state.to_dict())
    restored = AuxiliaryShadow(restored_state)
    snapshot = restored.snapshot()

    assert snapshot["engaged"] is False
    assert snapshot["shadow_mode"] is True
    assert snapshot["sensors"]["co2"]["mu"] == 400

    malformed = AuxiliaryShadowState.from_dict(
        {
            "e_aux": "nan",
            "last_tick": "not-a-date",
            "sensors": {
                "co2": {
                    "mu": "inf",
                    "mad": "nan",
                    "contribution": "-inf",
                }
            },
        }
    )
    assert malformed.e_aux == 0
    assert malformed.last_tick is None
    assert malformed.sensors["co2"].mu is None
