import json
import logging

from pathlib import Path
from datetime import timedelta
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util
from homeassistant.helpers import config_validation as cv

from .const import (
    DOMAIN,
    ATTR_START_TIME,
    ATTR_END_TIME,
    ATTR_OUTPUT_FILE,
)

from .probabilities import (
    MOTION_PROB_GIVEN_TRUE,
    MOTION_PROB_GIVEN_FALSE,
    MEDIA_PROB_GIVEN_TRUE,
    MEDIA_PROB_GIVEN_FALSE,
    APPLIANCE_PROB_GIVEN_TRUE,
    APPLIANCE_PROB_GIVEN_FALSE,
)


_LOGGER = logging.getLogger(__name__)


async def async_setup_services(hass: HomeAssistant):
    """Register custom services for area occupancy."""

    async def export_calculations(call):
        """Export probability calculations to a JSON file."""
        now = dt_util.utcnow()
        start_time = call.data.get(ATTR_START_TIME, now - timedelta(days=7))
        end_time = call.data.get(ATTR_END_TIME, now)
        output_file = call.data.get(ATTR_OUTPUT_FILE, "occupancy_calculations.json")

        try:
            entry_id = call.data["entry_id"]
            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]

            # Get the calculator instance
            calculator = coordinator._calculator
            sensor_states = coordinator._sensor_states
            motion_timestamps = coordinator._motion_timestamps

            # Calculate current probabilities
            result = await calculator.calculate(
                sensor_states,
                motion_timestamps,
                coordinator._timeslot_data.get("slots", {}),
            )

            # Get prior probabilities for each sensor
            sensor_priors = {}
            for entity_id in sensor_states:
                p_true, p_false = calculator.get_sensor_priors(entity_id)
                sensor_priors[entity_id] = {
                    "prob_given_true": round(p_true, 4),
                    "prob_given_false": round(p_false, 4),
                }

            # Prepare detailed calculation data
            export_data = {
                "calculation_time": now.isoformat(),
                "time_period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                },
                "current_probability": round(result["probability"], 4),
                "is_occupied": result["is_occupied"],
                "sensor_data": {
                    entity_id: {
                        "state": state["state"],
                        "current_probability": round(
                            result["sensor_probabilities"].get(entity_id, 0.0), 4
                        ),
                        "priors": sensor_priors.get(entity_id, {}),
                        "is_active": entity_id in result["active_triggers"],
                    }
                    for entity_id, state in sensor_states.items()
                },
                "active_triggers": result["active_triggers"],
                "device_states": result["device_states"],
                "decay_status": {
                    sensor: round(value, 4)
                    for sensor, value in result["decay_status"].items()
                },
            }

            # Calculate rolled up probabilities by type
            sensor_types = {
                "motion": {
                    "sensors": calculator.motion_sensors,
                    "priors": (MOTION_PROB_GIVEN_TRUE, MOTION_PROB_GIVEN_FALSE),
                },
                "media": {
                    "sensors": calculator.media_devices,
                    "priors": (MEDIA_PROB_GIVEN_TRUE, MEDIA_PROB_GIVEN_FALSE),
                },
                "appliances": {
                    "sensors": calculator.appliances,
                    "priors": (APPLIANCE_PROB_GIVEN_TRUE, APPLIANCE_PROB_GIVEN_FALSE),
                },
            }

            rolled_up_data = {}
            for sensor_type, config in sensor_types.items():
                active_sensors = [
                    s for s in config["sensors"] if s in result["active_triggers"]
                ]

                rolled_up_data[sensor_type] = {
                    "sensors": config["sensors"],
                    "active_sensors": active_sensors,
                    "active_count": len(active_sensors),
                    "total_count": len(config["sensors"]),
                    "prob_given_true": round(config["priors"][0], 4),
                    "prob_given_false": round(config["priors"][1], 4),
                    "current_probability": round(
                        max(
                            [
                                result["sensor_probabilities"].get(s, 0.0)
                                for s in config["sensors"]
                            ]
                            or [0.0]
                        ),
                        4,
                    ),
                }

            # Update export data with rolled up calculations
            export_data.update(
                {
                    "rolled_up_calculations": rolled_up_data,
                }
            )

            output_path = hass.config.path(output_file)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=4, default=str)

            _LOGGER.info("Successfully exported calculations to %s", output_path)

            hass.components.persistent_notification.create(
                f"Exported probability calculations to {output_file}",
                title="Area Occupancy",
            )
        except KeyError as err:
            raise HomeAssistantError(
                f"Invalid entry_id or coordinator not found: {err}"
            ) from err
        except (OSError, IOError) as err:
            raise HomeAssistantError(f"Failed to write file: {err}") from err
        except ValueError as err:
            raise HomeAssistantError(f"Invalid data format: {err}") from err

    service_schema_calculations = {
        vol.Required("entry_id"): str,
        vol.Optional(
            ATTR_START_TIME, default=lambda: dt_util.utcnow() - timedelta(days=7)
        ): cv.datetime,
        vol.Optional(ATTR_END_TIME, default=dt_util.utcnow): cv.datetime,
        vol.Optional(ATTR_OUTPUT_FILE, default="occupancy_calculations.json"): str,
    }

    hass.services.async_register(
        DOMAIN,
        "export_calculations",
        export_calculations,
        schema=vol.Schema(service_schema_calculations),
    )

    async def export_historical_analysis(call):
        """Export historical analysis to a JSON file."""
        days = call.data.get("days", 7)
        output_file = call.data.get(ATTR_OUTPUT_FILE, "historical_analysis.json")

        try:
            coordinator = hass.data[DOMAIN][call.data["entry_id"]]["coordinator"]
            calculator = coordinator._calculator

            # Get sensors from coordinator or fall back to empty list
            sensors = coordinator.get_configured_sensors() if coordinator else []
            if not sensors:
                raise HomeAssistantError(
                    "No sensors configured or coordinator not ready"
                )

            timeslots = await calculator.calculate_timeslots(
                sensors, history_period=days
            )

            # Use hass.config.path() to get the correct output path
            output_path = hass.config.path(output_file)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(timeslots, f, indent=4)

            _LOGGER.info("Successfully exported historical analysis to %s", output_path)

            hass.components.persistent_notification.create(
                f"Exported historical analysis to {output_file}", title="Area Occupancy"
            )
        except KeyError as err:
            raise HomeAssistantError(
                f"Invalid entry_id or coordinator not found: {err}"
            ) from err
        except (OSError, IOError) as err:
            raise HomeAssistantError(f"Failed to write file: {err}") from err
        except ValueError as err:
            raise HomeAssistantError(f"Invalid data format: {err}") from err

    service_schema_historical = {
        vol.Required("entry_id"): str,
        vol.Optional("days", default=7): vol.All(int, vol.Range(min=1, max=30)),
        vol.Optional(ATTR_OUTPUT_FILE, default="historical_analysis.json"): str,
    }

    hass.services.async_register(
        DOMAIN,
        "export_historical_analysis",
        export_historical_analysis,
        schema=vol.Schema(service_schema_historical),
    )
