"""Service implementations for Area Occupancy Detection."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

import voluptuous as vol

from homeassistant.const import ATTR_ENTITY_ID
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    ATTR_START_TIME,
    ATTR_END_TIME,
    ATTR_OUTPUT_FILE,
    ATTR_DAYS,
    DEFAULT_PRIOR_OUTPUT,
    DEFAULT_TIMESLOT_OUTPUT,
)
from .historical_analysis import HistoricalAnalysis

_LOGGER = logging.getLogger(__name__)

# Schema for calculate_prior service
CALCULATE_PRIOR_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_id,
        vol.Optional(ATTR_START_TIME): cv.datetime,
        vol.Optional(ATTR_END_TIME): cv.datetime,
        vol.Optional(ATTR_OUTPUT_FILE, default=DEFAULT_PRIOR_OUTPUT): cv.string,
    }
)

# Schema for calculate_timeslots service
CALCULATE_TIMESLOTS_SCHEMA = vol.Schema(
    {
        vol.Required(ATTR_ENTITY_ID): cv.entity_ids,
        vol.Optional(ATTR_DAYS, default=7): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=30)
        ),
        vol.Optional(ATTR_OUTPUT_FILE, default=DEFAULT_TIMESLOT_OUTPUT): cv.string,
    }
)


async def async_setup_services(hass: HomeAssistant) -> None:
    """Set up services for Area Occupancy integration."""

    async def calculate_prior(call: ServiceCall) -> None:
        """Service to calculate prior probabilities for an entity."""
        try:
            entity_id = call.data[ATTR_ENTITY_ID]

            # Get time range
            end_time = call.data.get(ATTR_END_TIME, dt_util.utcnow())
            start_time = call.data.get(ATTR_START_TIME, end_time - timedelta(days=7))

            # Validate entity belongs to an area occupancy instance
            entry_id = None
            for entry in hass.config_entries.async_entries(DOMAIN):
                coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
                if entity_id in coordinator.get_configured_sensors():
                    entry_id = entry.entry_id
                    break

            if not entry_id:
                raise HomeAssistantError(
                    f"Entity {entity_id} is not configured in any Area Occupancy instance"
                )

            # Create analyzer instance
            analyzer = HistoricalAnalysis(hass)

            # Calculate prior probabilities
            prob_true, prob_false = await analyzer.calculate_prior(
                entity_id, start_time, end_time
            )

            # Prepare output data
            output_data = {
                "entity_id": entity_id,
                "calculation_time": dt_util.utcnow().isoformat(),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "probability_given_true": prob_true,
                "probability_given_false": prob_false,
            }

            # Write to file
            output_file = call.data[ATTR_OUTPUT_FILE]
            await hass.async_add_executor_job(
                _write_json_file, hass.config.path(output_file), output_data
            )

            _LOGGER.info(
                "Prior probability calculation completed for %s: %s",
                entity_id,
                output_file,
            )

        except Exception as err:
            _LOGGER.error("Error calculating prior probabilities: %s", err)
            raise HomeAssistantError(
                f"Failed to calculate prior probabilities: {err}"
            ) from err

    async def calculate_timeslots(call: ServiceCall) -> None:
        """Service to calculate timeslot probabilities."""
        try:
            entity_ids = call.data[ATTR_ENTITY_ID]
            days = call.data[ATTR_DAYS]

            # Validate at least one entity belongs to an area occupancy instance
            entry_id = None
            for entry in hass.config_entries.async_entries(DOMAIN):
                coordinator = hass.data[DOMAIN][entry.entry_id]["coordinator"]
                if any(
                    entity_id in coordinator.get_configured_sensors()
                    for entity_id in entity_ids
                ):
                    entry_id = entry.entry_id
                    break

            if not entry_id:
                raise HomeAssistantError(
                    "None of the provided entities are configured in any Area Occupancy instance"
                )

            # Create analyzer instance
            analyzer = HistoricalAnalysis(hass)

            # Calculate timeslot data
            timeslots = await analyzer.calculate_timeslots(entity_ids, days)

            # Prepare output data with more detailed information
            output_data = {
                "entities": entity_ids,
                "calculation_time": dt_util.utcnow().isoformat(),
                "analysis_period_days": days,
                "timeslots": timeslots,
                "metadata": {
                    "slot_count": len(timeslots),
                    "time_periods": [
                        f"{hour:02d}:{minute:02d}"
                        for hour in range(24)
                        for minute in (0, 30)
                    ],
                },
            }

            # Write to file
            output_file = call.data[ATTR_OUTPUT_FILE]
            await hass.async_add_executor_job(
                _write_json_file, hass.config.path(output_file), output_data
            )

            _LOGGER.info(
                "Timeslot calculation completed for %s: %s",
                ", ".join(entity_ids),
                output_file,
            )

        except Exception as err:
            _LOGGER.error("Error calculating timeslots: %s", err)
            raise HomeAssistantError(f"Failed to calculate timeslots: {err}") from err

    def _write_json_file(file_path: str, data: dict[str, Any]) -> None:
        """Write data to JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError as err:
            raise HomeAssistantError(f"Failed to write output file: {err}") from err

    # Register services
    hass.services.async_register(
        DOMAIN,
        "calculate_prior",
        calculate_prior,
        schema=CALCULATE_PRIOR_SCHEMA,
    )

    hass.services.async_register(
        DOMAIN,
        "calculate_timeslots",
        calculate_timeslots,
        schema=CALCULATE_TIMESLOTS_SCHEMA,
    )


async def async_unload_services(hass: HomeAssistant) -> None:
    """Unload Area Occupancy services."""
    for service in ["calculate_prior", "calculate_timeslots"]:
        if hass.services.has_service(DOMAIN, service):
            hass.services.async_remove(DOMAIN, service)
