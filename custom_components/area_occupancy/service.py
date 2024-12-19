import logging

from datetime import timedelta
import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    DEFAULT_HISTORY_PERIOD,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_services(hass: HomeAssistant):
    """Register custom services for area occupancy."""

    async def update_priors(call):
        """Manually trigger an update of learned priors."""
        try:
            entry_id = call.data["entry_id"]
            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]

            # Get history period from service call or use configured default
            history_period = call.data.get("history_period")
            if history_period:
                start_time = dt_util.utcnow() - timedelta(days=history_period)
            else:
                # Use configured history period
                start_time = dt_util.utcnow() - timedelta(
                    days=coordinator.config.get(
                        "history_period", DEFAULT_HISTORY_PERIOD
                    )
                )

            end_time = dt_util.utcnow()

            # Update priors for all configured sensors
            sensors = coordinator.get_configured_sensors()
            updated_count = 0

            for entity_id in sensors:
                try:
                    await coordinator.calculate_sensor_prior(
                        entity_id, start_time, end_time
                    )
                    updated_count += 1
                except (HomeAssistantError, ValueError, RuntimeError) as err:
                    _LOGGER.error(
                        "Error updating prior for sensor %s: %s", entity_id, err
                    )

            # Trigger a coordinator refresh
            await coordinator.async_refresh()

            message = (
                f"Updated priors for {updated_count} out of {len(sensors)} sensors"
            )
            _LOGGER.info(message)

        except KeyError as err:
            raise HomeAssistantError(
                f"Invalid entry_id or coordinator not found: {err}"
            ) from err
        except (HomeAssistantError, ValueError, RuntimeError) as err:
            raise HomeAssistantError(f"Failed to update priors: {err}") from err

    service_schema_update_priors = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional(
                "history_period",
                default=DEFAULT_HISTORY_PERIOD,
            ): vol.All(int, vol.Range(min=1, max=90)),
        }
    )

    hass.services.async_register(
        DOMAIN,
        "update_priors",
        update_priors,
        schema=service_schema_update_priors,
    )
