"""Service definitions for the Area Occupancy Detection integration."""

import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import DEFAULT_HISTORY_PERIOD, DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_services(hass: HomeAssistant):
    """Register custom services for area occupancy."""

    async def update_priors(call):
        """Manually trigger an update of learned priors."""
        try:
            entry_id = call.data["entry_id"]
            _LOGGER.debug("Updating priors for entry_id: %s", entry_id)

            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]
            _LOGGER.debug("Found coordinator for entry_id")

            # Get history period from service call or use configured default
            history_period = call.data.get("history_period", DEFAULT_HISTORY_PERIOD)
            _LOGGER.debug(
                "Using history period: %s (type: %s)",
                history_period,
                type(history_period),
            )

            if history_period:
                _LOGGER.debug(
                    "Calling update_learned_priors with period: %s", history_period
                )
                try:
                    await coordinator.update_learned_priors(history_period)
                except Exception:
                    _LOGGER.exception("Error in update_learned_priors")
                    raise
            else:
                _LOGGER.debug("Calling update_learned_priors with default period")
                await coordinator.update_learned_priors()

            # Trigger a coordinator refresh
            _LOGGER.debug("Triggering coordinator refresh")
            await coordinator.async_refresh()
            _LOGGER.debug("Prior update completed successfully")

        except KeyError as err:
            _LOGGER.error("Invalid entry_id or coordinator not found: %s", err)
            raise HomeAssistantError(
                f"Invalid entry_id or coordinator not found: {err}"
            ) from err
        except (HomeAssistantError, ValueError, RuntimeError) as err:
            _LOGGER.error("Failed to update priors: %s (type: %s)", err, type(err))
            raise HomeAssistantError(f"Failed to update priors: {err}") from err
        except Exception as err:
            _LOGGER.exception("Unexpected error during prior update")
            raise HomeAssistantError(
                f"Unexpected error during prior update: {err}"
            ) from err

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
