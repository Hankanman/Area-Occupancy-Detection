"""Service definitions for the Area Occupancy Detection integration."""

import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD, DOMAIN
from .exceptions import CalculationError

_LOGGER = logging.getLogger(__name__)


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    async def update_priors(call):
        """Manually trigger an update of learned priors."""
        entry_id = call.data["entry_id"]

        def raise_error(msg: str, err: Exception | None = None) -> None:
            """Raise appropriate error with consistent format."""
            error_msg = f"{msg}: {err}" if err else msg
            _LOGGER.error(
                "Failed to update priors for instance %s: %s", entry_id, error_msg
            )
            raise HomeAssistantError(error_msg)

        try:
            _LOGGER.debug("Updating priors for entry_id: %s", entry_id)

            # Find the config entry and get coordinator
            config_entry = None
            for entry in hass.config_entries.async_entries(DOMAIN):
                if entry.entry_id == entry_id:
                    config_entry = entry
                    break

            if not config_entry:
                raise_error(f"Config entry {entry_id} not found")

            coordinator = config_entry.runtime_data

            _LOGGER.debug("Found coordinator for entry_id %s", entry_id)

            # Get history period from service call, fallback to coordinator config, then to default
            history_period = call.data.get("history_period")
            if history_period is None:
                history_period = coordinator.config.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                )

            _LOGGER.debug(
                "Using history period: %s (type: %s) for instance %s",
                history_period,
                type(history_period),
                entry_id,
            )

            # Always call update_learned_priors, passing the determined period
            _LOGGER.debug(
                "Calling update_learned_priors with period: %s for instance %s",
                history_period,
                entry_id,
            )
            try:
                await coordinator.update_learned_priors(history_period)
            except (CalculationError, HomeAssistantError) as err:
                _LOGGER.error(
                    "Error in update_learned_priors for instance %s: %s",
                    entry_id,
                    err,
                )
                raise_error("Failed to update learned priors", err)

            # Trigger a coordinator refresh
            _LOGGER.debug("Triggering coordinator refresh for instance %s", entry_id)
            await coordinator.async_refresh()
            _LOGGER.debug(
                "Prior update completed successfully for instance %s", entry_id
            )

        except KeyError as err:
            raise_error("Invalid entry_id or coordinator not found", err)
        except (HomeAssistantError, ValueError, RuntimeError) as err:
            raise_error("Failed to update priors", err)

    service_schema_update_priors = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional(
                "history_period",
            ): vol.All(int, vol.Range(min=1, max=90)),
        }
    )

    hass.services.async_register(
        DOMAIN,
        "update_priors",
        update_priors,
        schema=service_schema_update_priors,
    )
