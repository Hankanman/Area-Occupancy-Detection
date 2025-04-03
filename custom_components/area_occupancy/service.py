"""Service definitions for the Area Occupancy Detection integration."""

import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import CONF_NAME, DEFAULT_HISTORY_PERIOD, DOMAIN
from .exceptions import CalculationError, StorageError

_LOGGER = logging.getLogger(__name__)


async def async_setup_services(hass: HomeAssistant):
    """Register custom services for area occupancy."""

    async def update_priors(call):
        """Manually trigger an update of learned priors."""
        try:
            entry_id = call.data["entry_id"]
            _LOGGER.debug("Updating priors for entry_id: %s", entry_id)

            if DOMAIN not in hass.data or entry_id not in hass.data[DOMAIN]:
                raise HomeAssistantError(
                    f"Integration or instance {entry_id} not found"
                )

            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]
            _LOGGER.debug("Found coordinator for entry_id %s", entry_id)

            # Get history period from service call or use configured default
            history_period = call.data.get("history_period", DEFAULT_HISTORY_PERIOD)
            _LOGGER.debug(
                "Using history period: %s (type: %s) for instance %s",
                history_period,
                type(history_period),
                entry_id,
            )

            if history_period:
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
                    raise
            else:
                _LOGGER.debug(
                    "Calling update_learned_priors with default period for instance %s",
                    entry_id,
                )
                await coordinator.update_learned_priors()

            # Load existing data first to ensure we have all instances
            try:
                _LOGGER.debug(
                    "Loading existing storage data before saving for instance %s",
                    entry_id,
                )
                await coordinator.storage.async_load()
            except StorageError as err:
                _LOGGER.warning(
                    "Error loading existing data for instance %s: %s", entry_id, err
                )

            # Immediately save the updated priors to storage
            _LOGGER.debug("Saving updated priors to storage for instance %s", entry_id)
            await coordinator.storage.async_save_prior_state(
                coordinator.config[CONF_NAME], coordinator.prior_state, immediate=True
            )

            # Trigger a coordinator refresh
            _LOGGER.debug("Triggering coordinator refresh for instance %s", entry_id)
            await coordinator.async_refresh()
            _LOGGER.debug(
                "Prior update completed successfully for instance %s", entry_id
            )

        except KeyError as err:
            _LOGGER.error("Invalid entry_id or coordinator not found: %s", err)
            raise HomeAssistantError(
                f"Invalid entry_id or coordinator not found: {err}"
            ) from err
        except (HomeAssistantError, ValueError, RuntimeError) as err:
            _LOGGER.error(
                "Failed to update priors for instance %s: %s (type: %s)",
                entry_id,
                err,
                type(err),
            )
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
