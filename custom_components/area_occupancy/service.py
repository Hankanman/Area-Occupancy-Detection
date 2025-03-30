import voluptuous as vol

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    DEFAULT_HISTORY_PERIOD,
)


async def async_setup_services(hass: HomeAssistant):
    """Register custom services for area occupancy."""

    async def update_priors(call):
        """Manually trigger an update of learned priors."""
        try:
            entry_id = call.data["entry_id"]
            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]

            # Get history period from service call or use configured default
            history_period = call.data.get("history_period", DEFAULT_HISTORY_PERIOD)
            if history_period:
                await coordinator.update_learned_priors(history_period)
            else:
                await coordinator.update_learned_priors()

            # Trigger a coordinator refresh
            await coordinator.async_refresh()

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
