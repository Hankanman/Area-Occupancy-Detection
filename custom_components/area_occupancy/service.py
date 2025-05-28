"""Service definitions for the Area Occupancy Detection integration."""

import logging

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD, DOMAIN
from .exceptions import CalculationError

_LOGGER = logging.getLogger(__name__)


async def async_setup_services(hass: HomeAssistant):
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

            if DOMAIN not in hass.data or entry_id not in hass.data[DOMAIN]:
                raise_error(f"Integration or instance {entry_id} not found")

            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]
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

    async def train_environmental_model(call):
        """Service to train or retrain the environmental ML model."""
        entry_id = call.data.get("entry_id")
        force_retrain = call.data.get("force_retrain", False)
        coordinator = hass.data[DOMAIN].get(entry_id)
        if not coordinator or not hasattr(coordinator, "ml_model_manager"):
            raise HomeAssistantError("ML model manager not available for this entry.")
        model_manager = coordinator.ml_model_manager
        training_data = await coordinator.get_environmental_training_data()
        try:
            await model_manager.train_model(training_data, force_retrain=force_retrain)
        except Exception as err:
            raise HomeAssistantError(f"Model training failed: {err}") from err

    async def analyze_environmental_patterns(call):
        """Service to analyze environmental sensor patterns for debugging."""
        entry_id = call.data.get("entry_id")
        sensor_types = call.data.get("sensor_types", [])
        coordinator = hass.data[DOMAIN].get(entry_id)
        if not coordinator or not hasattr(coordinator, "environmental_analyzer"):
            raise HomeAssistantError(
                "Environmental analyzer not available for this entry."
            )
        analyzer = coordinator.environmental_analyzer
        try:
            result = await analyzer.analyze_patterns(sensor_types)
            return result
        except Exception as err:
            raise HomeAssistantError(f"Pattern analysis failed: {err}") from err

    hass.services.async_register(
        DOMAIN,
        "update_priors",
        update_priors,
        schema=service_schema_update_priors,
    )
    hass.services.async_register(
        DOMAIN,
        "train_environmental_model",
        train_environmental_model,
    )
    hass.services.async_register(
        DOMAIN,
        "analyze_environmental_patterns",
        analyze_environmental_patterns,
    )
