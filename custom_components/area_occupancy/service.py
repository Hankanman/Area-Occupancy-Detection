"""Service definitions for the Area Occupancy Detection integration."""

import logging

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


def _get_coordinator(hass: HomeAssistant, entry_id: str):
    """Get coordinator from entry_id with error handling."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == entry_id:
            return entry.runtime_data
    raise HomeAssistantError(f"Config entry {entry_id} not found")


async def _update_priors(hass: HomeAssistant, call: ServiceCall):
    """Manually trigger an update of learned priors."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        # Get history period from service call or use coordinator default
        history_period = (
            call.data.get("history_period") or coordinator.config.decay.history_period
        )

        _LOGGER.info(
            "Updating priors for entry %s with %d days history",
            entry_id,
            history_period,
        )
        await coordinator.update_learned_priors(history_period)
        await coordinator.async_refresh()

        # Collect the updated priors to return
        priors_data = {}
        for entity_id, entity in coordinator.entities.entities.items():
            priors_data[entity_id] = {
                "prior": entity.prior.prior,
                "prob_given_true": entity.prior.prob_given_true,
                "prob_given_false": entity.prior.prob_given_false,
                "last_updated": entity.prior.last_updated.isoformat(),
                "type": entity.prior.type.value,
                "entity_type": entity.type.input_type.value,
            }

        _LOGGER.info("Prior update completed successfully for entry %s", entry_id)

        return {
            "updated_priors": priors_data,
            "history_period": history_period,
            "total_entities": len(priors_data),
            "update_timestamp": dt_util.utcnow().isoformat(),
        }

    except (HomeAssistantError, ValueError, RuntimeError) as err:
        error_msg = f"Failed to update priors for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _reset_entities(hass: HomeAssistant, call: ServiceCall):
    """Reset all entity probabilities and learned data."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info("Resetting entities for entry %s", entry_id)

        # Reset entities to fresh state
        await coordinator.entities.reset_entities()

        # Clear storage if requested
        if call.data.get("clear_storage", False):
            await coordinator.storage.async_reset()

        await coordinator.async_refresh()

        _LOGGER.info("Entity reset completed successfully for entry %s", entry_id)

    except Exception as err:
        error_msg = f"Failed to reset entities for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _get_entity_metrics(hass: HomeAssistant, call: ServiceCall):
    """Get basic entity metrics for diagnostics."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)
        entities = coordinator.entities.entities

        metrics = {
            "total_entities": len(entities),
            "active_entities": sum(1 for e in entities.values() if e.is_active),
            "available_entities": sum(1 for e in entities.values() if e.available),
            "unavailable_entities": sum(
                1 for e in entities.values() if not e.available
            ),
            "decaying_entities": sum(
                1 for e in entities.values() if e.decay.is_decaying
            ),
        }

        _LOGGER.info("Retrieved entity metrics for entry %s", entry_id)

    except Exception as err:
        error_msg = f"Failed to get entity metrics for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err

    else:
        return {"metrics": metrics}


async def _get_problematic_entities(hass: HomeAssistant, call: ServiceCall):
    """Get entities that may need attention."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)
        entities = coordinator.entities.entities
        now = dt_util.utcnow()

        problems = {
            "unavailable": [eid for eid, e in entities.items() if not e.available],
            "stale_updates": [
                eid
                for eid, e in entities.items()
                if e.last_updated and (now - e.last_updated).total_seconds() > 3600
            ],
        }

        _LOGGER.info("Retrieved problematic entities for entry %s", entry_id)

    except Exception as err:
        error_msg = f"Failed to get problematic entities for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err

    else:
        return {"problems": problems}


async def _get_entity_details(hass: HomeAssistant, call: ServiceCall):
    """Get detailed information about specific entities."""
    entry_id = call.data["entry_id"]
    entity_ids = call.data.get("entity_ids", [])

    try:
        coordinator = _get_coordinator(hass, entry_id)
        entities = coordinator.entities

        details = {}
        if not entity_ids:
            # Get all entities if none specified
            entity_ids = list(entities.entities.keys())

        for entity_id in entity_ids:
            try:
                entity = entities.get_entity(entity_id)
                details[entity_id] = {
                    "state": entity.state,
                    "is_active": entity.is_active,
                    "available": entity.available,
                    "last_updated": entity.last_updated.isoformat(),
                    "probability": entity.probability,
                    "decay_factor": entity.decay.decay_factor,
                    "is_decaying": entity.decay.is_decaying,
                    "decay_start_time": entity.decay.decay_start_time.isoformat()
                    if entity.decay.decay_start_time
                    else None,
                    "entity_type": {
                        "input_type": entity.type.input_type.value,
                        "weight": entity.type.weight,
                        "prob_true": entity.type.prob_true,
                        "prob_false": entity.type.prob_false,
                        "prior": entity.type.prior,
                        "active_states": entity.type.active_states,
                        "active_range": entity.type.active_range,
                    },
                    "prior": {
                        "prior": entity.prior.prior,
                        "prob_given_true": entity.prior.prob_given_true,
                        "prob_given_false": entity.prior.prob_given_false,
                        "last_updated": entity.prior.last_updated.isoformat(),
                    },
                }
            except ValueError:
                details[entity_id] = {"error": "Entity not found"}

        _LOGGER.info(
            "Retrieved details for %d entities in entry %s", len(details), entry_id
        )
    except Exception as err:
        error_msg = f"Failed to get entity details for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"entity_details": details}


async def _force_entity_update(hass: HomeAssistant, call: ServiceCall):
    """Force immediate update of specific entities."""
    entry_id = call.data["entry_id"]
    entity_ids = call.data.get("entity_ids", [])

    try:
        coordinator = _get_coordinator(hass, entry_id)
        entities = coordinator.entities

        if not entity_ids:
            entity_ids = list(entities.entities.keys())

        updated_count = 0
        for entity_id in entity_ids:
            try:
                entity = entities.get_entity(entity_id)
                await entity.async_update()
                updated_count += 1
            except ValueError:
                _LOGGER.warning("Entity %s not found for forced update", entity_id)

        await coordinator.async_refresh()

        _LOGGER.info("Force updated %d entities in entry %s", updated_count, entry_id)

    except Exception as err:
        error_msg = f"Failed to force entity updates for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"updated_entities": updated_count}


async def _get_area_status(hass: HomeAssistant, call: ServiceCall):
    """Get current area occupancy status and confidence."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        # Get current occupancy state
        area_name = coordinator.config.name
        occupancy_entity_id = (
            f"binary_sensor.{area_name.lower().replace(' ', '_')}_occupancy"
        )

        # Get occupancy sensor state from hass
        occupancy_state = hass.states.get(occupancy_entity_id)
        occupancy_probability = None

        if occupancy_state:
            occupancy_probability = occupancy_state.attributes.get("probability")

        # Get entity metrics for additional context
        entities = coordinator.entities.entities
        metrics = {
            "total_entities": len(entities),
            "active_entities": sum(1 for e in entities.values() if e.is_active),
            "available_entities": sum(1 for e in entities.values() if e.available),
            "unavailable_entities": sum(
                1 for e in entities.values() if not e.available
            ),
            "decaying_entities": sum(
                1 for e in entities.values() if e.decay.is_decaying
            ),
        }

        status = {
            "area_name": area_name,
            "is_occupied": occupancy_state.state == "on" if occupancy_state else None,
            "occupancy_probability": occupancy_probability,
            "confidence_level": (
                "high"
                if occupancy_probability and occupancy_probability > 0.8
                else "medium"
                if occupancy_probability and occupancy_probability > 0.2
                else "low"
            )
            if occupancy_probability is not None
            else "unknown",
            "total_entities": metrics["total_entities"],
            "active_entities": metrics["active_entities"],
            "available_entities": metrics["available_entities"],
            "unavailable_entities": metrics["unavailable_entities"],
            "decaying_entities": metrics["decaying_entities"],
            "last_updated": occupancy_state.last_updated.isoformat()
            if occupancy_state
            else None,
        }

        _LOGGER.info("Retrieved area status for entry %s", entry_id)

    except Exception as err:
        error_msg = f"Failed to get area status for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"area_status": status}


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Service schemas
    entry_id_schema = vol.Schema({vol.Required("entry_id"): str})

    update_priors_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional("history_period"): vol.All(int, vol.Range(min=1, max=90)),
        }
    )

    reset_entities_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional("clear_storage", default=False): bool,
        }
    )

    entity_details_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional("entity_ids", default=[]): [str],
        }
    )

    force_update_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional("entity_ids", default=[]): [str],
        }
    )

    # Create async wrapper functions to properly handle the service calls
    async def handle_update_priors(call: ServiceCall):
        return await _update_priors(hass, call)

    async def handle_reset_entities(call: ServiceCall):
        return await _reset_entities(hass, call)

    async def handle_get_entity_metrics(call: ServiceCall):
        return await _get_entity_metrics(hass, call)

    async def handle_get_problematic_entities(call: ServiceCall):
        return await _get_problematic_entities(hass, call)

    async def handle_get_entity_details(call: ServiceCall):
        return await _get_entity_details(hass, call)

    async def handle_force_entity_update(call: ServiceCall):
        return await _force_entity_update(hass, call)

    async def handle_get_area_status(call: ServiceCall):
        return await _get_area_status(hass, call)

    # Register services with async wrapper functions
    hass.services.async_register(
        DOMAIN,
        "update_priors",
        handle_update_priors,
        schema=update_priors_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "reset_entities",
        handle_reset_entities,
        schema=reset_entities_schema,
    )

    hass.services.async_register(
        DOMAIN,
        "get_entity_metrics",
        handle_get_entity_metrics,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_problematic_entities",
        handle_get_problematic_entities,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_entity_details",
        handle_get_entity_details,
        schema=entity_details_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "force_entity_update",
        handle_force_entity_update,
        schema=force_update_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_area_status",
        handle_get_area_status,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("Registered %d services for %s integration", 7, DOMAIN)
