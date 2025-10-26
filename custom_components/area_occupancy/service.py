"""Service definitions for the Area Occupancy Detection integration."""

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .utils import ensure_timezone_aware

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


def _get_coordinator(hass: HomeAssistant, entry_id: str) -> "AreaOccupancyCoordinator":
    """Get coordinator from entry_id with error handling."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == entry_id:
            return entry.runtime_data  # type: ignore[no-any-return]
    raise HomeAssistantError(f"Config entry {entry_id} not found")


async def _run_analysis(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of sensor likelihoods."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info("Running analysis for entry %s", entry_id)

        await coordinator.run_analysis()

        _LOGGER.info("Analysis completed successfully for entry %s", entry_id)

        entity_ids = [eid for eid in set(coordinator.config.entity_ids) if eid]

        # Check entity states
        entity_states = {}
        for entity_id in entity_ids:
            state = hass.states.get(entity_id)
            if state:
                entity_states[entity_id] = state.state
            else:
                entity_states[entity_id] = "NOT_FOUND"

        # Collect the updated likelihoods to return
        likelihood_data = {}

        for entity_id, entity in coordinator.entities.entities.items():
            entity_likelihood_data = {
                "type": entity.type.input_type.value,
                "weight": entity.type.weight,
                "prob_given_true": entity.prob_given_true,
                "prob_given_false": entity.prob_given_false,
            }

            likelihood_data[entity_id] = entity_likelihood_data

        response_data = {
            "area_name": coordinator.config.name,
            "current_prior": coordinator.area_prior,
            "global_prior": coordinator.prior.global_prior,
            "time_prior": coordinator.prior.time_prior,
            "prior_entity_ids": coordinator.prior.sensor_ids,
            "total_entities": len(coordinator.entities.entities),
            "entity_states": entity_states,
            "likelihoods": likelihood_data,
            "update_timestamp": dt_util.utcnow().isoformat(),
        }

    except Exception as err:
        error_msg = f"Failed to run analysis for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return response_data


async def _reset_entities(hass: HomeAssistant, call: ServiceCall) -> None:
    """Reset all entity probabilities and learned data."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info("Resetting entities for entry %s", entry_id)

        # Reset entities to fresh state
        await coordinator.entities.cleanup()

        await coordinator.async_refresh()

        _LOGGER.info("Entity reset completed successfully for entry %s", entry_id)

    except Exception as err:
        error_msg = f"Failed to reset entities for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _recover_database(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger database recovery for a specific entry."""
    entry_id = call.data["entry_id"]

    def _raise_recovery_error(entry_id: str) -> None:
        """Raise an error when database recovery fails."""
        _LOGGER.error("Database recovery failed for entry %s", entry_id)
        raise HomeAssistantError(f"Database recovery failed for entry {entry_id}")

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info("Triggering database recovery for entry %s", entry_id)

        # Get database status before recovery
        status_before = coordinator.db.get_database_status()

        # Trigger manual recovery
        recovery_success = coordinator.db.manual_recovery_trigger()

        # Get database status after recovery
        status_after = coordinator.db.get_database_status()

        if not recovery_success:
            _raise_recovery_error(entry_id)

        _LOGGER.info("Database recovery completed successfully for entry %s", entry_id)
        return {
            "entry_id": entry_id,
            "area_name": coordinator.config.name,
            "recovery_success": recovery_success,
            "status_before": status_before,
            "status_after": status_after,
            "recovery_timestamp": dt_util.utcnow().isoformat(),
        }

    except HomeAssistantError:
        # Re-raise our own errors
        raise
    except Exception as err:
        error_msg = f"Failed to recover database for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _get_database_status(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Get database status information for a specific entry."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.debug("Getting database status for entry %s", entry_id)

        status = coordinator.db.get_database_status()

        return {
            "entry_id": entry_id,
            "area_name": coordinator.config.name,
            "database_status": status,
            "timestamp": dt_util.utcnow().isoformat(),
        }

    except Exception as err:
        error_msg = f"Failed to get database status for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _get_entity_metrics(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Get basic entity metrics for diagnostics."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)
        entities = coordinator.entities.entities

        total_entities = len(entities)
        active_entities = sum(1 for e in entities.values() if e.evidence)
        available_entities = sum(1 for e in entities.values() if e.available)
        unavailable_entities = sum(1 for e in entities.values() if not e.available)
        decaying_entities = sum(1 for e in entities.values() if e.decay.is_decaying)

        metrics = {
            "total_entities": total_entities,
            "active_entities": active_entities,
            "available_entities": available_entities,
            "unavailable_entities": unavailable_entities,
            "decaying_entities": decaying_entities,
            "availability_percentage": round(
                (available_entities / total_entities * 100), 1
            )
            if total_entities > 0
            else 0,
            "activity_percentage": round((active_entities / total_entities * 100), 1)
            if total_entities > 0
            else 0,
            "summary": f"{total_entities} total entities: {active_entities} active, {available_entities} available, {unavailable_entities} unavailable, {decaying_entities} decaying",
        }

        _LOGGER.info("Retrieved entity metrics for entry %s", entry_id)

    except Exception as err:
        error_msg = f"Failed to get entity metrics for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err

    else:
        return {"metrics": metrics}


async def _get_problematic_entities(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Get entities that may need attention."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)
        entities = coordinator.entities.entities
        now = dt_util.utcnow()

        unavailable = [eid for eid, e in entities.items() if not e.available]
        stale_updates = [
            eid
            for eid, e in entities.items()
            if e.last_updated
            and (now - ensure_timezone_aware(e.last_updated)).total_seconds() > 3600
        ]

        problems = {
            "unavailable": unavailable,
            "stale_updates": stale_updates,
            "total_problems": len(unavailable) + len(stale_updates),
            "summary": f"Found {len(unavailable)} unavailable and {len(stale_updates)} stale entities out of {len(entities)} total",
        }

        _LOGGER.info("Retrieved problematic entities for entry %s", entry_id)

    except Exception as err:
        error_msg = f"Failed to get problematic entities for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err

    else:
        return {"problems": problems}


async def _get_area_status(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Get current area occupancy status and confidence."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        # Get current occupancy state
        area_name = coordinator.config.name
        occupancy_probability = coordinator.probability

        # Get entity metrics for additional context
        entities = coordinator.entities.entities
        metrics = {
            "total_entities": len(entities),
            "active_entities": sum(1 for e in entities.values() if e.evidence),
            "available_entities": sum(1 for e in entities.values() if e.available),
            "unavailable_entities": sum(
                1 for e in entities.values() if not e.available
            ),
            "decaying_entities": sum(
                1 for e in entities.values() if e.decay.is_decaying
            ),
        }

        # Format confidence level with more detail
        if occupancy_probability is not None:
            if occupancy_probability > 0.8:
                confidence_level = "high"
                confidence_description = "Very confident in occupancy status"
            elif occupancy_probability > 0.6:
                confidence_level = "medium-high"
                confidence_description = "Fairly confident in occupancy status"
            elif occupancy_probability > 0.2:
                confidence_level = "medium"
                confidence_description = "Moderate confidence in occupancy status"
            else:
                confidence_level = "low"
                confidence_description = "Low confidence in occupancy status"
        else:
            confidence_level = "unknown"  # type: ignore[unreachable]
            confidence_description = "Unable to determine confidence level"

        status = {
            "area_name": area_name,
            "occupied": coordinator.occupied,
            "occupancy_probability": round(occupancy_probability, 4)
            if occupancy_probability is not None
            else None,
            "area_baseline_prior": round(coordinator.prior.value, 4),
            "confidence_level": confidence_level,
            "confidence_description": confidence_description,
            "entity_summary": {
                "total_entities": metrics["total_entities"],
                "active_entities": metrics["active_entities"],
                "available_entities": metrics["available_entities"],
                "unavailable_entities": metrics["unavailable_entities"],
                "decaying_entities": metrics["decaying_entities"],
            },
            "status_summary": f"Area '{area_name}' is {'occupied' if coordinator.occupied else 'not occupied'} with {confidence_level} confidence ({round(occupancy_probability * 100, 1) if occupancy_probability else 0}% probability)",
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

    # Create async wrapper functions to properly handle the service calls

    async def handle_run_analysis(call: ServiceCall) -> dict[str, Any]:
        return await _run_analysis(hass, call)

    async def handle_reset_entities(call: ServiceCall) -> None:
        return await _reset_entities(hass, call)

    async def handle_recover_database(call: ServiceCall) -> dict[str, Any]:
        return await _recover_database(hass, call)

    async def handle_get_database_status(call: ServiceCall) -> dict[str, Any]:
        return await _get_database_status(hass, call)

    async def handle_get_entity_metrics(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_metrics(hass, call)

    async def handle_get_problematic_entities(call: ServiceCall) -> dict[str, Any]:
        return await _get_problematic_entities(hass, call)

    async def handle_get_area_status(call: ServiceCall) -> dict[str, Any]:
        return await _get_area_status(hass, call)

    # Register services with async wrapper functions
    hass.services.async_register(
        DOMAIN,
        "run_analysis",
        handle_run_analysis,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN, "reset_entities", handle_reset_entities, schema=entry_id_schema
    )

    hass.services.async_register(
        DOMAIN,
        "recover_database",
        handle_recover_database,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_database_status",
        handle_get_database_status,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
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
        "get_area_status",
        handle_get_area_status,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("Registered %d services for %s integration", 5, DOMAIN)
