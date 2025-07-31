"""Service definitions for the Area Occupancy Detection integration."""

from datetime import timedelta
import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .state_intervals import filter_intervals

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


def _get_coordinator(hass: HomeAssistant, entry_id: str) -> "AreaOccupancyCoordinator":
    """Get coordinator from entry_id with error handling."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == entry_id:
            return entry.runtime_data
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

        import_stats = coordinator.storage.import_stats
        total_imported = sum(import_stats.values())
        total_intervals = await coordinator.storage.get_total_intervals_count()

        response_data = {
            "area_name": coordinator.config.name,
            "current_prior": coordinator.area_prior,
            "global_prior": coordinator.prior.global_prior,
            "occupancy_prior": coordinator.prior.occupancy_prior,
            "primary_sensors_prior": coordinator.prior.primary_sensors_prior,
            "prior_entity_ids": coordinator.prior.sensor_ids,
            "total_entities": len(coordinator.entities.entities),
            "import_stats": import_stats,
            "total_imported": total_imported,
            "total_intervals": total_intervals,
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
            if e.last_updated and (now - e.last_updated).total_seconds() > 3600
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


async def _get_entity_details(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
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
                    "evidence": entity.evidence,
                    "available": entity.available,
                    "last_updated": entity.last_updated.isoformat(),
                    "probability": entity.probability,
                    "decay_factor": entity.decay.decay_factor,
                    "is_decaying": entity.decay.is_decaying,
                    "entity_type": {
                        "input_type": entity.type.input_type.value,
                        "weight": entity.type.weight,
                        "prob_true": entity.type.prob_true,
                        "prob_false": entity.type.prob_false,
                        "prior": entity.type.prior,
                        "active_states": entity.type.active_states,
                        "active_range": entity.type.active_range,
                    },
                    "prob_given_true": entity.prob_given_true,
                    "prob_given_false": entity.prob_given_false,
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
            confidence_level = "unknown"
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


async def _get_entity_type_learned_data(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Return the learned entity_type data for an entry_id."""
    entry_id = call.data["entry_id"]
    try:
        coordinator = _get_coordinator(hass, entry_id)
        entity_types = coordinator.entity_types.entity_types
        learned_data = {}
        for input_type, et in entity_types.items():
            learned_data[input_type.value] = {
                "prior": et.prior,
                "prob_true": et.prob_true,
                "prob_false": et.prob_false,
                "weight": et.weight,
                "active_states": et.active_states,
                "active_range": et.active_range,
            }
        _LOGGER.info("Retrieved learned entity_type data for entry %s", entry_id)
    except Exception as err:
        error_msg = f"Failed to get entity_type learned data for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"entity_types": learned_data}


async def _debug_database_state(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Debug service to check current database state."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info("Debug: Checking database state for entry %s", entry_id)

        # Check if state_intervals table is empty
        is_empty = await coordinator.storage.is_state_intervals_empty()
        total_intervals = await coordinator.storage.get_total_intervals_count()

        _LOGGER.info(
            "Debug: State intervals empty: %s, Total count: %d",
            is_empty,
            total_intervals,
        )

        # Get some sample intervals for a few entities
        entity_ids = coordinator.config.entity_ids[:3]  # Just first 3 entities
        sample_data = {}

        for entity_id in entity_ids:
            try:
                # Get recent intervals for this entity
                intervals = await coordinator.storage.get_historical_intervals(
                    entity_id,
                    start_time=dt_util.utcnow() - timedelta(days=1),
                    end_time=dt_util.utcnow(),
                )
                sample_data[entity_id] = {
                    "intervals_found": len(intervals),
                    "latest_intervals": [
                        {
                            "state": interval["state"],
                            "start": interval["start"].isoformat(),
                            "end": interval["end"].isoformat(),
                            "duration_minutes": round(
                                (interval["end"] - interval["start"]).total_seconds()
                                / 60,
                                2,
                            ),
                        }
                        for interval in intervals[:3]
                    ],
                }
                _LOGGER.info(
                    "Debug: Entity %s has %d intervals in last 24h",
                    entity_id,
                    len(intervals),
                )
            except (HomeAssistantError, OSError) as err:
                sample_data[entity_id] = {"error": str(err)}

        # Get database statistics
        stats = await coordinator.storage.async_get_stats()

    except Exception as err:
        error_msg = f"Debug database state failed for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {
            "database_state": {
                "state_intervals_empty": is_empty,
                "total_intervals": total_intervals,
                "sample_entities": sample_data,
                "database_stats": stats,
            },
            "configuration": {
                "entity_ids": coordinator.config.entity_ids,
            },
        }


async def _purge_intervals(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Purge state_intervals table based on filter_intervals and retention period."""
    entry_id = call.data["entry_id"]
    retention_days = call.data.get("retention_days", 365)
    entity_ids = call.data.get("entity_ids", [])

    try:
        coordinator = _get_coordinator(hass, entry_id)
        storage = coordinator.storage
        cutoff_date = dt_util.utcnow() - timedelta(days=retention_days)

        if not entity_ids:
            entity_ids = coordinator.config.entity_ids

        # Count all intervals in the db for this entry_id (across all entity_ids)
        total_intervals_in_db = await storage.get_total_intervals_count()

        # 1. Delete all intervals older than the retention period using ORM
        total_deleted_old = await storage.cleanup_old_intervals(retention_days)

        # 2. For intervals within retention, filter and delete those that don't pass
        total_checked = 0
        total_deleted_filtered = 0
        total_kept = 0
        details = {}

        for entity_id in entity_ids:
            # Fetch intervals within retention
            intervals = await storage.get_historical_intervals(
                entity_id,
                start_time=cutoff_date,
                end_time=None,
                limit=None,
                page_size=1000,
            )
            total_checked += len(intervals)
            filtered = filter_intervals(intervals)
            to_delete = [iv for iv in intervals if iv not in filtered]
            total_deleted_filtered += len(to_delete)
            total_kept += len(filtered)

            if to_delete:
                # Delete filtered intervals using ORM
                def _delete_filtered(to_delete=to_delete, storage=storage):
                    return storage.executor.execute_in_session(
                        lambda session: storage.queries.delete_specific_intervals(
                            to_delete, session
                        )
                    )

                await hass.async_add_executor_job(_delete_filtered)

            details[entity_id] = {
                "intervals_checked": len(intervals),
                "intervals_deleted_filtered": len(to_delete),
                "intervals_kept": len(filtered),
            }

        _LOGGER.info(
            "Purged intervals for entry %s: deleted old %d, checked %d, deleted filtered %d, kept %d",
            entry_id,
            total_deleted_old,
            total_checked,
            total_deleted_filtered,
            total_kept,
        )

    except Exception as err:
        error_msg = f"Failed to purge intervals for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {
            "entry_id": entry_id,
            "retention_days": retention_days,
            "entity_ids": entity_ids,
            "total_deleted_old": total_deleted_old,
            "total_checked": total_checked,
            "total_deleted_filtered": total_deleted_filtered,
            "total_kept": total_kept,
            "details": details,
            "total_intervals_in_db": total_intervals_in_db,
        }


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Service schemas
    entry_id_schema = vol.Schema({vol.Required("entry_id"): str})

    entity_details_schema = vol.Schema(
        {vol.Required("entry_id"): str, vol.Optional("entity_ids", default=[]): [str]}
    )

    purge_intervals_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional("retention_days", default=365): int,
            vol.Optional("entity_ids", default=[]): [str],
        }
    )

    # Create async wrapper functions to properly handle the service calls

    async def handle_run_analysis(call: ServiceCall) -> dict[str, Any]:
        return await _run_analysis(hass, call)

    async def handle_reset_entities(call: ServiceCall) -> None:
        return await _reset_entities(hass, call)

    async def handle_get_entity_metrics(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_metrics(hass, call)

    async def handle_get_problematic_entities(call: ServiceCall) -> dict[str, Any]:
        return await _get_problematic_entities(hass, call)

    async def handle_get_entity_details(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_details(hass, call)

    async def handle_get_area_status(call: ServiceCall) -> dict[str, Any]:
        return await _get_area_status(hass, call)

    async def handle_get_entity_type_learned_data(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_type_learned_data(hass, call)

    async def handle_debug_database_state(call: ServiceCall) -> dict[str, Any]:
        return await _debug_database_state(hass, call)

    async def handle_purge_intervals(call: ServiceCall) -> dict[str, Any]:
        return await _purge_intervals(hass, call)

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
        "get_area_status",
        handle_get_area_status,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_entity_type_learned_data",
        handle_get_entity_type_learned_data,
        schema=entry_id_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "debug_database_state",
        handle_debug_database_state,
        schema=vol.Schema(
            {
                vol.Required("entry_id"): str,
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "purge_intervals",
        handle_purge_intervals,
        schema=purge_intervals_schema,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("Registered %d services for %s integration", 9, DOMAIN)
