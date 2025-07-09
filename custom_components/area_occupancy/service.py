"""Service definitions for the Area Occupancy Detection integration."""

from datetime import timedelta
import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DOMAIN

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


def _get_coordinator(hass: HomeAssistant, entry_id: str) -> "AreaOccupancyCoordinator":
    """Get coordinator from entry_id with error handling."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == entry_id:
            return entry.runtime_data
    raise HomeAssistantError(f"Config entry {entry_id} not found")


async def _update_area_prior(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of the area baseline prior."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        # Remove history_period handling, always use coordinator.config.history.period
        history_period = coordinator.config.history.period

        _LOGGER.info(
            "Updating area baseline prior for entry %s with %d days history",
            entry_id,
            history_period,
        )

        # Update area baseline prior with forced recalculation
        area_baseline_prior = await coordinator.prior.update(
            force=True, history_period=history_period
        )

        # Determine which prior was used (occupancy_entity or sensors)
        all_sensors_prior = getattr(coordinator.prior, "_all_sensors_prior", None)
        occupancy_entity_prior = getattr(
            coordinator.prior, "_occupancy_entity_prior", None
        )
        prior_source = getattr(coordinator.prior, "_prior_source", "unknown")

        # Collect calculation details
        calculation_details = {
            "motion_sensors": coordinator.prior.sensor_ids,
            "sensor_count": len(coordinator.prior.sensor_ids),
            "calculation_method": "Max of (average of individual sensor occupancy ratios + 5% buffer, occupancy_entity_id prior + 5% buffer)",
            "all_sensors_prior": all_sensors_prior,
            "occupancy_entity_prior": occupancy_entity_prior,
            "prior_source": prior_source,
        }

        # Add per-sensor details if data is available
        if coordinator.prior.data:
            sensor_details = {}
            total_ratio = 0.0

            for sensor_id, sensor_data in coordinator.prior.data.items():
                total_seconds = (
                    sensor_data["end_time"] - sensor_data["start_time"]
                ).total_seconds()

                sensor_details[sensor_id] = {
                    "occupancy_ratio": round(sensor_data["ratio"], 4),
                    "occupied_seconds": sensor_data["occupied_seconds"],
                    "total_seconds": int(total_seconds),
                    "states_found": sensor_data["states_count"],
                    "intervals_found": len(sensor_data["intervals"]),
                }

                total_ratio += sensor_data["ratio"]

            raw_average = (
                total_ratio / len(coordinator.prior.data)
                if coordinator.prior.data
                else 0.0
            )

            calculation_details.update(
                {
                    "sensor_details": sensor_details,
                    "raw_average_ratio": round(raw_average, 4),
                    "buffer_multiplier": 1.05,
                    "final_prior": round(area_baseline_prior, 4),
                    "calculation": f"({raw_average:.4f} average) × 1.05 buffer = {area_baseline_prior:.4f}",
                }
            )
        else:
            calculation_details.update(
                {
                    "sensor_details": "No sensor data available",
                    "note": f"Using default prior value of {area_baseline_prior}",
                }
            )

        await coordinator.async_refresh()

        _LOGGER.info("Area prior update completed successfully for entry %s", entry_id)

        return {
            "area_prior": area_baseline_prior,
            "history_period": history_period,
            "update_timestamp": dt_util.utcnow().isoformat(),
            "prior_source": prior_source,
            "calculation_details": calculation_details,
        }

    except (HomeAssistantError, ValueError, RuntimeError) as err:
        error_msg = f"Failed to update area prior for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _update_likelihoods(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of sensor likelihoods."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        # Remove history_period handling, always use coordinator.config.history.period
        history_period = coordinator.config.history.period

        _LOGGER.info(
            "Updating sensor likelihoods for entry %s with %d days history",
            entry_id,
            history_period,
        )

        # Update individual sensor likelihoods with forced recalculation
        updated_count = await coordinator.entities.update_all_entity_likelihoods(
            history_period, force=True
        )
        await coordinator.async_refresh()

        # Collect the updated likelihoods to return
        likelihood_data = {}

        for entity_id, entity in coordinator.entities.entities.items():
            entity_likelihood_data = {
                "type": entity.type.input_type.value,
                "weight": entity.type.weight,
                "prob_given_true": entity.likelihood.prob_given_true,
                "prob_given_false": entity.likelihood.prob_given_false,
                "prob_given_true_raw": entity.likelihood.prob_given_true_raw,
                "prob_given_false_raw": entity.likelihood.prob_given_false_raw,
            }

            likelihood_data[entity_id] = entity_likelihood_data

        _LOGGER.info("Likelihood update completed successfully for entry %s", entry_id)

        response_data = {
            "updated_entities": updated_count,
            "history_period": history_period,
            "total_entities": len(coordinator.entities.entities),
            "update_timestamp": dt_util.utcnow().isoformat(),
            "prior": coordinator.area_prior,
            "likelihoods": likelihood_data,
        }

    except (HomeAssistantError, ValueError, RuntimeError) as err:
        error_msg = f"Failed to update likelihoods for {entry_id}: {err}"
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

        # Clear storage if requested
        if call.data.get("clear_storage", False):
            await coordinator.sqlite_store.async_reset()

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
                    "prior": {
                        "prob_given_true": entity.likelihood.prob_given_true,
                        "prob_given_false": entity.likelihood.prob_given_false,
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


async def _force_entity_update(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Force immediate update of specific entities."""
    entry_id = call.data["entry_id"]
    entity_ids = call.data.get("entity_ids", [])

    try:
        coordinator = _get_coordinator(hass, entry_id)
        entities = coordinator.entities

        if not entity_ids:
            entity_ids = list(entities.entities.keys())

        updated_count = len(entity_ids)

        await coordinator.async_refresh()

        _LOGGER.info("Force updated %d entities in entry %s", updated_count, entry_id)

    except Exception as err:
        error_msg = f"Failed to force entity updates for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"updated_entities": updated_count}


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

        status = {
            "area_name": area_name,
            "occupied": coordinator.occupied,
            "occupancy_probability": occupancy_probability,
            "area_baseline_prior": coordinator.prior,
            "confidence_level": (
                (
                    "high"
                    if occupancy_probability and occupancy_probability > 0.8
                    else (
                        "medium"
                        if occupancy_probability and occupancy_probability > 0.2
                        else "low"
                    )
                )
                if occupancy_probability is not None
                else "unknown"
            ),
            "total_entities": metrics["total_entities"],
            "active_entities": metrics["active_entities"],
            "available_entities": metrics["available_entities"],
            "unavailable_entities": metrics["unavailable_entities"],
            "decaying_entities": metrics["decaying_entities"],
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


async def _debug_import_intervals(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Debug service to manually trigger state intervals import."""
    entry_id = call.data["entry_id"]
    days = call.data.get("days", 10)

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info("Debug: Manual state intervals import for entry %s", entry_id)

        # Get entity IDs from configuration
        entity_ids = coordinator.config.entity_ids
        _LOGGER.info(
            "Debug: Found %d entity_ids in config: %s", len(entity_ids), entity_ids
        )

        # Remove duplicates and empty strings
        entity_ids = [eid for eid in set(entity_ids) if eid]
        _LOGGER.info(
            "Debug: Filtered to %d entity_ids: %s", len(entity_ids), entity_ids
        )

        if not entity_ids:
            return {
                "error": "No entity IDs found in configuration",
                "entity_ids": [],
            }

        # Check entity states
        entity_states = {}
        for entity_id in entity_ids:
            state = hass.states.get(entity_id)
            if state:
                entity_states[entity_id] = state.state
                _LOGGER.info(
                    "Debug: Entity %s exists with state: %s", entity_id, state.state
                )
            else:
                entity_states[entity_id] = "NOT_FOUND"
                _LOGGER.warning(
                    "Debug: Entity %s does not exist in Home Assistant", entity_id
                )

        # Perform import
        import_counts = await coordinator.sqlite_store.import_intervals_from_recorder(
            entity_ids, days=days
        )
        total_imported = sum(import_counts.values())

        _LOGGER.info("Debug: Import completed with results: %s", import_counts)

        # Check final state of database
        total_intervals = await coordinator.sqlite_store.get_total_intervals_count()

    except Exception as err:
        error_msg = f"Debug import failed for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {
            "entity_ids": entity_ids,
            "entity_states": entity_states,
            "import_counts": import_counts,
            "total_imported": total_imported,
            "total_intervals_in_db": total_intervals,
            "days_imported": days,
        }


async def _debug_database_state(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Debug service to check current database state."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info("Debug: Checking database state for entry %s", entry_id)

        # Check if state_intervals table is empty
        is_empty = await coordinator.sqlite_store.is_state_intervals_empty()
        total_intervals = await coordinator.sqlite_store.get_total_intervals_count()

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
                intervals = await coordinator.sqlite_store.get_historical_intervals(
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
        stats = await coordinator.sqlite_store.async_get_stats()

        # Add schema information
        schema_info = {
            "tables": [
                "entities",
                "state_intervals",
                "area_occupancy",
                "area_entity_config",
                "metadata",
            ],
            "simplified_schema": True,
            "removed_tables": ["area_history"],
            "removed_fields": {
                "area_occupancy": ["probability", "prior", "occupied"],
                "area_entity_config": ["attributes", "last_state"],
                "entities": ["domain"],
            },
            "indexes_count": 3,  # Simplified from 11 to 3
        }

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
                "schema_info": schema_info,
            },
            "configuration": {
                "entity_ids": coordinator.config.entity_ids,
                "history_enabled": coordinator.config.history.enabled,
                "history_period": coordinator.config.history.period,
            },
        }


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Service schemas
    entry_id_schema = vol.Schema({vol.Required("entry_id"): str})

    # Remove history_period from schemas
    update_area_prior_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
        }
    )

    update_likelihoods_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
        }
    )

    reset_entities_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional("clear_storage", default=False): bool,
        }
    )

    entity_details_schema = vol.Schema(
        {vol.Required("entry_id"): str, vol.Optional("entity_ids", default=[]): [str]}
    )

    force_update_schema = vol.Schema(
        {vol.Required("entry_id"): str, vol.Optional("entity_ids", default=[]): [str]}
    )

    entity_type_learned_schema = vol.Schema({vol.Required("entry_id"): str})

    # Create async wrapper functions to properly handle the service calls
    async def handle_update_area_prior(call: ServiceCall) -> dict[str, Any]:
        return await _update_area_prior(hass, call)

    async def handle_update_likelihoods(call: ServiceCall) -> dict[str, Any]:
        return await _update_likelihoods(hass, call)

    async def handle_reset_entities(call: ServiceCall) -> None:
        return await _reset_entities(hass, call)

    async def handle_get_entity_metrics(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_metrics(hass, call)

    async def handle_get_problematic_entities(call: ServiceCall) -> dict[str, Any]:
        return await _get_problematic_entities(hass, call)

    async def handle_get_entity_details(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_details(hass, call)

    async def handle_force_entity_update(call: ServiceCall) -> dict[str, Any]:
        return await _force_entity_update(hass, call)

    async def handle_get_area_status(call: ServiceCall) -> dict[str, Any]:
        return await _get_area_status(hass, call)

    async def handle_get_entity_type_learned_data(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_type_learned_data(hass, call)

    async def handle_debug_import_intervals(call: ServiceCall) -> dict[str, Any]:
        return await _debug_import_intervals(hass, call)

    async def handle_debug_database_state(call: ServiceCall) -> dict[str, Any]:
        return await _debug_database_state(hass, call)

    # Register services with async wrapper functions
    hass.services.async_register(
        DOMAIN,
        "update_area_prior",
        handle_update_area_prior,
        schema=update_area_prior_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "update_likelihoods",
        handle_update_likelihoods,
        schema=update_likelihoods_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN, "reset_entities", handle_reset_entities, schema=reset_entities_schema
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

    hass.services.async_register(
        DOMAIN,
        "get_entity_type_learned_data",
        handle_get_entity_type_learned_data,
        schema=entity_type_learned_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "debug_import_intervals",
        handle_debug_import_intervals,
        schema=vol.Schema(
            {
                vol.Required("entry_id"): str,
                vol.Optional("days", default=10): int,
            }
        ),
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

    _LOGGER.info("Registered %d services for %s integration", 11, DOMAIN)
