"""Service definitions for the Area Occupancy Detection integration."""

from datetime import timedelta
import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DOMAIN, HA_RECORDER_DAYS
from .schema import state_intervals_table
from .state_intervals import filter_intervals
from .utils import get_current_time_slot, get_time_slot_name

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


def _get_coordinator(hass: HomeAssistant, entry_id: str) -> "AreaOccupancyCoordinator":
    """Get coordinator from entry_id with error handling."""
    for entry in hass.config_entries.async_entries(DOMAIN):
        if entry.entry_id == entry_id:
            return entry.runtime_data
    raise HomeAssistantError(f"Config entry {entry_id} not found")


async def _update_likelihoods(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of sensor likelihoods."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info(
            "Updating sensor likelihoods for entry %s",
            entry_id,
        )

        # Update individual sensor likelihoods with forced recalculation
        updated_count = await coordinator.entities.update_all_entity_likelihoods()
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


async def _update_area_prior(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of time-based priors and return the current priors in a human-readable format."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        _LOGGER.info(
            "Starting time-based priors update for entry %s",
            entry_id,
        )

        # Calculate all priors with forced recalculation
        await coordinator.prior.update()
        await coordinator.async_refresh()

        _LOGGER.info("Area prior update completed successfully for entry %s", entry_id)

        # Now, return the current time-based priors in a human-readable format (moved from _get_time_based_priors)
        current_day, current_slot = get_current_time_slot()

        # Get all time-based priors from database
        time_priors = {}
        if coordinator.sqlite_store:
            try:
                records = await coordinator.sqlite_store.get_time_priors_for_entry(
                    entry_id
                )
                for record in records:
                    time_priors[(record.day_of_week, record.time_slot)] = (
                        record.prior_value
                    )
            except HomeAssistantError as err:
                _LOGGER.debug("Failed to get time priors from database: %s", err)

        # Format the data in a human-readable way
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        # Create a summary by day
        daily_summaries = {}
        for day_of_week in range(7):
            day_name = day_names[day_of_week]
            day_priors = {}

            # Get priors for this day
            for time_slot in range(48):
                prior_value = time_priors.get((day_of_week, time_slot), None)
                if prior_value is not None:
                    hour = time_slot // 2
                    minute = (time_slot % 2) * 30
                    time_str = f"{hour:02d}:{minute:02d}"
                    day_priors[time_str] = round(prior_value, 4)

            if day_priors:
                daily_summaries[day_name] = day_priors

        # Create a summary of key time periods
        key_periods = {
            "Early Morning (06:00-08:00)": [],
            "Morning (08:00-12:00)": [],
            "Afternoon (12:00-17:00)": [],
            "Evening (17:00-21:00)": [],
            "Night (21:00-06:00)": [],
        }

        for day_of_week in range(7):
            day_name = day_names[day_of_week]

            # Calculate averages for key periods
            early_morning = []
            morning = []
            afternoon = []
            evening = []
            night = []

            for time_slot in range(48):
                prior_value = time_priors.get((day_of_week, time_slot), None)
                if prior_value is not None:
                    hour = time_slot // 2
                    if 6 <= hour < 8:
                        early_morning.append(prior_value)
                    elif 8 <= hour < 12:
                        morning.append(prior_value)
                    elif 12 <= hour < 17:
                        afternoon.append(prior_value)
                    elif 17 <= hour < 21:
                        evening.append(prior_value)
                    else:
                        night.append(prior_value)

            # Calculate averages
            if early_morning:
                key_periods["Early Morning (06:00-08:00)"].append(
                    {
                        "day": day_name,
                        "average": round(sum(early_morning) / len(early_morning), 4),
                    }
                )
            if morning:
                key_periods["Morning (08:00-12:00)"].append(
                    {"day": day_name, "average": round(sum(morning) / len(morning), 4)}
                )
            if afternoon:
                key_periods["Afternoon (12:00-17:00)"].append(
                    {
                        "day": day_name,
                        "average": round(sum(afternoon) / len(afternoon), 4),
                    }
                )
            if evening:
                key_periods["Evening (17:00-21:00)"].append(
                    {"day": day_name, "average": round(sum(evening) / len(evening), 4)}
                )
            if night:
                key_periods["Night (21:00-06:00)"].append(
                    {"day": day_name, "average": round(sum(night) / len(night), 4)}
                )

        # Remove empty periods
        key_periods = {k: v for k, v in key_periods.items() if v}

        return {
            "area_name": coordinator.config.name,
            "area_prior": coordinator.prior.value,
            "time_prior": getattr(
                coordinator.prior.get_time_slot_prior(), "prior", None
            ),
            "global_prior": getattr(
                coordinator.prior.get_global_prior(), "prior", None
            ),
            "primary_sensors": coordinator.prior.sensor_ids,
            "last_updated": coordinator.prior.last_updated.isoformat()
            if coordinator.prior.last_updated
            else None,
            "total_time_slots_available": len(time_priors),
            "current_time_slot": get_time_slot_name(current_day, current_slot),
            "daily_summaries": daily_summaries,
            "key_periods": key_periods,
        }

    except (HomeAssistantError, ValueError, RuntimeError) as err:
        error_msg = f"Failed to update and get time-based priors for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


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


async def _debug_import_intervals(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Debug service to manually trigger state intervals import."""
    entry_id = call.data["entry_id"]

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
            entity_ids, days=HA_RECORDER_DAYS
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
            "days_imported": HA_RECORDER_DAYS,
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
            },
        }


async def _purge_intervals(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Purge state_intervals table based on filter_intervals and retention period."""
    entry_id = call.data["entry_id"]
    retention_days = call.data.get("retention_days", 365)
    entity_ids = call.data.get("entity_ids", [])

    try:
        coordinator = _get_coordinator(hass, entry_id)
        sqlite_store = coordinator.sqlite_store
        cutoff_date = dt_util.utcnow() - timedelta(days=retention_days)

        if not entity_ids:
            entity_ids = coordinator.config.entity_ids

        # Count all intervals in the db for this entry_id (across all entity_ids)
        def _count_all_intervals():
            with sqlite_store.engine.connect() as conn:
                result = conn.execute(
                    state_intervals_table.select().where(
                        state_intervals_table.c.entity_id.in_(entity_ids)
                    )
                )
                return len(result.fetchall())

        total_intervals_in_db = await hass.async_add_executor_job(_count_all_intervals)

        # 1. Delete all intervals older than the retention period
        def _delete_old():
            with sqlite_store.engine.connect() as conn:
                result = conn.execute(
                    state_intervals_table.delete().where(
                        state_intervals_table.c.entity_id.in_(entity_ids)
                        & (state_intervals_table.c.end_time < cutoff_date)
                    )
                )
                conn.commit()
                return result.rowcount

        total_deleted_old = await hass.async_add_executor_job(_delete_old)

        # 2. For intervals within retention, filter and delete those that don't pass
        total_checked = 0
        total_deleted_filtered = 0
        total_kept = 0
        details = {}
        for entity_id in entity_ids:
            # Fetch intervals within retention
            intervals = await sqlite_store.get_historical_intervals(
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

                def _delete(to_delete=to_delete):
                    with sqlite_store.engine.connect() as conn:
                        for iv in to_delete:
                            conn.execute(
                                state_intervals_table.delete().where(
                                    (
                                        state_intervals_table.c.entity_id
                                        == iv["entity_id"]
                                    )
                                    & (
                                        state_intervals_table.c.start_time
                                        == iv["start"]
                                    )
                                    & (state_intervals_table.c.end_time == iv["end"])
                                    & (state_intervals_table.c.state == iv["state"])
                                )
                            )
                        conn.commit()

                await hass.async_add_executor_job(_delete)

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
        }
    )

    entity_details_schema = vol.Schema(
        {vol.Required("entry_id"): str, vol.Optional("entity_ids", default=[]): [str]}
    )

    entity_type_learned_schema = vol.Schema({vol.Required("entry_id"): str})

    purge_intervals_schema = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional("retention_days", default=365): int,
            vol.Optional("entity_ids", default=[]): [str],
        }
    )

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

    async def handle_get_area_status(call: ServiceCall) -> dict[str, Any]:
        return await _get_area_status(hass, call)

    async def handle_get_entity_type_learned_data(call: ServiceCall) -> dict[str, Any]:
        return await _get_entity_type_learned_data(hass, call)

    async def handle_debug_import_intervals(call: ServiceCall) -> dict[str, Any]:
        return await _debug_import_intervals(hass, call)

    async def handle_debug_database_state(call: ServiceCall) -> dict[str, Any]:
        return await _debug_database_state(hass, call)

    async def handle_purge_intervals(call: ServiceCall) -> dict[str, Any]:
        return await _purge_intervals(hass, call)

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

    hass.services.async_register(
        DOMAIN,
        "purge_intervals",
        handle_purge_intervals,
        schema=purge_intervals_schema,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("Registered %d services for %s integration", 12, DOMAIN)
