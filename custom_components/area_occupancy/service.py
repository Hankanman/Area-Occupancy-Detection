"""Service definitions for the Area Occupancy Detection integration."""

from datetime import timedelta
import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DOMAIN, HA_RECORDER_DAYS
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


async def _update_area_prior(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of the area baseline prior."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        history_period = HA_RECORDER_DAYS

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

        history_period = HA_RECORDER_DAYS

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


async def _update_time_based_priors(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Manually trigger an update of time-based priors."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

        history_period = HA_RECORDER_DAYS

        _LOGGER.info(
            "Starting time-based priors update for entry %s with %d days history",
            entry_id,
            history_period,
        )

        # Start the calculation in a background task to avoid blocking
        async def _calculate_in_background():
            """Calculate time-based priors in background."""
            try:
                # Calculate time-based priors with forced recalculation
                time_priors = await coordinator.prior.calculate_time_based_priors(
                    history_period=history_period, force=True
                )

                current_day, current_slot = get_current_time_slot()
                current_prior = time_priors.get(
                    (current_day, current_slot), coordinator.prior.value
                )

                await coordinator.async_refresh()

                _LOGGER.info(
                    "Time-based priors update completed successfully for entry %s",
                    entry_id,
                )

                return {
                    "status": "completed",
                    "time_priors_calculated": len(time_priors),
                    "current_time_prior": current_prior,
                    "history_period": history_period,
                    "completion_timestamp": dt_util.utcnow().isoformat(),
                }

            except HomeAssistantError as err:
                _LOGGER.error(
                    "Background time-based priors calculation failed for entry %s: %s",
                    entry_id,
                    err,
                )
                return {
                    "status": "failed",
                    "error": str(err),
                    "completion_timestamp": dt_util.utcnow().isoformat(),
                }

        # Start the background task
        hass.async_create_task(_calculate_in_background())

        # Return immediately with status
        return {
            "status": "started",
            "message": f"Time-based priors calculation started for entry {entry_id}",
            "history_period_days": history_period,
            "start_timestamp": dt_util.utcnow().isoformat(),
            "note": "This is a background operation. Check logs for completion status.",
        }

    except (HomeAssistantError, ValueError, RuntimeError) as err:
        error_msg = f"Failed to start time-based priors update for {entry_id}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _get_time_based_priors(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Get current time-based priors in a human-readable format."""
    entry_id = call.data["entry_id"]

    try:
        coordinator = _get_coordinator(hass, entry_id)

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
            "current_time_slot": get_time_slot_name(current_day, current_slot),
            "current_prior": round(coordinator.prior.value, 4),
            "time_prior": round(coordinator.prior.time_prior, 4),
            "global_prior": round(coordinator.prior.global_prior, 4),
            "total_time_slots_available": len(time_priors),
            "daily_summaries": daily_summaries,
            "key_periods": key_periods,
            "note": "Time-based priors show the learned occupancy probability for specific times of day and days of the week.",
        }

    except Exception as err:
        error_msg = f"Failed to get time-based priors for {entry_id}: {err}"
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
    days = HA_RECORDER_DAYS

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

    async def handle_update_time_based_priors(call: ServiceCall) -> dict[str, Any]:
        return await _update_time_based_priors(hass, call)

    async def handle_get_time_based_priors(call: ServiceCall) -> dict[str, Any]:
        return await _get_time_based_priors(hass, call)

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
        DOMAIN,
        "update_time_based_priors",
        handle_update_time_based_priors,
        schema=vol.Schema({vol.Required("entry_id"): str}),
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_time_based_priors",
        handle_get_time_based_priors,
        schema=vol.Schema({vol.Required("entry_id"): str}),
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

    _LOGGER.info("Registered %d services for %s integration", 12, DOMAIN)
