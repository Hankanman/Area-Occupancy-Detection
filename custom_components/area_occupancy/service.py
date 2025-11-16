"""Service definitions for the Area Occupancy Detection integration."""

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.selector import (
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
)
from homeassistant.util import dt as dt_util

from .const import ALL_AREAS_IDENTIFIER, DOMAIN
from .utils import ensure_timezone_aware

if TYPE_CHECKING:
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)


def _get_coordinator(hass: HomeAssistant) -> "AreaOccupancyCoordinator":
    """Get global coordinator from hass.data with error handling."""
    coordinator = hass.data.get(DOMAIN)
    if coordinator is None:
        raise HomeAssistantError(
            "Area Occupancy coordinator not found. Ensure integration is configured."
        )
    return coordinator


def _get_area_name_from_service_call(hass: HomeAssistant, call: ServiceCall) -> str:
    """Extract area_name from service call, handling backward compatibility with entry_id.

    Args:
        hass: Home Assistant instance
        call: The service call

    Returns:
        str: The area name (or ALL_AREAS_IDENTIFIER for "all")
    """
    # Check for deprecated entry_id parameter
    if "entry_id" in call.data:
        _LOGGER.warning(
            "The 'entry_id' parameter is deprecated and will be removed in a future version. "
            "Use 'area_name' instead. For backward compatibility, using first area."
        )
        # For backward compatibility, use first area
        coordinator = _get_coordinator(hass)
        area_names = coordinator.get_area_names()
        if area_names:
            area_name = area_names[0]
            _LOGGER.debug(
                "Using first area '%s' for deprecated entry_id parameter", area_name
            )
            return area_name
        # Fallback to "all" if no areas
        return ALL_AREAS_IDENTIFIER

    # Normal parameter handling
    area_name = call.data.get("area_name", "all")
    if area_name == "all":
        area_name = ALL_AREAS_IDENTIFIER
    return area_name


def _validate_area_exists(
    coordinator: "AreaOccupancyCoordinator", area_name: str
) -> None:
    """Validate that the area exists.

    Args:
        coordinator: The area occupancy coordinator
        area_name: The area name to validate

    Raises:
        HomeAssistantError: If area does not exist
    """
    if area_name not in coordinator.get_area_names():
        raise HomeAssistantError(f"Area '{area_name}' not found")

    area_check = coordinator.get_area_or_default(area_name)
    if area_check is None:
        raise HomeAssistantError(f"Area '{area_name}' not found")


async def _run_analysis(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of sensor likelihoods.

    Supports area_name parameter. If area_name is "all" or ALL_AREAS_IDENTIFIER,
    runs analysis for all areas.

    Backward compatibility: Supports deprecated 'entry_id' parameter.
    """
    area_name = _get_area_name_from_service_call(hass, call)

    try:
        coordinator = _get_coordinator(hass)

        if area_name == ALL_AREAS_IDENTIFIER:
            _LOGGER.info("Running analysis for all areas")
            await coordinator.run_analysis()
            _LOGGER.info("Analysis completed successfully for all areas")

            # Aggregate data from all areas
            all_areas_data = {}
            for area_name_item in coordinator.get_area_names():
                area = coordinator.get_area_or_default(area_name_item)
                entity_ids = list(area.entities.entities.keys())
                entity_states = {}
                for entity_id in entity_ids:
                    state = hass.states.get(entity_id)
                    if state:
                        entity_states[entity_id] = state.state
                    else:
                        entity_states[entity_id] = "NOT_FOUND"

                likelihood_data = {}
                for entity_id, entity in area.entities.entities.items():
                    likelihood_data[entity_id] = {
                        "type": entity.type.input_type.value,
                        "weight": entity.type.weight,
                        "prob_given_true": entity.prob_given_true,
                        "prob_given_false": entity.prob_given_false,
                    }

                all_areas_data[area_name_item] = {
                    "area_name": area_name_item,
                    "current_prior": area.area_prior(),
                    "global_prior": area.prior.global_prior,
                    "time_prior": area.prior.time_prior,
                    "prior_entity_ids": area.prior.sensor_ids,
                    "total_entities": len(area.entities.entities),
                    "entity_states": entity_states,
                    "likelihoods": likelihood_data,
                }

            return {
                "areas": all_areas_data,
                "update_timestamp": dt_util.utcnow().isoformat(),
            }

        # Run for specific area
        _validate_area_exists(coordinator, area_name)
        area = coordinator.get_area_or_default(area_name)

        _LOGGER.info("Running analysis for area %s", area_name)
        await coordinator.run_analysis()
        _LOGGER.info("Analysis completed successfully for area %s", area_name)

        entity_ids = list(area.entities.entities.keys())
        entity_states = {}
        for entity_id in entity_ids:
            state = hass.states.get(entity_id)
            if state:
                entity_states[entity_id] = state.state
            else:
                entity_states[entity_id] = "NOT_FOUND"

        likelihood_data = {}
        for entity_id, entity in area.entities.entities.items():
            likelihood_data[entity_id] = {
                "type": entity.type.input_type.value,
                "weight": entity.type.weight,
                "prob_given_true": entity.prob_given_true,
                "prob_given_false": entity.prob_given_false,
            }

        return {
            "area_name": area_name,
            "current_prior": area.area_prior(),
            "global_prior": area.prior.global_prior,
            "time_prior": area.prior.time_prior,
            "prior_entity_ids": area.prior.sensor_ids,
            "total_entities": len(area.entities.entities),
            "entity_states": entity_states,
            "likelihoods": likelihood_data,
            "update_timestamp": dt_util.utcnow().isoformat(),
        }

    except Exception as err:
        error_msg = f"Failed to run analysis for {area_name}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _reset_entities(hass: HomeAssistant, call: ServiceCall) -> None:
    """Reset all entity probabilities and learned data.

    Supports area_name parameter. If area_name is "all" or ALL_AREAS_IDENTIFIER,
    resets entities for all areas.

    Backward compatibility: Supports deprecated 'entry_id' parameter.
    """
    area_name = _get_area_name_from_service_call(hass, call)

    try:
        coordinator = _get_coordinator(hass)

        if area_name == ALL_AREAS_IDENTIFIER:
            _LOGGER.info("Resetting entities for all areas")
            for area_name_item in coordinator.get_area_names():
                area = coordinator.get_area_or_default(area_name_item)
                await area.entities.cleanup()
            await coordinator.async_refresh()
            _LOGGER.info("Entity reset completed successfully for all areas")
        else:
            _validate_area_exists(coordinator, area_name)
            area = coordinator.get_area_or_default(area_name)

            _LOGGER.info("Resetting entities for area %s", area_name)
            await area.entities.cleanup()
            await coordinator.async_refresh()
            _LOGGER.info("Entity reset completed successfully for area %s", area_name)

    except Exception as err:
        error_msg = f"Failed to reset entities for {area_name}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _get_entity_metrics(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Get basic entity metrics for diagnostics.

    Supports area_name parameter. If area_name is "all" or ALL_AREAS_IDENTIFIER,
    returns metrics for all areas.

    Backward compatibility: Supports deprecated 'entry_id' parameter.
    """
    area_name = _get_area_name_from_service_call(hass, call)

    try:
        coordinator = _get_coordinator(hass)

        if area_name == ALL_AREAS_IDENTIFIER:
            # Aggregate metrics from all areas
            all_metrics = {}
            for area_name_item in coordinator.get_area_names():
                area = coordinator.get_area_or_default(area_name_item)
                entities = area.entities.entities
                total_entities = len(entities)
                active_entities = sum(1 for e in entities.values() if e.evidence)
                available_entities = sum(1 for e in entities.values() if e.available)
                unavailable_entities = sum(
                    1 for e in entities.values() if not e.available
                )
                decaying_entities = sum(
                    1 for e in entities.values() if e.decay.is_decaying
                )

                all_metrics[area_name_item] = {
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
                    "activity_percentage": round(
                        (active_entities / total_entities * 100), 1
                    )
                    if total_entities > 0
                    else 0,
                    "summary": f"{total_entities} total entities: {active_entities} active, {available_entities} available, {unavailable_entities} unavailable, {decaying_entities} decaying",
                }

            _LOGGER.info("Retrieved entity metrics for all areas")
            return {"areas": all_metrics}

        _validate_area_exists(coordinator, area_name)
        area = coordinator.get_area_or_default(area_name)

        entities = area.entities.entities
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

        _LOGGER.info("Retrieved entity metrics for area %s", area_name)

    except Exception as err:
        error_msg = f"Failed to get entity metrics for {area_name}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"metrics": metrics}


async def _get_problematic_entities(
    hass: HomeAssistant, call: ServiceCall
) -> dict[str, Any]:
    """Get entities that may need attention.

    Supports area_name parameter. If area_name is "all" or ALL_AREAS_IDENTIFIER,
    returns problematic entities for all areas.

    Backward compatibility: Supports deprecated 'entry_id' parameter.
    """
    area_name = _get_area_name_from_service_call(hass, call)

    try:
        coordinator = _get_coordinator(hass)
        now = dt_util.utcnow()

        if area_name == ALL_AREAS_IDENTIFIER:
            # Aggregate problematic entities from all areas
            all_problems = {}
            for area_name_item in coordinator.get_area_names():
                area = coordinator.get_area_or_default(area_name_item)
                entities = area.entities.entities
                unavailable = [eid for eid, e in entities.items() if not e.available]
                stale_updates = [
                    eid
                    for eid, e in entities.items()
                    if e.last_updated
                    and (now - ensure_timezone_aware(e.last_updated)).total_seconds()
                    > 3600
                ]

                all_problems[area_name_item] = {
                    "unavailable": unavailable,
                    "stale_updates": stale_updates,
                    "total_problems": len(unavailable) + len(stale_updates),
                    "summary": f"Found {len(unavailable)} unavailable and {len(stale_updates)} stale entities out of {len(entities)} total",
                }

            _LOGGER.info("Retrieved problematic entities for all areas")
            return {"areas": all_problems}

        _validate_area_exists(coordinator, area_name)
        area = coordinator.get_area_or_default(area_name)

        entities = area.entities.entities
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

        _LOGGER.info("Retrieved problematic entities for area %s", area_name)

    except Exception as err:
        error_msg = f"Failed to get problematic entities for {area_name}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"problems": problems}


async def _get_area_status(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Get current area occupancy status and confidence.

    Supports area_name parameter. If area_name is "all" or ALL_AREAS_IDENTIFIER,
    returns status for all areas.

    Backward compatibility: Supports deprecated 'entry_id' parameter.
    """
    area_name = _get_area_name_from_service_call(hass, call)

    try:
        coordinator = _get_coordinator(hass)

        def _format_confidence(prob: float | None) -> tuple[str, str]:
            """Format confidence level from probability."""
            if prob is not None:
                if prob > 0.8:
                    return ("high", "Very confident in occupancy status")
                if prob > 0.6:
                    return ("medium-high", "Fairly confident in occupancy status")
                if prob > 0.2:
                    return ("medium", "Moderate confidence in occupancy status")
                return ("low", "Low confidence in occupancy status")
            return ("unknown", "Unable to determine confidence level")

        if area_name == ALL_AREAS_IDENTIFIER:
            # Get status for all areas
            all_status = {}
            for area_name_item in coordinator.get_area_names():
                area = coordinator.get_area_or_default(area_name_item)
                occupancy_probability = area.probability()
                confidence_level, confidence_description = _format_confidence(
                    occupancy_probability
                )

                entities = area.entities.entities
                metrics = {
                    "total_entities": len(entities),
                    "active_entities": sum(1 for e in entities.values() if e.evidence),
                    "available_entities": sum(
                        1 for e in entities.values() if e.available
                    ),
                    "unavailable_entities": sum(
                        1 for e in entities.values() if not e.available
                    ),
                    "decaying_entities": sum(
                        1 for e in entities.values() if e.decay.is_decaying
                    ),
                }

                all_status[area_name_item] = {
                    "area_name": area_name_item,
                    "occupied": area.occupied(),
                    "occupancy_probability": round(occupancy_probability, 4)
                    if occupancy_probability is not None
                    else None,
                    "area_baseline_prior": round(area.area_prior(), 4),
                    "confidence_level": confidence_level,
                    "confidence_description": confidence_description,
                    "entity_summary": {
                        "total_entities": metrics["total_entities"],
                        "active_entities": metrics["active_entities"],
                        "available_entities": metrics["available_entities"],
                        "unavailable_entities": metrics["unavailable_entities"],
                        "decaying_entities": metrics["decaying_entities"],
                    },
                    "status_summary": f"Area '{area_name_item}' is {'occupied' if area.occupied() else 'not occupied'} with {confidence_level} confidence ({round(occupancy_probability * 100, 1) if occupancy_probability else 0}% probability)",
                }

            _LOGGER.info("Retrieved area status for all areas")
            return {"areas": all_status}

        _validate_area_exists(coordinator, area_name)
        area = coordinator.get_area_or_default(area_name)

        # Get current occupancy state
        occupancy_probability = area.probability()
        confidence_level, confidence_description = _format_confidence(
            occupancy_probability
        )

        # Get entity metrics for additional context
        entities = area.entities.entities
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
            "occupied": area.occupied(),
            "occupancy_probability": round(occupancy_probability, 4)
            if occupancy_probability is not None
            else None,
            "area_baseline_prior": round(area.area_prior(), 4),
            "confidence_level": confidence_level,
            "confidence_description": confidence_description,
            "entity_summary": {
                "total_entities": metrics["total_entities"],
                "active_entities": metrics["active_entities"],
                "available_entities": metrics["available_entities"],
                "unavailable_entities": metrics["unavailable_entities"],
                "decaying_entities": metrics["decaying_entities"],
            },
            "status_summary": f"Area '{area_name}' is {'occupied' if area.occupied() else 'not occupied'} with {confidence_level} confidence ({round(occupancy_probability * 100, 1) if occupancy_probability else 0}% probability)",
        }

        _LOGGER.info("Retrieved area status for area %s", area_name)

    except Exception as err:
        error_msg = f"Failed to get area status for {area_name}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return {"area_status": status}


def _create_area_selector_schema(hass: HomeAssistant) -> vol.Schema:
    """Create dynamic schema with area selector options.

    Gets area names from coordinator if available, otherwise falls back to
    simple text field with "all" option.

    Args:
        hass: Home Assistant instance

    Returns:
        vol.Schema with SelectSelector for area_name field
    """
    # Try to get coordinator to build dynamic options
    coordinator = hass.data.get(DOMAIN)
    options = [{"value": "all", "label": "All Areas"}]

    if coordinator is not None:
        try:
            area_names = coordinator.get_area_names()
            options.extend(
                [{"value": area_name, "label": area_name} for area_name in area_names]
            )
        except (AttributeError, RuntimeError, ValueError) as err:
            # If coordinator exists but get_area_names fails, use fallback
            _LOGGER.debug(
                "Could not get area names from coordinator, using fallback schema: %s",
                err,
            )

    return vol.Schema(
        {
            vol.Optional("area_name", default="all"): SelectSelector(
                SelectSelectorConfig(options=options, mode=SelectSelectorMode.DROPDOWN)
            ),
            vol.Optional("entry_id"): str,  # Deprecated, for backward compatibility
        }
    )


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Create dynamic schema with area selector options
    # Schema will be populated with actual area names from coordinator
    area_name_schema = _create_area_selector_schema(hass)

    # Create async wrapper functions to properly handle the service calls

    async def handle_run_analysis(call: ServiceCall) -> dict[str, Any]:
        return await _run_analysis(hass, call)

    async def handle_reset_entities(call: ServiceCall) -> None:
        return await _reset_entities(hass, call)

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
        schema=area_name_schema,
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN, "reset_entities", handle_reset_entities, schema=area_name_schema
    )

    hass.services.async_register(
        DOMAIN,
        "get_entity_metrics",
        handle_get_entity_metrics,
        schema=area_name_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_problematic_entities",
        handle_get_problematic_entities,
        schema=area_name_schema,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "get_area_status",
        handle_get_area_status,
        schema=area_name_schema,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("Registered %d services for %s integration", 5, DOMAIN)
