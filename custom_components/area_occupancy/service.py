"""Service definitions for the Area Occupancy Detection integration."""

import logging
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .utils import get_coordinator

if TYPE_CHECKING:
    from .area.area import Area

_LOGGER = logging.getLogger(__name__)


def _collect_entity_states(hass: HomeAssistant, area: "Area") -> dict[str, str]:
    """Collect current states for all entities in an area.

    Args:
        hass: Home Assistant instance
        area: The area to collect entity states for

    Returns:
        Dictionary mapping entity_id to state (or "NOT_FOUND" if unavailable)
    """
    entity_states = {}
    for entity_id in area.entities.entities:
        state = hass.states.get(entity_id)
        if state:
            entity_states[entity_id] = state.state
        else:
            entity_states[entity_id] = "NOT_FOUND"
    return entity_states


def _collect_likelihood_data(area: "Area") -> dict[str, dict[str, Any]]:
    """Collect likelihood data for all entities in an area.

    Args:
        area: The area to collect likelihood data for

    Returns:
        Dictionary mapping entity_id to likelihood data dict
    """
    likelihood_data = {}
    for entity_id, entity in area.entities.entities.items():
        likelihood_data[entity_id] = {
            "type": entity.type.input_type.value,
            "weight": entity.type.weight,
            "prob_given_true": entity.prob_given_true,
            "prob_given_false": entity.prob_given_false,
        }
    return likelihood_data


def _build_analysis_data(
    hass: HomeAssistant, area: "Area", area_name: str
) -> dict[str, Any]:
    """Build analysis data dictionary for an area.

    Args:
        hass: Home Assistant instance
        area: The area to build analysis data for
        area_name: The name of the area

    Returns:
        Dictionary containing analysis data for the area
    """
    entity_states = _collect_entity_states(hass, area)
    likelihood_data = _collect_likelihood_data(area)

    return {
        "area_name": area_name,
        "current_prior": area.area_prior(),
        "global_prior": area.prior.global_prior,
        "time_prior": area.prior.time_prior,
        "prior_entity_ids": area.prior.sensor_ids,
        "total_entities": len(area.entities.entities),
        "entity_states": entity_states,
        "likelihoods": likelihood_data,
    }


async def _run_analysis(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of sensor likelihoods.

    Always runs analysis for all areas.
    """
    try:
        coordinator = get_coordinator(hass)

        _LOGGER.info("Running analysis for all areas")
        await coordinator.run_analysis()

        # Aggregate data from all areas
        all_areas_data = {}
        for area_name_item in coordinator.get_area_names():
            area = coordinator.get_area(area_name_item)
            all_areas_data[area_name_item] = _build_analysis_data(
                hass, area, area_name_item
            )

        return {
            "areas": all_areas_data,
            "update_timestamp": dt_util.utcnow().isoformat(),
        }
    except Exception as err:
        error_msg = f"Failed to run analysis: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def _run_nightly_tasks(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger nightly aggregation + correlation tasks."""
    try:
        coordinator = get_coordinator(hass)

        _LOGGER.info("Running nightly aggregation + correlation tasks for all areas")
        summary = await coordinator.run_interval_aggregation_job(dt_util.utcnow())

        return {
            "results": summary,
            "update_timestamp": dt_util.utcnow().isoformat(),
        }
    except Exception as err:
        error_msg = f"Failed to run nightly tasks: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Create async wrapper function to properly handle the service call
    async def handle_run_analysis(call: ServiceCall) -> dict[str, Any]:
        return await _run_analysis(hass, call)

    async def handle_run_nightly_tasks(call: ServiceCall) -> dict[str, Any]:
        return await _run_nightly_tasks(hass, call)

    # Register service with async wrapper function
    hass.services.async_register(
        DOMAIN,
        "run_analysis",
        handle_run_analysis,
        schema=None,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "run_nightly_tasks",
        handle_run_nightly_tasks,
        schema=None,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("Registered %d service for %s integration", 2, DOMAIN)
