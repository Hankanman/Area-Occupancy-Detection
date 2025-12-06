"""Service definitions for the Area Occupancy Detection integration."""

import logging
import time
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import DEVICE_SW_VERSION, DOMAIN
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
        # Get runtime likelihood values (uses Gaussian params if available, falls back to defaults)
        prob_given_true, prob_given_false = entity.get_likelihoods()

        # Prepare active_range for JSON serialization (only for numeric sensors)
        # Binary sensors shouldn't show active_range
        active_range_val = None
        if entity.active_range and not entity.active_states:
            # Only include active_range for numeric sensors (those without active_states)
            try:
                active_range_val = []
                # Ensure active_range is iterable (tuple or list)
                # Mock objects might report having active_range but fail iteration if not configured
                range_iter = entity.active_range
                if not isinstance(range_iter, (tuple, list)) and not hasattr(
                    range_iter, "__iter__"
                ):
                    # Fallback for unconfigured mocks
                    active_range_val = None
                else:
                    for val in range_iter:
                        # JSON doesn't support infinity, use None for open bounds
                        if val == float("inf") or val == float("-inf"):
                            active_range_val.append(None)
                        else:
                            active_range_val.append(val)
            except TypeError:
                # Handle case where iteration fails (e.g. non-iterable Mock)
                active_range_val = None

        raw_data = {
            "type": entity.type.input_type.value,
            "weight": entity.type.weight,
            "prob_given_true": prob_given_true,  # Runtime calculated value
            "prob_given_false": prob_given_false,  # Runtime calculated value
            "active_states": entity.active_states,
            "active_range": active_range_val,
            "is_active": entity.active,
        }

        # Always include analysis data and errors (even if None) for visibility
        analysis_data = getattr(entity, "learned_gaussian_params", None)
        analysis_error = getattr(entity, "analysis_error", None)
        correlation_type = getattr(entity, "correlation_type", None)

        raw_data["analysis_data"] = analysis_data
        raw_data["analysis_error"] = analysis_error
        raw_data["correlation_type"] = correlation_type

        # Filter out keys with None values, but keep analysis_data, analysis_error, and correlation_type
        # even if None so users can see which entities have been analyzed
        filtered_data = {
            k: v
            for k, v in raw_data.items()
            if k in ("analysis_data", "analysis_error", "correlation_type")
            or v is not None
        }
        likelihood_data[entity_id] = filtered_data
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

    data = {
        "area_name": area_name,
        "purpose": area.purpose.name,
        "current_probability": area.probability(),
        "current_occupied": area.occupied(),
        "current_threshold": area.threshold(),
        "current_prior": area.area_prior(),
        "global_prior": area.prior.global_prior,
        "time_prior": area.prior.time_prior,
        "prior_entity_ids": area.prior.sensor_ids,
        "total_entities": len(area.entities.entities),
        "entity_states": entity_states,
        "likelihoods": likelihood_data,
    }
    # Filter out keys with None values
    return {k: v for k, v in data.items() if v is not None}


async def _run_analysis(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Manually trigger an update of sensor likelihoods.

    Always runs analysis for all areas.
    """
    try:
        coordinator = get_coordinator(hass)

        _LOGGER.info("Running analysis for all areas")
        analysis_start_time = time.perf_counter()
        await coordinator.run_analysis()
        analysis_time_ms = (time.perf_counter() - analysis_start_time) * 1000

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
            "analysis_time_ms": analysis_time_ms,
            "device_sw_version": DEVICE_SW_VERSION,
        }
    except Exception as err:
        error_msg = f"Failed to run analysis: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Create async wrapper function to properly handle the service call
    async def handle_run_analysis(call: ServiceCall) -> dict[str, Any]:
        return await _run_analysis(hass, call)

    # Register service with async wrapper function
    hass.services.async_register(
        DOMAIN,
        "run_analysis",
        handle_run_analysis,
        schema=None,
        supports_response=SupportsResponse.ONLY,
    )
