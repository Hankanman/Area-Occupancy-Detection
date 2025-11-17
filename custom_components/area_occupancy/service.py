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
from .utils import get_coordinator, normalize_area_name, validate_area_exists

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

    Supports area_name parameter. If area_name is "all" or ALL_AREAS_IDENTIFIER,
    runs analysis for all areas.
    """
    area_name = normalize_area_name(call.data.get("area_name", "all"))

    try:
        coordinator = get_coordinator(hass)

        if area_name == ALL_AREAS_IDENTIFIER:
            _LOGGER.info("Running analysis for all areas")
            await coordinator.run_analysis()

            # Aggregate data from all areas
            all_areas_data = {}
            for area_name_item in coordinator.get_area_names():
                area = coordinator.get_area_or_default(area_name_item)
                all_areas_data[area_name_item] = _build_analysis_data(
                    hass, area, area_name_item
                )

            return {
                "areas": all_areas_data,
                "update_timestamp": dt_util.utcnow().isoformat(),
            }

        # Run for specific area
        validate_area_exists(coordinator, area_name)
        area = coordinator.get_area_or_default(area_name)

        _LOGGER.info("Running analysis for area %s", area_name)
        await coordinator.run_analysis()

        result = _build_analysis_data(hass, area, area_name)
        result["update_timestamp"] = dt_util.utcnow().isoformat()
    except Exception as err:
        error_msg = f"Failed to run analysis for {area_name}: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return result


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
        }
    )


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Create dynamic schema with area selector options
    # Schema will be populated with actual area names from coordinator
    area_name_schema = _create_area_selector_schema(hass)

    # Create async wrapper function to properly handle the service call
    async def handle_run_analysis(call: ServiceCall) -> dict[str, Any]:
        return await _run_analysis(hass, call)

    # Register service with async wrapper function
    hass.services.async_register(
        DOMAIN,
        "run_analysis",
        handle_run_analysis,
        schema=area_name_schema,
        supports_response=SupportsResponse.ONLY,
    )

    _LOGGER.info("Registered %d service for %s integration", 1, DOMAIN)
