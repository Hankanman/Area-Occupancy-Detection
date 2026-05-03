"""Service definitions for the Area Occupancy Detection integration."""

import contextlib
from dataclasses import asdict
import logging
import time
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.util import dt as dt_util

from .const import CONF_AREA_ID, DEVICE_SW_VERSION, DOMAIN
from .data.purpose import get_default_decay_half_life
from .utils import get_coordinator

if TYPE_CHECKING:
    from .area.area import Area
    from .coordinator import AreaOccupancyCoordinator

_LOGGER = logging.getLogger(__name__)

PURGE_AREA_HISTORY_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_AREA_ID): vol.All(str, vol.Length(min=1)),
    }
)


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
        gaussian_params = getattr(entity, "learned_gaussian_params", None)
        analysis_data = asdict(gaussian_params) if gaussian_params else None
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

    # Resolve half_life: use configured value, or derive from purpose if set to auto (0)
    half_life = area.config.decay.half_life
    if half_life == 0:
        half_life = get_default_decay_half_life(area.config.purpose)

    data = {
        "area_name": area_name,
        "purpose": area.purpose.name,
        "half_life": half_life,
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


async def _export_config(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Export the complete integration configuration as YAML."""
    try:
        coordinator = get_coordinator(hass)
        config_entry = coordinator.config_entry

        config = dict(config_entry.data) | dict(config_entry.options)

        # Reorder area dicts so area_id comes first
        if "areas" in config:
            config["areas"] = [
                {
                    "area_id": area["area_id"],
                    **{k: v for k, v in area.items() if k != "area_id"},
                }
                for area in config["areas"]
            ]
    except Exception as err:
        error_msg = f"Failed to export config: {err}"
        _LOGGER.error(error_msg)
        raise HomeAssistantError(error_msg) from err
    else:
        return config


def _find_area_by_area_id(
    coordinator: Any, area_id: str
) -> tuple[str | None, "Area | None"]:
    """Look up an area by its Home Assistant area_id.

    Args:
        coordinator: AreaOccupancyCoordinator instance
        area_id: Home Assistant area_id stored on each area's config

    Returns:
        Tuple of (area_name, area) — both None when no match is found.
    """
    for area_name, area in coordinator.areas.items():
        if area.config.area_id == area_id:
            return area_name, area
    return None, None


async def async_purge_area_data(
    hass: HomeAssistant,
    coordinator: "AreaOccupancyCoordinator",
    area_name: str,
    area: "Area",
) -> dict[str, Any]:
    """Purge DB rows + in-memory state for a single configured area.

    Deletes all database rows for the area (intervals, priors, correlations,
    caches, etc.) without removing the area from configuration. The area's
    in-memory prior cache is cleared and a coordinator refresh is requested so
    the UI immediately reflects the purge.

    Shared by the public ``purge_area_history`` service and the options-flow
    "Reset learning" action — both want the same effect (data wiped, area
    config preserved). Caller is responsible for the area-id → name lookup
    and any user-facing validation messaging; this helper assumes the area
    exists in the coordinator.

    Returns the same result dict the service handler exposes:
    ``{"area_id", "area_name", "entities_deleted", "shell_repersisted",
    "purged_at"}``. Raises ``HomeAssistantError`` on hard DB failure.
    """
    try:
        deleted = await hass.async_add_executor_job(
            coordinator.db.delete_area_data, area_name
        )
    except Exception as err:
        error_msg = f"Failed to purge database records for area '{area_name}': {err}"
        _LOGGER.exception(error_msg)
        raise HomeAssistantError(error_msg) from err

    # Reset in-memory prior state so next calculation rebuilds cleanly.
    with contextlib.suppress(AttributeError):
        area.prior.clear_cache()

    # Re-persist the area shell so subsequent operations still find it.
    # A failure here does not invalidate the purge itself (the user-visible
    # history has been deleted) — the shell is re-created on the next save
    # cycle. Surfaced via the response and at warning level so callers can
    # see partial failures without the service raising.
    shell_repersisted = True
    try:
        await hass.async_add_executor_job(coordinator.db.save_area_data, area_name)
    except Exception:  # noqa: BLE001
        shell_repersisted = False
        _LOGGER.warning(
            "Failed to re-persist area shell for '%s' after purge; "
            "it will be recreated on the next save cycle",
            area_name,
            exc_info=True,
        )

    # Reload priors/correlations/entity state from DB (now empty for this area).
    try:
        await coordinator.db.load_data()
    except Exception:  # noqa: BLE001
        _LOGGER.warning("db.load_data() after purge raised; continuing", exc_info=True)

    try:
        await coordinator.async_refresh_correlations()
    except Exception:  # noqa: BLE001
        _LOGGER.warning(
            "async_refresh_correlations() after purge raised; continuing",
            exc_info=True,
        )

    try:
        await coordinator.async_request_refresh()
    except Exception:  # noqa: BLE001
        _LOGGER.warning(
            "async_request_refresh() after purge raised; continuing", exc_info=True
        )

    return {
        "area_id": area.config.area_id,
        "area_name": area_name,
        "entities_deleted": int(deleted),
        "shell_repersisted": shell_repersisted,
        "purged_at": dt_util.utcnow().isoformat(),
    }


async def _purge_area_history(hass: HomeAssistant, call: ServiceCall) -> dict[str, Any]:
    """Service handler: purge learned history for a single configured area.

    Resolves the caller-supplied ``area_id`` and delegates to
    ``async_purge_area_data``. The service-call layer owns the user-facing
    validation messaging; the helper is kept ServiceCall-free so the
    options-flow "Reset learning" action can reuse it.
    """
    coordinator = get_coordinator(hass)
    area_id = call.data[CONF_AREA_ID]

    area_name, area = _find_area_by_area_id(coordinator, area_id)
    if area_name is None or area is None:
        known = sorted(
            a.config.area_id
            for a in coordinator.areas.values()
            if isinstance(a.config.area_id, str)
        )
        raise ServiceValidationError(
            f"No configured area found for area_id '{area_id}'. "
            f"Known area_ids: {', '.join(known) if known else '(none)'}"
        )

    _LOGGER.info(
        "Purging learned history for area '%s' (area_id=%s) on user request",
        area_name,
        area_id,
    )

    return await async_purge_area_data(hass, coordinator, area_name, area)


async def async_setup_services(hass: HomeAssistant) -> None:
    """Register custom services for area occupancy."""

    # Create async wrapper function to properly handle the service call
    async def handle_run_analysis(call: ServiceCall) -> dict[str, Any]:
        return await _run_analysis(hass, call)

    async def handle_export_config(call: ServiceCall) -> dict[str, Any]:
        return await _export_config(hass, call)

    async def handle_purge_area_history(call: ServiceCall) -> dict[str, Any]:
        return await _purge_area_history(hass, call)

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
        "export_config",
        handle_export_config,
        schema=None,
        supports_response=SupportsResponse.ONLY,
    )

    hass.services.async_register(
        DOMAIN,
        "purge_area_history",
        handle_purge_area_history,
        schema=PURGE_AREA_HISTORY_SCHEMA,
        supports_response=SupportsResponse.OPTIONAL,
    )
