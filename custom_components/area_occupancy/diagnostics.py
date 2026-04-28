"""Diagnostics support for Area Occupancy Detection.

Exposes a structured, JSON-serializable snapshot of the integration's
runtime state so users (and triagers) can inspect priors, per-entity
weights/likelihoods, current evidence, decay state, learned correlations,
sensor health, and database row counts without enabling DEBUG logging.

Each section is captured defensively — a failure in one area must not
prevent the rest of the diagnostic from being produced.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.core import HomeAssistant

from .const import CONF_VERSION, CONF_VERSION_MINOR, DEVICE_SW_VERSION
from .db import queries

if TYPE_CHECKING:
    from homeassistant.config_entries import ConfigEntry

    from .area.area import Area
    from .coordinator import AreaOccupancyCoordinator
    from .data.entity import Entity

_LOGGER = logging.getLogger(__name__)

# Canonical DB exception tuple used across db/queries.py and db/operations.py.
_DB_EXCEPTIONS: tuple[type[BaseException], ...] = (
    SQLAlchemyError,
    ValueError,
    TypeError,
    RuntimeError,
    OSError,
)


def _isoformat(value: datetime | None) -> str | None:
    """Return an ISO-8601 string for a datetime, or None."""
    return value.isoformat() if value is not None else None


def _entity_snapshot(entity: Entity, correlation: float | None) -> dict[str, Any]:
    """Capture a JSON-friendly snapshot of an Entity's diagnostic fields."""
    decay = entity.decay
    gaussian = entity.learned_gaussian_params
    return {
        "entity_id": entity.entity_id,
        "input_type": entity.type.input_type.value,
        "weight": entity.type.weight,
        "prob_given_true": entity.prob_given_true,
        "prob_given_false": entity.prob_given_false,
        "evidence": entity.evidence,
        "previous_evidence": entity.previous_evidence,
        "last_updated": _isoformat(entity.last_updated),
        "analysis_error": entity.analysis_error,
        "correlation_type": entity.correlation_type,
        "correlation_strength": correlation,
        "learned_active_range": (
            list(entity.learned_active_range)
            if entity.learned_active_range is not None
            else None
        ),
        "learned_gaussian_params": (
            {
                "mean_occupied": gaussian.mean_occupied,
                "std_occupied": gaussian.std_occupied,
                "mean_unoccupied": gaussian.mean_unoccupied,
                "std_unoccupied": gaussian.std_unoccupied,
            }
            if gaussian is not None
            else None
        ),
        "decay": {
            "is_decaying": decay.is_decaying,
            "half_life": decay.half_life,
            "decay_start": _isoformat(decay.decay_start),
            "decay_factor": decay.decay_factor,
        },
    }


def _area_config_snapshot(area: Area) -> dict[str, Any]:
    """Capture the area's configuration shape (counts, not entity IDs)."""
    config = area.config
    sensors = config.sensors
    weights = config.weights
    decay = config.decay
    wasp = config.wasp_in_box
    sensor_counts = {
        "motion": len(sensors.motion),
        "media": len(sensors.media),
        "appliance": len(sensors.appliance),
        "door": len(sensors.door),
        "window": len(sensors.window),
        "cover": len(sensors.cover),
        "power": len(sensors.power),
        "illuminance": len(sensors.illuminance),
        "humidity": len(sensors.humidity),
        "temperature": len(sensors.temperature),
        "co2": len(sensors.co2),
        "co": len(sensors.co),
        "sound_pressure": len(sensors.sound_pressure),
        "pressure": len(sensors.pressure),
        "air_quality": len(sensors.air_quality),
        "voc": len(sensors.voc),
        "pm25": len(sensors.pm25),
        "pm10": len(sensors.pm10),
    }
    return {
        "decay": {"enabled": decay.enabled, "half_life": decay.half_life},
        "wasp_in_box": {
            "enabled": wasp.enabled,
            "motion_timeout": wasp.motion_timeout,
            "weight": wasp.weight,
            "max_duration": wasp.max_duration,
            "verification_delay": wasp.verification_delay,
        },
        "weights": {
            "motion": weights.motion,
            "media": weights.media,
            "appliance": weights.appliance,
            "door": weights.door,
            "window": weights.window,
            "cover": weights.cover,
            "environmental": weights.environmental,
            "power": weights.power,
            "wasp": weights.wasp,
        },
        "min_prior_override": getattr(config, "min_prior_override", None),
        "motion_timeout": sensors.motion_timeout,
        "sensor_counts": sensor_counts,
    }


def _health_snapshot(area: Area) -> dict[str, Any]:
    """Capture cached health issues without re-running checks.

    ``HealthIssue.entity_id`` and ``HealthIssue.input_type`` are both
    ``None`` for pipeline-scope issues (``insufficient_priors`` etc.) — the
    issue applies to the whole area. Both are null-safe here so a
    pipeline-scope issue doesn't poison the entire area's diagnostic.
    """
    monitor = area.health_monitor
    return {
        "issue_count": len(monitor.issues),
        "last_check": _isoformat(monitor.last_check),
        "issues": [
            {
                "entity_id": issue.entity_id,
                "issue_type": issue.issue_type.value,
                "input_type": (
                    issue.input_type.value if issue.input_type is not None else None
                ),
                "since": _isoformat(issue.since),
                "duration_hours": issue.duration_hours,
                "details": issue.details,
            }
            for issue in monitor.issues
        ],
    }


def _area_snapshot(
    coordinator: AreaOccupancyCoordinator, area_name: str, area: Area
) -> dict[str, Any]:
    """Capture a snapshot for a single area, defending each subsection.

    Each subsection is wrapped in a broad ``except Exception`` because the
    underlying calls touch a wide surface (probability math, state lookups,
    cache reads, sub-property access on lazily-initialized components) and
    we don't want a single failure to discard the rest of the diagnostic.
    Failures are surfaced both in the diagnostic dump (as a
    ``<section>_error`` key) and in the HA log with a stack trace, so a
    triager can act on them.
    """
    snapshot: dict[str, Any] = {"area_name": area_name}

    try:
        snapshot["area_id"] = area.config.area_id
        snapshot["purpose"] = area.config.purpose
        snapshot["threshold"] = area.config.threshold
    except (AttributeError, KeyError) as err:
        _LOGGER.warning(
            "Diagnostics: failed to read area config for '%s': %s",
            area_name,
            err,
            exc_info=True,
        )
        snapshot["config_error"] = repr(err)

    try:
        snapshot["current"] = {
            "probability": area.probability(),
            "occupied": area.occupied(),
            "decay_factor": area.decay(),
            "active_entity_count": len(area.entities.active_entities),
            "decaying_entity_count": len(area.entities.decaying_entities),
            "entity_count": len(area.entities.entities),
        }
    except Exception as err:  # noqa: BLE001 — see docstring
        _LOGGER.warning(
            "Diagnostics: failed to compute current state for '%s': %s",
            area_name,
            err,
            exc_info=True,
        )
        snapshot["current_error"] = repr(err)

    try:
        snapshot["prior"] = area.prior.diagnostic_snapshot()
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning(
            "Diagnostics: failed to read prior snapshot for '%s': %s",
            area_name,
            err,
            exc_info=True,
        )
        snapshot["prior_error"] = repr(err)

    try:
        snapshot["config"] = _area_config_snapshot(area)
    except (AttributeError, TypeError) as err:
        _LOGGER.warning(
            "Diagnostics: failed to capture config snapshot for '%s': %s",
            area_name,
            err,
            exc_info=True,
        )
        snapshot["config_snapshot_error"] = repr(err)

    try:
        correlations = coordinator.get_cached_correlations(area_name)
        snapshot["entities"] = [
            _entity_snapshot(entity, correlations.get(entity.entity_id))
            for entity in area.entities.entities.values()
        ]
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning(
            "Diagnostics: failed to capture entity snapshots for '%s': %s",
            area_name,
            err,
            exc_info=True,
        )
        snapshot["entities_error"] = repr(err)

    try:
        snapshot["health"] = _health_snapshot(area)
    except (AttributeError, TypeError) as err:
        _LOGGER.warning(
            "Diagnostics: failed to capture health snapshot for '%s': %s",
            area_name,
            err,
            exc_info=True,
        )
        snapshot["health_error"] = repr(err)

    return snapshot


def _collect_db_stats(coordinator: AreaOccupancyCoordinator) -> dict[str, Any]:
    """Collect database row counts and per-area cache freshness.

    Runs on the executor pool — caller must wrap with async_add_executor_job.
    """
    db = coordinator.db
    stats: dict[str, Any] = {}

    try:
        with db.get_session() as session:
            stats["interval_count"] = session.query(db.Intervals).count()
            stats["prior_count"] = session.query(db.Priors).count()
            stats["correlation_count"] = session.query(db.Correlations).count()
            stats["entity_count"] = session.query(db.Entities).count()
            stats["area_count"] = session.query(db.Areas).count()
    except _DB_EXCEPTIONS as err:
        _LOGGER.warning(
            "Diagnostics: failed to read DB row counts: %s", err, exc_info=True
        )
        stats["counts_error"] = repr(err)

    cache_status: dict[str, dict[str, Any]] = {}
    for area_name in coordinator.areas:
        try:
            cache_status[area_name] = {
                "valid": queries.is_occupied_intervals_cache_valid(db, area_name),
            }
        except _DB_EXCEPTIONS as err:
            _LOGGER.warning(
                "Diagnostics: failed to check cache validity for '%s': %s",
                area_name,
                err,
                exc_info=True,
            )
            cache_status[area_name] = {"error": repr(err)}
    stats["occupied_intervals_cache"] = cache_status

    return stats


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    """Return a diagnostic snapshot of integration runtime state."""
    coordinator: AreaOccupancyCoordinator | None = getattr(entry, "runtime_data", None)
    if coordinator is None:
        return {
            "error": "coordinator_not_initialized",
            "entry_id": entry.entry_id,
            "entry_state": str(entry.state),
        }

    integration_section: dict[str, Any] = {
        "version": DEVICE_SW_VERSION,
        "config_version": CONF_VERSION,
        "config_version_minor": CONF_VERSION_MINOR,
        "entry_id": entry.entry_id,
        "entry_title": entry.title,
        "setup_complete": coordinator.setup_complete,
        "area_count": len(coordinator.areas),
    }
    try:
        integration_config = coordinator.integration_config
        integration_section["sleep_start"] = integration_config.sleep_start
        integration_section["sleep_end"] = integration_config.sleep_end
        integration_section["people_count"] = len(integration_config.people)
    except (AttributeError, TypeError, KeyError) as err:
        _LOGGER.warning(
            "Diagnostics: failed to read integration config: %s", err, exc_info=True
        )
        integration_section["config_error"] = repr(err)

    areas_section = [
        _area_snapshot(coordinator, area_name, area)
        for area_name, area in coordinator.areas.items()
    ]

    try:
        database_section = await hass.async_add_executor_job(
            _collect_db_stats, coordinator
        )
    except Exception as err:  # noqa: BLE001 — executor failure is opaque
        _LOGGER.warning(
            "Diagnostics: failed to dispatch DB stats collection: %s",
            err,
            exc_info=True,
        )
        database_section = {"error": repr(err)}

    return {
        "integration": integration_section,
        "areas": areas_section,
        "database": database_section,
    }
