"""Database CRUD operations."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

from homeassistant import helpers
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import (
    GLOBAL_PRIOR_HISTORY_COUNT,
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PROBABILITY,
    MIN_WEIGHT,
    RETENTION_DAYS,
)
from . import maintenance, queries
from .constants import (
    DEFAULT_ENTITY_PROB_GIVEN_FALSE,
    DEFAULT_ENTITY_PROB_GIVEN_TRUE,
    DEFAULT_ENTITY_WEIGHT,
)

ar = helpers.area_registry

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)


def _validate_area_data(
    db: AreaOccupancyDB, area_data: dict[str, Any], area_name_item: str
) -> list[tuple[str, str]]:
    """Validate area data and return list of validation errors.

    Args:
        db: Database instance
        area_data: Dictionary containing area data to validate
        area_name_item: Name of the area being validated (for error messages)

    Returns:
        List of (area_name, error_message) tuples for any validation failures
    """
    failures: list[tuple[str, str]] = []

    if not area_data.get("entry_id"):
        failures.append((area_name_item, "entry_id is empty or None"))
    if not area_data.get("area_name"):
        failures.append((area_name_item, "area_name is empty or None"))
    if not area_data.get("area_id"):
        failures.append((area_name_item, "area_id is empty or None"))
    if not area_data.get("purpose"):
        failures.append((area_name_item, "purpose is empty or None"))
    if area_data.get("threshold") is None:
        failures.append((area_name_item, "threshold is None"))

    return failures


async def load_data(db: AreaOccupancyDB) -> None:
    """Load the data from the database for all areas.

    This method iterates over all configured areas and loads data for each.
    """
    # Import here to avoid circular imports

    def _read_data_operation(area_name: str) -> tuple[Any, list[Any], list[str]]:
        """Read data WITHOUT lock (parallel-safe) for a specific area."""
        # Ensure tables exist (important for in-memory databases where
        # each connection gets its own database)
        if not maintenance.verify_all_tables_exist(db):
            maintenance.init_db(db)
        stale_entity_ids = []
        with db.get_session() as session:
            # Query by area_name instead of entry_id
            area = session.query(db.Areas).filter_by(area_name=area_name).first()
            entities = (
                session.query(db.Entities)
                .filter_by(area_name=area_name)
                .order_by(db.Entities.entity_id)
                .all()
            )
            if entities:
                # Get the area's entity manager to check if entities exist
                # Area is guaranteed to exist when area_name comes from get_area_names()
                area_data = db.coordinator.get_area(area_name)
                for entity_obj in entities:
                    # Check if entity exists in current coordinator config
                    try:
                        area_data.entities.get_entity(entity_obj.entity_id)
                    except ValueError:
                        # Entity not found in coordinator - check if it's stale
                        # EntityManager always has entity_ids property
                        current_entity_ids = set(area_data.entities.entity_ids)
                        if entity_obj.entity_id not in current_entity_ids:
                            stale_entity_ids.append(entity_obj.entity_id)
        return area, entities, stale_entity_ids

    def _delete_stale_operation(area_name: str, stale_ids: list[str]) -> None:
        """Delete stale entities (requires lock to prevent race conditions)."""
        with db.get_locked_session() as session:
            for entity_id in stale_ids:
                _LOGGER.info(
                    "Deleting stale entity %s from database for area %s (not in current config)",
                    entity_id,
                    area_name,
                )
                session.query(db.Entities).filter_by(
                    area_name=area_name, entity_id=entity_id
                ).delete()
            session.commit()

    try:
        # Load data for each configured area
        for area_name in db.coordinator.get_area_names():
            area_data = db.coordinator.get_area(area_name)

            # Phase 1: Read without lock (all instances in parallel)
            _area, entities, stale_ids = await db.hass.async_add_executor_job(
                _read_data_operation, area_name
            )

            # Update prior from GlobalPriors table
            global_prior_data = db.get_global_prior(area_name)
            if global_prior_data:
                area_data.prior.set_global_prior(global_prior_data["prior_value"])

            # Process entities
            if entities:
                for entity_obj in entities:
                    if entity_obj.entity_id in stale_ids:
                        # Skip stale entities, will be deleted in phase 2
                        continue

                    # Try to get existing entity from coordinator
                    try:
                        existing_entity = area_data.entities.get_entity(
                            entity_obj.entity_id
                        )
                        # Update existing entity with database values (preserve database timestamp)
                        existing_entity.update_decay(
                            entity_obj.decay_start,
                            entity_obj.is_decaying,
                        )
                        existing_entity.update_likelihood(
                            entity_obj.prob_given_true,
                            entity_obj.prob_given_false,
                        )
                        # DB weight takes priority over configured defaults when valid
                        if hasattr(existing_entity, "type") and hasattr(
                            existing_entity.type, "weight"
                        ):
                            try:
                                weight_val = float(entity_obj.weight)
                                if MIN_WEIGHT <= weight_val <= MAX_WEIGHT:
                                    existing_entity.type.weight = weight_val
                            except (TypeError, ValueError):
                                pass
                        existing_entity.last_updated = entity_obj.last_updated
                        existing_entity.previous_evidence = entity_obj.evidence
                    except ValueError:
                        # Entity should exist but doesn't - create it from database
                        # (This handles cases where we can't determine current config, like in tests)
                        _LOGGER.warning(
                            "Entity %s not found in coordinator for area %s but is in config, creating from database",
                            entity_obj.entity_id,
                            area_name,
                        )
                        new_entity = area_data.factory.create_from_db(entity_obj)
                        area_data.entities.add_entity(new_entity)

            # Phase 2: Only lock if cleanup needed (rare)
            if stale_ids:
                await db.hass.async_add_executor_job(
                    _delete_stale_operation, area_name, stale_ids
                )

    except (
        sa.exc.SQLAlchemyError,
        HomeAssistantError,
        TimeoutError,
        OSError,
        RuntimeError,
    ) as err:
        _LOGGER.error("Failed to load area occupancy data: %s", err)
        # Don't raise the error, just log it and continue
        # This allows the integration to start even if data loading fails


def save_area_data(db: AreaOccupancyDB, area_name: str | None = None) -> None:
    """Save the area data to the database.

    Args:
        db: Database instance
        area_name: Optional area name to save. If None, saves all areas.

    With single-instance architecture, no file lock is required.
    """
    # Determine which areas to save
    if area_name is not None:
        areas_to_save = [area_name]
    else:
        areas_to_save = db.coordinator.get_area_names()

    def _attempt(session: Any) -> bool:
        """Save area data for all configured areas."""
        failures: list[
            tuple[str, str]
        ] = []  # List of (area_name, error_message) tuples
        has_failures = False
        area_objects = []  # Collect all area objects for batch merge

        for area_name_item in areas_to_save:
            area_data_obj = db.coordinator.get_area(area_name_item)
            # Area is guaranteed to exist when area_name comes from get_area_names()
            # or when area_name is validated before calling this function
            cfg = area_data_obj.config

            # area_id is validated in config flow before areas are created
            area_data = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name_item,
                "area_id": cfg.area_id,
                "purpose": cfg.purpose,
                "threshold": cfg.threshold,
                "updated_at": dt_util.utcnow(),
            }

            # Validate required fields using helper method
            validation_failures = _validate_area_data(db, area_data, area_name_item)
            if validation_failures:
                for area_name_fail, error_msg in validation_failures:
                    _LOGGER.error(
                        "%s, cannot insert area '%s'", error_msg, area_name_fail
                    )
                failures.extend(validation_failures)
                has_failures = True
                continue

            # Collect area object for batch merge
            area_obj = db.Areas.from_dict(area_data)
            area_objects.append(area_obj)

        # Perform all merges in batch (SQLite requires individual merges due to upsert limitations,
        # but collecting first allows SQLAlchemy to optimize the batch)
        for area_obj in area_objects:
            session.merge(area_obj)

        if has_failures:
            # Log concise summary of all failures
            failed_areas = [f"{area} ({error})" for area, error in failures]
            _LOGGER.error(
                "Failed to save area data for %d area(s): %s",
                len(failures),
                "; ".join(failed_areas),
            )
            # Rollback and return False
            session.rollback()
            return False

        session.commit()
        return True

    try:
        # Retry with backoff under a file lock
        backoffs = [0.1, 0.25, 0.5, 1.0]
        for attempt, delay in enumerate(backoffs, start=1):
            try:
                with db.get_locked_session() as session:
                    success = _attempt(session)
                if not success:
                    raise ValueError(
                        "Area data validation failed; required fields missing or invalid"
                    )
                # Update debounce timestamp only after a successful attempt
                db.last_area_save_ts = time.monotonic()
                break
            except (sa.exc.OperationalError, sa.exc.TimeoutError) as err:
                _LOGGER.warning("save_area_data attempt %d failed: %s", attempt, err)
                if attempt == len(backoffs):
                    # Ensure next call is not debounced due to a recent success
                    db.last_area_save_ts = 0.0
                    raise
                time.sleep(delay)
    except (
        sa.exc.SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as err:
        _LOGGER.error("Failed to save area data: %s", err)
        raise


def save_entity_data(db: AreaOccupancyDB) -> None:
    """Save the entity data to the database for all areas.

    With single-instance architecture, no file lock is required.
    """

    def _iter_area_entities() -> Iterable[tuple[str, Any]]:
        """Yield (area_name, entity) tuples for all configured areas."""
        for area_name in db.coordinator.get_area_names():
            area_data = db.coordinator.get_area(area_name)
            entities_container = getattr(area_data.entities, "entities", None)
            if not entities_container:
                continue

            try:
                entities_iter = entities_container.values()
            except AttributeError:
                continue

            for entity in entities_iter:
                yield area_name, entity

    def _prepare_entity_payload(area_name: str, entity: Any) -> dict[str, Any] | None:
        """Prepare normalized entity data for persistence."""
        if not hasattr(entity, "type") or not entity.type:
            _LOGGER.warning(
                "Entity %s has no type information, skipping",
                getattr(entity, "entity_id", "unknown"),
            )
            return None

        entity_type = getattr(entity.type, "input_type", None)
        if entity_type is None:
            _LOGGER.warning("Entity %s has no input_type, skipping", entity.entity_id)
            return None

        # Normalize entity_type to plain string (handle Enum instances)
        entity_type_value = (
            entity_type.value if hasattr(entity_type, "value") else str(entity_type)
        )

        # Normalize values before persisting
        try:
            weight = float(getattr(entity.type, "weight", DEFAULT_ENTITY_WEIGHT))
        except (TypeError, ValueError):
            weight = DEFAULT_ENTITY_WEIGHT
        weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weight))

        try:
            prob_true = float(entity.prob_given_true)
        except (TypeError, ValueError):
            prob_true = DEFAULT_ENTITY_PROB_GIVEN_TRUE
        prob_true = max(MIN_PROBABILITY, min(MAX_PROBABILITY, prob_true))

        try:
            prob_false = float(entity.prob_given_false)
        except (TypeError, ValueError):
            prob_false = DEFAULT_ENTITY_PROB_GIVEN_FALSE
        prob_false = max(MIN_PROBABILITY, min(MAX_PROBABILITY, prob_false))

        last_updated = getattr(entity, "last_updated", None) or dt_util.utcnow()

        evidence_source = getattr(entity, "previous_evidence", None)
        if evidence_source is None:
            evidence_source = getattr(entity, "evidence", None)
        evidence_val = bool(evidence_source) if evidence_source is not None else False

        return {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity.entity_id,
            "entity_type": entity_type_value,
            "weight": weight,
            "prob_given_true": prob_true,
            "prob_given_false": prob_false,
            "last_updated": last_updated,
            "is_decaying": entity.decay.is_decaying,
            "decay_start": entity.decay.decay_start,
            "evidence": evidence_val,
        }

    def _attempt(session: Any) -> int:
        """Attempt to save entity data with batched merges for better performance."""
        merges_count = 0
        # Collect all entity objects first, then merge in batches
        # This allows SQLAlchemy to optimize the operations
        entity_objects = []

        for area_name, entity in _iter_area_entities():
            entity_data = _prepare_entity_payload(area_name, entity)
            if entity_data is None:
                continue

            entity_obj = db.Entities.from_dict(entity_data)
            entity_objects.append(entity_obj)
            merges_count += 1

        # Perform all merges (SQLite requires individual merges due to upsert limitations)
        # Collecting first allows SQLAlchemy to optimize the batch
        for entity_obj in entity_objects:
            session.merge(entity_obj)

        # Single commit for all merges
        session.commit()
        return merges_count

    try:
        backoffs = [0.1, 0.25, 0.5, 1.0]
        for attempt, delay in enumerate(backoffs, start=1):
            try:
                with db.get_locked_session() as session:
                    _attempt(session)
                # Update debounce timestamp after any successful attempt,
                # regardless of whether merges occurred, to avoid rapid retries
                db.last_entities_save_ts = time.monotonic()
                # Whether merges happened or not, no further retries are useful
                break
            except (sa.exc.OperationalError, sa.exc.TimeoutError) as err:
                _LOGGER.warning("save_entity_data attempt %d failed: %s", attempt, err)
                if attempt == len(backoffs):
                    # Ensure next call is not debounced due to a recent success
                    db.last_entities_save_ts = 0.0
                    raise
                time.sleep(delay)

        # Clean up any orphaned entities after saving current ones
        try:
            cleaned_count = cleanup_orphaned_entities(db)
            if cleaned_count > 0:
                _LOGGER.info(
                    "Cleaned up %d orphaned entities after saving", cleaned_count
                )
        except (
            sa.exc.SQLAlchemyError,
            HomeAssistantError,
            TimeoutError,
            OSError,
            RuntimeError,
        ) as cleanup_err:
            _LOGGER.error("Failed to cleanup orphaned entities: %s", cleanup_err)

    except (
        sa.exc.SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as err:
        _LOGGER.error("Failed to save entity data: %s", err)
        raise


def save_data(db: AreaOccupancyDB) -> None:
    """Save both area and entity data to the database."""
    save_area_data(db)
    save_entity_data(db)


def cleanup_orphaned_entities(db: AreaOccupancyDB) -> int:
    """Clean up entities from database that are no longer in the current configuration.

    This method removes entities and their associated intervals that exist in the database
    but are no longer present in the coordinator's current entity configuration.

    Returns:
        int: Number of entities that were cleaned up
    """
    total_cleaned = 0
    try:
        for area_name in db.coordinator.get_area_names():
            area_data = db.coordinator.get_area(area_name)

            def _cleanup_operation(area_name: str, area_data: Any) -> int:
                with db.get_session() as session:
                    # Get all entity IDs currently configured for this area
                    # EntityManager always has entity_ids property
                    current_entity_ids = set(area_data.entities.entity_ids)

                    # Query all entities for this area_name from database
                    db_entities = (
                        session.query(db.Entities).filter_by(area_name=area_name).all()
                    )

                    # Find entities that exist in database but not in current config
                    orphaned_entities = [
                        entity
                        for entity in db_entities
                        if entity.entity_id not in current_entity_ids
                    ]

                    if not orphaned_entities:
                        return 0

                    # Collect orphaned entity IDs for bulk operations
                    orphaned_entity_ids = [
                        entity.entity_id for entity in orphaned_entities
                    ]

                    # Log orphaned entities being removed
                    for entity_id in orphaned_entity_ids:
                        _LOGGER.info(
                            "Removing orphaned entity %s from database for area %s (no longer in config)",
                            entity_id,
                            area_name,
                        )

                    # Bulk delete all intervals for orphaned entities in a single query
                    # Filter by both area_name and entity_id to avoid deleting intervals
                    # for entities with the same ID in other areas
                    intervals_deleted = (
                        session.query(db.Intervals)
                        .filter(db.Intervals.area_name == area_name)
                        .filter(db.Intervals.entity_id.in_(orphaned_entity_ids))
                        .delete(synchronize_session=False)
                    )

                    # Bulk delete all orphaned entities in a single query
                    # Filter by both area_name and entity_id to avoid deleting entities
                    # with the same ID in other areas
                    entities_deleted = (
                        session.query(db.Entities)
                        .filter(db.Entities.area_name == area_name)
                        .filter(db.Entities.entity_id.in_(orphaned_entity_ids))
                        .delete(synchronize_session=False)
                    )

                    session.commit()
                    _LOGGER.info(
                        "Cleaned up %d orphaned entities for area %s (deleted %d intervals)",
                        entities_deleted,
                        area_name,
                        intervals_deleted,
                    )
                    return entities_deleted

            result = _cleanup_operation(area_name, area_data)
            total_cleaned += result

    except (
        sa.exc.SQLAlchemyError,
        HomeAssistantError,
        OSError,
        RuntimeError,
    ) as err:
        _LOGGER.error("Failed to cleanup orphaned entities: %s", err)
        return total_cleaned
    return total_cleaned


def delete_area_data(db: AreaOccupancyDB, area_name: str) -> int:
    """Delete all database data for a removed area.

    This includes:
    - All entities for the area
    - All intervals for those entities (filtered by area_name to avoid
      deleting intervals for entities with the same ID in other areas)
    - All priors for the area
    - All global priors for the area
    - All occupied intervals cache entries for the area
    - The area record itself

    Args:
        db: Database instance
        area_name: Name of the area to delete

    Returns:
        int: Number of entities deleted
    """
    deleted_count = 0
    try:
        with db.get_session() as session:
            # Get all entity IDs for this area first, then bulk delete intervals
            # SQLAlchemy doesn't allow delete() on queries with join()
            entity_ids = [
                entity_id
                for (entity_id,) in session.query(db.Entities.entity_id)
                .filter_by(area_name=area_name)
                .all()
            ]

            # Bulk delete all intervals for entities in this area
            # Filter by both area_name and entity_id to avoid deleting intervals
            # for entities with the same ID in other areas
            query = session.query(db.Intervals).filter(
                db.Intervals.area_name == area_name
            )
            if entity_ids:
                query = query.filter(db.Intervals.entity_id.in_(entity_ids))
            intervals_deleted = query.delete(synchronize_session=False)

            # Delete all entities for this area
            entities_deleted = (
                session.query(db.Entities).filter_by(area_name=area_name).delete()
            )
            deleted_count = entities_deleted

            # Delete priors for this area
            priors_deleted = (
                session.query(db.Priors).filter_by(area_name=area_name).delete()
            )

            # Delete global priors for this area
            global_priors_deleted = (
                session.query(db.GlobalPriors).filter_by(area_name=area_name).delete()
            )

            # Delete occupied intervals cache for this area
            cache_deleted = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .delete()
            )

            # Delete the area record itself
            area_deleted = (
                session.query(db.Areas).filter_by(area_name=area_name).delete()
            )

            session.commit()
            _LOGGER.info(
                "Deleted all data for removed area %s (%d entities, %d intervals, %d priors, %d global priors, %d cache entries, %d area records)",
                area_name,
                deleted_count,
                intervals_deleted,
                priors_deleted,
                global_priors_deleted,
                cache_deleted,
                area_deleted,
            )
    except (SQLAlchemyError, OSError) as err:
        _LOGGER.error("Failed to delete data for removed area %s: %s", area_name, err)
    return deleted_count


def _create_data_hash(
    area_name: str,
    period_start: datetime,
    period_end: datetime,
    total_occupied: float,
    interval_count: int,
) -> str:
    """Create a hash of underlying data for validation.

    Args:
        area_name: Area name
        period_start: Period start time
        period_end: Period end time
        total_occupied: Total occupied seconds
        interval_count: Number of intervals

    Returns:
        Hash string
    """
    data = {
        "area_name": area_name,
        "period_start": period_start.isoformat(),
        "period_end": period_end.isoformat(),
        "total_occupied": total_occupied,
        "interval_count": interval_count,
    }
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()


def _prune_old_global_priors(
    db: AreaOccupancyDB,
    session: Any,
    area_name: str,
) -> None:
    """Prune old global prior calculations, keeping only the most recent N.

    Args:
        db: Database instance
        session: Database session
        area_name: Area name
    """
    try:
        # Get all global priors for this area, ordered by calculation date
        # Note: GlobalPriors has unique constraint on area_name, so there's only one
        # But we keep this function for future use if we change to allow history
        priors = (
            session.query(db.GlobalPriors)
            .filter_by(area_name=area_name)
            .order_by(db.GlobalPriors.calculation_date.desc())
            .all()
        )

        # Keep only the most recent N calculations
        if len(priors) > GLOBAL_PRIOR_HISTORY_COUNT:
            to_delete = priors[GLOBAL_PRIOR_HISTORY_COUNT:]
            for prior in to_delete:
                session.delete(prior)
            session.commit()
            _LOGGER.debug(
                "Pruned %d old global prior calculations for %s",
                len(to_delete),
                area_name,
            )

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError) as e:
        _LOGGER.warning("Error pruning old global priors: %s", e)
        # Don't raise - this is cleanup, not critical


async def ensure_area_exists(db: AreaOccupancyDB) -> None:
    """Ensure that the area record exists in the database."""
    try:
        # Check if area exists
        existing_area = queries.get_area_data(db, db.coordinator.entry_id)
        if existing_area:
            _LOGGER.debug(
                "Area already exists for entry_id: %s", db.coordinator.entry_id
            )
            return

        # Area doesn't exist, force create it
        _LOGGER.info(
            "Area not found, forcing creation for entry_id: %s",
            db.coordinator.entry_id,
        )
        # Call save_area_data directly to avoid circular dependency
        save_area_data(db)

        # Verify it was created
        new_area = queries.get_area_data(db, db.coordinator.entry_id)
        if new_area:
            _LOGGER.info("Successfully created area: %s", new_area)
        else:
            _LOGGER.error(
                "Failed to create area for entry_id: %s", db.coordinator.entry_id
            )

    except (
        SQLAlchemyError,
        HomeAssistantError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
        TimeoutError,
    ) as e:
        _LOGGER.error("Error ensuring area exists: %s", e)


def prune_old_intervals(db: AreaOccupancyDB, force: bool = False) -> int:
    """Delete intervals older than RETENTION_DAYS (coordinated across instances).

    Args:
        db: Database instance
        force: If True, skip the recent-prune check

    Returns:
        Number of intervals deleted
    """
    cutoff_date = dt_util.utcnow() - timedelta(days=RETENTION_DAYS)
    _LOGGER.debug("Pruning intervals older than %s", cutoff_date)

    try:
        with db.get_locked_session() as session:
            # Re-check last_prune inside locked session to prevent concurrent bypass
            # This ensures the throttle cannot be bypassed by concurrent instances
            if not force:
                result = (
                    session.query(db.Metadata).filter_by(key="last_prune_time").first()
                )
                if result:
                    try:
                        last_prune = datetime.fromisoformat(result.value)
                        time_since_prune = (
                            dt_util.utcnow() - last_prune
                        ).total_seconds()
                        if time_since_prune < 3600:  # 1 hour
                            _LOGGER.debug(
                                "Skipping prune - last run was %d minutes ago",
                                int(time_since_prune / 60),
                            )
                            return 0
                    except (ValueError, AttributeError) as e:
                        _LOGGER.debug(
                            "Failed to parse last prune time, proceeding: %s", e
                        )

            # Count intervals to be deleted for logging
            count_query = session.query(func.count(db.Intervals.id)).filter(
                db.Intervals.start_time < cutoff_date
            )
            intervals_to_delete = count_query.scalar() or 0

            if intervals_to_delete == 0:
                _LOGGER.debug("No old intervals to prune")
                # Still record the prune attempt to prevent other instances from trying
                maintenance.set_last_prune_time(db, dt_util.utcnow(), session)
                return 0

            # Delete old intervals
            delete_query = session.query(db.Intervals).filter(
                db.Intervals.start_time < cutoff_date
            )
            deleted_count = delete_query.delete(synchronize_session=False)

            session.commit()

            _LOGGER.info(
                "Pruned %d intervals older than %d days (cutoff: %s)",
                deleted_count,
                RETENTION_DAYS,
                cutoff_date,
            )

            # Record successful prune
            maintenance.set_last_prune_time(db, dt_util.utcnow(), session)

            return deleted_count

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error during interval pruning: %s", e)
        return 0


def save_global_prior(
    db: AreaOccupancyDB,
    area_name: str,
    prior_value: float,
    data_period_start: datetime,
    data_period_end: datetime,
    total_occupied_seconds: float,
    total_period_seconds: float,
    interval_count: int,
    calculation_method: str = "interval_analysis",
    confidence: float | None = None,
) -> bool:
    """Save global prior calculation to GlobalPriors table.

    Args:
        db: Database instance
        area_name: Area name
        prior_value: Calculated prior probability
        data_period_start: Start of data period used
        data_period_end: End of data period used
        total_occupied_seconds: Total occupied time in period
        total_period_seconds: Total period duration
        interval_count: Number of intervals used
        calculation_method: Method used for calculation
        confidence: Confidence in calculation (0.0-1.0)

    Returns:
        True if saved successfully, False otherwise
    """
    _LOGGER.debug(
        "Saving global prior for area: %s, value: %.4f", area_name, prior_value
    )

    try:
        with db.get_locked_session() as session:
            # Create hash of underlying data for validation
            data_hash = _create_data_hash(
                area_name,
                data_period_start,
                data_period_end,
                total_occupied_seconds,
                interval_count,
            )

            # Check if global prior already exists for this area
            existing = (
                session.query(db.GlobalPriors).filter_by(area_name=area_name).first()
            )

            if existing:
                # Update existing record
                existing.prior_value = prior_value
                existing.calculation_date = dt_util.utcnow()
                existing.data_period_start = data_period_start
                existing.data_period_end = data_period_end
                existing.total_occupied_seconds = total_occupied_seconds
                existing.total_period_seconds = total_period_seconds
                existing.interval_count = interval_count
                existing.confidence = confidence
                existing.calculation_method = calculation_method
                existing.underlying_data_hash = data_hash
                existing.updated_at = dt_util.utcnow()
            else:
                # Create new record
                global_prior = db.GlobalPriors(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    prior_value=prior_value,
                    calculation_date=dt_util.utcnow(),
                    data_period_start=data_period_start,
                    data_period_end=data_period_end,
                    total_occupied_seconds=total_occupied_seconds,
                    total_period_seconds=total_period_seconds,
                    interval_count=interval_count,
                    confidence=confidence,
                    calculation_method=calculation_method,
                    underlying_data_hash=data_hash,
                )
                session.add(global_prior)

            session.commit()

            # Prune old global prior history (keep only last N)
            _prune_old_global_priors(db, session, area_name)

            _LOGGER.debug("Global prior saved successfully")
            return True

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error saving global prior: %s", e)
        session.rollback()
        return False


def save_occupied_intervals_cache(
    db: AreaOccupancyDB,
    area_name: str,
    intervals: list[tuple[datetime, datetime]],
    data_source: str = "merged",
) -> bool:
    """Save occupied intervals to OccupiedIntervalsCache table.

    Args:
        db: Database instance
        area_name: Area name
        intervals: List of (start_time, end_time) tuples
        data_source: Source of intervals ('primary_sensor', 'motion_sensors', 'merged')

    Returns:
        True if saved successfully, False otherwise
    """
    _LOGGER.debug(
        "Saving %d occupied intervals to cache for area: %s",
        len(intervals),
        area_name,
    )

    try:
        with db.get_locked_session() as session:
            calculation_date = dt_util.utcnow()

            # Delete existing cached intervals for this area
            session.query(db.OccupiedIntervalsCache).filter_by(
                area_name=area_name
            ).delete(synchronize_session=False)

            # Insert new intervals
            for start_time, end_time in intervals:
                duration_seconds = (end_time - start_time).total_seconds()

                cached_interval = db.OccupiedIntervalsCache(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration_seconds,
                    calculation_date=calculation_date,
                    data_source=data_source,
                )
                session.add(cached_interval)

            session.commit()
            _LOGGER.debug("Occupied intervals cache saved successfully")
            return True

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error saving occupied intervals cache: %s", e)
        session.rollback()
        return False
