"""Database CRUD operations."""

from __future__ import annotations

from collections.abc import Iterable
import logging
import time
from typing import TYPE_CHECKING, Any

import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError

from homeassistant import helpers
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import MAX_PROBABILITY, MAX_WEIGHT, MIN_PROBABILITY, MIN_WEIGHT
from . import maintenance
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
                area_data = db.coordinator.get_area_or_default(area_name)
                if area_data:
                    for entity_obj in entities:
                        # Check if entity exists in current coordinator config
                        try:
                            area_data.entities.get_entity(entity_obj.entity_id)
                        except ValueError:
                            # Entity not found in coordinator - identify if stale
                            should_delete = False
                            if hasattr(area_data.entities, "entity_ids"):
                                current_entity_ids = set(area_data.entities.entity_ids)
                                if entity_obj.entity_id not in current_entity_ids:
                                    should_delete = True
                            elif hasattr(area_data.entities, "entities"):
                                # Fallback for mock objects that have entities dict
                                current_entity_ids = set(
                                    area_data.entities.entities.keys()
                                )
                                if entity_obj.entity_id not in current_entity_ids:
                                    should_delete = True
                            else:
                                # Can't determine current config - assume entity is stale
                                should_delete = True

                            if should_delete:
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
            area_data = db.coordinator.get_area_or_default(area_name)

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
            area_data_obj = db.coordinator.get_area_or_default(area_name_item)
            if area_data_obj is None:
                error_msg = f"Area '{area_name_item}' not found"
                _LOGGER.error("%s, cannot insert area", error_msg)
                failures.append((area_name_item, error_msg))
                has_failures = True
                continue

            cfg = area_data_obj.config

            # Get area_id from config
            area_id = cfg.area_id
            if not area_id:
                error_msg = "area_id is missing"
                _LOGGER.warning("Skipping area '%s': %s", area_name_item, error_msg)
                failures.append((area_name_item, error_msg))
                has_failures = True
                continue

            area_data = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name_item,
                "area_id": area_id,
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
    except Exception as err:
        _LOGGER.error("Failed to save area data: %s", err)
        raise


def save_entity_data(db: AreaOccupancyDB) -> None:
    """Save the entity data to the database for all areas.

    With single-instance architecture, no file lock is required.
    """

    def _iter_area_entities() -> Iterable[tuple[str, Any]]:
        """Yield (area_name, entity) tuples for all configured areas."""
        for area_name in db.coordinator.get_area_names():
            area_data = db.coordinator.get_area_or_default(area_name)
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
        ) as cleanup_err:
            _LOGGER.error("Failed to cleanup orphaned entities: %s", cleanup_err)

    except Exception as err:
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
            area_data = db.coordinator.get_area_or_default(area_name)

            def _cleanup_operation(area_name: str, area_data: Any) -> int:
                with db.get_session() as session:
                    # Get all entity IDs currently configured for this area
                    # Handle cases where entities might be a SimpleNamespace or mock object
                    if hasattr(area_data.entities, "entity_ids"):
                        current_entity_ids = set(area_data.entities.entity_ids)
                    elif hasattr(area_data.entities, "entities"):
                        # Fallback for mock objects that have entities dict
                        current_entity_ids = set(area_data.entities.entities.keys())
                    else:
                        # If we can't determine current entities, skip cleanup
                        return 0

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
