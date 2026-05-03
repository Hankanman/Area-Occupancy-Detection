"""Area relationship management and influence weight calculations.

This module handles storage and retrieval of area relationships (adjacent areas,
shared walls, etc.) and calculates influence weights for cross-area probability
adjustments.

Status (feat/adjacent-areas, discussion #431): the config flow writes
adjacent area_ids into ``Areas.adjacent_areas`` and the area-save path now
calls ``sync_adjacent_areas_from_config`` to reconcile the
``AreaRelationships`` table. The Bayesian / decay consumers
(``calculate_adjacent_influence`` and friends) are still uncalled — they're
the Phase 4 work, gated on transition learning landing in
``db.transitions`` (Phase 3 in progress).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..time_utils import to_db_utc

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)

# Default influence weights by relationship type
DEFAULT_INFLUENCE_WEIGHTS = {
    "adjacent": 0.3,  # Adjacent areas have moderate influence
    "shared_wall": 0.4,  # Shared walls have higher influence
    "shared_entrance": 0.5,  # Shared entrances have strong influence
    "open_connection": 0.6,  # Open connections (no door) have very strong influence
}


def save_area_relationship(
    db: AreaOccupancyDB,
    area_name: str,
    related_area_name: str,
    relationship_type: str = "adjacent",
    influence_weight: float | None = None,
    distance: float | None = None,
) -> bool:
    """Save or update an area relationship.

    Args:
        db: Database instance
        area_name: Source area name
        related_area_name: Related/adjacent area name
        relationship_type: Type of relationship (adjacent, shared_wall, etc.)
        influence_weight: Influence weight (0.0-1.0). If None, uses default for type.
        distance: Physical distance if applicable

    Returns:
        True if saved successfully, False otherwise
    """
    _LOGGER.debug(
        "Saving relationship: %s -> %s (type: %s)",
        area_name,
        related_area_name,
        relationship_type,
    )

    try:
        with db.get_session() as session:
            # Use default weight if not provided
            if influence_weight is None:
                influence_weight = DEFAULT_INFLUENCE_WEIGHTS.get(
                    relationship_type, DEFAULT_INFLUENCE_WEIGHTS["adjacent"]
                )

            # Clamp influence weight to valid range
            influence_weight = max(0.0, min(1.0, influence_weight))

            # Check if relationship already exists
            existing = (
                session.query(db.AreaRelationships)
                .filter_by(area_name=area_name, related_area_name=related_area_name)
                .first()
            )

            if existing:
                # Update existing relationship
                existing.relationship_type = relationship_type
                existing.influence_weight = influence_weight
                existing.distance = distance
                existing.updated_at = to_db_utc(dt_util.utcnow())
            else:
                # Create new relationship
                relationship = db.AreaRelationships(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    related_area_name=related_area_name,
                    relationship_type=relationship_type,
                    influence_weight=influence_weight,
                    distance=distance,
                )
                session.add(relationship)

            session.commit()
            _LOGGER.debug("Relationship saved successfully")
            return True

    except (
        HomeAssistantError,
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as e:
        _LOGGER.error("Error saving relationship: %s", e)
        return False


def get_adjacent_areas(db: AreaOccupancyDB, area_name: str) -> list[dict[str, Any]]:
    """Get all adjacent/related areas for an area.

    Args:
        db: Database instance
        area_name: Area name

    Returns:
        List of relationship dictionaries
    """
    try:
        with db.get_session() as session:
            relationships = (
                session.query(db.AreaRelationships).filter_by(area_name=area_name).all()
            )

            return [
                {
                    "related_area_name": rel.related_area_name,
                    "relationship_type": rel.relationship_type,
                    "influence_weight": rel.influence_weight,
                    "distance": rel.distance,
                }
                for rel in relationships
            ]

    except SQLAlchemyError as e:
        _LOGGER.error("Database error getting adjacent areas: %s", e)
        return []


def get_influence_weight(
    db: AreaOccupancyDB, area_name: str, related_area_name: str
) -> float:
    """Get the influence weight between two areas.

    Args:
        db: Database instance
        area_name: Source area name
        related_area_name: Related area name

    Returns:
        Influence weight (0.0-1.0), or 0.0 if no relationship exists
    """
    try:
        with db.get_session() as session:
            relationship = (
                session.query(db.AreaRelationships)
                .filter_by(area_name=area_name, related_area_name=related_area_name)
                .first()
            )

            if relationship:
                return float(relationship.influence_weight)

            return 0.0

    except SQLAlchemyError as e:
        _LOGGER.error("Database error getting influence weight: %s", e)
        return 0.0


def calculate_adjacent_influence(
    db: AreaOccupancyDB, area_name: str, base_probability: float
) -> float:
    """Calculate probability adjustment based on adjacent area occupancy.

    Args:
        db: Database instance
        area_name: Area name
        base_probability: Base probability before adjustment

    Returns:
        Adjusted probability (0.0-1.0)
    """
    try:
        adjacent_areas = get_adjacent_areas(db, area_name)

        if not adjacent_areas:
            return base_probability

        # Get current occupancy probabilities for adjacent areas
        adjustment = 0.0
        total_weight = 0.0

        for adj in adjacent_areas:
            related_area = db.coordinator.get_area(adj["related_area_name"])
            if related_area:
                # Get current probability of adjacent area
                adj_prob = related_area.probability()
                influence_weight = adj["influence_weight"]

                # Calculate adjustment: if adjacent area is occupied, increase probability
                # Adjustment is proportional to adjacent area's probability and influence weight
                adjustment += adj_prob * influence_weight
                total_weight += influence_weight

        # Normalize adjustment
        if total_weight > 0:
            avg_adjustment = adjustment / total_weight
            # Apply adjustment: increase base probability by weighted average
            # Clamp to valid range
            adjusted_prob = min(1.0, base_probability + (avg_adjustment * 0.2))

            _LOGGER.debug(
                "Adjacent area influence: base=%.3f, adjusted=%.3f (from %d areas)",
                base_probability,
                adjusted_prob,
                len(adjacent_areas),
            )

            return adjusted_prob

    except (ValueError, TypeError, RuntimeError, AttributeError) as e:
        _LOGGER.warning("Error calculating adjacent influence: %s", e)
        return base_probability
    else:
        return base_probability


def sync_adjacent_areas_from_config(db: AreaOccupancyDB, area_name: str) -> bool:
    """Sync adjacent areas from area configuration to AreaRelationships table.

    Reads the canonical ``Areas.adjacent_areas`` JSON column for the given
    area and reconciles the ``AreaRelationships`` rows of type ``adjacent``
    so they exactly match: rows for neighbours that have been removed are
    deleted, missing rows are inserted, weights/types are preserved on
    existing rows. Performs the read, delete, and merge in a single
    session so partial failures don't leave the table in an inconsistent
    state.

    Args:
        db: Database instance
        area_name: Area name

    Returns:
        True if synced successfully, False otherwise
    """
    _LOGGER.debug("Syncing adjacent areas from config for area: %s", area_name)

    try:
        with db.get_session() as session:
            area_record = session.query(db.Areas).filter_by(area_name=area_name).first()
            if not area_record:
                _LOGGER.warning("Area record not found in database: %s", area_name)
                return False

            target_neighbours: set[str] = {
                str(a) for a in (area_record.adjacent_areas or []) if a
            }

            existing_rows = (
                session.query(db.AreaRelationships)
                .filter(
                    db.AreaRelationships.area_name == area_name,
                    db.AreaRelationships.relationship_type == "adjacent",
                )
                .all()
            )
            existing_by_neighbour = {
                row.related_area_name: row for row in existing_rows
            }

            # Delete rows for neighbours no longer in the config.
            removed = 0
            for neighbour, row in existing_by_neighbour.items():
                if neighbour not in target_neighbours:
                    session.delete(row)
                    removed += 1

            # Add rows for new neighbours.
            default_weight = DEFAULT_INFLUENCE_WEIGHTS["adjacent"]
            added = 0
            for neighbour in target_neighbours:
                if neighbour in existing_by_neighbour:
                    continue
                session.add(
                    db.AreaRelationships(
                        entry_id=db.coordinator.entry_id,
                        area_name=area_name,
                        related_area_name=neighbour,
                        relationship_type="adjacent",
                        influence_weight=default_weight,
                    )
                )
                added += 1

            session.commit()

            if added or removed:
                _LOGGER.info(
                    "Synced adjacency for %s: %d added, %d removed (%d total now)",
                    area_name,
                    added,
                    removed,
                    len(target_neighbours),
                )
            return True

    except (ValueError, TypeError, RuntimeError, AttributeError, SQLAlchemyError) as e:
        _LOGGER.error("Error syncing adjacent areas: %s", e)
        return False
