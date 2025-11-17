"""Database utility functions."""

from __future__ import annotations

import logging
from typing import Any

import sqlalchemy as sa

from homeassistant.exceptions import HomeAssistantError

from . import maintenance
from .constants import INVALID_STATES

_LOGGER = logging.getLogger(__name__)


def is_valid_state(state: Any) -> bool:
    """Check if a state is valid."""
    return state not in INVALID_STATES


def is_intervals_empty(db: Any) -> bool:
    """Check if the intervals table is empty using ORM (read-only, no lock)."""
    try:
        with db.get_session() as session:
            count = session.query(db.Intervals).count()
            return bool(count == 0)
    except (
        sa.exc.SQLAlchemyError,
        HomeAssistantError,
        TimeoutError,
        OSError,
        RuntimeError,
    ) as e:
        # If table doesn't exist, it's considered empty
        if "no such table" in str(e).lower():
            _LOGGER.debug("Intervals table doesn't exist yet, considering empty")
            return True
        _LOGGER.error("Failed to check if intervals empty: %s", e)
        # Return True as fallback to trigger data population
        return True


def safe_is_intervals_empty(db: Any) -> bool:
    """Safely check if intervals table is empty (fast, no integrity checks).

    Note: Database integrity checks are deferred to background health check
    task that runs 60 seconds after startup to avoid blocking integration loading.

    Returns:
        bool: True if intervals are empty, False if intervals exist
    """
    try:
        # Quick check - assume database is healthy during startup
        # Integrity checks will be performed by background health check task
        return is_intervals_empty(db)
    except (
        sa.exc.SQLAlchemyError,
        HomeAssistantError,
        TimeoutError,
        OSError,
        RuntimeError,
    ) as e:
        # If we hit a corruption error, log it but don't block startup
        if maintenance.is_database_corrupted(db, e):
            _LOGGER.warning(
                "Database may be corrupted (error: %s). "
                "Background health check will attempt recovery in 60 seconds.",
                e,
            )
        else:
            _LOGGER.error("Error checking intervals: %s", e)

        # Assume empty to trigger data population, but don't block startup
        return True
