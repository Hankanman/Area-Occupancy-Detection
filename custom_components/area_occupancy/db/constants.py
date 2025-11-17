"""Database-specific constants."""

from __future__ import annotations

# Database filename
DB_NAME = "area_occupancy.db"

# Database schema version for migrations
DB_VERSION = 5  # New schema: redesigned for single integration with multiple areas

# States to exclude from intervals
INVALID_STATES = {"unknown", "unavailable", None, "", "NaN"}

# Default values
DEFAULT_AREA_PRIOR = 0.15
DEFAULT_ENTITY_WEIGHT = 0.85
DEFAULT_ENTITY_PROB_GIVEN_TRUE = 0.8
DEFAULT_ENTITY_PROB_GIVEN_FALSE = 0.05
