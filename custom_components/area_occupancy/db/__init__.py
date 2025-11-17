"""Database package for area occupancy detection."""

from __future__ import annotations

from .constants import (
    DB_NAME,
    DB_VERSION,
    DEFAULT_ENTITY_PROB_GIVEN_FALSE,
    DEFAULT_ENTITY_PROB_GIVEN_TRUE,
    DEFAULT_ENTITY_WEIGHT,
    INVALID_STATES,
)
from .core import AreaOccupancyDB
from .schema import (
    AreaRelationships,
    Areas,
    Base,
    CrossAreaStats,
    Entities,
    EntityStatistics,
    GlobalPriors,
    IntervalAggregates,
    Intervals,
    Metadata,
    NumericAggregates,
    NumericCorrelations,
    NumericSamples,
    OccupiedIntervalsCache,
    Priors,
)

__all__ = [
    "DB_NAME",
    "DB_VERSION",
    "DEFAULT_ENTITY_PROB_GIVEN_FALSE",
    "DEFAULT_ENTITY_PROB_GIVEN_TRUE",
    "DEFAULT_ENTITY_WEIGHT",
    "INVALID_STATES",
    "AreaOccupancyDB",
    "AreaRelationships",
    "Areas",
    "Base",
    "CrossAreaStats",
    "Entities",
    "EntityStatistics",
    "GlobalPriors",
    "IntervalAggregates",
    "Intervals",
    "Metadata",
    "NumericAggregates",
    "NumericCorrelations",
    "NumericSamples",
    "OccupiedIntervalsCache",
    "Priors",
]
