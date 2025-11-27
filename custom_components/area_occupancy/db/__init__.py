"""Database package for area occupancy detection."""

from __future__ import annotations

from ..const import (
    DB_NAME,
    DB_VERSION,
    DEFAULT_AREA_PRIOR,
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
    Correlations,
    CrossAreaStats,
    Entities,
    EntityStatistics,
    GlobalPriors,
    IntervalAggregates,
    Intervals,
    Metadata,
    NumericAggregates,
    NumericSamples,
    OccupiedIntervalsCache,
    Priors,
)

__all__ = [
    "DB_NAME",
    "DB_VERSION",
    "DEFAULT_AREA_PRIOR",
    "DEFAULT_ENTITY_PROB_GIVEN_FALSE",
    "DEFAULT_ENTITY_PROB_GIVEN_TRUE",
    "DEFAULT_ENTITY_WEIGHT",
    "INVALID_STATES",
    "AreaOccupancyDB",
    "AreaRelationships",
    "Areas",
    "Base",
    "Correlations",
    "CrossAreaStats",
    "Entities",
    "EntityStatistics",
    "GlobalPriors",
    "IntervalAggregates",
    "Intervals",
    "Metadata",
    "NumericAggregates",
    "NumericSamples",
    "OccupiedIntervalsCache",
    "Priors",
]
