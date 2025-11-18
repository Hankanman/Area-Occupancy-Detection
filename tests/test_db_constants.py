"""Tests for database constants."""

from custom_components.area_occupancy.db import (
    DB_NAME as EXPORTED_DB_NAME,
    DB_VERSION as EXPORTED_DB_VERSION,
    DEFAULT_AREA_PRIOR as EXPORTED_DEFAULT_AREA_PRIOR,
    DEFAULT_ENTITY_PROB_GIVEN_FALSE as EXPORTED_DEFAULT_ENTITY_PROB_GIVEN_FALSE,
    DEFAULT_ENTITY_PROB_GIVEN_TRUE as EXPORTED_DEFAULT_ENTITY_PROB_GIVEN_TRUE,
    DEFAULT_ENTITY_WEIGHT as EXPORTED_DEFAULT_ENTITY_WEIGHT,
    INVALID_STATES as EXPORTED_INVALID_STATES,
)
from custom_components.area_occupancy.db.constants import (
    DB_NAME,
    DB_VERSION,
    DEFAULT_AREA_PRIOR,
    DEFAULT_ENTITY_PROB_GIVEN_FALSE,
    DEFAULT_ENTITY_PROB_GIVEN_TRUE,
    DEFAULT_ENTITY_WEIGHT,
    INVALID_STATES,
)


class TestDatabaseConstants:
    """Test database constant values."""

    def test_db_name(self):
        """Test DB_NAME constant."""
        assert DB_NAME == "area_occupancy.db"
        assert isinstance(DB_NAME, str)

    def test_db_version(self):
        """Test DB_VERSION constant."""
        assert DB_VERSION == 5
        assert isinstance(DB_VERSION, int)
        assert DB_VERSION > 0

    def test_default_area_prior(self):
        """Test DEFAULT_AREA_PRIOR constant."""
        assert DEFAULT_AREA_PRIOR == 0.15
        assert isinstance(DEFAULT_AREA_PRIOR, float)
        assert 0.0 <= DEFAULT_AREA_PRIOR <= 1.0

    def test_default_entity_weight(self):
        """Test DEFAULT_ENTITY_WEIGHT constant."""
        assert DEFAULT_ENTITY_WEIGHT == 0.85
        assert isinstance(DEFAULT_ENTITY_WEIGHT, float)
        assert 0.0 <= DEFAULT_ENTITY_WEIGHT <= 1.0

    def test_default_entity_prob_given_true(self):
        """Test DEFAULT_ENTITY_PROB_GIVEN_TRUE constant."""
        assert DEFAULT_ENTITY_PROB_GIVEN_TRUE == 0.8
        assert isinstance(DEFAULT_ENTITY_PROB_GIVEN_TRUE, float)
        assert 0.0 <= DEFAULT_ENTITY_PROB_GIVEN_TRUE <= 1.0

    def test_default_entity_prob_given_false(self):
        """Test DEFAULT_ENTITY_PROB_GIVEN_FALSE constant."""
        assert DEFAULT_ENTITY_PROB_GIVEN_FALSE == 0.05
        assert isinstance(DEFAULT_ENTITY_PROB_GIVEN_FALSE, float)
        assert 0.0 <= DEFAULT_ENTITY_PROB_GIVEN_FALSE <= 1.0

    def test_invalid_states(self):
        """Test INVALID_STATES constant."""
        assert isinstance(INVALID_STATES, set)
        assert "unknown" in INVALID_STATES
        assert "unavailable" in INVALID_STATES
        assert None in INVALID_STATES
        assert "" in INVALID_STATES
        assert "NaN" in INVALID_STATES

    def test_constants_exported(self):
        """Test that constants are properly exported from db package."""
        assert EXPORTED_DB_NAME == DB_NAME
        assert EXPORTED_DB_VERSION == DB_VERSION
        assert EXPORTED_DEFAULT_AREA_PRIOR == DEFAULT_AREA_PRIOR
        assert (
            EXPORTED_DEFAULT_ENTITY_PROB_GIVEN_FALSE == DEFAULT_ENTITY_PROB_GIVEN_FALSE
        )
        assert EXPORTED_DEFAULT_ENTITY_PROB_GIVEN_TRUE == DEFAULT_ENTITY_PROB_GIVEN_TRUE
        assert EXPORTED_DEFAULT_ENTITY_WEIGHT == DEFAULT_ENTITY_WEIGHT
        assert EXPORTED_INVALID_STATES == INVALID_STATES
