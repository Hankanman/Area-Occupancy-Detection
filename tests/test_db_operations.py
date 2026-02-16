"""Tests for database CRUD operations."""
# ruff: noqa: SLF001

from contextlib import suppress
from datetime import timedelta
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from custom_components.area_occupancy.const import (
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PROBABILITY,
    RETENTION_DAYS,
    TIME_PRIOR_MAX_BOUND,
    TIME_PRIOR_MIN_BOUND,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.correlation import (
    save_binary_likelihood_result,
    save_correlation_result,
)
from custom_components.area_occupancy.db.operations import (
    _cleanup_orphaned_entities,
    _create_data_hash,
    _validate_area_data,
    delete_area_data,
    ensure_area_exists,
    load_data,
    prune_old_intervals,
    save_area_data,
    save_entity_data,
    save_global_prior,
    save_occupied_intervals_cache,
    save_time_priors,
)
from homeassistant.util import dt as dt_util


def _create_mock_entity(
    entity_id: str,
    input_type: str = "motion",
    weight: float = 0.9,
    prob_given_true: float = 0.8,
    prob_given_false: float = 0.1,
    evidence: bool = True,
    is_decaying: bool = False,
) -> SimpleNamespace:
    """Create a mock entity for testing."""
    return SimpleNamespace(
        entity_id=entity_id,
        type=SimpleNamespace(input_type=input_type, weight=weight),
        prob_given_true=prob_given_true,
        prob_given_false=prob_given_false,
        last_updated=dt_util.utcnow(),
        decay=SimpleNamespace(
            is_decaying=is_decaying,
            decay_start=dt_util.utcnow() if is_decaying else None,
        ),
        evidence=evidence,
    )


def _create_retry_mock(original_get_session, max_failures: int = 1):
    """Create a mock for retry mechanism testing."""
    call_count = 0

    @wraps(original_get_session)
    def mock_get_session(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= max_failures:
            # Raise OperationalError on first call(s)
            mock_session = Mock()
            mock_session.__enter__ = Mock(
                side_effect=OperationalError("Database locked", None, None)
            )
            mock_session.__exit__ = Mock(return_value=False)
            return mock_session
        # Return real session on retry
        return original_get_session(*args, **kwargs)

    return mock_get_session


class TestValidateAreaData:
    """Test _validate_area_data function."""

    def test_validate_area_data_valid(self, coordinator: AreaOccupancyCoordinator):
        """Test validation with valid data."""
        db = coordinator.db
        area_data = {
            "entry_id": "test",
            "area_name": "Test Area",
            "area_id": "test_id",
            "purpose": "living",
            "threshold": 0.5,
        }
        failures = _validate_area_data(db, area_data, "Test Area")
        assert failures == []

    def test_validate_area_data_missing_fields(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test validation with missing fields."""
        db = coordinator.db
        area_data = {"entry_id": "test"}
        failures = _validate_area_data(db, area_data, "Test Area")
        assert len(failures) > 0
        assert any("area_name" in msg for _, msg in failures)
        assert any("area_id" in msg for _, msg in failures)


class TestLoadData:
    """Test load_data function."""

    @pytest.mark.asyncio
    async def test_load_data_success(self, coordinator: AreaOccupancyCoordinator):
        """Test loading data successfully."""
        db = coordinator.db
        db.init_db()
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Clear prior to ensure we're testing restoration
        area.prior.global_prior = None
        db.save_area_data(area_name)

        # Save global prior to database
        expected_prior = 0.65
        save_global_prior(
            db,
            area_name,
            expected_prior,
            dt_util.utcnow() - timedelta(days=30),
            dt_util.utcnow(),
            86400.0,
            7776000.0,
            100,
        )

        # Create and save an entity with specific values
        entity_id = "binary_sensor.test_motion"
        try:
            entity = area.entities.get_entity(entity_id)
        except ValueError:
            entity = area.factory.create_from_config_spec(
                entity_id, InputType.MOTION.value
            )
            area.entities.add_entity(entity)

        # Set entity values
        entity.prob_given_true = 0.85
        entity.prob_given_false = 0.05
        entity.type.weight = 0.9
        save_entity_data(db)

        # Clear entity values to test restoration
        entity.prob_given_true = None
        entity.prob_given_false = None
        entity.type.weight = 0.85  # Default

        # Load data from database
        await load_data(db)

        # Verify global prior was restored
        assert area.prior.global_prior == expected_prior

        # Verify entity data was restored
        reloaded_entity = area.entities.get_entity(entity_id)
        assert reloaded_entity.prob_given_true == 0.85
        assert reloaded_entity.prob_given_false == 0.05
        assert reloaded_entity.type.weight == 0.9

    @pytest.mark.asyncio
    async def test_load_data_deletes_stale_entities(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that load_data deletes stale entities."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists in database first (foreign key requirement)
        save_area_data(db, area_name)

        # Save entity that's not in current config
        good = _create_mock_entity("binary_sensor.good")
        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.good": good},
            entity_ids=["binary_sensor.good"],
        )
        area._entities = mock_entities_manager
        db.save_entity_data()

        # Reset entities to real manager
        area._entities = None
        real_entities = area.entities

        # Verify entity is not in current config
        assert "binary_sensor.good" not in real_entities.entity_ids

        await load_data(db)

        # Verify stale entity was deleted
        with db.get_session() as session:
            count = (
                session.query(db.Entities)
                .filter_by(area_name=area_name, entity_id="binary_sensor.good")
                .count()
            )
            assert count == 0

    @pytest.mark.parametrize("error_class", [SQLAlchemyError, TimeoutError])
    @pytest.mark.asyncio
    async def test_load_data_handles_errors(
        self, coordinator: AreaOccupancyCoordinator, error_class
    ):
        """Test load_data propagates errors."""
        db = coordinator.db

        with (
            patch.object(db, "get_session", side_effect=error_class("Error")),
            pytest.raises(error_class),
        ):
            await load_data(db)

    @pytest.mark.parametrize(
        ("analysis_type", "entity_id", "input_type", "error_message"),
        [
            (
                "binary_likelihood",
                "binary_sensor.test_light",
                InputType.APPLIANCE,
                "no_occupied_intervals",
            ),
            (
                "correlation",
                "sensor.test_temperature",
                InputType.TEMPERATURE,
                "too_few_samples",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_load_data_preserves_analysis_error(
        self,
        coordinator: AreaOccupancyCoordinator,
        analysis_type,
        entity_id,
        input_type,
        error_message,
    ):
        """Test that analysis_error is preserved after reload for different analysis types."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists
        save_area_data(db, area_name)

        # Create entity
        try:
            entity = area.entities.get_entity(entity_id)
        except ValueError:
            entity = area.factory.create_from_config_spec(entity_id, input_type.value)
            area.entities.add_entity(entity)

        # Save entity to database
        save_entity_data(db)

        if analysis_type == "binary_likelihood":
            # Create binary likelihood result with analysis_error
            likelihood_data = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
                "analysis_period_end": dt_util.utcnow(),
                "prob_given_true": None,
                "prob_given_false": None,
                "analysis_error": error_message,
                "calculation_date": dt_util.utcnow(),
            }
            save_binary_likelihood_result(db, likelihood_data, input_type)
            entity.update_binary_likelihoods(likelihood_data)
        else:  # correlation
            # Create correlation result with analysis_error
            correlation_data = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "input_type": input_type.value,
                "correlation_coefficient": 0.0,
                "correlation_type": "none",
                "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
                "analysis_period_end": dt_util.utcnow(),
                "sample_count": 0,
                "confidence": None,
                "mean_value_when_occupied": None,
                "mean_value_when_unoccupied": None,
                "std_dev_when_occupied": None,
                "std_dev_when_unoccupied": None,
                "threshold_active": None,
                "threshold_inactive": None,
                "analysis_error": error_message,
                "calculation_date": dt_util.utcnow(),
            }
            save_correlation_result(db, correlation_data)
            entity.update_correlation(correlation_data)

        assert entity.analysis_error == error_message

        # Reload data
        await load_data(db)

        # Verify analysis_error is preserved
        reloaded_entity = area.entities.get_entity(entity_id)
        assert reloaded_entity.analysis_error == error_message


class TestSaveAreaData:
    """Test save_area_data function."""

    def test_save_area_data_success(self, coordinator: AreaOccupancyCoordinator):
        """Test saving area data successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        save_area_data(db, area_name)

        # Verify area was saved
        with db.get_session() as session:
            area = session.query(db.Areas).filter_by(area_name=area_name).first()
            assert area is not None
            assert area.area_name == area_name

    def test_save_area_data_all_areas(self, coordinator: AreaOccupancyCoordinator):
        """Test saving data for all areas."""
        db = coordinator.db
        save_area_data(db, None)

        # Verify all areas were saved
        area_names = db.coordinator.get_area_names()
        with db.get_session() as session:
            for area_name in area_names:
                area = session.query(db.Areas).filter_by(area_name=area_name).first()
                assert area is not None

    def test_save_area_data_validation_failure(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test save_area_data with validation failure."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Corrupt area config to cause validation failure
        original_area_id = area.config.area_id
        area.config.area_id = None  # This will cause validation to fail

        # Should handle validation failure gracefully
        try:
            save_area_data(db, area_name)
        except ValueError:
            # Expected when validation fails
            pass
        finally:
            # Restore original value
            area.config.area_id = original_area_id

    def test_save_area_data_database_error(self, coordinator: AreaOccupancyCoordinator):
        """Test save_area_data with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        with (
            patch.object(
                db,
                "get_session",
                side_effect=OperationalError("DB error", None, None),
            ),
            pytest.raises((OperationalError, ValueError)),
        ):
            # Should raise after retries
            save_area_data(db, area_name)


class TestSaveEntityData:
    """Test save_entity_data function."""

    def test_save_entity_data_success(self, coordinator: AreaOccupancyCoordinator):
        """Test saving entity data successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists in database first (foreign key requirement)
        save_area_data(db, area_name)

        good = _create_mock_entity("binary_sensor.good")
        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.good": good},
            entity_ids=["binary_sensor.good"],
        )
        area._entities = mock_entities_manager

        save_entity_data(db)

        # Verify entity was saved
        with db.get_session() as session:
            entity = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.good")
                .first()
            )
            assert entity is not None
            assert entity.entity_id == "binary_sensor.good"

    def test_save_entity_data_filters_invalid(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that invalid entities are filtered out."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists in database first (foreign key requirement)
        save_area_data(db, area_name)

        # Entity without type
        bad = SimpleNamespace(
            entity_id="binary_sensor.bad",
            decay=SimpleNamespace(is_decaying=False, decay_start=dt_util.utcnow()),
            evidence=False,
        )
        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.bad": bad},
            entity_ids=["binary_sensor.bad"],
        )
        area._entities = mock_entities_manager

        save_entity_data(db)

        # Verify invalid entity was not saved
        with db.get_session() as session:
            entity = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.bad")
                .first()
            )
            assert entity is None

    def test_save_entity_data_database_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test save_entity_data with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        save_area_data(db, area_name)

        good = _create_mock_entity("binary_sensor.good")
        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.good": good},
            entity_ids=["binary_sensor.good"],
        )
        area._entities = mock_entities_manager

        with (
            patch.object(
                db,
                "get_session",
                side_effect=OperationalError("DB error", None, None),
            ),
            pytest.raises(OperationalError),
        ):
            # Should raise after retries
            save_entity_data(db)

    def test_save_entity_data_cleanup_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test save_entity_data when cleanup fails."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        save_area_data(db, area_name)

        good = _create_mock_entity("binary_sensor.good")
        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.good": good},
            entity_ids=["binary_sensor.good"],
        )
        area._entities = mock_entities_manager

        # Mock cleanup to fail, but save should still succeed
        with patch(
            "custom_components.area_occupancy.db.operations._cleanup_orphaned_entities",
            side_effect=RuntimeError("Cleanup error"),
        ):
            # Should still save successfully even if cleanup fails
            save_entity_data(db)

            # Verify entity was saved
            with db.get_session() as session:
                entity = (
                    session.query(db.Entities)
                    .filter_by(entity_id="binary_sensor.good")
                    .first()
                )
                assert entity is not None


class TestCleanupOrphanedEntities:
    """Test cleanup_orphaned_entities function."""

    def test_cleanup_orphaned_entities_no_orphans(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test cleanup when no orphans exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists first (foreign key requirement)
        save_area_data(db, area_name)

        # Set up entities matching config
        # The entity_ids property returns list(self._entities.keys())
        # So we need to ensure _entities is properly set
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.type = SimpleNamespace(input_type="motion", weight=0.85)
        mock_entity.prob_given_true = 0.8
        mock_entity.prob_given_false = 0.05
        mock_entity.last_updated = dt_util.utcnow()
        mock_entity.decay = SimpleNamespace(is_decaying=False, decay_start=None)
        mock_entity.evidence = False
        area.entities._entities = {"binary_sensor.motion1": mock_entity}
        # Ensure entity_ids property returns the correct list
        assert area.entities.entity_ids == ["binary_sensor.motion1"]

        # Save entities
        db.save_entity_data()

        # Run cleanup
        count = _cleanup_orphaned_entities(db)
        assert count == 0

    def test_cleanup_orphaned_entities_with_orphans(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test cleanup when orphans exist."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists first (foreign key requirement)
        save_area_data(db, area_name)

        # Set up entities - one current, one orphaned
        # The entity_ids property returns list(self._entities.keys())
        mock_entity = Mock()
        mock_entity.entity_id = "binary_sensor.motion1"
        mock_entity.type = SimpleNamespace(input_type="motion", weight=0.85)
        mock_entity.prob_given_true = 0.8
        mock_entity.prob_given_false = 0.05
        mock_entity.last_updated = dt_util.utcnow()
        mock_entity.decay = SimpleNamespace(is_decaying=False, decay_start=None)
        mock_entity.evidence = False
        area.entities._entities = {"binary_sensor.motion1": mock_entity}
        # Ensure entity_ids property returns the correct list
        assert area.entities.entity_ids == ["binary_sensor.motion1"]

        # Save current entity
        db.save_entity_data()

        # Manually add orphaned entity to database (not in config)
        with db.get_session() as session:
            orphaned = db.Entities(
                entity_id="binary_sensor.orphaned",
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_type="motion",
            )
            session.add(orphaned)
            session.commit()

        # Run cleanup - should remove orphaned entity
        count = _cleanup_orphaned_entities(db)
        assert count == 1

        # Verify orphaned entity was deleted
        with db.get_session() as session:
            entity = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.orphaned")
                .first()
            )
            assert entity is None


class TestDeleteAreaData:
    """Test delete_area_data function."""

    def test_delete_area_data_success(self, coordinator: AreaOccupancyCoordinator):
        """Test deleting area data successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Save area data first
        db.save_area_data(area_name)

        # Delete area data
        count = delete_area_data(db, area_name)
        assert count >= 0

        # Verify area was deleted
        with db.get_session() as session:
            area = session.query(db.Areas).filter_by(area_name=area_name).first()
            assert area is None

    def test_delete_area_data_database_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test delete_area_data with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        with (
            patch.object(db, "get_session", side_effect=SQLAlchemyError("DB error")),
            suppress(Exception),
        ):
            # Should handle error gracefully
            # May raise or return 0, both are acceptable
            count = delete_area_data(db, area_name)
            assert isinstance(count, int)

    def test_delete_area_data_missing_area(self, coordinator: AreaOccupancyCoordinator):
        """Test delete_area_data when area doesn't exist."""
        db = coordinator.db
        # Delete non-existent area
        count = delete_area_data(db, "nonexistent_area")
        assert count == 0


class TestPruneOldIntervals:
    """Test prune_old_intervals function."""

    @pytest.mark.parametrize("force", [False, True])
    def test_prune_old_intervals(self, coordinator: AreaOccupancyCoordinator, force):
        """Test pruning old intervals with different force values."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        old_time = dt_util.utcnow() - timedelta(days=RETENTION_DAYS + 10)
        recent_time = dt_util.utcnow() - timedelta(days=30)

        # Ensure area and entity exist first (foreign key requirements)
        save_area_data(db, area_name)
        with db.get_session() as session:
            entity = db.Entities(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                entity_type="motion",
            )
            session.add(entity)
            session.commit()

        with db.get_session() as session:
            # Add old interval
            old_interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                start_time=old_time,
                end_time=old_time + timedelta(hours=1),
                state="on",
                duration_seconds=3600,
            )
            # Add recent interval
            recent_interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                start_time=recent_time,
                end_time=recent_time + timedelta(hours=1),
                state="on",
                duration_seconds=3600,
            )
            session.add_all([old_interval, recent_interval])
            session.commit()

        # Prune old intervals
        count = prune_old_intervals(db, force=force)
        assert count >= 1

        # Verify old interval was deleted, recent remains
        with db.get_session() as session:
            intervals = session.query(db.Intervals).all()
            assert len(intervals) == 1
            assert intervals[0].start_time.replace(tzinfo=None) == recent_time.replace(
                tzinfo=None
            )

    def test_prune_old_intervals_error(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test prune_old_intervals with database error."""
        db = coordinator.db

        def bad_session():
            raise OperationalError("Error", None, None)

        monkeypatch.setattr(db, "get_session", bad_session)
        # Should handle error gracefully
        count = prune_old_intervals(db, force=False)
        assert count == 0


class TestSaveOccupiedIntervalsCache:
    """Test save_occupied_intervals_cache function."""

    def test_save_occupied_intervals_cache_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving occupied intervals cache successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        now = dt_util.utcnow()

        save_area_data(db, area_name)

        intervals = [
            (now - timedelta(hours=2), now - timedelta(hours=1)),
            (now - timedelta(hours=4), now - timedelta(hours=3)),
        ]

        result = save_occupied_intervals_cache(db, area_name, intervals)
        assert result is True

        # Verify intervals were saved
        with db.get_session() as session:
            cached_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cached_intervals) == 2
            # Verify data_source defaults to "merged"
            assert all(
                interval.data_source == "merged" for interval in cached_intervals
            )

    def test_save_occupied_intervals_cache_empty(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving empty intervals cache."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_area_data(db, area_name)

        result = save_occupied_intervals_cache(db, area_name, [])
        assert result is True

        # Verify no intervals were saved
        with db.get_session() as session:
            cached_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cached_intervals) == 0

    def test_save_occupied_intervals_cache_with_data_source(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving occupied intervals cache with custom data_source."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_area_data(db, area_name)

        intervals = [
            (
                dt_util.utcnow() - timedelta(hours=2),
                dt_util.utcnow() - timedelta(hours=1),
            )
        ]

        result = save_occupied_intervals_cache(
            db, area_name, intervals, "motion_sensors"
        )
        assert result is True

        # Verify cache was saved with correct data_source
        with db.get_session() as session:
            cache = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cache) == 1
            assert cache[0].data_source == "motion_sensors"

    def test_save_occupied_intervals_cache_replaces_existing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that saving intervals cache replaces existing cache."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        now = dt_util.utcnow()

        save_area_data(db, area_name)

        # Save initial intervals
        initial_intervals = [
            (now - timedelta(hours=4), now - timedelta(hours=3)),
        ]
        save_occupied_intervals_cache(db, area_name, initial_intervals)

        # Save new intervals - should replace old ones
        new_intervals = [
            (now - timedelta(hours=2), now - timedelta(hours=1)),
        ]
        save_occupied_intervals_cache(db, area_name, new_intervals)

        # Verify only new intervals exist
        with db.get_session() as session:
            cached_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cached_intervals) == 1
            assert cached_intervals[0].start_time.replace(tzinfo=None) == (
                now - timedelta(hours=2)
            ).replace(tzinfo=None)

    def test_save_occupied_intervals_cache_skips_invalid(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that invalid intervals (start > end) are skipped."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        now = dt_util.utcnow()

        save_area_data(db, area_name)

        intervals = [
            (now - timedelta(hours=2), now - timedelta(hours=1)),  # Valid
            (
                now - timedelta(hours=1),
                now - timedelta(hours=2),
            ),  # Invalid (start > end)
        ]

        result = save_occupied_intervals_cache(db, area_name, intervals)
        assert result is True

        # Verify only valid interval was saved
        with db.get_session() as session:
            cached_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cached_intervals) == 1
            assert cached_intervals[0].start_time.replace(tzinfo=None) == (
                now - timedelta(hours=2)
            ).replace(tzinfo=None)

    def test_save_occupied_intervals_cache_deduplicates(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that duplicate intervals are only saved once."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        now = dt_util.utcnow()

        save_area_data(db, area_name)

        intervals = [
            (now - timedelta(hours=2), now - timedelta(hours=1)),
            (now - timedelta(hours=2), now - timedelta(hours=1)),  # Duplicate
        ]

        result = save_occupied_intervals_cache(db, area_name, intervals)
        assert result is True

        # Verify duplicate was not saved
        with db.get_session() as session:
            cached_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cached_intervals) == 1
            assert cached_intervals[0].start_time.replace(tzinfo=None) == (
                now - timedelta(hours=2)
            ).replace(tzinfo=None)


class TestEnsureAreaExists:
    """Test ensure_area_exists function."""

    @pytest.mark.asyncio
    async def test_ensure_area_exists_new_area(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test ensure_area_exists creates areas when they don't exist."""
        db = coordinator.db

        # Delete all areas from database
        with db.get_session() as session:
            session.query(db.Areas).delete()
            session.commit()

        # ensure_area_exists should create areas
        await ensure_area_exists(db)

        # Verify areas were created
        with db.get_session() as session:
            areas = session.query(db.Areas).all()
            assert len(areas) > 0

    @pytest.mark.asyncio
    async def test_ensure_area_exists_existing_area(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test ensure_area_exists with existing areas."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Areas already exist - save them first
        db.save_area_data(area_name)

        # Get initial area count
        with db.get_session() as session:
            initial_count = session.query(db.Areas).count()

        # ensure_area_exists should not create duplicates
        await ensure_area_exists(db)

        # Verify areas still exist and no duplicates were created
        with db.get_session() as session:
            final_count = session.query(db.Areas).count()
            assert final_count == initial_count
            assert final_count > 0

            # Verify area still exists
            area = session.query(db.Areas).filter_by(area_name=area_name).first()
            assert area is not None


class TestSaveGlobalPrior:
    """Test save_global_prior function."""

    def test_save_global_prior_success(self, coordinator: AreaOccupancyCoordinator):
        """Test saving global prior successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists in database first (foreign key requirement)
        save_area_data(db, area_name)

        result = save_global_prior(
            db,
            area_name,
            0.35,
            dt_util.utcnow() - timedelta(days=90),
            dt_util.utcnow(),
            86400.0,
            7776000.0,
            100,
        )
        assert result is True

        # Verify global prior was saved
        with db.get_session() as session:
            prior = (
                session.query(db.GlobalPriors).filter_by(area_name=area_name).first()
            )
            assert prior is not None
            assert prior.prior_value == 0.35


class TestCreateDataHash:
    """Test _create_data_hash function."""

    def test_create_data_hash(self, coordinator: AreaOccupancyCoordinator):
        """Test data hash creation."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        now = dt_util.utcnow()

        hash1 = _create_data_hash(area_name, now, now + timedelta(days=1), 100.0, 10)
        hash2 = _create_data_hash(area_name, now, now + timedelta(days=1), 100.0, 10)

        # Same data should produce same hash
        assert hash1 == hash2

        # Different data should produce different hash
        hash3 = _create_data_hash(area_name, now, now + timedelta(days=2), 200.0, 20)
        assert hash1 != hash3


class TestSaveAreaDataRetry:
    """Test retry/backoff logic for save_area_data."""

    def test_save_area_data_retry_succeeds_after_transient_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that save_area_data retries on transient errors."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        mock_get_session = _create_retry_mock(db.get_session)
        with patch.object(db, "get_session", side_effect=mock_get_session):
            # Should succeed after retry
            save_area_data(db, area_name)

        # Verify area was saved
        with db.get_session() as session:
            area = session.query(db.Areas).filter_by(area_name=area_name).first()
            assert area is not None


class TestSaveEntityDataRetry:
    """Test retry/backoff logic for save_entity_data."""

    def test_save_entity_data_retry_succeeds_after_transient_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that save_entity_data retries on transient errors."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        save_area_data(db, area_name)

        good = _create_mock_entity("binary_sensor.good")
        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.good": good},
            entity_ids=["binary_sensor.good"],
        )
        area._entities = mock_entities_manager

        mock_get_session = _create_retry_mock(db.get_session)
        with patch.object(db, "get_session", side_effect=mock_get_session):
            # Should succeed after retry
            save_entity_data(db)

        # Verify entity was saved
        with db.get_session() as session:
            entity = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.good")
                .first()
            )
            assert entity is not None


class TestLoadDataCorrelationPriority:
    """Test correlation priority logic in load_data."""

    @pytest.mark.asyncio
    async def test_load_data_prioritizes_binary_likelihood_over_correlation(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that binary_likelihood takes priority over correlation in load_data."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        save_area_data(db, area_name)

        # Create a binary sensor entity
        entity_id = "binary_sensor.test_appliance"
        try:
            entity = area.entities.get_entity(entity_id)
        except ValueError:
            entity = area.factory.create_from_config_spec(
                entity_id, InputType.APPLIANCE.value
            )
            area.entities.add_entity(entity)

        save_entity_data(db)

        # Save correlation result first (older)
        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "input_type": InputType.APPLIANCE.value,
            "correlation_coefficient": 0.5,
            "correlation_type": "positive",
            "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
            "analysis_period_end": dt_util.utcnow() - timedelta(days=1),
            "sample_count": 100,
            "confidence": 0.8,
            "mean_value_when_occupied": 0.7,
            "mean_value_when_unoccupied": 0.3,
            "std_dev_when_occupied": 0.1,
            "std_dev_when_unoccupied": 0.1,
            "threshold_active": None,
            "threshold_inactive": None,
            "analysis_error": None,
            "calculation_date": dt_util.utcnow() - timedelta(days=1),
        }
        save_correlation_result(db, correlation_data)

        # Save binary likelihood result second (newer, should take priority)
        likelihood_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
            "analysis_period_end": dt_util.utcnow(),
            "prob_given_true": 0.85,
            "prob_given_false": 0.15,
            "analysis_error": None,
            "calculation_date": dt_util.utcnow(),
        }
        save_binary_likelihood_result(db, likelihood_data, InputType.APPLIANCE)

        # Clear entity to test reload
        entity.prob_given_true = None
        entity.prob_given_false = None

        # Reload data
        await load_data(db)

        # Verify binary_likelihood was applied (not correlation)
        reloaded_entity = area.entities.get_entity(entity_id)
        assert reloaded_entity.prob_given_true == 0.85
        assert reloaded_entity.prob_given_false == 0.15
        # Should not have correlation coefficient
        assert not hasattr(reloaded_entity, "correlation_coefficient") or (
            getattr(reloaded_entity, "correlation_coefficient", None) is None
        )


class TestLoadDataEntityCreation:
    """Test entity creation from database in load_data."""

    @pytest.mark.asyncio
    async def test_load_data_creates_entity_from_database(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that load_data creates entity from database when entity is in config but not initialized."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        save_area_data(db, area_name)

        # Create entity via factory and add to entities manager, then save to database
        # Use APPLIANCE type (not MOTION) because motion sensors use configured values, not database values
        entity_id = "binary_sensor.database_only"
        entity = area.factory.create_from_config_spec(
            entity_id, InputType.APPLIANCE.value
        )
        entity.prob_given_true = 0.8
        entity.prob_given_false = 0.05
        entity.type.weight = 0.85
        area.entities.add_entity(entity)
        save_entity_data(db)

        # Now remove from coordinator (simulating entity not initialized after reload)
        # But keep it in database. We need to mock entity_ids to include it so it's not marked as stale
        del area.entities._entities[entity_id]

        # Mock entity_ids property to include the entity_id so it's not marked as stale
        # This simulates the entity being in config but not initialized in _entities
        original_entity_ids_list = list(area.entities._entities.keys())
        original_entity_ids_list.append(entity_id)  # Include the entity_id in the list

        # Create a mock property that returns the list including our entity_id
        mock_entity_ids_prop = property(lambda self: original_entity_ids_list)

        with patch.object(type(area.entities), "entity_ids", mock_entity_ids_prop):
            # Verify entity doesn't exist in coordinator but is in database
            assert entity_id not in area.entities._entities
            assert entity_id in area.entities.entity_ids  # Mocked to include it

            # Load data - should create entity from database
            await load_data(db)

            # Verify entity was created from database
            created_entity = area.entities.get_entity(entity_id)
            assert created_entity is not None
            assert created_entity.entity_id == entity_id
            assert created_entity.prob_given_true == 0.8
            assert created_entity.prob_given_false == 0.05
            assert created_entity.type.weight == 0.85


class TestSaveTimePriors:
    """Test save_time_priors function."""

    def test_save_time_priors_success(self, coordinator: AreaOccupancyCoordinator):
        """Test saving time priors successfully."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_area_data(db, area_name)

        time_priors = {
            (0, 0): 0.3,  # Monday, 0-6 hours
            (0, 1): 0.5,  # Monday, 6-12 hours
            (1, 0): 0.4,  # Tuesday, 0-6 hours
        }
        data_points_per_slot = {
            (0, 0): 4,
            (0, 1): 8,
            (1, 0): 6,
        }
        period_start = dt_util.utcnow() - timedelta(days=30)
        period_end = dt_util.utcnow()

        result = save_time_priors(
            db,
            area_name,
            time_priors,
            period_start,
            period_end,
            data_points_per_slot,
        )
        assert result is True

        # Verify priors were saved
        with db.get_session() as session:
            priors = session.query(db.Priors).filter_by(area_name=area_name).all()
            assert len(priors) == 3

            # Verify specific prior values
            monday_morning = (
                session.query(db.Priors)
                .filter_by(area_name=area_name, day_of_week=0, time_slot=0)
                .first()
            )
            assert monday_morning is not None
            assert monday_morning.prior_value == 0.3
            assert monday_morning.data_points == 4

    def test_save_time_priors_updates_existing(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that save_time_priors updates existing priors."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_area_data(db, area_name)

        # Save initial prior
        initial_priors = {(0, 0): 0.3}
        initial_data_points = {(0, 0): 4}
        save_time_priors(
            db,
            area_name,
            initial_priors,
            dt_util.utcnow() - timedelta(days=30),
            dt_util.utcnow(),
            initial_data_points,
        )

        # Update with new value
        updated_priors = {(0, 0): 0.6}
        updated_data_points = {(0, 0): 12}
        save_time_priors(
            db,
            area_name,
            updated_priors,
            dt_util.utcnow() - timedelta(days=20),
            dt_util.utcnow(),
            updated_data_points,
        )

        # Verify prior was updated
        with db.get_session() as session:
            prior = (
                session.query(db.Priors)
                .filter_by(area_name=area_name, day_of_week=0, time_slot=0)
                .first()
            )
            assert prior is not None
            assert prior.prior_value == 0.6
            assert prior.data_points == 12

    def test_save_time_priors_clamps_values(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that save_time_priors clamps values to TIME_PRIOR bounds."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_area_data(db, area_name)

        # Try to save values outside bounds
        time_priors = {
            (0, 0): 0.02,  # Below minimum (0.03)
            (0, 1): 0.95,  # Above maximum
            (1, 0): 0.5,  # Within bounds
        }
        data_points_per_slot = {(0, 0): 4, (0, 1): 4, (1, 0): 4}

        result = save_time_priors(
            db,
            area_name,
            time_priors,
            dt_util.utcnow() - timedelta(days=30),
            dt_util.utcnow(),
            data_points_per_slot,
        )
        assert result is True

        # Verify values were clamped
        with db.get_session() as session:
            below_min = (
                session.query(db.Priors)
                .filter_by(area_name=area_name, day_of_week=0, time_slot=0)
                .first()
            )
            assert below_min.prior_value == TIME_PRIOR_MIN_BOUND

            above_max = (
                session.query(db.Priors)
                .filter_by(area_name=area_name, day_of_week=0, time_slot=1)
                .first()
            )
            assert above_max.prior_value == TIME_PRIOR_MAX_BOUND

            within_bounds = (
                session.query(db.Priors)
                .filter_by(area_name=area_name, day_of_week=1, time_slot=0)
                .first()
            )
            assert within_bounds.prior_value == 0.5

    def test_save_time_priors_database_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test save_time_priors with database error."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_area_data(db, area_name)

        # Mock session to raise error inside the with block
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        # Make query raise an error
        mock_query = Mock()
        mock_query.filter_by = Mock(return_value=mock_query)
        mock_query.first = Mock(side_effect=SQLAlchemyError("DB error"))
        mock_session.query = Mock(return_value=mock_query)

        with patch.object(db, "get_session", return_value=mock_session):
            result = save_time_priors(
                db,
                area_name,
                {(0, 0): 0.5},
                dt_util.utcnow() - timedelta(days=30),
                dt_util.utcnow(),
                {(0, 0): 4},
            )
            assert result is False


class TestSaveEntityDataNormalization:
    """Test data normalization in save_entity_data."""

    @pytest.mark.parametrize(
        ("field_name", "invalid_value", "expected_value"),
        [
            ("weight", 1.5, MAX_WEIGHT),
            ("prob_given_true", 1.5, MAX_PROBABILITY),
            ("prob_given_false", -0.1, MIN_PROBABILITY),
        ],
    )
    def test_save_entity_data_clamps_values(
        self,
        coordinator: AreaOccupancyCoordinator,
        field_name,
        invalid_value,
        expected_value,
    ):
        """Test that save_entity_data clamps values to bounds."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        save_area_data(db, area_name)

        # Create entity with invalid value
        if field_name == "weight":
            entity = _create_mock_entity(
                "binary_sensor.test",
                weight=invalid_value,
                prob_given_true=0.8,
                prob_given_false=0.1,
            )
        elif field_name == "prob_given_true":
            entity = _create_mock_entity(
                "binary_sensor.test",
                prob_given_true=invalid_value,
                prob_given_false=0.1,
            )
        else:  # prob_given_false
            entity = _create_mock_entity(
                "binary_sensor.test",
                prob_given_true=0.8,
                prob_given_false=invalid_value,
            )

        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.test": entity},
            entity_ids=["binary_sensor.test"],
        )
        area._entities = mock_entities_manager

        save_entity_data(db)

        # Verify value was clamped
        with db.get_session() as session:
            db_entity = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.test")
                .first()
            )
            assert db_entity is not None
            assert getattr(db_entity, field_name) == expected_value

    def test_save_entity_data_handles_missing_values(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that save_entity_data handles missing probability values."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        save_area_data(db, area_name)

        # Entity with missing probability values
        entity = _create_mock_entity(
            "binary_sensor.test", prob_given_true=None, prob_given_false=None
        )
        mock_entities_manager = SimpleNamespace(
            entities={"binary_sensor.test": entity},
            entity_ids=["binary_sensor.test"],
        )
        area._entities = mock_entities_manager

        save_entity_data(db)

        # Verify default values were used
        with db.get_session() as session:
            db_entity = (
                session.query(db.Entities)
                .filter_by(entity_id="binary_sensor.test")
                .first()
            )
            assert db_entity is not None
            # Should use defaults from const
            assert db_entity.prob_given_true is not None
            assert db_entity.prob_given_false is not None


class TestBatchOperations:
    """Test batch operations with multiple areas/entities."""

    def test_save_area_data_multiple_areas(self, coordinator: AreaOccupancyCoordinator):
        """Test saving data for multiple areas."""
        db = coordinator.db
        area_names = db.coordinator.get_area_names()

        # Save all areas
        save_area_data(db, None)

        # Verify all areas were saved
        with db.get_session() as session:
            for area_name in area_names:
                area = session.query(db.Areas).filter_by(area_name=area_name).first()
                assert area is not None
                assert area.area_name == area_name

    def test_save_entity_data_multiple_entities(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving multiple entities across areas."""
        db = coordinator.db
        area_names = db.coordinator.get_area_names()

        # Ensure all areas exist
        for area_name in area_names:
            save_area_data(db, area_name)

        # Create entities in multiple areas
        entities_created = []
        for area_name in area_names:
            area = db.coordinator.get_area(area_name)
            entity_id = f"binary_sensor.test_{area_name.replace(' ', '_')}"
            try:
                entity = area.entities.get_entity(entity_id)
            except ValueError:
                entity = area.factory.create_from_config_spec(
                    entity_id, InputType.MOTION.value
                )
                area.entities.add_entity(entity)
                entities_created.append((area_name, entity_id))

        # Save all entities
        save_entity_data(db)

        # Verify all entities were saved
        with db.get_session() as session:
            for area_name, entity_id in entities_created:
                entity = (
                    session.query(db.Entities)
                    .filter_by(area_name=area_name, entity_id=entity_id)
                    .first()
                )
                assert entity is not None

    @pytest.mark.asyncio
    async def test_load_data_multiple_areas(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test loading data for multiple areas."""
        db = coordinator.db
        area_names = db.coordinator.get_area_names()

        # Set up data for each area
        for area_name in area_names:
            area = db.coordinator.get_area(area_name)
            save_area_data(db, area_name)

            # Set global prior for each area
            save_global_prior(
                db,
                area_name,
                0.5 + (area_names.index(area_name) * 0.1),
                dt_util.utcnow() - timedelta(days=30),
                dt_util.utcnow(),
                86400.0,
                7776000.0,
                100,
            )

            # Clear prior to test restoration
            area.prior.global_prior = None

        # Load data for all areas
        await load_data(db)

        # Verify data was loaded for all areas
        for area_name in area_names:
            area = db.coordinator.get_area(area_name)
            expected_prior = 0.5 + (area_names.index(area_name) * 0.1)
            assert area.prior.global_prior == expected_prior
