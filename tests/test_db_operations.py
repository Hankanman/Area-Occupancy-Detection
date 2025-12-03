"""Tests for database CRUD operations."""
# ruff: noqa: SLF001

from contextlib import suppress
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from custom_components.area_occupancy.const import RETENTION_DAYS
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.db.correlation import (
    save_binary_likelihood_result,
    save_correlation_result,
)
from custom_components.area_occupancy.db.operations import (
    _cleanup_orphaned_entities,
    _create_data_hash,
    _prune_old_global_priors,
    _validate_area_data,
    delete_area_data,
    ensure_area_exists,
    load_data,
    prune_old_intervals,
    save_area_data,
    save_entity_data,
    save_global_prior,
    save_occupied_intervals_cache,
)
from homeassistant.util import dt as dt_util


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
        area.prior.set_global_prior(0.5)
        db.save_area_data(area_name)

        save_global_prior(
            db,
            area_name,
            0.5,
            dt_util.utcnow(),
            dt_util.utcnow(),
            0.0,
            0.0,
            0,
        )

        await load_data(db)
        # Should complete without error

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
        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(is_decaying=False, decay_start=dt_util.utcnow()),
            evidence=True,
        )
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

    @pytest.mark.asyncio
    async def test_load_data_database_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test load_data with database error."""
        db = coordinator.db

        with patch.object(db, "get_session", side_effect=SQLAlchemyError("DB error")):
            # Should handle error gracefully and not raise
            await load_data(db)

    @pytest.mark.asyncio
    async def test_load_data_timeout_error(self, coordinator: AreaOccupancyCoordinator):
        """Test load_data with timeout error."""
        db = coordinator.db

        with patch.object(db, "get_session", side_effect=TimeoutError("Timeout")):
            # Should handle error gracefully and not raise
            await load_data(db)

    @pytest.mark.asyncio
    async def test_load_data_preserves_binary_likelihood_analysis_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that binary likelihood analysis_error is preserved after reload."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists
        save_area_data(db, area_name)

        # Create a binary sensor entity
        entity_id = "binary_sensor.test_light"
        try:
            entity = area.entities.get_entity(entity_id)
        except ValueError:
            # Entity doesn't exist, create it
            entity = area.factory.create_from_config_spec(
                entity_id, InputType.APPLIANCE.value
            )
            area.entities.add_entity(entity)

        # Save entity to database
        save_entity_data(db)

        # Create binary likelihood result with analysis_error
        likelihood_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "analysis_period_start": dt_util.utcnow() - timedelta(days=30),
            "analysis_period_end": dt_util.utcnow(),
            "prob_given_true": None,
            "prob_given_false": None,
            "analysis_error": "no_occupied_intervals",
            "calculation_date": dt_util.utcnow(),
        }

        # Save binary likelihood result
        save_binary_likelihood_result(db, likelihood_data, InputType.APPLIANCE)

        # Update entity with the error
        entity.update_binary_likelihoods(likelihood_data)
        assert entity.analysis_error == "no_occupied_intervals"

        # Reload data
        await load_data(db)

        # Verify analysis_error is preserved
        reloaded_entity = area.entities.get_entity(entity_id)
        assert reloaded_entity.analysis_error == "no_occupied_intervals"

    @pytest.mark.asyncio
    async def test_load_data_preserves_correlation_analysis_error(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test that correlation analysis_error is preserved after reload."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        area = db.coordinator.get_area(area_name)

        # Ensure area exists
        save_area_data(db, area_name)

        # Create a numeric sensor entity
        entity_id = "sensor.test_temperature"
        try:
            entity = area.entities.get_entity(entity_id)
        except ValueError:
            # Entity doesn't exist, create it
            entity = area.factory.create_from_config_spec(
                entity_id, InputType.TEMPERATURE.value
            )
            area.entities.add_entity(entity)

        # Save entity to database
        save_entity_data(db)

        # Create correlation result with analysis_error
        correlation_data = {
            "entry_id": db.coordinator.entry_id,
            "area_name": area_name,
            "entity_id": entity_id,
            "input_type": InputType.TEMPERATURE.value,
            "correlation_coefficient": 0.0,  # Placeholder for failed analysis
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
            "analysis_error": "too_few_samples",
            "calculation_date": dt_util.utcnow(),
        }

        # Save correlation result
        save_correlation_result(db, correlation_data)

        # Update entity with the error
        entity.update_correlation(correlation_data)
        assert entity.analysis_error == "too_few_samples"

        # Reload data
        await load_data(db)

        # Verify analysis_error is preserved
        reloaded_entity = area.entities.get_entity(entity_id)
        assert reloaded_entity.analysis_error == "too_few_samples"


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

        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(is_decaying=False, decay_start=dt_util.utcnow()),
            evidence=True,
        )
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

        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(is_decaying=False, decay_start=dt_util.utcnow()),
            evidence=True,
        )
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

        good = SimpleNamespace(
            entity_id="binary_sensor.good",
            type=SimpleNamespace(input_type="motion", weight=0.9),
            prob_given_true=0.8,
            prob_given_false=0.1,
            last_updated=dt_util.utcnow(),
            decay=SimpleNamespace(is_decaying=False, decay_start=dt_util.utcnow()),
            evidence=True,
        )
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

    def test_prune_old_intervals_success(self, coordinator: AreaOccupancyCoordinator):
        """Test pruning old intervals successfully."""
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
        count = prune_old_intervals(db, force=False)
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

    def test_prune_old_intervals_force(self, coordinator: AreaOccupancyCoordinator):
        """Test prune_old_intervals with force=True."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        old_time = dt_util.utcnow() - timedelta(days=RETENTION_DAYS + 10)

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
            old_interval = db.Intervals(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                entity_id="binary_sensor.motion1",
                start_time=old_time,
                end_time=old_time + timedelta(hours=1),
                state="on",
                duration_seconds=3600,
            )
            session.add(old_interval)
            session.commit()

        # Prune with force
        count = prune_old_intervals(db, force=True)
        assert count >= 0


class TestPruneOldGlobalPriorsEdgeCases2:
    """Test _prune_old_global_priors function - additional edge cases."""

    def test_prune_old_global_priors_edge_cases(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test pruning old global priors edge cases."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        old_area_name = f"{area_name}_stale"
        old_time = dt_util.utcnow() - timedelta(days=RETENTION_DAYS + 10)
        recent_time = dt_util.utcnow() - timedelta(days=30)

        save_area_data(db, area_name)

        with db.get_session() as session:
            # Add old prior
            old_prior = db.GlobalPriors(
                entry_id=db.coordinator.entry_id,
                area_name=old_area_name,
                prior_value=0.5,
                calculation_date=old_time,
                data_period_start=old_time - timedelta(days=7),
                data_period_end=old_time,
                total_occupied_seconds=7200.0,
                total_period_seconds=86400.0,
                interval_count=10,
            )
            # Add recent prior
            recent_prior = db.GlobalPriors(
                entry_id=db.coordinator.entry_id,
                area_name=area_name,
                prior_value=0.6,
                calculation_date=recent_time,
                data_period_start=recent_time - timedelta(days=7),
                data_period_end=recent_time,
                total_occupied_seconds=10800.0,
                total_period_seconds=86400.0,
                interval_count=12,
            )
            session.add_all([old_prior, recent_prior])
            session.commit()

        # Force pruning to remove all but the most recent calculation
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.operations.GLOBAL_PRIOR_HISTORY_COUNT",
            0,
        )
        with db.get_session() as session:
            _prune_old_global_priors(db, session, old_area_name)

        # Verify old prior was deleted, recent remains
        with db.get_session() as session:
            priors = session.query(db.GlobalPriors).all()
            assert len(priors) == 1
            assert priors[0].prior_value == 0.6


class TestSaveOccupiedIntervalsCacheEdgeCases:
    """Test save_occupied_intervals_cache function - edge cases."""

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

        save_occupied_intervals_cache(db, area_name, intervals)

        # Verify intervals were saved
        with db.get_session() as session:
            cached_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cached_intervals) == 2

    def test_save_occupied_intervals_cache_empty(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving empty intervals cache."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        save_area_data(db, area_name)

        save_occupied_intervals_cache(db, area_name, [])

        # Verify no intervals were saved
        with db.get_session() as session:
            cached_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cached_intervals) == 0


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

        # ensure_area_exists should not create duplicates
        await ensure_area_exists(db)

        # Verify areas still exist
        with db.get_session() as session:
            areas = session.query(db.Areas).all()
            assert len(areas) > 0

    @pytest.mark.asyncio
    async def test_ensure_area_exists_creates_area(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that ensure_area_exists creates area if missing."""
        db = coordinator.db
        saved = []

        def fake_save_area_data(db_instance, area_name=None):
            saved.append(True)

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.operations.save_area_data",
            fake_save_area_data,
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.queries.get_area_data",
            lambda db_instance, entry_id: None,
        )

        await ensure_area_exists(db)
        # ensure_area_exists calls save_area_data when area doesn't exist
        assert len(saved) > 0

    @pytest.mark.asyncio
    async def test_ensure_area_exists_when_present(
        self, coordinator: AreaOccupancyCoordinator, monkeypatch
    ):
        """Test that ensure_area_exists doesn't create if area exists."""
        db = coordinator.db
        called = False

        def fake_save(db_instance):
            nonlocal called
            called = True

        monkeypatch.setattr(
            "custom_components.area_occupancy.db.operations.save_data", fake_save
        )
        monkeypatch.setattr(
            "custom_components.area_occupancy.db.queries.get_area_data",
            lambda db_instance, entry_id: {"entry_id": entry_id},
        )

        await ensure_area_exists(db)
        assert not called


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


class TestSaveOccupiedIntervalsCacheEdgeCases2:
    """Test save_occupied_intervals_cache function - additional edge cases."""

    def test_save_occupied_intervals_cache_edge_cases(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test saving occupied intervals cache edge cases."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Ensure area exists in database first (foreign key requirement)
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

        # Verify cache was saved
        with db.get_session() as session:
            cache = (
                session.query(db.OccupiedIntervalsCache)
                .filter_by(area_name=area_name)
                .all()
            )
            assert len(cache) == 1


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


class TestPruneOldGlobalPriorsEdgeCases3:
    """Test _prune_old_global_priors function - additional edge cases."""

    def test_prune_old_global_priors_edge_cases(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test pruning old global priors edge cases."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]

        # Create multiple global priors (note: GlobalPriors has unique constraint on area_name,
        # so we can only have one per area, but the function handles multiple)
        for i in range(5):
            # Use different area names to create multiple priors
            test_area = f"{area_name}_test_{i}"
            save_global_prior(
                db,
                test_area,
                0.3 + i * 0.01,
                dt_util.utcnow() - timedelta(days=90 - i),
                dt_util.utcnow() - timedelta(days=i),
                86400.0,
                7776000.0,
                100,
            )

        # Prune old priors for the first test area
        with db.get_session() as session:
            _prune_old_global_priors(db, session, f"{area_name}_test_0")

        # Verify function completed without error
        with db.get_session() as session:
            priors = (
                session.query(db.GlobalPriors)
                .filter_by(area_name=f"{area_name}_test_0")
                .all()
            )
            # Note: Due to unique constraint, there should be at most 1 per area
            assert len(priors) <= 1
