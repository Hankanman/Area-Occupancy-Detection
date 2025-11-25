"""Tests for binary likelihood analysis functions."""

from datetime import timedelta

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.db.correlation import analyze_binary_likelihoods
from homeassistant.util import dt as dt_util


class TestAnalyzeBinaryLikelihoods:
    """Test analyze_binary_likelihoods function."""

    def test_analyze_binary_likelihoods_success(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test successful binary likelihood analysis."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"
        now = dt_util.utcnow()

        # Ensure area exists first
        db.save_area_data(area_name)

        # Create motion sensor intervals for occupied periods
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]
        with db.get_locked_session() as session:
            # Create motion intervals (occupied periods)
            for i in range(10):
                start = now - timedelta(hours=24 - i * 2)
                end = start + timedelta(hours=1)
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=motion_entity_id,
                    start_time=start,
                    end_time=end,
                    state="on",
                    duration_seconds=3600,
                )
                session.add(interval)

            # Create light intervals (active during some occupied periods)
            for i in range(10):
                start = now - timedelta(hours=24 - i * 2)
                end = start + timedelta(hours=1)
                # Light is on during first 5 occupied periods
                state = "on" if i < 5 else "off"
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    start_time=start,
                    end_time=end,
                    state=state,
                    duration_seconds=3600,
                )
                session.add(interval)
            session.commit()

        # Calculate and save occupied intervals cache from motion sensors
        # Create simple intervals list from motion sensor intervals
        motion_intervals = [
            (now - timedelta(hours=24 - i * 2), now - timedelta(hours=23 - i * 2))
            for i in range(10)
        ]
        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        # Analyze binary likelihoods
        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            active_states=["on"],
        )

        assert result is not None
        assert result["prob_given_true"] is not None
        assert result["prob_given_false"] is not None
        assert result["analysis_error"] is None
        # Light should be more likely on when occupied
        assert result["prob_given_true"] > result["prob_given_false"]
        # Probabilities should be clamped
        assert 0.05 <= result["prob_given_true"] <= 0.95
        assert 0.05 <= result["prob_given_false"] <= 0.95

    def test_analyze_binary_likelihoods_no_active_states(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis without active states."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"

        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            active_states=None,
        )

        assert result is None

    def test_analyze_binary_likelihoods_no_occupied_intervals(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis with no occupied intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create light intervals but no motion intervals
        with db.get_locked_session() as session:
            for i in range(5):
                start = now - timedelta(hours=24 - i * 2)
                end = start + timedelta(hours=1)
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=entity_id,
                    start_time=start,
                    end_time=end,
                    state="on",
                    duration_seconds=3600,
                )
                session.add(interval)
            session.commit()

        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            active_states=["on"],
        )

        assert result is not None
        assert result["prob_given_true"] is None
        assert result["prob_given_false"] is None
        assert result["analysis_error"] == "no_occupied_intervals"

    def test_analyze_binary_likelihoods_no_sensor_data(
        self, coordinator: AreaOccupancyCoordinator
    ):
        """Test binary likelihood analysis with no sensor intervals."""
        db = coordinator.db
        area_name = db.coordinator.get_area_names()[0]
        entity_id = "light.test_light"
        now = dt_util.utcnow()

        db.save_area_data(area_name)

        # Create motion intervals but no light intervals
        motion_entity_id = db.coordinator.get_area(area_name).config.sensors.motion[0]
        with db.get_locked_session() as session:
            for i in range(5):
                start = now - timedelta(hours=24 - i * 2)
                end = start + timedelta(hours=1)
                interval = db.Intervals(
                    entry_id=db.coordinator.entry_id,
                    area_name=area_name,
                    entity_id=motion_entity_id,
                    start_time=start,
                    end_time=end,
                    state="on",
                    duration_seconds=3600,
                )
                session.add(interval)
            session.commit()

        # Calculate and save occupied intervals cache from motion sensors
        # Create simple intervals list from motion sensor intervals
        motion_intervals = [
            (now - timedelta(hours=24 - i * 2), now - timedelta(hours=23 - i * 2))
            for i in range(10)
        ]
        db.save_occupied_intervals_cache(area_name, motion_intervals, "motion_sensors")

        result = analyze_binary_likelihoods(
            db,
            area_name,
            entity_id,
            analysis_period_days=30,
            active_states=["on"],
        )

        assert result is not None
        assert result["prob_given_true"] is None
        assert result["prob_given_false"] is None
        assert result["analysis_error"] == "no_sensor_data"
