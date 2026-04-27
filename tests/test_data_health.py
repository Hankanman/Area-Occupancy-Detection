"""Tests for sensor health monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.data.health import HealthIssueType, HealthMonitor
from homeassistant.const import STATE_ON
from homeassistant.util import dt as dt_util


def _make_entity(
    entity_id: str,
    input_type: InputType,
    *,
    state: str | None = "off",
    last_updated: datetime | None = None,
    naive_last_updated: bool = False,
    evidence: bool | None = False,
) -> Entity:
    """Create a minimal Entity for health testing.

    When ``naive_last_updated`` is True, the supplied ``last_updated`` is
    stripped of tzinfo before being passed to the constructor. This mirrors
    the SQLite DB-restore path, where ``DateTime(timezone=True)`` columns
    return naive values; ``Entity.__post_init__`` is expected to normalize.
    """
    if last_updated is None:
        last_updated = dt_util.utcnow()
    if naive_last_updated:
        last_updated = last_updated.replace(tzinfo=None)

    entity_type = EntityType(
        input_type=input_type,
        weight=0.5,
        active_states=[STATE_ON],
    )
    return Entity(
        entity_id=entity_id,
        type=entity_type,
        prob_given_true=0.9,
        prob_given_false=0.1,
        decay=Decay(half_life=300),
        state_provider=lambda eid: Mock(state=state) if state is not None else None,
        last_updated=last_updated,
        previous_evidence=evidence,
    )


@pytest.fixture
def mock_hass() -> Mock:
    """Create a mock Home Assistant instance."""
    return Mock()


@pytest.fixture
def monitor(mock_hass: Mock) -> HealthMonitor:
    """Create a HealthMonitor for testing."""
    return HealthMonitor("test_area", "test_area_id", mock_hass)


# --- Stuck Active Tests ---


class TestStuckActive:
    """Tests for stuck active sensor detection."""

    def test_motion_stuck_active_above_threshold(self, monitor: HealthMonitor) -> None:
        """Motion sensor stuck 'on' for 3h should trigger stuck_active."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="on",
            last_updated=dt_util.utcnow() - timedelta(hours=3),
            evidence=True,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.STUCK_ACTIVE
        assert issues[0].entity_id == "binary_sensor.motion_1"
        assert issues[0].input_type == InputType.MOTION

    def test_motion_stuck_active_below_threshold(self, monitor: HealthMonitor) -> None:
        """Motion sensor stuck 'on' for 1h should NOT trigger (below 2h threshold)."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="on",
            last_updated=dt_util.utcnow() - timedelta(hours=1),
            evidence=True,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity})

        assert len(issues) == 0

    def test_door_stuck_active_below_threshold(self, monitor: HealthMonitor) -> None:
        """Door sensor open for 24h should NOT trigger (threshold is 48h)."""
        entity = _make_entity(
            "binary_sensor.door_1",
            InputType.DOOR,
            state="on",
            last_updated=dt_util.utcnow() - timedelta(hours=24),
            evidence=True,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"door_1": entity})

        assert len(issues) == 0

    def test_door_stuck_active_above_threshold(self, monitor: HealthMonitor) -> None:
        """Door sensor open for 50h should trigger (threshold is 48h)."""
        entity = _make_entity(
            "binary_sensor.door_1",
            InputType.DOOR,
            state="on",
            last_updated=dt_util.utcnow() - timedelta(hours=50),
            evidence=True,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"door_1": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.STUCK_ACTIVE


# --- Stuck Inactive Tests ---


class TestStuckInactive:
    """Tests for stuck inactive sensor detection."""

    def test_motion_inactive_above_threshold(self, monitor: HealthMonitor) -> None:
        """Motion sensor inactive for 8 days should trigger stuck_inactive."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=8),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.STUCK_INACTIVE

    def test_door_inactive_3_days_no_issue(self, monitor: HealthMonitor) -> None:
        """Door sensor closed for 3 days should NOT trigger (threshold is 14 days)."""
        entity = _make_entity(
            "binary_sensor.door_1",
            InputType.DOOR,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=3),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"door_1": entity})

        assert len(issues) == 0

    def test_appliance_inactive_above_threshold(self, monitor: HealthMonitor) -> None:
        """Appliance inactive for 30 days should trigger (threshold is 28 days)."""
        entity = _make_entity(
            "binary_sensor.oven",
            InputType.APPLIANCE,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=30),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"oven": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.STUCK_INACTIVE


# --- Unavailable Tests ---


class TestUnavailable:
    """Tests for unavailable sensor detection."""

    def test_unavailable_above_threshold(self, monitor: HealthMonitor) -> None:
        """Entity unavailable for 2h should trigger unavailable issue."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,  # unavailable
            last_updated=dt_util.utcnow() - timedelta(hours=2),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.UNAVAILABLE

    def test_unavailable_below_threshold(self, monitor: HealthMonitor) -> None:
        """Entity unavailable for 30min should NOT trigger (below 1h threshold)."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(minutes=30),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity})

        assert len(issues) == 0

    def test_unavailable_environmental_detected(self, monitor: HealthMonitor) -> None:
        """Environmental sensor unavailable should still be detected."""
        entity = _make_entity(
            "sensor.temperature_1",
            InputType.TEMPERATURE,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=3),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"temp_1": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.UNAVAILABLE


# --- Environmental Exclusion Tests ---


class TestEnvironmentalExclusion:
    """Tests that environmental sensors are excluded from stuck checks."""

    def test_temperature_not_stuck_checked(self, monitor: HealthMonitor) -> None:
        """Temperature sensor with old last_updated should NOT trigger stuck."""
        entity = _make_entity(
            "sensor.temperature_1",
            InputType.TEMPERATURE,
            state="22.5",
            last_updated=dt_util.utcnow() - timedelta(days=30),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"temp_1": entity})

        # Should not have stuck issues (environmental excluded from stuck checks)
        stuck_issues = [
            i
            for i in issues
            if i.issue_type
            in (HealthIssueType.STUCK_ACTIVE, HealthIssueType.STUCK_INACTIVE)
        ]
        assert len(stuck_issues) == 0

    def test_humidity_not_stuck_checked(self, monitor: HealthMonitor) -> None:
        """Humidity sensor with old last_updated should NOT trigger stuck."""
        entity = _make_entity(
            "sensor.humidity_1",
            InputType.HUMIDITY,
            state="65.0",
            last_updated=dt_util.utcnow() - timedelta(days=30),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"humidity_1": entity})

        stuck_issues = [
            i
            for i in issues
            if i.issue_type
            in (HealthIssueType.STUCK_ACTIVE, HealthIssueType.STUCK_INACTIVE)
        ]
        assert len(stuck_issues) == 0


# --- Sleep Sensor Exclusion Tests ---


class TestSleepExclusion:
    """Tests that sleep sensors are excluded from all health checks."""

    def test_sleep_sensor_excluded(self, monitor: HealthMonitor) -> None:
        """Sleep sensor should be excluded from all health checks."""
        entity = _make_entity(
            "binary_sensor.sleeping",
            InputType.SLEEP,
            state=None,  # unavailable
            last_updated=dt_util.utcnow() - timedelta(days=30),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"sleeping": entity})

        assert len(issues) == 0


# --- Never Triggered Tests ---


class TestNeverTriggered:
    """Tests for never-triggered sensor detection."""

    def test_never_triggered_old_last_updated(self, monitor: HealthMonitor) -> None:
        """Sensor with last_updated >7 days old and never active should trigger."""
        entity = _make_entity(
            "binary_sensor.oven",
            InputType.APPLIANCE,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=10),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"oven": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.NEVER_TRIGGERED

    def test_never_triggered_recent_last_updated(self, monitor: HealthMonitor) -> None:
        """Sensor with last_updated <7 days old should NOT trigger."""
        entity = _make_entity(
            "binary_sensor.oven",
            InputType.APPLIANCE,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=3),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"oven": entity})

        never_triggered = [
            i for i in issues if i.issue_type == HealthIssueType.NEVER_TRIGGERED
        ]
        assert len(never_triggered) == 0

    def test_previously_active_sensor_not_flagged(self, monitor: HealthMonitor) -> None:
        """Sensor with previous_evidence=True should NOT be flagged."""
        entity = _make_entity(
            "binary_sensor.oven",
            InputType.APPLIANCE,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=10),
            evidence=False,
        )
        # Simulate that the sensor was previously active
        entity.previous_evidence = True
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"oven": entity})

        never_triggered = [
            i for i in issues if i.issue_type == HealthIssueType.NEVER_TRIGGERED
        ]
        assert len(never_triggered) == 0


# --- Issue Resolution Tests ---


class TestIssueResolution:
    """Tests that issues resolve when sensors recover."""

    def test_issues_clear_when_resolved(self, monitor: HealthMonitor) -> None:
        """Issues should clear when sensors come back to normal."""
        # First check - sensor unavailable
        entity_unavail = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=5),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity_unavail})
        assert len(issues) == 1

        # Second check - sensor recovered
        entity_ok = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="off",
            last_updated=dt_util.utcnow(),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity_ok})
        assert len(issues) == 0


# --- Repair Issue Tests ---


class TestRepairIssues:
    """Tests for HA repair issue creation and deletion."""

    def test_repair_created_on_issues(self, monitor: HealthMonitor) -> None:
        """Repair issue should be created when health issues are found."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=5),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir") as mock_ir:
            monitor.check_health({"motion_1": entity})

        mock_ir.async_create_issue.assert_called_once()
        call_kwargs = mock_ir.async_create_issue.call_args
        assert call_kwargs.kwargs["translation_key"] == "sensor_health_unavailable"
        assert (
            call_kwargs.kwargs["translation_placeholders"]["entity_id"]
            == "binary_sensor.motion_1"
        )
        assert call_kwargs.kwargs["translation_placeholders"]["area"] == "test_area"

    def test_repair_deleted_when_resolved(self, monitor: HealthMonitor) -> None:
        """Repair issue should be deleted when the problem resolves."""
        # First: create an issue
        entity_bad = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=5),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            monitor.check_health({"motion_1": entity_bad})

        # Then: resolve the issue
        entity_ok = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="off",
            last_updated=dt_util.utcnow(),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir") as mock_ir:
            monitor.check_health({"motion_1": entity_ok})

        mock_ir.async_delete_issue.assert_called_once()

    def test_repair_not_recreated_if_unchanged(self, monitor: HealthMonitor) -> None:
        """Repair issues should still be created each check (idempotent)."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=5),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir") as mock_ir:
            monitor.check_health({"motion_1": entity})
            monitor.check_health({"motion_1": entity})

        # async_create_issue is called on every check (it's idempotent in HA)
        assert mock_ir.async_create_issue.call_count == 2
        # But no deletes since nothing resolved
        mock_ir.async_delete_issue.assert_not_called()


# --- Multiple Issues Tests ---


class TestMultipleIssues:
    """Tests for multiple issues in one area."""

    def test_multiple_issues_aggregated(self, monitor: HealthMonitor) -> None:
        """Multiple issues should be detected and aggregated."""
        entities = {
            "motion_1": _make_entity(
                "binary_sensor.motion_1",
                InputType.MOTION,
                state=None,
                last_updated=dt_util.utcnow() - timedelta(hours=5),
                evidence=None,
            ),
            "door_1": _make_entity(
                "binary_sensor.door_1",
                InputType.DOOR,
                state="on",
                last_updated=dt_util.utcnow() - timedelta(hours=50),
                evidence=True,
            ),
        }
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health(entities)

        assert len(issues) == 2
        issue_types = {i.issue_type for i in issues}
        assert HealthIssueType.UNAVAILABLE in issue_types
        assert HealthIssueType.STUCK_ACTIVE in issue_types


# --- Excluded Entity IDs Tests ---


class TestExcludedEntities:
    """Tests for entity exclusion by ID."""

    def test_excluded_entity_ids_skipped(self, monitor: HealthMonitor) -> None:
        """Entities in excluded set should not be checked."""
        entity = _make_entity(
            "binary_sensor.wasp",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=5),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health(
                {"wasp": entity},
                excluded_entity_ids={"binary_sensor.wasp"},
            )

        assert len(issues) == 0


# --- get_issue_for_entity Tests ---


class TestGetIssueForEntity:
    """Tests for the get_issue_for_entity lookup method."""

    def test_returns_issue_when_exists(self, monitor: HealthMonitor) -> None:
        """Should return the issue for a specific entity."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=5),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            monitor.check_health({"motion_1": entity})

        issue = monitor.get_issue_for_entity("binary_sensor.motion_1")
        assert issue is not None
        assert issue.issue_type == HealthIssueType.UNAVAILABLE

    def test_returns_none_when_healthy(self, monitor: HealthMonitor) -> None:
        """Should return None for a healthy entity."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="off",
            last_updated=dt_util.utcnow(),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            monitor.check_health({"motion_1": entity})

        issue = monitor.get_issue_for_entity("binary_sensor.motion_1")
        assert issue is None


# --- Properties Tests ---


class TestProperties:
    """Tests for HealthMonitor properties."""

    def test_has_critical_issues(self, monitor: HealthMonitor) -> None:
        """has_critical_issues should be True when stuck_active or unavailable."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="on",
            last_updated=dt_util.utcnow() - timedelta(hours=5),
            evidence=True,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            monitor.check_health({"motion_1": entity})

        assert monitor.has_critical_issues is True

    def test_no_critical_issues_for_inactive(self, monitor: HealthMonitor) -> None:
        """has_critical_issues should be False for only stuck_inactive."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=10),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            monitor.check_health({"motion_1": entity})

        assert monitor.has_critical_issues is False

    def test_last_check_updated(self, monitor: HealthMonitor) -> None:
        """last_check should be updated after each check."""
        assert monitor.last_check is None
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="off",
            last_updated=dt_util.utcnow(),
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            monitor.check_health({"motion_1": entity})

        assert monitor.last_check is not None


# --- Naive last_updated regression tests (PR #446 / issue #445) ---


class TestNaiveLastUpdatedRegression:
    """Regression: naive last_updated must not break health checks.

    Naive datetimes can leak in from SQLite-backed DB restoration —
    ``DateTime(timezone=True)`` columns return naive values on SQLite. The
    ``Entity`` contract is that ``last_updated`` is always tz-aware UTC, so
    ``Entity.__post_init__`` normalizes any naive input via ``to_utc()``.
    These tests pin both ends of that contract: the normalization itself,
    and the consumer-side health checks correctly handling caller-supplied
    naive timestamps.
    """

    def test_post_init_normalizes_naive_last_updated(self) -> None:
        """Construction with a naive last_updated yields tz-aware UTC."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            last_updated=dt_util.utcnow(),
            naive_last_updated=True,
        )
        assert entity.last_updated is not None
        assert entity.last_updated.tzinfo is dt_util.UTC

    def test_check_stuck_sensor_with_naive_input(self, monitor: HealthMonitor) -> None:
        """Stuck-active fires for motion sensor whose caller supplied naive."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state="on",
            last_updated=dt_util.utcnow() - timedelta(hours=3),
            naive_last_updated=True,
            evidence=True,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.STUCK_ACTIVE

    def test_check_unavailable_with_naive_input(self, monitor: HealthMonitor) -> None:
        """Unavailable fires for entity whose caller supplied naive."""
        entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=3),
            naive_last_updated=True,
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"motion_1": entity})

        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.UNAVAILABLE

    def test_check_never_triggered_with_naive_input(
        self, monitor: HealthMonitor
    ) -> None:
        """Never-triggered fires for appliance whose caller supplied naive."""
        entity = _make_entity(
            "binary_sensor.oven",
            InputType.APPLIANCE,
            state="off",
            last_updated=dt_util.utcnow() - timedelta(days=10),
            naive_last_updated=True,
            evidence=False,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_health({"oven": entity})

        never_triggered = [
            i for i in issues if i.issue_type == HealthIssueType.NEVER_TRIGGERED
        ]
        assert len(never_triggered) == 1


# --- Pipeline-scope health tests ---


class TestPipelineHealth:
    """Cover pipeline-scope checks emitted by ``check_pipeline_health``."""

    def test_no_issues_when_all_inputs_healthy(self, monitor: HealthMonitor) -> None:
        """Healthy inputs produce no pipeline issues."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 30,  # mature area
                has_global_prior=True,
                cache_age_hours=1.0,
                last_analysis_duration_ms=2_000.0,
                correlation_failure_count=0,
                correlatable_entity_count=10,
            )
        assert issues == []

    def test_insufficient_priors_after_grace_period(
        self, monitor: HealthMonitor
    ) -> None:
        """Mature area with no global prior triggers insufficient_priors."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 14,  # 14 days
                has_global_prior=False,
                cache_age_hours=1.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        assert len(issues) == 1
        assert issues[0].issue_type == HealthIssueType.INSUFFICIENT_PRIORS
        assert issues[0].entity_id is None

    def test_insufficient_priors_within_grace_period(
        self, monitor: HealthMonitor
    ) -> None:
        """Young area with no prior is *not* flagged — still warming up."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 3,  # 3 days
                has_global_prior=False,
                cache_age_hours=1.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        assert issues == []

    def test_stale_cache_above_threshold(self, monitor: HealthMonitor) -> None:
        """Cache older than threshold triggers stale_intervals_cache."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 30,
                has_global_prior=True,
                cache_age_hours=48.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        types = {i.issue_type for i in issues}
        assert HealthIssueType.STALE_INTERVALS_CACHE in types

    def test_missing_cache_after_grace_period(self, monitor: HealthMonitor) -> None:
        """Mature area with no cache at all also flags stale_intervals_cache."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 14,
                has_global_prior=True,
                cache_age_hours=None,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        types = {i.issue_type for i in issues}
        assert HealthIssueType.STALE_INTERVALS_CACHE in types

    def test_missing_cache_within_grace_period(self, monitor: HealthMonitor) -> None:
        """Young area with no cache yet is not flagged."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 1,  # 1 day
                has_global_prior=False,
                cache_age_hours=None,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        assert issues == []

    def test_slow_analysis_above_threshold(self, monitor: HealthMonitor) -> None:
        """Last analysis > 30s triggers slow_analysis."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 30,
                has_global_prior=True,
                cache_age_hours=1.0,
                last_analysis_duration_ms=45_000.0,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        types = {i.issue_type for i in issues}
        assert HealthIssueType.SLOW_ANALYSIS in types

    def test_correlation_failures_above_ratio(self, monitor: HealthMonitor) -> None:
        """≥50% correlatable entities failed → correlation_failures."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 30,
                has_global_prior=True,
                cache_age_hours=1.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=6,
                correlatable_entity_count=10,
            )
        types = {i.issue_type for i in issues}
        assert HealthIssueType.CORRELATION_FAILURES in types

    def test_correlation_failures_below_ratio(self, monitor: HealthMonitor) -> None:
        """Failures under the 50% ratio are tolerated."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 30,
                has_global_prior=True,
                cache_age_hours=1.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=3,
                correlatable_entity_count=10,
            )
        assert issues == []

    def test_correlation_failures_zero_correlatable(
        self, monitor: HealthMonitor
    ) -> None:
        """No correlatable entities → no failures issue (avoid div by zero)."""
        with patch("custom_components.area_occupancy.data.health.ir"):
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 30,
                has_global_prior=True,
                cache_age_hours=1.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        assert issues == []

    def test_pipeline_issues_use_distinct_repair_id_namespace(
        self, monitor: HealthMonitor
    ) -> None:
        """Pipeline issues are registered with the ``pipeline_health_*`` prefix."""
        with patch("custom_components.area_occupancy.data.health.ir") as mock_ir:
            monitor.check_pipeline_health(
                area_age_hours=24 * 14,
                has_global_prior=False,
                cache_age_hours=1.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )

        mock_ir.async_create_issue.assert_called_once()
        call = mock_ir.async_create_issue.call_args
        repair_id = call.args[2]
        assert repair_id.startswith("pipeline_health_")
        assert call.kwargs["translation_key"] == "pipeline_health_insufficient_priors"

    def test_pipeline_issues_merge_with_sensor_issues(
        self, monitor: HealthMonitor
    ) -> None:
        """Sensor issues from check_health survive a check_pipeline_health pass."""
        sensor_entity = _make_entity(
            "binary_sensor.motion_1",
            InputType.MOTION,
            state=None,
            last_updated=dt_util.utcnow() - timedelta(hours=3),
            evidence=None,
        )
        with patch("custom_components.area_occupancy.data.health.ir"):
            monitor.check_health({"motion_1": sensor_entity})
            issues = monitor.check_pipeline_health(
                area_age_hours=24 * 14,
                has_global_prior=False,
                cache_age_hours=1.0,
                last_analysis_duration_ms=None,
                correlation_failure_count=0,
                correlatable_entity_count=0,
            )
        types = {i.issue_type for i in issues}
        assert HealthIssueType.UNAVAILABLE in types  # from check_health
        assert HealthIssueType.INSUFFICIENT_PRIORS in types  # from pipeline check
