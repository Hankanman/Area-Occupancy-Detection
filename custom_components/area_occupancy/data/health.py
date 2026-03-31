"""Sensor health monitoring for Area Occupancy Detection.

Detects stuck, unavailable, and never-triggered sensors and raises
Home Assistant repair issues so users can see and act on degraded sensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
import logging
from typing import TYPE_CHECKING

from homeassistant.helpers import issue_registry as ir
from homeassistant.util import dt as dt_util

from ..const import DOMAIN
from .entity_type import BINARY_INPUT_TYPES, InputType

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

    from .entity import Entity

_LOGGER = logging.getLogger(__name__)

# Input types eligible for stuck-sensor detection (binary + power + motion)
_STUCK_CHECK_TYPES: set[InputType] = BINARY_INPUT_TYPES | {
    InputType.MOTION,
    InputType.POWER,
    InputType.COVER,
}

# Input types excluded from all health checks
_EXCLUDED_TYPES: set[InputType] = {
    InputType.SLEEP,
}

# How long a binary sensor can be stuck in its *active* state before flagging
STUCK_ACTIVE_THRESHOLDS: dict[InputType, timedelta] = {
    InputType.MOTION: timedelta(hours=2),
    InputType.MEDIA: timedelta(hours=12),
    InputType.APPLIANCE: timedelta(hours=24),
    InputType.DOOR: timedelta(hours=48),
    InputType.WINDOW: timedelta(hours=72),
    InputType.COVER: timedelta(hours=24),
}

# How long a binary sensor can remain *inactive* before flagging
STUCK_INACTIVE_THRESHOLDS: dict[InputType, timedelta] = {
    InputType.MOTION: timedelta(days=7),
    InputType.MEDIA: timedelta(days=14),
    InputType.APPLIANCE: timedelta(days=28),
    InputType.DOOR: timedelta(days=14),
    InputType.WINDOW: timedelta(days=14),
    InputType.COVER: timedelta(days=14),
    InputType.POWER: timedelta(days=14),
}

# Report unavailable sensors after this duration
UNAVAILABLE_THRESHOLD: timedelta = timedelta(hours=1)

# Only flag "never triggered" after this much integration uptime
NEVER_TRIGGERED_MIN_UPTIME: timedelta = timedelta(days=7)


class HealthIssueType(StrEnum):
    """Types of sensor health issues."""

    STUCK_ACTIVE = "stuck_active"
    STUCK_INACTIVE = "stuck_inactive"
    UNAVAILABLE = "unavailable"
    NEVER_TRIGGERED = "never_triggered"


@dataclass(frozen=True)
class HealthIssue:
    """A detected sensor health issue."""

    entity_id: str
    issue_type: HealthIssueType
    input_type: InputType
    since: datetime
    duration_hours: float
    details: str


def _issue_id(area_name: str, entity_id: str, issue_type: HealthIssueType) -> str:
    """Build a unique repair issue ID."""
    # Replace dots with underscores for cleaner issue IDs
    safe_entity = entity_id.replace(".", "_")
    return f"sensor_health_{area_name}_{safe_entity}_{issue_type}"


class HealthMonitor:
    """Monitors sensor health for a single area.

    Runs during the hourly analysis pipeline, checking each entity for:
    - Stuck active/inactive states (binary sensors only)
    - Prolonged unavailability
    - Sensors that have never triggered
    """

    def __init__(self, area_name: str, hass: HomeAssistant) -> None:
        """Initialize health monitor for an area."""
        self._area_name = area_name
        self._hass = hass
        self._issues: list[HealthIssue] = []
        self._last_check: datetime | None = None
        self._start_time: datetime = dt_util.utcnow()
        # Track active issue IDs so we can delete resolved ones
        self._active_issue_ids: set[str] = set()

    @property
    def issues(self) -> list[HealthIssue]:
        """Current health issues."""
        return self._issues

    @property
    def issue_count(self) -> int:
        """Number of current health issues."""
        return len(self._issues)

    @property
    def has_critical_issues(self) -> bool:
        """Whether any critical issues exist (stuck active or unavailable)."""
        return any(
            issue.issue_type
            in (HealthIssueType.STUCK_ACTIVE, HealthIssueType.UNAVAILABLE)
            for issue in self._issues
        )

    @property
    def last_check(self) -> datetime | None:
        """Timestamp of last health check."""
        return self._last_check

    def get_issue_for_entity(self, entity_id: str) -> HealthIssue | None:
        """Get the health issue for a specific entity, if any."""
        for issue in self._issues:
            if issue.entity_id == entity_id:
                return issue
        return None

    def check_health(
        self,
        entities: dict[str, Entity],
        excluded_entity_ids: set[str] | None = None,
    ) -> list[HealthIssue]:
        """Run all health checks on entities and update repair issues.

        Args:
            entities: All entities in the area
            excluded_entity_ids: Entity IDs to skip (e.g., wasp/sleep virtual sensors)

        Returns:
            List of detected health issues
        """
        now = dt_util.utcnow()
        self._last_check = now
        excluded = excluded_entity_ids or set()
        issues: list[HealthIssue] = []

        for entity in entities.values():
            if entity.entity_id in excluded:
                continue
            if entity.type.input_type in _EXCLUDED_TYPES:
                continue

            issue = self._check_unavailable(entity, now)
            if issue:
                issues.append(issue)
                continue  # Skip other checks if unavailable

            issue = self._check_stuck_sensor(entity, now)
            if issue:
                issues.append(issue)

        # Never-triggered check requires minimum uptime
        if now - self._start_time >= NEVER_TRIGGERED_MIN_UPTIME:
            for entity in entities.values():
                if entity.entity_id in excluded:
                    continue
                if entity.type.input_type in _EXCLUDED_TYPES:
                    continue
                # Only if not already flagged for another issue
                if any(i.entity_id == entity.entity_id for i in issues):
                    continue
                issue = self._check_never_triggered(entity, now)
                if issue:
                    issues.append(issue)

        self._issues = issues
        self._update_repair_issues()
        return issues

    def _check_stuck_sensor(self, entity: Entity, now: datetime) -> HealthIssue | None:
        """Check if a binary sensor is stuck in one state too long."""
        if entity.type.input_type not in _STUCK_CHECK_TYPES:
            return None

        if entity.last_updated is None:
            return None

        duration = now - entity.last_updated
        evidence = entity.evidence

        # Check stuck active
        if evidence is True:
            threshold = STUCK_ACTIVE_THRESHOLDS.get(entity.type.input_type)
            if threshold and duration >= threshold:
                hours = duration.total_seconds() / 3600
                return HealthIssue(
                    entity_id=entity.entity_id,
                    issue_type=HealthIssueType.STUCK_ACTIVE,
                    input_type=entity.type.input_type,
                    since=entity.last_updated,
                    duration_hours=round(hours, 1),
                    details=(
                        f"{entity.type.input_type.value} sensor has been active "
                        f"for {hours:.0f}h (threshold: "
                        f"{threshold.total_seconds() / 3600:.0f}h)"
                    ),
                )

        # Check stuck inactive
        if evidence is False:
            threshold = STUCK_INACTIVE_THRESHOLDS.get(entity.type.input_type)
            if threshold and duration >= threshold:
                hours = duration.total_seconds() / 3600
                return HealthIssue(
                    entity_id=entity.entity_id,
                    issue_type=HealthIssueType.STUCK_INACTIVE,
                    input_type=entity.type.input_type,
                    since=entity.last_updated,
                    duration_hours=round(hours, 1),
                    details=(
                        f"{entity.type.input_type.value} sensor hasn't changed "
                        f"state for {hours / 24:.0f} days (threshold: "
                        f"{threshold.total_seconds() / 86400:.0f} days)"
                    ),
                )

        return None

    def _check_unavailable(self, entity: Entity, now: datetime) -> HealthIssue | None:
        """Check if a sensor has been unavailable for too long."""
        if entity.available:
            return None

        if entity.last_updated is None:
            return None

        duration = now - entity.last_updated
        if duration < UNAVAILABLE_THRESHOLD:
            return None

        hours = duration.total_seconds() / 3600
        return HealthIssue(
            entity_id=entity.entity_id,
            issue_type=HealthIssueType.UNAVAILABLE,
            input_type=entity.type.input_type,
            since=entity.last_updated,
            duration_hours=round(hours, 1),
            details=(
                f"Sensor has been unavailable for {hours:.0f}h "
                f"(possible dead battery or connectivity issue)"
            ),
        )

    def _check_never_triggered(
        self, entity: Entity, now: datetime
    ) -> HealthIssue | None:
        """Check if a binary sensor has never been active."""
        if entity.type.input_type not in _STUCK_CHECK_TYPES:
            return None

        if entity.last_updated is None:
            return None

        # If evidence is currently active, it has triggered
        if entity.evidence is True:
            return None

        # Check if last_updated is close to start time (never had a transition)
        # A sensor that has transitioned would have last_updated >> start_time
        time_since_start = now - self._start_time
        time_since_update = now - entity.last_updated

        # If the sensor was last updated close to when we started tracking
        # (within a small margin) and has been running for 7+ days, flag it
        if time_since_update >= time_since_start - timedelta(minutes=5):
            days = time_since_start.total_seconds() / 86400
            return HealthIssue(
                entity_id=entity.entity_id,
                issue_type=HealthIssueType.NEVER_TRIGGERED,
                input_type=entity.type.input_type,
                since=self._start_time,
                duration_hours=round(days * 24, 1),
                details=(
                    f"{entity.type.input_type.value} sensor has never been "
                    f"active in {days:.0f} days of monitoring "
                    f"(possible misconfiguration)"
                ),
            )

        return None

    def _update_repair_issues(self) -> None:
        """Create or delete HA repair issues based on current health state."""
        current_issue_ids: set[str] = set()

        # Severity mapping
        severity_map = {
            HealthIssueType.STUCK_ACTIVE: ir.IssueSeverity.ERROR,
            HealthIssueType.UNAVAILABLE: ir.IssueSeverity.ERROR,
            HealthIssueType.STUCK_INACTIVE: ir.IssueSeverity.WARNING,
            HealthIssueType.NEVER_TRIGGERED: ir.IssueSeverity.WARNING,
        }

        for issue in self._issues:
            repair_id = _issue_id(self._area_name, issue.entity_id, issue.issue_type)
            current_issue_ids.add(repair_id)

            ir.async_create_issue(
                self._hass,
                DOMAIN,
                repair_id,
                is_fixable=False,
                severity=severity_map.get(issue.issue_type, ir.IssueSeverity.WARNING),
                translation_key=f"sensor_health_{issue.issue_type}",
                translation_placeholders={
                    "area": self._area_name,
                    "entity_id": issue.entity_id,
                    "duration": str(round(issue.duration_hours)),
                    "sensor_type": issue.input_type.value,
                },
            )

        # Delete resolved issues
        resolved_ids = self._active_issue_ids - current_issue_ids
        for resolved_id in resolved_ids:
            ir.async_delete_issue(self._hass, DOMAIN, resolved_id)

        if resolved_ids:
            _LOGGER.info(
                "Resolved %d sensor health issue(s) in area '%s'",
                len(resolved_ids),
                self._area_name,
            )

        if self._issues and current_issue_ids != self._active_issue_ids:
            _LOGGER.warning(
                "Sensor health issues in area '%s': %s",
                self._area_name,
                ", ".join(f"{i.entity_id} ({i.issue_type})" for i in self._issues),
            )

        self._active_issue_ids = current_issue_ids
