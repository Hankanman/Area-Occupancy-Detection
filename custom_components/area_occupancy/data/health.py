"""Sensor and pipeline health monitoring for Area Occupancy Detection.

Two families of checks are run per area:

- **Sensor health** — per-entity anomalies (stuck active/inactive,
  unavailable, never-triggered).
- **Pipeline health** — per-area calculation anomalies (priors haven't
  trained, occupied-intervals cache is stale, last analysis cycle was
  slow, correlation analysis is failing for too many sensors).

Both surfaces emit Home Assistant repair issues with distinct
translation-key namespaces (``sensor_health_*`` vs ``pipeline_health_*``)
and share the same ``HealthIssue`` shape so callers (e.g. diagnostics)
can iterate them uniformly.
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

# A sensor whose last_updated is older than this and has never been active
# is flagged as "never triggered" (uses persisted last_updated, survives restarts)
NEVER_TRIGGERED_THRESHOLD: timedelta = timedelta(days=7)

# --- Pipeline-health thresholds ---

# An area is "old enough" that priors should have trained after this many
# days. Below this grace period we assume the integration is still
# warming up and don't flag missing global priors.
PRIORS_TRAINING_GRACE_PERIOD: timedelta = timedelta(days=7)

# Occupied-intervals cache is rebuilt hourly; flag if it's older than this
# (a slightly larger margin than the 24h validity window in db.queries).
STALE_CACHE_THRESHOLD: timedelta = timedelta(hours=25)

# Last full analysis cycle is flagged as "slow" if it took longer than this.
SLOW_ANALYSIS_THRESHOLD_MS: float = 30_000.0

# Fraction of correlatable entities whose ``analysis_error`` indicates a real
# failure (not a designed exclusion) before we flag the area's correlation
# pipeline as broken. 0.5 = "half or more failed".
CORRELATION_FAILURE_RATIO: float = 0.5

# ``analysis_error`` strings that represent real correlation failures (not
# "by design" states like ``not_analyzed`` or ``motion_sensor_excluded``).
# Sourced from db/correlation.py — keep this in sync when new failure
# branches are added there. Public because the analysis pipeline reads it
# when counting per-area failures to feed into ``check_pipeline_health``.
CORRELATION_FAILURE_ERRORS: frozenset[str] = frozenset(
    {
        "no_occupied_intervals",
        "no_occupied_time",
        "no_unoccupied_time",
        "no_occupancy_data",
        "no_sensor_data",
        "no_state_changes",
        "no_active_intervals",
        "no_active_during_occupied",
        "no_occupied_samples",
        "no_unoccupied_samples",
        "no_correlation",
        "insufficient_data",
        "too_few_samples",
        "too_few_samples_after_filtering",
        "zero_samples_after_filtering",
    }
)


class HealthIssueType(StrEnum):
    """Types of health issues — sensor-scope and pipeline-scope."""

    # Sensor-scope (per-entity)
    STUCK_ACTIVE = "stuck_active"
    STUCK_INACTIVE = "stuck_inactive"
    UNAVAILABLE = "unavailable"
    NEVER_TRIGGERED = "never_triggered"

    # Pipeline-scope (per-area, no entity_id)
    INSUFFICIENT_PRIORS = "insufficient_priors"
    STALE_INTERVALS_CACHE = "stale_intervals_cache"
    SLOW_ANALYSIS = "slow_analysis"
    CORRELATION_FAILURES = "correlation_failures"


_PIPELINE_ISSUE_TYPES: frozenset[HealthIssueType] = frozenset(
    {
        HealthIssueType.INSUFFICIENT_PRIORS,
        HealthIssueType.STALE_INTERVALS_CACHE,
        HealthIssueType.SLOW_ANALYSIS,
        HealthIssueType.CORRELATION_FAILURES,
    }
)


@dataclass(frozen=True)
class HealthIssue:
    """A detected health issue.

    For sensor-scope issues, ``entity_id`` and ``input_type`` identify the
    affected sensor. For pipeline-scope issues both are ``None`` and the
    issue applies to the whole area.
    """

    entity_id: str | None
    issue_type: HealthIssueType
    input_type: InputType | None
    since: datetime
    duration_hours: float
    details: str


def _issue_id(area_id: str, entity_id: str | None, issue_type: HealthIssueType) -> str:
    """Build a unique repair issue ID using the stable area_id.

    Sensor-scope issues key on ``area_id`` + ``entity_id`` + type; pipeline
    issues drop the entity_id and use a separate prefix so the two
    namespaces don't collide.
    """
    if issue_type in _PIPELINE_ISSUE_TYPES:
        return f"pipeline_health_{area_id}_{issue_type}"
    safe_entity = (entity_id or "").replace(".", "_")
    return f"sensor_health_{area_id}_{safe_entity}_{issue_type}"


class HealthMonitor:
    """Monitors sensor health for a single area.

    Runs during the hourly analysis pipeline, checking each entity for:
    - Stuck active/inactive states (binary sensors only)
    - Prolonged unavailability
    - Sensors that have never triggered

    Uses the stable area_id (HA area registry ID) for repair issue keys
    so issues survive area renames.
    """

    def __init__(self, area_name: str, area_id: str, hass: HomeAssistant) -> None:
        """Initialize health monitor for an area.

        Args:
            area_name: Human-readable area name (for log messages and translations)
            area_id: Stable HA area registry ID (for repair issue keys)
            hass: Home Assistant instance
        """
        self._area_name = area_name
        self._area_id = area_id
        self._hass = hass
        self._issues: list[HealthIssue] = []
        self._checked_count: int = 0
        self._last_check: datetime | None = None
        # Seed active issue IDs from the persisted issue registry so that
        # resolved issues can be cleaned up even after a restart.
        self._active_issue_ids: set[str] = self._load_existing_issue_ids()

    def _load_existing_issue_ids(self) -> set[str]:
        """Load existing repair issue IDs for this area from the HA issue registry.

        This ensures that issues created before a restart can be properly
        cleaned up if the underlying sensor or pipeline state has recovered
        while HA was down. Loads both sensor-scope (``sensor_health_*``)
        and pipeline-scope (``pipeline_health_*``) namespaces so neither
        family leaks orphaned issues across restarts.
        """
        prefixes = (
            f"sensor_health_{self._area_id}_",
            f"pipeline_health_{self._area_id}_",
        )
        try:
            registry = ir.async_get(self._hass)
            return {
                issue_id
                for (domain, issue_id) in registry.issues
                if domain == DOMAIN and issue_id.startswith(prefixes)
            }
        except (AttributeError, KeyError, TypeError):
            _LOGGER.debug(
                "Could not load existing health issues for area '%s'",
                self._area_name,
                exc_info=True,
            )
            return set()

    @property
    def issues(self) -> list[HealthIssue]:
        """Current health issues."""
        return self._issues

    @property
    def issue_count(self) -> int:
        """Number of current health issues."""
        return len(self._issues)

    @property
    def checked_count(self) -> int:
        """Number of entities that were actually checked (excludes virtual sensors)."""
        return self._checked_count

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
        checked = 0

        for entity in entities.values():
            if entity.entity_id in excluded:
                continue
            if entity.type.input_type in _EXCLUDED_TYPES:
                continue

            checked += 1
            issue = self._check_unavailable(entity, now)
            if issue:
                issues.append(issue)
                continue  # Skip other checks if unavailable

            issue = self._check_stuck_sensor(entity, now)
            if issue:
                issues.append(issue)
                continue

            # Never-triggered uses persisted last_updated, so it survives restarts
            issue = self._check_never_triggered(entity, now)
            if issue:
                issues.append(issue)

        self._checked_count = checked
        self._issues = issues
        self._update_repair_issues()
        return issues

    def cleanup(self) -> None:
        """Remove all repair issues for this area.

        Call when the area is unloaded, removed, or the integration shuts down
        to prevent orphaned repair entries.
        """
        for issue_id in self._active_issue_ids:
            ir.async_delete_issue(self._hass, DOMAIN, issue_id)

        if self._active_issue_ids:
            _LOGGER.debug(
                "Cleaned up %d health repair issue(s) for area '%s'",
                len(self._active_issue_ids),
                self._area_name,
            )
        self._active_issue_ids.clear()
        self._issues.clear()

    def check_pipeline_health(
        self,
        *,
        area_age_hours: float | None,
        has_global_prior: bool,
        cache_age_hours: float | None,
        last_analysis_duration_ms: float | None,
        correlation_failure_count: int,
        correlatable_entity_count: int,
    ) -> list[HealthIssue]:
        """Run pipeline-scope checks and merge results with sensor issues.

        This method is intentionally parameter-driven (rather than taking
        an ``Area`` object) so it stays easy to unit-test and decoupled
        from the Area's runtime surface. Callers in the analysis pipeline
        gather the inputs and pass them in; the monitor decides what to
        flag.

        Pipeline issues are merged into ``self._issues`` and persisted to
        the HA repair registry alongside any existing sensor-scope issues
        from the most recent ``check_health`` run.

        Returns the full list of current issues (sensor + pipeline).
        """
        now = dt_util.utcnow()
        # Carry over only sensor-scope issues from the most recent
        # ``check_health`` run; drop any prior pipeline issues so that a
        # check whose condition has cleared can't survive as a stale entry
        # if ``check_pipeline_health`` is called twice without an
        # intervening ``check_health``.
        new_issues: list[HealthIssue] = [
            issue
            for issue in self._issues
            if issue.issue_type not in _PIPELINE_ISSUE_TYPES
        ]

        issue = self._check_insufficient_priors(area_age_hours, has_global_prior, now)
        if issue:
            new_issues.append(issue)

        issue = self._check_stale_cache(cache_age_hours, area_age_hours, now)
        if issue:
            new_issues.append(issue)

        issue = self._check_slow_analysis(last_analysis_duration_ms, now)
        if issue:
            new_issues.append(issue)

        issue = self._check_correlation_failures(
            correlation_failure_count, correlatable_entity_count, now
        )
        if issue:
            new_issues.append(issue)

        self._issues = new_issues
        self._update_repair_issues()
        return new_issues

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
        """Check if a binary sensor has never been active.

        Uses the persisted entity.last_updated field (stored in DB, survives
        restarts). An entity that has never transitioned will have last_updated
        still at its initial value from when it was first created. If that
        timestamp is older than NEVER_TRIGGERED_THRESHOLD and the entity is
        currently inactive, it has likely never triggered.

        This intentionally overlaps with stuck_inactive detection for long
        durations but provides a distinct "misconfiguration" message for sensors
        that have literally never been active.
        """
        if entity.type.input_type not in _STUCK_CHECK_TYPES:
            return None

        if entity.last_updated is None:
            return None

        # If evidence is currently active, it has triggered
        if entity.evidence is True:
            return None

        # If previous_evidence was ever set to True (and is now False),
        # the sensor HAS triggered before - this is just normal inactive
        if entity.previous_evidence is True:
            return None

        # Check if last_updated is old enough to flag as never-triggered.
        # Entity.last_updated is persisted in the DB and only advances on
        # evidence transitions, so a sensor that has never triggered will
        # have last_updated close to its creation time.
        time_since_update = now - entity.last_updated
        if time_since_update < NEVER_TRIGGERED_THRESHOLD:
            return None

        days = time_since_update.total_seconds() / 86400
        return HealthIssue(
            entity_id=entity.entity_id,
            issue_type=HealthIssueType.NEVER_TRIGGERED,
            input_type=entity.type.input_type,
            since=entity.last_updated,
            duration_hours=round(days * 24, 1),
            details=(
                f"{entity.type.input_type.value} sensor has never been "
                f"active in {days:.0f} days of monitoring "
                f"(possible misconfiguration)"
            ),
        )

    def _check_insufficient_priors(
        self,
        area_age_hours: float | None,
        has_global_prior: bool,
        now: datetime,
    ) -> HealthIssue | None:
        """Flag areas past the warm-up period that still have no global prior.

        ``global_prior`` is None until enough occupied-vs-total time has
        accumulated for the prior calculator to produce a value. If the
        area has been running for longer than ``PRIORS_TRAINING_GRACE_PERIOD``
        and the global prior is still missing, the user should know — the
        integration is silently falling back to ``MIN_PRIOR`` for every
        Bayesian update.
        """
        if area_age_hours is None:
            return None
        if has_global_prior:
            return None
        grace_hours = PRIORS_TRAINING_GRACE_PERIOD.total_seconds() / 3600
        if area_age_hours < grace_hours:
            return None
        return HealthIssue(
            entity_id=None,
            issue_type=HealthIssueType.INSUFFICIENT_PRIORS,
            input_type=None,
            since=now,
            duration_hours=round(area_age_hours, 1),
            details=(
                f"Area has been running for {area_age_hours / 24:.0f} days "
                f"but no global prior has been learned yet "
                f"(grace period: {grace_hours / 24:.0f} days)"
            ),
        )

    def _check_stale_cache(
        self,
        cache_age_hours: float | None,
        area_age_hours: float | None,
        now: datetime,
    ) -> HealthIssue | None:
        """Flag stale occupied-intervals cache.

        Two cases:
        - Cache exists but is older than ``STALE_CACHE_THRESHOLD`` — the
          analysis pipeline that rebuilds it has stopped running for this
          area.
        - Cache is missing entirely (``None``) and the area is past its
          grace period — population step has never succeeded.
        """
        threshold_hours = STALE_CACHE_THRESHOLD.total_seconds() / 3600
        grace_hours = PRIORS_TRAINING_GRACE_PERIOD.total_seconds() / 3600

        if cache_age_hours is None:
            if area_age_hours is None or area_age_hours < grace_hours:
                return None
            return HealthIssue(
                entity_id=None,
                issue_type=HealthIssueType.STALE_INTERVALS_CACHE,
                input_type=None,
                since=now,
                duration_hours=round(area_age_hours, 1),
                details=(
                    "Occupied-intervals cache has never been populated "
                    f"({area_age_hours / 24:.0f} days since area was added)"
                ),
            )

        if cache_age_hours < threshold_hours:
            return None
        return HealthIssue(
            entity_id=None,
            issue_type=HealthIssueType.STALE_INTERVALS_CACHE,
            input_type=None,
            since=now,
            duration_hours=round(cache_age_hours, 1),
            details=(
                f"Occupied-intervals cache is {cache_age_hours:.0f}h old "
                f"(threshold: {threshold_hours:.0f}h)"
            ),
        )

    def _check_slow_analysis(
        self,
        last_analysis_duration_ms: float | None,
        now: datetime,
    ) -> HealthIssue | None:
        """Flag analysis runs that took longer than ``SLOW_ANALYSIS_THRESHOLD_MS``."""
        if last_analysis_duration_ms is None:
            return None
        if last_analysis_duration_ms < SLOW_ANALYSIS_THRESHOLD_MS:
            return None
        return HealthIssue(
            entity_id=None,
            issue_type=HealthIssueType.SLOW_ANALYSIS,
            input_type=None,
            since=now,
            duration_hours=round(last_analysis_duration_ms / 3_600_000, 4),
            details=(
                f"Last analysis cycle took {last_analysis_duration_ms / 1000:.1f}s "
                f"(threshold: {SLOW_ANALYSIS_THRESHOLD_MS / 1000:.0f}s)"
            ),
        )

    def _check_correlation_failures(
        self,
        failure_count: int,
        total_count: int,
        now: datetime,
    ) -> HealthIssue | None:
        """Flag when too many correlatable entities have failed correlation analysis."""
        if total_count <= 0:
            return None
        ratio = failure_count / total_count
        if ratio < CORRELATION_FAILURE_RATIO:
            return None
        return HealthIssue(
            entity_id=None,
            issue_type=HealthIssueType.CORRELATION_FAILURES,
            input_type=None,
            since=now,
            duration_hours=0.0,
            details=(
                f"{failure_count} of {total_count} correlatable sensors "
                f"({ratio * 100:.0f}%) failed correlation analysis"
            ),
        )

    def _update_repair_issues(self) -> None:
        """Create or delete HA repair issues based on current health state."""
        current_issue_ids: set[str] = set()

        # Severity mapping
        severity_map = {
            HealthIssueType.STUCK_ACTIVE: ir.IssueSeverity.ERROR,
            HealthIssueType.UNAVAILABLE: ir.IssueSeverity.ERROR,
            HealthIssueType.STUCK_INACTIVE: ir.IssueSeverity.WARNING,
            HealthIssueType.NEVER_TRIGGERED: ir.IssueSeverity.WARNING,
            HealthIssueType.INSUFFICIENT_PRIORS: ir.IssueSeverity.WARNING,
            HealthIssueType.STALE_INTERVALS_CACHE: ir.IssueSeverity.ERROR,
            HealthIssueType.SLOW_ANALYSIS: ir.IssueSeverity.WARNING,
            HealthIssueType.CORRELATION_FAILURES: ir.IssueSeverity.WARNING,
        }

        for issue in self._issues:
            repair_id = _issue_id(self._area_id, issue.entity_id, issue.issue_type)
            current_issue_ids.add(repair_id)

            if issue.issue_type in _PIPELINE_ISSUE_TYPES:
                translation_key = f"pipeline_health_{issue.issue_type}"
                placeholders = {
                    "area": self._area_name,
                    "duration": str(round(issue.duration_hours)),
                    "details": issue.details,
                }
            else:
                translation_key = f"sensor_health_{issue.issue_type}"
                placeholders = {
                    "area": self._area_name,
                    "entity_id": issue.entity_id or "",
                    "duration": str(round(issue.duration_hours)),
                    "sensor_type": (issue.input_type.value if issue.input_type else ""),
                }

            ir.async_create_issue(
                self._hass,
                DOMAIN,
                repair_id,
                is_fixable=False,
                severity=severity_map.get(issue.issue_type, ir.IssueSeverity.WARNING),
                translation_key=translation_key,
                translation_placeholders=placeholders,
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

        new_issue_ids = current_issue_ids - self._active_issue_ids
        if new_issue_ids:
            new_issues = [
                i
                for i in self._issues
                if _issue_id(self._area_id, i.entity_id, i.issue_type) in new_issue_ids
            ]
            _LOGGER.warning(
                "New sensor health issues in area '%s': %s",
                self._area_name,
                ", ".join(f"{i.entity_id} ({i.issue_type})" for i in new_issues),
            )

        self._active_issue_ids = current_issue_ids
