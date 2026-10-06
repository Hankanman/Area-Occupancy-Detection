"""Area Occupancy Coordinator."""

from __future__ import annotations

# Standard library imports
from collections import deque
import contextlib
from datetime import datetime, timedelta
import logging
from typing import Any

# Home Assistant imports
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, STATE_ON
from homeassistant.core import CALLBACK_TYPE, Event, HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    floor_registry as fr,
)
from homeassistant.helpers.event import (
    async_track_point_in_time,
    async_track_state_change_event,
)
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util

# Local imports
from .area import AllAreas, Area, AreaDeviceHandle, FloorAreas
from .config_helpers import iter_area_subentries
from .const import (
    ACCURACY_HISTORY_DAYS,
    ACCURACY_STORE_KEY_PREFIX,
    ACCURACY_STORE_VERSION,
    ACCURACY_TICK_BUFFER_MAXLEN,
    ADJACENCY_TRANSITION_WINDOW_S,
    AWAY_SHADOW_STORE_KEY_PREFIX,
    AWAY_SHADOW_STORE_VERSION,
    CONF_AREA_ID,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_NAME,
    DOMAIN,
    FUSION_STORE_KEY_PREFIX,
    FUSION_STORE_VERSION,
    GROUND_TRUTH_STORE_KEY_PREFIX,
    GROUND_TRUTH_STORE_VERSION,
    LIKELIHOOD_SHADOW_STORE_KEY_PREFIX,
    LIKELIHOOD_SHADOW_STORE_VERSION,
    ONLINE_PRIOR_STORE_KEY_PREFIX,
    ONLINE_PRIOR_STORE_VERSION,
    SAVE_INTERVAL,
    TRANSITION_SHADOW_STORE_KEY_PREFIX,
    TRANSITION_SHADOW_STORE_VERSION,
)
from .data.adjacency import (
    BoostContribution,
    DecayModifierContribution,
    Trajectory,
    compute_adjacency_boost,
    compute_decay_modifier,
)
from .data.analysis import run_full_analysis
from .data.away_shadow import AwayShadow, AwayShadowState
from .data.config import IntegrationConfig
from .data.entity_type import (
    BINARY_INPUT_TYPES,
    GROUND_TRUTH_INPUT_TYPES,
    NUMERIC_INPUT_TYPES,
    PRESENCE_INPUT_TYPES,
    InputType,
)
from .data.fusion import FusionLearner, FusionState, FusionTick
from .data.ground_truth import LabelerState, LiveLabeler
from .data.likelihood_shadow import LikelihoodShadow, LikelihoodShadowState
from .data.metrics import AccuracyMetrics, TickSample, accuracy_summary
from .data.online_prior import OnlinePriorEstimator, OnlinePriorState
from .data.trajectory import TrajectoryTracker
from .data.transition_shadow import TransitionShadow, TransitionShadowState
from .db import AreaOccupancyDB
from .db.transitions import AdjacencySnapshot, load_adjacency_snapshot
from .time_utils import to_local
from .utils import clamp_probability, evidence_value, format_area_names, logit

_LOGGER = logging.getLogger(__name__)


def _ground_truth_present(area: Area, *, include_stuck: bool = False) -> bool:
    """Return whether any of the area's ground-truth sensors is active.

    Motion, media and sleep: the evidence ``db.queries.get_occupied_intervals``
    builds the learned priors and area transitions from.

    Args:
        area: The area.
        include_stuck: Count sensors flagged stuck active. The trajectory
            tracker needs this: flagging a sensor that is still on is not
            anyone leaving, and dropping it would record a departure.
    """
    return any(
        entity.evidence is True
        and entity.type.input_type in GROUND_TRUTH_INPUT_TYPES
        and (include_stuck or getattr(entity, "is_stuck", False) is not True)
        for entity in area.entities.entities.values()
    )


def _arrived_after(area: Area, departed_at: datetime | None) -> bool:
    """Return whether one of the area's presence sensors fired after a departure.

    The sensor must have become active within ``ADJACENCY_TRANSITION_WINDOW_S``
    of ``departed_at`` and still be active. That is how the transition
    learner recognises a move: the next area's evidence starts within the
    window after the previous area's evidence ends.
    """
    if departed_at is None:
        return False
    window_end = departed_at + timedelta(seconds=ADJACENCY_TRANSITION_WINDOW_S)
    return any(
        entity.evidence is True
        and entity.type.input_type in PRESENCE_INPUT_TYPES
        and entity.last_updated is not None
        and departed_at <= entity.last_updated <= window_end
        for entity in area.entities.entities.values()
    )


class OnlinePriorStore(Store[dict[str, dict]]):
    """Store for the online-prior shadow state, with v1->v2 passthrough.

    v2 (2026.9.1) added the weekly slot accumulators and divergence
    history to ``OnlinePriorState``. Nothing needs rewriting on upgrade:
    ``OnlinePriorState.from_dict`` tolerates a v1 payload (new fields
    default to empty), so the migration just hands the old data through —
    but the hook must exist, or ``Store.async_load`` raises
    ``NotImplementedError`` on the version mismatch and every user's
    scalar accumulators silently reset.
    """

    async def _async_migrate_func(
        self, old_version: int, old_data: dict[str, dict]
    ) -> dict[str, dict]:
        return old_data


class AreaOccupancyCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Manage fetching and combining data for area occupancy."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DEFAULT_NAME,
            update_interval=None,
            setup_method=self.setup,
            update_method=self.update,
        )
        self.config_entry = config_entry
        self.entry_id = config_entry.entry_id
        self.db = AreaOccupancyDB(self)

        # Integration-level configuration (global settings for entire integration)
        self.integration_config = IntegrationConfig(self, config_entry)

        # Multi-area architecture: dict[str, Area] keyed by area name
        self.areas: dict[str, Area] = {}
        self._area_handles: dict[str, AreaDeviceHandle] = {}

        # All Areas aggregator (lazy initialization)
        self._all_areas: AllAreas | None = None

        # Floor-based aggregators keyed by floor ID.
        self._floor_aggregators: dict[str, FloorAreas] = {}

        # Per-area state listeners (area_name -> callback)
        self._area_state_listeners: dict[str, CALLBACK_TYPE] = {}
        self._global_decay_timer: CALLBACK_TYPE | None = None
        self._analysis_timer: CALLBACK_TYPE | None = None
        self._save_timer: CALLBACK_TYPE | None = None
        self._setup_complete: bool = False
        self._analysis_running: bool = False
        self._cached_correlations: dict[str, dict[str, float]] = {}
        # Most recent full-analysis duration in milliseconds, written by
        # data.analysis.run_full_analysis at the end of each pipeline run.
        # Read by HealthMonitor.check_pipeline_health to flag slow analysis.
        self._last_analysis_duration_ms: float | None = None
        # Set to True by the EVENT_HOMEASSISTANT_STOP listener so the
        # in-flight analysis pipeline (and the sync correlation work it
        # dispatches to the executor) can bail at the next loop boundary
        # instead of riding through HA's "final writes shutdown stage".
        # Read by data.analysis.run_full_analysis and the per-area /
        # per-entity loops in db.correlation.
        self._stop_requested: bool = False
        self._stop_listener_remove: CALLBACK_TYPE | None = None

        # Adjacent-areas Phase 4 runtime state. The trajectory tracker
        # records departures across the household so the per-area
        # boost / decay-modifier paths can read a consistent snapshot.
        # ``_ground_truth_presence`` holds whether each area's ground-truth
        # sensors were active on the previous tick, which is how ``update``
        # spots a departure.
        # ``_lagged_probabilities`` holds the *previous* tick's
        # probability per area — captured at the start of ``update``
        # so the decay modifier and any future per-tick reader can't
        # feed back on this tick's own outputs.
        self._trajectory_tracker = TrajectoryTracker()
        self._ground_truth_presence: dict[str, bool] = {}
        self._lagged_probabilities: dict[str, float] = {}
        # In-memory adjacency index + transition counts. ``update`` runs on
        # every sensor state change, so it must not query SQLite; this is
        # reloaded by ``async_load_adjacency_snapshot`` whenever the
        # underlying tables can change (setup, options update, area purge,
        # hourly analysis).
        self._adjacency_snapshot = AdjacencySnapshot()
        # Adjacency boosts computed once per tick from the snapshot, then
        # read synchronously by ``Area.probability``.
        self._adjacency_boosts: dict[str, BoostContribution] = {}
        # Decay modifiers (Option 3a) precomputed alongside the boosts
        # and applied to each entity's ``Decay.modifier_factor``.
        self._adjacency_decay_modifiers: dict[str, DecayModifierContribution] = {}
        # Trust-score epic (#499), shadow mode: rolling per-area tick
        # observations scored hourly against motion-confirmed ground
        # truth. maxlen bounds memory to ~24h at the 10s decay cadence.
        # Nothing here feeds back into probability or thresholds.
        self._accuracy_ticks: dict[str, deque[TickSample]] = {}
        self._accuracy_metrics: dict[str, AccuracyMetrics] = {}
        # Spans the away-mode entity was on over the learning window,
        # refreshed before each prior recalculation; prior learning leaves
        # them out (#574/#584). Empty without an away-mode entity.
        self.away_spans: list[tuple[datetime, datetime]] = []
        # Daily summaries of the above, persisted so the metric's stability
        # can be judged across restarts (the tick window is memory-only).
        self._accuracy_history: dict[str, list[dict[str, Any]]] = {}
        self._accuracy_store: Store[dict[str, list]] = Store(
            hass,
            ACCURACY_STORE_VERSION,
            f"{ACCURACY_STORE_KEY_PREFIX}.{self.entry_id}",
        )
        # DB-retirement epic (#500), shadow mode: per-area online prior
        # estimators fed from live motion evidence each tick, persisted
        # via the HA storage helper, and diffed against the DB-computed
        # prior each analysis cycle. Never read by the probability path.
        self._online_priors: dict[str, OnlinePriorEstimator] = {}
        self._online_prior_store: Store[dict[str, dict]] = OnlinePriorStore(
            hass,
            ONLINE_PRIOR_STORE_VERSION,
            f"{ONLINE_PRIOR_STORE_KEY_PREFIX}.{self.entry_id}",
        )
        # Live ground-truth labellers (#603): label each tick by the same
        # rule the database applies after the fact, so learners don't need
        # the database to know the truth.
        self._labelers: dict[str, LiveLabeler] = {}
        self._ground_truth_store: Store[dict[str, dict]] = Store(
            hass,
            GROUND_TRUTH_STORE_VERSION,
            f"{GROUND_TRUTH_STORE_KEY_PREFIX}.{self.entry_id}",
        )
        # Shadow sensor likelihoods (#603): learned tick by tick against the
        # live label, compared daily with the database's values.
        self._likelihood_shadow: dict[str, LikelihoodShadow] = {}
        self._likelihood_shadow_store: Store[dict[str, dict]] = Store(
            hass,
            LIKELIHOOD_SHADOW_STORE_VERSION,
            f"{LIKELIHOOD_SHADOW_STORE_KEY_PREFIX}.{self.entry_id}",
        )
        # Live transition counts (#603), fed every area's latest label.
        self._latest_labels: dict[str, bool] = {}
        self._transition_shadow = TransitionShadow()
        self._transition_shadow_store: Store[dict] = Store(
            hass,
            TRANSITION_SHADOW_STORE_VERSION,
            f"{TRANSITION_SHADOW_STORE_KEY_PREFIX}.{self.entry_id}",
        )
        # Away-mode shadow evidence (#584): what lowering every room to the
        # away prior would have done. Never read by the probability path.
        self._away_shadow: dict[str, AwayShadow] = {}
        self._away_shadow_store: Store[dict[str, dict]] = Store(
            hass,
            AWAY_SHADOW_STORE_VERSION,
            f"{AWAY_SHADOW_STORE_KEY_PREFIX}.{self.entry_id}",
        )
        # Learned-fusion shadow state (#501): per-area training ticks
        # (sparse per-entity features) and per-area learners, persisted
        # like the online priors above. Never read by the probability path.
        self._fusion_ticks: dict[str, deque[FusionTick]] = {}
        self._fusion_learners: dict[str, FusionLearner] = {}
        self._fusion_store: Store[dict[str, dict]] = Store(
            hass,
            FUSION_STORE_VERSION,
            f"{FUSION_STORE_KEY_PREFIX}.{self.entry_id}",
        )

    async def async_init_database(self) -> None:
        """Initialize the database asynchronously to avoid blocking the event loop.

        This method should be called after coordinator creation but before
        async_config_entry_first_refresh().

        This ensures database tables exist and basic integrity before setup() loads data.
        The initialization is idempotent - calling it multiple times is safe even if the
        database was pre-initialized.
        """
        try:
            await self.hass.async_add_executor_job(self.db.initialize_database)
            _LOGGER.debug(
                "Database initialization completed for entry %s", self.entry_id
            )
        except Exception as err:
            _LOGGER.error(
                "Failed to initialize database for entry %s: %s", self.entry_id, err
            )
            raise

    async def async_refresh_correlations(self) -> None:
        """Refresh cached entity correlations for all areas from database.

        Loads correlations via executor to avoid blocking the event loop,
        then stores them on the coordinator for synchronous access during
        probability calculations.
        """
        if self.db is None:
            return

        for area_name in self.areas:
            try:
                from .db.correlation import get_entity_correlations  # noqa: PLC0415

                correlations = await self.hass.async_add_executor_job(
                    get_entity_correlations, self.db, area_name
                )
                self._cached_correlations[area_name] = correlations
            except ImportError:
                _LOGGER.debug(
                    "Failed to import correlation module for area %s",
                    area_name,
                    exc_info=True,
                )
                self._cached_correlations[area_name] = {}
            except (AttributeError, ValueError, OSError, RuntimeError):
                _LOGGER.debug(
                    "Failed to load entity correlations for area %s",
                    area_name,
                    exc_info=True,
                )
                self._cached_correlations[area_name] = {}

    def get_cached_correlations(self, area_name: str) -> dict[str, float]:
        """Return cached correlation strengths for the given area.

        Args:
            area_name: Name of the area to get correlations for

        Returns:
            Dict of entity_id -> correlation strength. Empty dict if no data.
        """
        return self._cached_correlations.get(area_name, {})

    def _load_areas_from_config(
        self, target_dict: dict[str, Area] | None = None
    ) -> list[str]:
        """Load areas from the config entry's area subentries.

        Reads one area configuration per ``SUBENTRY_TYPE_AREA`` subentry;
        the ``CONF_AREAS`` list it used to read is gone as of CONF_VERSION 19.

        Args:
            target_dict: Optional dict to load areas into. If None, loads into self.areas.

        Returns:
            List of area_ids that were not found in the Home Assistant registry
            (orphaned areas whose HA area was deleted).
        """
        # Use target_dict if provided, otherwise use self.areas
        areas_dict = target_dict if target_dict is not None else self.areas
        orphaned_area_ids: list[str] = []

        area_reg = ar.async_get(self.hass)

        for subentry_id, area_data in iter_area_subentries(self.config_entry):
            area_id = area_data.get(CONF_AREA_ID)

            if not area_id:
                _LOGGER.warning("Skipping area config without area ID")
                continue

            # Validate that area ID exists in Home Assistant.
            area_entry = area_reg.async_get_area(area_id)
            if not area_entry:
                _LOGGER.warning(
                    "Area ID '%s' not found in Home Assistant registry. "
                    "Area may have been deleted. Skipping.",
                    area_id,
                )
                orphaned_area_ids.append(area_id)
                continue

            # Resolve area name from ID.
            area_name = area_entry.name

            # Check for duplicate area names.
            if area_name in areas_dict:
                _LOGGER.warning("Duplicate area name %s, skipping", area_name)
                continue

            # Create Area for this area.
            areas_dict[area_name] = Area(
                coordinator=self,
                area_name=area_name,
                area_data=area_data,
                subentry_id=subentry_id,
            )
            self.get_area_handle(area_name).attach(areas_dict[area_name])
            _LOGGER.debug("Loaded area: %s (ID: %s)", area_name, area_id)

        return orphaned_area_ids

    def get_area_handle(self, area_name: str) -> AreaDeviceHandle:
        """Return a stable handle for the requested area."""
        handle = self._area_handles.get(area_name)
        if handle is None:
            handle = AreaDeviceHandle(self, area_name)
            self._area_handles[area_name] = handle
        return handle

    def get_area(self, area_name: str | None = None) -> Area | None:
        """Get area by name, or return first area if None.

        Args:
            area_name: Optional area name, None returns first area

        Returns:
            Area instance (always returns first area when area_name is None),
            or None if the specified area_name doesn't exist or no areas exist
        """
        if area_name is None:
            # Return first area (at least one area always exists in normal operation)
            # Handle empty case for tests/edge cases
            if not self.areas:
                return None
            return next(iter(self.areas.values()))
        return self.areas.get(area_name)

    def get_area_names(self) -> list[str]:
        """Get list of all configured area names."""
        return list(self.areas.keys())

    def find_area_for_entity(self, entity_id: str) -> str | None:
        """Find which area contains a specific entity.

        Args:
            entity_id: The entity ID to search for

        Returns:
            Area name if entity is found, None otherwise
        """
        for area_name, area in self.areas.items():
            try:
                area.entities.get_entity(entity_id)
            except ValueError:
                continue
            else:
                return area_name
        return None

    def get_all_areas(self) -> AllAreas:
        """Get or create the AllAreas aggregator instance.

        Returns:
            AllAreas instance for aggregating data across all areas
        """
        if self._all_areas is None:
            self._all_areas = AllAreas(self)
        return self._all_areas

    def _build_floor_aggregators(self) -> None:
        """Discover floors from configured areas and create FloorAreas aggregators."""
        area_reg = ar.async_get(self.hass)
        floor_reg = fr.async_get(self.hass)

        seen_floors: dict[str, str] = {}  # floor_id -> floor_name
        for area in self.areas.values():
            if not area.config.area_id:
                continue
            area_entry = area_reg.async_get_area(area.config.area_id)
            if (
                area_entry
                and area_entry.floor_id
                and area_entry.floor_id not in seen_floors
            ):
                floor_entry = floor_reg.async_get_floor(area_entry.floor_id)
                if floor_entry:
                    seen_floors[area_entry.floor_id] = floor_entry.name

        self._floor_aggregators = {
            floor_id: FloorAreas(self, floor_id, floor_name)
            for floor_id, floor_name in seen_floors.items()
        }

        if self._floor_aggregators:
            _LOGGER.debug(
                "Built %d floor aggregator(s): %s",
                len(self._floor_aggregators),
                ", ".join(f.floor_name for f in self._floor_aggregators.values()),
            )

    def get_floor_aggregators(self) -> dict[str, FloorAreas]:
        """Return floor-based aggregators keyed by floor_id."""
        return self._floor_aggregators

    @property
    def setup_complete(self) -> bool:
        """Return whether setup is complete."""
        return self._setup_complete

    @property
    def last_analysis_duration_ms(self) -> float | None:
        """Return the most recent full-analysis duration in milliseconds, if any."""
        return self._last_analysis_duration_ms

    @property
    def stop_requested(self) -> bool:
        """Return whether HA has signalled shutdown.

        The analysis pipeline and the sync correlation loops poll this so
        they can bail at the next loop boundary instead of riding through
        HA's "final writes shutdown stage" and tripping its
        ``Thread … is still running at shutdown`` warning.
        """
        return self._stop_requested

    def _on_homeassistant_stop(self, _event: Event) -> None:
        """Handle HA's ``EVENT_HOMEASSISTANT_STOP``.

        Setting the flag is enough for the analysis pipeline (which polls
        it inside long-running sync work). Cancelling timers here too
        prevents a fresh callback from kicking off DB / decay work after
        HA has already entered shutdown but before ``async_shutdown``
        runs. The ``_handle_*_timer`` handlers also poll the flag — that
        catches the race where a callback is already in-flight when this
        runs (we can't yank it from the loop, but we can stop it from
        rearming and from starting new executor work).
        """
        self._stop_requested = True
        if self._analysis_timer is not None:
            self._analysis_timer()
            self._analysis_timer = None
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None
        # ``_save_timer`` was missed in the original wiring. ``db.save_data``
        # runs in the executor pool, so a save callback in flight during
        # shutdown reproduces the "Thread is still running at shutdown"
        # warning the analysis pipeline used to trigger. Cancel here AND
        # have ``_handle_save_timer`` return early on the flag.
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

    # --- Public Methods ---
    def _validate_areas_configured(self) -> None:
        """Validate that at least one area is configured.

        Raises:
            HomeAssistantError: If no areas are configured
        """
        if not self.areas:
            raise HomeAssistantError("No areas configured")

    def _reconcile_entity_state(self) -> None:
        """Reconcile restored entity state with current HA sensor states.

        After loading from the database, entities may have stale decay and
        evidence state from the previous session.  This method:
        1. Ticks all decays so that any that have expired are resolved.
        2. Calls has_new_evidence() on every entity so that previous_evidence
           is reconciled against the live HA state.
        """
        for area in self.areas.values():
            # Resolve expired decays immediately
            area.tick_decay()
            # Reconcile previous_evidence with current HA state
            for entity in area.entities.entities.values():
                entity.has_new_evidence()

    async def setup(self) -> None:
        """Initialize the coordinator and its components (fast startup mode)."""
        try:
            # Load areas from config entry
            orphaned_area_ids = self._load_areas_from_config()

            # Clean up areas whose HA area was deleted (removes config, device,
            # entities, and database records so the user isn't left with orphans)
            if orphaned_area_ids:
                await self._cleanup_orphaned_areas(orphaned_area_ids)

            # Prune DB rows for areas no longer tracked by either area_name or
            # area_id (e.g. historical rows left behind by reconfiguration).
            try:
                await self._prune_fully_orphaned_db_areas()
            except (HomeAssistantError, OSError, RuntimeError):
                _LOGGER.debug(
                    "Pruning fully-orphaned DB areas failed at startup",
                    exc_info=True,
                )

            self._validate_areas_configured()

            # Restore online-prior shadow state (#500) for known areas
            stored_priors = await self._online_prior_store.async_load() or {}
            for area_name in self.areas:
                if area_name in stored_priors:
                    self._online_priors[area_name] = OnlinePriorEstimator(
                        OnlinePriorState.from_dict(stored_priors[area_name])
                    )

            # Restore the accuracy history (#499) for known areas
            stored_accuracy = await self._accuracy_store.async_load() or {}
            for area_name in self.areas:
                history = stored_accuracy.get(area_name)
                if isinstance(history, list):
                    self._accuracy_history[area_name] = history

            # Restore the live ground-truth labellers (#603) for known areas
            stored_labels = await self._ground_truth_store.async_load() or {}
            for area_name in self.areas:
                if area_name in stored_labels:
                    self._labelers[area_name] = LiveLabeler(
                        LabelerState.from_dict(stored_labels[area_name])
                    )

            # Restore the shadow likelihoods (#603) for known areas
            stored_likelihoods = await self._likelihood_shadow_store.async_load() or {}
            for area_name in self.areas:
                if area_name in stored_likelihoods:
                    self._likelihood_shadow[area_name] = LikelihoodShadow(
                        LikelihoodShadowState.from_dict(stored_likelihoods[area_name])
                    )

            # Restore the live transition counts (#603)
            stored_transitions = await self._transition_shadow_store.async_load()
            if stored_transitions:
                self._transition_shadow = TransitionShadow(
                    TransitionShadowState.from_dict(stored_transitions)
                )

            # Restore the away-mode shadow evidence (#584) for known areas
            stored_away = await self._away_shadow_store.async_load() or {}
            for area_name in self.areas:
                if area_name in stored_away:
                    self._away_shadow[area_name] = AwayShadow(
                        AwayShadowState.from_dict(stored_away[area_name])
                    )

            # Restore learned-fusion shadow state (#501) for known areas
            stored_fusion = await self._fusion_store.async_load() or {}
            for area_name in self.areas:
                if area_name in stored_fusion:
                    self._fusion_learners[area_name] = FusionLearner(
                        FusionState.from_dict(stored_fusion[area_name])
                    )

            _LOGGER.info(
                "Initializing Area Occupancy for %d area(s): %s",
                len(self.areas),
                ", ".join(self.areas.keys()),
            )

            # Initialize each area
            for area_name in self.areas:
                _LOGGER.debug("Initializing area: %s", area_name)

                # Load stored data from database for this area
                # Note: Database load will restore priors and entities per area
                # Database integrity checks are deferred to background (60s after startup)
                _LOGGER.debug(
                    "Loading entity data from database for area %s (deferring heavy operations)",
                    area_name,
                )

            # Load data from database
            await self.db.load_data()

            # Load cached correlations for probability calculations
            await self.async_refresh_correlations()

            # Reconcile restored entity state with current HA reality.
            # Database may have stale decay/evidence from before the reload.
            self._reconcile_entity_state()

            # Ensure areas and entities exist in database and persist configuration/state
            # This must happen before analysis runs so that get_occupied_intervals() can
            # properly JOIN Intervals with Entities table
            try:
                # Save both area and entity data for all areas in one operation
                await self.hass.async_add_executor_job(self.db.save_data)
            except (HomeAssistantError, OSError, RuntimeError) as e:
                _LOGGER.warning(
                    "Failed to save area and entity data, continuing setup: %s", e
                )

            # save_data syncs AreaRelationships from each area's configured
            # neighbours, so load the adjacency snapshot after it.
            await self.async_load_adjacency_snapshot()

            # Track entity state changes for all areas
            all_entity_ids = []
            for area in self.areas.values():
                all_entity_ids.extend(area.entities.entity_ids)

            # Remove duplicates
            all_entity_ids = list(set(all_entity_ids))
            await self.track_entity_state_changes(all_entity_ids)

            # Start timers only after everything is ready
            self._start_decay_timer()
            self._start_save_timer()
            # Analysis timer is async and runs in background
            await self._start_analysis_timer()

            # Wire up an early shutdown signal so an in-flight analysis
            # pipeline can bail before HA's "final writes shutdown stage"
            # rather than after, which is what triggered the
            # ``Task … was still running after final writes shutdown stage``
            # warning users were seeing on restart.
            if self._stop_listener_remove is None:
                self._stop_listener_remove = self.hass.bus.async_listen_once(
                    EVENT_HOMEASSISTANT_STOP, self._on_homeassistant_stop
                )

            # Build floor-based aggregators from area floor assignments
            self._build_floor_aggregators()

            # Mark setup as complete before initial refresh to prevent debouncer conflicts
            self._setup_complete = True

            # Log initialization summary
            total_entities = sum(
                len(area.entities.entities) for area in self.areas.values()
            )
            _LOGGER.info(
                "Successfully initialized %d area(s) with %d total entities",
                len(self.areas),
                total_entities,
            )
        except HomeAssistantError as err:
            _LOGGER.error("Failed to set up coordinator: %s", err)
            raise ConfigEntryNotReady(f"Failed to set up coordinator: {err}") from err
        except (OSError, RuntimeError) as err:
            _LOGGER.exception("Unexpected error during coordinator setup")
            raise ConfigEntryNotReady(f"Setup failed: {err}. Will retry.") from err

    async def update(self) -> dict[str, Any]:
        """Update and return the current coordinator data (in-memory only).

        Returns:
            Dictionary with area data keyed by area name
        """
        # Snapshot the previous tick's probabilities and occupancy
        # state BEFORE this tick recomputes, so the adjacency boost and
        # decay modifier read lagged values rather than feeding back on
        # the in-progress recompute.
        previous = self.data or {}
        self._lagged_probabilities = {
            name: float(entry.get("probability") or 0.0)
            for name, entry in previous.items()
        }

        now = dt_util.utcnow()
        # Record departures before this tick's boosts are computed. An area
        # is left when its ground-truth sensors go quiet, which is the event
        # the transition learner records. Its probability dropping below the
        # threshold is not a departure: that happens minutes later as the
        # decay tail runs out, and in an empty house it would make every
        # fading area look like someone moving on to its neighbours.
        for area_name, area in self.areas.items():
            present = _ground_truth_present(area, include_stuck=True)
            self._trajectory_tracker.observe(
                area_name,
                was_present=self._ground_truth_presence.get(area_name, False),
                is_present=present,
                now=now,
            )
            self._ground_truth_presence[area_name] = present
        # Compute adjacency boosts and decay modifiers from the in-memory
        # snapshot. Nothing in this method awaits, so a state-change
        # refresh completes without yielding to the event loop and
        # without waiting on the executor pool. ``Area.probability``
        # reads boosts via the cached dict; entity ``Decay`` instances
        # pick up the modifier through ``set_modifier_factor`` below.
        (
            self._adjacency_boosts,
            self._adjacency_decay_modifiers,
        ) = self._compute_adjacency_state(now)
        for area_name, modifier in self._adjacency_decay_modifiers.items():
            area = self.areas.get(area_name)
            if area is None:
                continue
            for entity in area.entities.entities.values():
                entity.decay.set_modifier_factor(modifier.decay_modifier)

        result = {}
        for area_name, area in self.areas.items():
            probability = area.probability()
            is_occupied = probability >= area.threshold()
            self._record_shadow_tick(area_name, area, now, probability, is_occupied)
            result[area_name] = {
                "probability": probability,
                "occupied": is_occupied,
                "threshold": area.threshold(),
                "prior": area.area_prior(),
                "decay": area.decay(),
                "last_updated": now,
            }
        return result

    def _record_shadow_tick(
        self,
        area_name: str,
        area: Area,
        now: datetime,
        probability: float,
        is_occupied: bool,
    ) -> None:
        """Record one shadow-mode accuracy/online-prior sample for an area.

        Read-only: does not touch ``self.data`` or notify listeners, so
        calling it outside a full ``update()`` (see
        ``_handle_decay_timer``) has no effect on entity state or the
        production refresh cadence.
        """
        truth = self._label_ground_truth(area_name, area, now)
        self._observe_likelihoods(area_name, area, now, truth)
        self._latest_labels[area_name] = truth
        self._transition_shadow.observe(
            now=now,
            labels=self._latest_labels,
            adjacency=self._adjacency_snapshot.adjacency_index,
        )
        self._accuracy_ticks.setdefault(
            area_name, deque(maxlen=ACCURACY_TICK_BUFFER_MAXLEN)
        ).append(
            TickSample(
                timestamp=now,
                probability=probability,
                occupied=is_occupied,
                truth=truth,
            )
        )
        # Match db.queries.get_occupied_intervals' ground-truth definition
        # (motion ∪ media ∪ sleep) rather than motion alone, so the online
        # numerator and the DB-computed prior it's being diffed against
        # measure the same thing. Motion's timeout extension isn't
        # replicated here — see module docstring's known approximations.
        presence_active = _ground_truth_present(area)
        self._online_priors.setdefault(area_name, OnlinePriorEstimator()).observe(
            motion_active=truth, now=now
        )
        # Away mode (#584, shadow): what the away prior would have decided.
        # The extra probability calculation only runs while away.
        away = self.household_away()
        away_shadow = (
            self._away_shadow.setdefault(area_name, AwayShadow())
            if away
            else self._away_shadow.get(area_name)
        )
        if away_shadow is not None:
            away_occupied = (
                area.away_adjusted_probability() >= area.config.threshold
                if away
                else is_occupied
            )
            away_shadow.observe(
                now=now,
                away=away,
                present=presence_active,
                live_occupied=is_occupied,
                away_occupied=away_occupied,
            )
        # Learned-fusion training row (#501): the bias and per-entity
        # feature products the live pipeline would use, minus the weight
        # being learned. evidence_value() is the same helper
        # sigmoid_probability uses, so the features cannot drift from the
        # live math. MOTION/SLEEP are excluded — the ground-truth labels
        # are derived from them (self-labelling); correlation analysis
        # excludes them for the same reason.
        correlations = self.get_cached_correlations(area_name)
        features: dict[str, float] = {}
        # Σ effective_weight · x, the part of the live logit being learned.
        learnable = 0.0
        for entity_id, entity in area.entities.entities.items():
            if entity.weight <= 0 or entity.type.input_type in (
                InputType.MOTION,
                InputType.SLEEP,
            ):
                continue
            evidence = evidence_value(entity)
            if evidence <= 0.0:
                continue
            correlation = correlations.get(entity_id, 1.0)
            strength_multiplier = getattr(entity.type, "strength_multiplier", 2.0)
            features[entity_id] = (
                evidence * correlation * entity.prob_given_true * strength_multiplier
            )
            learnable += (
                getattr(entity, "effective_weight", entity.weight) * features[entity_id]
            )
        self._fusion_ticks.setdefault(
            area_name, deque(maxlen=ACCURACY_TICK_BUFFER_MAXLEN)
        ).append(
            FusionTick(
                timestamp=now,
                bias=logit(area.prior.value),
                features=features,
                truth=truth,
                probability=probability,
                fixed_logit=logit(clamp_probability(probability)) - learnable,
            )
        )

    def _label_ground_truth(self, area_name: str, area: Area, now: datetime) -> bool:
        """This tick's ground-truth label for an area (data.ground_truth).

        Motion sensors are tracked one by one (for the PIR-or-presence
        call); media and sleep count as present while active. Sensors
        flagged stuck active don't count, as in the database path.
        """
        motion: dict[str, bool] = {}
        other_present = False
        for entity_id, entity in area.entities.entities.items():
            input_type = entity.type.input_type
            if input_type not in GROUND_TRUTH_INPUT_TYPES:
                continue
            active = (
                entity.evidence is True
                and getattr(entity, "is_stuck", False) is not True
            )
            if input_type == InputType.MOTION:
                motion[entity_id] = active
            elif active:
                other_present = True
        labeler = self._labelers.setdefault(area_name, LiveLabeler())
        return labeler.observe(
            now=now,
            motion=motion,
            other_present=other_present,
            timeout_seconds=area.config.sensors.motion_timeout,
        )

    def _observe_likelihoods(
        self, area_name: str, area: Area, now: datetime, truth: bool
    ) -> None:
        """Feed this tick into the area's shadow likelihoods (#603)."""
        binary: dict[str, bool] = {}
        numeric: dict[str, float] = {}
        for entity_id, entity in area.entities.entities.items():
            input_type = entity.type.input_type
            if input_type in BINARY_INPUT_TYPES:
                binary[entity_id] = entity.evidence is True
            elif input_type in NUMERIC_INPUT_TYPES:
                try:
                    numeric[entity_id] = float(entity.state)
                except (TypeError, ValueError):
                    continue
        self._likelihood_shadow.setdefault(area_name, LikelihoodShadow()).observe(
            now=now, label=truth, binary=binary, numeric=numeric
        )

    @property
    def transition_shadow(self) -> TransitionShadow:
        """The live transition counts (#603)."""
        return self._transition_shadow

    def db_transition_counts(self) -> dict[str, dict[int, dict[str, float]]]:
        """The database's transition counts, keyed like the live shadow."""
        return {
            f"{from_area}|{mid_area}": by_hour
            for (
                from_area,
                mid_area,
            ), by_hour in self._adjacency_snapshot.counts.items()
        }

    async def async_save_transition_shadow(self) -> None:
        """Persist the live transition counts via the HA storage helper."""
        await self._transition_shadow_store.async_save(
            self._transition_shadow.state.to_dict()
        )

    def likelihood_shadow_for(self, area_name: str) -> LikelihoodShadow | None:
        """Return the area's shadow likelihoods, if any were recorded."""
        return self._likelihood_shadow.get(area_name)

    async def async_save_likelihood_shadow(self) -> None:
        """Persist the shadow likelihoods via the HA storage helper."""
        await self._likelihood_shadow_store.async_save(
            {
                name: shadow.state.to_dict()
                for name, shadow in self._likelihood_shadow.items()
            }
        )

    def pulse_classification(self, area_name: str) -> dict[str, bool]:
        """The live labeller's PIR-like classification for an area's sensors.

        The database path uses it too, so both apply the motion timeout to
        the same sensors (#603).
        """
        labeler = self._labelers.get(area_name)
        return dict(labeler.state.pulse) if labeler is not None else {}

    async def async_save_ground_truth(self) -> None:
        """Persist the live labellers via the HA storage helper."""
        await self._ground_truth_store.async_save(
            {name: labeler.state.to_dict() for name, labeler in self._labelers.items()}
        )

    # --- Adjacent-areas (Phase 4) accessors ---
    @property
    def lagged_probabilities(self) -> dict[str, float]:
        """Return the previous tick's per-area probability snapshot.

        Read by the decay modifier so its silence-score is computed
        against last-tick occupancy of adjacent areas, not the values
        being recomputed in the current tick.
        """
        return self._lagged_probabilities

    def adjacency_boost_for(self, area_name: str) -> BoostContribution | None:
        """Return the cached adjacency boost for ``area_name`` this tick.

        Returns ``None`` when no boost has been computed this tick (e.g.
        called outside an active ``update``) or the area isn't in the
        cache yet. ``Area.probability`` calls this and falls through
        unmodified when ``None``.
        """
        return self._adjacency_boosts.get(area_name)

    def adjacency_decay_modifier_for(
        self, area_name: str
    ) -> DecayModifierContribution | None:
        """Return the cached decay modifier for ``area_name`` this tick."""
        return self._adjacency_decay_modifiers.get(area_name)

    def household_away(self) -> bool:
        """Whether the away-mode entity says the household is away (#584).

        Only an explicit away-mode entity reading ``on``. Person tracking,
        ``unknown``, ``unavailable`` or no entity all mean "not away", so
        nobody's readings change unless they set the entity.
        """
        entity_id = self.integration_config.away_mode_entity
        if not entity_id:
            return False
        state = self.hass.states.get(entity_id)
        return state is not None and state.state == STATE_ON

    async def async_refresh_away_spans(self) -> None:
        """Load when the away-mode entity was on, for prior learning (#584).

        Only an explicit away-mode entity counts; person tracking never
        shapes learning. Any state other than ``on`` (including
        ``unavailable``) counts as home, as it does for health alerts. A
        recorder failure leaves the spans empty, which is today's behaviour.
        """
        entity_id = self.integration_config.away_mode_entity
        if not entity_id:
            self.away_spans = []
            return
        # Lazy import: recorder history is only needed with an away entity.
        from homeassistant.components.recorder.history import (  # noqa: PLC0415
            get_significant_states,
        )
        from homeassistant.helpers.recorder import get_instance  # noqa: PLC0415

        end = dt_util.utcnow()
        start = end - timedelta(days=DEFAULT_LOOKBACK_DAYS)
        try:
            history = await get_instance(self.hass).async_add_executor_job(
                lambda: get_significant_states(
                    self.hass, start, end, [entity_id], minimal_response=False
                )
            )
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.debug("Could not read away-mode history: %s", err)
            self.away_spans = []
            return
        states = sorted(
            (history or {}).get(entity_id, []), key=lambda st: st.last_changed
        )
        spans: list[tuple[datetime, datetime]] = []
        for current, following in zip(states, [*states[1:], None], strict=True):
            if current.state != STATE_ON:
                continue
            span_end = following.last_changed if following is not None else end
            spans.append((current.last_changed, span_end))
        self.away_spans = spans

    # --- Trust score (#499, shadow mode) accessors ---
    def accuracy_samples_for(self, area_name: str) -> list[TickSample]:
        """Return the rolling tick observations for an area, oldest first."""
        return list(self._accuracy_ticks.get(area_name, ()))

    def accuracy_metrics_for(self, area_name: str) -> AccuracyMetrics | None:
        """Return the most recent shadow accuracy snapshot for an area.

        ``None`` until the hourly analysis pipeline has scored the area
        at least once (or when the area has no tick history yet).
        """
        return self._accuracy_metrics.get(area_name)

    def set_accuracy_metrics(self, area_name: str, metrics: AccuracyMetrics) -> None:
        """Cache an area's shadow accuracy snapshot (analysis pipeline).

        Also records it as the day's summary: each hourly run overwrites
        the day's entry, so a day keeps its latest 24-hour window.
        """
        self._accuracy_metrics[area_name] = metrics
        day = to_local(dt_util.utcnow()).date().isoformat()
        record = {"date": day, **accuracy_summary(metrics)}
        history = self._accuracy_history.setdefault(area_name, [])
        if history and history[-1].get("date") == day:
            history[-1] = record
        else:
            history.append(record)
        del history[:-ACCURACY_HISTORY_DAYS]

    def accuracy_history_for(self, area_name: str) -> list[dict[str, Any]]:
        """Return an area's persisted daily accuracy summaries, oldest first."""
        return list(self._accuracy_history.get(area_name, ()))

    async def async_save_accuracy_history(self) -> None:
        """Persist the daily accuracy summaries via the HA storage helper."""
        await self._accuracy_store.async_save(dict(self._accuracy_history))

    # --- Online prior (#500, shadow mode) accessors ---
    def online_prior_for(self, area_name: str) -> OnlinePriorEstimator | None:
        """Return the area's shadow online-prior estimator, if any ticks seen."""
        return self._online_priors.get(area_name)

    def reset_online_prior(self, area_name: str, signature: str) -> None:
        """Discard an area's online-prior statistics and start again.

        Args:
            area_name: The area.
            signature: Its current ground-truth fingerprint, recorded on
                the fresh statistics.
        """
        self._online_priors[area_name] = OnlinePriorEstimator(
            OnlinePriorState(config_signature=signature)
        )

    async def async_save_online_priors(self) -> None:
        """Persist online-prior shadow state via the HA storage helper."""
        await self._online_prior_store.async_save(
            {name: est.state.to_dict() for name, est in self._online_priors.items()}
        )

    def away_shadow_for(self, area_name: str) -> AwayShadow | None:
        """Return the area's away-mode shadow evidence, if any was recorded."""
        return self._away_shadow.get(area_name)

    async def async_save_away_shadow(self) -> None:
        """Persist the away-mode shadow evidence via the HA storage helper."""
        await self._away_shadow_store.async_save(
            {name: shadow.state.to_dict() for name, shadow in self._away_shadow.items()}
        )

    def fusion_learner_for(self, area_name: str) -> FusionLearner | None:
        """Return the area's shadow fusion learner, if any ticks/state exist."""
        return self._fusion_learners.get(area_name)

    def ensure_fusion_learner(self, area_name: str) -> FusionLearner:
        """Return the area's shadow fusion learner, creating it if absent."""
        return self._fusion_learners.setdefault(area_name, FusionLearner())

    def fusion_ticks_for(self, area_name: str) -> list[FusionTick]:
        """Return a snapshot of the area's fusion training ticks."""
        return list(self._fusion_ticks.get(area_name, ()))

    async def async_save_fusion_state(self) -> None:
        """Persist learned-fusion shadow state via the HA storage helper."""
        await self._fusion_store.async_save(
            {
                name: learner.state.to_dict()
                for name, learner in self._fusion_learners.items()
            }
        )

    def _compute_adjacency_state(
        self, now: datetime
    ) -> tuple[dict[str, BoostContribution], dict[str, DecayModifierContribution]]:
        """Compute boosts and decay modifiers for every area.

        Reads only the in-memory adjacency snapshot, so it is safe to
        call on the event loop from every refresh.
        """
        boosts: dict[str, BoostContribution] = {}
        modifiers: dict[str, DecayModifierContribution] = {}
        snapshot = self._adjacency_snapshot
        adjacency_index = snapshot.adjacency_index
        lagged = self._lagged_probabilities

        for area_name, area in self.areas.items():
            trajectory = self.trajectory_for(area_name, now=now)
            # Boost only an area whose own sensors confirm the move: one of
            # them fired after the departure, within the window the learner
            # uses for a transition. Without that check the boost acts on
            # areas nobody walked into, including ones still decaying from
            # earlier, and lifting those back over the threshold is what
            # made probabilities spike in an empty house.
            if trajectory.prev_area is not None and _arrived_after(
                area, trajectory.prev_end_time
            ):
                boosts[area_name] = compute_adjacency_boost(
                    target_area=area_name,
                    trajectory=trajectory,
                    lookup=snapshot.lookup,
                )
            # Decay modifier still fires even with no trajectory — the
            # 1-hop fallback ``P(target → neighbour)`` is meaningful when
            # adjacent neighbours are silent. ``base_half_life_seconds=1.0``
            # makes the diagnostic ``effective_half_life_seconds`` field
            # carry the unit-less modifier value; the actual per-entity
            # stretch is applied via ``Decay.set_modifier_factor`` after
            # this returns.
            if adjacency_index.get(area_name):
                modifiers[area_name] = compute_decay_modifier(
                    target_area=area_name,
                    adjacency_index=adjacency_index,
                    lagged_probabilities=lagged,
                    trajectory=trajectory,
                    lookup=snapshot.lookup,
                    base_half_life_seconds=1.0,
                )
        return boosts, modifiers

    async def async_load_adjacency_snapshot(self) -> None:
        """Reload the in-memory adjacency index and transition counts.

        Called after anything that can change ``AreaRelationships`` or
        ``AreaTransitions``. On a database error the previous snapshot is
        kept, so a transient failure doesn't drop every learned
        transition until the next reload.
        """
        snapshot = await self.hass.async_add_executor_job(
            load_adjacency_snapshot, self.db, self.entry_id
        )
        if snapshot is not None:
            self._adjacency_snapshot = snapshot

    def trajectory_for(self, target_area: str, *, now: datetime) -> Trajectory:
        """Return the trajectory describing recent ends excluding target.

        Hour-of-week is computed from ``now`` in the user's local
        timezone — matches the bucketing convention in
        ``db.transitions._hour_of_week``.
        """
        local = to_local(now)
        hour_of_week = local.weekday() * 24 + local.hour
        return self._trajectory_tracker.trajectory_for(
            target_area, hour_of_week=hour_of_week, now=now
        )

    async def async_shutdown(self) -> None:
        """Shutdown the coordinator.

        Cleanup order is important to prevent circular references and memory leaks:
        1. Cancel timers and listeners first
        2. Save final state
        3. Clean up areas (which clears their internal caches and references)
        4. Reset aggregators
        5. Dispose database engine (after all areas are cleaned up)
        6. Call parent shutdown
        """
        _LOGGER.info(
            "Starting coordinator shutdown for areas: %s",
            format_area_names(self),
        )

        # Mark stop requested so any analysis run that races shutdown bails
        # at the next loop boundary. Idempotent with the EVENT_HOMEASSISTANT_STOP
        # listener — config-entry unloads (e.g. options-flow reload) reach this
        # path without ever firing the bus event.
        self._stop_requested = True

        # Step 0: Drop the EVENT_HOMEASSISTANT_STOP listener if it's still
        # registered. ``async_listen_once`` self-removes when fired, so this
        # is the unload-without-shutdown case (options reload, integration
        # remove). Dropping it prevents a leaked listener after reload.
        if self._stop_listener_remove is not None:
            self._stop_listener_remove()
            self._stop_listener_remove = None

        # Step 1: Cancel periodic save timer before cleanup and perform final save
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Step 2: Perform final save to ensure no data loss
        try:
            await self.hass.async_add_executor_job(self.db.save_data)
            _LOGGER.info(
                "Final database save completed for areas: %s",
                format_area_names(self),
            )
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error(
                "Failed final save for areas: %s: %s",
                format_area_names(self),
                err,
            )

        # Step 2b: Persist online-prior shadow state so a restart doesn't
        # silently drop the occupied-seconds numerator while the period
        # denominator (anchored at first_observation) keeps growing —
        # previously this was only saved once per hour by the analysis
        # pipeline, biasing the online prior low across every restart.
        try:
            await self.async_save_online_priors()
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.warning(
                "Failed to save online-prior shadow state for areas: %s: %s",
                format_area_names(self),
                err,
            )
        try:
            await self.async_save_transition_shadow()
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.warning("Failed to save live transition counts: %s", err)
        try:
            await self.async_save_likelihood_shadow()
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.warning(
                "Failed to save shadow likelihoods for areas: %s: %s",
                format_area_names(self),
                err,
            )
        try:
            await self.async_save_ground_truth()
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.warning(
                "Failed to save ground-truth labellers for areas: %s: %s",
                format_area_names(self),
                err,
            )
        try:
            await self.async_save_away_shadow()
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.warning(
                "Failed to save away-mode shadow state for areas: %s: %s",
                format_area_names(self),
                err,
            )
        try:
            await self.async_save_accuracy_history()
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.warning(
                "Failed to save accuracy history for areas: %s: %s",
                format_area_names(self),
                err,
            )
        try:
            await self.async_save_fusion_state()
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.warning(
                "Failed to save learned-fusion shadow state for areas: %s: %s",
                format_area_names(self),
                err,
            )

        # Step 3: Cancel all area state listeners
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Step 4: Cancel prior update tracker
        if self._global_decay_timer is not None:
            self._global_decay_timer()
            self._global_decay_timer = None

        # Step 5: Clean up historical timer
        if self._analysis_timer is not None:
            self._analysis_timer()
            self._analysis_timer = None

        # Step 6: Clean up periodic save timer (defensive check)
        if self._save_timer is not None:
            self._save_timer()
            self._save_timer = None

        # Step 7: Clean up all areas (clears caches, entities, and internal references)
        for area in list(self.areas.values()):
            await area.async_cleanup()

        # Step 8: Reset AllAreas and floor aggregators to release references to old areas
        # This must be done after areas are cleaned up to break circular references
        self._all_areas = None
        self._floor_aggregators.clear()

        # Step 9: Dispose database engine to close all connections
        # This must be done after all areas are cleaned up to ensure no active sessions
        try:
            if hasattr(self.db, "engine") and self.db.engine is not None:
                self.db.engine.dispose(close=True)
        except (OSError, RuntimeError) as err:
            _LOGGER.warning("Error disposing database engine: %s", err)

        # Step 10: Clear areas dict to release all area references
        # This helps break any remaining circular references
        # Format area names before clearing (since we need them for logging)
        area_names_str = format_area_names(self)
        self.areas.clear()

        _LOGGER.info("Coordinator shutdown completed for areas: %s", area_names_str)
        await super().async_shutdown()

    async def _cleanup_removed_area(self, area_name: str, area: Area) -> None:
        """Clean up a removed area from registries and database.

        Handles cleanup of:
        - Area async cleanup
        - Entity registry entries
        - Device registry entries
        - Database records

        Args:
            area_name: Name of the area to clean up
            area: Area instance to clean up
        """
        # Clean up removed area
        await area.async_cleanup()
        self.get_area_handle(area_name).attach(None)

        # Look up device for this area (used for both entity and device removal)
        device_registry = dr.async_get(self.hass)
        device = device_registry.async_get_device_by_identifier(
            (DOMAIN, area.config.area_id), self.entry_id
        )

        # Remove entities from entity registry that belong to this device
        entity_registry = er.async_get(self.hass)
        entities_removed = 0
        target_device_id = device.id if device else None
        for entity_id, entity_entry in list(entity_registry.entities.items()):
            if (
                entity_entry.config_entry_id == self.entry_id
                and target_device_id is not None
                and entity_entry.device_id == target_device_id
            ):
                try:
                    entity_registry.async_remove(entity_id)
                    entities_removed += 1
                    _LOGGER.debug(
                        "Removed entity %s from registry for removed area %s",
                        entity_id,
                        area_name,
                    )
                except (ValueError, KeyError, AttributeError) as remove_err:
                    _LOGGER.warning(
                        "Failed to remove entity %s from registry: %s",
                        entity_id,
                        remove_err,
                    )

        if entities_removed > 0:
            _LOGGER.info(
                "Removed %d entities from registry for removed area %s",
                entities_removed,
                area_name,
            )

        # Remove device from device registry (reusing device object from above)
        if device:
            try:
                device_registry.async_remove_device(device.id)
                _LOGGER.info(
                    "Removed device %s from registry for removed area %s",
                    device.id,
                    area_name,
                )
            except (ValueError, KeyError, AttributeError) as remove_err:
                _LOGGER.warning(
                    "Failed to remove device %s from registry: %s",
                    device.id,
                    remove_err,
                )
        else:
            _LOGGER.debug(
                "No device found for removed area %s (area_id: %s)",
                area_name,
                area.config.area_id,
            )

        # Delete all database records for this area
        try:
            deleted_count = await self.hass.async_add_executor_job(
                self.db.delete_area_data, area_name
            )
            _LOGGER.debug(
                "Deleted %d database records for removed area %s",
                deleted_count,
                area_name,
            )
        except (HomeAssistantError, OSError, RuntimeError) as db_err:
            _LOGGER.error(
                "Failed to delete database records for removed area %s: %s",
                area_name,
                db_err,
            )

        _LOGGER.info("Cleaned up removed area: %s", area_name)

    async def _cleanup_orphaned_areas(self, orphaned_area_ids: list[str]) -> None:
        """Clean up areas whose Home Assistant area was deleted.

        When a user deletes an area from Home Assistant, the AOD config still
        references it. This method removes the orphaned config entries and
        cleans up associated devices, entities, and database records.

        Args:
            orphaned_area_ids: List of area_ids no longer in the HA registry.
        """
        if not orphaned_area_ids:
            return

        _LOGGER.info(
            "Cleaning up %d orphaned area(s) (HA area deleted): %s",
            len(orphaned_area_ids),
            ", ".join(orphaned_area_ids),
        )

        device_registry = dr.async_get(self.hass)
        entity_registry = er.async_get(self.hass)

        for area_id in orphaned_area_ids:
            # Look up the area_name from the database (needed for DB cleanup)
            area_name: str | None = None
            try:
                area_name = await self.hass.async_add_executor_job(
                    self._get_area_name_from_db, area_id
                )
            except (HomeAssistantError, OSError, RuntimeError) as err:
                _LOGGER.debug(
                    "Could not look up area_name for area_id '%s' from DB: %s",
                    area_id,
                    err,
                )

            # Clean up device and entities from registries
            device = device_registry.async_get_device_by_identifier(
                (DOMAIN, area_id), self.entry_id
            )
            if device:
                # Remove entities belonging to this device
                for entity_id, entity_entry in list(entity_registry.entities.items()):
                    if (
                        entity_entry.config_entry_id == self.entry_id
                        and entity_entry.device_id == device.id
                    ):
                        with contextlib.suppress(ValueError, KeyError, AttributeError):
                            entity_registry.async_remove(entity_id)

                # Remove the device itself
                try:
                    device_registry.async_remove_device(device.id)
                    _LOGGER.debug("Removed device for orphaned area_id '%s'", area_id)
                except (ValueError, KeyError, AttributeError) as err:
                    _LOGGER.warning(
                        "Failed to remove device for orphaned area_id '%s': %s",
                        area_id,
                        err,
                    )

            # Clean up database records if we found the area_name
            if area_name:
                try:
                    await self.hass.async_add_executor_job(
                        self.db.delete_area_data, area_name
                    )
                    _LOGGER.debug(
                        "Deleted database records for orphaned area '%s' (area_id: %s)",
                        area_name,
                        area_id,
                    )
                except (HomeAssistantError, OSError, RuntimeError) as err:
                    _LOGGER.warning(
                        "Failed to delete database records for orphaned area '%s': %s",
                        area_name,
                        err,
                    )

            _LOGGER.info(
                "Cleaned up orphaned area (area_id: %s, name: %s)",
                area_id,
                area_name or "unknown",
            )

        # Drop subentries for areas whose HA area was deleted, so the warning
        # does not repeat on every reload.
        orphaned_set = set(orphaned_area_ids)
        orphaned_subentry_ids = [
            subentry_id
            for subentry_id, area_data in iter_area_subentries(self.config_entry)
            if area_data.get(CONF_AREA_ID) in orphaned_set
        ]

        if orphaned_subentry_ids:
            try:
                for subentry_id in orphaned_subentry_ids:
                    self.hass.config_entries.async_remove_subentry(
                        self.config_entry, subentry_id
                    )
                _LOGGER.info(
                    "Removed %d orphaned area(s) from configuration",
                    len(orphaned_subentry_ids),
                )
            except (ValueError, KeyError, AttributeError) as err:
                _LOGGER.error(
                    "Failed to update config entry after orphan cleanup: %s", err
                )

    def _get_area_name_from_db(self, area_id: str) -> str | None:
        """Look up area_name from the database by area_id.

        Args:
            area_id: The Home Assistant area ID.

        Returns:
            The area_name if found, None otherwise.
        """
        try:
            with self.db.get_session() as session:
                result = (
                    session.query(self.db.Areas.area_name)
                    .filter_by(area_id=area_id)
                    .first()
                )
                return result[0] if result else None
        except (OSError, RuntimeError, HomeAssistantError):
            _LOGGER.exception(
                "Failed to look up area_name for area_id '%s' from DB", area_id
            )
            return None

    def _list_db_areas(self) -> list[tuple[str, str | None]]:
        """Return (area_name, area_id) pairs in the Areas table for this entry.

        Scoped to ``self.entry_id`` so a coordinator never considers rows
        owned by a different config entry when deciding what to prune.
        """
        try:
            with self.db.get_session() as session:
                rows = (
                    session.query(self.db.Areas.area_name, self.db.Areas.area_id)
                    .filter(self.db.Areas.entry_id == self.entry_id)
                    .all()
                )
                return [(row[0], row[1]) for row in rows]
        except (OSError, RuntimeError, HomeAssistantError):
            _LOGGER.exception(
                "Failed to list DB areas for entry '%s' during orphan prune",
                self.entry_id,
            )
            return []

    async def _prune_fully_orphaned_db_areas(self) -> None:
        """Delete DB rows for areas orphaned by both area_name and area_id.

        The existing orphan cleanup handles HA-area deletions by looking up the
        area_name from the DB via area_id. That path only runs when an area_id
        is no longer present in the HA registry. DB rows can still linger when
        a user removes an area from the integration and re-adds it under a
        different HA area (new area_id and area_name) — neither the original
        area_name nor the original area_id appears in the current config, so
        those rows are fully orphaned.

        Scoped to this coordinator's entry (see ``_list_db_areas``) and
        conservative: only deletes a DB area when BOTH its ``area_name`` and
        its ``area_id`` are absent from the currently loaded configuration.
        """
        configured_names = set(self.areas.keys())
        configured_ids: set[str] = {
            area.config.area_id for area in self.areas.values() if area.config.area_id
        }

        db_areas = await self.hass.async_add_executor_job(self._list_db_areas)
        if not db_areas:
            return

        stale_names = [
            area_name
            for area_name, area_id in db_areas
            if area_name not in configured_names
            and (area_id is None or area_id not in configured_ids)
        ]

        if not stale_names:
            return

        _LOGGER.info(
            "Pruning %d fully-orphaned area record(s) from database: %s",
            len(stale_names),
            ", ".join(stale_names),
        )

        for area_name in stale_names:
            try:
                await self.hass.async_add_executor_job(
                    self.db.delete_area_data, area_name
                )
            except (HomeAssistantError, OSError, RuntimeError) as err:
                _LOGGER.warning(
                    "Failed to prune orphaned DB area '%s': %s", area_name, err
                )

    async def async_update_options(self, options: dict[str, Any]) -> None:
        """Update coordinator options.

        Args:
            options: Updated options dict (may contain CONF_AREAS for multi-area updates)
        """
        if self.config_entry is None:
            raise HomeAssistantError("Cannot update options: config_entry is None")

        # Load new areas into a temporary dict first to avoid race condition
        # where self.areas is empty while platform entities are still active
        new_areas: dict[str, Area] = {}
        self._load_areas_from_config(
            target_dict=new_areas
        )  # orphans already cleaned at setup

        # Identify areas that will be removed by comparing old and new area names
        removed_area_names = set(self.areas.keys()) - set(new_areas.keys())
        if removed_area_names:
            _LOGGER.info(
                "Cleaning up %d removed area(s): %s",
                len(removed_area_names),
                ", ".join(removed_area_names),
            )

            for area_name in removed_area_names:
                area = self.areas.get(area_name)
                if area:
                    await self._cleanup_removed_area(area_name, area)

        # Cancel existing entity state listeners (will be recreated with new entity lists)
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Reset AllAreas aggregator to release references to old areas
        self._all_areas = None
        _LOGGER.debug("Reset AllAreas aggregator after area update")

        # Atomically replace self.areas with new_areas
        # This ensures self.areas is never empty when platform entities can access it
        self.areas = new_areas

        # Update area handles to point to new Area objects
        # This ensures platform entities can access the updated areas
        for area_name, area in self.areas.items():
            self.get_area_handle(area_name).attach(area)

        # Update each area's configuration
        # Clean up and recreate entities from new config first
        for area_name, area in self.areas.items():
            _LOGGER.info(
                "Configuration updated, re-initializing entities for area: %s",
                area_name,
            )

            # Clean up existing entity tracking and recreate from new config
            # This ensures entities match the updated configuration
            await area.entities.cleanup()

            # Area components are now initialized synchronously in __init__

        # Reload database data for new areas (restores priors, entity states)
        # This must happen AFTER entities are recreated from config so database
        # state can be applied to the correctly configured entities
        # This is critical to restore state after config changes without requiring a full reload
        await self.db.load_data()

        # Refresh cached correlations for new areas
        await self.async_refresh_correlations()

        # Reconcile restored entity state with current HA reality
        self._reconcile_entity_state()

        # Re-establish entity state tracking with new entity lists
        all_entity_ids = []
        for area in self.areas.values():
            all_entity_ids.extend(area.entities.entity_ids)

        # Remove duplicates
        all_entity_ids = list(set(all_entity_ids))
        await self.track_entity_state_changes(all_entity_ids)

        # Rebuild floor-based aggregators for updated areas
        self._build_floor_aggregators()

        # Force immediate save after configuration changes
        await self.hass.async_add_executor_job(self.db.save_data)

        # Neighbour lists may have changed; save_data has just re-synced them.
        await self.async_load_adjacency_snapshot()

        # Only request refresh if setup is complete to avoid debouncer conflicts
        if self.setup_complete:
            await self.async_request_refresh()

    # --- Entity State Tracking ---
    async def track_entity_state_changes(self, entity_ids: list[str]) -> None:
        """Track state changes for a list of entity_ids across all areas."""
        # Clean up existing listeners
        for listener in self._area_state_listeners.values():
            listener()
        self._area_state_listeners.clear()

        # Only create listener if we have entities to track
        if entity_ids:

            async def _refresh_on_state_change(event: Any) -> None:
                entity_id = event.data.get("entity_id")
                if not entity_id:
                    return

                # Find which area(s) this entity belongs to
                affected_areas = []
                health_enabled = self.integration_config.health_enabled
                for area_name, area in self.areas.items():
                    try:
                        entity = area.entities.get_entity(entity_id)
                    except ValueError:
                        # Entity doesn't belong to this area, skip it
                        continue
                    changed = entity.has_new_evidence()
                    # Take down repairs the change resolved (sensor back
                    # online, no longer stuck) now, not at the next hourly
                    # check. The virtual sensors are never checked.
                    if (
                        health_enabled
                        and entity_id not in (area.wasp_entity_id, area.sleep_entity_id)
                        and area.health_monitor.entity_changed(entity)
                    ):
                        changed = True
                    if changed:
                        affected_areas.append(area_name)

                # If entity affects any area and setup is complete, refresh
                if affected_areas and self.setup_complete:
                    await self.async_refresh()

            # Create single listener for all entities (more efficient than per-area listeners)
            listener = async_track_state_change_event(
                self.hass, entity_ids, _refresh_on_state_change
            )
            # Store listener (using a single key since we have one listener for all)
            self._area_state_listeners["_all"] = listener

    # --- Save Timer Handling ---
    def _start_save_timer(self) -> None:
        """Start the periodic database save timer (runs every 10 minutes)."""
        if self._save_timer is not None or not self.hass:
            return

        next_save = dt_util.utcnow() + timedelta(seconds=SAVE_INTERVAL)

        self._save_timer = async_track_point_in_time(
            self.hass, self._handle_save_timer, next_save
        )

    async def _handle_save_timer(self, _now: datetime) -> None:
        """Handle periodic save timer firing - save data and reschedule."""
        self._save_timer = None

        # Bail before kicking off executor work if shutdown has been
        # signalled. ``db.save_data`` runs in the executor pool and
        # would otherwise reproduce the "Thread is still running at
        # shutdown" warning. ``async_shutdown`` does its own final
        # save, so skipping this tick is safe.
        if self._stop_requested:
            return

        try:
            await self.hass.async_add_executor_job(self.db.save_data)
            _LOGGER.debug(
                "Periodic database save completed for areas: %s",
                format_area_names(self),
            )
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error(
                "Failed periodic save for areas: %s: %s",
                format_area_names(self),
                err,
            )

        # Reschedule the timer — but not if shutdown was signalled
        # while ``save_data`` was running in the executor. Otherwise we
        # leak a registered callback that ``async_shutdown`` will then
        # have to clean up.
        if self._stop_requested:
            return
        self._start_save_timer()

    # --- Decay Timer Handling ---
    def _start_decay_timer(self) -> None:
        """Start the global decay timer (always-on implementation)."""
        if self._global_decay_timer is not None or not self.hass:
            return

        next_update = dt_util.utcnow() + timedelta(
            seconds=self.integration_config.decay_interval
        )

        self._global_decay_timer = async_track_point_in_time(
            self.hass, self._handle_decay_timer, next_update
        )

    async def _handle_decay_timer(self, _now: datetime) -> None:
        """Handle decay timer firing - refresh coordinator and always reschedule."""
        self._global_decay_timer = None

        # Skip the tick + refresh if shutdown was signalled. The
        # work itself is in-process and quick (no executor), but
        # ``async_refresh`` can fan out to listeners that don't
        # expect to be called once the integration is unwinding.
        if self._stop_requested:
            return

        # Tick decay for all areas to update state (e.g., stop decay when factor reaches zero)
        # This must be done before refresh to ensure state transitions happen
        for area in self.areas.values():
            if area.config.decay.enabled:
                area.tick_decay()

        # Refresh the coordinator if decay is enabled for any area
        decay_enabled = any(area.config.decay.enabled for area in self.areas.values())
        if decay_enabled:
            await self.async_refresh()
        else:
            # Shadow-mode estimators (#499/#500) still need a steady tick
            # cadence even when no area has decay enabled — without this,
            # they'd only observe on evidence-triggered refreshes, whose
            # gaps routinely exceed the online-prior's downtime threshold
            # and starve its occupied-seconds numerator. Record ticks
            # directly rather than going through ``async_refresh()`` so
            # this doesn't change entity refresh cadence for users who
            # disabled decay to reduce state churn.
            now = dt_util.utcnow()
            for area_name, area in self.areas.items():
                probability = area.probability()
                is_occupied = probability >= area.threshold()
                self._record_shadow_tick(area_name, area, now, probability, is_occupied)

        # Reschedule the timer — but not if shutdown was signalled
        # during ``async_refresh``. Same rearm-leak avoidance as the
        # save and analysis timers.
        if self._stop_requested:
            return
        self._start_decay_timer()

    # --- Analysis Timer Handling ---
    async def _start_analysis_timer(self) -> None:
        """Start the historical data import timer.

        Uses async_at_started to ensure the analysis timer doesn't fire during
        Home Assistant bootstrap, which could cause startup timeouts.

        Note: No staggering needed with single-instance architecture.
        """
        if self._analysis_timer is not None or not self.hass:
            return

        async def _schedule_first_analysis(_: HomeAssistant) -> None:
            """Schedule the first analysis after HA is fully started."""
            if self._analysis_timer is not None:
                return  # Already scheduled by another call

            # First analysis: 5 minutes after HA is fully started
            # This ensures we don't block bootstrap
            next_update = dt_util.utcnow() + timedelta(minutes=5)

            _LOGGER.info(
                "Home Assistant started - scheduling analysis timer for areas: %s",
                format_area_names(self),
            )

            self._analysis_timer = async_track_point_in_time(
                self.hass, self.run_analysis, next_update
            )

        # Use async_at_started to defer analysis until after bootstrap completes
        # This prevents the analysis from blocking HA startup
        async_at_started(self.hass, _schedule_first_analysis)

    async def run_analysis(self, _now: datetime | None = None) -> None:
        """Handle the historical data import timer.

        Always runs analysis for all areas. Prevents concurrent runs by checking
        and setting the _analysis_running flag.

        Args:
            _now: Optional timestamp for the analysis run (used by timer)
        """
        if _now is None:
            _now = dt_util.utcnow()

        # Don't start a new pipeline if HA is shutting down. The
        # EVENT_HOMEASSISTANT_STOP listener already cancelled the timer,
        # but a previously-fired timer's callback can still land here
        # before the listener runs.
        if self._stop_requested:
            return

        # Prevent concurrent analysis runs
        if self._analysis_running:
            _LOGGER.debug("Analysis already running, skipping this trigger")
            # Cancel existing timer before rescheduling to avoid orphaned timers
            if self._analysis_timer is not None:
                self._analysis_timer()
                self._analysis_timer = None
            # Don't re-arm if shutdown has been signalled while another
            # analysis is in flight — the EVENT_HOMEASSISTANT_STOP listener
            # already cancelled the timer slot, and a callback we know
            # will hit the stop_requested guard is just a registry leak
            # until ``async_shutdown`` cleans it up.
            if self._stop_requested:
                return
            # Reschedule to try again later
            next_update = _now + timedelta(minutes=5)
            self._analysis_timer = async_track_point_in_time(
                self.hass, self.run_analysis, next_update
            )
            return

        # Clear timer reference now that we're actually running
        # (timer has fired and called this function)
        self._analysis_timer = None

        self._analysis_running = True
        _failed = False
        try:
            # Run the full analysis chain
            await run_full_analysis(self, _now)
        except (HomeAssistantError, OSError, RuntimeError) as err:
            _LOGGER.error("Failed to run historical analysis: %s", err)
            _failed = True
        except Exception:
            # Last-resort safety net — keeps the analysis timer alive
            _LOGGER.exception("Unexpected analysis error")
            _failed = True
        finally:
            self._analysis_running = False

        # All exceptions are handled above, so this runs on every path.
        # If shutdown was signalled DURING the await above, the
        # EVENT_HOMEASSISTANT_STOP listener has already cancelled
        # the timer slot and set ``_stop_requested``. Re-arming
        # here would register a fresh callback that immediately
        # no-ops in the guard at the top of this method — and
        # would only be cleaned up later by ``async_shutdown``.
        # Skip the re-arm entirely.
        if self._stop_requested:
            return
        # Always reschedule — retry sooner on failure
        if _failed:
            next_update = _now + timedelta(minutes=15)
        else:
            next_update = _now + timedelta(
                seconds=self.integration_config.analysis_interval
            )
        self._analysis_timer = async_track_point_in_time(
            self.hass, self.run_analysis, next_update
        )
