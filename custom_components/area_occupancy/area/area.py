"""Area class for individual device areas.

The Area class encapsulates all area-specific behavior and components,
including configuration, entities, prior probability, and purpose management.
This represents a single device area in the multi-area architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.helpers.device_registry import DeviceInfo

from ..const import (
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from ..data.activity import DetectedActivity, detect_activity
from ..data.analysis import start_prior_analysis
from ..utils import (
    combined_probability as calc_combined,
    environmental_confidence as calc_env,
    presence_probability as calc_presence,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from ..data.config import AreaConfig
    from ..data.entity import EntityFactory, EntityManager
    from ..data.prior import Prior
    from ..data.purpose import Purpose
else:
    from ..data.config import AreaConfig
    from ..data.entity import EntityFactory, EntityManager
    from ..data.prior import Prior
    from ..data.purpose import Purpose

_LOGGER = logging.getLogger(__name__)


class AreaDeviceHandle:
    """Lightweight reference to an Area instance that survives reloads."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, area_name: str) -> None:
        """Initialize a new handle for the given area.

        Args:
            coordinator: Coordinator that manages the area.
            area_name: Name of the area represented by this handle.
        """
        self.coordinator = coordinator
        self.area_name = area_name
        self._area: Area | None = None

    def attach(self, area: Area | None) -> None:
        """Attach the latest Area implementation."""
        self._area = area

    @property
    def area(self) -> Area | None:
        """Return the currently attached Area."""
        return self._area

    def resolve(self) -> Area | None:
        """Return the current Area, refreshing from the coordinator."""
        area = self.coordinator.get_area(self.area_name)
        self._area = area
        return area

    def device_info(self) -> DeviceInfo | None:
        """Return DeviceInfo for the attached Area, if available."""
        area = self.resolve()
        if area is None:
            return None
        return area.device_info()


class Area:
    """Represents an individual device area in the multi-area architecture.

    The Area class encapsulates all area-specific components and behavior:
    - Configuration (sensors, weights, thresholds)
    - Entity management (tracking sensor states)
    - Prior probability calculation
    - Purpose management (area purpose and decay settings)

    This class is self-contained and handles all area-specific operations.
    """

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
        area_data: dict | None = None,
    ) -> None:
        """Initialize the Area instance.

        Args:
            coordinator: The coordinator instance managing this area
            area_name: Name/identifier for this area
            area_data: Optional area-specific configuration data

        Note:
            The area must be added to coordinator.areas BEFORE components
            are initialized. Components will be initialized lazily on first access
            to ensure the area exists in coordinator.areas first.
        """
        self.coordinator = coordinator
        self.area_name = area_name
        self.config = AreaConfig(coordinator, area_name=area_name, area_data=area_data)

        # Components will be initialized lazily after area is added to coordinator.areas
        # This avoids circular dependency issues during initialization
        self._factory: EntityFactory | None = None
        self._prior: Prior | None = None
        self._purpose: Purpose | None = None
        self._entities: EntityManager | None = None

        # Entity IDs for platform entities (set by platform modules)
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None
        self.sleep_entity_id: str | None = None

        # Activity detection cache
        self._activity_cache: DetectedActivity | None = None
        self._activity_cache_key: tuple[frozenset[str], float] | None = None

    @property
    def factory(self) -> EntityFactory:
        """Get or create the EntityFactory for this area."""
        if self._factory is None:
            self._factory = EntityFactory(self.coordinator, area_name=self.area_name)
        return self._factory

    @property
    def prior(self) -> Prior:
        """Get or create the Prior instance for this area."""
        if self._prior is None:
            self._prior = Prior(self.coordinator, area_name=self.area_name)
        return self._prior

    @property
    def purpose(self) -> Purpose:
        """Get or create the Purpose for this area."""
        if self._purpose is None:
            purpose_value = getattr(self.config, "purpose", None)
            self._purpose = Purpose(purpose=purpose_value)
        return self._purpose

    @property
    def entities(self) -> EntityManager:
        """Get or create the EntityManager for this area."""
        if self._entities is None:
            self._entities = EntityManager(self.coordinator, area_name=self.area_name)
        return self._entities

    async def run_prior_analysis(self) -> None:
        """Run prior analysis for this area."""
        await start_prior_analysis(self.coordinator, self.area_name, self.prior)

    async def async_cleanup(self) -> None:
        """Clean up the area's resources.

        This should be called when the area is being removed or the
        integration is shutting down.
        """
        # Clear prior cache first to release cached data
        if self._prior is not None:
            self._prior.clear_cache()
        await self.entities.cleanup()
        self.purpose.cleanup()

    def device_info(self) -> DeviceInfo:
        """Return device info for this area.

        Returns:
            DeviceInfo for this area
        """
        # Use area_id for device identifier (stable even if area is renamed)
        # Fallback to area_name if area_id is not available
        device_identifier = self.config.area_id or self.area_name
        return DeviceInfo(
            identifiers={(DOMAIN, device_identifier)},
            name=self.config.name,
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    def probability(self) -> float:
        """Calculate combined occupancy probability using sigmoid model.

        Combines presence probability (from strong binary indicators) with
        environmental confidence (from environmental sensors) using weighted
        averaging in logit space.

        Returns:
            Probability value (0.0-1.0)
        """
        entities = self.entities.entities
        if not entities:
            return MIN_PROBABILITY

        presence = self.presence_probability()
        env = self.environmental_confidence()

        return calc_combined(presence, env)

    def presence_probability(self) -> float:
        """Calculate presence probability from strong binary indicators.

        Uses motion, media, appliances, doors, windows, covers, and power
        sensors to determine presence likelihood.

        Returns:
            Probability value (0.0-1.0)
        """
        entities = self.entities.entities
        if not entities:
            return MIN_PROBABILITY

        correlations = self._get_entity_correlations()

        return calc_presence(
            entities, prior=self.prior.value, correlations=correlations
        )

    def environmental_confidence(self) -> float:
        """Calculate environmental support confidence.

        Uses temperature, humidity, illuminance, CO2, and other environmental
        sensors to determine how much the environment supports occupancy.

        Returns:
            Confidence value (0.0-1.0), where 0.5 is neutral
        """
        entities = self.entities.entities
        if not entities:
            return 0.5  # Neutral when no entities

        correlations = self._get_entity_correlations()

        return calc_env(entities, correlations=correlations)

    def _get_entity_correlations(self) -> dict[str, float]:
        """Get cached correlation strengths for this area.

        Returns correlations loaded asynchronously by the coordinator.
        No DB calls are made in this method.

        Returns:
            Dict of entity_id -> correlation strength. Empty dict if no data.
        """
        return self.coordinator.get_cached_correlations(self.area_name)

    def area_prior(self) -> float:
        """Get the area's baseline occupancy prior from historical data.

        This returns the pure P(area occupied) without any sensor weighting.

        Returns:
            Prior probability (0.0-1.0)
        """
        return self.prior.value

    def decay(self) -> float:
        """Calculate the current decay probability (0.0-1.0) for this area.

        Returns:
            Decay probability (0.0-1.0)
        """
        entities = self.entities.entities
        if not entities:
            return 1.0

        decay_sum = sum(entity.decay.decay_factor for entity in entities.values())
        return decay_sum / len(entities)

    def tick_decay(self) -> None:
        """Tick all entity decays to update their state.

        This method should be called periodically (e.g., by the decay timer)
        to transition decay states when factors drop below threshold.
        """
        for entity in self.entities.entities.values():
            entity.decay.tick()

    def occupied(self) -> bool:
        """Return the current occupancy state (True/False) for this area.

        Returns:
            True if occupied, False otherwise
        """
        return self.probability() >= self.config.threshold

    def detected_activity(self) -> DetectedActivity:
        """Detect the current activity in this area.

        Results are cached and recomputed only when the set of active
        entity IDs or the occupancy probability changes.

        Returns:
            DetectedActivity with activity_id, confidence, and matching indicators.
        """
        active_ids = frozenset(e.entity_id for e in self.entities.active_entities)
        prob = round(self.probability(), 4)
        cache_key = (active_ids, prob)

        if self._activity_cache_key == cache_key and self._activity_cache is not None:
            return self._activity_cache

        result = detect_activity(self)
        self._activity_cache = result
        self._activity_cache_key = cache_key
        return result

    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0) for this area.

        Returns:
            Threshold value (0.0-1.0)
        """
        return self.config.threshold
