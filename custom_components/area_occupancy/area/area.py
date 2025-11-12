"""Area class for individual device areas.

The Area class encapsulates all area-specific behavior and components,
including configuration, entities, prior probability, and purpose management.
This represents a single device area in the multi-area architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    async def async_cleanup(self) -> None:
        """Clean up the area's resources.

        This should be called when the area is being removed or the
        integration is shutting down.
        """
        await self.entities.cleanup()
        self.purpose.cleanup()
