"""Area-specific data structure for multi-area coordinator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from .config import Config
    from .entity import EntityFactory, EntityManager
    from .prior import Prior
    from .purpose import PurposeManager
else:
    from .config import Config
    from .entity import EntityFactory, EntityManager
    from .prior import Prior
    from .purpose import PurposeManager


class AreaData:
    """Container for area-specific coordinator components."""

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
        area_data: dict | None = None,
    ):
        """Initialize area-specific data.

        Args:
            coordinator: The coordinator instance
            area_name: Name/identifier for this area
            area_data: Optional area-specific configuration data
        """
        self.coordinator = coordinator
        self.area_name = area_name
        self.config = Config(coordinator, area_name=area_name, area_data=area_data)
        # Components will be initialized lazily by coordinator after areas dict is populated
        # This avoids circular dependency issues
        self._factory: EntityFactory | None = None
        self._prior: Prior | None = None
        self._purpose: PurposeManager | None = None
        self._entities: EntityManager | None = None
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None

    @property
    def factory(self) -> EntityFactory:
        """Get or create the EntityFactory for this area."""
        if self._factory is None:
            # Temporarily set coordinator.config to this area's config
            # so components can access it correctly
            old_config = getattr(self.coordinator, "config", None)
            self.coordinator.config = self.config
            try:
                self._factory = EntityFactory(
                    self.coordinator, area_name=self.area_name
                )
            finally:
                if old_config is not None:
                    self.coordinator.config = old_config
        return self._factory

    @property
    def prior(self) -> Prior:
        """Get or create the Prior for this area."""
        if self._prior is None:
            old_config = getattr(self.coordinator, "config", None)
            self.coordinator.config = self.config
            try:
                self._prior = Prior(self.coordinator, area_name=self.area_name)
            finally:
                if old_config is not None:
                    self.coordinator.config = old_config
        return self._prior

    @property
    def purpose(self) -> PurposeManager:
        """Get or create the PurposeManager for this area."""
        if self._purpose is None:
            old_config = getattr(self.coordinator, "config", None)
            self.coordinator.config = self.config
            try:
                self._purpose = PurposeManager(
                    self.coordinator, area_name=self.area_name
                )
            finally:
                if old_config is not None:
                    self.coordinator.config = old_config
        return self._purpose

    @property
    def entities(self) -> EntityManager:
        """Get or create the EntityManager for this area."""
        if self._entities is None:
            old_config = getattr(self.coordinator, "config", None)
            self.coordinator.config = self.config
            try:
                self._entities = EntityManager(
                    self.coordinator, area_name=self.area_name
                )
            finally:
                if old_config is not None:
                    self.coordinator.config = old_config
        return self._entities
