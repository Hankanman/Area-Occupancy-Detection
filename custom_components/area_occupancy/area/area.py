"""Area class for individual device areas.

The Area class encapsulates all area-specific behavior and components,
including configuration, entities, prior probability, and purpose management.
This represents a single device area in the multi-area architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError

from ..data.analysis import start_likelihood_analysis, start_prior_analysis

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

    async def run_prior_analysis(self) -> None:
        """Run prior analysis for this area.

        This triggers the full prior analysis workflow:
        - Calculates global prior from sensor data
        - Writes global prior to database
        - Updates in-memory state
        - Calculates time priors and writes to database
        - Invalidates caches

        Raises:
            ValueError: If analysis fails due to data error
            SQLAlchemyError: If database operations fail
            Exception: For any other unexpected errors
        """
        _LOGGER.debug("Running prior analysis for area: %s", self.area_name)
        try:
            await start_prior_analysis(self.coordinator, self.area_name, self.prior)
        except (ValueError, TypeError, ZeroDivisionError):
            _LOGGER.exception("Prior analysis failed due to data error")
            raise
        except SQLAlchemyError:
            _LOGGER.exception("Prior analysis failed due to database error")
            raise
        except Exception:
            _LOGGER.exception("Prior analysis failed due to unexpected error")
            raise

    async def run_likelihood_analysis(self) -> None:
        """Run likelihood analysis for this area.

        This triggers the full likelihood analysis workflow:
        - Gets occupied intervals from Prior
        - Calculates likelihoods for all entities
        - Writes likelihoods to database
        - Updates Entity objects in memory

        Raises:
            ValueError: If analysis fails due to data error
            SQLAlchemyError: If database operations fail
            Exception: For any other unexpected errors
        """
        _LOGGER.debug("Running likelihood analysis for area: %s", self.area_name)
        try:
            await start_likelihood_analysis(
                self.coordinator, self.area_name, self.entities
            )
        except (ValueError, TypeError, ZeroDivisionError):
            _LOGGER.exception("Likelihood analysis failed due to data error")
            raise
        except SQLAlchemyError:
            _LOGGER.exception("Likelihood analysis failed due to database error")
            raise
        except Exception:
            _LOGGER.exception("Likelihood analysis failed due to unexpected error")
            raise

    async def async_cleanup(self) -> None:
        """Clean up the area's resources.

        This should be called when the area is being removed or the
        integration is shutting down.
        """
        await self.entities.cleanup()
        self.purpose.cleanup()
