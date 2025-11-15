"""Area class for individual device areas.

The Area class encapsulates all area-specific behavior and components,
including configuration, entities, prior probability, and purpose management.
This represents a single device area in the multi-area architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.helpers.device_registry import DeviceInfo

from ..const import (
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MIN_PROBABILITY,
)
from ..data.analysis import start_likelihood_analysis, start_prior_analysis
from ..data.entity_type import InputType
from ..utils import bayesian_probability

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
        _LOGGER.debug("Cleaning up area: %s", self.area_name)
        # Clear prior cache first to release cached data
        if self._prior is not None:
            self._prior.clear_cache()
        await self.entities.cleanup()
        self.purpose.cleanup()
        _LOGGER.debug("Area cleanup completed: %s", self.area_name)

    def device_info(self) -> DeviceInfo:
        """Return device info for this area.

        Returns:
            DeviceInfo for this area
        """
        # Use area_id for device identifier (stable even if area is renamed)
        # Fallback to area_name for legacy compatibility
        device_identifier = self.config.area_id or self.area_name
        return DeviceInfo(
            identifiers={(DOMAIN, device_identifier)},
            name=self.config.name,
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    def probability(self) -> float:
        """Calculate and return the current occupancy probability (0.0-1.0) for this area.

        Returns:
            Probability value (0.0-1.0)
        """
        entities = self.entities.entities
        if not entities:
            return MIN_PROBABILITY

        return bayesian_probability(
            entities=entities,
            prior=self.prior.value,
        )

    def type_probabilities(self) -> dict[str, float]:
        """Calculate and return the current occupancy probabilities for each entity type (0.0-1.0).

        Returns:
            Dictionary mapping input types to probabilities
        """
        entities = self.entities.entities
        if not entities:
            return {}

        return {
            InputType.MOTION: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MOTION),
                prior=self.prior.value,
            ),
            InputType.MEDIA: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.MEDIA),
                prior=self.prior.value,
            ),
            InputType.APPLIANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.APPLIANCE),
                prior=self.prior.value,
            ),
            InputType.DOOR: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.DOOR),
                prior=self.prior.value,
            ),
            InputType.WINDOW: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.WINDOW),
                prior=self.prior.value,
            ),
            InputType.ILLUMINANCE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.ILLUMINANCE
                ),
                prior=self.prior.value,
            ),
            InputType.HUMIDITY: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(InputType.HUMIDITY),
                prior=self.prior.value,
            ),
            InputType.TEMPERATURE: bayesian_probability(
                entities=self.entities.get_entities_by_input_type(
                    InputType.TEMPERATURE
                ),
                prior=self.prior.value,
            ),
        }

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

    def occupied(self) -> bool:
        """Return the current occupancy state (True/False) for this area.

        Returns:
            True if occupied, False otherwise
        """
        return self.probability() >= self.config.threshold

    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0) for this area.

        Returns:
            Threshold value (0.0-1.0)
        """
        return self.config.threshold
