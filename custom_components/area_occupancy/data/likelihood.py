"""Likelihood calculations for individual sensors in Area Occupancy Detection."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from ..const import (
    DEFAULT_PROB_GIVEN_FALSE,
    DEFAULT_PROB_GIVEN_TRUE,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from ..utils import (
    TimeInterval,
    get_states_from_recorder,
    states_to_intervals,
    validate_datetime,
    validate_prior,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from .entity import Entity

_LOGGER = logging.getLogger(__name__)


@dataclass
class Likelihood:
    """Holds learned likelihood data for an individual sensor.

    This class stores conditional probabilities for individual sensors - these are
    the likelihoods used in Bayes' theorem.

    The Likelihood class is responsible for:
    - Storing learned sensor conditional probabilities (likelihoods)
    - P(sensor active | area occupied) and P(sensor active | area not occupied)
    - Providing sensor-specific probabilities for Bayesian calculations
    - Calculating its own values from historical data
    """

    prob_given_true: float  # P(sensor active | area occupied)
    prob_given_false: float  # P(sensor active | area not occupied)
    last_updated: datetime

    def __post_init__(self):
        """Validate properties after initialization."""
        self.prob_given_true = validate_prior(self.prob_given_true)
        self.prob_given_false = validate_prior(self.prob_given_false)
        self.last_updated = validate_datetime(self.last_updated)

    def to_dict(self) -> dict[str, Any]:
        """Convert likelihood to dictionary for storage."""
        return {
            "prob_given_true": self.prob_given_true,
            "prob_given_false": self.prob_given_false,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Likelihood":
        """Create likelihood from dictionary."""
        last_updated = validate_datetime(dt_util.parse_datetime(data["last_updated"]))

        return cls(
            prob_given_true=data["prob_given_true"],
            prob_given_false=data["prob_given_false"],
            last_updated=last_updated,
        )

    @classmethod
    def from_defaults(cls, entity_type) -> "Likelihood":
        """Create likelihood with default values from entity type."""
        return cls(
            prob_given_true=entity_type.prob_true,
            prob_given_false=entity_type.prob_false,
            last_updated=validate_datetime(None),
        )

    async def update(
        self,
        coordinator: "AreaOccupancyCoordinator",
        entity: "Entity",
        history_period: int | None = None,
    ) -> None:
        """Calculate and update this likelihood's values from historical data.

        Uses the area baseline prior to determine occupancy periods, then calculates
        sensor likelihoods against those periods. This breaks the circular dependency
        of using sensors to determine occupancy to calculate sensor likelihoods.

        Args:
            coordinator: The area occupancy coordinator
            entity: The entity to calculate likelihood for
            history_period: Number of days of history to analyze

        """
        hass: HomeAssistant = coordinator.hass
        entity_id = entity.entity_id
        entity_type = entity.type

        # Use provided history period or default from config
        effective_period = history_period or coordinator.config.history.period
        start_time = dt_util.utcnow() - timedelta(days=effective_period)
        end_time = dt_util.utcnow()

        _LOGGER.debug(
            "Calculating likelihood for entity: %s over %d days",
            entity_id,
            effective_period,
        )

        # Use default likelihoods as fallback values
        fallback_likelihood = self.from_defaults(entity_type)

        # Check if history analysis is disabled
        if (
            hasattr(coordinator.config, "history")
            and not coordinator.config.history.enabled
        ):
            self.prob_given_true = fallback_likelihood.prob_given_true
            self.prob_given_false = fallback_likelihood.prob_given_false
            self.last_updated = fallback_likelihood.last_updated
            return

        try:
            # Get area occupancy periods from the baseline prior
            total_history_seconds = (end_time - start_time).total_seconds()
            area_baseline_prior = coordinator.prior.area_baseline_prior

            _LOGGER.debug(
                "Area baseline prior for %s: %.3f (method: %s)",
                entity_id,
                area_baseline_prior,
                coordinator.prior.method_used,
            )

            occupied_seconds = coordinator.prior.time_active(effective_period)
            unoccupied_seconds = total_history_seconds - occupied_seconds

            if occupied_seconds <= 0 or unoccupied_seconds <= 0:
                _LOGGER.warning(
                    "Invalid occupancy periods for %s: occupied=%.1fs, unoccupied=%.1fs, area_prior=%.3f, total=%.1fs. Using defaults",
                    entity_id,
                    occupied_seconds,
                    unoccupied_seconds,
                    area_baseline_prior,
                    total_history_seconds,
                )
                self.prob_given_true = fallback_likelihood.prob_given_true
                self.prob_given_false = fallback_likelihood.prob_given_false
                self.last_updated = fallback_likelihood.last_updated
                return

            # Get sensor state history
            entity_states = await get_states_from_recorder(
                hass, entity_id, start_time, end_time
            )

            if not entity_states:
                _LOGGER.warning(
                    "No states found for likelihood calculation (%s). Using defaults",
                    entity_id,
                )
                self.prob_given_true = fallback_likelihood.prob_given_true
                self.prob_given_false = fallback_likelihood.prob_given_false
                self.last_updated = fallback_likelihood.last_updated
                return

            entity_state_objects = [
                state for state in entity_states if isinstance(state, State)
            ]

            if not entity_state_objects:
                _LOGGER.warning(
                    "No valid states found for likelihood calculation (%s). Using defaults",
                    entity_id,
                )
                self.prob_given_true = fallback_likelihood.prob_given_true
                self.prob_given_false = fallback_likelihood.prob_given_false
                self.last_updated = fallback_likelihood.last_updated
                return

            # Convert states to intervals
            entity_intervals = await states_to_intervals(
                entity_state_objects, start_time, end_time
            )

            # Calculate sensor active time during occupied and unoccupied periods
            # Use entity's active states or fall back to entity type's active states
            active_states = entity.active_states or entity.type.active_states or []

            if not active_states and entity.type.active_range is None:
                _LOGGER.warning(
                    "Entity %s has no active states or active range defined. Using defaults",
                    entity_id,
                )
                self.prob_given_true = fallback_likelihood.prob_given_true
                self.prob_given_false = fallback_likelihood.prob_given_false
                self.last_updated = fallback_likelihood.last_updated
                return

            sensor_active_during_occupied = self._calculate_sensor_active_time(
                entity_intervals,
                active_states,
                occupied_seconds,
                total_history_seconds,
                True,  # During occupied periods
                entity.type.active_range,  # Pass active range for numeric sensors
            )

            sensor_active_during_unoccupied = self._calculate_sensor_active_time(
                entity_intervals,
                active_states,
                unoccupied_seconds,
                total_history_seconds,
                False,  # During unoccupied periods
                entity.type.active_range,  # Pass active range for numeric sensors
            )

            # Calculate conditional probabilities
            prob_given_true = (
                sensor_active_during_occupied / occupied_seconds
                if occupied_seconds > 0
                else DEFAULT_PROB_GIVEN_TRUE
            )

            prob_given_false = (
                sensor_active_during_unoccupied / unoccupied_seconds
                if unoccupied_seconds > 0
                else DEFAULT_PROB_GIVEN_FALSE
            )

            # Apply bounds to prevent extreme values
            prob_given_true = max(
                MIN_PROBABILITY, min(prob_given_true, MAX_PROBABILITY)
            )
            prob_given_false = max(
                MIN_PROBABILITY, min(prob_given_false, MAX_PROBABILITY)
            )

            # Update this likelihood's values
            self.prob_given_true = prob_given_true
            self.prob_given_false = prob_given_false
            self.last_updated = validate_datetime(None)

            # Log the learned vs default values for debugging
            _LOGGER.debug(
                "Likelihood calculation for %s: "
                "occupied_time=%.1fs, unoccupied_time=%.1fs, "
                "sensor_active_occupied=%.1fs, sensor_active_unoccupied=%.1fs, "
                "prob_given_true=%.3f (default=%.3f), prob_given_false=%.3f (default=%.3f)",
                entity_id,
                occupied_seconds,
                unoccupied_seconds,
                sensor_active_during_occupied,
                sensor_active_during_unoccupied,
                prob_given_true,
                fallback_likelihood.prob_given_true,
                prob_given_false,
                fallback_likelihood.prob_given_false,
            )

        except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
            _LOGGER.warning(
                "Could not calculate likelihood for %s: %s. Using defaults",
                entity_id,
                err,
            )
            self.prob_given_true = fallback_likelihood.prob_given_true
            self.prob_given_false = fallback_likelihood.prob_given_false
            self.last_updated = fallback_likelihood.last_updated

    def _calculate_sensor_active_time(
        self,
        entity_intervals: list[TimeInterval],
        active_states: list[str],
        period_duration: float,
        total_duration: float,
        during_occupied: bool,
        active_range: tuple[float, float] | None = None,
    ) -> float:
        """Calculate how long the sensor was active during occupied or unoccupied periods.

        For now, this is a simplified calculation that assumes uniform distribution
        of occupied/unoccupied time throughout the history period. A more sophisticated
        approach would reconstruct the actual occupied/unoccupied intervals from the prior.

        Args:
            entity_intervals: Time intervals for the entity
            active_states: States considered active for the entity
            period_duration: Duration of the target period (occupied or unoccupied time)
            total_duration: Total duration of the history period
            during_occupied: If True, calculate for occupied periods; if False, for unoccupied
            active_range: Active range for numeric sensors

        Returns:
            Total seconds the sensor was active during the target period type

        """
        # Calculate total sensor active time
        total_sensor_active_time = 0
        for interval in entity_intervals:
            is_active = False

            if active_states:
                # Check if state matches active states
                is_active = interval["state"] in active_states
            elif active_range:
                # Check if numeric state is within active range
                try:
                    value = float(interval["state"])
                    min_val, max_val = active_range
                    is_active = min_val <= value <= max_val
                except (ValueError, TypeError):
                    is_active = False

            if is_active:
                total_sensor_active_time += (
                    interval["end"] - interval["start"]
                ).total_seconds()

        if total_sensor_active_time == 0:
            return 0.0

        # Simplified assumption: sensor activity is distributed proportionally
        # between occupied and unoccupied periods based on their durations
        if during_occupied:
            occupancy_ratio = period_duration / total_duration
        else:
            occupancy_ratio = period_duration / total_duration

        # Distribute sensor active time proportionally
        return total_sensor_active_time * occupancy_ratio
