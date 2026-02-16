"""Entity model."""

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from typing import TYPE_CHECKING, Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from ..const import (
    MAX_WEIGHT,
    MIN_WEIGHT,
    SLEEP_PRESENCE_HALF_LIFE,
    get_sensor_type_mapping,
)
from ..time_utils import to_utc
from ..utils import map_binary_state_to_semantic
from .decay import Decay
from .entity_type import (
    BINARY_INPUT_TYPES,
    DEFAULT_TYPES,
    AnalysisStatus,
    CorrelationType,
    EntityType,
    InputType,
)
from .purpose import get_default_decay_half_life
from .types import GaussianParams

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
    from ..db import AreaOccupancyDB as DB

_LOGGER = logging.getLogger(__name__)


@dataclass
class Entity:
    """Type for sensor state information."""

    # --- Core Data ---
    entity_id: str
    type: EntityType
    prob_given_true: float
    prob_given_false: float
    decay: Decay
    hass: HomeAssistant | None = None
    state_provider: Callable[[str], Any] | None = None
    last_updated: datetime | None = None
    previous_evidence: bool | None = None
    learned_active_range: tuple[float, float] | None = None
    learned_gaussian_params: GaussianParams | None = None
    analysis_error: str | None = None
    correlation_type: str | None = None

    def __post_init__(self) -> None:
        """Validate that either hass or state_provider is provided.

        Either hass or state_provider must be provided, and they are mutually
        exclusive. This ensures proper state retrieval behavior:

        - **HA-backed instances**: Provide `hass` to use Home Assistant's state
          registry for entity state retrieval. This is the standard usage pattern
          for entities managed within a Home Assistant integration.

        - **State-provider-only instances**: Provide `state_provider` (a callable
          that takes an entity_id and returns state) for testing or external state
          management scenarios where Home Assistant is not available.

        Raises:
            ValueError: If neither hass nor state_provider is provided, or if both
                are provided simultaneously.
        """
        if self.hass is None and self.state_provider is None:
            raise ValueError("Either hass or state_provider must be provided")
        if self.hass is not None and self.state_provider is not None:
            raise ValueError("Cannot provide both hass and state_provider")
        if self.last_updated is None:
            self.last_updated = dt_util.utcnow()

        # Store the static probability values in protected attributes
        # These are used as fallbacks when Gaussian calculation is not available
        self._prob_given_true = self.prob_given_true
        self._prob_given_false = self.prob_given_false

    def _calculate_gaussian_density(
        self, value: float, mean: float, std: float
    ) -> float:
        """Calculate Gaussian probability density function.

        Args:
            value: The current sensor value
            mean: The mean of the distribution
            std: The standard deviation of the distribution

        Returns:
            The probability density at the given value
        """
        if std <= 0:
            return 0.0
        exponent = -0.5 * ((value - mean) / std) ** 2
        return (1.0 / (std * math.sqrt(2 * math.pi))) * math.exp(exponent)

    def get_likelihoods(self) -> tuple[float, float]:
        """Get dynamic likelihoods based on current state.

        Motion sensors: Always use configured prob_given_true/prob_given_false.
        Binary sensors: Use static probabilities if learned from analysis, otherwise EntityType defaults.
        Numeric sensors: Use Gaussian PDF if available, otherwise EntityType defaults.

        Returns:
            Tuple of (prob_given_true, prob_given_false)
        """
        # Motion sensors: Always use configured values (not fallback)
        if self.type.input_type == InputType.MOTION:
            return (self.prob_given_true, self.prob_given_false)

        # Binary sensors: Use static probabilities if learned, otherwise EntityType defaults
        if self.type.input_type in BINARY_INPUT_TYPES:
            # If analysis has been run (not "not_analyzed"), use learned probabilities
            # Otherwise fall back to EntityType defaults
            if self.analysis_error != AnalysisStatus.NOT_ANALYZED:
                # Analysis has been run - use stored probabilities
                return (self.prob_given_true, self.prob_given_false)
            # Not analyzed yet - use EntityType defaults
            return (self.type.prob_given_true, self.type.prob_given_false)

        # Numeric sensors: Use Gaussian PDF if available
        if self.learned_gaussian_params:
            # Validate mean values first
            mean_occupied = self.learned_gaussian_params.mean_occupied
            mean_unoccupied = self.learned_gaussian_params.mean_unoccupied

            # Check for NaN/inf in mean values
            if math.isnan(mean_occupied) or math.isinf(mean_occupied):
                _LOGGER.warning(
                    "Invalid mean_occupied value (NaN/inf) for %s, using EntityType defaults",
                    self.entity_id,
                )
                return (self.type.prob_given_true, self.type.prob_given_false)

            if math.isnan(mean_unoccupied) or math.isinf(mean_unoccupied):
                _LOGGER.warning(
                    "Invalid mean_unoccupied value (NaN/inf) for %s, using EntityType defaults",
                    self.entity_id,
                )
                return (self.type.prob_given_true, self.type.prob_given_false)

            # Try to get current state value, fall back to mean if unavailable
            val = None
            if self.state is not None:
                with suppress(ValueError, TypeError):
                    # State is not a valid number (e.g., "unknown", "unavailable")
                    val = float(self.state)
                    # Validate state value for NaN/inf
                    if math.isnan(val) or math.isinf(val):
                        _LOGGER.warning(
                            "Invalid state value (NaN/inf) for %s, using mean of means",
                            self.entity_id,
                        )
                        val = None

            # If state is unavailable or invalid, use mean of occupied and unoccupied as representative value
            if val is None:
                # Use average of means as representative value (means are already validated)
                val = (mean_occupied + mean_unoccupied) / 2.0

            try:
                # Clamp std dev to minimum 0.05 to prevent numerical issues (division by zero)
                # We do NOT clamp maximum as numeric sensors (e.g. CO2) can have large variance
                std_occupied = max(0.05, self.learned_gaussian_params.std_occupied)
                std_unoccupied = max(0.05, self.learned_gaussian_params.std_unoccupied)

                # Calculate density for occupied state
                p_true = self._calculate_gaussian_density(
                    val,
                    mean_occupied,
                    std_occupied,
                )

                # Calculate density for unoccupied state
                p_false = self._calculate_gaussian_density(
                    val,
                    mean_unoccupied,
                    std_unoccupied,
                )

                # Validate calculated densities for NaN/inf
                if math.isnan(p_true) or math.isinf(p_true):
                    _LOGGER.warning(
                        "Invalid p_true density (NaN/inf) for %s, using EntityType defaults",
                        self.entity_id,
                    )
                    return (self.type.prob_given_true, self.type.prob_given_false)

                if math.isnan(p_false) or math.isinf(p_false):
                    _LOGGER.warning(
                        "Invalid p_false density (NaN/inf) for %s, using EntityType defaults",
                        self.entity_id,
                    )
                    return (self.type.prob_given_true, self.type.prob_given_false)

            except (ValueError, TypeError) as err:
                _LOGGER.warning(
                    f"Gaussian likelihood calculation failed for {self.entity_id}: {err}",  # noqa: G004
                )
            else:
                return (p_true, p_false)

        # Fallback: Use EntityType defaults (not stored prob_given_true/false)
        return (self.type.prob_given_true, self.type.prob_given_false)

    @property
    def is_continuous_likelihood(self) -> bool:
        """Check if entity uses continuous likelihood calculation."""
        return self.learned_gaussian_params is not None

    @property
    def name(self) -> str | None:
        """Get the entity name from Home Assistant state or state provider."""
        if self.state_provider:
            state_obj = self.state_provider(self.entity_id)
            if state_obj and hasattr(state_obj, "name"):
                return state_obj.name
            return None
        ha_state = self.hass.states.get(self.entity_id)
        return ha_state.name if ha_state else None

    @property
    def available(self) -> bool:
        """Get the entity availability."""
        return self.state is not None

    @property
    def state(self) -> str | float | bool | None:
        """Get the entity state from Home Assistant or state provider."""
        if self.state_provider:
            state_obj = self.state_provider(self.entity_id)
            if state_obj is None:
                return None
            # Handle both object with .state attribute and direct value
            if hasattr(state_obj, "state"):
                state_value = state_obj.state
            else:
                state_value = state_obj
        else:
            ha_state = self.hass.states.get(self.entity_id)
            if ha_state is None:
                return None
            state_value = ha_state.state

        # Check if state is valid
        if state_value in [
            "unknown",
            "unavailable",
            None,
            "",
            "NaN",
        ]:
            return None
        return state_value

    @property
    def weight(self) -> float:
        """Get the entity weight."""
        return self.type.weight

    @property
    def evidence(self) -> bool | None:
        """Determine if entity is active.

        Returns:
            bool | None: True if entity is active, False if inactive, None if state unknown

        """
        if self.state is None:
            return None

        if self.active_states:
            mapped_state = map_binary_state_to_semantic(
                str(self.state), self.active_states
            )
            return mapped_state in self.active_states
        if self.active_range:
            min_val, max_val = self.active_range
            try:
                return min_val <= float(self.state) <= max_val
            except (ValueError, TypeError):
                return False

        return None

    @property
    def active(self) -> bool:
        """Get the entity active status."""
        return bool(self.evidence) or self.decay.is_decaying

    @property
    def active_states(self) -> list[str] | None:
        """Get the active states for the entity."""
        return self.type.active_states

    @property
    def active_range(self) -> tuple[float, float] | None:
        """Get the active range for the entity."""
        if self.learned_active_range is not None:
            return self.learned_active_range
        return self.type.active_range

    @property
    def decay_factor(self) -> float:
        """Get decay factor that considers current evidence state.

        Returns 1.0 if evidence is currently True, otherwise returns the normal decay factor.
        This prevents inconsistent states where evidence is True but decay is being applied.
        """
        if self.evidence is True:
            return 1.0
        return self.decay.decay_factor

    def update_correlation(self, correlation_data: dict[str, Any]) -> None:
        """Update the learned active range based on correlation data."""
        if not correlation_data:
            return

        # Get correlation details
        confidence = correlation_data.get("confidence", 0.0)
        correlation_type = correlation_data.get("correlation_type")
        mean_unoccupied = correlation_data.get("mean_value_when_unoccupied")
        std_unoccupied = correlation_data.get("std_dev_when_unoccupied")

        # Store analysis error (None means no error, analysis succeeded)
        # This will be None if analysis succeeded, or a string error reason if it failed
        self.analysis_error = correlation_data.get("analysis_error")
        # Store correlation type for reporting
        self.correlation_type = correlation_type

        # Get occupied stats
        mean_occupied = correlation_data.get("mean_value_when_occupied")
        std_occupied = correlation_data.get("std_dev_when_occupied")

        # If any required data is missing or type is none, reset learned range
        if (
            mean_unoccupied is None
            or std_unoccupied is None
            or correlation_type == CorrelationType.NONE
        ):
            self.learned_active_range = None
            self.learned_gaussian_params = None
            return

        # Store Gaussian parameters if available
        if mean_occupied is not None and std_occupied is not None:
            self.learned_gaussian_params = GaussianParams(
                mean_occupied=mean_occupied,
                std_occupied=std_occupied,
                mean_unoccupied=mean_unoccupied,
                std_unoccupied=std_unoccupied,
            )
            # When using Gaussian params, we don't update static likelihoods
            # because they are calculated dynamically
        else:
            self.learned_gaussian_params = None

        # Only set learned_active_range for numeric sensors
        if self.type.input_type in BINARY_INPUT_TYPES:
            # Binary sensors: don't set learned_active_range
            self.learned_active_range = None
        else:
            # Numeric sensors: calculate thresholds (mean ± 2σ)
            k_factor = 2.0

            if correlation_type in (
                CorrelationType.STRONG_POSITIVE,
                CorrelationType.POSITIVE,
            ):
                # Active > mean_unoccupied + K*std_unoccupied
                # Same logic for both strong and weak positive correlations
                lower_bound = mean_unoccupied + (k_factor * std_unoccupied)

                # Try to determine upper bound from occupied stats
                if mean_occupied is not None and std_occupied is not None:
                    # Cap at mean_occupied + K*std_occupied
                    upper_bound = mean_occupied + (k_factor * std_occupied)
                    # Ensure logical consistency (upper > lower)
                    if upper_bound <= lower_bound:
                        # Fallback to open-ended if stats overlap significantly
                        upper_bound = float("inf")
                else:
                    upper_bound = float("inf")

                self.learned_active_range = (lower_bound, upper_bound)

            elif correlation_type in (
                CorrelationType.STRONG_NEGATIVE,
                CorrelationType.NEGATIVE,
            ):
                # Active < mean_unoccupied - K*std_unoccupied
                # Same logic for both strong and weak negative correlations
                upper_bound = mean_unoccupied - (k_factor * std_unoccupied)

                # Try to determine lower bound from occupied stats
                if mean_occupied is not None and std_occupied is not None:
                    # Floor at mean_occupied - K*std_occupied
                    lower_bound = mean_occupied - (k_factor * std_occupied)
                    # Ensure logical consistency (lower < upper)
                    if lower_bound >= upper_bound:
                        # Fallback to open-ended if stats overlap significantly
                        lower_bound = float("-inf")
                else:
                    lower_bound = float("-inf")

                self.learned_active_range = (lower_bound, upper_bound)
            else:
                self.learned_active_range = None

        _LOGGER.debug(
            "Updated learned active range for %s: %s (type=%s, conf=%.2f, p_true=%.2f, p_false=%.2f)",
            self.entity_id,
            self.learned_active_range,
            correlation_type,
            confidence,
            self.prob_given_true,
            self.prob_given_false,
        )

    def update_binary_likelihoods(self, likelihood_data: dict[str, Any]) -> None:
        """Update binary sensor likelihoods from duration-based analysis.

        Binary sensors use static probabilities calculated from interval overlap
        durations, not Gaussian PDFs.

        Args:
            likelihood_data: Dictionary with prob_given_true, prob_given_false,
                and analysis_error
        """
        if not likelihood_data:
            return

        # Store analysis error (None means no error, analysis succeeded)
        self.analysis_error = likelihood_data.get("analysis_error")
        # Store correlation type for reporting (binary likelihoods have type "binary_likelihood")
        self.correlation_type = likelihood_data.get("correlation_type")

        # Get probability values
        prob_given_true = likelihood_data.get("prob_given_true")
        prob_given_false = likelihood_data.get("prob_given_false")

        # If analysis failed or data is missing, reset to defaults
        if (
            self.analysis_error is not None
            or prob_given_true is None
            or prob_given_false is None
        ):
            # Reset to EntityType defaults
            self.prob_given_true = self.type.prob_given_true
            self.prob_given_false = self.type.prob_given_false
            self.learned_gaussian_params = None
            self.learned_active_range = None
            _LOGGER.debug(
                "Binary likelihood analysis failed for %s: %s, using defaults",
                self.entity_id,
                self.analysis_error or "missing data",
            )
            return

        # Update static probabilities
        self.prob_given_true = float(prob_given_true)
        self.prob_given_false = float(prob_given_false)

        # Binary sensors don't use Gaussian params or active ranges
        self.learned_gaussian_params = None
        self.learned_active_range = None

        _LOGGER.debug(
            "Updated binary likelihoods for %s: prob_given_true=%.3f, prob_given_false=%.3f",
            self.entity_id,
            self.prob_given_true,
            self.prob_given_false,
        )

    def update_decay(self, decay_start: datetime | None, is_decaying: bool) -> None:
        """Update the decay of the entity."""
        self.decay.decay_start = (
            to_utc(decay_start) if decay_start is not None else decay_start
        )
        self.decay.is_decaying = is_decaying

    def has_new_evidence(self) -> bool:
        """Update decay on actual evidence transitions.

        Handles three cases:
        1. Normal transitions (True→False, False→True)
        2. Entity becoming unavailable (True/False→None)
        3. Entity becoming available (None→True/False)

        Returns:
            bool: True if evidence transition occurred, False otherwise

        """
        # Pure calculation from current HA state
        current_evidence = self.evidence

        # Capture previous evidence before updating it
        previous_evidence = self.previous_evidence

        # Handle entity becoming unavailable (evidence was known, now None)
        if current_evidence is None:
            if previous_evidence is True:
                # Entity had evidence and became unavailable - treat as evidence lost
                # Start decay since we lost positive evidence
                self.decay.start_decay()
                self.last_updated = dt_util.utcnow()
            # Update previous evidence to track the unavailable state
            self.previous_evidence = current_evidence
            return False  # No "new evidence" but decay state updated

        # Handle entity becoming available (previous was None, now has evidence)
        if previous_evidence is None:
            # Entity just became available - update previous and don't trigger transition
            # If it has positive evidence, stop any lingering decay
            if current_evidence:
                self.decay.stop_decay()
            self.previous_evidence = current_evidence
            return False  # Entity just became available, not a true transition

        # Fix inconsistent state: if evidence is True but decay is running, stop decay
        if current_evidence and self.decay.is_decaying:
            self.decay.stop_decay()

        # Check for evidence transitions
        transition_occurred = current_evidence != previous_evidence

        # Handle evidence transitions
        if transition_occurred:
            self.last_updated = dt_util.utcnow()
            if current_evidence:  # FALSE→TRUE transition
                self.decay.stop_decay()
            else:  # TRUE→FALSE transition
                # Evidence lost - start decay
                self.decay.start_decay()

        # Update previous evidence for next comparison
        self.previous_evidence = current_evidence
        return transition_occurred


class EntityFactory:
    """Factory for creating entities from various sources."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
        area_name: str,
    ) -> None:
        """Initialize the factory.

        Args:
            coordinator: The coordinator instance
            area_name: Area name for multi-area support
        """
        self.coordinator = coordinator
        self.area_name = area_name
        # Validate area_name exists and retrieve config from coordinator.areas
        if area_name not in coordinator.areas:
            available = list(coordinator.areas.keys())
            raise ValueError(
                f"Area '{area_name}' not found. "
                f"Available areas: {available or '(none)'}"
            )
        self.config = coordinator.areas[area_name].config

    def create_from_db(self, entity_obj: "DB.Entities") -> Entity:
        """Create entity from storage data.

        Args:
            entity_obj: SQLAlchemy Entities object from database

        Returns:
            Entity: Properly constructed Entity object with Python types

        Raises:
            ValueError: If required fields are missing or invalid
            TypeError: If type conversion fails
        """
        # Convert SQLAlchemy objects to Python types using to_dict()
        # This ensures we get proper Python types rather than SQLAlchemy Column objects
        entity_data = entity_obj.to_dict()

        # Extract and validate required string fields
        entity_id = str(entity_data["entity_id"])
        entity_type_str = str(entity_data["entity_type"])

        if not entity_id:
            raise ValueError("Entity ID cannot be empty")
        if not entity_type_str:
            raise ValueError("Entity type cannot be empty")

        # Convert numeric fields with validation
        try:
            db_weight = float(entity_data["weight"])
            prob_given_true = float(entity_data["prob_given_true"])
            prob_given_false = float(entity_data["prob_given_false"])
        except (TypeError, ValueError) as e:
            raise TypeError(f"Failed to convert numeric fields: {e}") from e

        # Convert boolean field
        is_decaying = bool(entity_data["is_decaying"])

        # Handle datetime fields - ensure they're proper Python datetime objects
        decay_start = entity_data["decay_start"]
        last_updated = entity_data["last_updated"]

        # Validate datetime objects are timezone-aware
        if decay_start is not None:
            decay_start = to_utc(decay_start)

        if last_updated is not None:
            last_updated = to_utc(last_updated)

        # Convert evidence field - handle None case
        previous_evidence = entity_data["evidence"]
        if previous_evidence is not None:
            previous_evidence = bool(previous_evidence)

        # Create the entity type directly
        input_type = InputType(entity_type_str)

        # Extract overrides from config
        config_weight = None
        active_states = None
        active_range = None

        weights = getattr(self.config, "weights", None)
        if weights:
            weight_attr = getattr(weights, input_type.value, None)
            if weight_attr is not None:
                config_weight = weight_attr

        sensor_states = getattr(self.config, "sensor_states", None)
        if sensor_states:
            states_attr = getattr(sensor_states, input_type.value, None)
            if states_attr is not None:
                active_states = states_attr

        range_config_attr = f"{input_type.value}_active_range"
        range_attr = getattr(self.config, range_config_attr, None)
        if range_attr is not None:
            active_range = range_attr

        entity_type = EntityType(
            input_type,
            weight=config_weight,
            active_states=active_states,
            active_range=active_range,
        )

        # DB weight should take priority over configured default
        try:
            if MIN_WEIGHT <= db_weight <= MAX_WEIGHT:
                entity_type.weight = db_weight
        except (TypeError, ValueError):
            # Weight is invalid, keep the default from EntityType initialization
            pass

        # Motion sensors use configured likelihoods (user-configurable per area)
        # Do not use learned likelihoods from database for motion sensors
        if input_type == InputType.MOTION:
            # Get configured values from area config, fall back to defaults if not configured
            motion_prob_given_true = getattr(
                self.config.sensors,
                "motion_prob_given_true",
                DEFAULT_TYPES[InputType.MOTION]["prob_given_true"],
            )
            motion_prob_given_false = getattr(
                self.config.sensors,
                "motion_prob_given_false",
                DEFAULT_TYPES[InputType.MOTION]["prob_given_false"],
            )
            prob_given_true = float(motion_prob_given_true)
            prob_given_false = float(motion_prob_given_false)
            _LOGGER.debug(
                "Using configured likelihoods for motion sensor %s: prob_given_true=%.2f, prob_given_false=%.2f",
                entity_id,
                prob_given_true,
                prob_given_false,
            )

        # Create decay object
        # Wasp-in-Box sensors should not have decay (immediate vacancy)
        # Sleep sensors use a very long half-life (persistent presence)
        half_life = self.config.decay.half_life
        # If half_life is 0, resolve from purpose
        if half_life == 0:
            half_life = get_default_decay_half_life(self.config.purpose)

        area = self.coordinator.areas.get(self.area_name)
        is_wasp = area and area.wasp_entity_id == entity_id
        is_sleep = area and area.sleep_entity_id == entity_id
        if is_wasp:
            half_life = 0.1  # Effectively zero decay (clears in <0.5s)
        elif is_sleep:
            half_life = SLEEP_PRESENCE_HALF_LIFE

        # Get sleep settings from integration config
        # For WASP/SLEEP entities, bypass sleeping semantics
        if is_wasp or is_sleep:
            purpose_for_decay = None
            sleep_start = None
            sleep_end = None
        else:
            sleep_start = getattr(
                self.coordinator.integration_config, "sleep_start", None
            )
            sleep_end = getattr(self.coordinator.integration_config, "sleep_end", None)
            purpose_for_decay = getattr(self.config, "purpose", None)

        decay = Decay(
            half_life=half_life,
            is_decaying=is_decaying,
            decay_start=decay_start,
            purpose=purpose_for_decay,
            sleep_start=sleep_start,
            sleep_end=sleep_end,
        )

        # Set default analysis_error based on entity type
        # Motion and Sleep sensors are excluded from correlation analysis
        analysis_error = (
            AnalysisStatus.MOTION_EXCLUDED
            if input_type in (InputType.MOTION, InputType.SLEEP)
            else AnalysisStatus.NOT_ANALYZED
        )

        return Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            decay=decay,
            hass=self.coordinator.hass,
            last_updated=last_updated,
            previous_evidence=previous_evidence,
            analysis_error=analysis_error,
        )

    def create_from_config_spec(self, entity_id: str, input_type: str) -> Entity:
        """Create entity from configuration specification."""
        # Create the entity type directly
        input_type_enum = InputType(input_type)

        # Extract overrides from config
        weight = None
        active_states = None
        active_range = None

        weights = getattr(self.config, "weights", None)
        if weights:
            weight_attr = getattr(weights, input_type_enum.value, None)
            if weight_attr is not None:
                weight = weight_attr

        sensor_states = getattr(self.config, "sensor_states", None)
        if sensor_states:
            states_attr = getattr(sensor_states, input_type_enum.value, None)
            if states_attr is not None:
                active_states = states_attr

        range_config_attr = f"{input_type_enum.value}_active_range"
        range_attr = getattr(self.config, range_config_attr, None)
        if range_attr is not None:
            active_range = range_attr

        entity_type = EntityType(
            input_type_enum,
            weight=weight,
            active_states=active_states,
            active_range=active_range,
        )

        # Wasp-in-Box sensors should not have decay (immediate vacancy)
        # Sleep sensors use a very long half-life (persistent presence)
        half_life = self.config.decay.half_life
        # If half_life is 0, resolve from purpose
        if half_life == 0:
            half_life = get_default_decay_half_life(self.config.purpose)

        area = self.coordinator.areas.get(self.area_name)
        is_wasp = area and area.wasp_entity_id == entity_id
        is_sleep = area and area.sleep_entity_id == entity_id
        if is_wasp:
            half_life = 0.1  # Effectively zero decay (clears in <0.5s)
        elif is_sleep:
            half_life = SLEEP_PRESENCE_HALF_LIFE

        # Get sleep settings from integration config
        # For WASP/SLEEP entities, bypass sleeping semantics
        if is_wasp or is_sleep:
            purpose_for_decay = None
            sleep_start = None
            sleep_end = None
        else:
            sleep_start = getattr(
                self.coordinator.integration_config, "sleep_start", None
            )
            sleep_end = getattr(self.coordinator.integration_config, "sleep_end", None)
            purpose_for_decay = getattr(self.config, "purpose", None)

        decay = Decay(
            half_life=half_life,
            is_decaying=False,
            decay_start=dt_util.utcnow(),
            purpose=purpose_for_decay,
            sleep_start=sleep_start,
            sleep_end=sleep_end,
        )

        # Motion sensors use configured likelihoods (user-configurable per area)
        # Other sensors use defaults from EntityType
        if input_type_enum == InputType.MOTION:
            motion_prob_given_true = getattr(
                self.config.sensors,
                "motion_prob_given_true",
                entity_type.prob_given_true,
            )
            motion_prob_given_false = getattr(
                self.config.sensors,
                "motion_prob_given_false",
                entity_type.prob_given_false,
            )
            prob_given_true = float(motion_prob_given_true)
            prob_given_false = float(motion_prob_given_false)
        else:
            prob_given_true = entity_type.prob_given_true
            prob_given_false = entity_type.prob_given_false

        # Set default analysis_error based on entity type
        # Motion and Sleep sensors are excluded from correlation analysis
        analysis_error = (
            "motion_sensor_excluded"
            if input_type_enum in (InputType.MOTION, InputType.SLEEP)
            else "not_analyzed"
        )

        return Entity(
            entity_id=entity_id,
            type=entity_type,
            prob_given_true=prob_given_true,
            prob_given_false=prob_given_false,
            decay=decay,
            hass=self.coordinator.hass,
            last_updated=dt_util.utcnow(),
            previous_evidence=None,
            analysis_error=analysis_error,
        )

    def create_all_from_config(self) -> dict[str, Entity]:
        """Create all entities from current configuration."""
        entity_type_mapping = self.get_entity_type_mapping()
        entities = {}

        for entity_id, input_type in entity_type_mapping.items():
            _LOGGER.debug("Creating entity %s with type %s", entity_id, input_type)
            entities[entity_id] = self.create_from_config_spec(entity_id, input_type)

        return entities

    def get_entity_type_mapping(self) -> dict[str, str]:
        """Get entity type mapping for all configured entities.

        Returns a mapping of entity_id -> input_type string that can be used
        directly for entity creation.
        """
        specs = {}

        # Process each sensor type using the mapping
        for sensor_type, input_type in get_sensor_type_mapping().items():
            # Skip sleep — handled separately below (virtual sensor like wasp)
            if sensor_type == "sleep":
                continue

            sensor_list = getattr(self.config.sensors, sensor_type, [])

            # Special handling for motion sensors (includes wasp)
            if sensor_type == "motion":
                sensor_list = self.config.sensors.get_motion_sensors(self.coordinator)

            for entity_id in sensor_list:
                specs[entity_id] = input_type.value

        # Add sleep presence sensor if registered for this area
        sleep_sensors = self.config.sensors.get_sleep_sensors(self.coordinator)
        for entity_id in sleep_sensors:
            specs[entity_id] = InputType.SLEEP.value

        return specs


class EntityManager:
    """Manages entities with simplified creation and storage logic."""

    def __init__(
        self,
        coordinator: "AreaOccupancyCoordinator",
        area_name: str | None = None,
    ) -> None:
        """Initialize the entity manager.

        Args:
            coordinator: The coordinator instance
            area_name: Required area name for multi-area support. Used to look up
                the area configuration from coordinator.areas.
        """
        self.coordinator = coordinator
        self.area_name = area_name
        # Validate area_name and retrieve config from coordinator.areas
        if not area_name:
            raise ValueError("Area name is required in multi-area architecture")
        if area_name not in coordinator.areas:
            available = list(coordinator.areas.keys())
            raise ValueError(
                f"Area '{area_name}' not found. "
                f"Available areas: {available or '(none)'}"
            )
        self.config = coordinator.areas[area_name].config
        self.hass = coordinator.hass
        self._factory = EntityFactory(coordinator, area_name=area_name)
        self._entities: dict[str, Entity] = self._factory.create_all_from_config()

    @property
    def entities(self) -> dict[str, Entity]:
        """Get the entities."""
        return self._entities

    def get_entities_by_input_type(
        self, input_type: "InputType"
    ) -> dict[str, "Entity"]:
        """Get entities filtered by InputType."""
        return {
            entity_id: entity
            for entity_id, entity in self._entities.items()
            if entity.type.input_type == input_type
        }

    @property
    def entity_ids(self) -> list[str]:
        """Get the entity IDs."""
        return list(self._entities.keys())

    @property
    def active_entities(self) -> list[Entity]:
        """Get the active entities."""
        return [
            entity
            for entity in self._entities.values()
            if entity.evidence or entity.decay.is_decaying
        ]

    @property
    def inactive_entities(self) -> list[Entity]:
        """Get the inactive entities."""
        return [
            entity
            for entity in self._entities.values()
            if not entity.evidence and not entity.decay.is_decaying
        ]

    @property
    def decaying_entities(self) -> list[Entity]:
        """Get the decaying entities."""
        return [
            entity for entity in self._entities.values() if entity.decay.is_decaying
        ]

    def get_entity(self, entity_id: str) -> Entity:
        """Get the entity from an entity ID."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity not found for entity: {entity_id}")
        return self._entities[entity_id]

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the manager."""
        self._entities[entity.entity_id] = entity

    def register_entity(self, entity_id: str, input_type: str) -> None:
        """Create and register an entity from a config spec if not already tracked."""
        if entity_id not in self._entities:
            entity = self._factory.create_from_config_spec(entity_id, input_type)
            self._entities[entity_id] = entity

    def deregister_entity(self, entity_id: str) -> None:
        """Remove an entity from the manager if it exists."""
        self._entities.pop(entity_id, None)

    async def cleanup(self) -> None:
        """Clean up resources and recreate from config.

        This method clears all entity references to release memory
        and prevent leaks when areas are removed or reconfigured.
        """
        _LOGGER.debug("Cleaning up EntityManager for area: %s", self.area_name)
        # Clear all entity references to release memory
        # This ensures entities and their internal state (decay, etc.) are released
        self._entities.clear()
        # Recreate entities from config (needed for reconfiguration scenarios)
        self._entities = self._factory.create_all_from_config()
        _LOGGER.debug("EntityManager cleanup completed for area: %s", self.area_name)
