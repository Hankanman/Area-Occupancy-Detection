"""Probability model for Area Occupancy Detection."""

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Dict, Optional

from homeassistant.util import dt as dt_util

from ..const import MAX_PROBABILITY, MIN_PROBABILITY
from ..utils import validate_prob, validate_datetime
from ..coordinator import AreaOccupancyCoordinator
from .entity import Entity

_LOGGER = logging.getLogger(__name__)


@dataclass
class Probability:
    """Represents the probability state for an entity."""

    entity_id: str
    probability: float
    weighted_probability: float
    decayed_probability: float
    decay_factor: float
    last_updated: datetime
    is_active: bool

    def __post_init__(self):
        """Validate properties after initialization."""
        self.probability = validate_prob(self.probability)
        self.weighted_probability = validate_prob(self.weighted_probability)
        self.decayed_probability = validate_prob(self.decayed_probability)
        self.decay_factor = validate_prob(self.decay_factor)
        self.last_updated = validate_datetime(self.last_updated)

    def update(
        self,
        probability: Optional[float] = None,
        weighted_probability: Optional[float] = None,
        decayed_probability: Optional[float] = None,
        decay_factor: Optional[float] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        """Update probability values.

        Args:
            probability: New probability value
            weighted_probability: New weighted probability value
            decayed_probability: New decayed probability value
            decay_factor: New decay factor
            is_active: New active status
        """
        if probability is not None:
            self.probability = validate_prob(probability)
        if weighted_probability is not None:
            self.weighted_probability = validate_prob(weighted_probability)
        if decayed_probability is not None:
            self.decayed_probability = validate_prob(decayed_probability)
        if decay_factor is not None:
            self.decay_factor = validate_prob(decay_factor)
        if is_active is not None:
            self.is_active = is_active
        self.last_updated = dt_util.utcnow()


class ProbabilityManager:
    """Manages probability calculations for entities."""

    def __init__(self, coordinator: AreaOccupancyCoordinator) -> None:
        """Initialize the probability manager."""
        self.coordinator = coordinator
        self.config = coordinator.config
        self._probabilities: Dict[str, Probability] = {}

    @property
    def probabilities(self) -> Dict[str, Probability]:
        """Get all stored probabilities."""
        return self._probabilities

    @property
    def complementary_probability(self) -> float:
        """Calculate the complementary probability across all entities.
        
        This represents the overall probability that the area is occupied,
        taking into account all entity probabilities and their weights.
        
        Returns:
            float: The complementary probability (0.0-1.0)
        """
        if not self._probabilities:
            return MIN_PROBABILITY

        # Calculate weighted sum of all entity probabilities
        total_weight = 0.0
        weighted_sum = 0.0

        for prob in self._probabilities.values():
            if prob.is_active:  # Only consider active entities
                weight = self.coordinator.entities.get_entity(prob.entity_id).type.weight
                total_weight += weight
                weighted_sum += prob.decayed_probability * weight

        if total_weight == 0:
            return MIN_PROBABILITY

        # Calculate final probability
        final_probability = weighted_sum / total_weight
        return validate_prob(final_probability)

    def get_probability(self, entity_id: str) -> Optional[Probability]:
        """Get the probability for an entity."""
        return self._probabilities.get(entity_id)

    def update_probability(
        self,
        entity_id: str,
        probability: float,
        weighted_probability: float,
        decayed_probability: float,
        decay_factor: float,
        is_active: bool,
    ) -> None:
        """Update the probability for an entity.
        
        Args:
            entity_id: The entity ID
            probability: The raw probability value
            weighted_probability: The weighted probability value
            decayed_probability: The decayed probability value
            decay_factor: The decay factor applied
            is_active: Whether the entity is active
        """
        if entity_id not in self._probabilities:
            self._probabilities[entity_id] = Probability(
                entity_id=entity_id,
                probability=probability,
                weighted_probability=weighted_probability,
                decayed_probability=decayed_probability,
                decay_factor=decay_factor,
                last_updated=dt_util.utcnow(),
                is_active=is_active,
            )
        else:
            self._probabilities[entity_id].update(
                probability=probability,
                weighted_probability=weighted_probability,
                decayed_probability=decayed_probability,
                decay_factor=decay_factor,
                is_active=is_active,
            )

    def remove_probability(self, entity_id: str) -> None:
        """Remove the probability for an entity."""
        self._probabilities.pop(entity_id, None)

    def clear_probabilities(self) -> None:
        """Clear all stored probabilities."""
        self._probabilities.clear()

    def calculate_entity_probability(self, entity: Entity) -> Probability:
        """Calculate probability for a single entity.
        
        Args:
            entity: The entity to calculate probability for
            
        Returns:
            Probability: The calculated probability state

        """
        # Get the entity's type and weight
        entity_type = entity.type
        weight = entity_type.weight

        # Calculate raw probability using Bayesian update
        raw_probability = self._calculate_bayesian_probability(
            entity_type.prior,
            entity_type.prob_true,
            entity_type.prob_false,
            entity.is_active,
        )

        # Calculate weighted probability
        weighted_probability = raw_probability * weight

        # Get decay values from the entity
        decayed_probability = entity.probability.decayed_probability
        decay_factor = entity.probability.decay_factor

        # Create and store the probability
        probability = Probability(
            entity_id=entity.entity_id,
            probability=raw_probability,
            weighted_probability=weighted_probability,
            decayed_probability=decayed_probability,
            decay_factor=decay_factor,
            last_updated=dt_util.utcnow(),
            is_active=entity.is_active,
        )

        self._probabilities[entity.entity_id] = probability
        return probability

    @staticmethod
    def _calculate_bayesian_probability(
        prior: float,
        prob_true: float,
        prob_false: float,
        is_active: bool,
    ) -> float:
        """Calculate probability using Bayesian update.
        
        Args:
            prior: The prior probability
            prob_true: Probability given true
            prob_false: Probability given false
            is_active: Whether the entity is active
            
        Returns:
            float: The calculated probability

        """
        if is_active:
            # P(occupied | active) = P(active | occupied) * P(occupied) / P(active)
            numerator = prob_true * prior
            denominator = (prob_true * prior) + (prob_false * (1 - prior))
        else:
            # P(occupied | inactive) = P(inactive | occupied) * P(occupied) / P(inactive)
            numerator = (1 - prob_true) * prior
            denominator = ((1 - prob_true) * prior) + ((1 - prob_false) * (1 - prior))

        if denominator == 0:
            return MIN_PROBABILITY

        probability = numerator / denominator
        return validate_prob(probability)