"""Bayesian probability calculations for Room Occupancy Detection."""

from __future__ import annotations

import logging
from typing import List
import numpy as np

_LOGGER = logging.getLogger(__name__)


class BayesianProbability:
    """Class to handle Bayesian probability calculations."""

    def __init__(self) -> None:
        """Initialize the probability calculator."""
        self._prior = 0.5  # Start with a neutral prior

    def update_prior(self, prior: float) -> None:
        """Update the prior probability."""
        self._prior = max(0.0, min(1.0, prior))

    def _odds(self, p: float) -> float:
        """Convert probability to odds."""
        return p / (1 - p) if p != 1 else float("inf")

    def _probability(self, o: float) -> float:
        """Convert odds to probability."""
        return o / (1 + o) if o != float("inf") else 1

    def calculate_probability(self, probabilities: List[float]) -> float:
        """
        Calculate the combined probability using Bayesian inference.

        Args:
            probabilities: List of individual probabilities from different sensors

        Returns:
            Combined probability as a float between 0 and 1
        """
        if not probabilities:
            return self._prior

        try:
            # Start with the prior odds
            combined_odds = self._odds(self._prior)

            # Multiply by likelihood ratios
            for p in probabilities:
                if 0 < p < 1:  # Avoid division by zero or infinity
                    combined_odds *= self._odds(p)

            # Convert back to probability
            return self._probability(combined_odds)

        except (ZeroDivisionError, ValueError) as err:
            _LOGGER.error("Error calculating probability: %s", err)
            return self._prior

    def calculate_weighted_probability(
        self, probabilities: List[float], weights: List[float]
    ) -> float:
        """
        Calculate weighted probability using Bayesian inference.

        Args:
            probabilities: List of individual probabilities
            weights: List of weights corresponding to each probability

        Returns:
            Weighted combined probability as a float between 0 and 1
        """
        if not probabilities or not weights or len(probabilities) != len(weights):
            return self._prior

        try:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)

            # Calculate weighted log odds
            weighted_odds = self._odds(self._prior)
            for p, w in zip(probabilities, weights):
                if 0 < p < 1:
                    weighted_odds *= self._odds(p) ** w

            return self._probability(weighted_odds)

        except (ZeroDivisionError, ValueError) as err:
            _LOGGER.error("Error calculating weighted probability: %s", err)
            return self._prior
