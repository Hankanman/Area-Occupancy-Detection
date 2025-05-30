"""ML Hybrid probability calculation for Area Occupancy Detection.

This module handles the fusion of Bayesian and ML probabilities using
configurable weighting and switching logic.
"""

from __future__ import annotations

import logging
from typing import Any

from .types import AnalysisMethod, MLHybridResult, MLPrediction

_LOGGER = logging.getLogger(__name__)


def combine_probabilities(
    bayesian_prob: float,
    ml_prediction: MLPrediction | None,
    analysis_method: AnalysisMethod,
    ml_confidence_threshold: float = 0.7,
) -> MLHybridResult:
    """Combine Bayesian and ML probabilities using specified method.

    Args:
        bayesian_prob: Probability from Bayesian calculation (0-1)
        ml_prediction: ML prediction result or None if not available
        analysis_method: Method for combining probabilities
        ml_confidence_threshold: Minimum confidence to trust ML prediction

    Returns:
        MLHybridResult with combined probability and metadata

    """
    # Initialize result with Bayesian probability as fallback
    result = MLHybridResult(
        final_probability=bayesian_prob,
        ml_probability=None,
        ml_confidence=None,
        bayesian_probability=bayesian_prob,
        method_used=AnalysisMethod.DETERMINISTIC,
        feature_data=None,
    )

    # If no ML prediction available, return Bayesian result
    if ml_prediction is None:
        _LOGGER.debug(
            "No ML prediction available, using Bayesian probability: %.3f",
            bayesian_prob,
        )
        return result

    # Update result with ML information
    result.ml_probability = ml_prediction.probability
    result.ml_confidence = ml_prediction.confidence
    result.feature_data = ml_prediction.feature_importance

    # Apply combination strategy based on analysis method
    if analysis_method == AnalysisMethod.ML:
        # Use ML only if confidence is sufficient
        if ml_prediction.confidence >= ml_confidence_threshold:
            result.final_probability = ml_prediction.probability
            result.method_used = AnalysisMethod.ML
            _LOGGER.debug(
                "Using ML prediction: %.3f (confidence: %.3f)",
                ml_prediction.probability,
                ml_prediction.confidence,
            )
        else:
            _LOGGER.debug(
                "ML confidence too low (%.3f < %.3f), falling back to Bayesian: %.3f",
                ml_prediction.confidence,
                ml_confidence_threshold,
                bayesian_prob,
            )

    elif analysis_method == AnalysisMethod.HYBRID:
        # Combine based on ML confidence
        combined_prob = _hybrid_combination(
            bayesian_prob, ml_prediction, ml_confidence_threshold
        )
        result.final_probability = combined_prob
        result.method_used = AnalysisMethod.HYBRID
        _LOGGER.debug(
            "Hybrid combination: Bayesian=%.3f, ML=%.3f (conf=%.3f), Final=%.3f",
            bayesian_prob,
            ml_prediction.probability,
            ml_prediction.confidence,
            combined_prob,
        )

    else:  # DETERMINISTIC
        # Use Bayesian only
        _LOGGER.debug("Using deterministic (Bayesian) method: %.3f", bayesian_prob)

    return result


def _hybrid_combination(
    bayesian_prob: float,
    ml_prediction: MLPrediction,
    confidence_threshold: float,
) -> float:
    """Combine Bayesian and ML probabilities using confidence-weighted approach.

    Args:
        bayesian_prob: Bayesian probability
        ml_prediction: ML prediction with confidence
        confidence_threshold: Minimum confidence threshold

    Returns:
        Combined probability

    """
    # If ML confidence is below threshold, use Bayesian
    if ml_prediction.confidence < confidence_threshold:
        return bayesian_prob

    # Confidence-weighted combination
    # Higher confidence = more weight on ML
    # Lower confidence = more weight on Bayesian

    # Normalize confidence to 0-1 range above threshold
    normalized_confidence = min(
        (ml_prediction.confidence - confidence_threshold)
        / (1.0 - confidence_threshold),
        1.0,
    )

    # Weight ML prediction by normalized confidence
    ml_weight = normalized_confidence
    bayesian_weight = 1.0 - ml_weight

    combined = ml_weight * ml_prediction.probability + bayesian_weight * bayesian_prob

    return combined


def calculate_confidence_score(
    ml_prediction: MLPrediction | None,
    bayesian_prob: float,
    feature_quality: dict[str, Any] | None = None,
) -> float:
    """Calculate overall confidence score for the hybrid result.

    Args:
        ml_prediction: ML prediction result
        bayesian_prob: Bayesian probability
        feature_quality: Optional feature quality metrics

    Returns:
        Overall confidence score (0-1)

    """
    confidence_factors = []

    # ML confidence factor
    if ml_prediction is not None:
        confidence_factors.append(ml_prediction.confidence)

    # Bayesian confidence factor (based on how decisive the probability is)
    # Probabilities closer to 0 or 1 are more confident
    bayesian_confidence = 2 * abs(bayesian_prob - 0.5)
    confidence_factors.append(bayesian_confidence)

    # Feature quality factor
    if feature_quality:
        # Simple heuristic: higher feature coverage = higher confidence
        feature_coverage = feature_quality.get("coverage", 0.5)
        confidence_factors.append(feature_coverage)

    # Return average of confidence factors
    return (
        sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
    )


def get_method_explanation(result: MLHybridResult) -> str:
    """Get human-readable explanation of the method used.

    Args:
        result: Hybrid calculation result

    Returns:
        Explanation string

    """
    if result.method_used == AnalysisMethod.ML:
        return (
            f"ML model prediction (confidence: {result.ml_confidence:.1%}) "
            f"was used with high confidence."
        )
    elif result.method_used == AnalysisMethod.HYBRID:
        return (
            f"Hybrid approach combining Bayesian ({result.bayesian_probability:.1%}) "
            f"and ML ({result.ml_probability:.1%}) predictions."
        )
    else:  # DETERMINISTIC
        if result.ml_probability is None:
            return "Bayesian calculation used (ML model not available)."
        else:
            return (
                f"Bayesian calculation used (ML confidence {result.ml_confidence:.1%} "
                f"was too low)."
            )


def analyze_prediction_agreement(
    bayesian_prob: float,
    ml_prediction: MLPrediction | None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Analyze agreement between Bayesian and ML predictions.

    Args:
        bayesian_prob: Bayesian probability
        ml_prediction: ML prediction result
        threshold: Threshold for binary classification

    Returns:
        Dictionary with agreement analysis

    """
    analysis = {
        "agreement": None,
        "difference": None,
        "bayesian_decision": bayesian_prob >= threshold,
        "ml_decision": None,
        "confidence_level": "unknown",
    }

    if ml_prediction is None:
        return analysis

    analysis["ml_decision"] = ml_prediction.probability >= threshold
    analysis["difference"] = abs(bayesian_prob - ml_prediction.probability)
    analysis["agreement"] = analysis["bayesian_decision"] == analysis["ml_decision"]

    # Categorize confidence level
    if ml_prediction.confidence >= 0.8:
        analysis["confidence_level"] = "high"
    elif ml_prediction.confidence >= 0.6:
        analysis["confidence_level"] = "medium"
    else:
        analysis["confidence_level"] = "low"

    return analysis


def recommend_analysis_method(
    recent_agreements: list[bool],
    ml_availability: float,
    computational_resources: str = "normal",
) -> AnalysisMethod:
    """Recommend optimal analysis method based on performance history.

    Args:
        recent_agreements: List of recent agreement results between methods
        ml_availability: Fraction of time ML predictions are available (0-1)
        computational_resources: 'low', 'normal', or 'high'

    Returns:
        Recommended analysis method

    """
    # If ML is rarely available, use deterministic
    if ml_availability < 0.3:
        return AnalysisMethod.DETERMINISTIC

    # If computational resources are low, prefer simpler methods
    if computational_resources == "low":
        return AnalysisMethod.DETERMINISTIC

    # If we have recent agreement data, use it to decide
    if recent_agreements:
        agreement_rate = sum(recent_agreements) / len(recent_agreements)

        # High agreement suggests both methods work well - use hybrid
        if agreement_rate > 0.8:
            return AnalysisMethod.HYBRID
        # Low agreement suggests conflicting methods - investigate further
        elif agreement_rate < 0.5:
            return AnalysisMethod.DETERMINISTIC  # Fall back to known method
        else:
            return AnalysisMethod.HYBRID  # Medium agreement - hybrid is reasonable

    # Default to hybrid for balanced approach
    return AnalysisMethod.HYBRID
