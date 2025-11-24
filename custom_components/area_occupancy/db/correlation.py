"""Numeric sensor correlation analysis.

This module calculates correlations between sensor values (numeric and binary)
and area occupancy to identify sensors that can be used as occupancy indicators.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import (
    CORRELATION_MODERATE_THRESHOLD,
    CORRELATION_STRONG_THRESHOLD,
    MIN_CORRELATION_SAMPLES,
    NUMERIC_CORRELATION_HISTORY_COUNT,
)
from .utils import (
    get_occupied_intervals_for_analysis,
    is_timestamp_occupied,
    validate_occupied_intervals,
    validate_sample_count,
)

if TYPE_CHECKING:
    from .core import AreaOccupancyDB

_LOGGER = logging.getLogger(__name__)


def calculate_pearson_correlation(
    x_values: list[float], y_values: list[float]
) -> tuple[float, float]:
    """Calculate Pearson correlation coefficient and p-value.

    Args:
        x_values: First variable values
        y_values: Second variable values

    Returns:
        Tuple of (correlation_coefficient, p_value)
        Returns (0.0, 1.0) if insufficient data or calculation fails
    """
    if len(x_values) != len(y_values):
        _LOGGER.warning("Mismatched array lengths for correlation calculation")
        return (0.0, 1.0)

    if len(x_values) < MIN_CORRELATION_SAMPLES:
        _LOGGER.debug(
            "Insufficient samples for correlation: %d < %d",
            len(x_values),
            MIN_CORRELATION_SAMPLES,
        )
        return (0.0, 1.0)

    try:
        # Convert to numpy arrays
        x = np.array(x_values)
        y = np.array(y_values)

        # Calculate correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

        # Handle NaN or invalid values
        if np.isnan(correlation) or not np.isfinite(correlation):
            _LOGGER.debug("Invalid correlation value: %s", correlation)
            return (0.0, 1.0)

        # Calculate p-value using t-test
        # For large samples, we can approximate
        n = len(x_values)
        if n < 3:
            p_value = 1.0
        elif abs(abs(correlation) - 1.0) < 1e-10:
            # Handle perfect correlation (correlation = ±1.0)
            # When correlation is exactly 1.0 or -1.0, p-value should be very small (near 0)
            p_value = 0.0  # Perfect correlation has p-value near 0
        else:
            # t-statistic
            # Avoid division by zero when correlation is very close to ±1.0
            denominator = 1 - correlation**2
            if abs(denominator) < 1e-10:
                p_value = 0.0  # Very strong correlation
            else:
                t_stat = correlation * np.sqrt((n - 2) / denominator)
                # Approximate p-value (two-tailed)
                # Use a more robust approximation that ensures p-value is in [0, 1]
                # For large n, p-value ≈ 2 * (1 - Φ(|t|)) where Φ is standard normal CDF
                # Simplified approximation: clamp to valid range
                p_approx = 2 * (1 - min(1.0, abs(t_stat) / np.sqrt(n)))
                p_value = max(0.0, min(1.0, p_approx))

        return (float(correlation), float(p_value))

    except (ValueError, TypeError, RuntimeError) as e:
        _LOGGER.warning("Error calculating correlation: %s", e)
        return (0.0, 1.0)


def convert_intervals_to_samples(
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    period_start: datetime,
    period_end: datetime,
    active_states: list[str] | None,
    session: Any,
) -> list[Any]:
    """Convert binary sensor intervals to numeric samples.

    For binary sensors, creates samples with value 1.0 for active intervals
    and 0.0 for inactive intervals. One sample is created per interval at
    the interval midpoint.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Entity ID
        period_start: Analysis period start
        period_end: Analysis period end
        active_states: List of active states for the binary sensor
        session: Database session

    Returns:
        List of sample objects (similar to NumericSamples)
    """
    if not active_states:
        return []

    # Query intervals for the entity within the period
    intervals = (
        session.query(db.Intervals)
        .filter(
            db.Intervals.entity_id == entity_id,
            db.Intervals.start_time >= period_start,
            db.Intervals.start_time <= period_end,
        )
        .order_by(db.Intervals.start_time)
        .all()
    )

    samples = []
    for interval in intervals:
        # Calculate midpoint
        midpoint = interval.start_time + (interval.end_time - interval.start_time) / 2

        # Determine value based on state
        is_active = interval.state in active_states
        value = 1.0 if is_active else 0.0

        # Create a simple object mimicking NumericSamples
        # We use a simple class or dict to be compatible with the analysis code
        class SimpleSample:
            def __init__(self, timestamp: datetime, value: float) -> None:
                self.timestamp = timestamp
                self.value = value

        samples.append(SimpleSample(midpoint, value))

    return samples


def analyze_correlation(
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    analysis_period_days: int = 30,
    is_binary: bool = False,
    active_states: list[str] | None = None,
) -> dict[str, Any] | None:
    """Analyze correlation between sensor values and occupancy.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Sensor entity ID
        analysis_period_days: Number of days to analyze
        is_binary: Whether the entity is a binary sensor
        active_states: List of active states (required if is_binary is True)

    Returns:
        Dictionary with correlation results, or None if insufficient data
    """
    _LOGGER.debug(
        "Analyzing correlation for %s in area %s over %d days (binary=%s)",
        entity_id,
        area_name,
        analysis_period_days,
        is_binary,
    )

    try:
        with db.get_session() as session:
            # Get analysis period
            period_end = dt_util.utcnow()
            period_start = period_end - timedelta(days=analysis_period_days)

            # Base result structure with defaults for required fields
            base_result = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "analysis_period_start": period_start,
                "analysis_period_end": period_end,
                "calculation_date": dt_util.utcnow(),
                "correlation_coefficient": 0.0,
                "sample_count": 0,
                "correlation_type": "none",
                "confidence": 0.0,
            }

            if is_binary:
                if not active_states:
                    _LOGGER.warning(
                        "Cannot analyze binary correlation for %s: no active states provided",
                        entity_id,
                    )
                    return None
                samples = convert_intervals_to_samples(
                    db,
                    area_name,
                    entity_id,
                    period_start,
                    period_end,
                    active_states,
                    session,
                )
            else:
                # Get numeric samples for the entity
                samples = (
                    session.query(db.NumericSamples)
                    .filter(
                        db.NumericSamples.area_name == area_name,
                        db.NumericSamples.entity_id == entity_id,
                        db.NumericSamples.timestamp >= period_start,
                        db.NumericSamples.timestamp <= period_end,
                    )
                    .order_by(db.NumericSamples.timestamp)
                    .all()
                )

            if error := validate_sample_count(samples):
                base_result.update(error)
                return base_result

            # Get occupied intervals for the area
            occupied_intervals = get_occupied_intervals_for_analysis(
                db, area_name, period_start, period_end
            )

            if error := validate_occupied_intervals(occupied_intervals, len(samples)):
                base_result.update(error)
                return base_result

            # Create occupancy flags for each sample
            sample_values: list[float] = []
            occupancy_flags: list[float] = []  # 1.0 for occupied, 0.0 for unoccupied

            for sample in samples:
                # Check if sample timestamp falls within any occupied interval
                is_occupied = is_timestamp_occupied(
                    sample.timestamp, occupied_intervals
                )

                sample_values.append(float(sample.value))
                occupancy_flags.append(1.0 if is_occupied else 0.0)

            if error := validate_sample_count(
                sample_values, error_type="too_few_samples_after_filtering"
            ):
                base_result.update(error)
                return base_result

            # Calculate correlation
            correlation, _p_value = calculate_pearson_correlation(
                sample_values, occupancy_flags
            )

            # Calculate statistics for occupied vs unoccupied
            occupied_values = [
                val
                for val, occ in zip(sample_values, occupancy_flags, strict=True)
                if occ == 1.0
            ]
            unoccupied_values = [
                val
                for val, occ in zip(sample_values, occupancy_flags, strict=True)
                if occ == 0.0
            ]

            if not occupied_values:
                base_result.update(
                    {
                        "sample_count": len(sample_values),
                        "analysis_error": "no_occupied_samples",
                    }
                )
                return base_result
            if not unoccupied_values:
                base_result.update(
                    {
                        "sample_count": len(sample_values),
                        "analysis_error": "no_unoccupied_samples",
                    }
                )
                return base_result

            mean_occupied = float(np.mean(occupied_values)) if occupied_values else None
            mean_unoccupied = (
                float(np.mean(unoccupied_values)) if unoccupied_values else None
            )
            std_occupied = float(np.std(occupied_values)) if occupied_values else None
            std_unoccupied = (
                float(np.std(unoccupied_values)) if unoccupied_values else None
            )

            # Clamp std dev for binary sensors to avoid numerical issues
            if is_binary:
                if std_occupied is not None:
                    std_occupied = max(0.05, min(0.95, std_occupied))
                if std_unoccupied is not None:
                    std_unoccupied = max(0.05, min(0.95, std_unoccupied))

            # Determine correlation type
            abs_correlation = abs(correlation)
            correlation_type = "none"
            analysis_error = None

            if (
                abs_correlation >= CORRELATION_STRONG_THRESHOLD
                or abs_correlation >= CORRELATION_MODERATE_THRESHOLD
            ):
                if correlation > 0:
                    correlation_type = "occupancy_positive"
                else:
                    correlation_type = "occupancy_negative"
            else:
                correlation_type = "none"
                analysis_error = "no_correlation"

            # Calculate confidence (based on correlation strength and sample size)
            # Confidence increases with stronger correlation and more samples
            sample_count = len(sample_values)
            confidence = min(
                1.0,
                abs_correlation * (1.0 - (MIN_CORRELATION_SAMPLES / sample_count)),
            )

            # Calculate thresholds (mean ± 1 std for active/inactive)
            threshold_active = (
                mean_occupied + std_occupied
                if mean_occupied is not None and std_occupied is not None
                else None
            )
            threshold_inactive = (
                mean_unoccupied - std_unoccupied
                if mean_unoccupied is not None and std_unoccupied is not None
                else None
            )

            result = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "correlation_coefficient": correlation,
                "correlation_type": correlation_type,
                "analysis_error": analysis_error,
                "analysis_period_start": period_start,
                "analysis_period_end": period_end,
                "sample_count": sample_count,
                "confidence": confidence,
                "mean_value_when_occupied": mean_occupied,
                "mean_value_when_unoccupied": mean_unoccupied,
                "std_dev_when_occupied": std_occupied,
                "std_dev_when_unoccupied": std_unoccupied,
                "threshold_active": threshold_active,
                "threshold_inactive": threshold_inactive,
                "calculation_date": dt_util.utcnow(),
            }

            _LOGGER.info(
                "Correlation analysis for %s: coefficient=%.3f, type=%s, confidence=%.3f",
                entity_id,
                correlation,
                correlation_type,
                confidence,
            )

            return result

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error during correlation analysis: %s", e)
        return None


def save_correlation_result(
    db: AreaOccupancyDB, correlation_data: dict[str, Any]
) -> bool:
    """Save correlation analysis result to database.

    Args:
        db: Database instance
        correlation_data: Correlation analysis result dictionary

    Returns:
        True if saved successfully, False otherwise
    """
    _LOGGER.debug(
        "Saving correlation result for %s in area %s",
        correlation_data["entity_id"],
        correlation_data["area_name"],
    )

    session = None
    try:
        with db.get_locked_session() as session:
            # Check if correlation already exists for this period
            existing = (
                session.query(db.NumericCorrelations)
                .filter_by(
                    area_name=correlation_data["area_name"],
                    entity_id=correlation_data["entity_id"],
                    analysis_period_start=correlation_data["analysis_period_start"],
                )
                .first()
            )

            if existing:
                # Update existing record
                for key, value in correlation_data.items():
                    if key not in (
                        "entry_id",
                        "area_name",
                        "entity_id",
                        "analysis_period_start",
                    ):
                        setattr(existing, key, value)
                existing.updated_at = dt_util.utcnow()
            else:
                # Create new record
                # Ensure calculation_date is set if not provided
                if "calculation_date" not in correlation_data:
                    correlation_data["calculation_date"] = dt_util.utcnow()
                correlation = db.NumericCorrelations(**correlation_data)
                session.add(correlation)

            session.commit()

            # Prune old correlation results (keep only last N)
            _prune_old_correlations(
                db,
                session,
                correlation_data["area_name"],
                correlation_data["entity_id"],
            )

            _LOGGER.debug("Correlation result saved successfully")
            return True

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error saving correlation result: %s", e)
        if session is not None:
            session.rollback()
        return False


def _prune_old_correlations(
    db: AreaOccupancyDB,
    session: Any,
    area_name: str,
    entity_id: str,
) -> None:
    """Prune old correlation results, keeping only the most recent N.

    Args:
        db: Database instance
        session: Database session
        area_name: Area name
        entity_id: Entity ID
    """
    try:
        # Get all correlations for this entity, ordered by calculation date
        correlations = (
            session.query(db.NumericCorrelations)
            .filter_by(area_name=area_name, entity_id=entity_id)
            .order_by(db.NumericCorrelations.calculation_date.desc())
            .all()
        )

        # Keep only the most recent N correlations
        if len(correlations) > NUMERIC_CORRELATION_HISTORY_COUNT:
            to_delete = correlations[NUMERIC_CORRELATION_HISTORY_COUNT:]
            for correlation in to_delete:
                session.delete(correlation)
            session.commit()
            _LOGGER.debug(
                "Pruned %d old correlation results for %s",
                len(to_delete),
                entity_id,
            )

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.warning("Error pruning old correlations: %s", e)
        # Don't raise - this is cleanup, not critical


def analyze_and_save_correlation(
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    analysis_period_days: int = 30,
    is_binary: bool = False,
    active_states: list[str] | None = None,
) -> dict[str, Any] | None:
    """Analyze and save correlation for a sensor (numeric or binary).

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Sensor entity ID
        analysis_period_days: Number of days to analyze
        is_binary: Whether the entity is a binary sensor
        active_states: List of active states (required if is_binary is True)

    Returns:
        Correlation data if analysis completed and saved, None otherwise
    """
    correlation_data = analyze_correlation(
        db, area_name, entity_id, analysis_period_days, is_binary, active_states
    )

    if correlation_data is None:
        _LOGGER.debug(
            "No correlation data generated for %s in area %s",
            entity_id,
            area_name,
        )
        return None

    if save_correlation_result(db, correlation_data):
        return correlation_data
    return None


def get_correlation_for_entity(
    db: AreaOccupancyDB, area_name: str, entity_id: str
) -> dict[str, Any] | None:
    """Get the most recent correlation result for an entity.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Entity ID

    Returns:
        Most recent correlation result as dictionary, or None if not found
    """
    try:
        with db.get_session() as session:
            correlation = (
                session.query(db.NumericCorrelations)
                .filter_by(area_name=area_name, entity_id=entity_id)
                .order_by(db.NumericCorrelations.calculation_date.desc())
                .first()
            )

            if correlation:
                return {
                    "correlation_coefficient": correlation.correlation_coefficient,
                    "correlation_type": correlation.correlation_type,
                    "confidence": correlation.confidence,
                    "mean_value_when_occupied": correlation.mean_value_when_occupied,
                    "mean_value_when_unoccupied": correlation.mean_value_when_unoccupied,
                    "threshold_active": correlation.threshold_active,
                    "threshold_inactive": correlation.threshold_inactive,
                    "calculation_date": correlation.calculation_date,
                }

            return None

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error getting correlation: %s", e)
        return None
