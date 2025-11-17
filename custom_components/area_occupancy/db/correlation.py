"""Numeric sensor correlation analysis.

This module calculates correlations between numeric sensor values (temperature,
humidity, CO2, etc.) and area occupancy to identify sensors that can be used
as occupancy indicators.
"""

from __future__ import annotations

from datetime import timedelta
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
        else:
            # t-statistic
            t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
            # Approximate p-value (two-tailed)
            # For simplicity, use a rough approximation
            # In production, use scipy.stats.pearsonr for accurate p-values
            p_value = 2 * (1 - abs(t_stat) / np.sqrt(n))

        return (float(correlation), float(p_value))

    except (ValueError, TypeError, RuntimeError) as e:
        _LOGGER.warning("Error calculating correlation: %s", e)
        return (0.0, 1.0)


def analyze_numeric_correlation(
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    analysis_period_days: int = 30,
) -> dict[str, Any] | None:
    """Analyze correlation between numeric sensor values and occupancy.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Numeric sensor entity ID
        analysis_period_days: Number of days to analyze

    Returns:
        Dictionary with correlation results, or None if insufficient data
    """
    _LOGGER.debug(
        "Analyzing correlation for %s in area %s over %d days",
        entity_id,
        area_name,
        analysis_period_days,
    )

    try:
        with db.get_session() as session:
            # Get analysis period
            period_end = dt_util.utcnow()
            period_start = period_end - timedelta(days=analysis_period_days)

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

            if len(samples) < MIN_CORRELATION_SAMPLES:
                _LOGGER.debug(
                    "Insufficient samples for correlation: %d < %d",
                    len(samples),
                    MIN_CORRELATION_SAMPLES,
                )
                return None

            # Get occupied intervals for the area
            occupied_intervals = (
                session.query(db.OccupiedIntervalsCache)
                .filter(
                    db.OccupiedIntervalsCache.area_name == area_name,
                    db.OccupiedIntervalsCache.start_time >= period_start,
                    db.OccupiedIntervalsCache.end_time <= period_end,
                )
                .all()
            )

            if not occupied_intervals:
                _LOGGER.debug("No occupied intervals found for correlation analysis")
                return None

            # Create occupancy flags for each sample
            sample_values: list[float] = []
            occupancy_flags: list[float] = []  # 1.0 for occupied, 0.0 for unoccupied

            for sample in samples:
                # Check if sample timestamp falls within any occupied interval
                is_occupied = False
                for interval in occupied_intervals:
                    if interval.start_time <= sample.timestamp <= interval.end_time:
                        is_occupied = True
                        break

                sample_values.append(float(sample.value))
                occupancy_flags.append(1.0 if is_occupied else 0.0)

            if len(sample_values) < MIN_CORRELATION_SAMPLES:
                _LOGGER.debug(
                    "Insufficient samples after filtering: %d < %d",
                    len(sample_values),
                    MIN_CORRELATION_SAMPLES,
                )
                return None

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

            mean_occupied = float(np.mean(occupied_values)) if occupied_values else None
            mean_unoccupied = (
                float(np.mean(unoccupied_values)) if unoccupied_values else None
            )
            std_occupied = float(np.std(occupied_values)) if occupied_values else None
            std_unoccupied = (
                float(np.std(unoccupied_values)) if unoccupied_values else None
            )

            # Determine correlation type
            abs_correlation = abs(correlation)
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

    except SQLAlchemyError as e:
        _LOGGER.error("Database error during correlation analysis: %s", e)
        return None
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Unexpected error during correlation analysis: %s", e)
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

    except SQLAlchemyError as e:
        _LOGGER.error("Database error saving correlation result: %s", e)
        if session is not None:
            session.rollback()
        return False
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Unexpected error saving correlation result: %s", e)
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

    except (SQLAlchemyError, ValueError, TypeError, RuntimeError) as e:
        _LOGGER.warning("Error pruning old correlations: %s", e)
        # Don't raise - this is cleanup, not critical


def analyze_and_save_correlation(
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    analysis_period_days: int = 30,
) -> bool:
    """Analyze and save correlation for a numeric sensor.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Numeric sensor entity ID
        analysis_period_days: Number of days to analyze

    Returns:
        True if analysis completed and saved, False otherwise
    """
    correlation_data = analyze_numeric_correlation(
        db, area_name, entity_id, analysis_period_days
    )

    if correlation_data is None:
        _LOGGER.debug(
            "No correlation data generated for %s in area %s",
            entity_id,
            area_name,
        )
        return False

    return save_correlation_result(db, correlation_data)


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

    except SQLAlchemyError as e:
        _LOGGER.error("Database error getting correlation: %s", e)
        return None
