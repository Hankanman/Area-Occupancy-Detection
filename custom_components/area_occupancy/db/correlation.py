"""Numeric sensor correlation analysis.

This module calculates correlations between sensor values (numeric and binary)
and area occupancy to identify sensors that can be used as occupancy indicators.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from homeassistant.util import dt as dt_util

from ..const import (
    AGGREGATION_PERIOD_HOURLY,
    CORRELATION_MODERATE_THRESHOLD,
    CORRELATION_MONTHS_TO_KEEP,
    CORRELATION_WEAK_THRESHOLD,
    MIN_CORRELATION_SAMPLES,
    RETENTION_RAW_NUMERIC_SAMPLES_DAYS,
)
from ..data.entity_type import CorrelationType, InputType
from ..time_utils import from_db_utc, to_db_utc, to_local, to_utc
from ..utils import clamp_probability, map_binary_state_to_semantic
from .utils import (
    get_occupied_intervals_for_analysis,
    is_timestamp_occupied,
    validate_sample_count,
)

if TYPE_CHECKING:
    from ..coordinator import AreaOccupancyCoordinator
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
        # Suppress numpy RuntimeWarning for zero-variance arrays (constant values
        # produce stddev=0, causing divide-by-zero in corrcoef). The resulting
        # NaN is handled below.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
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


class SimpleSample:
    """Simple sample object mimicking NumericSamples for binary sensor conversion."""

    def __init__(self, timestamp: datetime, value: float) -> None:
        """Initialize sample with timestamp and value.

        Args:
            timestamp: Sample timestamp
            value: Sample value (0.0 or 1.0 for binary sensors)
        """
        self.timestamp = timestamp
        self.value = value


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
    the interval midpoint. Only intervals whose midpoints fall within the
    analysis window are included.

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

    # Runtime comparisons use aware UTC; DB filters use naive UTC
    period_start_utc = to_utc(period_start)
    period_end_utc = to_utc(period_end)
    period_start_db = to_db_utc(period_start_utc)
    period_end_db = to_db_utc(period_end_utc)

    # Query intervals for the entity, filtered by area_name and entry_id
    # Filter intervals that overlap with the analysis period
    intervals = (
        session.query(db.Intervals)
        .filter(
            db.Intervals.entry_id == db.coordinator.entry_id,
            db.Intervals.area_name == area_name,
            db.Intervals.entity_id == entity_id,
            # Include intervals that overlap with the period:
            # - Start time before period_end (interval starts before period ends)
            # - End time after period_start (interval ends after period starts)
            db.Intervals.start_time < period_end_db,
            db.Intervals.end_time > period_start_db,
        )
        .order_by(db.Intervals.start_time)
        .all()
    )

    samples = []

    for interval in intervals:
        # DB stores naive UTC
        interval_start = from_db_utc(interval.start_time)
        interval_end = from_db_utc(interval.end_time)

        # Clamp interval bounds to period boundaries
        clamped_start = max(interval_start, period_start_utc)
        clamped_end = min(interval_end, period_end_utc)

        # Calculate midpoint of clamped interval
        midpoint = clamped_start + (clamped_end - clamped_start) / 2

        # Ensure midpoint falls within the analysis window
        if midpoint < period_start_utc or midpoint > period_end_utc:
            continue

        # Determine value based on state
        is_active = interval.state in active_states
        value = 1.0 if is_active else 0.0

        samples.append(SimpleSample(midpoint, value))

    return samples


def convert_hourly_aggregates_to_samples(
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    period_start: datetime,
    period_end: datetime,
    session: Any,
) -> list[SimpleSample]:
    """Convert hourly numeric aggregates to sample objects.

    For numeric sensors, creates samples from hourly aggregates using avg_value
    as the representative value for each hour. Uses hour midpoint as timestamp.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Entity ID
        period_start: Analysis period start
        period_end: Analysis period end
        session: Database session

    Returns:
        List of SimpleSample objects created from hourly aggregates
    """
    # Runtime comparisons use aware UTC; DB filters use naive UTC
    period_start_utc = to_utc(period_start)
    period_end_utc = to_utc(period_end)
    period_start_db = to_db_utc(period_start_utc)
    period_end_db = to_db_utc(period_end_utc)

    # Query hourly aggregates for the entity that overlap with the analysis period
    aggregates = (
        session.query(db.NumericAggregates)
        .filter(
            db.NumericAggregates.entry_id == db.coordinator.entry_id,
            db.NumericAggregates.area_name == area_name,
            db.NumericAggregates.entity_id == entity_id,
            db.NumericAggregates.aggregation_period == AGGREGATION_PERIOD_HOURLY,
            # Include aggregates that overlap with the period:
            # - period_start before aggregate period_end (aggregate starts before period ends)
            # - period_end after aggregate period_start (aggregate ends after period starts)
            db.NumericAggregates.period_start < period_end_db,
            db.NumericAggregates.period_end > period_start_db,
        )
        .order_by(db.NumericAggregates.period_start)
        .all()
    )

    samples = []

    for aggregate in aggregates:
        # DB stores naive UTC
        agg_start = from_db_utc(aggregate.period_start)
        agg_end = from_db_utc(aggregate.period_end)

        # Clamp aggregate bounds to period boundaries
        clamped_start = max(agg_start, period_start_utc)
        clamped_end = min(agg_end, period_end_utc)

        # Calculate midpoint of clamped aggregate period
        midpoint = clamped_start + (clamped_end - clamped_start) / 2

        # Ensure midpoint falls within the analysis window
        if midpoint < period_start_utc or midpoint > period_end_utc:
            continue

        # avg_value is guaranteed by the aggregator (always computed from raw samples)
        samples.append(SimpleSample(midpoint, float(aggregate.avg_value)))

    return samples


def analyze_binary_likelihoods(
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    analysis_period_days: int = 30,
    active_states: list[str] | None = None,
) -> dict[str, Any] | None:
    """Analyze binary sensor likelihoods using duration-based probability calculation.

    Calculates P(active|occupied) and P(active|unoccupied) by measuring interval
    overlap durations rather than using correlation analysis.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Binary sensor entity ID
        analysis_period_days: Number of days to analyze
        active_states: List of active states (e.g., ["on", "playing"])

    Returns:
        Dictionary with likelihood results, or None if insufficient data
    """
    _LOGGER.debug(
        "Analyzing binary likelihoods for %s in area %s over %d days",
        entity_id,
        area_name,
        analysis_period_days,
    )

    if not active_states:
        _LOGGER.warning("No active_states provided for binary sensor %s", entity_id)
        return None

    try:
        with db.get_session() as session:
            # Get analysis period
            period_end = dt_util.utcnow()
            period_start = period_end - timedelta(days=analysis_period_days)
            period_start_utc = to_utc(period_start)
            period_end_utc = to_utc(period_end)
            period_start_db = to_db_utc(period_start_utc)
            period_end_db = to_db_utc(period_end_utc)

            # Base result structure
            base_result = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "analysis_period_start": period_start,
                "analysis_period_end": period_end,
                # Stored in DB as naive UTC
                "calculation_date": to_db_utc(dt_util.utcnow()),
            }

            # Get occupied intervals for the area
            occupied_intervals = get_occupied_intervals_for_analysis(
                db, area_name, period_start, period_end
            )

            if not occupied_intervals:
                base_result.update(
                    {
                        "prob_given_true": None,
                        "prob_given_false": None,
                        "analysis_error": "no_occupied_intervals",
                    }
                )
                return base_result

            # Calculate total occupied and unoccupied durations
            # occupied_intervals are already timezone-aware UTC from get_occupied_intervals_for_analysis
            total_seconds_occupied = 0.0
            for occ_start, occ_end in occupied_intervals:
                # Intervals are already timezone-aware UTC, but normalize defensively
                occ_start_utc = to_utc(occ_start)
                occ_end_utc = to_utc(occ_end)
                # Clamp to analysis period
                clamped_start = max(occ_start_utc, period_start_utc)
                clamped_end = min(occ_end_utc, period_end_utc)
                if clamped_start < clamped_end:
                    total_seconds_occupied += (
                        clamped_end - clamped_start
                    ).total_seconds()
            total_period_seconds = (period_end_utc - period_start_utc).total_seconds()
            total_seconds_unoccupied = total_period_seconds - total_seconds_occupied

            if total_seconds_occupied <= 0:
                base_result.update(
                    {
                        "prob_given_true": None,
                        "prob_given_false": None,
                        "analysis_error": "no_occupied_time",
                    }
                )
                return base_result

            if total_seconds_unoccupied <= 0:
                base_result.update(
                    {
                        "prob_given_true": None,
                        "prob_given_false": None,
                        "analysis_error": "no_unoccupied_time",
                    }
                )
                return base_result

            # Get binary sensor intervals
            binary_intervals = (
                session.query(db.Intervals)
                .filter(
                    db.Intervals.entry_id == db.coordinator.entry_id,
                    db.Intervals.area_name == area_name,
                    db.Intervals.entity_id == entity_id,
                    db.Intervals.start_time < period_end_db,
                    db.Intervals.end_time > period_start_db,
                )
                .all()
            )

            if not binary_intervals:
                base_result.update(
                    {
                        "prob_given_true": None,
                        "prob_given_false": None,
                        "analysis_error": "no_sensor_data",
                    }
                )
                return base_result

            # Calculate overlap durations
            seconds_active_and_occupied = 0.0
            seconds_active_and_unoccupied = 0.0
            found_active_intervals = False
            unique_states = set()

            for interval in binary_intervals:
                interval_start = from_db_utc(interval.start_time)
                interval_end = from_db_utc(interval.end_time)
                # Clamp to analysis period
                clamped_start = max(interval_start, period_start_utc)
                clamped_end = min(interval_end, period_end_utc)
                interval_duration = (clamped_end - clamped_start).total_seconds()

                if interval_duration <= 0:
                    continue

                # Track unique states for diagnostic logging
                unique_states.add(interval.state)

                # Map binary state to semantic state if needed (e.g., 'off'/'on' → 'closed'/'open')
                mapped_state = map_binary_state_to_semantic(
                    interval.state, active_states
                )

                # Check if sensor is active (using mapped state)
                is_active = mapped_state in active_states

                if not is_active:
                    # Skip inactive intervals
                    continue

                found_active_intervals = True

                # Calculate overlap with occupied periods
                # occupied_intervals are already timezone-aware UTC from get_occupied_intervals_for_analysis
                occupied_overlap = 0.0
                for occ_start, occ_end in occupied_intervals:
                    occ_start_utc = to_utc(occ_start)
                    occ_end_utc = to_utc(occ_end)
                    # Calculate overlap
                    overlap_start = max(clamped_start, occ_start_utc)
                    overlap_end = min(clamped_end, occ_end_utc)
                    if overlap_start < overlap_end:
                        occupied_overlap += (
                            overlap_end - overlap_start
                        ).total_seconds()

                # Calculate unoccupied overlap: remainder of interval duration after occupied overlap
                # The entire clamped interval duration is either occupied or unoccupied (no gaps)
                # Clamp to ensure non-negative and not exceeding interval_duration (defensive check)
                unoccupied_overlap = max(
                    0.0, min(interval_duration - occupied_overlap, interval_duration)
                )

                # Accumulate active durations
                seconds_active_and_occupied += occupied_overlap
                seconds_active_and_unoccupied += unoccupied_overlap

            # Check if sensor has intervals but none are active
            if not found_active_intervals:
                # Sensor has intervals but none match active_states
                _LOGGER.debug(
                    "Sensor %s in area %s has %d intervals but none match active_states %s. "
                    "Found states: %s",
                    entity_id,
                    area_name,
                    len(binary_intervals),
                    active_states,
                    sorted(unique_states),
                )
                base_result.update(
                    {
                        "prob_given_true": None,
                        "prob_given_false": None,
                        "analysis_error": "no_active_intervals",
                    }
                )
                return base_result

            # Calculate probabilities
            prob_given_true = (
                seconds_active_and_occupied / total_seconds_occupied
                if total_seconds_occupied > 0
                else 0.0
            )
            prob_given_false = (
                seconds_active_and_unoccupied / total_seconds_unoccupied
                if total_seconds_unoccupied > 0
                else 0.0
            )

            # If sensor was never active during occupied periods, treat as insufficient data
            # This allows the entity to use type defaults instead of clamped low values
            if seconds_active_and_occupied == 0.0:
                if seconds_active_and_unoccupied > 0.0:
                    # Sensor was active but never during occupied periods
                    _LOGGER.debug(
                        "Sensor %s in area %s was active for %.1fs but never during occupied periods. "
                        "Found states: %s, active_states: %s. Using type defaults instead of clamped values.",
                        entity_id,
                        area_name,
                        seconds_active_and_unoccupied,
                        sorted(unique_states),
                        active_states,
                    )
                else:
                    # Sensor was never active at all (should have been caught earlier, but defensive check)
                    _LOGGER.debug(
                        "Sensor %s in area %s has active intervals but zero active duration. "
                        "This should have been caught earlier. Using type defaults.",
                        entity_id,
                        area_name,
                    )
                # Return error to use type defaults instead of clamped 0.05
                base_result.update(
                    {
                        "prob_given_true": None,
                        "prob_given_false": None,
                        "analysis_error": "no_active_during_occupied",
                    }
                )
                return base_result

            # Clamp probabilities to avoid "black hole" values
            prob_given_true = clamp_probability(
                prob_given_true, min_val=0.05, max_val=0.95
            )
            prob_given_false = clamp_probability(
                prob_given_false, min_val=0.05, max_val=0.95
            )

            _LOGGER.debug(
                "Binary likelihood analysis for %s in area %s: "
                "prob_given_true=%.3f, prob_given_false=%.3f, "
                "active_occupied=%.1fs, active_unoccupied=%.1fs, "
                "total_occupied=%.1fs, total_unoccupied=%.1fs",
                entity_id,
                area_name,
                prob_given_true,
                prob_given_false,
                seconds_active_and_occupied,
                seconds_active_and_unoccupied,
                total_seconds_occupied,
                total_seconds_unoccupied,
            )

            base_result.update(
                {
                    "prob_given_true": prob_given_true,
                    "prob_given_false": prob_given_false,
                    "analysis_error": None,
                }
            )
            return base_result

    except SQLAlchemyError as e:
        _LOGGER.error(
            "Database error analyzing binary likelihoods for %s: %s",
            entity_id,
            e,
        )
        return None
    except (ValueError, TypeError, RuntimeError, OSError) as e:
        _LOGGER.error("Error analyzing binary likelihoods for %s: %s", entity_id, e)
        return None


def analyze_correlation(  # noqa: C901
    db: AreaOccupancyDB,
    area_name: str,
    entity_id: str,
    analysis_period_days: int = 30,
    is_binary: bool = False,
    active_states: list[str] | None = None,
    input_type: InputType | None = None,
) -> dict[str, Any] | None:
    """Analyze correlation between sensor values and occupancy.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Sensor entity ID
        analysis_period_days: Number of days to analyze
        is_binary: Whether the entity is a binary sensor
        active_states: List of active states (required if is_binary is True)
        input_type: InputType of the sensor (e.g., InputType.HUMIDITY)

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
            period_start_utc = to_utc(period_start)
            period_end_utc = to_utc(period_end)
            period_start_db = to_db_utc(period_start_utc)
            period_end_db = to_db_utc(period_end_utc)

            # Base result structure with defaults for required fields
            base_result = {
                "entry_id": db.coordinator.entry_id,
                "area_name": area_name,
                "entity_id": entity_id,
                "input_type": input_type.value
                if input_type
                else InputType.UNKNOWN.value,
                "analysis_period_start": period_start,
                "analysis_period_end": period_end,
                "calculation_date": to_db_utc(dt_util.utcnow()),
                "correlation_coefficient": 0.0,
                "sample_count": 0,
                "correlation_type": CorrelationType.NONE,
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
                _LOGGER.debug(
                    "Converted intervals to %d samples for binary sensor %s in area %s",
                    len(samples),
                    entity_id,
                    area_name,
                )
            else:
                # Get numeric samples from both raw samples and hourly aggregates
                # Split period at retention cutoff to use appropriate data source
                retention_cutoff = period_end_utc - timedelta(
                    days=RETENTION_RAW_NUMERIC_SAMPLES_DAYS
                )

                # Recent period: Use raw NumericSamples (within retention period)
                recent_start = max(period_start_utc, retention_cutoff)
                recent_samples = (
                    session.query(db.NumericSamples)
                    .filter(
                        db.NumericSamples.entry_id == db.coordinator.entry_id,
                        db.NumericSamples.area_name == area_name,
                        db.NumericSamples.entity_id == entity_id,
                        db.NumericSamples.timestamp >= to_db_utc(recent_start),
                        db.NumericSamples.timestamp <= period_end_db,
                    )
                    .order_by(db.NumericSamples.timestamp)
                    .all()
                )

                # Historical period: Use hourly NumericAggregates (older than retention)
                aggregate_samples: list[SimpleSample] = []
                if period_start_utc < retention_cutoff:
                    historical_end = min(period_end_utc, retention_cutoff)
                    aggregate_samples = convert_hourly_aggregates_to_samples(
                        db,
                        area_name,
                        entity_id,
                        period_start_utc,
                        historical_end,
                        session,
                    )

                # Combine both sources
                # Convert raw samples to SimpleSample objects for consistency
                all_samples: list[SimpleSample] = [
                    SimpleSample(from_db_utc(sample.timestamp), float(sample.value))
                    for sample in recent_samples
                ]
                all_samples.extend(aggregate_samples)

                # Sort by timestamp to ensure chronological order
                all_samples.sort(key=lambda s: s.timestamp)

                # Use combined samples for correlation analysis
                samples = all_samples

            if error := validate_sample_count(samples):
                base_result.update(error)
                return base_result

            # Get occupied intervals for the area
            occupied_intervals = get_occupied_intervals_for_analysis(
                db, area_name, period_start, period_end
            )

            # Initialize diagnostic counters (used for both binary and numeric)
            intervals_overlapping_occupied = 0
            intervals_overlapping_unoccupied = 0

            # For binary sensors, use time-based chunking to avoid duplicate samples
            if is_binary:
                # Query intervals to calculate time-based chunked samples
                binary_intervals = (
                    session.query(db.Intervals)
                    .filter(
                        db.Intervals.entry_id == db.coordinator.entry_id,
                        db.Intervals.area_name == area_name,
                        db.Intervals.entity_id == entity_id,
                        db.Intervals.start_time < period_end_db,
                        db.Intervals.end_time > period_start_db,
                    )
                    .all()
                )

                sample_values = []
                occupancy_flags = []
                samples_in_occupied = 0
                samples_in_unoccupied = 0
                active_samples_in_occupied = 0
                active_samples_in_unoccupied = 0
                inactive_samples_in_occupied = 0
                inactive_samples_in_unoccupied = 0

                # Time chunk granularity: 60 seconds (1 minute)
                # This ensures proper weighting and avoids duplicate samples
                chunk_duration_seconds = 60.0

                for interval in binary_intervals:
                    interval_start = from_db_utc(interval.start_time)
                    interval_end = from_db_utc(interval.end_time)
                    # Clamp to analysis period
                    clamped_start = max(interval_start, period_start_utc)
                    clamped_end = min(interval_end, period_end_utc)
                    interval_duration = (clamped_end - clamped_start).total_seconds()

                    if interval_duration <= 0:
                        continue

                    # Determine value based on state
                    is_active = interval.state in active_states
                    value = 1.0 if is_active else 0.0

                    # Subdivide interval into time chunks
                    # Each chunk is assigned to either occupied or unoccupied
                    current_time = clamped_start
                    chunks_in_occupied = 0
                    chunks_in_unoccupied = 0

                    while current_time < clamped_end:
                        # Calculate chunk end (don't exceed interval end)
                        chunk_end = min(
                            current_time + timedelta(seconds=chunk_duration_seconds),
                            clamped_end,
                        )

                        # Use chunk midpoint to determine occupancy
                        chunk_midpoint = current_time + (chunk_end - current_time) / 2

                        # Check if chunk midpoint falls within any occupied interval
                        is_occupied = is_timestamp_occupied(
                            chunk_midpoint, occupied_intervals
                        )

                        # Create one sample per chunk with unambiguous occupancy flag
                        sample_values.append(value)
                        occupancy_flags.append(1.0 if is_occupied else 0.0)

                        if is_occupied:
                            samples_in_occupied += 1
                            chunks_in_occupied += 1
                            if value == 1.0:
                                active_samples_in_occupied += 1
                            else:
                                inactive_samples_in_occupied += 1
                        else:
                            samples_in_unoccupied += 1
                            chunks_in_unoccupied += 1
                            if value == 1.0:
                                active_samples_in_unoccupied += 1
                            else:
                                inactive_samples_in_unoccupied += 1

                        # Move to next chunk
                        current_time = chunk_end

                    # Track interval overlap for diagnostics
                    if chunks_in_occupied > 0:
                        intervals_overlapping_occupied += 1
                    if chunks_in_unoccupied > 0:
                        intervals_overlapping_unoccupied += 1
            else:
                # For numeric sensors, use existing midpoint-based approach
                sample_values: list[float] = []
                occupancy_flags: list[
                    float
                ] = []  # 1.0 for occupied, 0.0 for unoccupied

                # Diagnostic counters
                samples_in_occupied = 0
                samples_in_unoccupied = 0
                active_samples_in_occupied = 0
                active_samples_in_unoccupied = 0
                inactive_samples_in_occupied = 0
                inactive_samples_in_unoccupied = 0

                for sample in samples:
                    # Check if sample timestamp falls within any occupied interval
                    is_occupied = is_timestamp_occupied(
                        sample.timestamp, occupied_intervals
                    )

                    sample_value = float(sample.value)
                    sample_values.append(sample_value)
                    occupancy_flags.append(1.0 if is_occupied else 0.0)

                    if is_occupied:
                        samples_in_occupied += 1
                        if (
                            sample_value > 0.5
                        ):  # For numeric, consider > 0.5 as "active"
                            active_samples_in_occupied += 1
                        else:
                            inactive_samples_in_occupied += 1
                    else:
                        samples_in_unoccupied += 1
                        if sample_value > 0.5:
                            active_samples_in_unoccupied += 1
                        else:
                            inactive_samples_in_unoccupied += 1

            if error := validate_sample_count(
                sample_values, error_type="too_few_samples_after_filtering"
            ):
                base_result.update(error)
                return base_result

            # Log diagnostic information for binary sensors
            if is_binary:
                _LOGGER.debug(
                    "Binary sensor correlation diagnostics for %s in area %s: "
                    "total_samples=%d, samples_in_occupied=%d, samples_in_unoccupied=%d, "
                    "active_in_occupied=%d, active_in_unoccupied=%d, "
                    "inactive_in_occupied=%d, inactive_in_unoccupied=%d, "
                    "intervals_overlapping_occupied=%d, intervals_overlapping_unoccupied=%d",
                    entity_id,
                    area_name,
                    len(sample_values),
                    samples_in_occupied,
                    samples_in_unoccupied,
                    active_samples_in_occupied,
                    active_samples_in_unoccupied,
                    inactive_samples_in_occupied,
                    inactive_samples_in_unoccupied,
                    intervals_overlapping_occupied,
                    intervals_overlapping_unoccupied,
                )

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

            # Log detailed statistics for binary sensors
            if is_binary:
                _LOGGER.debug(
                    "Binary sensor correlation statistics for %s in area %s: "
                    "correlation=%.3f, mean_occupied=%.3f, mean_unoccupied=%.3f, "
                    "std_occupied=%.3f, std_unoccupied=%.3f, "
                    "occupied_samples=%d, unoccupied_samples=%d",
                    entity_id,
                    area_name,
                    correlation,
                    mean_occupied or 0.0,
                    mean_unoccupied or 0.0,
                    std_occupied or 0.0,
                    std_unoccupied or 0.0,
                    len(occupied_values),
                    len(unoccupied_values),
                )

            # Clamp std dev for binary sensors to avoid numerical issues
            if is_binary:
                if std_occupied is not None:
                    std_occupied = max(0.05, min(0.95, std_occupied))
                if std_unoccupied is not None:
                    std_unoccupied = max(0.05, min(0.95, std_unoccupied))

            # Determine correlation type
            abs_correlation = abs(correlation)
            correlation_type: str = CorrelationType.NONE
            analysis_error = None

            if abs_correlation >= CORRELATION_MODERATE_THRESHOLD:
                # Strong correlation (>= 0.4)
                if correlation > 0:
                    correlation_type = CorrelationType.STRONG_POSITIVE
                else:
                    correlation_type = CorrelationType.STRONG_NEGATIVE
            elif abs_correlation >= CORRELATION_WEAK_THRESHOLD:
                # Weak correlation (0.15 to 0.4)
                if correlation > 0:
                    correlation_type = CorrelationType.POSITIVE
                else:
                    correlation_type = CorrelationType.NEGATIVE
            else:
                # Very weak correlation (< 0.15) - no meaningful correlation
                correlation_type = CorrelationType.NONE
                analysis_error = "no_correlation"

            # Calculate confidence (based on correlation strength and sample size)
            # Confidence increases with stronger correlation and more samples
            sample_count = len(sample_values)
            # Defensive check: avoid division by zero (should not happen due to validation above)
            if sample_count == 0:
                base_result.update(
                    {
                        "sample_count": 0,
                        "analysis_error": "zero_samples_after_filtering",
                    }
                )
                return base_result
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
                "input_type": input_type.value
                if input_type
                else InputType.UNKNOWN.value,
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

            _LOGGER.debug(
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


def save_binary_likelihood_result(
    db: AreaOccupancyDB,
    likelihood_data: dict[str, Any],
    input_type: InputType | None = None,
) -> bool:
    """Save binary likelihood analysis result to database.

    Binary likelihood results are stored in the Correlations table with
    correlation_type="binary_likelihood" to distinguish them from numeric correlations.

    Args:
        db: Database instance
        likelihood_data: Binary likelihood analysis result dictionary
        input_type: InputType of the sensor (e.g., InputType.MEDIA, InputType.APPLIANCE)

    Returns:
        True if saved successfully, False otherwise
    """
    _LOGGER.debug(
        "Saving binary likelihood result for %s in area %s",
        likelihood_data["entity_id"],
        likelihood_data["area_name"],
    )

    try:
        # Convert binary likelihood data to correlation table format
        # Use correlation_type="binary_likelihood" to distinguish from numeric correlations
        input_type_value = (
            input_type.value
            if input_type
            else likelihood_data.get("input_type", "unknown")
        )
        correlation_data = {
            "entry_id": likelihood_data["entry_id"],
            "area_name": likelihood_data["area_name"],
            "entity_id": likelihood_data["entity_id"],
            "input_type": input_type_value,
            "correlation_coefficient": 0.0,  # Binary sensors don't have correlation
            "correlation_type": CorrelationType.BINARY_LIKELIHOOD,
            "analysis_period_start": to_db_utc(
                likelihood_data["analysis_period_start"]
            ),
            "analysis_period_end": to_db_utc(likelihood_data["analysis_period_end"]),
            "sample_count": likelihood_data.get(
                "sample_count", 0
            ),  # Binary sensors don't use samples
            "confidence": None,  # Binary sensors don't have confidence scores
            # Store prob_given_true and prob_given_false in mean fields
            "mean_value_when_occupied": likelihood_data.get("prob_given_true"),
            "mean_value_when_unoccupied": likelihood_data.get("prob_given_false"),
            "std_dev_when_occupied": None,  # Binary sensors don't have std dev
            "std_dev_when_unoccupied": None,
            "threshold_active": None,
            "threshold_inactive": None,
            "analysis_error": likelihood_data.get("analysis_error"),
            "calculation_date": to_db_utc(
                likelihood_data.get("calculation_date", dt_util.utcnow())
            ),
        }

        with db.get_session() as session:
            # Check if correlation already exists for this period
            existing = (
                session.query(db.Correlations)
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
                existing.updated_at = to_db_utc(dt_util.utcnow())
            else:
                # Create new record
                correlation = db.Correlations(**correlation_data)
                session.add(correlation)

            session.commit()

            # Prune old correlation results (keep only last N)
            _prune_old_correlations(
                db,
                session,
                correlation_data["area_name"],
                correlation_data["entity_id"],
            )

            _LOGGER.debug("Binary likelihood result saved successfully")
            return True

    except (
        SQLAlchemyError,
        ValueError,
        TypeError,
        RuntimeError,
        OSError,
    ) as e:
        _LOGGER.error("Error saving binary likelihood result: %s", e)
        return False


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

    # Ensure input_type is set (required field)
    if "input_type" not in correlation_data or correlation_data["input_type"] is None:
        correlation_data["input_type"] = InputType.UNKNOWN.value

    try:
        # Normalize datetime fields for DB persistence (naive UTC)
        if (
            "analysis_period_start" in correlation_data
            and correlation_data["analysis_period_start"] is not None
        ):
            correlation_data["analysis_period_start"] = to_db_utc(
                correlation_data["analysis_period_start"]
            )
        if (
            "analysis_period_end" in correlation_data
            and correlation_data["analysis_period_end"] is not None
        ):
            correlation_data["analysis_period_end"] = to_db_utc(
                correlation_data["analysis_period_end"]
            )
        if (
            "calculation_date" in correlation_data
            and correlation_data["calculation_date"] is not None
        ):
            correlation_data["calculation_date"] = to_db_utc(
                correlation_data["calculation_date"]
            )

        with db.get_session() as session:
            # Check if correlation already exists for this period
            existing = (
                session.query(db.Correlations)
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
                existing.updated_at = to_db_utc(dt_util.utcnow())
            else:
                # Create new record
                # Ensure calculation_date is set if not provided
                if "calculation_date" not in correlation_data:
                    correlation_data["calculation_date"] = to_db_utc(dt_util.utcnow())
                correlation = db.Correlations(**correlation_data)
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
        return False


def _prune_old_correlations(
    db: AreaOccupancyDB,
    session: Any,
    area_name: str,
    entity_id: str,
) -> None:
    """Prune old correlation results, keeping only the most recent N months.

    Args:
        db: Database instance
        session: Database session
        area_name: Area name
        entity_id: Entity ID
    """
    try:
        # Get all correlations for this entity, ordered by analysis_period_start
        correlations = (
            session.query(db.Correlations)
            .filter_by(area_name=area_name, entity_id=entity_id)
            .order_by(db.Correlations.analysis_period_start.desc())
            .all()
        )

        if not correlations:
            return

        # Group correlations by year-month
        monthly_correlations: dict[tuple[int, int], list[Any]] = defaultdict(list)
        for corr in correlations:
            # Group by local wall-clock month (policy decision)
            period_start_local = to_local(from_db_utc(corr.analysis_period_start))
            month_key = (period_start_local.year, period_start_local.month)
            monthly_correlations[month_key].append(corr)

        # For each month, keep only the most recent record (by calculation_date)
        # Then keep only the last N months
        months_to_keep = []
        duplicates_deleted = False
        for month_key in sorted(monthly_correlations.keys(), reverse=True):
            month_corrs = monthly_correlations[month_key]
            # Keep the most recent record for this month
            most_recent = max(
                month_corrs, key=lambda c: from_db_utc(c.calculation_date)
            )
            months_to_keep.append(most_recent)
            # Delete other records for this month
            for corr in month_corrs:
                if corr.id != most_recent.id:
                    session.delete(corr)
                    duplicates_deleted = True

        # Commit within-month duplicate deletions before proceeding to pruning
        if duplicates_deleted:
            session.commit()

        # Keep only the last N months
        if len(months_to_keep) > CORRELATION_MONTHS_TO_KEEP:
            to_delete = months_to_keep[CORRELATION_MONTHS_TO_KEEP:]
            for correlation in to_delete:
                session.delete(correlation)
            session.commit()
            _LOGGER.debug(
                "Pruned %d old correlation results for %s (kept %d months)",
                len(to_delete),
                entity_id,
                CORRELATION_MONTHS_TO_KEEP,
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
    input_type: InputType | None = None,
) -> dict[str, Any] | None:
    """Analyze and save correlation for a sensor (numeric or binary).

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Sensor entity ID
        analysis_period_days: Number of days to analyze
        is_binary: Whether the entity is a binary sensor
        active_states: List of active states (required if is_binary is True)
        input_type: InputType of the sensor (e.g., InputType.HUMIDITY)

    Returns:
        Correlation data if analysis completed and saved, None otherwise
    """
    correlation_data = analyze_correlation(
        db,
        area_name,
        entity_id,
        analysis_period_days,
        is_binary,
        active_states,
        input_type,
    )

    if correlation_data is None:
        _LOGGER.debug(
            "No correlation data generated for %s in area %s",
            entity_id,
            area_name,
        )
        return None

    # Save correlation results even when they have errors, so analysis_error is preserved
    # This allows failed analyses to be restored after entity reload
    analysis_error = correlation_data.get("analysis_error")
    correlation_coefficient = correlation_data.get("correlation_coefficient")

    # Only skip saving if correlation_coefficient is invalid (None, NaN, or infinite)
    # and there's no analysis_error to preserve
    if analysis_error is None:
        # For successful analyses, validate correlation_coefficient
        if correlation_coefficient is None or not np.isfinite(correlation_coefficient):
            _LOGGER.debug(
                "Skipping save of correlation with invalid coefficient for %s in area %s",
                entity_id,
                area_name,
            )
            return None
        # Save all valid correlations (including weak correlations)
        # Weak correlations (0.15-0.4) are now saved and used for occupancy detection
    else:
        # For failed analyses, save the error even if correlation_coefficient is invalid
        # Use a placeholder value for correlation_coefficient if it's invalid
        if correlation_coefficient is None or not np.isfinite(correlation_coefficient):
            correlation_data["correlation_coefficient"] = 0.0
        # Ensure sample_count is set (required field)
        if correlation_data.get("sample_count") is None:
            correlation_data["sample_count"] = 0
        _LOGGER.debug(
            "Saving correlation result with analysis_error for %s in area %s: error=%s",
            entity_id,
            area_name,
            analysis_error,
        )

    if save_correlation_result(db, correlation_data):
        return correlation_data
    return None


def get_entity_correlations(db: AreaOccupancyDB, area_name: str) -> dict[str, float]:
    """Get normalized correlation strengths for all entities in an area.

    Queries the correlations table and returns a dict mapping entity_id to
    correlation strength (0.0-1.0) for use in sigmoid probability calculation.
    Only entities with positive correlations and sufficient data are included.

    Args:
        db: Database instance
        area_name: Area name to query correlations for

    Returns:
        Dict mapping entity_id to correlation strength (0.0-1.0).
        Entities with insufficient data or negative/no correlation return 0.0
        and are excluded from the result.
    """
    try:
        with db.get_session() as session:
            # Query all correlations for this area, newest first
            correlations = (
                session.query(db.Correlations)
                .filter(
                    db.Correlations.entry_id == db.coordinator.entry_id,
                    db.Correlations.area_name == area_name,
                )
                .order_by(db.Correlations.calculation_date.desc())
                .all()
            )

            result: dict[str, float] = {}
            for corr in correlations:
                # Only accept the first (newest) row per entity
                if corr.entity_id in result:
                    continue

                # Get correlation coefficient (may be None for binary likelihoods)
                coef = corr.correlation_coefficient
                if coef is None:
                    # Binary likelihoods don't have correlation coefficient
                    # Use mean_value_when_occupied (prob_given_true) as proxy
                    # Normalize: higher prob_given_true = stronger correlation
                    prob_true = corr.mean_value_when_occupied
                    if prob_true is not None and prob_true > 0:
                        # Scale prob_given_true to 0-1 correlation strength
                        # 0.5 -> 0.5, 0.95 -> 0.95
                        result[corr.entity_id] = min(1.0, max(0.0, float(prob_true)))
                    continue

                # Skip if insufficient samples for reliability
                if corr.sample_count < MIN_CORRELATION_SAMPLES:
                    continue

                # Only use positive correlations with sufficient confidence
                if coef > 0:
                    # Normalize to 0-1 range
                    # Correlation coefficient is typically -1 to 1
                    result[corr.entity_id] = min(1.0, max(0.0, float(coef)))

            return result

    except (
        SQLAlchemyError,
        OSError,
        RuntimeError,
    ) as err:
        _LOGGER.warning(
            "Error getting entity correlations for area %s: %s", area_name, err
        )
        return {}


def get_correlation_for_entity(
    db: AreaOccupancyDB, area_name: str, entity_id: str
) -> dict[str, Any] | None:
    """Get the most recent correlation result for an entity.

    Gets the current month's correlation record (by analysis_period_start) to ensure
    we get the accumulating monthly record even if it was calculated earlier in the month.

    Args:
        db: Database instance
        area_name: Area name
        entity_id: Entity ID

    Returns:
        Most recent correlation result as dictionary, or None if not found
    """
    try:
        with db.get_session() as session:
            # Get current month's period_start
            now_local = to_local(dt_util.utcnow())
            current_month_start_local = now_local.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            current_month_start_db = to_db_utc(current_month_start_local)

            # Try to get current month's record first
            correlation = (
                session.query(db.Correlations)
                .filter_by(
                    area_name=area_name,
                    entity_id=entity_id,
                    analysis_period_start=current_month_start_db,
                )
                .first()
            )

            # If not found, fall back to most recent by calculation_date
            if not correlation:
                correlation = (
                    session.query(db.Correlations)
                    .filter_by(area_name=area_name, entity_id=entity_id)
                    .order_by(db.Correlations.calculation_date.desc())
                    .first()
                )

            if correlation:
                return {
                    "correlation_coefficient": correlation.correlation_coefficient,
                    "correlation_type": correlation.correlation_type,
                    "confidence": correlation.confidence,
                    "mean_value_when_occupied": correlation.mean_value_when_occupied,
                    "mean_value_when_unoccupied": correlation.mean_value_when_unoccupied,
                    "std_dev_when_occupied": correlation.std_dev_when_occupied,
                    "std_dev_when_unoccupied": correlation.std_dev_when_unoccupied,
                    "threshold_active": correlation.threshold_active,
                    "threshold_inactive": correlation.threshold_inactive,
                    "analysis_error": correlation.analysis_error,
                    "calculation_date": from_db_utc(correlation.calculation_date),
                    "input_type": getattr(correlation, "input_type", None),
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


def get_correlatable_entities_by_area(
    coordinator: AreaOccupancyCoordinator,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return mapping of area_name to correlatable entities with metadata.

    Args:
        coordinator: The coordinator instance containing areas

    Returns:
        Dict mapping area_name to dict of entity_id -> {
            'is_binary': bool,
            'active_states': list[str] | None
        }
    """
    # Binary sensors that should be analyzed (excluding MOTION)
    binary_inputs = {
        InputType.MEDIA,
        InputType.APPLIANCE,
        InputType.DOOR,
        InputType.WINDOW,
    }

    # Numeric sensors
    numeric_inputs = {
        InputType.TEMPERATURE,
        InputType.HUMIDITY,
        InputType.ILLUMINANCE,
        InputType.ENVIRONMENTAL,
        InputType.CO2,
        InputType.CO,
        InputType.POWER,
        InputType.SOUND_PRESSURE,
        InputType.PRESSURE,
        InputType.AIR_QUALITY,
        InputType.VOC,
        InputType.PM25,
        InputType.PM10,
    }

    correlatable_entities: dict[str, dict[str, dict[str, Any]]] = {}

    for area_name, area in coordinator.areas.items():
        entities_container = getattr(area.entities, "entities", {})
        area_entities: dict[str, dict[str, Any]] = {}

        for entity_id, entity in entities_container.items():
            entity_type = getattr(entity, "type", None)
            input_type = getattr(entity_type, "input_type", None)

            if input_type in binary_inputs:
                # Binary sensor - get active_states
                active_states = getattr(entity_type, "active_states", None)
                if active_states:
                    area_entities[entity_id] = {
                        "is_binary": True,
                        "active_states": active_states,
                        "input_type": input_type,
                    }
            elif input_type in numeric_inputs:
                # Numeric sensor
                area_entities[entity_id] = {
                    "is_binary": False,
                    "active_states": None,
                    "input_type": input_type,
                }

        if area_entities:
            correlatable_entities[area_name] = area_entities

    return correlatable_entities


async def run_correlation_analysis(
    coordinator: AreaOccupancyCoordinator,
    return_results: bool = False,
) -> list[dict[str, Any]] | None:
    """Run correlation analysis for numeric sensors and binary likelihood analysis.

    Numeric sensors use correlation analysis (Gaussian PDF).
    Binary sensors use duration-based probability calculation (static values).
    Requires OccupiedIntervalsCache to be populated.

    Args:
        coordinator: The coordinator instance containing areas and database
        return_results: If True, returns a list of correlation results for summary

    Returns:
        List of correlation result dictionaries if return_results is True, None otherwise.
        Each dictionary contains: area, entity_id, type, success, and optionally error.
    """
    correlatable_entities = get_correlatable_entities_by_area(coordinator)
    results: list[dict[str, Any]] = []

    if correlatable_entities:
        _LOGGER.debug(
            "Starting sensor analysis for %d area(s)",
            len(correlatable_entities),
        )
        for area_name, entities in correlatable_entities.items():
            for entity_id, entity_info in entities.items():
                try:
                    if entity_info["is_binary"]:
                        # Binary sensors: Use duration-based likelihood analysis
                        likelihood_result = (
                            await coordinator.hass.async_add_executor_job(
                                coordinator.db.analyze_binary_likelihoods,
                                area_name,
                                entity_id,
                                30,  # analysis_period_days
                                entity_info["active_states"],
                            )
                        )

                        # Save binary likelihood results to database (including errors)
                        if likelihood_result:
                            input_type = entity_info.get("input_type")
                            await coordinator.hass.async_add_executor_job(
                                save_binary_likelihood_result,
                                coordinator.db,
                                likelihood_result,
                                input_type,
                            )

                        # Apply analysis results to live entities immediately
                        if likelihood_result and area_name in coordinator.areas:
                            area = coordinator.areas[area_name]
                            try:
                                entity = area.entities.get_entity(entity_id)
                                entity.update_binary_likelihoods(likelihood_result)
                            except ValueError as e:
                                # Entity might have been removed during analysis
                                _LOGGER.debug(
                                    "Entity %s in area %s no longer exists: %s",
                                    entity_id,
                                    area_name,
                                    e,
                                )

                        # Track result if requested
                        if return_results:
                            results.append(
                                {
                                    "area": area_name,
                                    "entity_id": entity_id,
                                    "type": "binary_likelihood",
                                    "success": bool(likelihood_result),
                                }
                            )
                    else:
                        # Numeric sensors: Use correlation analysis
                        correlation_result = (
                            await coordinator.hass.async_add_executor_job(
                                coordinator.db.analyze_and_save_correlation,
                                area_name,
                                entity_id,
                                30,  # analysis_period_days
                                False,  # is_binary
                                None,  # active_states (not used for numeric)
                                entity_info.get("input_type"),  # Pass input_type
                            )
                        )

                        # Apply analysis results to live entities immediately
                        if correlation_result and area_name in coordinator.areas:
                            area = coordinator.areas[area_name]
                            try:
                                entity = area.entities.get_entity(entity_id)
                                entity.update_correlation(correlation_result)
                            except ValueError as e:
                                # Entity might have been removed during analysis
                                _LOGGER.debug(
                                    "Entity %s in area %s no longer exists: %s",
                                    entity_id,
                                    area_name,
                                    e,
                                )

                        # Track result if requested
                        if return_results:
                            results.append(
                                {
                                    "area": area_name,
                                    "entity_id": entity_id,
                                    "type": "correlation",
                                    "success": bool(correlation_result),
                                }
                            )
                except (
                    SQLAlchemyError,
                    ValueError,
                    TypeError,
                    RuntimeError,
                    OSError,
                ) as err:
                    _LOGGER.error(
                        "Sensor analysis failed for %s (%s): %s",
                        area_name,
                        entity_id,
                        err,
                    )
                    # Track error result if requested
                    if return_results:
                        results.append(
                            {
                                "area": area_name,
                                "entity_id": entity_id,
                                "success": False,
                                "error": str(err),
                            }
                        )
    else:
        _LOGGER.debug("Skipping sensor analysis - no correlatable entities configured")

    return results if return_results else None
