"""Pattern analysis for Area Occupancy Detection."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from homeassistant.util import dt as dt_util

from .. import SensorState

_LOGGER = logging.getLogger(__name__)


@dataclass
class TimeSlotData:
    """Data structure for time slot occupancy patterns."""

    total_samples: int = 0
    occupied_samples: int = 0
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=dt_util.utcnow)

    def update(self, is_occupied: bool) -> None:
        """Update time slot statistics."""
        self.total_samples += 1
        if is_occupied:
            self.occupied_samples += 1
        self.confidence = min(self.total_samples / 100, 1.0)
        self.last_updated = dt_util.utcnow()

    @property
    def occupancy_rate(self) -> float:
        """Calculate occupancy rate for this time slot."""
        return (
            self.occupied_samples / self.total_samples
            if self.total_samples > 0
            else 0.0
        )


@dataclass
class PatternData:
    """Container for pattern analysis data."""

    daily_patterns: Dict[str, TimeSlotData] = field(default_factory=dict)
    weekly_patterns: Dict[str, Dict[str, TimeSlotData]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    recent_states: deque[Tuple[datetime, bool]] = field(
        default_factory=lambda: deque(maxlen=288)
    )  # 24 hours worth of 5-minute samples
    pattern_changes: List[Tuple[datetime, float]] = field(default_factory=list)
    last_analysis: datetime = field(default_factory=dt_util.utcnow)


class PatternAnalyzer:
    """Analyzes occupancy patterns over time."""

    def __init__(
        self,
        slot_minutes: int = 30,
        min_confidence: float = 0.3,
        max_pattern_age: int = 30,
    ):
        """Initialize the pattern analyzer."""
        self.slot_minutes = slot_minutes
        self.min_confidence = min_confidence
        self.max_pattern_age = max_pattern_age
        self._patterns: Dict[str, PatternData] = {}
        self._change_threshold = 0.2
        self._cache: Dict[str, Tuple[float, float, datetime]] = {}
        self._cache_duration = timedelta(minutes=5)

    def calculate(self, sensor_states: dict[str, SensorState]) -> float:
        """Calculate probability adjustment based on patterns."""
        area_id = self._get_area_id(sensor_states)
        if not area_id or area_id not in self._patterns:
            return 0.0

        # Check cache
        if self._is_cache_valid(area_id):
            return self._cache[area_id][0]

        # Calculate new probability
        probability, confidence = self._calculate_pattern_probability(area_id)

        # Cache result
        self._cache[area_id] = (probability, confidence, dt_util.utcnow())

        return probability if confidence >= self.min_confidence else 0.0

    def update_pattern(
        self, area_id: str, timestamp: datetime, is_occupied: bool
    ) -> None:
        """Update pattern data with new occupancy information."""
        if area_id not in self._patterns:
            self._patterns[area_id] = PatternData()

        pattern = self._patterns[area_id]
        time_slot = self._get_time_slot(timestamp)
        day_name = timestamp.strftime("%A").lower()

        # Update daily pattern
        if time_slot not in pattern.daily_patterns:
            pattern.daily_patterns[time_slot] = TimeSlotData()
        pattern.daily_patterns[time_slot].update(is_occupied)

        # Update weekly pattern
        if time_slot not in pattern.weekly_patterns[day_name]:
            pattern.weekly_patterns[day_name][time_slot] = TimeSlotData()
        pattern.weekly_patterns[day_name][time_slot].update(is_occupied)

        # Update recent states
        pattern.recent_states.append((timestamp, is_occupied))

        # Detect pattern changes
        self._detect_pattern_changes(area_id, timestamp)

        # Invalidate cache
        if area_id in self._cache:
            del self._cache[area_id]

    def _calculate_pattern_probability(self, area_id: str) -> Tuple[float, float]:
        """Calculate probability based on historical patterns."""
        pattern = self._patterns[area_id]
        current_time = dt_util.utcnow()
        time_slot = self._get_time_slot(current_time)
        day_name = current_time.strftime("%A").lower()

        # Get probabilities from patterns
        daily_prob, daily_conf = self._get_slot_probability(
            pattern.daily_patterns.get(time_slot)
        )
        weekly_prob, weekly_conf = self._get_slot_probability(
            pattern.weekly_patterns[day_name].get(time_slot)
        )

        # Calculate trend adjustment
        trend_adjustment = self._calculate_trend_adjustment(pattern.recent_states)

        # Combine probabilities based on confidence
        if daily_conf >= self.min_confidence and weekly_conf >= self.min_confidence:
            combined_prob = (daily_prob * daily_conf + weekly_prob * weekly_conf) / (
                daily_conf + weekly_conf
            )
            combined_conf = (daily_conf + weekly_conf) / 2
        elif daily_conf >= self.min_confidence:
            combined_prob = daily_prob
            combined_conf = daily_conf
        elif weekly_conf >= self.min_confidence:
            combined_prob = weekly_prob
            combined_conf = weekly_conf
        else:
            return 0.0, 0.0

        # Apply trend adjustment
        final_prob = max(0.0, min(1.0, combined_prob + trend_adjustment))

        return final_prob, combined_conf

    def _get_slot_probability(
        self, slot_data: TimeSlotData | None
    ) -> Tuple[float, float]:
        """Get probability and confidence from time slot data."""
        if not slot_data or slot_data.total_samples == 0:
            return 0.0, 0.0

        return slot_data.occupancy_rate, slot_data.confidence

    def _calculate_trend_adjustment(
        self, recent_states: deque[Tuple[datetime, bool]]
    ) -> float:
        """Calculate probability adjustment based on recent trends."""
        if len(recent_states) < 2:
            return 0.0

        # Calculate short-term trend (last hour)
        short_term = self._calculate_period_trend(recent_states, timedelta(hours=1))

        # Calculate medium-term trend (last 4 hours)
        medium_term = self._calculate_period_trend(recent_states, timedelta(hours=4))

        # Combine trends with different weights
        return (short_term * 0.7 + medium_term * 0.3) * 0.1  # Max 10% adjustment

    def _calculate_period_trend(
        self, states: deque[Tuple[datetime, bool]], period: timedelta
    ) -> float:
        """Calculate trend for a specific time period."""
        cutoff = dt_util.utcnow() - period
        relevant_states = [state for time, state in states if time >= cutoff]

        if not relevant_states:
            return 0.0

        # Calculate trend direction and strength
        occupied_ratio = sum(1 for state in relevant_states if state) / len(
            relevant_states
        )
        trend = occupied_ratio - 0.5  # -0.5 to 0.5 range
        return trend * 2  # Scale to -1.0 to 1.0 range

    def _detect_pattern_changes(self, area_id: str, timestamp: datetime) -> None:
        """Detect significant changes in occupancy patterns."""
        pattern = self._patterns[area_id]

        # Only analyze once per day
        if (timestamp - pattern.last_analysis) < timedelta(days=1):
            return

        changes = []
        week_ago = timestamp - timedelta(days=7)

        # Compare patterns
        for time_slot, current_data in pattern.daily_patterns.items():
            if current_data.last_updated < week_ago:
                continue

            historical_prob = self._get_historical_probability(
                area_id, time_slot, week_ago
            )
            current_prob = current_data.occupancy_rate

            if abs(current_prob - historical_prob) > self._change_threshold:
                changes.append((timestamp, current_prob - historical_prob))

        if changes:
            # Keep only recent changes
            month_ago = timestamp - timedelta(days=self.max_pattern_age)
            pattern.pattern_changes = [
                (t, c) for t, c in pattern.pattern_changes if t > month_ago
            ] + changes

        pattern.last_analysis = timestamp

    def _get_historical_probability(
        self, area_id: str, time_slot: str, before_date: datetime
    ) -> float:
        """Get historical probability for a time slot."""
        if area_id not in self._patterns:
            return 0.0

        pattern = self._patterns[area_id]
        slot_data = pattern.daily_patterns.get(time_slot)

        if not slot_data or slot_data.last_updated >= before_date:
            return 0.0

        return slot_data.occupancy_rate

    def _get_time_slot(self, timestamp: datetime) -> str:
        """Convert timestamp to time slot string."""
        minutes = timestamp.hour * 60 + timestamp.minute
        slot = (minutes // self.slot_minutes) * self.slot_minutes
        return f"{slot // 60:02d}:{slot % 60:02d}"

    def _is_cache_valid(self, area_id: str) -> bool:
        """Check if cached result is still valid."""
        if area_id not in self._cache:
            return False

        cache_time = self._cache[area_id][2]
        return dt_util.utcnow() - cache_time < self._cache_duration

    def _get_area_id(self, sensor_states: dict[str, SensorState]) -> str | None:
        """Extract area ID from sensor states."""
        for entity_id in sensor_states:
            parts = entity_id.split(".")
            if len(parts) == 2:
                return parts[0]
        return None

    def clear_old_data(self, area_id: str | None = None) -> None:
        """Clear pattern data older than max_pattern_age."""
        cutoff = dt_util.utcnow() - timedelta(days=self.max_pattern_age)

        areas = [area_id] if area_id else list(self._patterns.keys())

        for current_area in areas:
            if current_area not in self._patterns:
                continue

            pattern = self._patterns[current_area]

            # Clear old daily patterns
            pattern.daily_patterns = {
                slot: data
                for slot, data in pattern.daily_patterns.items()
                if data.last_updated >= cutoff
            }

            # Clear old weekly patterns
            for day in pattern.weekly_patterns:
                pattern.weekly_patterns[day] = {
                    slot: data
                    for slot, data in pattern.weekly_patterns[day].items()
                    if data.last_updated >= cutoff
                }

            # Clear old pattern changes
            pattern.pattern_changes = [
                (t, c) for t, c in pattern.pattern_changes if t >= cutoff
            ]

            # Clear cache
            if current_area in self._cache:
                del self._cache[current_area]
