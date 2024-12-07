"""Pattern analysis for area occupancy detection."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple
from collections import defaultdict

from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .types import PatternSummary, TimeSlot

_LOGGER = logging.getLogger(__name__)


def create_time_slot() -> TimeSlot:
    """Create a new TimeSlot with default values."""
    return {
        "total_samples": 0,
        "active_samples": 0,
        "occupied_samples": 0,
        "average_duration": 0.0,
        "confidence": 0.0,
        "last_updated": dt_util.utcnow(),
    }


class OccupancyPatternAnalyzer:
    """Analyzes and predicts occupancy patterns."""

    def __init__(self, slot_minutes: int = 30, min_confidence: float = 0.3) -> None:
        """Initialize the pattern analyzer."""
        self.slot_minutes = slot_minutes
        self.min_confidence = min_confidence
        self._patterns: Dict[str, Dict] = {}
        self._change_threshold = 0.2

    def get_time_slot(self, timestamp: datetime) -> str:
        """Convert timestamp to time slot string."""
        try:
            minutes = timestamp.hour * 60 + timestamp.minute
            slot = (minutes // self.slot_minutes) * self.slot_minutes
            return f"{slot // 60:02d}:{slot % 60:02d}"
        except Exception as err:
            _LOGGER.error("Error calculating time slot: %s", err)
            return "00:00"

    def update_pattern(
        self, area_id: str, timestamp: datetime, is_occupied: bool
    ) -> None:
        """Update pattern data with new occupancy information."""
        try:
            if area_id not in self._patterns:
                self._patterns[area_id] = {
                    "daily_patterns": defaultdict(create_time_slot),
                    "weekly_patterns": defaultdict(
                        lambda: defaultdict(create_time_slot)
                    ),
                    "pattern_changes": [],
                    "last_analysis": timestamp,
                }

            pattern = self._patterns[area_id]
            time_slot = self.get_time_slot(timestamp)
            day_name = timestamp.strftime("%A").lower()

            # Update daily pattern
            daily = pattern["daily_patterns"][time_slot]
            daily["total_samples"] += 1
            if is_occupied:
                daily["occupied_samples"] += 1
            daily["confidence"] = min(daily["total_samples"] / 100, 1.0)
            daily["last_updated"] = timestamp

            # Update weekly pattern
            weekly = pattern["weekly_patterns"][day_name][time_slot]
            weekly["total_samples"] += 1
            if is_occupied:
                weekly["occupied_samples"] += 1
            weekly["confidence"] = min(weekly["total_samples"] / 100, 1.0)
            weekly["last_updated"] = timestamp

            # Detect pattern changes
            self._detect_pattern_changes(area_id, timestamp)

        except Exception as err:
            _LOGGER.error("Error updating pattern: %s", err)
            raise HomeAssistantError(
                f"Failed to update occupancy pattern: {err}"
            ) from err

    def get_probability_adjustment(
        self, area_id: str, timestamp: datetime
    ) -> Tuple[float, float]:
        """Get probability adjustment based on patterns."""
        try:
            if area_id not in self._patterns:
                return 0.0, 0.0

            pattern = self._patterns[area_id]
            time_slot = self.get_time_slot(timestamp)
            day_name = timestamp.strftime("%A").lower()

            # Get daily pattern probability
            daily_data = pattern["daily_patterns"][time_slot]
            daily_prob = (
                daily_data["occupied_samples"] / daily_data["total_samples"]
                if daily_data["total_samples"] > 0
                else 0.0
            )
            daily_confidence = daily_data["confidence"]

            # Get weekly pattern probability
            weekly_data = pattern["weekly_patterns"][day_name][time_slot]
            weekly_prob = (
                weekly_data["occupied_samples"] / weekly_data["total_samples"]
                if weekly_data["total_samples"] > 0
                else 0.0
            )
            weekly_confidence = weekly_data["confidence"]

            # Combine probabilities based on confidence
            if (
                daily_confidence >= self.min_confidence
                and weekly_confidence >= self.min_confidence
            ):
                combined_prob = (
                    daily_prob * daily_confidence + weekly_prob * weekly_confidence
                ) / (daily_confidence + weekly_confidence)
                combined_confidence = (daily_confidence + weekly_confidence) / 2
            elif daily_confidence >= self.min_confidence:
                combined_prob = daily_prob
                combined_confidence = daily_confidence
            elif weekly_confidence >= self.min_confidence:
                combined_prob = weekly_prob
                combined_confidence = weekly_confidence
            else:
                return 0.0, 0.0

            return combined_prob, combined_confidence

        except Exception as err:
            _LOGGER.error("Error calculating probability adjustment: %s", err)
            return 0.0, 0.0

    def _detect_pattern_changes(self, area_id: str, timestamp: datetime) -> None:
        """Detect significant changes in occupancy patterns."""
        try:
            pattern = self._patterns[area_id]

            # Only analyze once per day
            if (timestamp - pattern["last_analysis"]) < timedelta(days=1):
                return

            daily_patterns = pattern["daily_patterns"]
            changes = []

            # Compare patterns over the last week
            week_ago = timestamp - timedelta(days=7)
            for time_slot, data in daily_patterns.items():
                if data["last_updated"] < week_ago:
                    continue

                historical_prob = self._get_historical_probability(
                    area_id, time_slot, week_ago
                )
                current_prob = (
                    data["occupied_samples"] / data["total_samples"]
                    if data["total_samples"] > 0
                    else 0.0
                )

                if abs(current_prob - historical_prob) > self._change_threshold:
                    changes.append((timestamp, current_prob - historical_prob))

            if changes:
                pattern["pattern_changes"].extend(changes)
                # Keep only last month of changes
                month_ago = timestamp - timedelta(days=30)
                pattern["pattern_changes"] = [
                    (t, c) for t, c in pattern["pattern_changes"] if t > month_ago
                ]

            pattern["last_analysis"] = timestamp

        except Exception as err:
            _LOGGER.error("Error detecting pattern changes: %s", err)

    def _get_historical_probability(
        self, area_id: str, time_slot: str, before_date: datetime
    ) -> float:
        """Get historical probability for a time slot before a specific date."""
        try:
            pattern = self._patterns[area_id]
            daily_data = pattern["daily_patterns"][time_slot]

            if daily_data["last_updated"] >= before_date:
                return 0.0

            return (
                daily_data["occupied_samples"] / daily_data["total_samples"]
                if daily_data["total_samples"] > 0
                else 0.0
            )
        except Exception as err:
            _LOGGER.error("Error getting historical probability: %s", err)
            return 0.0

    def get_pattern_summary(self, area_id: str) -> PatternSummary:
        """Get summary of detected patterns."""
        try:
            if area_id not in self._patterns:
                return PatternSummary([], 1.0, 0, 0)

            pattern = self._patterns[area_id]

            # Calculate peak occupancy times
            daily_peaks = []
            for time_slot, data in pattern["daily_patterns"].items():
                if data["total_samples"] > 0:
                    prob = data["occupied_samples"] / data["total_samples"]
                    if prob > 0.7:  # Consider slots with >70% occupancy as peaks
                        daily_peaks.append((time_slot, prob))

            # Sort by probability
            daily_peaks.sort(key=lambda x: x[1], reverse=True)

            # Calculate pattern stability
            stability = 1.0
            if pattern["pattern_changes"]:
                recent_changes = [
                    abs(change) for _, change in pattern["pattern_changes"][-10:]
                ]
                if recent_changes:
                    stability = 1.0 - min(
                        sum(recent_changes) / len(recent_changes), 1.0
                    )

            total_days = len(
                set(
                    d["last_updated"].date() for d in pattern["daily_patterns"].values()
                )
            )

            return PatternSummary(
                peak_times=daily_peaks[:3],  # Top 3 peak times
                pattern_stability=stability,
                total_analyzed_days=total_days,
                significant_changes=len(pattern["pattern_changes"]),
            )

        except Exception as err:
            _LOGGER.error("Error generating pattern summary: %s", err)
            return PatternSummary([], 1.0, 0, 0)

    def clear_old_data(self, area_id: str, before_date: datetime) -> None:
        """Clear pattern data older than the specified date."""
        try:
            if area_id not in self._patterns:
                return

            pattern = self._patterns[area_id]

            # Clear old daily patterns
            pattern["daily_patterns"] = {
                slot: data
                for slot, data in pattern["daily_patterns"].items()
                if data["last_updated"] >= before_date
            }

            # Clear old weekly patterns
            for day in pattern["weekly_patterns"]:
                pattern["weekly_patterns"][day] = {
                    slot: data
                    for slot, data in pattern["weekly_patterns"][day].items()
                    if data["last_updated"] >= before_date
                }

            # Clear old pattern changes
            pattern["pattern_changes"] = [
                (t, c) for t, c in pattern["pattern_changes"] if t >= before_date
            ]

        except Exception as err:
            _LOGGER.error("Error clearing old data: %s", err)
