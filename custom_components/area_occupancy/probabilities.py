"""Probability constants and defaults for Area Occupancy Detection."""

from __future__ import annotations
from typing import Final, Dict, Any

# Core weight limits - Ensure weights don't exceed reasonable bounds
MAX_MOTION_WEIGHT: Final[float] = 0.9
MAX_MEDIA_WEIGHT: Final[float] = 0.7
MAX_APPLIANCE_WEIGHT: Final[float] = 0.5
MAX_ENVIRONMENTAL_WEIGHT: Final[float] = 0.3

# Default weights for each detection type
DEFAULT_MOTION_WEIGHT: Final[float] = 0.7
DEFAULT_MEDIA_WEIGHT: Final[float] = 0.5
DEFAULT_APPLIANCE_WEIGHT: Final[float] = 0.1
DEFAULT_ENVIRONMENTAL_WEIGHT: Final[float] = 0.1

# Motion detection probabilities
MOTION_SINGLE_SENSOR_PROBABILITY: Final[float] = 0.85
MOTION_MULTIPLE_SENSORS_PROBABILITY: Final[float] = 0.9
MOTION_DECAY_FACTOR: Final[float] = 0.7
MAX_COMPOUND_DECAY: Final[float] = 0.8
MIN_MOTION_WEIGHT: Final[float] = 0.3

# Media device state probabilities
MEDIA_STATE_PROBABILITIES: Final[Dict[str, float]] = {
    "playing": 0.9,
    "paused": 0.7,
    "idle": 0.3,
    "off": 0.1,
    "default": 0.0,
}

# Appliance state probabilities
APPLIANCE_STATE_PROBABILITIES: Final[Dict[str, float]] = {
    "active": 0.8,
    "standby": 0.4,
    "off": 0.1,
    "default": 0.0,
}

# Environmental detection settings
ENVIRONMENTAL_SETTINGS: Final[Dict[str, Any]] = {
    "illuminance": {
        "change_threshold": 50.0,  # lux
        "significant_change_probability": 0.7,
        "minor_change_probability": 0.3,
    },
    "temperature": {
        "change_threshold": 1.5,  # degrees
        "baseline": 21.0,
        "weight": 0.6,
    },
    "humidity": {
        "change_threshold": 10.0,  # percent
        "baseline": 50.0,
        "weight": 0.5,
    },
}

# Correlation and confidence thresholds
CORRELATION_LIMIT: Final[float] = 0.7
MIN_CONFIDENCE: Final[float] = 0.2
HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.8
MEDIUM_CONFIDENCE_THRESHOLD: Final[float] = 0.5

# Pattern recognition thresholds
PATTERN_CHANGE_THRESHOLD: Final[float] = 0.2
PATTERN_PEAK_THRESHOLD: Final[float] = 0.7
MIN_PATTERN_SAMPLES: Final[int] = 100

# History analysis settings
HISTORY_UPDATE_INTERVAL: Final[int] = 3600  # seconds
MAX_HISTORY_SAMPLES: Final[int] = 1000
HISTORY_DECAY_RATE: Final[float] = 0.1

# Confidence calculation weights
CONFIDENCE_WEIGHTS: Final[Dict[str, float]] = {
    "sensor_availability": 0.4,
    "data_freshness": 0.3,
    "sample_size": 0.3,
}

# Safety bounds
MIN_PROBABILITY: Final[float] = 0.0
MAX_PROBABILITY: Final[float] = 1.0
MAX_WEIGHT_ADJUSTMENT: Final[float] = 0.5
MIN_SAMPLES_FOR_CONFIDENCE: Final[int] = 10

# Time windows
DEFAULT_TIME_WINDOW: Final[int] = 300  # seconds
MIN_TIME_WINDOW: Final[int] = 60  # seconds
MAX_TIME_WINDOW: Final[int] = 3600  # seconds
