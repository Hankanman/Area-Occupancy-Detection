"""Probability constants and defaults for Area Occupancy Detection."""

from __future__ import annotations
from typing import Final, Dict, Any

# Motion detection probabilities
MOTION_SINGLE_SENSOR_PROBABILITY: Final[float] = 0.85
MOTION_MULTIPLE_SENSORS_PROBABILITY: Final[float] = 0.9
MOTION_DECAY_FACTOR: Final[float] = 0.7
MAX_COMPOUND_DECAY: Final[float] = 0.8

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
    },
    "humidity": {
        "change_threshold": 10.0,  # percent
        "baseline": 50.0,
    },
}

# Safety bounds
MIN_PROBABILITY: Final[float] = 0.0
MAX_PROBABILITY: Final[float] = 1.0

# Default prior probabilities (should never be 0 or 1)
DEFAULT_PROB_GIVEN_TRUE: Final[float] = 0.8
DEFAULT_PROB_GIVEN_FALSE: Final[float] = 0.2

# Motion sensor specific defaults
MOTION_PROB_GIVEN_TRUE: Final[float] = 0.85
MOTION_PROB_GIVEN_FALSE: Final[float] = 0.15

# Media device specific defaults
MEDIA_PROB_GIVEN_TRUE: Final[float] = 0.75
MEDIA_PROB_GIVEN_FALSE: Final[float] = 0.25

# Appliance specific defaults
APPLIANCE_PROB_GIVEN_TRUE: Final[float] = 0.7
APPLIANCE_PROB_GIVEN_FALSE: Final[float] = 0.3

# Environmental sensor specific defaults
ENVIRONMENTAL_PROB_GIVEN_TRUE: Final[float] = 0.6
ENVIRONMENTAL_PROB_GIVEN_FALSE: Final[float] = 0.4
