"""Probability constants and defaults for Area Occupancy Detection."""

from __future__ import annotations
from typing import Final, Dict

# Default decay window
DEFAULT_DECAY_WINDOW = 600  # seconds
# Decay lambda such that at half of decay_window probability is 25% of original
DECAY_LAMBDA = 0.866433976

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
    "on": 0.8,
    "standby": 0.4,
    "off": 0.1,
    "default": 0.0,
}

# Environmental detection baseline settings
ENVIRONMENTAL_BASELINE_PERCENT: Final[float] = 0.05  # 5% deviation allowed around mean
ENVIRONMENTAL_MIN_ACTIVE_DURATION: Final[int] = 300  # seconds of active data needed

# Safety bounds
MIN_PROBABILITY: Final[float] = 0.01
MAX_PROBABILITY: Final[float] = 0.99

# Default prior probabilities
DEFAULT_PROB_GIVEN_TRUE: Final[float] = 0.8
DEFAULT_PROB_GIVEN_FALSE: Final[float] = 0.2

# Motion sensor defaults
MOTION_PROB_GIVEN_TRUE: Final[float] = 0.9
MOTION_PROB_GIVEN_FALSE: Final[float] = 0.1

# Media device defaults
MEDIA_PROB_GIVEN_TRUE: Final[float] = 0.8
MEDIA_PROB_GIVEN_FALSE: Final[float] = 0.2

# Appliance defaults
APPLIANCE_PROB_GIVEN_TRUE: Final[float] = 0.7
APPLIANCE_PROB_GIVEN_FALSE: Final[float] = 0.3

# Environmental defaults
ENVIRONMENTAL_PROB_GIVEN_TRUE: Final[float] = 0.6
ENVIRONMENTAL_PROB_GIVEN_FALSE: Final[float] = 0.4

# Minimum active duration for storing learned priors
MIN_ACTIVE_DURATION_FOR_PRIORS: Final[int] = 300

# Baseline cache TTL (to avoid hitting DB repeatedly)
BASELINE_CACHE_TTL: Final[int] = 21600  # 6 hours in seconds
