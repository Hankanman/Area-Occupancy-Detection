"""Probability constants and defaults for Area Occupancy Detection."""

from __future__ import annotations
from typing import Final, Dict, Literal

from homeassistant.const import (
    STATE_ON,
    STATE_OFF,
    STATE_CLOSED,
    STATE_OPEN,
    STATE_PLAYING,
    STATE_PAUSED,
)

# Default decay window
DEFAULT_DECAY_WINDOW = 600  # seconds
# Decay lambda such that at half of decay_window probability is 25% of original
DECAY_LAMBDA = 0.866433976

# Environmental detection baseline settings
ENVIRONMENTAL_BASELINE_PERCENT: Final[float] = 0.05  # 5% deviation allowed around mean
ENVIRONMENTAL_MIN_ACTIVE_DURATION: Final[int] = 300  # seconds of active data needed

# Safety bounds
MIN_PROBABILITY: Final[float] = 0.01
MAX_PROBABILITY: Final[float] = 0.99

# Default prior probabilities
DEFAULT_PROB_GIVEN_TRUE: Final[float] = 0.3
DEFAULT_PROB_GIVEN_FALSE: Final[float] = 0.02

# Motion sensor defaults
MOTION_PROB_GIVEN_TRUE: Final[float] = 0.25
MOTION_PROB_GIVEN_FALSE: Final[float] = 0.05

# Door sensor defaults
DOOR_PROB_GIVEN_TRUE: Final[float] = 0.2
DOOR_PROB_GIVEN_FALSE: Final[float] = 0.02

# Window sensor defaults
WINDOW_PROB_GIVEN_TRUE: Final[float] = 0.2
WINDOW_PROB_GIVEN_FALSE: Final[float] = 0.02

# Light sensor defaults
LIGHT_PROB_GIVEN_TRUE: Final[float] = 0.2
LIGHT_PROB_GIVEN_FALSE: Final[float] = 0.02

# Media device defaults
MEDIA_PROB_GIVEN_TRUE: Final[float] = 0.25
MEDIA_PROB_GIVEN_FALSE: Final[float] = 0.02

# Appliance defaults
APPLIANCE_PROB_GIVEN_TRUE: Final[float] = 0.2
APPLIANCE_PROB_GIVEN_FALSE: Final[float] = 0.02

# Environmental defaults
ENVIRONMENTAL_PROB_GIVEN_TRUE: Final[float] = 0.09
ENVIRONMENTAL_PROB_GIVEN_FALSE: Final[float] = 0.01

# Minimum active duration for storing learned priors
MIN_ACTIVE_DURATION_FOR_PRIORS: Final[int] = 300

# Baseline cache TTL (to avoid hitting DB repeatedly)
BASELINE_CACHE_TTL: Final[int] = 21600  # 6 hours in seconds

# Default Priors
DEFAULT_PRIOR: Final[float] = 0.1713
MOTION_DEFAULT_PRIOR: Final[float] = 0.35
MEDIA_DEFAULT_PRIOR: Final[float] = 0.30
APPLIANCE_DEFAULT_PRIOR: Final[float] = 0.2356
DOOR_DEFAULT_PRIOR: Final[float] = 0.1356
WINDOW_DEFAULT_PRIOR: Final[float] = 0.1569
LIGHT_DEFAULT_PRIOR: Final[float] = 0.3846
ENVIRONMENTAL_DEFAULT_PRIOR: Final[float] = 0.0769

# Sensor type weights (higher weight = more impact)
SENSOR_WEIGHTS: Final[Dict[str, float]] = {
    "motion": 0.85,
    "media": 0.7,
    "appliance": 0.3,
    "door": 0.3,
    "window": 0.2,
    "light": 0.2,
    "environmental": 0.1,
}

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

SensorType = Literal[
    "motion",
    "media",
    "appliance",
    "door",
    "window",
    "light",
    "environmental",
]

SENSOR_TYPE_CONFIGS = {
    "motion": {
        "prob_given_true": MOTION_PROB_GIVEN_TRUE,
        "prob_given_false": MOTION_PROB_GIVEN_FALSE,
        "default_prior": MOTION_DEFAULT_PRIOR,
        "weight": SENSOR_WEIGHTS["motion"],
        "active_states": {STATE_ON},
    },
    "media": {
        "prob_given_true": MEDIA_PROB_GIVEN_TRUE,
        "prob_given_false": MEDIA_PROB_GIVEN_FALSE,
        "default_prior": MEDIA_DEFAULT_PRIOR,
        "weight": SENSOR_WEIGHTS["media"],
        "active_states": {STATE_PLAYING, STATE_PAUSED},
    },
    "appliance": {
        "prob_given_true": APPLIANCE_PROB_GIVEN_TRUE,
        "prob_given_false": APPLIANCE_PROB_GIVEN_FALSE,
        "default_prior": APPLIANCE_DEFAULT_PRIOR,
        "weight": SENSOR_WEIGHTS["appliance"],
        "active_states": {STATE_ON},
    },
    "door": {
        "prob_given_true": DOOR_PROB_GIVEN_TRUE,
        "prob_given_false": DOOR_PROB_GIVEN_FALSE,
        "default_prior": DOOR_DEFAULT_PRIOR,
        "weight": SENSOR_WEIGHTS["door"],
        "active_states": {STATE_OFF, STATE_CLOSED},  # Closed door indicates occupancy
    },
    "window": {
        "prob_given_true": WINDOW_PROB_GIVEN_TRUE,
        "prob_given_false": WINDOW_PROB_GIVEN_FALSE,
        "default_prior": WINDOW_DEFAULT_PRIOR,
        "weight": SENSOR_WEIGHTS["window"],
        "active_states": {STATE_ON, STATE_OPEN},
    },
    "light": {
        "prob_given_true": LIGHT_PROB_GIVEN_TRUE,
        "prob_given_false": LIGHT_PROB_GIVEN_FALSE,
        "default_prior": LIGHT_DEFAULT_PRIOR,
        "weight": SENSOR_WEIGHTS["light"],
        "active_states": {STATE_ON},
    },
    "environmental": {
        "prob_given_true": ENVIRONMENTAL_PROB_GIVEN_TRUE,
        "prob_given_false": ENVIRONMENTAL_PROB_GIVEN_FALSE,
        "default_prior": ENVIRONMENTAL_DEFAULT_PRIOR,
        "weight": SENSOR_WEIGHTS["environmental"],
        "active_states": {
            STATE_ON
        },  # Environmental sensors typically use thresholds in coordinator
    },
}
