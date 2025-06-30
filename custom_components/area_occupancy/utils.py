"""Utility functions for the Area Occupancy component."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import logging
import re
from typing import TYPE_CHECKING, Any, TypedDict

from sqlalchemy.exc import SQLAlchemyError

from homeassistant.components.recorder.history import get_significant_states
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity import get_device_class, get_unit_of_measurement
from homeassistant.helpers.recorder import get_instance
from homeassistant.util import dt as dt_util

# Import official HA device class enums
try:
    from homeassistant.components.binary_sensor import BinarySensorDeviceClass
except ImportError:
    BinarySensorDeviceClass = None

try:
    from homeassistant.components.sensor import SensorDeviceClass
except ImportError:
    SensorDeviceClass = None

try:
    from homeassistant.components.number.const import NumberDeviceClass
except ImportError:
    NumberDeviceClass = None

try:
    from homeassistant.components.media_player import MediaPlayerDeviceClass
except ImportError:
    MediaPlayerDeviceClass = None

from .const import (
    MAX_PROBABILITY,
    MAX_WEIGHT,
    MIN_PROBABILITY,
    MIN_WEIGHT,
    ROUNDING_PRECISION,
)

if TYPE_CHECKING:
    from .data.entity import Entity
    from .data.entity_type import InputType

_LOGGER = logging.getLogger(__name__)


class TimeInterval(TypedDict):
    """Time interval with state information."""

    start: datetime
    end: datetime
    state: str


# ──────────────────────────────────── History Utilities ──────────────────────────────────
async def get_states_from_recorder(
    hass: HomeAssistant, entity_id: str, start_time: datetime, end_time: datetime
) -> list[State | dict[str, Any]] | None:
    """Fetch states history from recorder.

    Args:
        hass: Home Assistant instance
        entity_id: Entity ID to fetch history for
        start_time: Start time window
        end_time: End time window

    Returns:
        List of states or minimal state dicts if successful, None if error occurred

    Raises:
        HomeAssistantError: If recorder access fails
        SQLAlchemyError: If database query fails

    """
    _LOGGER.debug("Fetching states: %s [%s -> %s]", entity_id, start_time, end_time)

    # Check if recorder is available
    recorder = get_instance(hass)
    if recorder is None:
        _LOGGER.debug("Recorder not available for %s", entity_id)
        return None

    try:
        states = await recorder.async_add_executor_job(
            lambda: get_significant_states(
                hass,
                start_time,
                end_time,
                [entity_id],
                minimal_response=False,  # Must be false to include last_changed attribute
            )
        )

        entity_states = states.get(entity_id) if states else None

        if entity_states:
            _LOGGER.debug("Found %d states for %s", len(entity_states), entity_id)
        else:
            _LOGGER.debug("No states found for %s", entity_id)

    except (HomeAssistantError, SQLAlchemyError, TimeoutError) as err:
        _LOGGER.error("Error getting states for %s: %s", entity_id, err)
        # Re-raise the exception as documented, let the caller handle fallback
        raise

    else:
        return entity_states


async def states_to_intervals(
    states: Sequence[State], start: datetime, end: datetime
) -> list[TimeInterval]:
    """Convert state history to time intervals.

    Args:
        states: List of State objects
        start: Start time for analysis
        end: End time for analysis

    Returns:
        List of TimeInterval objects

    """
    intervals: list[TimeInterval] = []
    if not states:
        return intervals

    # Sort states chronologically
    sorted_states = sorted(states, key=lambda x: x.last_changed)

    # Determine the state that was active at the start time
    current_state = sorted_states[0].state
    for state in sorted_states:
        if state.last_changed <= start:
            current_state = state.state
        else:
            break

    current_time = start

    # Build intervals between state changes
    for state in sorted_states:
        if state.last_changed <= start:
            continue
        if state.last_changed > end:
            break
        intervals.append(
            TimeInterval(
                start=current_time, end=state.last_changed, state=current_state
            )
        )
        current_state = state.state
        current_time = state.last_changed

    # Final interval until end
    intervals.append(TimeInterval(start=current_time, end=end, state=current_state))

    return intervals


# ───────────────────────────────────────── Validation ────────────────────────
def validate_prob(value: complex) -> float:
    """Validate probability value, handling complex numbers."""
    # Handle complex numbers by taking the real part
    if isinstance(value, complex):
        _LOGGER.warning(
            "Complex number detected in probability calculation: %s, using real part",
            value,
        )
        value = value.real

    # Ensure it's a valid float
    if not isinstance(value, (int, float)) or not (-1e10 < value < 1e10):
        _LOGGER.warning("Invalid probability value: %s, using default", value)
        return 0.5

    return max(0.001, min(float(value), 1.0))


def validate_prior(value: float) -> float:
    """Validate prior probability value."""
    return max(0.000001, min(value, 0.999999))


def validate_datetime(value: datetime | None) -> datetime:
    """Validate datetime value."""
    return value if isinstance(value, datetime) else dt_util.utcnow()


def validate_weight(value: float) -> float:
    """Validate weight value."""
    return max(MIN_WEIGHT, min(value, MAX_WEIGHT))


def validate_decay_factor(value: float) -> float:
    """Validate decay factor value."""
    return max(MIN_PROBABILITY, min(value, MAX_PROBABILITY))


def format_float(value: float) -> float:
    """Format float value."""
    return round(float(value), ROUNDING_PRECISION)


def format_percentage(value: float) -> str:
    """Format float value as percentage."""
    return f"{value * 100:.2f}%"


def apply_decay(
    prob_given_true: float, prob_given_false: float, decay_factor: float
) -> tuple[float, float]:
    """Apply decay factor to likelihood probabilities.

    This maintains mathematical equivalence with applying decay as an exponent
    to the Bayes factor in the original bayesian_probability function.

    Args:
        prob_given_true: Original probability given true
        prob_given_false: Original probability given false
        decay_factor: Decay factor (0.0 = full decay, 1.0 = no decay)

    Returns:
        Tuple of (effective_prob_given_true, effective_prob_given_false)

    """
    if decay_factor == 1.0:
        return prob_given_true, prob_given_false

    if decay_factor == 0.0:
        # Full decay - return neutral probabilities
        return 0.5, 0.5

    # Ensure inputs are in valid range
    prob_given_true = max(0.001, min(prob_given_true, 0.999))
    prob_given_false = max(0.001, min(prob_given_false, 0.999))

    # Calculate the original bayes factor
    original_bf = prob_given_true / prob_given_false

    # Apply decay to the bayes factor (this is what bayesian_probability was doing)
    decayed_bf = original_bf**decay_factor

    # Calculate geometric mean to preserve overall magnitude
    geo_mean = (prob_given_true * prob_given_false) ** 0.5

    # Calculate new probabilities that give the decayed bayes factor
    # p_t_eff / p_f_eff = decayed_bf
    # p_t_eff * p_f_eff = geo_mean^2 (preserve geometric mean)
    # Solving: p_t_eff = geo_mean * sqrt(decayed_bf), p_f_eff = geo_mean / sqrt(decayed_bf)

    sqrt_bf = decayed_bf**0.5
    p_true_eff = geo_mean * sqrt_bf
    p_false_eff = geo_mean / sqrt_bf

    # Ensure probabilities are in valid range
    p_true_eff = max(0.001, min(0.999, p_true_eff))
    p_false_eff = max(0.001, min(0.999, p_false_eff))

    return p_true_eff, p_false_eff


EPS = 1e-12


# ────────────────────────────────────── Core Bayes ───────────────────────────
def bayesian_probability(
    *,  # keyword-only → prevents accidental positional mix-ups
    prior: float,
    prob_given_true: float,
    prob_given_false: float,
    evidence: bool | None,
) -> float:
    """Pure Bayesian probability update.

    This function now focuses solely on Bayesian calculation.
    Decay should be applied to prob_given_true and prob_given_false before calling this.

    Args:
        prior: Prior probability
        prob_given_true: Probability of evidence given true (decay already applied if needed)
        prob_given_false: Probability of evidence given false (decay already applied if needed)
        evidence: Evidence (True/False/None)

    Returns:
        Posterior probability

    """
    if evidence is None:
        return prior

    # Validate inputs first
    prob_given_true = max(0.001, min(prob_given_true, 0.999))
    prob_given_false = max(0.001, min(prob_given_false, 0.999))
    prior = max(0.001, min(prior, 0.999))

    # Calculate Bayes factor
    bayes_factor = (
        (prob_given_true + EPS) / (prob_given_false + EPS)
        if evidence
        else (1 - prob_given_true + EPS) / (1 - prob_given_false + EPS)
    )

    # Ensure bayes_factor is positive to avoid complex numbers
    bayes_factor = max(EPS, bayes_factor)

    # Calculate posterior odds
    odds = prior / (1.0 - prior + EPS)
    posterior_odds = odds * bayes_factor

    # Return posterior probability
    result = posterior_odds / (1.0 + posterior_odds)
    return validate_prob(result)


# ─────────────────────────────── Area-level fusion ───────────────────────────
def overall_probability(entities: dict[str, Entity], prior: float) -> float:
    """Combine weighted posteriors from active/decaying sensors."""

    contributing_entities = [
        e for e in entities.values() if e.evidence or e.decay.is_decaying
    ]

    if not contributing_entities:
        return validate_prob(prior)

    product = 1.0
    for e in contributing_entities:
        # Use Entity's effective probabilities (decay already applied)
        posterior = bayesian_probability(
            prior=prior,
            prob_given_true=e.effective_prob_given_true,
            prob_given_false=e.effective_prob_given_false,
            evidence=True,
        )
        product *= 1 - posterior

    return validate_prob(1 - product)


# ──────────────────────────────── Entity Type Detection ──────────────────────────────────


def get_device_class_with_fallback(hass: HomeAssistant, entity_id: str) -> str:
    """Get device class with intelligent fallback detection.

    This function first tries to get the device class from Home Assistant's
    built-in get_device_class function. It then applies intelligent corrections
    and uses fuzzy matching as a fallback if needed.

    Args:
        hass: Home Assistant instance
        entity_id: The entity ID to analyze

    Returns:
        Device class string (official HA device class or "Unknown")

    """
    # First try Home Assistant's built-in device class detection
    ha_device_class = get_device_class(hass, entity_id)
    unit = get_unit_of_measurement(hass, entity_id)

    # Always apply our intelligent detection/correction logic
    detected_class = detect_device_class(hass, entity_id, ha_device_class, unit)

    if detected_class:
        if ha_device_class and detected_class != ha_device_class:
            _LOGGER.debug(
                "Corrected device class for %s: %s -> %s",
                entity_id,
                ha_device_class,
                detected_class,
            )
        else:
            _LOGGER.debug("Using device class for %s: %s", entity_id, detected_class)
        return detected_class

    # If our detection failed but HA had a valid class, use that
    if ha_device_class and ha_device_class.lower() not in ["unknown", "none"]:
        _LOGGER.debug(
            "Fallback to HA device class for %s: %s", entity_id, ha_device_class
        )
        return ha_device_class

    _LOGGER.debug("Unable to determine device class for %s", entity_id)
    return "Unknown"


def detect_device_class(
    hass: HomeAssistant,
    entity_id: str,
    device_class: str | None = None,
    unit_of_measurement: str | None = None,
) -> str | None:
    """Detect Home Assistant device class using fuzzy matching with official enums.

    Args:
        hass: Home Assistant instance
        entity_id: The entity ID to analyze
        device_class: Device class from Home Assistant (may be None or incorrect)
        unit_of_measurement: Unit of measurement (may be None)

    Returns:
        Official HA device class string if detected, None if unable to determine

    """
    domain = entity_id.split(".")[0]

    _LOGGER.debug(
        "Detecting device class for %s: domain=%s, device_class=%s, unit=%s",
        entity_id,
        domain,
        device_class,
        unit_of_measurement,
    )

    # 1. Validate and correct existing device class if provided
    if device_class and device_class.lower() != "unknown":
        corrected_class = _validate_and_correct_device_class(
            entity_id, device_class, domain
        )
        if corrected_class:
            return corrected_class

    # 2. Fuzzy match based on entity ID and domain
    fuzzy_class = _fuzzy_match_device_class(entity_id, domain, unit_of_measurement)
    if fuzzy_class:
        return fuzzy_class

    return None


def _validate_and_correct_device_class(
    entity_id: str, device_class: str, domain: str
) -> str | None:
    """Validate device class against official enums and apply corrections."""
    device_class_lower = device_class.lower()
    entity_id_lower = entity_id.lower()

    # Get all valid device classes for the domain
    valid_classes = _get_valid_device_classes_for_domain(domain)

    # Check if device class is valid
    if device_class_lower in valid_classes:
        # Apply corrections for known misclassifications
        if device_class_lower == "door" and "window" in entity_id_lower:
            _LOGGER.debug(
                "Correcting device_class 'door' to 'window' for %s", entity_id
            )
            return "window"
        if device_class_lower == "window" and "door" in entity_id_lower:
            _LOGGER.debug(
                "Correcting device_class 'window' to 'door' for %s", entity_id
            )
            return "door"

        _LOGGER.debug(
            "Using validated device_class for %s: %s", entity_id, device_class_lower
        )
        return device_class_lower

    return None


def _get_valid_device_classes_for_domain(domain: str) -> set[str]:
    """Get all valid device classes for a given domain."""
    valid_classes = set()

    if domain == "binary_sensor" and BinarySensorDeviceClass:
        valid_classes.update(cls.value for cls in BinarySensorDeviceClass)
    elif domain == "sensor" and SensorDeviceClass:
        valid_classes.update(cls.value for cls in SensorDeviceClass)
    elif domain == "number" and NumberDeviceClass:
        valid_classes.update(cls.value for cls in NumberDeviceClass)
    elif domain == "media_player" and MediaPlayerDeviceClass:
        valid_classes.update(cls.value for cls in MediaPlayerDeviceClass)

    return valid_classes


def _fuzzy_match_device_class(
    entity_id: str, domain: str, unit_of_measurement: str | None
) -> str | None:
    """Use fuzzy matching to determine device class from entity patterns."""
    entity_id_lower = entity_id.lower()

    # Extract meaningful words from entity ID for matching
    entity_words = _extract_entity_keywords(entity_id_lower)

    # Get device classes for the domain
    if domain == "binary_sensor" and BinarySensorDeviceClass:
        return _fuzzy_match_binary_sensor(entity_words, entity_id_lower)
    if domain == "sensor" and SensorDeviceClass:
        return _fuzzy_match_sensor(entity_words, entity_id_lower, unit_of_measurement)
    if domain == "number" and NumberDeviceClass:
        return _fuzzy_match_number(entity_words, entity_id_lower, unit_of_measurement)
    if domain == "media_player" and MediaPlayerDeviceClass:
        return _fuzzy_match_media_player(entity_words, entity_id_lower)

    return None


def _extract_entity_keywords(entity_id_lower: str) -> list[str]:
    """Extract meaningful keywords from entity ID for fuzzy matching."""
    # Remove domain prefix and common suffixes
    entity_part = (
        entity_id_lower.split(".", 1)[1] if "." in entity_id_lower else entity_id_lower
    )

    # Split on common separators but preserve alphanumeric combinations like co2, pm25
    keywords = re.split(r"[_\-\s]+", entity_part)

    # Filter out empty strings, very short words, and pure numbers
    return [word for word in keywords if len(word) > 1 and not word.isdigit()]


def _fuzzy_match_binary_sensor(
    entity_words: list[str], entity_id_lower: str
) -> str | None:
    """Fuzzy match binary sensor device classes."""
    if not BinarySensorDeviceClass:
        return None

    # High-confidence exact matches first
    exact_matches = {
        "motion": BinarySensorDeviceClass.MOTION,
        "occupancy": BinarySensorDeviceClass.OCCUPANCY,
        "presence": BinarySensorDeviceClass.PRESENCE,
        "door": BinarySensorDeviceClass.DOOR,
        "window": BinarySensorDeviceClass.WINDOW,
        "moisture": BinarySensorDeviceClass.MOISTURE,
        "battery": BinarySensorDeviceClass.BATTERY,
        "smoke": BinarySensorDeviceClass.SMOKE,
        "gas": BinarySensorDeviceClass.GAS,
        "vibration": BinarySensorDeviceClass.VIBRATION,
        "sound": BinarySensorDeviceClass.SOUND,
        "connectivity": BinarySensorDeviceClass.CONNECTIVITY,
        "power": BinarySensorDeviceClass.POWER,
        "running": BinarySensorDeviceClass.RUNNING,
        "tamper": BinarySensorDeviceClass.TAMPER,
    }

    # Check for exact keyword matches
    for word in entity_words:
        if word in exact_matches:
            _LOGGER.debug(
                "Exact match for binary sensor %s: %s",
                entity_id_lower,
                exact_matches[word].value,
            )
            return exact_matches[word].value

    # Fuzzy matching for similar words
    fuzzy_matches = {
        "pir": BinarySensorDeviceClass.MOTION,
        "movement": BinarySensorDeviceClass.MOTION,
        "occupied": BinarySensorDeviceClass.OCCUPANCY,
        "leak": BinarySensorDeviceClass.MOISTURE,
        "flood": BinarySensorDeviceClass.MOISTURE,
        "wet": BinarySensorDeviceClass.MOISTURE,
        "fire": BinarySensorDeviceClass.SMOKE,
        "contact": None,  # Handle specially - needs context
        "garage": BinarySensorDeviceClass.GARAGE_DOOR,
        "entry": BinarySensorDeviceClass.DOOR,
        "entrance": BinarySensorDeviceClass.DOOR,
        "shutter": BinarySensorDeviceClass.WINDOW,
        "blind": BinarySensorDeviceClass.WINDOW,
        "opening": BinarySensorDeviceClass.OPENING,
        "shake": BinarySensorDeviceClass.VIBRATION,
        "connection": BinarySensorDeviceClass.CONNECTIVITY,
        "online": BinarySensorDeviceClass.CONNECTIVITY,
        "charging": BinarySensorDeviceClass.BATTERY_CHARGING,
    }

    # Check fuzzy matches
    for word in entity_words:
        if word in fuzzy_matches:
            match = fuzzy_matches[word]
            if match:
                _LOGGER.debug(
                    "Fuzzy match for binary sensor %s: %s", entity_id_lower, match.value
                )
                return match.value

    # Special handling for contact sensors
    if "contact" in entity_words:
        if any(w in entity_words for w in ["window", "casement", "sash"]):
            return BinarySensorDeviceClass.WINDOW.value
        if any(w in entity_words for w in ["door", "gate", "portal"]):
            return BinarySensorDeviceClass.DOOR.value
        # Default contact to door
        return BinarySensorDeviceClass.DOOR.value

    return None


def _fuzzy_match_media_player(
    entity_words: list[str], entity_id_lower: str
) -> str | None:
    """Fuzzy match media player device classes."""
    if not MediaPlayerDeviceClass:
        return None

    # High-confidence exact matches first
    exact_matches = {
        "tv": MediaPlayerDeviceClass.TV,
        "television": MediaPlayerDeviceClass.TV,
        "speaker": MediaPlayerDeviceClass.SPEAKER,
        "speakers": MediaPlayerDeviceClass.SPEAKER,
        "receiver": MediaPlayerDeviceClass.RECEIVER,
    }

    # Check for exact keyword matches
    for word in entity_words:
        if word in exact_matches:
            _LOGGER.debug(
                "Exact match for media player %s: %s",
                entity_id_lower,
                exact_matches[word].value,
            )
            return exact_matches[word].value

    # Fuzzy matching for similar words and patterns
    fuzzy_matches = {
        # TV/Television patterns
        "samsung": MediaPlayerDeviceClass.TV,  # Samsung TVs
        "lg": MediaPlayerDeviceClass.TV,  # LG TVs
        "sony": MediaPlayerDeviceClass.TV,  # Sony TVs
        "roku": MediaPlayerDeviceClass.TV,  # Roku devices
        "fire": MediaPlayerDeviceClass.TV,  # Fire TV
        "chromecast": MediaPlayerDeviceClass.TV,  # Chromecast
        "apple": MediaPlayerDeviceClass.TV,  # Apple TV (when combined with other words)
        "smart": MediaPlayerDeviceClass.TV,  # Smart TV
        "android": MediaPlayerDeviceClass.TV,  # Android TV
        "webos": MediaPlayerDeviceClass.TV,  # LG WebOS
        "tizen": MediaPlayerDeviceClass.TV,  # Samsung Tizen
        # Speaker patterns
        "sonos": MediaPlayerDeviceClass.SPEAKER,  # Sonos speakers
        "echo": MediaPlayerDeviceClass.SPEAKER,  # Amazon Echo
        "alexa": MediaPlayerDeviceClass.SPEAKER,  # Amazon Alexa
        "google": MediaPlayerDeviceClass.SPEAKER,  # Google Home/Nest
        "nest": MediaPlayerDeviceClass.SPEAKER,  # Google Nest
        "homepod": MediaPlayerDeviceClass.SPEAKER,  # Apple HomePod
        "bluetooth": MediaPlayerDeviceClass.SPEAKER,  # Bluetooth speakers
        "airplay": MediaPlayerDeviceClass.SPEAKER,  # AirPlay speakers
        "spotify": MediaPlayerDeviceClass.SPEAKER,  # Often speaker-like
        "music": MediaPlayerDeviceClass.SPEAKER,  # Music players often speakers
        # Receiver patterns
        "denon": MediaPlayerDeviceClass.RECEIVER,  # Denon receivers
        "yamaha": MediaPlayerDeviceClass.RECEIVER,  # Yamaha receivers
        "onkyo": MediaPlayerDeviceClass.RECEIVER,  # Onkyo receivers
        "marantz": MediaPlayerDeviceClass.RECEIVER,  # Marantz receivers
        "pioneer": MediaPlayerDeviceClass.RECEIVER,  # Pioneer receivers
        "avr": MediaPlayerDeviceClass.RECEIVER,  # Audio/Video Receiver
        "amplifier": MediaPlayerDeviceClass.RECEIVER,  # Amplifiers
        "amp": MediaPlayerDeviceClass.RECEIVER,  # Short for amplifier
        "stereo": MediaPlayerDeviceClass.RECEIVER,  # Stereo systems
    }

    # Check fuzzy matches
    for word in entity_words:
        if word in fuzzy_matches:
            match = fuzzy_matches[word]
            _LOGGER.debug(
                "Fuzzy match for media player %s: %s", entity_id_lower, match.value
            )
            return match.value

    # Apple TV detection (needs both words)
    if "apple" in entity_words and "tv" in entity_words:
        return MediaPlayerDeviceClass.TV.value

    # Fire TV detection
    if "fire" in entity_words and "tv" in entity_words:
        return MediaPlayerDeviceClass.TV.value

    # Google Home/Nest detection
    if "google" in entity_words and ("home" in entity_words or "nest" in entity_words):
        return MediaPlayerDeviceClass.SPEAKER.value

    # Default fallback - if no specific pattern detected, try to infer from context
    # Most streaming services default to speakers unless explicitly TV-related
    streaming_services = ["spotify", "pandora", "tidal", "apple", "amazon", "deezer"]
    if any(service in entity_words for service in streaming_services):
        return MediaPlayerDeviceClass.SPEAKER.value

    # No clear match found
    return None


def _fuzzy_match_sensor(
    entity_words: list[str], entity_id_lower: str, unit: str | None
) -> str | None:
    """Fuzzy match sensor device classes."""
    if not SensorDeviceClass:
        return None

    # Unit-based matching first (more reliable)
    if unit:
        unit_lower = unit.lower()
        unit_matches = {
            # Temperature
            "°c": SensorDeviceClass.TEMPERATURE,
            "°f": SensorDeviceClass.TEMPERATURE,
            "k": SensorDeviceClass.TEMPERATURE,
            # Humidity
            "%": None,  # Could be humidity, moisture, battery - check keywords
            "rh": SensorDeviceClass.HUMIDITY,
            # Illuminance
            "lx": SensorDeviceClass.ILLUMINANCE,
            "lm": SensorDeviceClass.ILLUMINANCE,
            # Pressure
            "hpa": SensorDeviceClass.ATMOSPHERIC_PRESSURE,
            "mbar": SensorDeviceClass.ATMOSPHERIC_PRESSURE,
            "pa": SensorDeviceClass.PRESSURE,
            "mmhg": SensorDeviceClass.ATMOSPHERIC_PRESSURE,
            "inhg": SensorDeviceClass.ATMOSPHERIC_PRESSURE,
            # Gas concentration
            "ppm": None,  # Could be CO, CO2, VOC - check keywords
            "ppb": None,
            # Particulate matter
            "µg/m³": None,  # Could be PM1, PM2.5, PM10, NO2, etc - check keywords
            "µg/m3": None,
            "ug/m3": None,
            "mg/m³": SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
            "mg/m3": SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
            # Sound
            "db": SensorDeviceClass.SOUND_PRESSURE,
            "dba": SensorDeviceClass.SOUND_PRESSURE,
            # Energy/Power
            "w": SensorDeviceClass.POWER,
            "kw": SensorDeviceClass.POWER,
            "mw": SensorDeviceClass.POWER,
            "kwh": SensorDeviceClass.ENERGY,
            "wh": SensorDeviceClass.ENERGY,
            # Current/Voltage
            "a": SensorDeviceClass.CURRENT,
            "ma": SensorDeviceClass.CURRENT,
            "v": SensorDeviceClass.VOLTAGE,
            "mv": SensorDeviceClass.VOLTAGE,
        }

        if unit_lower in unit_matches:
            match = unit_matches[unit_lower]
            if match:
                _LOGGER.debug(
                    "Unit-based match for sensor %s: %s (unit: %s)",
                    entity_id_lower,
                    match.value,
                    unit,
                )
                return match.value

    # Keyword-based matching
    keyword_matches = {
        "temperature": SensorDeviceClass.TEMPERATURE,
        "temp": SensorDeviceClass.TEMPERATURE,
        "humidity": SensorDeviceClass.HUMIDITY,
        "illuminance": SensorDeviceClass.ILLUMINANCE,
        "lux": SensorDeviceClass.ILLUMINANCE,
        "light": SensorDeviceClass.ILLUMINANCE,
        "pressure": SensorDeviceClass.PRESSURE,
        "atmospheric": SensorDeviceClass.ATMOSPHERIC_PRESSURE,
        "barometric": SensorDeviceClass.ATMOSPHERIC_PRESSURE,
        "co2": SensorDeviceClass.CO2,
        "carbon": SensorDeviceClass.CO2,  # Ambiguous - could be CO or CO2
        "dioxide": SensorDeviceClass.CO2,
        "monoxide": SensorDeviceClass.CO,
        "pm25": SensorDeviceClass.PM25,
        "pm10": SensorDeviceClass.PM10,
        "pm1": SensorDeviceClass.PM1,
        "particulate": SensorDeviceClass.PM25,  # Default to PM2.5
        "voc": SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
        "tvoc": SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
        "volatile": SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
        "organic": SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS,
        "noise": SensorDeviceClass.SOUND_PRESSURE,
        "sound": SensorDeviceClass.SOUND_PRESSURE,
        "power": SensorDeviceClass.POWER,
        "energy": SensorDeviceClass.ENERGY,
        "consumption": SensorDeviceClass.ENERGY,
        "current": SensorDeviceClass.CURRENT,
        "voltage": SensorDeviceClass.VOLTAGE,
        "battery": SensorDeviceClass.BATTERY,
        "signal": SensorDeviceClass.SIGNAL_STRENGTH,
        "rssi": SensorDeviceClass.SIGNAL_STRENGTH,
        "distance": SensorDeviceClass.DISTANCE,
        "weight": SensorDeviceClass.WEIGHT,
        "mass": SensorDeviceClass.WEIGHT,
        "frequency": SensorDeviceClass.FREQUENCY,
        "speed": SensorDeviceClass.SPEED,
        "volume": SensorDeviceClass.VOLUME,
        "flow": SensorDeviceClass.VOLUME_FLOW_RATE,
        "rate": SensorDeviceClass.VOLUME_FLOW_RATE,
        "water": SensorDeviceClass.WATER,
        "gas": SensorDeviceClass.GAS,
        "wind": SensorDeviceClass.WIND_SPEED,
        "direction": SensorDeviceClass.WIND_DIRECTION,
        "moisture": SensorDeviceClass.MOISTURE,
    }

    # Check for keyword matches
    for word in entity_words:
        if word in keyword_matches:
            match = keyword_matches[word]
            _LOGGER.debug(
                "Keyword match for sensor %s: %s", entity_id_lower, match.value
            )
            return match.value

    # Special handling for percentage unit with context
    if unit and unit.lower() == "%":
        if any(w in entity_words for w in ["humidity", "rh"]):
            return SensorDeviceClass.HUMIDITY.value
        if any(w in entity_words for w in ["moisture", "wet"]):
            return SensorDeviceClass.MOISTURE.value
        if any(w in entity_words for w in ["battery", "charge"]):
            return SensorDeviceClass.BATTERY.value

    # Special handling for ppm/ppb with context
    if unit and unit.lower() in ["ppm", "ppb"]:
        if any(w in entity_words for w in ["co2", "carbon", "dioxide"]):
            return SensorDeviceClass.CO2.value
        if any(w in entity_words for w in ["co", "monoxide"]):
            return SensorDeviceClass.CO.value
        if any(w in entity_words for w in ["voc", "volatile", "organic"]):
            return SensorDeviceClass.VOLATILE_ORGANIC_COMPOUNDS_PARTS.value

    # Special handling for µg/m³ with context
    if unit and unit.lower() in ["µg/m³", "µg/m3", "ug/m3"]:
        if any(w in entity_words for w in ["pm25", "pm2"]):
            return SensorDeviceClass.PM25.value
        if any(w in entity_words for w in ["pm10"]):
            return SensorDeviceClass.PM10.value
        if any(w in entity_words for w in ["pm1"]):
            return SensorDeviceClass.PM1.value
        if any(w in entity_words for w in ["no2", "nitrogen", "dioxide"]):
            return SensorDeviceClass.NITROGEN_DIOXIDE.value
        if any(w in entity_words for w in ["no", "nitrogen", "monoxide"]):
            return SensorDeviceClass.NITROGEN_MONOXIDE.value
        if any(w in entity_words for w in ["ozone", "o3"]):
            return SensorDeviceClass.OZONE.value
        if any(w in entity_words for w in ["so2", "sulphur", "sulfur"]):
            return SensorDeviceClass.SULPHUR_DIOXIDE.value

    return None


def _fuzzy_match_number(
    entity_words: list[str], entity_id_lower: str, unit: str | None
) -> str | None:
    """Fuzzy match number device classes."""
    if not NumberDeviceClass:
        return None

    # Number device classes are similar to sensor device classes
    # We can reuse much of the sensor logic
    sensor_match = _fuzzy_match_sensor(entity_words, entity_id_lower, unit)

    # Convert sensor device class to number device class if valid
    if sensor_match and hasattr(NumberDeviceClass, sensor_match.upper()):
        return sensor_match

    return None


def detect_entity_type_from_device_class(
    device_class: str | None,
) -> InputType | None:
    """Map Home Assistant device class to our internal InputType enum."""
    # Import here to avoid circular import
    from .data.entity_type import InputType  # noqa: PLC0415

    if not device_class:
        return None

    device_class_lower = device_class.lower()

    # Map HA device classes to our InputType system
    device_class_to_input_type = {
        # Motion/Occupancy/Presence
        "motion": InputType.MOTION,
        "occupancy": InputType.MOTION,
        "presence": InputType.MOTION,
        # Doors and Windows
        "door": InputType.DOOR,
        "garage_door": InputType.DOOR,
        "opening": InputType.DOOR,
        "window": InputType.WINDOW,
        # Environmental sensors
        "temperature": InputType.ENVIRONMENTAL,
        "humidity": InputType.ENVIRONMENTAL,
        "illuminance": InputType.ENVIRONMENTAL,
        "pressure": InputType.ENVIRONMENTAL,
        "atmospheric_pressure": InputType.ENVIRONMENTAL,
        "co2": InputType.ENVIRONMENTAL,
        "carbon_dioxide": InputType.ENVIRONMENTAL,
        "co": InputType.ENVIRONMENTAL,
        "carbon_monoxide": InputType.ENVIRONMENTAL,
        "pm25": InputType.ENVIRONMENTAL,
        "pm10": InputType.ENVIRONMENTAL,
        "pm1": InputType.ENVIRONMENTAL,
        "volatile_organic_compounds": InputType.ENVIRONMENTAL,
        "volatile_organic_compounds_parts": InputType.ENVIRONMENTAL,
        "sound_pressure": InputType.ENVIRONMENTAL,
        "moisture": InputType.ENVIRONMENTAL,
        "nitrogen_dioxide": InputType.ENVIRONMENTAL,
        "nitrogen_monoxide": InputType.ENVIRONMENTAL,
        "ozone": InputType.ENVIRONMENTAL,
        "sulphur_dioxide": InputType.ENVIRONMENTAL,
        "aqi": InputType.ENVIRONMENTAL,
        # Power/Appliances/Energy
        "power": InputType.APPLIANCE,
        "energy": InputType.APPLIANCE,
        "running": InputType.APPLIANCE,
        "plug": InputType.APPLIANCE,
        "current": InputType.APPLIANCE,
        "voltage": InputType.APPLIANCE,
        "battery": InputType.APPLIANCE,
        # Media Players
        "tv": InputType.MEDIA,
        "speaker": InputType.MEDIA,
        "receiver": InputType.MEDIA,
    }

    return device_class_to_input_type.get(device_class_lower)


def get_entity_type_description(hass: HomeAssistant, entity_id: str) -> dict[str, Any]:
    """Get comprehensive entity type information for debugging and validation."""
    ha_state = hass.states.get(entity_id)
    ha_device_class = get_device_class(hass, entity_id)
    unit = get_unit_of_measurement(hass, entity_id)

    # Get both original and fallback device classes
    fallback_device_class = get_device_class_with_fallback(hass, entity_id)
    detected_input_type = detect_entity_type_from_device_class(fallback_device_class)

    # Extract keywords for debugging
    entity_words = _extract_entity_keywords(entity_id.lower())

    return {
        "entity_id": entity_id,
        "domain": entity_id.split(".")[0],
        "friendly_name": ha_state.name if ha_state else "Unknown",
        "ha_device_class": ha_device_class or "Unknown",
        "fallback_device_class": fallback_device_class or "Unknown",
        "unit_of_measurement": unit or "Unknown",
        "detected_input_type": detected_input_type.value
        if detected_input_type
        else "Unknown",
        "extracted_keywords": entity_words,
        "current_state": ha_state.state if ha_state else "Unknown",
        "available": ha_state is not None
        and ha_state.state not in ["unknown", "unavailable"],
    }
