# custom_components/area_occupancy/validation_helpers.py

"""Validation helpers for Area Occupancy Detection."""


def validate_threshold(threshold: float) -> str | None:
    """Validate the threshold value."""
    if not 0 <= threshold <= 1:
        return "invalid_threshold"
    return None


def validate_decay_window(decay_window: int) -> str | None:
    """Validate the decay window."""
    if not 60 <= decay_window <= 3600:
        return "invalid_decay"
    return None


def validate_required_sensors(sensors: list[str]) -> str | None:
    """Validate that required sensors are provided."""
    if not sensors:
        return "no_motion_sensors"
    return None
