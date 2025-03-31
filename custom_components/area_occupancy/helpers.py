"""Helper functions for Area Occupancy Detection integration."""

from __future__ import annotations

import logging

from homeassistant.core import HomeAssistant

from .const import ROUNDING_PRECISION

_LOGGER = logging.getLogger(__name__)


def format_float(value: float) -> float:
    """Format float to consistently show 2 decimal places."""
    try:
        return round(float(value), ROUNDING_PRECISION)
    except (ValueError, TypeError):
        return 0.0


def get_friendly_names(hass: HomeAssistant, entity_ids: list[str]) -> list[str]:
    """Convert entity IDs to friendly names."""
    return [
        hass.states.get(entity_id).attributes.get("friendly_name", entity_id)
        for entity_id in entity_ids
        if hass.states.get(entity_id)
    ]
