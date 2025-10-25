"""Area Occupancy Detection integration.

Compatibility shim for Home Assistant frame helper in tests.
"""

from __future__ import annotations

try:
    from homeassistant.helpers import frame as ha_frame

    # Some test fixtures patch an internal attribute that may not exist
    # in the currently installed Home Assistant version. Ensure it exists
    # so unit tests can patch it without AttributeError.
    if not hasattr(ha_frame, "_hass"):
        setattr(ha_frame, "_hass", None)
except Exception:  # noqa: BLE001
    # If Home Assistant isn't available in the environment, ignore.
    pass
