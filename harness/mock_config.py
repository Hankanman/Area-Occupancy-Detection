"""Generation of a throwaway instance's ``configuration.yaml``.

The sensors an area config points at have to exist, or the integration loads
with every entity unavailable and nothing can be exercised. This module
renders a profile into real HA entities: an ``input_*`` helper per mock sensor
as the knob, and a template (or universal media player) entity on top of it as
the thing the integration watches. Flipping a knob moves the sensor the
integration sees, which is what makes a seeded instance interactive.

Only the components the harness actually needs are configured -- no
``default_config`` -- so an instance boots in seconds and its log is not
buried in missing-dependency errors for integrations nobody is testing.
"""

from __future__ import annotations

from typing import Any

import yaml

from .profiles import CHANNELS, AreaSpec, ChannelSpec, Profile

#: Prefix for the helper entities that drive the mock sensors.
KNOB_PREFIX = "mock"


def knob_id(area: AreaSpec, channel: str, index: int) -> str:
    """Object id of the helper entity behind one mock sensor.

    Args:
        area: The area the sensor belongs to.
        channel: Channel name.
        index: 1-based index within the channel.

    Returns:
        The helper's object id, without its domain.
    """
    return f"{KNOB_PREFIX}_{area.slug}_{channel}_{index}"


def _friendly(area: AreaSpec, channel: str, index: int) -> str:
    """Human-readable name whose slug matches the expected entity id.

    Args:
        area: The area the sensor belongs to.
        channel: Channel name.
        index: 1-based index within the channel.

    Returns:
        A name that HA slugifies back to ``<slug>_<channel>_<index>``.
    """
    return f"{area.name} {channel.replace('_', ' ').title()} {index}"


def _numeric_bounds(spec: ChannelSpec) -> tuple[float, float, float]:
    """Slider bounds and step for a numeric channel's knob.

    Args:
        spec: The channel being rendered.

    Returns:
        Tuple of ``(minimum, maximum, step)``.

    Raises:
        ValueError: If the channel is not numeric.
    """
    model = spec.numeric
    if model is None:
        raise ValueError("channel is not numeric")
    low = min(model.empty_mean, model.occupied_mean) - 6 * model.sd
    high = max(model.empty_mean, model.occupied_mean) + 6 * model.sd
    step = 1.0 if model.decimals == 0 else 10.0**-model.decimals
    return (round(low, model.decimals), round(high, model.decimals), step)


def _helpers(profile: Profile) -> dict[str, Any]:
    """Build the ``input_*`` helper sections that drive the mock sensors.

    Args:
        profile: The profile being rendered.

    Returns:
        Mapping of helper domain to its entity definitions, omitting domains
        the profile does not need.
    """
    booleans: dict[str, Any] = {}
    selects: dict[str, Any] = {}
    numbers: dict[str, Any] = {}

    for area in profile.areas:
        for channel, count in area.channels.items():
            spec = CHANNELS[channel]
            for index in range(1, count + 1):
                object_id = knob_id(area, channel, index)
                name = f"{area.name}: {channel} {index}"
                if spec.is_numeric:
                    low, high, step = _numeric_bounds(spec)
                    numbers[object_id] = {
                        "name": name,
                        "min": low,
                        "max": high,
                        "step": step,
                        "initial": spec.numeric.empty_mean,
                        "mode": "box",
                    }
                elif spec.domain == "media_player":
                    selects[object_id] = {
                        "name": name,
                        "options": [*spec.active_states, "idle", spec.idle_state],
                        "initial": spec.idle_state,
                    }
                    numbers[f"{object_id}_volume"] = {
                        "name": f"{name} volume",
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "initial": 40,
                    }
                else:
                    booleans[object_id] = {"name": name, "initial": "off"}

    sections: dict[str, Any] = {}
    if booleans:
        sections["input_boolean"] = booleans
    if selects:
        sections["input_select"] = selects
    if numbers:
        sections["input_number"] = numbers
    return sections


def _template_entities(profile: Profile) -> list[dict[str, Any]]:
    """Build the ``template:`` blocks mirroring each knob as a real sensor.

    Args:
        profile: The profile being rendered.

    Returns:
        A list suitable for the top-level ``template`` key, with one
        ``binary_sensor`` block and one ``sensor`` block where needed.
    """
    binary: list[dict[str, Any]] = []
    numeric: list[dict[str, Any]] = []

    for area in profile.areas:
        for channel, count in area.channels.items():
            spec = CHANNELS[channel]
            if spec.domain == "media_player":
                continue
            for index in range(1, count + 1):
                object_id = knob_id(area, channel, index)
                entity: dict[str, Any] = {
                    "name": _friendly(area, channel, index),
                    "unique_id": f"{area.slug}_{channel}_{index}",
                }
                if spec.device_class:
                    entity["device_class"] = spec.device_class
                if spec.is_numeric:
                    entity["unit_of_measurement"] = spec.unit
                    entity["state"] = f"{{{{ states('input_number.{object_id}') }}}}"
                    entity["state_class"] = "measurement"
                    numeric.append(entity)
                else:
                    entity["state"] = f"{{{{ states('input_boolean.{object_id}') }}}}"
                    binary.append(entity)

    blocks: list[dict[str, Any]] = []
    if binary:
        blocks.append({"binary_sensor": binary})
    if numeric:
        blocks.append({"sensor": numeric})
    return blocks


def _media_players(profile: Profile) -> list[dict[str, Any]]:
    """Build ``media_player:`` universal platforms backed by input helpers.

    Args:
        profile: The profile being rendered.

    Returns:
        A list of universal media player platform configs, empty when the
        profile has no media channels.
    """
    players: list[dict[str, Any]] = []
    for area in profile.areas:
        for channel, count in area.channels.items():
            if CHANNELS[channel].domain != "media_player":
                continue
            for index in range(1, count + 1):
                object_id = knob_id(area, channel, index)
                state_select = f"input_select.{object_id}"
                volume = f"input_number.{object_id}_volume"
                players.append(
                    {
                        "platform": "universal",
                        "name": _friendly(area, channel, index),
                        "unique_id": f"{area.slug}_{channel}_{index}",
                        "state_template": f"{{{{ states('{state_select}') }}}}",
                        "volume_level": f"{{{{ states('{volume}') | float / 100 }}}}",
                        "commands": {
                            command: {
                                "action": "input_select.select_option",
                                "data": {"entity_id": state_select, "option": option},
                            }
                            for command, option in (
                                ("turn_on", "idle"),
                                ("turn_off", "off"),
                                ("media_play", "playing"),
                                ("media_pause", "paused"),
                                ("media_stop", "idle"),
                            )
                        },
                    }
                )
    return players


def build(
    profile: Profile, *, time_zone: str, frontend: bool, port: int
) -> dict[str, Any]:
    """Assemble the full configuration for a profile.

    Args:
        profile: The profile to render.
        time_zone: IANA time zone for the instance. Prior time slots are
            local-time based, so this is worth varying deliberately.
        frontend: Whether to load the frontend. It is needed to click through
            the UI and to reach onboarding, and costs a few seconds of boot.
        port: Port to listen on. Pinned here rather than left at the default
            so several instances can run side by side.

    Returns:
        The configuration as a mapping ready to be dumped to YAML.
    """
    config: dict[str, Any] = {
        "homeassistant": {
            "name": f"AOD {profile.name}",
            "time_zone": time_zone,
            "unit_system": "metric",
            "country": "GB",
            "currency": "GBP",
            # Positioned so sun/elevation templates behave sensibly.
            "latitude": 51.5,
            "longitude": -0.12,
            "elevation": 25,
        },
        # Pinned so instances do not collide on 8123. Changing it means
        # rebuilding the instance, since it is baked in here.
        "http": {"server_port": port},
        # The recorder is not optional: the analysis pipeline's first step
        # imports entity history through it, and get_instance() raises
        # without it.
        "recorder": {"commit_interval": 1, "purge_keep_days": 30},
        "history": {},
        "logger": {
            "default": "warning",
            "logs": {"custom_components.area_occupancy": "debug"},
        },
    }
    if frontend:
        # Pulls in api, auth, config, onboarding, websocket_api and lovelace,
        # which is everything the harness client and a browsing dev need.
        config["frontend"] = {}
    else:
        config["api"] = {}
        config["auth"] = {}
        config["config"] = {}
        config["onboarding"] = {}
        config["websocket_api"] = {}

    config.update(_helpers(profile))
    if templates := _template_entities(profile):
        config["template"] = templates
    if players := _media_players(profile):
        config["media_player"] = players
    return config


def render(profile: Profile, *, time_zone: str, frontend: bool, port: int) -> str:
    """Render a profile's configuration as YAML text.

    Args:
        profile: The profile to render.
        time_zone: IANA time zone for the instance.
        frontend: Whether to load the frontend.
        port: Port to listen on.

    Returns:
        The YAML document, with a header explaining where it came from.
    """
    config = build(profile, time_zone=time_zone, frontend=frontend, port=port)
    body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False, width=100)
    return (
        "# Generated by scripts/harness -- do not edit.\n"
        f"# Profile: {profile.name} ({profile.description})\n"
        "#\n"
        "# Each mock sensor is a template entity over an input helper. Flip the\n"
        "# input_boolean / input_select / input_number to move what the\n"
        "# integration sees.\n\n"
    ) + body
