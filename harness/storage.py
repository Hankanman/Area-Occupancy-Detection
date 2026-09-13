"""Seeding of a throwaway instance's ``.storage`` files.

Writing the config entry directly, before Home Assistant starts, is what
makes migration testing possible: the config flow can only ever create an
entry at the current ``CONF_VERSION``, so a v18-shaped entry with its areas
still in the legacy list has to be fabricated. Seeding also skips the click
path entirely, so an instance comes up already configured.

Three files are written -- the area registry, the config entries store and
the HTTP config store. All are small, stable formats, and the version stamps
are imported from Home Assistant rather than hardcoded so they track the
pinned core.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from custom_components.area_occupancy.const import (
    CONF_ADJACENT_AREAS,
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_CUSTOM_ACTIVE_STATES,
    CONF_CUSTOM_ENTITY_ID,
    CONF_CUSTOM_SENSORS,
    CONF_CUSTOM_WEIGHT,
    CONF_DECAY_ENABLED,
    CONF_DECAY_HALF_LIFE,
    CONF_PURPOSE,
    CONF_THRESHOLD,
    CONF_VERSION,
    CONF_VERSION_MINOR,
    CONF_WASP_ENABLED,
    DEFAULT_DECAY_ENABLED,
    DEFAULT_DECAY_HALF_LIFE,
    DOMAIN,
    SUBENTRY_TYPE_AREA,
)
from homeassistant import config_entries as ha_config_entries
from homeassistant.components.http import config as ha_http_config
from homeassistant.helpers import area_registry as ha_area_registry
from homeassistant.util import dt as dt_util, ulid as ulid_util

from .profiles import CHANNELS, AreaSpec, Profile, area_entities

#: Entry version whose areas still live in the legacy ``CONF_AREAS`` list.
LEGACY_ENTRY_VERSION = 18

#: Entry versions the harness knows how to fabricate.
SUPPORTED_ENTRY_VERSIONS = (LEGACY_ENTRY_VERSION, CONF_VERSION)


def _now() -> str:
    """Current UTC time in the ISO form the stores use.

    Returns:
        An ISO 8601 timestamp with offset.
    """
    return dt_util.utcnow().isoformat()


def _store(key: str, version: int, minor: int, data: dict[str, Any]) -> dict[str, Any]:
    """Wrap payload data in the envelope ``Store`` expects on disk.

    Args:
        key: Storage key, which is also the file name.
        version: Major storage version.
        minor: Minor storage version.
        data: The payload.

    Returns:
        The full document to serialise.
    """
    return {"version": version, "minor_version": minor, "key": key, "data": data}


def _write(path: Path, document: dict[str, Any]) -> None:
    """Write a storage document, creating ``.storage`` if needed.

    Args:
        path: Destination file.
        document: The document to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def area_config(area: AreaSpec, profile: Profile) -> dict[str, Any]:
    """Build the stored configuration for one area.

    Only the keys a profile actually decides are written. Everything else is
    left absent so the integration's own defaults apply, which is both closer
    to a hand-configured entry and a standing test that those defaults hold.

    Args:
        area: The area to render.
        profile: The profile it belongs to, used to filter adjacency down to
            areas that exist.

    Returns:
        The area's configuration mapping.
    """
    config: dict[str, Any] = {
        CONF_AREA_ID: area.slug,
        CONF_PURPOSE: area.purpose,
        CONF_THRESHOLD: area.threshold,
        CONF_DECAY_ENABLED: DEFAULT_DECAY_ENABLED,
        # 0 means "use the purpose's half-life" -- the sentinel that has
        # regressed more than once, so seeds keep it in play by default.
        CONF_DECAY_HALF_LIFE: DEFAULT_DECAY_HALF_LIFE,
    }

    known = {other.slug for other in profile.areas}
    if adjacent := [slug for slug in area.adjacent if slug in known]:
        config[CONF_ADJACENT_AREAS] = adjacent

    for channel, entities in area_entities(area).items():
        spec = CHANNELS[channel]
        if spec.conf_key == CONF_CUSTOM_SENSORS:
            # Custom sensors carry their own states and weight per row.
            config[CONF_CUSTOM_SENSORS] = [
                {
                    CONF_CUSTOM_ENTITY_ID: entity,
                    CONF_CUSTOM_ACTIVE_STATES: list(spec.active_states),
                    CONF_CUSTOM_WEIGHT: 0.55,
                }
                for entity in entities
            ]
            continue
        config[spec.conf_key] = entities

    if area.wasp_enabled:
        config[CONF_WASP_ENABLED] = True

    return config


def write_area_registry(config_dir: Path, profile: Profile) -> None:
    """Create one HA area per profile area, keyed by its slug.

    The integration resolves an area's display name through the area
    registry, so a config entry pointing at an area id that does not exist
    there loads with fallback names and logs warnings.

    Args:
        config_dir: The instance's configuration directory.
        profile: The profile being seeded.
    """
    timestamp = _now()
    areas = [
        {
            "aliases": [],
            "floor_id": None,
            "humidity_entity_id": None,
            "icon": None,
            "id": area.slug,
            "labels": [],
            "name": area.name,
            "picture": None,
            "temperature_entity_id": None,
            "created_at": timestamp,
            "modified_at": timestamp,
        }
        for area in profile.areas
    ]
    _write(
        config_dir / ".storage" / ha_area_registry.STORAGE_KEY,
        _store(
            ha_area_registry.STORAGE_KEY,
            ha_area_registry.STORAGE_VERSION_MAJOR,
            ha_area_registry.STORAGE_VERSION_MINOR,
            {"areas": areas},
        ),
    )


def write_config_entry(
    config_dir: Path,
    profile: Profile,
    *,
    entry_version: int = CONF_VERSION,
    entry_id: str | None = None,
) -> str:
    """Write an ``area_occupancy`` config entry for the profile.

    Args:
        config_dir: The instance's configuration directory.
        profile: The profile being seeded.
        entry_version: ``CONF_VERSION`` to write a current entry with one
            subentry per area, or ``LEGACY_ENTRY_VERSION`` to write a
            pre-subentry entry whose areas sit in the legacy list, so that
            starting the instance runs the real migration.
        entry_id: Entry id to use, generated when omitted.

    Returns:
        The entry id that was written.

    Raises:
        ValueError: If ``entry_version`` is not one the harness can fabricate.
    """
    if entry_version not in SUPPORTED_ENTRY_VERSIONS:
        supported = ", ".join(str(version) for version in SUPPORTED_ENTRY_VERSIONS)
        raise ValueError(
            f"entry_version must be one of {supported}, got {entry_version}"
        )

    entry_id = entry_id or ulid_util.ulid_now()
    timestamp = _now()
    configs = [area_config(area, profile) for area in profile.areas]

    data: dict[str, Any] = {}
    subentries: list[dict[str, Any]] = []
    if entry_version == LEGACY_ENTRY_VERSION:
        data[CONF_AREAS] = configs
    else:
        subentries = [
            {
                "data": config,
                "subentry_id": ulid_util.ulid_now(),
                "subentry_type": SUBENTRY_TYPE_AREA,
                "title": area.name,
                "unique_id": area.slug,
            }
            for area, config in zip(profile.areas, configs, strict=True)
        ]

    entry = {
        "created_at": timestamp,
        "data": data,
        "disabled_by": None,
        "discovery_keys": {},
        "domain": DOMAIN,
        "entry_id": entry_id,
        # A legacy entry predates the minor version being used for anything,
        # so it carries the 1 that v18 instances have on disk.
        "minor_version": 1
        if entry_version == LEGACY_ENTRY_VERSION
        else CONF_VERSION_MINOR,
        "modified_at": timestamp,
        "options": {},
        "pref_disable_new_entities": False,
        "pref_disable_polling": False,
        "source": "user",
        "subentries": subentries,
        "title": "Area Occupancy Detection",
        "unique_id": None,
        "version": entry_version,
    }

    _write(
        config_dir / ".storage" / ha_config_entries.STORAGE_KEY,
        _store(
            ha_config_entries.STORAGE_KEY,
            ha_config_entries.STORAGE_VERSION,
            ha_config_entries.STORAGE_VERSION_MINOR,
            {"entries": [entry]},
        ),
    )
    return entry_id


def read_config_entry(config_dir: Path) -> dict[str, Any] | None:
    """Read the instance's ``area_occupancy`` config entry back from disk.

    Used by verification to inspect what a run actually persisted, including
    what the migration did to a seeded legacy entry.

    Args:
        config_dir: The instance's configuration directory.

    Returns:
        The entry mapping, or ``None`` if there is no store or no entry for
        the integration.
    """
    path = config_dir / ".storage" / ha_config_entries.STORAGE_KEY
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    for entry in document.get("data", {}).get("entries", []):
        if entry.get("domain") == DOMAIN:
            return entry
    return None


def write_http_config(config_dir: Path, port: int) -> None:
    """Pin the instance's port as Home Assistant's confirmed HTTP config.

    Since 2026.9 the HTTP config is a user-managed store with a confirmed
    ``stable`` config and an unconfirmed ``pending`` one. A port that arrives
    any other way -- in YAML, or as a changed built-in default -- is staged as
    a pending *trial*: if nothing promotes it within five minutes Home
    Assistant reverts to stable and restarts itself to do it. For a harness
    instance that means the process exits five minutes in, and the next start
    looks for a port nothing is listening on.

    Writing the port straight into ``stable`` with no pending config, and
    marking the YAML migration done so a stray ``http:`` block cannot restage
    it, is what makes an instance's port survive as long as the instance
    does.

    Args:
        config_dir: The instance's configuration directory.
        port: Port the instance should listen on.
    """
    stable = {
        **ha_http_config.HTTP_STORAGE_SCHEMA({"server_port": port}),
        "created_at": _now(),
        "error": None,
        "error_message": None,
    }
    _write(
        config_dir / ".storage" / ha_http_config.STORAGE_KEY,
        _store(
            ha_http_config.STORAGE_KEY,
            ha_http_config.STORAGE_VERSION,
            ha_http_config.STORAGE_MINOR_VERSION,
            {
                ha_http_config.KEY_STABLE: stable,
                ha_http_config.KEY_PENDING: None,
                ha_http_config.KEY_YAML_MIGRATION_DONE: True,
            },
        ),
    )
