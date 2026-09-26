"""Upgrade testing: live on one release, then swap in another and diff.

Every other harness instance runs the working tree against storage the
harness wrote in the *current* format. That cannot catch an upgrade bug,
because the thing an upgrade has to survive is state a *previous* release
wrote. This module builds that state the honest way:

1. ``upgrade-base`` installs a released version's integration code (a copy of
   ``git archive <ref>``, not the working-tree symlink), boots it, backfills
   the profile's synthetic history into the **recorder** so the old code
   learns everything itself, adds the integration through that version's real
   config flow, applies the customisations a user would have made, and
   captures and snapshots the result as ``baseline``.
2. ``upgrade`` restores that snapshot, swaps in another ref (or the working
   tree), boots it, and diffs a second capture against the baseline: config
   entry and per-area values, entity registry (renames, enabled state,
   unique ids), entity availability, services, persisted stores, the
   integration database, diagnostics, and log errors. The standard
   ``harness verify`` checks run on top.

The config flow is filled generically: each form's schema comes back from
Home Assistant, and every field the profile's area config has a value for is
filled in (inside collapsible sections too). Fields an old release does not
have are skipped, which is exactly what a user on that release could set.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from typing import Any

import aiohttp

from custom_components.area_occupancy.const import DOMAIN
from homeassistant.helpers import floor_registry as ha_floor_registry

from . import history, mock_config, storage
from .client import ApiError, Client
from .instance import MARKER, REPO_ROOT, Instance, InstanceError, free_port
from .profiles import CHANNELS, Profile, get_profile

#: Snapshot the baseline is stored under, next to the instance directory.
BASELINE = "baseline"

#: Purposes whose areas the base puts upstairs, so floor sensors get built.
UPSTAIRS_PURPOSES = frozenset({"sleeping", "bathroom"})

#: The customisations ``upgrade-base`` applies, which the upgrade must keep.
#: The aggregate occupancy sensor is renamed rather than an area one, so the
#: standard ``entities`` check (which finds area entities by name) still holds.
RENAMED_NAME = "Anyone Home"
THRESHOLD_OVERRIDE = 45.0

#: Wizard steps of the 2026.7-2026.8 single-entry config flow that need the
#: user to pick from a menu to continue.
MENU_ADD_AREA = "add_area"
MENU_FINISH = "finish_setup"

#: Log noise that is not the integration's fault.
_IGNORED_LOG = (
    "has not been tested by Home Assistant",
    # The harness client refreshing an expired token; the URL names us.
    "invalid authentication",
)


class UpgradeError(InstanceError):
    """The base could not be built, or the upgrade could not be run."""


# -- code installation ---------------------------------------------------


def resolve_ref(ref: str) -> tuple[str, str]:
    """Resolve a git ref to a short sha and the manifest version it carries.

    Args:
        ref: A tag, branch or sha, or ``working`` for the working tree.

    Returns:
        ``(sha, manifest_version)``.

    Raises:
        UpgradeError: If git cannot resolve the ref.
    """
    manifest = f"custom_components/{DOMAIN}/manifest.json"
    if ref == "working":
        sha = _git("rev-parse", "--short", "HEAD") + "+working"
        version = json.loads((REPO_ROOT / manifest).read_text(encoding="utf-8"))
        return sha, version["version"]
    sha = _git("rev-parse", "--short", f"{ref}^{{commit}}")
    version = json.loads(_git("show", f"{ref}:{manifest}"))
    return sha, version["version"]


def _git(*args: str) -> str:
    """Run a git command in the repo and return its stdout.

    Args:
        *args: Arguments after ``git``.

    Returns:
        Stripped standard output.

    Raises:
        UpgradeError: If git fails.
    """
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise UpgradeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def conf_version(ref: str) -> int:
    """The config entry version a ref's code creates entries at.

    Args:
        ref: A git ref, or ``working``.

    Returns:
        That ref's ``CONF_VERSION``.

    Raises:
        UpgradeError: If it cannot be found.
    """
    path = f"custom_components/{DOMAIN}/const.py"
    source = (
        (REPO_ROOT / path).read_text(encoding="utf-8")
        if ref == "working"
        else _git("show", f"{ref}:{path}")
    )
    match = re.search(r"^CONF_VERSION(?::\s*\w+)?\s*=\s*(\d+)", source, re.MULTILINE)
    if match is None:
        raise UpgradeError(f"no CONF_VERSION in {ref}:{path}")
    return int(match.group(1))


def _stored_entry_version(instance: Instance, timeout: float = 60.0) -> int | None:
    """Wait for the entry to reach disk and return its stored version.

    Home Assistant saves config entries on a delay, so an entry the flow
    just created is not in ``core.config_entries`` straight away.

    Args:
        instance: The running instance.
        timeout: Seconds to wait.

    Returns:
        The stored version, or ``None`` if it never appeared.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        entry = storage.read_config_entry(instance.path)
        if entry and entry.get("entry_id") == instance.entry_id:
            return int(entry["version"])
        time.sleep(1.0)
    return None


def install_code(instance: Instance, ref: str) -> str:
    """Point the instance's ``custom_components`` at a ref's integration code.

    ``working`` restores the harness's usual symlink to the working tree; any
    other ref is extracted as a real copy, so later edits to the checkout
    cannot leak into it.

    Args:
        instance: The (stopped) instance.
        ref: Git ref, or ``working``.

    Returns:
        A one-line description of what was installed.
    """
    sha, version = resolve_ref(ref)
    target = instance.path / "custom_components"
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)

    if ref == "working":
        target.symlink_to(REPO_ROOT / "custom_components", target_is_directory=True)
    else:
        archive = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "archive",
                ref,
                f"custom_components/{DOMAIN}",
            ],
            capture_output=True,
            check=True,
        ).stdout
        subprocess.run(
            ["tar", "-x", "-C", str(instance.path)],
            input=archive,
            check=True,
        )

    instance.meta["code"] = {"ref": ref, "sha": sha, "version": version}
    instance.save()
    return f"{ref} ({sha}, manifest {version})"


# -- building the base ---------------------------------------------------


def _write_floors(instance: Instance, profile: Profile) -> None:
    """Add two floors and put the profile's areas on them.

    Floors matter to an upgrade test because the integration builds floor
    sensors from them, and those are registry entries the upgrade must keep.

    Args:
        instance: The instance being built.
        profile: Its profile.
    """
    now = storage._now()  # noqa: SLF001
    floors = [
        {
            "aliases": [],
            "floor_id": floor_id,
            "icon": None,
            "labels": [],
            "level": level,
            "name": name,
            "created_at": now,
            "modified_at": now,
        }
        for floor_id, name, level in (
            ("ground_floor", "Ground Floor", 0),
            ("first_floor", "First Floor", 1),
        )
    ]
    storage._write(  # noqa: SLF001
        instance.path / ".storage" / ha_floor_registry.STORAGE_KEY,
        storage._store(  # noqa: SLF001
            ha_floor_registry.STORAGE_KEY,
            ha_floor_registry.STORAGE_VERSION_MAJOR,
            ha_floor_registry.STORAGE_VERSION_MINOR,
            {"floors": floors},
        ),
    )
    areas_path = instance.path / ".storage" / "core.area_registry"
    document = json.loads(areas_path.read_text(encoding="utf-8"))
    purposes = {area.slug: area.purpose for area in profile.areas}
    for area in document["data"]["areas"]:
        upstairs = purposes.get(area["id"]) in UPSTAIRS_PURPOSES
        area["floor_id"] = "first_floor" if upstairs else "ground_floor"
    areas_path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def write_recorder_history(
    instance: Instance, histories: dict[str, history.AreaHistory]
) -> int:
    """Backfill generated sensor history into the recorder database.

    The regular harness seeds the integration's own database directly, in the
    current schema. An old release has to learn from the recorder like it
    would in a real home, so here the same generated history becomes
    recorder ``states`` rows instead. Home Assistant must be stopped, and
    must have run once so the recorder schema exists.

    Args:
        instance: The stopped instance.
        histories: Generated history, keyed by area slug.

    Returns:
        The number of state rows written.

    Raises:
        UpgradeError: If the recorder database does not exist yet.
    """
    db_path = instance.path / "home-assistant_v2.db"
    if not db_path.exists():
        raise UpgradeError(f"{db_path} missing; start the instance once first")

    rows = 0
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        meta = dict(cursor.execute("select entity_id, metadata_id from states_meta"))
        for area_history in histories.values():
            for sensor in area_history.sensors:
                spec = CHANNELS[sensor.channel]
                if spec.is_numeric:
                    changes = [(ts, str(value)) for ts, value in sensor.samples]
                else:
                    changes = [
                        (
                            run.start,
                            spec.active_states[0] if run.active else spec.idle_state,
                        )
                        for run in sensor.runs
                    ]
                if not changes:
                    continue
                entity = sensor.entity_id
                if entity not in meta:
                    cursor.execute(
                        "insert into states_meta(entity_id) values (?)", (entity,)
                    )
                    meta[entity] = cursor.lastrowid
                first = changes[0][0].timestamp()
                # Drop whatever the first boot recorded before the window, so
                # the backfill is the entity's whole history.
                cursor.execute(
                    "delete from states where metadata_id = ? and last_updated_ts >= ?",
                    (meta[entity], first),
                )
                previous_id = previous_state = None
                for moment, state in sorted(changes):
                    if state == previous_state:
                        continue
                    cursor.execute(
                        "insert into states(state, last_updated_ts, old_state_id,"
                        " origin_idx, metadata_id) values (?, ?, ?, 0, ?)",
                        (state, moment.timestamp(), previous_id, meta[entity]),
                    )
                    previous_id, previous_state = cursor.lastrowid, state
                    rows += 1
        connection.commit()
    finally:
        connection.close()
    return rows


def _fill_form(form: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    """Answer a flow form from a flat config mapping.

    Args:
        form: The form result, carrying ``data_schema``.
        values: Flat area config; only keys the form asks for are used.

    Returns:
        The user input to submit, with section fields nested.
    """

    def answer(field: dict[str, Any]) -> Any:
        value = values[field["name"]]
        if "duration" in (field.get("selector") or {}) and isinstance(
            value, (int, float)
        ):
            return {"seconds": int(value)}
        return value

    user_input: dict[str, Any] = {}
    for field in form.get("data_schema") or []:
        if field.get("type") == "expandable":
            section = {
                child["name"]: answer(child)
                for child in field["schema"]
                if child["name"] in values
            }
            user_input[field["name"]] = section
        elif field["name"] in values:
            user_input[field["name"]] = answer(field)
    return user_input


def _base_area_config(profile: Profile) -> list[dict[str, Any]]:
    """Area configs to enter, with the customisations the base exercises.

    Starts from the harness's own ``storage.area_config`` so the base means
    the same house as every other instance, then adds the settings a user
    changes by hand and a migration has to carry: a custom decay half-life
    and motion timeout, a threshold, a minimum prior, exclusion from "All
    Areas", and an inverted door.

    Args:
        profile: The profile being built.

    Returns:
        One flat config per area, in profile order.
    """
    configs = []
    for area in profile.areas:
        config = storage.area_config(area, profile)
        if area.purpose == "social":
            config |= {"decay_half_life": 300, "threshold": 60.0}
        if area.purpose == "food_prep":
            config |= {"motion_timeout": 120}
        if area.purpose == "passageway":
            config |= {"exclude_from_all_areas": True}
        if area.purpose == "bathroom":
            config |= {
                "min_prior_override": 0.05,
                "door_active_state": "closed",
                "wasp_weight": 0.85,
            }
        configs.append(config)
    return configs


def run_config_flow(client: Client, profile: Profile) -> tuple[str, list[str]]:
    """Add the integration through the installed version's config flow.

    Supports the single-entry wizard of 2026.7-2026.8 (area wizard steps,
    then an ``add_area`` / ``finish_setup`` menu). Adjacent areas are only
    offered once they exist, so each area's adjacency is trimmed to the
    areas already added.

    Args:
        client: An authenticated client.
        profile: The profile to configure.

    Returns:
        The new entry id, and notes on config keys the flow had no field for.

    Raises:
        UpgradeError: If the flow shape is not one this supports, or a step
            is rejected.
    """
    flow = "/api/config/config_entries/flow"
    result = client.post(flow, {"handler": DOMAIN})
    flow_id = result["flow_id"]
    added: list[str] = []
    notes: list[str] = []

    def submit(user_input: dict[str, Any]) -> dict[str, Any]:
        try:
            step = client.post(f"{flow}/{flow_id}", user_input)
        except ApiError as err:
            client.request("DELETE", f"{flow}/{flow_id}")
            raise UpgradeError(f"config flow rejected {user_input}: {err}") from err
        if step.get("errors"):
            client.request("DELETE", f"{flow}/{flow_id}")
            raise UpgradeError(f"config flow errors {step['errors']} for {user_input}")
        return step

    for index, config in enumerate(_base_area_config(profile)):
        if index:
            if MENU_ADD_AREA not in (result.get("menu_options") or []):
                raise UpgradeError(f"expected an add_area menu, got {result}")
            result = submit({"next_step_id": MENU_ADD_AREA})
        wanted = dict(config)
        if "adjacent_areas" in wanted:
            wanted["adjacent_areas"] = [
                a for a in wanted["adjacent_areas"] if a in added
            ]
            if not wanted["adjacent_areas"]:
                del wanted["adjacent_areas"]
        used: set[str] = set()
        while result.get("type") == "form":
            answer = _fill_form(result, wanted)
            used |= set(answer) | {
                key
                for value in answer.values()
                if isinstance(value, dict)
                for key in value
            }
            result = submit(answer)
        if skipped := sorted(set(wanted) - used):
            notes.append(f"{config['area_id']}: no field for {', '.join(skipped)}")
        added.append(config["area_id"])

    if MENU_FINISH not in (result.get("menu_options") or []):
        raise UpgradeError(f"expected a finish_setup menu, got {result}")
    result = submit({"next_step_id": MENU_FINISH})
    if result.get("type") != "create_entry":
        raise UpgradeError(f"config flow did not create an entry: {result}")
    return result["result"]["entry_id"], notes


def add_person(client: Client, entry_id: str, profile: Profile) -> None:
    """Configure the owner as a person with sleep detection in the bedroom.

    Args:
        client: An authenticated client.
        entry_id: The integration's config entry.
        profile: The profile, used to find the sleeping area.

    Raises:
        UpgradeError: If the options flow does not accept the person.
    """
    bedroom = next((a for a in profile.areas if a.purpose == "sleeping"), None)
    if bedroom is None:
        return
    person = next(
        s["entity_id"] for s in client.states() if s["entity_id"].startswith("person.")
    )
    sensor = next(
        e["entity_id"]
        for e in client.states()
        if e["entity_id"].startswith("binary_sensor.")
        and bedroom.slug in e["entity_id"]
    )
    result = client.start_options_flow(entry_id)
    for user_input in (
        {"next_step_id": "manage_people"},
        {"selected_option": "add_person"},
        {
            "person_entity": person,
            "sleep_sensors": [sensor],
            "sleep_area_id": bedroom.slug,
            "confidence_threshold": 70,
        },
    ):
        result = client.advance_options_flow(result["flow_id"], user_input)
        if result.get("errors"):
            raise UpgradeError(f"person options flow rejected {user_input}: {result}")
    if result.get("type") != "create_entry":
        raise UpgradeError(f"person options flow did not finish: {result}")


def _ws(instance: Instance, messages: list[dict[str, Any]]) -> list[Any]:
    """Send websocket commands and return their results in order.

    The entity registry is only writable over the websocket API, which the
    stdlib client does not speak.

    Args:
        instance: A running instance.
        messages: Commands, without ids.

    Returns:
        Each command's result.
    """

    # Any authenticated call refreshes an expired token, so the websocket
    # authenticates with a live one.
    client = instance.client()
    client.get("/api/")

    async def run() -> list[Any]:
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(f"{instance.base_url}/api/websocket") as ws,
        ):
            await ws.receive_json()
            await ws.send_json({"type": "auth", "access_token": client.token})
            if (await ws.receive_json())["type"] != "auth_ok":
                raise UpgradeError("websocket auth failed")
            results = []
            for number, message in enumerate(messages, start=1):
                await ws.send_json({**message, "id": number})
                while True:
                    reply = await ws.receive_json()
                    if reply.get("id") == number and reply["type"] == "result":
                        if not reply["success"]:
                            raise UpgradeError(f"{message['type']}: {reply['error']}")
                        results.append(reply["result"])
                        break
            return results

    return asyncio.run(run())


def customise(instance: Instance, client: Client) -> list[str]:
    """Make the changes a user would, which the upgrade must preserve.

    Enables the (default-disabled) diagnostic entities of two areas, renames
    and re-ids the "All Areas" occupancy sensor, and moves a threshold slider.

    Args:
        instance: The running instance.
        client: An authenticated client.

    Returns:
        A description of each change.
    """
    registry = [
        entry
        for entry in _ws(instance, [{"type": "config/entity_registry/list"}])[0]
        if entry["platform"] == DOMAIN
    ]
    slugs = [area.slug for area in instance.profile.areas]
    enable = [
        entry["entity_id"]
        for entry in registry
        if entry["disabled_by"]
        and any(slug in entry["entity_id"] for slug in slugs[:2])
    ]
    occupancy = next(
        entry["entity_id"]
        for entry in registry
        if entry["entity_id"].startswith("binary_sensor.all_areas")
        and entry["entity_id"].endswith("occupancy_status")
    )
    renamed = "binary_sensor." + RENAMED_NAME.lower().replace(" ", "_")
    threshold = next(
        entry["entity_id"]
        for entry in registry
        if entry["entity_id"].startswith(f"number.{slugs[-1]}")
        and "threshold" in entry["entity_id"]
    )
    _ws(
        instance,
        [
            *(
                {
                    "type": "config/entity_registry/update",
                    "entity_id": eid,
                    "disabled_by": None,
                }
                for eid in enable
            ),
            {
                "type": "config/entity_registry/update",
                "entity_id": occupancy,
                "name": RENAMED_NAME,
                "new_entity_id": renamed,
            },
        ],
    )
    client.call_service(
        "number", "set_value", {"entity_id": threshold, "value": THRESHOLD_OVERRIDE}
    )
    return [
        f"enabled {len(enable)} diagnostic entities",
        f"renamed {occupancy} -> {renamed} ({RENAMED_NAME!r})",
        f"set {threshold} to {THRESHOLD_OVERRIDE}",
    ]


def walk(client: Client, profile: Profile, *, laps: int, dwell: float = 20.0) -> None:
    """Walk through the house live, one area's first motion sensor at a time.

    Gives the trajectory tracker, the online prior and the shadow learners
    real ticks to observe, which backfilled history alone does not.

    Args:
        client: An authenticated client.
        profile: The profile whose areas to walk.
        laps: How many passes through every area.
        dwell: Seconds to hold each area's motion on.
    """
    for _ in range(laps):
        for area in profile.areas:
            if "motion" not in area.channels:
                continue
            knob = f"input_boolean.{mock_config.knob_id(area, 'motion', 1)}"
            client.set_state(knob, "on")
            time.sleep(dwell)
            client.set_state(knob, "off")
            time.sleep(5.0)


def run_analysis(instance: Instance) -> None:
    """Run the integration's analysis and wait for it to finish.

    Args:
        instance: The running instance.
    """
    client = instance.client()
    client.timeout = 900.0
    client.call_service(DOMAIN, "run_analysis", {}, return_response=True)


# -- capture and diff ----------------------------------------------------


def _areas_by_id(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-area config from an entry, whichever shape it is stored in.

    Args:
        entry: A config entry as stored in ``core.config_entries``.

    Returns:
        Area id to that area's stored config.
    """
    areas = {a["area_id"]: a for a in entry["data"].get("areas", [])}
    for sub in entry.get("subentries") or []:
        if sub.get("subentry_type") == "area":
            areas[sub.get("unique_id") or sub["data"].get("area_id")] = sub["data"]
    return areas


def _db_summary(path: Path) -> dict[str, Any]:
    """Row counts, metadata and global priors from the integration database.

    Args:
        path: The database file.

    Returns:
        The summary, empty if there is no database.
    """
    if not path.exists():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table' and name not like 'sqlite_%'"
            )
        ]
        counts = {
            table: connection.execute(f'select count(*) from "{table}"').fetchone()[0]  # noqa: S608
            for table in tables
        }
        metadata = (
            dict(connection.execute("select key, value from metadata"))
            if "metadata" in tables
            else {}
        )
        priors = {}
        if "global_priors" in tables:
            priors = {
                name: round(value, 4)
                for name, value in connection.execute(
                    "select area_name, prior_value from global_priors"
                )
            }
        return {"tables": counts, "metadata": metadata, "global_priors": priors}
    finally:
        connection.close()


def _log_problems(instance: Instance) -> list[str]:
    """The integration's errors and warnings in the current run's log.

    An entry counts when its log line or its traceback mentions the
    integration, so an exception raised in our code but logged by core (a
    failed platform setup, say) is caught, while another component's
    missing dependency is not.

    Args:
        instance: The instance.

    Returns:
        Problem lines, each followed by its indented traceback lines.
    """
    log = instance.path / "home-assistant.log"
    if not log.exists():
        return []
    entries: list[list[str]] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        head = re.match(r"^\d{4}-\d\d-\d\d \S+ (\w+) ", line)
        if head:
            entries.append([head.group(1), line[:400]])
        elif entries and line.strip():
            entries[-1].append("    " + line.strip()[:300])
    problems: list[str] = []
    for level, *lines in entries:
        text = "\n".join(lines)
        if (
            level in ("ERROR", "CRITICAL", "WARNING")
            and DOMAIN in text
            and not any(noise in text for noise in _IGNORED_LOG)
        ):
            problems.extend(lines)
    return problems[:400]


def capture(instance: Instance, label: str) -> dict[str, Any]:
    """Record everything the diff compares, and save it in the instance.

    Args:
        instance: The running instance.
        label: Name for the capture file under ``captures/``.

    Returns:
        The capture.
    """
    client = instance.client()
    registry, devices = _ws(
        instance,
        [
            {"type": "config/entity_registry/list"},
            {"type": "config/device_registry/list"},
        ],
    )
    ours = [entry for entry in registry if entry["platform"] == DOMAIN]
    ids = {entry["entity_id"] for entry in ours}
    try:
        diagnostics = client.get(f"/api/diagnostics/config_entry/{instance.entry_id}")
        diagnostics_ok = True
    except ApiError as err:
        diagnostics, diagnostics_ok = str(err), False
    services = next(
        (d["services"] for d in client.get("/api/services") if d["domain"] == DOMAIN),
        {},
    )
    entry = storage.read_config_entry(instance.path) or {}
    stores = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((instance.path / ".storage").glob(f"{DOMAIN}.*"))
        if not path.name.endswith((".db", ".backup", "-wal", "-shm"))
    }
    data = {
        "captured_at": datetime.now(UTC).isoformat(),
        "code": instance.meta.get("code"),
        "entry": {
            "version": entry.get("version"),
            "minor_version": entry.get("minor_version"),
            "areas": _areas_by_id(entry) if entry else {},
            "data": {k: v for k, v in entry.get("data", {}).items() if k != "areas"},
            "options": entry.get("options"),
            "subentries": len(entry.get("subentries") or []),
        },
        "registry": {
            entry["unique_id"]: {
                key: entry.get(key)
                for key in (
                    "entity_id",
                    "name",
                    "original_name",
                    "disabled_by",
                    "hidden_by",
                    "entity_category",
                    "area_id",
                )
            }
            for entry in ours
        },
        "devices": sorted(
            (d.get("name_by_user") or d.get("name") or "?", d.get("sw_version"))
            for d in devices
            if instance.entry_id in (d.get("config_entries") or [])
        ),
        "states": {
            s["entity_id"]: {"state": s["state"], "attributes": sorted(s["attributes"])}
            for s in client.states()
            if s["entity_id"] in ids
        },
        "services": sorted(services),
        "diagnostics_ok": diagnostics_ok,
        "diagnostics": diagnostics,
        "stores": stores,
        "db": _db_summary(instance.path / ".storage" / f"{DOMAIN}.db"),
        "log_problems": _log_problems(instance),
    }
    folder = instance.path / "captures"
    folder.mkdir(exist_ok=True)
    (folder / f"{label}.json").write_text(
        json.dumps(data, indent=1, default=str), encoding="utf-8"
    )
    return data


def _changes(before: Any, after: Any, path: str = "") -> list[str]:
    """Leaf-level differences between two JSON-like values.

    Args:
        before: The old value.
        after: The new value.
        path: Dotted path of this value, for messages.

    Returns:
        ``- removed``, ``+ added`` and ``~ changed`` lines.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        lines = []
        for key in sorted(set(before) | set(after), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in after:
                lines.append(
                    f"- removed {sub} (was {json.dumps(before[key], default=str)[:100]})"
                )
            elif key not in before:
                lines.append(
                    f"+ added {sub} = {json.dumps(after[key], default=str)[:100]}"
                )
            else:
                lines.extend(_changes(before[key], after[key], sub))
        return lines
    if before != after:
        old, new = (
            json.dumps(before, default=str)[:80],
            json.dumps(after, default=str)[:80],
        )
        return [f"~ changed {path}: {old} -> {new}"]
    return []


def _lost(before: Any, after: Any, path: str = "") -> list[str]:
    """Non-empty fields present before that are gone after.

    Args:
        before: The old value.
        after: The new value.
        path: Dotted path of this value, for messages.

    Returns:
        Paths of lost fields.
    """
    if not (isinstance(before, dict) and isinstance(after, dict)):
        return []
    lost = []
    for key, value in before.items():
        sub = f"{path}.{key}" if path else str(key)
        if key not in after:
            if value not in (None, {}, [], 0, ""):
                lost.append(sub)
        else:
            lost.extend(_lost(value, after[key], sub))
    return lost


@dataclass(slots=True)
class Report:
    """The outcome of diffing two captures.

    Attributes:
        markdown: The full human-readable report.
        flags: Red flags: things a user would notice or lose.
    """

    markdown: str
    flags: list[str]


def _diff_registry(
    old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]], flags: list[str]
) -> list[str]:
    """Report entity registry differences, keyed by unique id.

    A unique id that vanished while its entity id reappears under another
    one is reported as a re-key rather than as an unrelated add and remove.

    Args:
        old: Baseline registry.
        new: Post-upgrade registry.
        flags: Red flags to append to.

    Returns:
        Report lines.
    """
    lines = []
    by_entity_id = {value["entity_id"]: uid for uid, value in new.items()}
    old_entity_ids = {value["entity_id"] for value in old.values()}
    for uid in sorted(set(old) | set(new)):
        if uid not in new:
            entity_id = old[uid]["entity_id"]
            moved = by_entity_id.get(entity_id)
            note = f" (entity_id now under unique_id {moved})" if moved else ""
            lines.append(f"- REMOVED {entity_id} [{uid}]{note}")
            flags.append(f"entity {entity_id} lost its registry entry{note}")
        elif uid not in old:
            entry = new[uid]
            if entry["entity_id"] not in old_entity_ids:
                lines.append(
                    f"- new {entry['entity_id']} (disabled_by={entry['disabled_by']}, "
                    f"category={entry['entity_category']})"
                )
        else:
            for change in _changes(old[uid], new[uid], old[uid]["entity_id"]):
                lines.append(f"- {change}")
                if any(
                    f".{key}" in change
                    for key in ("entity_id", "name:", "disabled_by", "hidden_by")
                ):
                    flags.append(f"registry {change}")
    return lines


def _diff_stores(
    old: dict[str, Any], new: dict[str, Any], flags: list[str]
) -> list[str]:
    """Report persisted store differences: versions and lost fields.

    Args:
        old: Baseline stores by file name.
        new: Post-upgrade stores by file name.
        flags: Red flags to append to.

    Returns:
        Report lines.
    """
    lines = []
    for name in sorted(set(old) | set(new)):
        old_store, new_store = old.get(name), new.get(name)
        if new_store is None:
            lines.append(f"- {name}: REMOVED")
            flags.append(f"store {name} removed")
        elif old_store is None:
            lines.append(f"- {name}: new, v{new_store.get('version')}")
        else:
            lines.append(
                f"- {name}: v{old_store.get('version')} -> v{new_store.get('version')}"
            )
            for path in _lost(old_store.get("data"), new_store.get("data")):
                lines.append(f"    - lost {path}")
                flags.append(f"store {name} lost {path}")
    return lines


def diff(before: dict[str, Any], after: dict[str, Any]) -> Report:
    """Compare a baseline capture with a post-upgrade one.

    Additions are reported but never flagged: new entities, new store
    fields and new config keys are what an upgrade is for. Losses and
    changes to anything a user set are flagged.

    Args:
        before: The baseline capture.
        after: The post-upgrade capture.

    Returns:
        The report.
    """
    flags: list[str] = []
    body: list[str] = []

    def section(title: str, lines: list[str]) -> None:
        body.extend([f"## {title}", "", *(lines or ["(no change)"]), ""])

    old, new = before["entry"], after["entry"]
    lines = [
        f"- version {old['version']}.{old['minor_version']} -> "
        f"{new['version']}.{new['minor_version']}; {new['subentries']} subentries"
    ]
    for area_id in sorted(set(old["areas"]) | set(new["areas"])):
        if area_id not in new["areas"]:
            flags.append(f"area {area_id} missing after upgrade")
            continue
        for change in _changes(
            old["areas"].get(area_id, {}), new["areas"][area_id], area_id
        ):
            lines.append(f"    {change}")
            if not change.startswith("+"):
                flags.append(f"area config {change}")
    for change in _changes(
        {"data": old["data"], "options": old["options"]},
        {"data": new["data"], "options": new["options"]},
    ):
        lines.append(f"    {change}")
        if change.startswith("- ") or (change.startswith("~ ") and "people" in change):
            flags.append(f"entry {change}")
    section("Config entry", lines)

    section(
        "Entity registry (by unique id)",
        _diff_registry(before["registry"], after["registry"], flags),
    )

    lines = []
    for entity_id, state in sorted(after["states"].items()):
        was = before["states"].get(entity_id)
        if state["state"] in ("unavailable", "unknown") and (
            was is None or was["state"] not in ("unavailable", "unknown")
        ):
            lines.append(f"- {entity_id}: {was and was['state']} -> {state['state']}")
            if was is not None:
                flags.append(f"{entity_id} became {state['state']}")
        if was:
            gained = sorted(set(state["attributes"]) - set(was["attributes"]))
            dropped = sorted(set(was["attributes"]) - set(state["attributes"]))
            if gained or dropped:
                lines.append(f"- {entity_id} attributes +{gained} -{dropped}")
    section("Entity states", lines)

    gone = sorted(set(before["services"]) - set(after["services"]))
    fresh = sorted(set(after["services"]) - set(before["services"]))
    flags.extend(f"service {name} removed" for name in gone)
    section(
        "Services", [f"- removed {n}" for n in gone] + [f"- added {n}" for n in fresh]
    )

    section("Persisted stores", _diff_stores(before["stores"], after["stores"], flags))

    old_db, new_db = before["db"], after["db"]
    lines = [f"- metadata {old_db.get('metadata')} -> {new_db.get('metadata')}"]
    for table in sorted(set(old_db.get("tables", {})) | set(new_db.get("tables", {}))):
        lines.append(
            f"- {table}: {old_db.get('tables', {}).get(table, '-')} -> "
            f"{new_db.get('tables', {}).get(table, '-')}"
        )
    for area, prior in sorted(new_db.get("global_priors", {}).items()):
        lines.append(
            f"- global prior {area}: {old_db.get('global_priors', {}).get(area)} -> {prior}"
        )
    section("Integration database", lines)

    if not after["diagnostics_ok"]:
        flags.append("diagnostics download failed")
    section(
        "Diagnostics", [f"- download {'ok' if after['diagnostics_ok'] else 'FAILED'}"]
    )

    problems = after["log_problems"]
    if any(" ERROR " in line or " CRITICAL " in line for line in problems):
        flags.append("errors in the log after the upgrade")
    section("Log problems after the upgrade", [f"    {line}" for line in problems])

    code_before, code_after = before["code"] or {}, after["code"] or {}
    head = [
        f"# Upgrade {code_before.get('ref')} ({code_before.get('version')}) -> "
        f"{code_after.get('ref')} ({code_after.get('version')}, {code_after.get('sha')})",
        "",
        f"**{len(flags)} red flag(s)**",
        *(f"- {flag}" for flag in flags),
        "",
    ]
    return Report(markdown="\n".join(head + body), flags=flags)


# -- snapshots -----------------------------------------------------------


def snapshot_path(instance: Instance, name: str) -> Path:
    """Where a named snapshot of an instance lives.

    Args:
        instance: The instance.
        name: Snapshot name.

    Returns:
        A sibling directory, ``<instance>@<name>``.
    """
    return instance.path.with_name(f"{instance.path.name}@{name}")


def snapshot(instance: Instance, name: str) -> Path:
    """Copy the stopped instance directory to a named snapshot.

    Args:
        instance: The instance; stopped first if running.
        name: Snapshot name.

    Returns:
        The snapshot directory.
    """
    instance.stop()
    target = snapshot_path(instance, name)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(instance.path, target, symlinks=True)
    return target


def restore(path: Path, name: str) -> Instance:
    """Replace an instance directory with one of its snapshots.

    Args:
        path: The instance directory.
        name: Snapshot name.

    Returns:
        The restored (stopped) instance.

    Raises:
        UpgradeError: If there is no such snapshot.
    """
    source = path.with_name(f"{path.name}@{name}")
    if not (source / MARKER).is_file():
        raise UpgradeError(f"no snapshot {source}")
    if path.exists():
        Instance.load(path).stop()
        shutil.rmtree(path)
    shutil.copytree(source, path, symlinks=True)
    instance = Instance.load(path)
    instance.meta["pid"] = None
    instance.save()
    return instance


# -- the two workflows ---------------------------------------------------


def build_base(
    path: Path,
    *,
    ref: str,
    profile_name: str,
    days: int,
    seed: int,
    time_zone: str,
    port: int,
    laps: int,
    force: bool,
    log: Callable[[str], None],
) -> Instance:
    """Build a lived-in instance on a released version and snapshot it.

    Args:
        path: Instance directory to create.
        ref: Git ref whose integration code to run (usually a release tag).
        profile_name: Harness profile to build.
        days: Days of history to backfill into the recorder.
        seed: Random seed for the history.
        time_zone: Instance time zone.
        port: Port to listen on, 0 for a free one.
        laps: Live walks through the house before the baseline capture.
        force: Replace an existing instance and its snapshots.
        log: Progress callback.

    Returns:
        The running instance, at its captured baseline.

    Raises:
        UpgradeError: If the path exists and ``force`` is not set.
    """
    if path.exists():
        if not force:
            raise UpgradeError(f"{path} already exists; pass --force to replace it")
        Instance.load(path).destroy()
    profile = get_profile(profile_name)
    port = port or free_port()

    path.mkdir(parents=True)
    (path / "configuration.yaml").write_text(
        mock_config.render(profile, time_zone=time_zone, frontend=True, port=port),
        encoding="utf-8",
    )
    storage.write_http_config(path, port)
    storage.write_area_registry(path, profile)
    instance = Instance(
        path=path,
        meta={
            "profile": profile.name,
            "entry_id": None,
            "entry_version": None,
            "days": days,
            "seed": seed,
            "time_zone": time_zone,
            "port": port,
            "frontend": True,
            "rows": {},
            "token": None,
            "refresh_token": None,
            "pid": None,
            "kind": "upgrade",
        },
    )
    instance.save()
    _write_floors(instance, profile)
    log(f"installed {install_code(instance, ref)}")

    log("first boot (onboarding, recorder schema)")
    instance.start()
    instance.wait_for_running()
    instance.stop()

    histories = history.generate(profile, days=days, time_zone=time_zone, seed=seed)
    rows = write_recorder_history(instance, histories)
    instance.meta["rows"] = {"recorder_states": rows}
    instance.save()
    log(f"backfilled {rows} recorder states over {days} days")

    client = instance.start()
    instance.wait_for_running()
    entry_id, notes = run_config_flow(client, profile)
    instance.meta["entry_id"] = entry_id
    instance.save()
    entry = instance.wait_for_entry()
    version = _stored_entry_version(instance)
    expected = conf_version(ref)
    if version != expected:
        raise UpgradeError(
            f"the config flow created a v{version} entry but {ref} writes "
            f"v{expected}: Home Assistant is not running the installed code"
        )
    instance.meta["entry_version"] = version
    instance.save()
    log(f"config flow created entry v{version} ({entry.get('state')})")
    for note in notes:
        log(f"  {note}")

    add_person(client, entry_id, profile)
    instance.wait_for_entry()
    log("added a person with sleep detection")
    for change in customise(instance, client):
        log(change)
    # Enabling entities schedules an entry reload about 30s later.
    time.sleep(40)
    instance.wait_for_entry()

    log(f"walking the house live ({laps} laps)")
    walk(client, profile, laps=laps)
    run_analysis(instance)
    base = capture(instance, BASELINE)
    log(
        f"baseline captured: {len(base['registry'])} entities, "
        f"{len(base['log_problems'])} log problem lines"
    )
    target = snapshot(instance, BASELINE)
    log(f"snapshot {target}")
    instance.start()
    instance.wait_for_running()
    return instance


def run_upgrade(
    path: Path,
    *,
    ref: str,
    settle: float,
    restore_first: bool,
    log: Callable[[str], None],
) -> tuple[Instance, Report]:
    """Upgrade a base instance to another ref and diff against its baseline.

    Args:
        path: The instance directory built by ``build_base``.
        ref: Git ref to upgrade to, or ``working`` for the working tree.
        settle: Seconds to let the upgraded instance run before capturing.
        restore_first: Restore the baseline snapshot before upgrading, so
            every run starts from the same pre-upgrade state.
        log: Progress callback.

    Returns:
        The running upgraded instance, and the diff report.
    """
    if restore_first:
        instance = restore(path, BASELINE)
        log(f"restored {BASELINE} ({(instance.meta.get('code') or {}).get('ref')})")
    else:
        instance = Instance.load(path)
        instance.stop()
    baseline = json.loads(
        (instance.path / "captures" / f"{BASELINE}.json").read_text(encoding="utf-8")
    )
    log(f"installed {install_code(instance, ref)}")
    # Keep the previous run's log out of this run's problems.
    for name in ("home-assistant.log", "home-assistant.log.1"):
        (instance.path / name).unlink(missing_ok=True)

    instance.start()
    entry = instance.wait_for_entry()
    instance.wait_for_running()
    log(f"entry {entry.get('state')} after upgrade; settling {settle:.0f}s")
    time.sleep(settle)
    run_analysis(instance)
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", ref)
    after = capture(instance, label)
    report = diff(baseline, after)
    (instance.path / "captures" / f"{BASELINE}--{label}.md").write_text(
        report.markdown, encoding="utf-8"
    )
    return instance, report
