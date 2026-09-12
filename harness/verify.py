"""End-to-end checks a throwaway instance can be held to.

These are the checks a unit test cannot make. They run against a real Home
Assistant with a real frontend API in front of it, so they see what a user
sees: whether the config entry actually loaded, whether the migration moved
the areas and re-homed their entities, whether every step of every flow has
a handler and a schema core will accept, and whether the numbers the
integration derives from seeded history are sane rather than pinned.

Each check returns a ``Result``; nothing raises for a failure, so one run
reports every problem rather than only the first.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import sqlalchemy as sa

from custom_components.area_occupancy.const import (
    CONF_AREA_ID,
    CONF_AREAS,
    CONF_VERSION,
    DB_NAME,
    DOMAIN,
    SUBENTRY_TYPE_AREA,
)
from custom_components.area_occupancy.db.schema import GlobalPriors

from . import mock_config, storage
from .client import ApiError, Client
from .instance import Instance
from .profiles import CHANNELS, area_entities

#: Entities every configured area must expose, by the name they carry after
#: the device name. Matched on friendly name rather than entity id on
#: purpose: Home Assistant 2026.9 builds ids from area, device and entity
#: name, so the ids depend on both the HA area and platform load order while
#: the names stay stable. Only entities enabled by default are listed --
#: the diagnostic ones are registry-disabled and have no state.
AREA_ENTITY_NAMES = (
    "Occupancy Status",
    "Occupancy Probability",
    "Occupancy Threshold",
)

#: How deep to walk the flow menus. Three covers hub -> area -> spoke.
FLOW_DEPTH = 3


@dataclass(slots=True)
class Result:
    """The outcome of one check.

    Attributes:
        name: Check name.
        passed: Whether it passed.
        detail: One-line explanation, shown either way.
        notes: Extra lines shown for a failure or in verbose mode.
    """

    name: str
    passed: bool
    detail: str
    notes: list[str] = field(default_factory=list)


def _ok(name: str, detail: str, notes: list[str] | None = None) -> Result:
    """Build a passing result.

    Args:
        name: Check name.
        detail: Explanation.
        notes: Extra lines.

    Returns:
        The result.
    """
    return Result(name=name, passed=True, detail=detail, notes=notes or [])


def _fail(name: str, detail: str, notes: list[str] | None = None) -> Result:
    """Build a failing result.

    Args:
        name: Check name.
        detail: Explanation.
        notes: Extra lines.

    Returns:
        The result.
    """
    return Result(name=name, passed=False, detail=detail, notes=notes or [])


def check_entry_loaded(instance: Instance, client: Client) -> Result:
    """The seeded config entry set up without error.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    entries = [
        entry
        for entry in client.config_entries(DOMAIN)
        if entry["entry_id"] == instance.entry_id
    ]
    if not entries:
        return _fail("entry_loaded", "the seeded config entry is not present")
    entry = entries[0]
    if entry.get("state") != "loaded":
        return _fail(
            "entry_loaded",
            f"config entry state is {entry.get('state')!r}, not 'loaded'",
            [entry.get("reason") or "", instance.log_tail(15)],
        )
    return _ok("entry_loaded", f"entry {entry['title']!r} loaded")


def check_migration(instance: Instance, client: Client) -> Result:
    """A legacy-seeded entry migrated to subentries and dropped the old key.

    Read from the config entries store rather than the API, which does not
    expose an entry's version. That is only safe once the instance has
    finished starting, which ``Instance.wait_for_running`` guarantees before
    any check runs.

    Skipped with a pass when the instance was seeded at the current version,
    since there is nothing to migrate.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    seeded = int(instance.meta["entry_version"])
    if seeded == CONF_VERSION:
        return _ok("migration", f"seeded at v{CONF_VERSION}, nothing to migrate")

    entry = storage.read_config_entry(instance.path)
    if entry is None:
        return _fail("migration", "no config entry on disk")

    notes: list[str] = []
    problems: list[str] = []
    if entry["version"] != CONF_VERSION:
        problems.append(f"entry is still at v{entry['version']}")
    if CONF_AREAS in entry["data"] or CONF_AREAS in entry["options"]:
        problems.append(f"the legacy {CONF_AREAS!r} key survived the migration")

    subentries = [
        sub for sub in entry["subentries"] if sub["subentry_type"] == SUBENTRY_TYPE_AREA
    ]
    expected = {area.slug for area in instance.profile.areas}
    found = {sub["unique_id"] for sub in subentries}
    if found != expected:
        problems.append(
            f"subentries {sorted(found)} do not match areas {sorted(expected)}"
        )
    notes.append(f"{len(subentries)} area subentries: {', '.join(sorted(found))}")

    if problems:
        return _fail("migration", "; ".join(problems), notes)
    return _ok(
        "migration", f"v{seeded} -> v{CONF_VERSION} with one subentry per area", notes
    )


def _by_friendly_name(client: Client) -> dict[str, dict[str, Any]]:
    """Index every state in the instance by its friendly name.

    Args:
        client: An authenticated client.

    Returns:
        Mapping of friendly name to state object.
    """
    return {
        state["attributes"]["friendly_name"]: state
        for state in client.states()
        if state.get("attributes", {}).get("friendly_name")
    }


def check_entities(instance: Instance, client: Client) -> Result:
    """Every area exposes its entities, and none are unavailable.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    states = _by_friendly_name(client)
    missing: list[str] = []
    unavailable: list[str] = []
    checked = 0

    for area in instance.profile.areas:
        for label in AREA_ENTITY_NAMES:
            name = f"{area.name} {label}"
            checked += 1
            state = states.get(name)
            if state is None:
                missing.append(name)
            elif state["state"] in ("unavailable", "unknown"):
                unavailable.append(f"{name} ({state['entity_id']}) = {state['state']}")

    if missing or unavailable:
        return _fail(
            "entities",
            f"{len(missing)} missing, {len(unavailable)} unavailable of {checked}",
            [*(f"missing: {name}" for name in missing), *unavailable],
        )
    return _ok("entities", f"all {checked} area entities present and available")


def check_mock_sensors(instance: Instance, client: Client) -> Result:
    """Every configured mock sensor exists, so nothing is silently ignored.

    A config entry pointing at entity ids that do not exist is the quietest
    way for a seeded instance to be useless, so it is worth its own check.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    missing: list[str] = []
    total = 0
    for area in instance.profile.areas:
        for entities in area_entities(area).values():
            for entity_id in entities:
                total += 1
                if client.state(entity_id) is None:
                    missing.append(entity_id)

    if missing:
        return _fail(
            "mock_sensors",
            f"{len(missing)} of {total} configured sensors do not exist",
            missing[:12],
        )
    return _ok("mock_sensors", f"all {total} configured mock sensors exist")


def check_sensor_response(instance: Instance, client: Client) -> Result:
    """Turning a motion sensor on raises the area's probability.

    This is the one check that exercises the live calculation path rather
    than configuration: evidence in, probability up.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    area = next(
        (spec for spec in instance.profile.areas if spec.channels.get("motion")), None
    )
    if area is None:
        return _ok("sensor_response", "no motion channel in this profile, skipped")

    probability = _by_friendly_name(client).get(f"{area.name} Occupancy Probability")
    if probability is None:
        return _fail(
            "sensor_response", f"{area.name} has no occupancy probability sensor"
        )
    probability_id = probability["entity_id"]
    knob = f"input_boolean.{mock_config.knob_id(area, 'motion', 1)}"

    client.set_state(knob, "off")
    idle = _settle(client, probability_id)

    client.set_state(knob, "on")
    active = _wait_for_change(client, probability_id, idle)

    client.set_state(knob, "off")

    if idle is None or active is None:
        return _fail("sensor_response", f"{probability_id} is not numeric")
    if active <= idle:
        return _fail(
            "sensor_response",
            f"{area.name} probability did not rise with motion: {idle} -> {active}",
        )
    return _ok(
        "sensor_response", f"{area.name} probability {idle}% -> {active}% on motion"
    )


def _settle(client: Client, entity_id: str, *, seconds: float = 3.0) -> float | None:
    """Wait for an entity to stop moving, then read it.

    Args:
        client: An authenticated client.
        entity_id: Entity to read.
        seconds: How long to let the coordinator catch up.

    Returns:
        The entity's numeric state, or ``None`` if it is not a number.
    """
    time.sleep(seconds)
    return _numeric_state(client, entity_id)


def _wait_for_change(
    client: Client,
    entity_id: str,
    previous: float | None,
    *,
    timeout: float = 20.0,
) -> float | None:
    """Poll an entity until its value differs from ``previous``.

    Args:
        client: An authenticated client.
        entity_id: Entity to poll.
        previous: The value to wait to move away from.
        timeout: Seconds to keep polling.

    Returns:
        The new value, or the last value read if it never changed.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _numeric_state(client, entity_id)
        if value is not None and value != previous:
            return value
        time.sleep(0.5)
    return _numeric_state(client, entity_id)


def _numeric_state(client: Client, entity_id: str) -> float | None:
    """Read an entity's state as a float.

    Args:
        client: An authenticated client.
        entity_id: Entity to read.

    Returns:
        The numeric state, or ``None`` if it is not a number.
    """
    state = client.state(entity_id)
    if state is None:
        return None
    try:
        return float(state["state"])
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class FlowNode:
    """One reachable step in a flow, found by walking the menus.

    Attributes:
        path: The menu choices taken to reach it.
        step_id: The step's id.
        step_type: ``form``, ``menu``, ``create_entry`` or ``abort``.
        error: The API error, if the step could not be reached at all.
    """

    path: tuple[str, ...]
    step_id: str
    step_type: str
    error: str | None = None


def _walk_flow(
    client: Client,
    *,
    start: Any,
    advance: Any,
    abort: Any,
    depth: int = FLOW_DEPTH,
) -> list[FlowNode]:
    """Walk every menu path of a flow, one fresh flow per path.

    Only menu navigation is submitted, never form data, so the walk cannot
    change the instance's configuration. Each path gets its own flow because
    a flow is a state machine -- stepping into one spoke rules out its
    siblings.

    Args:
        client: An authenticated client.
        start: Callable returning a fresh first step.
        advance: Callable taking ``(flow_id, user_input)``.
        abort: Callable taking ``flow_id``.
        depth: Maximum menu depth to explore.

    Returns:
        Every node reached, including ones that errored.
    """
    nodes: list[FlowNode] = []
    queue: deque[tuple[str, ...]] = deque([()])
    seen: set[tuple[str, ...]] = set()

    while queue:
        path = queue.popleft()
        if path in seen:
            continue
        seen.add(path)

        flow_id: str | None = None
        try:
            step = start()
            flow_id = step.get("flow_id")
            for choice in path:
                step = advance(flow_id, {"next_step_id": choice})
                flow_id = step.get("flow_id", flow_id)
        except ApiError as err:
            nodes.append(
                FlowNode(
                    path=path,
                    step_id=path[-1] if path else "(start)",
                    step_type="error",
                    error=str(err),
                )
            )
            continue

        step_type = step.get("type", "?")
        node = FlowNode(
            path=path, step_id=step.get("step_id") or step_type, step_type=step_type
        )
        nodes.append(node)

        if step_type == "menu" and len(path) < depth:
            options = step.get("menu_options") or []
            if isinstance(options, dict):
                options = list(options)
            for option in options:
                queue.append((*path, option))

        if flow_id and step_type in ("form", "menu"):
            abort(flow_id)

    return nodes


def check_options_flow(instance: Instance, client: Client) -> Result:
    """Every reachable options-flow step renders.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    nodes = _walk_flow(
        client,
        start=lambda: client.start_options_flow(instance.entry_id),
        advance=client.advance_options_flow,
        abort=client.abort_options_flow,
    )
    return _flow_result("options_flow", nodes)


def check_subentry_flow(instance: Instance, client: Client) -> Result:
    """Every reachable step of an area's reconfigure flow renders.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    entry = storage.read_config_entry(instance.path)
    subentries = [
        sub
        for sub in (entry or {}).get("subentries", [])
        if sub["subentry_type"] == SUBENTRY_TYPE_AREA
    ]
    if not subentries:
        return _fail("subentry_flow", "no area subentries to reconfigure")

    subentry_id = subentries[0]["subentry_id"]
    nodes = _walk_flow(
        client,
        start=lambda: client.start_subentry_flow(
            instance.entry_id, SUBENTRY_TYPE_AREA, subentry_id=subentry_id
        ),
        advance=client.advance_subentry_flow,
        abort=client.abort_subentry_flow,
    )
    return _flow_result(f"subentry_flow[{subentries[0]['title']}]", nodes)


def _flow_result(name: str, nodes: list[FlowNode]) -> Result:
    """Summarise a flow walk.

    Args:
        name: Check name.
        nodes: Nodes the walk reached.

    Returns:
        The result, failing if any node errored.
    """
    broken = [node for node in nodes if node.error]
    notes = [
        f"{'/'.join(node.path) or '(start)'} -> {node.step_id} [{node.step_type}]"
        for node in nodes
    ]
    if broken:
        return Result(
            name=name,
            passed=False,
            detail=f"{len(broken)} of {len(nodes)} steps failed to render",
            notes=[f"{'/'.join(node.path)}: {node.error}" for node in broken],
        )
    return _ok(name, f"{len(nodes)} steps render", notes)


def check_analysis(instance: Instance, client: Client) -> Result:
    """The analysis pipeline runs and derives a believable prior.

    The recomputed prior is not compared tightly against the generated
    occupancy on purpose: the integration derives occupancy from motion
    intervals, and a motion sensor is only active for part of a visit, so the
    two legitimately differ. What is worth asserting is that every area gets
    a prior in the same ballpark and, above all, that none is pinned at the
    probability bounds -- the failure this project has hit more than once.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    if int(instance.meta["days"]) <= 0:
        return _ok("analysis", "instance has no seeded history, skipped")

    try:
        client.call_service("area_occupancy", "run_analysis", return_response=True)
    except ApiError as err:
        return _fail("analysis", f"run_analysis failed: {err}", [instance.log_tail(20)])

    priors = _read_global_priors(instance.path)
    if not priors:
        return _fail("analysis", "no global priors were written")

    notes: list[str] = []
    problems: list[str] = []
    for area in instance.profile.areas:
        value = priors.get(area.name)
        if value is None:
            problems.append(f"{area.name} has no global prior")
            continue
        notes.append(f"{area.name}: prior {value:.3f}")
        if not 0.02 <= value <= 0.95:
            problems.append(f"{area.name} prior is pinned at {value:.3f}")

    if problems:
        return _fail("analysis", "; ".join(problems), notes)
    return _ok("analysis", f"priors derived for {len(priors)} areas", notes)


def _read_global_priors(config_dir: Path) -> dict[str, float]:
    """Read every area's global prior from the integration database.

    Args:
        config_dir: The instance's configuration directory.

    Returns:
        Mapping of area name to prior value, empty if there is no database.
    """
    db_path = config_dir / ".storage" / DB_NAME
    if not db_path.is_file():
        return {}
    engine = sa.create_engine(f"sqlite:///{db_path}")
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                sa.select(GlobalPriors.area_name, GlobalPriors.prior_value)
            ).all()
    finally:
        engine.dispose()
    return {str(name): float(value) for name, value in rows}


def check_export_config(instance: Instance, client: Client) -> Result:
    """The config export describes the whole instance, areas included.

    Worth its own check because the export is what users paste into bug
    reports and the simulator, and because it reads the areas through a
    different path from everything else -- it once kept reading the legacy
    list after the areas had moved into subentries, and came back empty.

    Args:
        instance: The instance under test.
        client: An authenticated client.

    Returns:
        The result.
    """
    try:
        response = client.call_service(
            "area_occupancy", "export_config", {}, return_response=True
        )
    except ApiError as err:
        return _fail("export_config", f"export_config failed: {err}")

    exported = (response or {}).get("service_response") or {}
    areas = exported.get(CONF_AREAS) or []
    expected = {area.slug for area in instance.profile.areas}
    found = {area.get(CONF_AREA_ID) for area in areas if isinstance(area, dict)}

    problems: list[str] = []
    if found != expected:
        problems.append(
            f"exported areas {sorted(found)} do not match {sorted(expected)}"
        )
    if not any(
        area.get(CHANNELS["motion"].conf_key)
        for area in areas
        if isinstance(area, dict)
    ):
        problems.append("no area exported its motion sensors")

    if problems:
        return _fail("export_config", "; ".join(problems))
    return _ok("export_config", f"{len(areas)} areas exported with their sensors")


def check_subentry_linkage(instance: Instance) -> Result:
    """Each area's entities are registered under that area's subentry.

    Read from disk after shutdown rather than over the API: the entity
    registry is only exposed over the websocket API, and Home Assistant
    defers registry writes while it is still starting, so the file is only
    authoritative once the instance has stopped cleanly.

    Args:
        instance: The stopped instance.

    Returns:
        The result.
    """
    registry = instance.path / ".storage" / "core.entity_registry"
    if not registry.is_file():
        return _fail("subentry_linkage", "no entity registry was written")

    entry = storage.read_config_entry(instance.path)
    subentry_by_area = {
        sub["unique_id"]: sub["subentry_id"]
        for sub in (entry or {}).get("subentries", [])
        if sub["subentry_type"] == SUBENTRY_TYPE_AREA
    }
    if not subentry_by_area:
        return _fail("subentry_linkage", "the entry has no area subentries")

    entities = [
        item
        for item in json.loads(registry.read_text(encoding="utf-8"))["data"]["entities"]
        if item.get("platform") == DOMAIN
    ]

    wrong: list[str] = []
    linked = 0
    aggregates = 0
    for item in entities:
        entity_id = item["entity_id"]
        object_id = entity_id.split(".", 1)[1]
        area = next(
            (slug for slug in subentry_by_area if object_id.startswith(f"{slug}_")),
            None,
        )
        if area is None:
            # "All Areas" and per-floor entities span areas, so they belong
            # to the entry itself and must not carry a subentry.
            aggregates += 1
            if item.get("config_subentry_id"):
                wrong.append(f"{entity_id} is an aggregate but has a subentry")
            continue
        if item.get("config_subentry_id") != subentry_by_area[area]:
            wrong.append(
                f"{entity_id} has subentry {item.get('config_subentry_id')!r},"
                f" expected {subentry_by_area[area]!r}"
            )
        else:
            linked += 1

    if wrong:
        return _fail(
            "subentry_linkage",
            f"{len(wrong)} of {len(entities)} entities misfiled",
            wrong[:12],
        )
    return _ok(
        "subentry_linkage",
        f"{linked} area entities filed under their subentry, {aggregates} aggregates on the entry",
    )


#: Checks that run against a live instance.
LIVE_CHECKS = (
    check_entry_loaded,
    check_migration,
    check_mock_sensors,
    check_entities,
    check_sensor_response,
    check_options_flow,
    check_subentry_flow,
    check_export_config,
    check_analysis,
)

#: Checks that need the instance stopped so its stores are flushed.
STOPPED_CHECKS = (check_subentry_linkage,)


def run_all(instance: Instance, client: Client) -> list[Result]:
    """Run the live checks, stop the instance, then run the on-disk checks.

    Args:
        instance: The instance under test, already started.
        client: An authenticated client.

    Returns:
        Every result, in the order the checks ran.
    """
    results: list[Result] = []
    for check in LIVE_CHECKS:
        try:
            results.append(check(instance, client))
        except Exception as err:  # noqa: BLE001 - a broken check is a failure, not a crash
            results.append(
                _fail(check.__name__.removeprefix("check_"), f"check raised: {err!r}")
            )

    instance.stop()

    for stopped_check in STOPPED_CHECKS:
        try:
            results.append(stopped_check(instance))
        except Exception as err:  # noqa: BLE001
            results.append(
                _fail(
                    stopped_check.__name__.removeprefix("check_"),
                    f"check raised: {err!r}",
                )
            )
    return results


__all__ = ["AREA_ENTITY_NAMES", "LIVE_CHECKS", "FlowNode", "Result", "run_all"]
