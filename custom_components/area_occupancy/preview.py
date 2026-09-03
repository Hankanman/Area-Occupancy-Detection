"""Live form preview for the options flow.

Home Assistant's generic flow preview lets a form stream a ``state`` plus
``attributes`` next to the fields while the user edits them. The frontend
subscribes to ``<domain>/start_preview`` with the flow id and the current
(unsaved) form values, and re-subscribes on every change.

The estimate shown here is the **sensor-only base probability** the area
would read *right now* with the candidate weights, likelihoods and
threshold applied to the live sensor evidence and learned priors. It
deliberately excludes the activity and adjacency boosts, wasp-in-box and
decay timing, which depend on history rather than on the values being
edited; the current live probability is included as an attribute so the two
can be compared.

The flow publishes a small context (entry id, area id, draft config) under
``hass.data`` keyed by flow id so the websocket handler does not need to
reach into the flow manager's private state; the flow removes it when it
ends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import voluptuous as vol

from homeassistant.components import websocket_api
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.event import async_track_state_change_event

from .config_helpers import flatten_sectioned_input
from .const import (
    CONF_MOTION_PROB_GIVEN_FALSE,
    CONF_MOTION_PROB_GIVEN_TRUE,
    CONF_THRESHOLD,
    CONF_WEIGHT_APPLIANCE,
    CONF_WEIGHT_COVER,
    CONF_WEIGHT_DOOR,
    CONF_WEIGHT_ENVIRONMENTAL,
    CONF_WEIGHT_LOCK,
    CONF_WEIGHT_MEDIA,
    CONF_WEIGHT_MOTION,
    CONF_WEIGHT_POWER,
    CONF_WEIGHT_WIFI_CLIENTS,
    CONF_WEIGHT_WINDOW,
    DOMAIN,
)
from .data.entity_type import InputType
from .utils import combined_probability, environmental_confidence, presence_probability

if TYPE_CHECKING:
    from collections.abc import Callable

    from .area.area import Area
    from .coordinator import AreaOccupancyCoordinator
    from .data.entity import Entity

_LOGGER = logging.getLogger(__name__)

# ``preview=`` value passed to ``async_show_form``. Any name the frontend has
# no dedicated component for is rendered by ``flow-preview-generic``, which
# subscribes to ``<name>/start_preview``.
PREVIEW_COMPONENT = DOMAIN
PREVIEW_DATA_KEY = f"{DOMAIN}_flow_previews"

_ENVIRONMENTAL_TYPES = {
    InputType.TEMPERATURE,
    InputType.HUMIDITY,
    InputType.ILLUMINANCE,
    InputType.CO2,
    InputType.CO,
    InputType.SOUND_PRESSURE,
    InputType.PRESSURE,
    InputType.AIR_QUALITY,
    InputType.VOC,
    InputType.PM25,
    InputType.PM10,
    InputType.ENVIRONMENTAL,
}

# Which candidate-config key carries the weight for each input type.
_WEIGHT_KEY_FOR_INPUT_TYPE: dict[InputType, str] = {
    InputType.MOTION: CONF_WEIGHT_MOTION,
    InputType.MEDIA: CONF_WEIGHT_MEDIA,
    InputType.APPLIANCE: CONF_WEIGHT_APPLIANCE,
    InputType.DOOR: CONF_WEIGHT_DOOR,
    InputType.LOCK: CONF_WEIGHT_LOCK,
    InputType.WINDOW: CONF_WEIGHT_WINDOW,
    InputType.COVER: CONF_WEIGHT_COVER,
    InputType.POWER: CONF_WEIGHT_POWER,
    InputType.WIFI_CLIENTS: CONF_WEIGHT_WIFI_CLIENTS,
    **dict.fromkeys(_ENVIRONMENTAL_TYPES, CONF_WEIGHT_ENVIRONMENTAL),
}


class PreviewEntity:
    """A live entity seen through candidate configuration.

    Delegates everything (evidence, decay state, type, correlation data) to
    the real ``Entity`` and overrides only what the form can change: the
    weight and, for motion sensors, the likelihoods.
    """

    def __init__(
        self,
        live: Entity,
        weight: float,
        prob_given_true: float,
        prob_given_false: float,
    ) -> None:
        """Wrap ``live`` with the candidate weight and likelihoods."""
        self._live = live
        self._weight = weight
        self._prob_given_true = prob_given_true
        self._prob_given_false = prob_given_false

    def __getattr__(self, name: str) -> Any:
        """Delegate anything not overridden to the live entity."""
        return getattr(self._live, name)

    @property
    def weight(self) -> float:
        """Candidate weight for this entity's type."""
        return self._weight

    @property
    def prob_given_true(self) -> float:
        """Candidate P(active | occupied)."""
        return self._prob_given_true

    @property
    def prob_given_false(self) -> float:
        """Candidate P(active | not occupied)."""
        return self._prob_given_false

    @property
    def information_gain(self) -> float:
        """Information gain under the candidate likelihoods.

        Unchanged likelihoods keep the live value (which also carries the
        analysis-error clamp); overridden ones use the same formula on the
        candidate pair. Only motion likelihoods are user-editable and motion
        is excluded from correlation analysis, so the clamp cannot apply.
        """
        live = self._live
        if (
            self._prob_given_true == live.prob_given_true
            and self._prob_given_false == live.prob_given_false
        ):
            return live.information_gain
        pgt, pgf = self._prob_given_true, self._prob_given_false
        return min(1.0, abs(pgt - pgf) / max(pgt, pgf, 0.01))

    @property
    def effective_weight(self) -> float:
        """Weight scaled by information gain, as the sigmoid model expects."""
        return self.weight * self.information_gain


def build_preview_entities(
    area: Area, candidate: dict[str, Any]
) -> dict[str, PreviewEntity]:
    """Wrap the area's live entities with the candidate configuration."""
    wrapped: dict[str, PreviewEntity] = {}
    for entity_id, entity in area.entities.entities.items():
        input_type = entity.type.input_type
        weight = float(entity.weight)
        weight_key = _WEIGHT_KEY_FOR_INPUT_TYPE.get(input_type)
        if weight_key is not None and weight_key in candidate:
            weight = float(candidate[weight_key])

        pgt, pgf = float(entity.prob_given_true), float(entity.prob_given_false)
        if input_type == InputType.MOTION:
            pgt = float(candidate.get(CONF_MOTION_PROB_GIVEN_TRUE, pgt))
            pgf = float(candidate.get(CONF_MOTION_PROB_GIVEN_FALSE, pgf))

        wrapped[entity_id] = PreviewEntity(entity, weight, pgt, pgf)
    return wrapped


def compute_area_preview(
    area: Area, candidate: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Return ``(state, attributes)`` for the generic preview row.

    ``state`` is the sensor-only occupancy probability as a percentage
    string; the attributes carry the verdict against the candidate
    threshold, the live probability for comparison, and which sensors are
    currently contributing.
    """
    entities = build_preview_entities(area, candidate)
    prior = float(area.prior.value)
    correlations = area.coordinator.get_cached_correlations(area.area_name)

    presence = presence_probability(entities, prior=prior, correlations=correlations)
    env = environmental_confidence(entities, correlations=correlations)
    base = presence if env == 0.5 else combined_probability(presence, env)

    threshold_pct = float(candidate.get(CONF_THRESHOLD, area.config.threshold * 100))
    occupied = base >= threshold_pct / 100.0

    active = sorted(eid for eid, e in entities.items() if e.evidence is True)
    decaying = sorted(
        eid
        for eid, e in entities.items()
        if e.evidence is not True and e.decay.is_decaying
    )

    verdict = "occupied" if occupied else "not occupied"
    attributes: dict[str, Any] = {
        # The generic preview row shows only name and state, so the verdict
        # against the candidate threshold rides along in the name.
        "friendly_name": f"{area.area_name} · {verdict} (threshold {threshold_pct:g}%)",
        "unit_of_measurement": "%",
        "occupied": occupied,
        "threshold": threshold_pct,
        "current_probability": round(area.probability() * 100, 1),
        "prior": round(prior * 100, 1),
        "presence_probability": round(presence * 100, 1),
        "environmental_confidence": round(env * 100, 1),
        "active_sensors": active,
        "decaying_sensors": decaying,
        "note": (
            "Sensor-only estimate from current sensor states and learned priors "
            "with the values in this form applied. Activity and adjacency "
            "boosts, Wasp in Box and decay timing are not simulated."
        ),
    }
    return f"{base * 100:.1f}", attributes


# ── Flow-side context ────────────────────────────────────────────────


@callback
def register_preview_context(
    hass: HomeAssistant,
    flow_id: str,
    entry_id: str,
    area_id: str | None,
    draft: dict[str, Any],
) -> None:
    """Publish what the websocket handler needs to preview this flow."""
    hass.data.setdefault(PREVIEW_DATA_KEY, {})[flow_id] = {
        "entry_id": entry_id,
        "area_id": area_id,
        "draft": dict(draft),
    }


@callback
def unregister_preview_context(hass: HomeAssistant, flow_id: str) -> None:
    """Drop the preview context once the flow has finished."""
    hass.data.get(PREVIEW_DATA_KEY, {}).pop(flow_id, None)


def _find_area(
    coordinator: AreaOccupancyCoordinator | None, area_id: str | None
) -> Area | None:
    if coordinator is None or not area_id:
        return None
    for area in coordinator.areas.values():
        if area.config.area_id == area_id:
            return area
    return None


# ── Websocket API ────────────────────────────────────────────────────


async def async_setup_preview(hass: HomeAssistant) -> None:
    """Register the preview websocket command (called once by the flow manager)."""
    websocket_api.async_register_command(hass, ws_start_preview)


@websocket_api.websocket_command(
    {
        vol.Required("type"): f"{DOMAIN}/start_preview",
        vol.Required("flow_id"): str,
        vol.Required("flow_type"): vol.Any("config_flow", "options_flow"),
        vol.Required("user_input"): dict,
    }
)
@callback
def ws_start_preview(
    hass: HomeAssistant,
    connection: websocket_api.ActiveConnection,
    msg: dict[str, Any],
) -> None:
    """Stream a preview for the area being edited in an options flow."""
    context = hass.data.get(PREVIEW_DATA_KEY, {}).get(msg["flow_id"])
    if context is None:
        connection.send_error(
            msg["id"], "not_found", "No preview is available for this flow"
        )
        return

    entry = hass.config_entries.async_get_entry(context["entry_id"])
    coordinator = getattr(entry, "runtime_data", None) if entry is not None else None
    area = _find_area(coordinator, context["area_id"])

    try:
        candidate = {**context["draft"], **flatten_sectioned_input(msg["user_input"])}
    except (TypeError, ValueError, AttributeError) as err:
        connection.send_error(
            msg["id"], "invalid_input", f"Cannot read form values: {err}"
        )
        return

    @callback
    def _send(*_: Any) -> None:
        if area is None:
            state, attributes = (
                "unavailable",
                {
                    "friendly_name": "Preview",
                    "reason": "The preview is available once the area has been "
                    "saved and loaded.",
                },
            )
        else:
            try:
                state, attributes = compute_area_preview(area, candidate)
            except Exception as err:  # noqa: BLE001 - a preview must never break the form
                _LOGGER.debug("Preview computation failed: %s", err)
                state, attributes = "unavailable", {"error": str(err)}
        connection.send_message(
            websocket_api.event_message(
                msg["id"], {"state": state, "attributes": attributes}
            )
        )

    connection.send_result(msg["id"])
    _send()

    unsubscribers: list[Callable[[], None]] = []
    if area is not None and coordinator is not None:
        entity_ids = list(area.entities.entities)
        if entity_ids:
            unsubscribers.append(
                async_track_state_change_event(hass, entity_ids, _send)
            )
        # Decay ticks and analysis runs refresh the coordinator without any
        # entity state event; follow those too so the number keeps moving.
        unsubscribers.append(coordinator.async_add_listener(_send))

    @callback
    def _unsubscribe() -> None:
        for unsub in unsubscribers:
            unsub()

    connection.subscriptions[msg["id"]] = _unsubscribe
