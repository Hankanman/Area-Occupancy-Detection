"""Tests for the options-flow live preview (custom_components/area_occupancy/preview.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.config_flow import (
    AreaOccupancyOptionsFlow,
    AreaSubentryFlowHandler,
)
from custom_components.area_occupancy.const import (
    CONF_MOTION_PROB_GIVEN_FALSE,
    CONF_MOTION_PROB_GIVEN_TRUE,
    CONF_THRESHOLD,
    CONF_WEIGHT_CUSTOM_BINARY,
    CONF_WEIGHT_CUSTOM_NUMERIC,
    CONF_WEIGHT_MOTION,
    DOMAIN,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.entity import EntityFactory
from custom_components.area_occupancy.data.entity_type import InputType
from custom_components.area_occupancy.preview import (
    PREVIEW_COMPONENT,
    PREVIEW_DATA_KEY,
    PREVIEW_FLOW_TYPES,
    PreviewEntity,
    build_preview_entities,
    compute_area_preview,
    register_preview_context,
    unregister_preview_context,
    ws_start_preview,
)
from homeassistant.core import HomeAssistant


class TestPreviewEntity:
    """The proxy overrides only what the form can change."""

    def test_overrides_and_delegation(
        self, coordinator: AreaOccupancyCoordinator, default_area: Any
    ) -> None:
        area = default_area
        live = next(
            e
            for e in area.entities.entities.values()
            if e.type.input_type == InputType.MOTION
        )

        proxy = PreviewEntity(
            live, weight=0.25, prob_given_true=0.9, prob_given_false=0.1
        )

        assert proxy.weight == 0.25
        assert proxy.prob_given_true == 0.9
        assert proxy.prob_given_false == 0.1
        # Delegated
        assert proxy.entity_id == live.entity_id
        assert proxy.type is live.type
        assert proxy.evidence == live.evidence
        assert proxy.decay is live.decay
        # Information gain recomputed from the candidate pair
        assert proxy.information_gain == pytest.approx(min(1.0, 0.8 / 0.9))
        assert proxy.effective_weight == pytest.approx(0.25 * proxy.information_gain)

    def test_unchanged_likelihoods_keep_live_information_gain(
        self, coordinator: AreaOccupancyCoordinator, default_area: Any
    ) -> None:
        area = default_area
        live = next(iter(area.entities.entities.values()))
        proxy = PreviewEntity(
            live, live.weight, live.prob_given_true, live.prob_given_false
        )
        assert proxy.information_gain == live.information_gain


class TestComputeAreaPreview:
    """The estimate reacts to candidate values and reports its inputs."""

    def test_candidate_weights_are_applied(
        self, coordinator: AreaOccupancyCoordinator, default_area: Any
    ) -> None:
        area = default_area
        entities = build_preview_entities(area, {CONF_WEIGHT_MOTION: 0.42})
        motion = [e for e in entities.values() if e.type.input_type == InputType.MOTION]
        assert motion
        assert all(e.weight == 0.42 for e in motion)
        # Non-motion entities keep their live weight
        others = [e for e in entities.values() if e.type.input_type != InputType.MOTION]
        for proxy in others:
            assert proxy.weight == area.entities.entities[proxy.entity_id].weight

    def test_zero_weights_return_the_prior(
        self, coordinator: AreaOccupancyCoordinator, default_area: Any
    ) -> None:
        area = default_area
        candidate = dict.fromkeys(
            (
                "weight_motion",
                "weight_media",
                "weight_appliance",
                "weight_door",
                "weight_lock",
                "weight_window",
                "weight_cover",
                "weight_environmental",
                "weight_power",
                "weight_wifi_clients",
            ),
            0.0,
        )
        candidate[CONF_THRESHOLD] = 50

        state, attributes = compute_area_preview(area, candidate)

        # With no sensor allowed to contribute, the sigmoid model returns the prior.
        assert state == f"{area.prior.value * 100:.1f}"
        assert attributes["unit_of_measurement"] == "%"
        assert attributes["threshold"] == 50
        assert attributes["occupied"] == (area.prior.value >= 0.5)
        assert attributes["prior"] == round(area.prior.value * 100, 1)
        assert isinstance(attributes["current_probability"], float)
        assert isinstance(attributes["active_sensors"], list)
        assert "note" in attributes

    def test_threshold_defaults_to_area_config(
        self, coordinator: AreaOccupancyCoordinator, default_area: Any
    ) -> None:
        area = default_area
        _, attributes = compute_area_preview(area, {})
        assert attributes["threshold"] == pytest.approx(area.config.threshold * 100)

    def test_active_sensor_raises_estimate(
        self, coordinator: AreaOccupancyCoordinator, default_area: Any
    ) -> None:
        area = default_area
        motion_id = next(
            eid
            for eid, e in area.entities.entities.items()
            if e.type.input_type == InputType.MOTION
        )
        candidate = {
            CONF_WEIGHT_MOTION: 1.0,
            CONF_MOTION_PROB_GIVEN_TRUE: 0.95,
            CONF_MOTION_PROB_GIVEN_FALSE: 0.01,
        }
        with patch.object(
            type(area.entities.entities[motion_id]), "evidence", new=False
        ):
            idle_state, _ = compute_area_preview(area, candidate)
        with patch.object(
            type(area.entities.entities[motion_id]), "evidence", new=True
        ):
            active_state, attributes = compute_area_preview(area, candidate)

        assert float(active_state) > float(idle_state)
        assert motion_id in attributes["active_sensors"]


class TestPreviewContext:
    """Flow-side registration under hass.data."""

    def test_register_and_unregister(self, hass: HomeAssistant) -> None:
        register_preview_context(hass, "flow-1", "entry-1", "living_room", {"a": 1})
        assert hass.data[PREVIEW_DATA_KEY]["flow-1"] == {
            "entry_id": "entry-1",
            "area_id": "living_room",
            "draft": {"a": 1},
        }
        unregister_preview_context(hass, "flow-1")
        assert "flow-1" not in hass.data[PREVIEW_DATA_KEY]
        # Idempotent
        unregister_preview_context(hass, "flow-1")

    def test_component_name_is_domain(self) -> None:
        # The frontend derives the websocket type from this name.
        assert PREVIEW_COMPONENT == DOMAIN


class TestPreviewIsReachableFromEveryFlow:
    """Every flow that shows a preview must be able to serve one.

    Home Assistant registers the preview websocket command once per flow
    *class* that renders a form with ``preview=``. A class that advertises a
    preview without providing ``async_setup_preview`` leaves the frontend
    subscribing to a command nobody registered, and the preview panel shows
    "Unknown command" -- unless another flow happened to register it earlier
    in the same run, which is why this only appeared when reconfiguring an
    area from the integration page in a fresh session.
    """

    @pytest.mark.parametrize(
        "handler", [AreaOccupancyOptionsFlow, AreaSubentryFlowHandler]
    )
    def test_every_previewing_flow_registers_the_command(self, handler) -> None:
        assert hasattr(handler, "async_setup_preview"), (
            f"{handler.__name__} shows a preview but never registers its command"
        )

    @pytest.mark.parametrize(
        "handler", [AreaOccupancyOptionsFlow, AreaSubentryFlowHandler]
    )
    async def test_setup_preview_registers_the_websocket_command(
        self, hass: HomeAssistant, handler
    ) -> None:
        with patch(
            "custom_components.area_occupancy.preview.websocket_api.async_register_command"
        ) as register:
            await handler.async_setup_preview(hass)

        register.assert_called_once()
        assert register.call_args[0][1] is ws_start_preview

    def test_subentry_flow_type_is_accepted(self) -> None:
        # The frontend reports a subentry flow as "config_subentries_flow"
        # (plural); a schema that only knows the singular rejects the main
        # editing path's preview.
        assert "config_subentries_flow" in PREVIEW_FLOW_TYPES
        assert {"config_flow", "options_flow"} <= set(PREVIEW_FLOW_TYPES)


class _FakeConnection:
    """Minimal stand-in for websocket_api.ActiveConnection."""

    def __init__(self) -> None:
        self.results: list[int] = []
        self.errors: list[tuple[int, str, str]] = []
        self.messages: list[dict[str, Any]] = []
        self.subscriptions: dict[int, Any] = {}

    def send_result(self, msg_id: int) -> None:
        self.results.append(msg_id)

    def send_error(self, msg_id: int, code: str, message: str) -> None:
        self.errors.append((msg_id, code, message))

    def send_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)


class TestWsStartPreview:
    """The websocket handler streams state/attributes and cleans up."""

    def _msg(
        self,
        flow_id: str = "flow-1",
        flow_type: str = "options_flow",
        **user_input: Any,
    ) -> dict[str, Any]:
        return {
            "id": 7,
            "type": f"{DOMAIN}/start_preview",
            "flow_id": flow_id,
            "flow_type": flow_type,
            "user_input": user_input,
        }

    def test_a_subentry_flow_gets_a_preview(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Any,
    ) -> None:
        # Reconfiguring an area from the integration page is a subentry flow,
        # and it is the main editing path -- its preview has to work.
        area = default_area
        register_preview_context(
            hass, "flow-sub", coordinator.config_entry.entry_id, area.config.area_id, {}
        )
        connection = _FakeConnection()
        with patch.object(
            hass.config_entries,
            "async_get_entry",
            return_value=Mock(runtime_data=coordinator),
        ):
            ws_start_preview(
                hass,
                connection,
                self._msg("flow-sub", flow_type="config_subentries_flow"),
            )

        assert connection.results == [7]
        assert not connection.errors
        assert float(connection.messages[0]["event"]["state"]) >= 0

    def test_unknown_flow_is_an_error(self, hass: HomeAssistant) -> None:
        connection = _FakeConnection()
        ws_start_preview(hass, connection, self._msg("nope"))
        assert connection.errors and connection.errors[0][1] == "not_found"
        assert not connection.results

    def test_streams_preview_and_subscribes(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Any,
    ) -> None:
        area = default_area
        register_preview_context(
            hass, "flow-1", coordinator.config_entry.entry_id, area.config.area_id, {}
        )
        entry = Mock(runtime_data=coordinator)
        connection = _FakeConnection()

        with patch.object(hass.config_entries, "async_get_entry", return_value=entry):
            ws_start_preview(
                hass,
                connection,
                self._msg(
                    **{
                        CONF_THRESHOLD: 40,
                        "wasp_in_box": {"wasp_enabled": False},
                        "decay_half_life": {"hours": 0, "minutes": 5, "seconds": 0},
                    }
                ),
            )

        assert connection.results == [7]
        assert len(connection.messages) == 1
        event = connection.messages[0]["event"]
        assert event["attributes"]["threshold"] == 40
        assert float(event["state"]) >= 0
        # Subscribed to entity state changes and coordinator updates
        assert 7 in connection.subscriptions
        before = len(connection.messages)
        coordinator.async_update_listeners()
        assert len(connection.messages) == before + 1
        connection.subscriptions[7]()
        coordinator.async_update_listeners()
        assert len(connection.messages) == before + 1

    def test_area_not_loaded_reports_unavailable(self, hass: HomeAssistant) -> None:
        register_preview_context(hass, "flow-2", "entry-x", "not_an_area", {})
        connection = _FakeConnection()
        with patch.object(hass.config_entries, "async_get_entry", return_value=None):
            ws_start_preview(hass, connection, self._msg("flow-2"))
        assert connection.results == [7]
        event = connection.messages[0]["event"]
        assert event["state"] == "unavailable"
        assert "reason" in event["attributes"]
        # Nothing to unsubscribe from, but the callable must still exist
        connection.subscriptions[7]()

    def test_bad_user_input_is_an_error(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Any,
    ) -> None:
        area = default_area
        register_preview_context(
            hass, "flow-3", coordinator.config_entry.entry_id, area.config.area_id, {}
        )
        connection = _FakeConnection()
        with patch.object(
            hass.config_entries,
            "async_get_entry",
            return_value=Mock(runtime_data=coordinator),
        ):
            ws_start_preview(
                hass, connection, self._msg("flow-3", decay_half_life={"minutes": "x"})
            )
        assert connection.errors and connection.errors[0][1] == "invalid_input"


class TestCustomSensorPreview:
    """The custom channels take their weight from the form like typed ones."""

    def _add_custom_entity(
        self,
        coordinator: AreaOccupancyCoordinator,
        input_type: InputType,
        default_area: Any,
    ) -> str:
        entity_id = f"sensor.{input_type.value}"
        area = default_area
        factory = EntityFactory(coordinator, area_name=area.area_name)
        area.entities.add_entity(
            factory.create_from_config_spec(entity_id, input_type.value)
        )
        return entity_id

    @pytest.mark.parametrize(
        ("input_type", "weight_key"),
        [
            (InputType.CUSTOM_BINARY, CONF_WEIGHT_CUSTOM_BINARY),
            (InputType.CUSTOM_NUMERIC, CONF_WEIGHT_CUSTOM_NUMERIC),
        ],
    )
    def test_candidate_weight_is_applied(
        self,
        coordinator: AreaOccupancyCoordinator,
        input_type: InputType,
        weight_key: str,
        default_area: Any,
    ) -> None:
        entity_id = self._add_custom_entity(coordinator, input_type, default_area)
        area = default_area

        entities = build_preview_entities(area, {weight_key: 0.9})

        assert entities[entity_id].weight == 0.9

    @pytest.mark.parametrize(
        "input_type", [InputType.CUSTOM_BINARY, InputType.CUSTOM_NUMERIC]
    )
    def test_absent_weight_keeps_the_live_value(
        self,
        coordinator: AreaOccupancyCoordinator,
        input_type: InputType,
        default_area: Any,
    ) -> None:
        entity_id = self._add_custom_entity(coordinator, input_type, default_area)
        area = default_area
        live_weight = area.entities.entities[entity_id].weight

        entities = build_preview_entities(area, {})

        assert entities[entity_id].weight == live_weight
