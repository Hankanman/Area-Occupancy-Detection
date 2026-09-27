"""Tests for the bundled time-priors Lovelace card registration (issue #559)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

import custom_components.area_occupancy as aod_init
from custom_components.area_occupancy import async_setup
from custom_components.area_occupancy.const import (
    DOMAIN,
    FRONTEND_DIR,
    FRONTEND_REGISTERED_KEY,
    FRONTEND_URL_BASE,
    TIME_PRIORS_CARD_FILENAME,
)
from homeassistant.core import HomeAssistant

COMPONENT_DIR = Path(aod_init.__file__).parent
CARD_URL = f"{FRONTEND_URL_BASE}/{TIME_PRIORS_CARD_FILENAME}"


@pytest.fixture
def frontend_mocks(hass: HomeAssistant):
    """Stub the http/frontend APIs async_setup talks to."""
    hass.http = Mock()
    hass.http.async_register_static_paths = AsyncMock()
    integration = Mock(version="2026.9.99")
    with (
        patch.object(aod_init, "add_extra_js_url") as add_js,
        patch.object(
            aod_init, "async_get_integration", AsyncMock(return_value=integration)
        ),
    ):
        yield hass.http.async_register_static_paths, add_js


async def test_registers_static_path_for_bundled_card(
    hass: HomeAssistant, frontend_mocks
) -> None:
    """The static path points at the card file shipped inside the integration."""
    register, _ = frontend_mocks

    assert await async_setup(hass, {}) is True

    register.assert_awaited_once()
    (configs,) = register.await_args.args
    assert len(configs) == 1
    config = configs[0]
    assert config.url_path == CARD_URL
    assert Path(config.path) == COMPONENT_DIR / FRONTEND_DIR / TIME_PRIORS_CARD_FILENAME
    assert config.cache_headers is True


async def test_registers_versioned_js_url_once(
    hass: HomeAssistant, frontend_mocks
) -> None:
    """The extra module URL carries the integration version and is added once."""
    register, add_js = frontend_mocks

    assert await async_setup(hass, {}) is True
    assert await async_setup(hass, {}) is True

    add_js.assert_called_once_with(hass, f"{CARD_URL}?v=2026.9.99")
    register.assert_awaited_once()
    assert hass.data[FRONTEND_REGISTERED_KEY] is True


async def test_missing_card_file_warns_and_setup_continues(
    hass: HomeAssistant, frontend_mocks, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing card file is logged and skipped; setup still succeeds."""
    register, add_js = frontend_mocks

    with (
        patch.object(aod_init, "TIME_PRIORS_CARD_FILENAME", "does-not-exist.js"),
        caplog.at_level(logging.WARNING, logger=aod_init.__name__),
    ):
        assert await async_setup(hass, {}) is True

    register.assert_not_awaited()
    add_js.assert_not_called()
    assert FRONTEND_REGISTERED_KEY not in hass.data
    assert "Time-priors card not found" in caplog.text


def test_card_ships_inside_integration() -> None:
    """The card lives in the integration package so the release zip carries it."""
    card = COMPONENT_DIR / FRONTEND_DIR / TIME_PRIORS_CARD_FILENAME
    assert card.is_file()
    source = card.read_text(encoding="utf-8")
    # A second load (old manual /local resource + bundled URL) must not throw.
    assert "if (customElements.get(CARD_TAG)) return;" in source
    # Loaded as an extra module the card can define itself before the
    # frontend patches CustomElementRegistry, which hides that definition from
    # Lovelace ("Configuration error"); it registers again after the patch.
    assert 'customElements.whenDefined("home-assistant").then(registerCard)' in source


def test_manifest_declares_http_and_frontend() -> None:
    """Serving the card needs http and frontend set up before the integration."""
    manifest = json.loads((COMPONENT_DIR / "manifest.json").read_text())
    assert manifest["domain"] == DOMAIN
    assert {"http", "frontend"} <= set(manifest["dependencies"])
