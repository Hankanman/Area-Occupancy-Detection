"""Reproducible documentation screenshots from a harness instance.

Documentation screenshots rot: the config flow changes, the pictures do not,
and a reader following a walkthrough ends up looking for a button that moved
two releases ago. Capturing them from a seeded harness instance makes them
reproducible -- same profile, same seed, same shots, any time the UI changes.

Playwright is an optional extra (``uv sync --extra shots``); everything else
in the harness works without it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

from .instance import Instance

#: Where the docs keep their images.
DEFAULT_OUTPUT = Path("docs/docs/images")

#: The config-flow dialog. Home Assistant nests the real, sized element
#: several shadow roots down -- the outer ha-dialog measures 1440x0 -- and
#: leaves inert dialogs in the DOM, so match the visible native one.
DIALOG = "dialog[open]:visible"

#: Wide enough for the integration page's two-column layout.
VIEWPORT = {"width": 1440, "height": 1000}


@dataclass(slots=True)
class Shot:
    """One screenshot to capture.

    Attributes:
        name: File name without extension; also the docs reference.
        description: What the shot shows, printed while capturing.
        steps: Clicks and waits to reach the screen, run in order.
        target: Selector to crop to, or ``None`` for the whole viewport.
        viewport: A taller window for this shot. Home Assistant sizes a
            dialog to the window, so a long form is clipped at the default
            height -- and the preview, which renders under the last field, is
            exactly what gets cut off.
    """

    name: str
    description: str
    steps: list[Callable[[Any], None]] = field(default_factory=list)
    target: str | None = None
    viewport: dict[str, int] | None = None


class ShotError(RuntimeError):
    """A screenshot could not be captured."""


def _install_auth(context: Any, instance: Instance) -> None:
    """Seed the instance's token so the browser never sees the login form.

    Installed as an init script rather than written after a navigation: the
    frontend redirects to the auth page as soon as it loads without a token,
    which destroys the execution context mid-write.

    Args:
        context: The Playwright browser context.
        instance: The instance being photographed.

    Raises:
        ShotError: If the instance has no token to install.
    """
    token = instance.meta.get("token")
    if not token:
        raise ShotError("the instance has no token; start it with the harness first")

    tokens = {
        "access_token": token,
        "token_type": "Bearer",
        "refresh_token": instance.meta.get("refresh_token"),
        "expires_in": 1800,
        "hassUrl": instance.base_url,
        "clientId": f"{instance.base_url}/",
        "expires": int(time.time() * 1000) + 1_500_000,
    }
    context.add_init_script(
        f"window.localStorage.setItem('hassTokens', {json.dumps(json.dumps(tokens))});"
        # Skip the onboarding "what's new" dialogs a fresh profile shows.
        f"window.localStorage.setItem('dockedSidebar', '\"always_hidden\"');"
    )


def _dismiss_overlays(page: Any) -> None:
    """Close anything Home Assistant shows over the page on first load.

    Args:
        page: The Playwright page.
    """
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)


def _click_text(page: Any, text: str, *, exact: bool = True, index: int = 0) -> None:
    """Click the nth element with this text, waiting for it to appear.

    Args:
        page: The Playwright page.
        text: Text to match.
        exact: Whether the match must be exact.
        index: Which match to click when several are present.

    Raises:
        ShotError: If nothing with that text turns up.
    """
    locator = page.get_by_text(text, exact=exact).nth(index)
    try:
        locator.wait_for(state="visible", timeout=15000)
    except Exception as err:
        raise ShotError(f"never found {text!r} on screen") from err
    locator.click()
    page.wait_for_timeout(900)


def _click_button(page: Any, name: str, *, index: int = 0) -> None:
    """Click a button by accessible name.

    Args:
        page: The Playwright page.
        name: The button's accessible name.
        index: Which match to click when several are present.

    Raises:
        ShotError: If the button never appears.
    """
    locator = page.get_by_role("button", name=name).nth(index)
    try:
        locator.wait_for(state="visible", timeout=15000)
    except Exception as err:
        raise ShotError(f"never found a {name!r} button") from err
    locator.click()
    page.wait_for_timeout(900)


def _open_integration_page(page: Any, instance: Instance) -> None:
    """Navigate to the integration's own page and let it settle.

    Args:
        page: The Playwright page.
        instance: The instance being photographed.
    """
    page.goto(f"{instance.base_url}/config/integrations/integration/area_occupancy")
    page.wait_for_timeout(3500)
    _dismiss_overlays(page)


def _open_area_menu(page: Any, instance: Instance, area: str = "Living Room") -> None:
    """Open one area's edit menu from the integration page.

    Args:
        page: The Playwright page.
        instance: The instance being photographed.
        area: Which area's menu to open.

    Raises:
        ShotError: If the area's row never appears.
    """
    _open_integration_page(page, instance)
    row = page.locator("ha-config-sub-entry-row", has_text=area).first
    try:
        row.wait_for(state="visible", timeout=15000)
    except Exception as err:
        raise ShotError(f"no subentry row for {area!r}") from err
    # The gear on an area's row opens that area's reconfigure flow.
    row.get_by_role("button", name="Reconfigure this area").first.click()
    page.wait_for_timeout(2000)


def build_shots(instance: Instance) -> list[Shot]:
    """The documentation set, in the order a reader meets them.

    Args:
        instance: The instance being photographed.

    Returns:
        Every shot to capture.
    """
    return [
        Shot(
            name="config_integration_page",
            description="areas as config subentries on the integration page",
            steps=[lambda page: _open_integration_page(page, instance)],
        ),
        Shot(
            name="config_add_area",
            description="adding an area: purpose and adjacent areas",
            steps=[
                lambda page: _open_integration_page(page, instance),
                lambda page: _click_button(page, "Add an area"),
            ],
            target=DIALOG,
        ),
        Shot(
            name="config_area_menu",
            description="an area's edit menu, each entry summarising its page",
            steps=[lambda page: _open_area_menu(page, instance)],
            target=DIALOG,
        ),
        Shot(
            name="config_area_behaviour",
            description="detection behaviour with the live preview beside it",
            steps=[
                lambda page: _open_area_menu(page, instance),
                lambda page: _click_text(page, "Detection behaviour"),
            ],
            target=DIALOG,
            viewport={"width": 1440, "height": 1400},
        ),
        Shot(
            name="config_area_sensors_menu",
            description="the additional-sensors menu, one entry per group",
            steps=[
                lambda page: _open_area_menu(page, instance),
                lambda page: _click_text(page, "Additional sensors"),
            ],
            target=DIALOG,
        ),
        Shot(
            name="config_custom_sensors",
            description="custom sensors: unfiltered binary and numeric entities",
            steps=[
                lambda page: _open_area_menu(page, instance),
                lambda page: _click_text(page, "Additional sensors"),
                lambda page: _click_text(page, "Custom Sensors"),
            ],
            target=DIALOG,
        ),
        Shot(
            name="config_options_menu",
            description="the Configure dialog: settings that are not per-area",
            steps=[
                lambda page: _open_integration_page(page, instance),
                lambda page: _click_button(page, "Configure"),
            ],
            target=DIALOG,
        ),
        Shot(
            name="config_global_settings",
            description="global settings: sleep schedule, health, precision",
            steps=[
                lambda page: _open_integration_page(page, instance),
                lambda page: _click_button(page, "Configure"),
                lambda page: _click_text(page, "Global Settings"),
            ],
            target=DIALOG,
        ),
    ]


def capture(
    instance: Instance,
    output: Path = DEFAULT_OUTPUT,
    *,
    only: list[str] | None = None,
) -> list[tuple[str, bool, str]]:
    """Capture the documentation screenshots.

    Args:
        instance: A started instance to photograph.
        output: Directory to write PNGs into.
        only: Capture just these shot names, or all of them when omitted.

    Returns:
        One ``(name, captured, detail)`` per shot attempted.

    Raises:
        ShotError: If Playwright is not installed.
    """
    try:
        from playwright.sync_api import sync_playwright  # noqa: PLC0415
    except ImportError as err:
        raise ShotError(
            "playwright is not installed; run `uv sync --extra shots`"
        ) from err

    output.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, bool, str]] = []
    shots = [shot for shot in build_shots(instance) if not only or shot.name in only]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/opt/pw-browsers/chromium"
        )
        context = browser.new_context(
            viewport=VIEWPORT, device_scale_factor=2, color_scheme="light"
        )
        _install_auth(context, instance)
        page = context.new_page()
        try:
            for shot in shots:
                destination = output / f"{shot.name}.png"
                try:
                    page.set_viewport_size(shot.viewport or VIEWPORT)
                    for step in shot.steps:
                        step(page)
                    page.wait_for_timeout(1200)
                    if shot.target:
                        element = page.locator(shot.target).last
                        element.wait_for(state="visible", timeout=10000)
                        element.screenshot(path=str(destination))
                    else:
                        page.screenshot(path=str(destination))
                except Exception as err:  # noqa: BLE001 - report, keep going
                    results.append((shot.name, False, f"{type(err).__name__}: {err}"))
                    continue
                size = destination.stat().st_size // 1024
                results.append((shot.name, True, f"{shot.description} ({size} KB)"))
        finally:
            browser.close()

    return results
