"""The user-facing strings match what the flows actually render.

Home Assistant silently falls back to the raw config key when a section,
field or error has no string, so a gap here is invisible in tests and only
shows up as ``custom_binary_sensors`` where a label should be. These checks
walk the real flow constants rather than a copy of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from custom_components.area_occupancy.config_flow import SENSOR_GROUPS

COMPONENT = Path("custom_components/area_occupancy")
STRINGS = COMPONENT / "strings.json"
TRANSLATIONS = COMPONENT / "translations/en.json"

# Every flow scope that renders the sectioned sensors step. The subentry
# flow is the one users reach from the integration page, so a gap there is
# the most visible of the three.
SENSOR_STEP_SCOPES: tuple[tuple[str, ...], ...] = (
    ("config", "step", "area_sensors"),
    ("options", "step", "area_sensors"),
    ("config_subentries", "area", "step", "area_sensors"),
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dig(data: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    for key in path:
        data = data[key]
    return data


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        out: dict[str, Any] = {}
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}.{key}"))
        return out
    return {prefix: data}


@pytest.fixture(name="strings")
def strings_fixture() -> dict[str, Any]:
    """The source strings file."""
    return _load(STRINGS)


class TestSensorSections:
    """Each sensor group needs a section string in every flow that shows it."""

    @pytest.mark.parametrize("scope", SENSOR_STEP_SCOPES)
    def test_every_group_has_a_section(
        self, strings: dict[str, Any], scope: tuple[str, ...]
    ) -> None:
        sections = _dig(strings, scope)["sections"]
        assert set(SENSOR_GROUPS) <= set(sections), (
            f"{'.'.join(scope)} is missing sections for "
            f"{sorted(set(SENSOR_GROUPS) - set(sections))}"
        )

    @pytest.mark.parametrize("scope", SENSOR_STEP_SCOPES)
    def test_every_section_is_a_real_group(
        self, strings: dict[str, Any], scope: tuple[str, ...]
    ) -> None:
        sections = _dig(strings, scope)["sections"]
        assert set(sections) <= set(SENSOR_GROUPS), (
            f"{'.'.join(scope)} has sections for groups the flow never "
            f"renders: {sorted(set(sections) - set(SENSOR_GROUPS))}"
        )

    @pytest.mark.parametrize("scope", SENSOR_STEP_SCOPES)
    def test_every_field_is_labelled(
        self, strings: dict[str, Any], scope: tuple[str, ...]
    ) -> None:
        for name, section in _dig(strings, scope)["sections"].items():
            assert section.get("name"), f"{'.'.join(scope)}.{name} has no name"
            assert section.get("data"), f"{'.'.join(scope)}.{name} has no labels"


class TestTranslationsMatchStrings:
    """``translations/en.json`` is generated from ``strings.json``."""

    def test_identical(self) -> None:
        assert _flatten(_load(STRINGS)) == _flatten(_load(TRANSLATIONS)), (
            "translations/en.json has drifted from strings.json"
        )
