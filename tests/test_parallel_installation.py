"""Tests for the parallel original and development integrations."""

import json
from pathlib import Path

from custom_components.area_occupancy.const import (
    DB_NAME as ORIGINAL_DB_NAME,
    DOMAIN as ORIGINAL_DOMAIN,
)
from custom_components.area_occupancy_dev.const import (
    DB_NAME as DEV_DB_NAME,
    DOMAIN as DEV_DOMAIN,
    FRONTEND_URL_BASE as DEV_FRONTEND_URL_BASE,
)


def test_dev_integration_uses_isolated_domain_and_database() -> None:
    """The development package must not share HA or SQLite namespaces."""
    assert ORIGINAL_DOMAIN == "area_occupancy"
    assert DEV_DOMAIN == "area_occupancy_dev"
    assert ORIGINAL_DOMAIN != DEV_DOMAIN
    assert ORIGINAL_DB_NAME == "area_occupancy.db"
    assert DEV_DB_NAME == "area_occupancy_dev.db"
    assert DEV_FRONTEND_URL_BASE == "/area_occupancy_dev/frontend"


def test_dev_manifest_and_release_target_are_consistent() -> None:
    """The fork release must install only the development integration."""
    repository_root = Path(__file__).parents[1]
    manifest = json.loads(
        (
            repository_root / "custom_components/area_occupancy_dev/manifest.json"
        ).read_text()
    )
    release_workflow = (repository_root / ".github/workflows/release.yml").read_text()

    assert manifest["domain"] == DEV_DOMAIN
    assert manifest["name"] == "Area Occupancy Detection Dev"
    assert "custom_components/area_occupancy_dev/manifest.json" in release_workflow
    assert "area_occupancy_dev.zip" in release_workflow
    assert "custom_components/area_occupancy\n" not in release_workflow
