"""Tests for diagnostics export."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.adjacency import (
    BoostContribution,
    DecayModifierContribution,
)
from custom_components.area_occupancy.diagnostics import (
    async_get_config_entry_diagnostics,
)
from homeassistant.core import HomeAssistant


@pytest.fixture
def entry_with_runtime_data(
    coordinator: AreaOccupancyCoordinator,
):
    """Wire the autouse coordinator into the config entry's runtime_data.

    Mirrors what __init__.async_setup_entry does in production so the
    diagnostics entry point can resolve the coordinator from the entry.
    """
    entry = coordinator.config_entry
    entry.runtime_data = coordinator
    return entry


class TestDiagnosticsExport:
    """Cover the public async_get_config_entry_diagnostics entry point."""

    @pytest.mark.asyncio
    async def test_returns_full_snapshot(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """Happy path: snapshot has integration, areas, database sections."""
        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        assert "integration" in result
        assert "areas" in result
        assert "database" in result

        integration = result["integration"]
        assert integration["entry_id"] == entry_with_runtime_data.entry_id
        assert integration["area_count"] == len(coordinator.areas)
        assert "version" in integration
        assert "config_version" in integration
        assert "setup_complete" in integration

    @pytest.mark.asyncio
    async def test_snapshot_is_json_serializable(
        self,
        hass: HomeAssistant,
        entry_with_runtime_data,
    ) -> None:
        """HA core serializes diagnostics as JSON; nothing exotic must leak."""
        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        # json.dumps will raise TypeError on datetimes, enums, dataclasses, etc.
        json.dumps(result)

    @pytest.mark.asyncio
    async def test_returns_error_when_coordinator_missing(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Diagnostics must degrade gracefully when called pre-setup."""
        entry = coordinator.config_entry
        entry.runtime_data = None

        result = await async_get_config_entry_diagnostics(hass, entry)

        assert result["error"] == "coordinator_not_initialized"
        assert result["entry_id"] == entry.entry_id

    @pytest.mark.asyncio
    async def test_each_area_has_expected_sections(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """Per-area snapshot exposes config, current state, prior, entities, health."""
        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        assert len(result["areas"]) == len(coordinator.areas)
        for area in result["areas"]:
            assert "area_name" in area
            assert "purpose" in area
            assert "threshold" in area
            assert "current" in area
            assert "prior" in area
            assert "config" in area
            assert "entities" in area
            assert "health" in area

            current = area["current"]
            assert "probability" in current
            assert "occupied" in current
            assert "decay_factor" in current
            assert "entity_count" in current

            prior = area["prior"]
            assert "prior_value" in prior
            assert "global_prior" in prior
            assert "time_prior" in prior
            assert "min_prior_floor_applied" in prior

    @pytest.mark.asyncio
    async def test_entity_snapshot_shape(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """Each entity entry surfaces likelihoods, evidence, and decay state."""
        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        for area in result["areas"]:
            for entity in area["entities"]:
                assert "entity_id" in entity
                assert "input_type" in entity
                assert "weight" in entity
                assert "prob_given_true" in entity
                assert "prob_given_false" in entity
                assert "evidence" in entity
                assert "decay" in entity

                decay = entity["decay"]
                assert "is_decaying" in decay
                assert "half_life" in decay
                assert "decay_factor" in decay

    @pytest.mark.asyncio
    async def test_health_section_present_with_no_issues(
        self,
        hass: HomeAssistant,
        entry_with_runtime_data,
    ) -> None:
        """Fresh coordinator has no health issues but still exports the section."""
        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        for area in result["areas"]:
            health = area["health"]
            assert "issue_count" in health
            assert health["issue_count"] == len(health["issues"])

    @pytest.mark.asyncio
    async def test_pipeline_health_issue_serializes_with_null_input_type(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """Pipeline issues have ``input_type=None``; diagnostics must not crash.

        Regression: an earlier version called ``issue.input_type.value``
        unconditionally. For pipeline-scope issues that's an
        ``AttributeError`` and the whole area's health snapshot fell back
        to a ``health_error`` field, hiding the issue from the dump.
        """
        # Seed a pipeline-scope issue on every area so the serializer hits
        # the null input_type branch.
        with patch("custom_components.area_occupancy.data.health.ir"):
            for area in coordinator.areas.values():
                area.health_monitor.check_pipeline_health(
                    area_age_hours=24 * 14,
                    has_global_prior=False,
                    cache_age_hours=1.0,
                    last_analysis_duration_ms=None,
                    correlation_failure_count=0,
                    correlatable_entity_count=0,
                )

        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        for area in result["areas"]:
            assert "health_error" not in area
            health = area["health"]
            pipeline = [
                i for i in health["issues"] if i["issue_type"] == "insufficient_priors"
            ]
            assert len(pipeline) == 1
            assert pipeline[0]["input_type"] is None
            assert pipeline[0]["entity_id"] is None

    @pytest.mark.asyncio
    async def test_database_section_includes_counts(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """Database section reports row counts and per-area cache freshness."""
        coordinator.db.init_db()

        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        database = result["database"]
        assert "interval_count" in database
        assert "prior_count" in database
        assert "correlation_count" in database
        assert "occupied_intervals_cache" in database

        cache = database["occupied_intervals_cache"]
        for area_name in coordinator.areas:
            assert area_name in cache

    @pytest.mark.asyncio
    async def test_resilient_to_per_area_failure(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """A failure in one section must not poison the whole diagnostic."""
        # Force area.probability() to raise; the other sections must still come back.
        with patch(
            "custom_components.area_occupancy.area.area.Area.probability",
            side_effect=RuntimeError("boom"),
        ):
            result = await async_get_config_entry_diagnostics(
                hass, entry_with_runtime_data
            )

        assert "areas" in result
        for area in result["areas"]:
            # The area entry exists and captured the failure of the failing section.
            assert "area_name" in area
            assert "current_error" in area
            # Other sections still populated.
            assert "prior" in area
            assert "entities" in area
            assert "health" in area

    @pytest.mark.asyncio
    async def test_adjacency_block_present_when_cached(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """Adjacency boost + decay modifier surface under current.adjacency."""
        area_name = coordinator.get_area_names()[0]
        coordinator._adjacency_boosts = {  # noqa: SLF001
            area_name: BoostContribution(
                fired=True,
                trajectory_prev="hallway",
                trajectory_prev_prev="kitchen",
                hour_of_week=42,
                raw_probability=0.8,
                fallback_level="2hop_hour_of_week",
                observed_count=8.0,
                total_count=10.0,
                logit_contribution=0.69,
            )
        }
        coordinator._adjacency_decay_modifiers = {  # noqa: SLF001
            area_name: DecayModifierContribution(
                fired=True,
                silence_score=0.8,
                decay_modifier=1.6,
                base_half_life_seconds=1.0,
                effective_half_life_seconds=1.6,
                silent_neighbours=[("hallway", 0.1, 0.8)],
            )
        }

        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        area_snapshot = next(a for a in result["areas"] if a["area_name"] == area_name)
        adjacency = area_snapshot["current"]["adjacency"]
        boost = adjacency["boost"]
        assert boost["fired"] is True
        assert boost["trajectory_prev"] == "hallway"
        assert boost["trajectory_prev_prev"] == "kitchen"
        assert boost["hour_of_week"] == 42
        assert boost["fallback_level"] == "2hop_hour_of_week"
        assert boost["logit_contribution"] == 0.69

        modifier = adjacency["decay_modifier"]
        assert modifier["fired"] is True
        assert modifier["silence_score"] == 0.8
        assert modifier["decay_modifier"] == 1.6
        assert modifier["silent_neighbours"] == [
            {
                "neighbour": "hallway",
                "lagged_probability": 0.1,
                "transition_probability": 0.8,
            }
        ]

        # And the whole thing must stay JSON-serializable
        json.dumps(result)

    @pytest.mark.asyncio
    async def test_adjacency_block_omitted_before_first_tick(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        entry_with_runtime_data,
    ) -> None:
        """No cached adjacency state → no adjacency key in the snapshot."""
        area_name = coordinator.get_area_names()[0]

        result = await async_get_config_entry_diagnostics(hass, entry_with_runtime_data)

        area_snapshot = next(a for a in result["areas"] if a["area_name"] == area_name)
        assert "adjacency" not in area_snapshot["current"]
