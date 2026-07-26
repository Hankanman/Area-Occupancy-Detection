"""Tests for the read-side forecast helpers (data/forecast.py)."""

import pytest

from custom_components.area_occupancy.const import MAX_PRIOR, MIN_PRIOR
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.forecast import (
    build_area_time_priors,
    forecast_prior,
)
from custom_components.area_occupancy.data.prior import (
    DEFAULT_SLOT_MINUTES,
    PRIOR_FACTOR,
)
from custom_components.area_occupancy.utils import combine_priors

# ruff: noqa: SLF001


def test_forecast_prior_without_global_returns_clamped_slot():
    """With no learned global prior, the slot's own value (clamped) is used."""
    assert forecast_prior(None, 0.6, prior_factor=PRIOR_FACTOR) == pytest.approx(
        max(MIN_PRIOR, min(MAX_PRIOR, 0.6))
    )


def test_forecast_prior_combines_global_and_slot():
    """A learned global prior is combined with the slot's time prior."""
    expected = max(MIN_PRIOR, min(MAX_PRIOR, combine_priors(0.5, 0.6) * PRIOR_FACTOR))
    assert forecast_prior(0.5, 0.6, prior_factor=PRIOR_FACTOR) == pytest.approx(
        expected
    )


def test_forecast_prior_within_bounds():
    """The forecast is always clamped to [MIN_PRIOR, MAX_PRIOR]."""
    result = forecast_prior(0.99, 0.9, prior_factor=PRIOR_FACTOR)
    assert MIN_PRIOR <= result <= MAX_PRIOR


def test_forecast_prior_respects_prior_factor():
    """prior_factor scales the combined value before clamping."""
    # factor 0 → adjusted 0 → clamped up to MIN_PRIOR
    assert forecast_prior(0.5, 0.5, prior_factor=0.0) == pytest.approx(MIN_PRIOR)


def test_build_area_time_priors_structure(coordinator: AreaOccupancyCoordinator):
    """build_area_time_priors emits area_id, global_prior, slot_minutes, slots."""
    area_name = coordinator.get_area_names()[0]
    area = coordinator.get_area(area_name)
    area.prior.global_prior = 0.5
    area.prior._cached_time_priors = {(0, 0): 0.4, (2, 10): 0.6}

    data = build_area_time_priors(area, DEFAULT_SLOT_MINUTES)

    assert data["area_id"] == area.config.area_id
    assert data["global_prior"] == 0.5
    assert data["slot_minutes"] == DEFAULT_SLOT_MINUTES
    assert set(data["slots"]) == {"0,0", "2,10"}
    # Response wires through Prior.prior_for (which delegates to forecast_prior).
    assert data["slots"]["2,10"] == round(area.prior.prior_for(2, 10), 4)
