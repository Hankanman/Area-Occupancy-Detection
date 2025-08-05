"""Tests for data.decay module."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from custom_components.area_occupancy.data.decay import Decay
from homeassistant.util import dt as dt_util


class TestDecay:
    """Test the Decay class."""

    @pytest.mark.parametrize(
        ("kwargs", "expected_is_decaying", "expected_half_life"),
        [
            ({}, False, 30.0),
            ({"is_decaying": True, "half_life": 60.0}, True, 60.0),
        ],
    )
    def test_initialization(
        self, kwargs: dict, expected_is_decaying: bool, expected_half_life: float
    ) -> None:
        """Test decay initialization."""
        decay = Decay(**kwargs)
        assert decay.is_decaying == expected_is_decaying
        assert decay.half_life == expected_half_life
        assert isinstance(decay.decay_start, datetime)

    @pytest.mark.parametrize(
        (
            "is_decaying",
            "decay_start",
            "half_life",
            "age_seconds",
            "expected_factor",
            "expected_is_decaying",
        ),
        [
            (False, None, 60.0, 0, 1.0, False),
            (True, dt_util.utcnow(), 60.0, 60.0, 0.5, True),
            (True, dt_util.utcnow(), 60.0, 1000.0, 0.0, False),
        ],
    )
    def test_decay_factor(
        self,
        is_decaying: bool,
        decay_start: datetime | None,
        half_life: float,
        age_seconds: float,
        expected_factor: float,
        expected_is_decaying: bool,
    ) -> None:
        """Test decay factor calculation."""
        decay = Decay(
            decay_start=decay_start,
            half_life=half_life,
            is_decaying=is_decaying,
        )

        # Mock datetime.now() to simulate time passing
        with patch("homeassistant.util.dt.utcnow") as mock_utcnow:
            if decay_start:
                mock_utcnow.return_value = decay_start + timedelta(seconds=age_seconds)
            else:
                mock_utcnow.return_value = dt_util.utcnow()

            factor = decay.decay_factor
            assert abs(factor - expected_factor) < 0.01
            assert decay.is_decaying == expected_is_decaying

    @pytest.mark.parametrize(
        ("initial_state", "method", "expected_is_decaying"),
        [
            (False, "start_decay", True),
            (True, "start_decay", True),  # Already decaying
            (True, "stop_decay", False),
            (False, "stop_decay", False),  # Already stopped
        ],
    )
    def test_decay_control_methods(
        self, initial_state: bool, method: str, expected_is_decaying: bool
    ) -> None:
        """Test decay control methods."""
        decay = Decay(is_decaying=initial_state)
        original_start = decay.decay_start

        # Call the method
        getattr(decay, method)()

        assert decay.is_decaying == expected_is_decaying

        # Check if decay_start was updated
        if method == "start_decay" and not initial_state:
            assert decay.decay_start > original_start
        else:
            assert decay.decay_start == original_start

    @pytest.mark.parametrize(
        ("kwargs", "expected_values"),
        [
            (
                {},
                {"is_decaying": False, "half_life": 30.0},
            ),
            (
                {
                    "decay_start": dt_util.utcnow(),
                    "half_life": 60.0,
                    "is_decaying": True,
                },
                {"is_decaying": True, "half_life": 60.0},
            ),
            (
                {"half_life": 120.0},
                {"is_decaying": False, "half_life": 120.0},
            ),
        ],
    )
    def test_create_classmethod(self, kwargs: dict, expected_values: dict) -> None:
        """Test create classmethod."""
        decay = Decay.create(**kwargs)
        assert decay.is_decaying == expected_values["is_decaying"]
        assert decay.half_life == expected_values["half_life"]
        assert isinstance(decay.decay_start, datetime)
