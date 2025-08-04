"""Tests for the decay module."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from custom_components.area_occupancy.data.decay import DEFAULT_HALF_LIFE, Decay


class TestDecay:
    """Test the Decay class."""

    @pytest.mark.parametrize(
        ("kwargs", "expected_attrs"),
        [
            (
                {},
                {
                    "half_life": DEFAULT_HALF_LIFE,
                    "is_decaying": False,
                },
            ),
            (
                {"decay_start": datetime.now(), "half_life": 60.0, "is_decaying": True},
                {
                    "half_life": 60.0,
                    "is_decaying": True,
                },
            ),
        ],
    )
    def test_initialization(self, kwargs, expected_attrs, freeze_time) -> None:
        """Test initialization with various parameters."""
        decay = Decay(**kwargs)

        # Check decay_start is always a datetime
        assert isinstance(decay.decay_start, datetime)

        # Check other attributes
        for attr, expected_value in expected_attrs.items():
            assert getattr(decay, attr) == expected_value

    @pytest.mark.parametrize(
        (
            "is_decaying",
            "decay_start",
            "half_life",
            "time_offset",
            "expected_factor",
            "expected_is_decaying",
        ),
        [
            # Not decaying
            (False, None, 60.0, 0, 1.0, False),
            # Decaying - one half-life
            (True, datetime.now(), 60.0, 60.0, 0.5, True),
            # Decaying - auto-stop when factor < 0.05
            (True, datetime.now(), 60.0, 1000.0, 0.0, False),
        ],
    )
    def test_decay_factor(
        self,
        is_decaying,
        decay_start,
        half_life,
        time_offset,
        expected_factor,
        expected_is_decaying,
        freeze_time,
    ) -> None:
        """Test decay factor calculations under various conditions."""
        # Use freeze_time if decay_start is None, otherwise use provided decay_start
        start_time = freeze_time if decay_start is None else decay_start

        decay = Decay(
            decay_start=start_time,
            half_life=half_life,
            is_decaying=is_decaying,
        )

        # Mock datetime.now() to simulate time passing
        with patch(
            "custom_components.area_occupancy.data.decay.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value = start_time + timedelta(seconds=time_offset)

            # Check decay factor
            assert abs(decay.decay_factor - expected_factor) < 0.001

            # Check if decay auto-stopped
            assert decay.is_decaying == expected_is_decaying

    @pytest.mark.parametrize(
        ("initial_state", "method", "expected_state", "should_update_start"),
        [
            # Start decay tests
            (False, "start_decay", True, True),
            (True, "start_decay", True, False),
            # Stop decay tests
            (True, "stop_decay", False, False),
            (False, "stop_decay", False, False),
        ],
    )
    def test_decay_control_methods(
        self, initial_state, method, expected_state, should_update_start
    ) -> None:
        """Test start_decay and stop_decay methods."""
        decay = Decay(is_decaying=initial_state)
        original_start = decay.decay_start

        # Call the method
        getattr(decay, method)()

        # Check final state
        assert decay.is_decaying == expected_state

        # Check if decay_start was updated (only for start_decay when not already decaying)
        if should_update_start:
            assert decay.decay_start != original_start
        else:
            assert decay.decay_start == original_start

    @pytest.mark.parametrize(
        ("kwargs", "expected_values"),
        [
            # Default values
            ({}, {"half_life": DEFAULT_HALF_LIFE, "is_decaying": False}),
            # Custom values
            (
                {
                    "decay_start": datetime.now(),
                    "half_life": 120.0,
                    "is_decaying": True,
                },
                {"half_life": 120.0, "is_decaying": True},
            ),
            # Partial values
            ({"half_life": 90.0}, {"half_life": 90.0, "is_decaying": False}),
        ],
    )
    def test_create_classmethod(self, kwargs, expected_values, freeze_time) -> None:
        """Test the create classmethod with various parameters."""
        decay = Decay.create(**kwargs)

        # Check decay_start is always a datetime
        assert isinstance(decay.decay_start, datetime)

        # Check other attributes
        for attr, expected_value in expected_values.items():
            assert getattr(decay, attr) == expected_value
