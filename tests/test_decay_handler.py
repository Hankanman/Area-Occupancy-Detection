"""Unit tests for the DecayHandler class."""

from datetime import datetime, timedelta

import pytest

from custom_components.area_occupancy.const import DEFAULT_DECAY_WINDOW
from custom_components.area_occupancy.decay_handler import DecayHandler


def test_decay_disabled_returns_current_probability():
    """Test that the decay handler returns the current probability when decay is disabled."""
    handler = DecayHandler(
        {"decay_enabled": False, "decay_window": DEFAULT_DECAY_WINDOW}
    )
    result = handler.calculate_decay(0.5, 0.4, False, None, None)
    assert result[0] == 0.5
    assert result[2] is False


def test_decay_window_validation():
    """Test that the decay handler raises an exception if the decay window is 0."""
    with pytest.raises(Exception):
        DecayHandler({"decay_enabled": True, "decay_window": 0})


def test_decay_start_and_stop():
    """Test that the decay handler starts and stops decay correctly."""
    handler = DecayHandler(
        {"decay_enabled": True, "decay_window": DEFAULT_DECAY_WINDOW}
    )
    # Start decay
    now = datetime.now()
    result = handler.calculate_decay(0.3, 0.5, False, None, None)
    assert result[2] is True
    # Continue decay
    result2 = handler.calculate_decay(0.2, 0.3, True, now - timedelta(seconds=10), 0.5)
    assert result2[2] is True or result2[0] >= 0.01


def test_decay_inconsistent_state():
    """Test that the decay handler returns True when is_decaying is True but no start time/probability."""
    handler = DecayHandler(
        {"decay_enabled": True, "decay_window": DEFAULT_DECAY_WINDOW}
    )
    # is_decaying True but no start time/probability
    result = handler.calculate_decay(0.2, 0.3, True, None, None)
    assert result[2] is True


def test_decay_error_handling():
    """Test that the decay handler raises a ValueError when the probabilities are invalid."""
    handler = DecayHandler(
        {"decay_enabled": True, "decay_window": DEFAULT_DECAY_WINDOW}
    )
    # Invalid probabilities
    with pytest.raises(ValueError):
        handler.calculate_decay(-1, 0.5, False, None, None)
        handler.calculate_decay(-1, 0.5, False, None, None)
