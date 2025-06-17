"""Tests for the decay module."""

from datetime import datetime, timedelta
import time
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.data.decay import Decay, DEFAULT_HALF_LIFE
from custom_components.area_occupancy.const import MIN_PROBABILITY


class TestDecay:
    """Test the Decay class."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        decay = Decay()
        assert decay.last_trigger_ts > 0
        assert decay.half_life == DEFAULT_HALF_LIFE
        assert decay.is_decaying is False

    def test_initialization_with_validation(self, freeze_time) -> None:
        """Test initialization with validation."""
        decay = Decay(
            last_trigger_ts=freeze_time.timestamp(),
            half_life=60.0,
            is_decaying=True,
        )
        assert decay.last_trigger_ts == freeze_time.timestamp()
        assert decay.half_life == 60.0
        assert decay.is_decaying is True

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        decay = Decay(
            last_trigger_ts=1000.0,
            half_life=60.0,
            is_decaying=True,
        )
        data = decay.to_dict()
        assert data["last_trigger_ts"] == 1000.0
        assert data["half_life"] == 60.0
        assert data["is_decaying"] is True

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "last_trigger_ts": 1000.0,
            "half_life": 60.0,
            "is_decaying": True,
        }
        decay = Decay.from_dict(data)
        assert decay.last_trigger_ts == 1000.0
        assert decay.half_life == 60.0
        assert decay.is_decaying is True

    def test_decay_factor_not_decaying(self) -> None:
        """Test decay factor when not decaying."""
        decay = Decay(is_decaying=False)
        assert decay.decay_factor == 1.0

    def test_decay_factor_decaying(self, freeze_time) -> None:
        """Test decay factor when decaying."""
        decay = Decay(
            last_trigger_ts=freeze_time.timestamp(),
            half_life=60.0,
            is_decaying=True,
        )
        # After one half-life, factor should be 0.5
        with patch("time.time", return_value=freeze_time.timestamp() + 60.0):
            assert abs(decay.decay_factor - 0.5) < 0.001

    def test_decay_factor_auto_stop(self, freeze_time) -> None:
        """Test auto-stop of decay when factor becomes negligible."""
        decay = Decay(
            last_trigger_ts=freeze_time.timestamp(),
            half_life=60.0,
            is_decaying=True,
        )
        # After many half-lives, decay should auto-stop
        with patch("time.time", return_value=freeze_time.timestamp() + 1000.0):
            assert decay.decay_factor == 0.0
            assert decay.is_decaying is False

    def test_should_start_decay(self) -> None:
        """Test should_start_decay logic."""
        decay = Decay(is_decaying=False)

        # Should start decay on falling edge
        assert decay.should_start_decay(True, False) is True

        # Should not start if already decaying
        decay.is_decaying = True
        assert decay.should_start_decay(True, False) is False

        # Should not start on rising edge
        decay.is_decaying = False
        assert decay.should_start_decay(False, True) is False

    def test_should_stop_decay(self) -> None:
        """Test should_stop_decay logic."""
        decay = Decay(is_decaying=True)

        # Should stop decay on rising edge
        assert decay.should_stop_decay(False, True) is True

        # Should not stop if not decaying
        decay.is_decaying = False
        assert decay.should_stop_decay(False, True) is False

        # Should not stop on falling edge
        decay.is_decaying = True
        assert decay.should_stop_decay(True, False) is False

    def test_is_decay_complete(self, freeze_time) -> None:
        """Test is_decay_complete logic."""
        decay = Decay(is_decaying=True)

        # Not decaying
        decay.is_decaying = False
        assert decay.is_decay_complete(0.5) is True

        # No trigger time
        decay.is_decaying = True
        decay.last_trigger_ts = 0
        assert decay.is_decay_complete(0.5) is True

        # Probability at minimum
        decay.last_trigger_ts = time.time()
        assert decay.is_decay_complete(0.0) is True

        # Probability at practical threshold
        assert decay.is_decay_complete(0.02) is True

        # Decay factor negligible - use time manipulation instead of patching
        decay.last_trigger_ts = freeze_time.timestamp()
        with patch("time.time", return_value=freeze_time.timestamp() + 1000.0):
            assert decay.is_decay_complete(0.5) is True

        # Not complete - use time manipulation instead of patching
        decay.last_trigger_ts = freeze_time.timestamp()
        decay.is_decaying = True  # Ensure decay is still active
        with patch("time.time", return_value=freeze_time.timestamp() + 30.0):
            # Check decay factor first to ensure it's not auto-stopping
            assert decay.decay_factor > 0.05  # Should be well above auto-stop threshold
            assert decay.is_decay_complete(0.5) is False
