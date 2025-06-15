"""Tests for data.decay module."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest

from custom_components.area_occupancy.data.decay import (
    DECAY_COMPLETION_THRESHOLD,
    DECAY_FACTOR_THRESHOLD,
    MAX_DECAY_DURATION_MULTIPLIER,
    Decay,
)
from homeassistant.util import dt as dt_util


class TestDecay:
    """Test Decay class."""

    def test_initialization(self) -> None:
        """Test decay initialization with valid parameters."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        assert not decay.is_decaying
        assert decay.decay_start_time is None
        assert decay.decay_start_probability == 0.5
        assert decay.decay_window == 300
        assert decay.decay_enabled
        assert decay.decay_factor == 1.0

    def test_initialization_with_validation(self) -> None:
        """Test initialization with invalid values that should be validated."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=-0.1,  # Invalid, should be clamped
            decay_window=0,  # Invalid, should be set to 1
            decay_enabled=True,
            decay_factor=1.5,  # Invalid, should be clamped
        )

        assert decay.decay_start_probability == 0.0  # Clamped
        assert decay.decay_window == 1  # Minimum value
        assert decay.decay_factor == 1.0  # Clamped

    def test_to_dict(self) -> None:
        """Test converting decay to dictionary."""
        now = dt_util.utcnow()
        decay = Decay(
            is_decaying=True,
            decay_start_time=now,
            decay_start_probability=0.8,
            decay_window=600,
            decay_enabled=True,
            decay_factor=0.9,
        )

        result = decay.to_dict()
        expected = {
            "is_decaying": True,
            "decay_start_time": now.isoformat(),
            "decay_start_probability": 0.8,
            "decay_window": 600,
            "decay_enabled": True,
            "decay_factor": 0.9,
        }

        assert result == expected

    def test_to_dict_with_none_time(self) -> None:
        """Test converting decay to dictionary with None start time."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        result = decay.to_dict()
        assert result["decay_start_time"] is None

    def test_from_dict(self) -> None:
        """Test creating decay from dictionary."""
        now = dt_util.utcnow()
        data = {
            "is_decaying": True,
            "decay_start_time": now.isoformat(),
            "decay_start_probability": 0.8,
            "decay_window": 600,
            "decay_enabled": True,
            "decay_factor": 0.9,
        }

        decay = Decay.from_dict(data)

        assert decay.is_decaying
        assert decay.decay_start_time == now
        assert decay.decay_start_probability == 0.8
        assert decay.decay_window == 600
        assert decay.decay_enabled
        assert decay.decay_factor == 0.9

    def test_from_dict_with_none_time(self) -> None:
        """Test creating decay from dictionary with None start time."""
        data = {
            "is_decaying": False,
            "decay_start_time": None,
            "decay_start_probability": 0.5,
            "decay_window": 300,
            "decay_enabled": True,
            "decay_factor": 1.0,
        }

        decay = Decay.from_dict(data)
        assert decay.decay_start_time is None

    @patch("homeassistant.util.dt.utcnow")
    def test_update_decay_disabled(self, mock_utcnow: Mock) -> None:
        """Test update_decay when decay is disabled."""
        mock_utcnow.return_value = dt_util.utcnow()

        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=False,  # Disabled
            decay_factor=1.0,
        )

        result_prob, result_factor = decay.update_decay(0.3, 0.5)

        assert result_prob == 0.3  # Current probability returned
        assert result_factor == 1.0

    @patch("homeassistant.util.dt.utcnow")
    def test_update_decay_probability_increased(self, mock_utcnow: Mock) -> None:
        """Test update_decay when probability increased."""
        mock_utcnow.return_value = dt_util.utcnow()

        decay = Decay(
            is_decaying=True,  # Was decaying
            decay_start_time=dt_util.utcnow() - timedelta(seconds=60),
            decay_start_probability=0.8,
            decay_window=300,
            decay_enabled=True,
            decay_factor=0.9,
        )

        # Probability increased from 0.3 to 0.5
        result_prob, result_factor = decay.update_decay(0.5, 0.3)

        assert result_prob == 0.5  # Current probability returned
        assert result_factor == 1.0
        assert not decay.is_decaying  # Decay stopped

    @patch("homeassistant.util.dt.utcnow")
    def test_update_decay_start_decay(self, mock_utcnow: Mock) -> None:
        """Test update_decay when starting decay."""
        now = dt_util.utcnow()
        mock_utcnow.return_value = now

        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        # Probability decreased from 0.8 to 0.3
        result_prob, result_factor = decay.update_decay(0.3, 0.8)

        assert result_prob == 0.8  # Previous probability maintained for first cycle
        assert result_factor == 1.0
        assert decay.is_decaying
        assert decay.decay_start_time == now
        assert decay.decay_start_probability == 0.8

    @patch("homeassistant.util.dt.utcnow")
    def test_update_decay_continue_decay(self, mock_utcnow: Mock) -> None:
        """Test update_decay when continuing active decay."""
        start_time = dt_util.utcnow() - timedelta(seconds=60)
        now = start_time + timedelta(seconds=60)
        mock_utcnow.return_value = now

        decay = Decay(
            is_decaying=True,
            decay_start_time=start_time,
            decay_start_probability=0.8,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        # Continue decay
        result_prob, result_factor = decay.update_decay(0.3, 0.7)

        # Should apply decay to starting probability
        assert result_prob >= 0.3  # Should be >= current probability (floor)
        assert result_prob < 0.8  # Should be < starting probability
        assert 0 < result_factor < 1  # Decay factor should be between 0 and 1

    def test_invalid_probabilities(self) -> None:
        """Test update_decay with invalid probabilities."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
            decay.update_decay(-0.1, 0.5)

        with pytest.raises(ValueError, match="Probabilities must be between 0 and 1"):
            decay.update_decay(0.5, 1.5)

    def test_should_start_decay(self) -> None:
        """Test should_start_decay logic."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        # Should start decay when transitioning from active to inactive
        assert decay.should_start_decay(True, False)

        # Should not start decay in other cases
        assert not decay.should_start_decay(False, True)
        assert not decay.should_start_decay(True, True)
        assert not decay.should_start_decay(False, False)

        # Should not start if already decaying
        decay.is_decaying = True
        assert not decay.should_start_decay(True, False)

    def test_should_stop_decay(self) -> None:
        """Test should_stop_decay logic."""
        decay = Decay(
            is_decaying=True,
            decay_start_time=dt_util.utcnow(),
            decay_start_probability=0.8,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        # Should stop decay when transitioning from inactive to active
        assert decay.should_stop_decay(False, True)

        # Should not stop decay in other cases
        assert not decay.should_stop_decay(True, False)
        assert not decay.should_stop_decay(True, True)
        assert not decay.should_stop_decay(False, False)

        # Should not stop if not decaying
        decay.is_decaying = False
        assert not decay.should_stop_decay(False, True)

    def test_start_decay(self) -> None:
        """Test start_decay method."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        with patch("homeassistant.util.dt.utcnow") as mock_utcnow:
            now = dt_util.utcnow()
            mock_utcnow.return_value = now

            decay.start_decay(0.8)

            assert decay.is_decaying
            assert decay.decay_start_time == now
            assert decay.decay_start_probability == 0.8
            assert decay.decay_factor == 1.0

    def test_start_decay_when_disabled(self) -> None:
        """Test start_decay when decay is disabled."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=False,  # Disabled
            decay_factor=1.0,
        )

        decay.start_decay(0.8)

        # Should not start decay when disabled
        assert not decay.is_decaying

    def test_stop_decay(self) -> None:
        """Test stop_decay method."""
        decay = Decay(
            is_decaying=True,
            decay_start_time=dt_util.utcnow(),
            decay_start_probability=0.8,
            decay_window=300,
            decay_enabled=True,
            decay_factor=0.9,
        )

        decay.stop_decay()

        assert not decay.is_decaying
        assert decay.decay_start_time is None
        assert decay.decay_start_probability == 0.0  # Reset to MIN_PROBABILITY
        assert decay.decay_factor == 1.0

    def test_is_decay_complete_not_decaying(self) -> None:
        """Test is_decay_complete when not decaying."""
        decay = Decay(
            is_decaying=False,
            decay_start_time=None,
            decay_start_probability=0.5,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        assert decay.is_decay_complete(0.5)

    def test_is_decay_complete_no_start_time(self) -> None:
        """Test is_decay_complete with no start time."""
        decay = Decay(
            is_decaying=True,
            decay_start_time=None,
            decay_start_probability=0.8,
            decay_window=300,
            decay_enabled=True,
            decay_factor=1.0,
        )

        assert decay.is_decay_complete(0.5)

    @patch("homeassistant.util.dt.utcnow")
    def test_is_decay_complete_various_conditions(self, mock_utcnow: Mock) -> None:
        """Test various decay completion conditions."""
        start_time = dt_util.utcnow()
        mock_utcnow.return_value = start_time + timedelta(seconds=100)

        decay = Decay(
            is_decaying=True,
            decay_start_time=start_time,
            decay_start_probability=0.8,
            decay_window=300,
            decay_enabled=True,
            decay_factor=0.5,
        )

        # Test minimum probability threshold
        assert decay.is_decay_complete(0.0)

        # Test completion threshold
        assert decay.is_decay_complete(DECAY_COMPLETION_THRESHOLD / 2)

        # Test decay factor threshold
        decay.decay_factor = DECAY_FACTOR_THRESHOLD / 2
        assert decay.is_decay_complete(0.5)

        # Test maximum duration
        mock_utcnow.return_value = start_time + timedelta(
            seconds=MAX_DECAY_DURATION_MULTIPLIER * 300 + 1
        )
        decay.decay_factor = 0.5  # Reset
        assert decay.is_decay_complete(0.5)

    def test_reset(self) -> None:
        """Test reset method."""
        decay = Decay(
            is_decaying=True,
            decay_start_time=dt_util.utcnow(),
            decay_start_probability=0.8,
            decay_window=600,
            decay_enabled=False,
            decay_factor=0.9,
        )

        decay.reset()

        assert not decay.is_decaying
        assert decay.decay_start_time is None
        assert decay.decay_start_probability == 0.0
        assert decay.decay_window == 300  # Default
        assert decay.decay_enabled  # Default
        assert decay.decay_factor == 1.0
