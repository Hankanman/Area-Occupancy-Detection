"""Tests for the decay module."""

from unittest.mock import patch

from custom_components.area_occupancy.data.decay import DEFAULT_HALF_LIFE, Decay


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
