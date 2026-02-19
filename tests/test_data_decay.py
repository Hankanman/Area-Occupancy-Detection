"""Tests for data.decay module."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.purpose import AreaPurpose, Purpose
from homeassistant.util import dt as dt_util

# Get decay values from purpose definitions for use in tests
SLEEPING_PURPOSE = Purpose(AreaPurpose.SLEEPING)
SLEEPING_HALF_LIFE = SLEEPING_PURPOSE.half_life
RELAXING_HALF_LIFE = SLEEPING_PURPOSE.awake_half_life


class TestDecay:
    """Test the Decay class."""

    @pytest.mark.parametrize(
        (
            "kwargs",
            "expected_is_decaying",
            "expected_half_life",
            "expected_purpose",
            "expected_sleep_start",
            "expected_sleep_end",
        ),
        [
            ({"half_life": 30.0}, False, 30.0, None, None, None),
            ({"is_decaying": True, "half_life": 60.0}, True, 60.0, None, None, None),
            (
                {
                    "half_life": 120.0,
                    "purpose": "social",
                    "sleep_start": "23:00:00",
                    "sleep_end": "07:00:00",
                },
                False,
                120.0,
                "social",
                "23:00:00",
                "07:00:00",
            ),
            (
                {
                    "half_life": 90.0,
                    "is_decaying": True,
                    "purpose": "relaxing",
                },
                True,
                90.0,
                "relaxing",
                None,
                None,
            ),
        ],
    )
    def test_initialization(
        self,
        kwargs: dict,
        expected_is_decaying: bool,
        expected_half_life: float,
        expected_purpose: str | None,
        expected_sleep_start: str | None,
        expected_sleep_end: str | None,
    ) -> None:
        """Test decay initialization with all constructor parameters."""
        decay = Decay(**kwargs)
        assert decay.is_decaying == expected_is_decaying
        assert decay.half_life == expected_half_life
        assert isinstance(decay.decay_start, datetime)
        if expected_purpose is not None:
            assert decay.purpose is not None
            assert decay.purpose.purpose.value == expected_purpose
        else:
            assert decay.purpose is None
        assert decay.sleep_start == expected_sleep_start
        assert decay.sleep_end == expected_sleep_end

    def test_initialization_decay_start_none(self) -> None:
        """Test that decay_start defaults to current time when None."""
        before_init = dt_util.utcnow()
        decay = Decay(half_life=60.0)
        after_init = dt_util.utcnow()

        assert decay.decay_start >= before_init
        assert decay.decay_start <= after_init

    @pytest.mark.parametrize(
        (
            "is_decaying",
            "decay_start",
            "half_life",
            "age_seconds",
            "expected_factor",
        ),
        [
            # Not decaying
            (False, None, 60.0, 0, 1.0),
            # Zero age
            (True, dt_util.utcnow(), 60.0, 0, 1.0),
            # Various ages relative to half_life
            (True, dt_util.utcnow(), 60.0, 15.0, 0.8409),  # 0.25x half_life
            (True, dt_util.utcnow(), 60.0, 30.0, 0.7071),  # 0.5x half_life
            (True, dt_util.utcnow(), 60.0, 45.0, 0.5946),  # 0.75x half_life
            (True, dt_util.utcnow(), 60.0, 60.0, 0.5),  # 1.0x half_life
            (True, dt_util.utcnow(), 60.0, 90.0, 0.3536),  # 1.5x half_life
            (True, dt_util.utcnow(), 60.0, 120.0, 0.25),  # 2.0x half_life
            # 5% threshold boundary - just above (returns calculated factor)
            (True, dt_util.utcnow(), 60.0, 258.0, 0.0501),  # ~0.0501, just above 0.05
            # 5% threshold boundary - just below (returns 0.0)
            (True, dt_util.utcnow(), 60.0, 260.0, 0.0),  # ~0.049, below 0.05
            # Very large age (returns 0.0)
            (True, dt_util.utcnow(), 60.0, 1000.0, 0.0),
            # Very small half_life (edge case)
            (True, dt_util.utcnow(), 10.0, 5.0, 0.7071),  # 0.5x half_life
            (True, dt_util.utcnow(), 10.0, 50.0, 0.0),  # Returns 0.0
        ],
    )
    def test_decay_factor(
        self,
        is_decaying: bool,
        decay_start: datetime | None,
        half_life: float,
        age_seconds: float,
        expected_factor: float,
    ) -> None:
        """Test decay factor calculation (pure read-only property, no state mutation)."""
        decay = Decay(
            decay_start=decay_start,
            half_life=half_life,
            is_decaying=is_decaying,
        )
        original_is_decaying = decay.is_decaying

        # Mock datetime.now() to simulate time passing
        with patch("homeassistant.util.dt.utcnow") as mock_utcnow:
            if decay_start:
                mock_utcnow.return_value = decay_start + timedelta(seconds=age_seconds)
            else:
                mock_utcnow.return_value = dt_util.utcnow()

            factor = decay.decay_factor
            assert abs(factor - expected_factor) < 0.01
            # Property should NOT mutate state
            assert decay.is_decaying == original_is_decaying

    @pytest.mark.parametrize(
        (
            "is_decaying",
            "decay_start",
            "half_life",
            "age_seconds",
            "expected_factor",
            "expected_is_decaying_after_tick",
        ),
        [
            # Not decaying - tick returns 1.0 and state remains False
            (False, None, 60.0, 0, 1.0, False),
            # Zero age - tick returns 1.0, state remains True
            (True, dt_util.utcnow(), 60.0, 0, 1.0, True),
            # Above threshold - tick returns factor, state remains True
            (True, dt_util.utcnow(), 60.0, 60.0, 0.5, True),
            # Just above 5% threshold - tick returns factor, state remains True
            (True, dt_util.utcnow(), 60.0, 258.0, 0.0501, True),
            # Below 5% threshold - tick returns 0.0 and stops decay
            (True, dt_util.utcnow(), 60.0, 260.0, 0.0, False),
            # Very large age - tick returns 0.0 and stops decay
            (True, dt_util.utcnow(), 60.0, 1000.0, 0.0, False),
        ],
    )
    def test_tick(
        self,
        is_decaying: bool,
        decay_start: datetime | None,
        half_life: float,
        age_seconds: float,
        expected_factor: float,
        expected_is_decaying_after_tick: bool,
    ) -> None:
        """Test tick() method updates decay state appropriately."""
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

            factor = decay.tick()
            assert abs(factor - expected_factor) < 0.01
            # tick() SHOULD update state when factor reaches zero
            assert decay.is_decaying == expected_is_decaying_after_tick

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
        decay = Decay(half_life=60.0, is_decaying=initial_state)
        original_start = decay.decay_start

        # Call the method
        getattr(decay, method)()

        assert decay.is_decaying == expected_is_decaying

        # Check if decay_start was updated
        if method == "start_decay" and not initial_state:
            assert decay.decay_start > original_start
        else:
            assert decay.decay_start == original_start

    def test_timezone_naive_datetime_handling(self) -> None:
        """Test that timezone-naive datetimes are handled correctly."""
        # Create a timezone-naive datetime
        naive_datetime = datetime(2023, 1, 1, 12, 0, 0)  # No timezone info

        # Create decay with naive datetime
        decay = Decay(half_life=60.0, decay_start=naive_datetime, is_decaying=True)

        # Verify the datetime is now timezone-aware
        assert decay.decay_start.tzinfo is not None
        assert decay.decay_start.tzinfo == dt_util.UTC

        # Test that decay_factor calculation works with the timezone-aware datetime
        with patch("homeassistant.util.dt.utcnow") as mock_utcnow:
            # Mock current time to be 60 seconds after the decay start
            mock_utcnow.return_value = naive_datetime.replace(
                tzinfo=dt_util.UTC
            ) + timedelta(seconds=60)

            # Should calculate decay factor without timezone errors
            factor = decay.decay_factor
            assert 0.0 <= factor <= 1.0

    def test_decay_edge_cases(self) -> None:
        """Test edge cases for decay behavior - idempotency of control methods."""
        # Test start_decay() when already decaying (should not update decay_start)
        decay = Decay(half_life=60.0, is_decaying=True)
        original_start = decay.decay_start
        # Wait a bit
        with patch(
            "homeassistant.util.dt.utcnow",
            return_value=original_start + timedelta(seconds=10),
        ):
            decay.start_decay()
            # decay_start should remain unchanged
            assert decay.decay_start == original_start
            assert decay.is_decaying is True

        # Test stop_decay() when not decaying (should be no-op)
        decay = Decay(half_life=60.0, is_decaying=False)
        original_start = decay.decay_start
        decay.stop_decay()
        assert decay.is_decaying is False
        assert decay.decay_start == original_start

    def test_decay_negative_age(self) -> None:
        """Test decay_factor when decay_start is in the future (negative age)."""
        decay_start = dt_util.utcnow()
        future_time = decay_start + timedelta(seconds=10)

        # Create decay with start time in the past, but mock current time to be before it
        decay = Decay(half_life=60.0, decay_start=decay_start, is_decaying=True)
        with patch("homeassistant.util.dt.utcnow", return_value=future_time):
            # Normal case: future_time is after decay_start, should decay
            factor = decay.decay_factor
            assert 0.0 <= factor <= 1.0

        # Now test negative age: decay_start is in the future relative to current time
        # Use fixed mocked time to ensure deterministic behavior
        mocked_now = datetime(2023, 1, 15, 12, 0, 0, tzinfo=dt_util.UTC)
        future_decay_start = mocked_now + timedelta(seconds=10)
        # Mock current time to be before future_decay_start
        with patch("homeassistant.util.dt.utcnow", return_value=mocked_now):
            decay = Decay(
                half_life=60.0, decay_start=future_decay_start, is_decaying=True
            )
            # Current time is before decay_start, so age is negative
            factor = decay.decay_factor
            # Should return 1.0 (no decay has occurred yet)
            assert factor == 1.0
            assert decay.is_decaying is True  # Should still be decaying

    @pytest.mark.parametrize(
        ("half_life", "is_decaying", "expected_factor"),
        [
            # Invalid half_life when not decaying (returns 1.0)
            (0.0, False, 1.0),
            (-10.0, False, 1.0),
            # Invalid half_life when decaying (returns 0.0 but property doesn't mutate)
            (0.0, True, 0.0),
            (-10.0, True, 0.0),
        ],
    )
    def test_decay_invalid_half_life_property(
        self,
        half_life: float,
        is_decaying: bool,
        expected_factor: float,
    ) -> None:
        """Test decay_factor property with invalid half_life values (pure read-only)."""
        decay_start = dt_util.utcnow() if is_decaying else None
        decay = Decay(
            half_life=half_life, decay_start=decay_start, is_decaying=is_decaying
        )
        original_is_decaying = decay.is_decaying
        factor = decay.decay_factor
        assert factor == expected_factor
        # Property should NOT mutate state
        assert decay.is_decaying == original_is_decaying

    @pytest.mark.parametrize(
        (
            "half_life",
            "is_decaying",
            "expected_factor",
            "expected_is_decaying_after_tick",
        ),
        [
            # Invalid half_life when not decaying (returns 1.0, state stays False)
            (0.0, False, 1.0, False),
            (-10.0, False, 1.0, False),
            # Invalid half_life when decaying (returns 0.0, tick stops decay)
            (0.0, True, 0.0, False),
            (-10.0, True, 0.0, False),
        ],
    )
    def test_decay_invalid_half_life_tick(
        self,
        half_life: float,
        is_decaying: bool,
        expected_factor: float,
        expected_is_decaying_after_tick: bool,
    ) -> None:
        """Test tick() with invalid half_life values (handles state transition)."""
        decay_start = dt_util.utcnow() if is_decaying else None
        decay = Decay(
            half_life=half_life, decay_start=decay_start, is_decaying=is_decaying
        )
        factor = decay.tick()
        assert factor == expected_factor
        # tick() SHOULD update state when factor reaches zero
        assert decay.is_decaying == expected_is_decaying_after_tick

    def test_decay_same_start_end_time(self) -> None:
        """Test SLEEPING purpose with same start and end time."""
        # When start == end, it's treated as same-day window
        # Only matches at the exact time
        # At exact time - should use base_half_life (within window)
        TestDecayHalfLife.check_sleep_window_half_life(
            "12:00:00", "12:00:00", "12:00:00", SLEEPING_HALF_LIFE, SLEEPING_HALF_LIFE
        )

        # One second before - should use RELAXING half_life (outside window)
        TestDecayHalfLife.check_sleep_window_half_life(
            "11:59:59", "12:00:00", "12:00:00", RELAXING_HALF_LIFE, SLEEPING_HALF_LIFE
        )

        # One second after - should use RELAXING half_life (outside window)
        TestDecayHalfLife.check_sleep_window_half_life(
            "12:00:01", "12:00:00", "12:00:00", RELAXING_HALF_LIFE, SLEEPING_HALF_LIFE
        )

    def test_decay_very_large_half_life(self) -> None:
        """Test decay_factor with very large half_life values."""
        decay_start = dt_util.utcnow()

        # Test with very large half_life (should not cause overflow)
        very_large_half_life = 1e10  # 10 billion seconds
        decay = Decay(
            half_life=very_large_half_life, decay_start=decay_start, is_decaying=True
        )

        # After 1 hour, decay should be minimal
        with patch(
            "homeassistant.util.dt.utcnow",
            return_value=decay_start + timedelta(seconds=3600),
        ):
            factor = decay.decay_factor
            # Should be very close to 1.0 (minimal decay)
            assert 0.99 < factor <= 1.0
            assert decay.is_decaying is True


class TestDecayHalfLife:
    """Test the Decay half_life property logic."""

    # Class-level constants that reference module-level constants
    SLEEPING_HALF_LIFE = SLEEPING_HALF_LIFE
    RELAXING_HALF_LIFE = RELAXING_HALF_LIFE

    @staticmethod
    def create_time_datetime(time_str: str) -> datetime:
        """Helper to create a datetime from a time string (HH:MM:SS)."""
        base_date = datetime(2023, 1, 15, 12, 0, 0)
        time_parts = time_str.split(":")
        return base_date.replace(
            hour=int(time_parts[0]),
            minute=int(time_parts[1]),
            second=int(time_parts[2]),
        )

    @staticmethod
    def check_sleep_window_half_life(
        current_time: str,
        sleep_start: str,
        sleep_end: str,
        expected_half_life: float,
        base_half_life: float,
    ) -> None:
        """Helper to test half_life for SLEEPING purpose with sleep window."""
        current_datetime = TestDecayHalfLife.create_time_datetime(current_time)
        decay = Decay(
            half_life=base_half_life,
            purpose=AreaPurpose.SLEEPING.value,
            sleep_start=sleep_start,
            sleep_end=sleep_end,
        )

        with (
            patch("homeassistant.util.dt.utcnow") as mock_utcnow,
            patch("homeassistant.util.dt.as_local") as mock_as_local,
        ):
            mock_utcnow.return_value = current_datetime.replace(tzinfo=dt_util.UTC)
            mock_as_local.return_value = current_datetime.replace(tzinfo=dt_util.UTC)
            assert decay.half_life == expected_half_life

    @pytest.mark.parametrize(
        ("purpose", "base_half_life", "expected_half_life"),
        [
            ("passageway", 45.0, 45.0),
            ("utility", 90.0, 90.0),
            ("bathroom", 450.0, 450.0),
            ("food_prep", 240.0, 240.0),
            ("eating", 480.0, 480.0),
            ("working", 600.0, 600.0),
            ("social", 480.0, 480.0),
            ("relaxing", 600.0, 600.0),
            (None, 100.0, 100.0),  # No purpose
        ],
    )
    def test_non_sleeping_purposes(
        self, purpose: str | None, base_half_life: float, expected_half_life: float
    ) -> None:
        """Test that non-SLEEPING purposes return base_half_life."""
        decay = Decay(half_life=base_half_life, purpose=purpose)
        assert decay.half_life == expected_half_life

    @pytest.mark.parametrize(
        ("sleep_start", "sleep_end"),
        [
            (None, None),
            (None, "07:00:00"),
            ("23:00:00", None),
        ],
    )
    def test_sleeping_without_sleep_config(
        self, sleep_start: str | None, sleep_end: str | None
    ) -> None:
        """Test SLEEPING purpose without complete sleep config returns base_half_life."""
        decay = Decay(
            half_life=self.SLEEPING_HALF_LIFE,
            purpose=AreaPurpose.SLEEPING.value,
            sleep_start=sleep_start,
            sleep_end=sleep_end,
        )
        assert decay.half_life == self.SLEEPING_HALF_LIFE

    @pytest.mark.parametrize(
        ("current_time", "sleep_start", "sleep_end", "expected_half_life"),
        [
            # Same-day window: 13:00-15:00
            ("13:00:00", "13:00:00", "15:00:00", SLEEPING_HALF_LIFE),  # At start
            ("14:00:00", "13:00:00", "15:00:00", SLEEPING_HALF_LIFE),  # Middle
            ("15:00:00", "13:00:00", "15:00:00", SLEEPING_HALF_LIFE),  # At end
            # Same-day window: outside
            ("12:59:59", "13:00:00", "15:00:00", RELAXING_HALF_LIFE),  # Before
            ("15:00:01", "13:00:00", "15:00:00", RELAXING_HALF_LIFE),  # After
        ],
    )
    def test_sleeping_same_day_window(
        self,
        current_time: str,
        sleep_start: str,
        sleep_end: str,
        expected_half_life: float,
    ) -> None:
        """Test SLEEPING purpose with same-day sleep window."""
        self.check_sleep_window_half_life(
            current_time, sleep_start, sleep_end, expected_half_life, SLEEPING_HALF_LIFE
        )

    @pytest.mark.parametrize(
        ("current_time", "sleep_start", "sleep_end", "expected_half_life"),
        [
            # Overnight window: 23:00-07:00
            ("23:00:00", "23:00:00", "07:00:00", SLEEPING_HALF_LIFE),  # At start
            ("01:00:00", "23:00:00", "07:00:00", SLEEPING_HALF_LIFE),  # Middle of night
            ("07:00:00", "23:00:00", "07:00:00", SLEEPING_HALF_LIFE),  # At end
            # Overnight window: outside
            ("22:59:59", "23:00:00", "07:00:00", RELAXING_HALF_LIFE),  # Before start
            ("07:00:01", "23:00:00", "07:00:00", RELAXING_HALF_LIFE),  # After end
            ("12:00:00", "23:00:00", "07:00:00", RELAXING_HALF_LIFE),  # Middle of day
        ],
    )
    def test_sleeping_overnight_window(
        self,
        current_time: str,
        sleep_start: str,
        sleep_end: str,
        expected_half_life: float,
    ) -> None:
        """Test SLEEPING purpose with overnight sleep window."""
        self.check_sleep_window_half_life(
            current_time, sleep_start, sleep_end, expected_half_life, SLEEPING_HALF_LIFE
        )

    def test_sleeping_error_handling_invalid_format(self) -> None:
        """Test SLEEPING purpose with invalid sleep time format falls back to base_half_life."""
        decay = Decay(
            half_life=self.SLEEPING_HALF_LIFE,
            purpose=AreaPurpose.SLEEPING.value,
            sleep_start="invalid",
            sleep_end="07:00:00",
        )

        # Should fall back to base_half_life due to ValueError in strptime
        assert decay.half_life == self.SLEEPING_HALF_LIFE

    def test_sleeping_error_handling_type_error(self) -> None:
        """Test SLEEPING purpose error handling for TypeError."""
        decay = Decay(
            half_life=self.SLEEPING_HALF_LIFE,
            purpose=AreaPurpose.SLEEPING.value,
            sleep_start="23:00:00",
            sleep_end="07:00:00",
        )

        # Mock as_local to raise TypeError
        with patch(
            "homeassistant.util.dt.as_local", side_effect=TypeError("Test error")
        ):
            # Should fall back to base_half_life
            assert decay.half_life == self.SLEEPING_HALF_LIFE

    def test_sleeping_relaxing_fallback(self) -> None:
        """Test SLEEPING purpose outside sleep window uses awake_half_life."""
        # Test with datetime outside sleep window (12:00:00)
        TestDecayHalfLife.check_sleep_window_half_life(
            "12:00:00",
            "23:00:00",
            "07:00:00",
            self.RELAXING_HALF_LIFE,
            self.SLEEPING_HALF_LIFE,
        )

        # Verify it's using the sleeping purpose's awake_half_life
        assert SLEEPING_PURPOSE.awake_half_life == self.RELAXING_HALF_LIFE

    def test_sleeping_fallback_when_no_awake_half_life(self) -> None:
        """Test purpose with no awake_half_life falls back to base_half_life."""
        # Create a decay with a purpose that has no awake_half_life (e.g. social)
        decay = Decay(
            half_life=520.0,
            purpose=AreaPurpose.SOCIAL.value,
            sleep_start="23:00:00",
            sleep_end="07:00:00",
        )

        # Social purpose has no awake_half_life, so should always use base
        assert decay.purpose.awake_half_life is None
        assert decay.half_life == 520.0
