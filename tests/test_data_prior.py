"""Tests for data.prior module - Area baseline prior calculations."""

from datetime import timedelta
from unittest.mock import AsyncMock, Mock, patch

from custom_components.area_occupancy.data.prior import Prior
from homeassistant.const import STATE_OFF, STATE_ON
from homeassistant.core import State
from homeassistant.util import dt as dt_util


# ruff: noqa: SLF001, PLC0415
class TestPrior:
    """Test Prior class for area baseline prior calculations."""

    def test_initialization(self, mock_coordinator: Mock) -> None:
        """Test Prior initialization."""
        prior = Prior(mock_coordinator)

        assert prior.coordinator == mock_coordinator
        assert prior.config == mock_coordinator.config
        assert prior._area_baseline_prior is None
        assert prior._method_used is None

    def test_area_baseline_prior_property_no_cache(
        self, mock_coordinator: Mock
    ) -> None:
        """Test area_baseline_prior property returns default when no cache."""
        from custom_components.area_occupancy.const import DEFAULT_PRIOR

        prior = Prior(mock_coordinator)
        result = prior.area_baseline_prior

        assert result == DEFAULT_PRIOR

    def test_area_baseline_prior_property_with_cache(
        self, mock_coordinator: Mock
    ) -> None:
        """Test area_baseline_prior property returns cached value."""
        prior = Prior(mock_coordinator)
        prior._area_baseline_prior = 0.25

        result = prior.area_baseline_prior

        assert result == 0.25

    def test_method_used_property(self, mock_coordinator: Mock) -> None:
        """Test method_used property."""
        prior = Prior(mock_coordinator)
        assert prior.method_used is None

        prior._method_used = "multi_motion_consensus"
        assert prior.method_used == "multi_motion_consensus"

    def test_clear_cache(self, mock_coordinator: Mock) -> None:
        """Test cache clearing."""
        prior = Prior(mock_coordinator)
        prior._area_baseline_prior = 0.25
        prior._method_used = "test_method"

        prior.clear_cache()

        assert prior._area_baseline_prior is None
        assert prior._method_used is None

    async def test_post_init_history_enabled(self, mock_coordinator: Mock) -> None:
        """Test __post_init__ when history is enabled."""
        mock_coordinator.config.history.enabled = True
        prior = Prior(mock_coordinator)

        with patch.object(
            prior, "calculate_area_baseline_prior", new_callable=AsyncMock
        ) as mock_calc:
            await prior.__post_init__()
            mock_calc.assert_called_once()

    async def test_post_init_history_disabled(self, mock_coordinator: Mock) -> None:
        """Test __post_init__ when history is disabled."""
        mock_coordinator.config.history.enabled = False
        prior = Prior(mock_coordinator)

        with patch.object(
            prior, "calculate_area_baseline_prior", new_callable=AsyncMock
        ) as mock_calc:
            await prior.__post_init__()
            mock_calc.assert_not_called()

    async def test_default_fallback_method(self, mock_coordinator: Mock) -> None:
        """Test default fallback when no viable method found."""
        from custom_components.area_occupancy.const import DEFAULT_PRIOR

        mock_coordinator.config.history.enabled = True
        mock_coordinator.config.history.period = 7
        mock_coordinator.config.sensors.motion = []
        mock_coordinator.config.sensors.media = []
        mock_coordinator.config.sensors.appliances = []
        mock_coordinator.config.sensors.doors = []
        mock_coordinator.config.sensors.windows = []
        mock_coordinator.config.sensors.lights = []
        mock_coordinator.config.sensors.illuminance = []
        mock_coordinator.config.sensors.humidity = []
        mock_coordinator.config.sensors.temperature = []
        mock_coordinator.config.sensors.primary_occupancy = None
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)

        result = await prior.calculate_area_baseline_prior()

        assert result == DEFAULT_PRIOR
        assert prior._method_used == "default_fallback"


class TestAreaBaselinePriorCalculation:
    """Test area baseline prior calculation methods."""

    async def test_calculate_history_disabled(self, mock_coordinator: Mock) -> None:
        """Test calculation when history is disabled."""
        from custom_components.area_occupancy.const import DEFAULT_PRIOR

        mock_coordinator.config.history.enabled = False
        prior = Prior(mock_coordinator)

        result = await prior.calculate_area_baseline_prior()

        assert result == DEFAULT_PRIOR
        assert prior._method_used == "disabled"

    async def test_calculate_with_cached_result(self, mock_coordinator: Mock) -> None:
        """Test calculation returns cached result when valid."""
        mock_coordinator.config.history.enabled = True
        mock_coordinator.config.history.period = 7
        prior = Prior(mock_coordinator)

        # Set up cached result
        now = dt_util.utcnow()
        prior._area_baseline_prior = 0.35
        prior._area_prior_last_updated = now - timedelta(hours=1)
        prior._method_used = "cached"

        result = await prior.calculate_area_baseline_prior()

        assert result == 0.35

    async def test_calculate_with_exception_fallback(
        self, mock_coordinator: Mock
    ) -> None:
        """Test calculation falls back to default when exception occurs."""
        from custom_components.area_occupancy.const import DEFAULT_PRIOR

        mock_coordinator.config.history.enabled = True
        mock_coordinator.config.sensors.motion = ["sensor.motion1", "sensor.motion2"]

        prior = Prior(mock_coordinator)

        with patch.object(
            prior, "_get_all_motion_sensors", side_effect=Exception("Test error")
        ):
            result = await prior.calculate_area_baseline_prior()

            assert result == DEFAULT_PRIOR
            assert prior._method_used == "error_fallback"


class TestHelperMethods:
    """Test helper methods."""

    def test_get_all_motion_sensors_basic(self, mock_coordinator: Mock) -> None:
        """Test _get_all_motion_sensors with basic configuration."""
        mock_coordinator.config.sensors.motion = ["sensor.motion1", "sensor.motion2"]
        mock_coordinator.config.sensors.primary_occupancy = None
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)
        result = prior._get_all_motion_sensors()

        assert result == ["sensor.motion1", "sensor.motion2"]

    def test_get_all_motion_sensors_with_primary(self, mock_coordinator: Mock) -> None:
        """Test _get_all_motion_sensors includes primary sensor."""
        mock_coordinator.config.sensors.motion = ["sensor.motion1"]
        mock_coordinator.config.sensors.primary_occupancy = "binary_sensor.primary"
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)
        result = prior._get_all_motion_sensors()

        assert "sensor.motion1" in result
        assert "binary_sensor.primary" in result
        assert len(result) == 2

    def test_get_all_motion_sensors_with_wasp(self, mock_coordinator: Mock) -> None:
        """Test _get_all_motion_sensors includes wasp sensor."""
        mock_coordinator.config.sensors.motion = ["sensor.motion1"]
        mock_coordinator.config.sensors.primary_occupancy = None
        mock_coordinator.config.wasp_in_box.enabled = True
        mock_coordinator.wasp_entity_id = "binary_sensor.wasp"

        prior = Prior(mock_coordinator)
        result = prior._get_all_motion_sensors()

        assert "sensor.motion1" in result
        assert "binary_sensor.wasp" in result
        assert len(result) == 2

    def test_get_all_motion_sensors_no_duplicates(self, mock_coordinator: Mock) -> None:
        """Test _get_all_motion_sensors avoids duplicates."""
        mock_coordinator.config.sensors.motion = [
            "sensor.motion1",
            "binary_sensor.primary",
        ]
        mock_coordinator.config.sensors.primary_occupancy = "binary_sensor.primary"
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)
        result = prior._get_all_motion_sensors()

        assert result.count("binary_sensor.primary") == 1
        assert len(result) == 2

    def test_has_multiple_sensor_types(self, mock_coordinator: Mock) -> None:
        """Test _has_multiple_sensor_types detection."""
        prior = Prior(mock_coordinator)

        # Single type
        assert not prior._has_multiple_sensor_types(["motion1"], [], [])
        assert not prior._has_multiple_sensor_types([], ["media1"], [])
        assert not prior._has_multiple_sensor_types([], [], ["appliance1"])

        # Multiple types
        assert prior._has_multiple_sensor_types(["motion1"], ["media1"], [])
        assert prior._has_multiple_sensor_types(["motion1"], [], ["appliance1"])
        assert prior._has_multiple_sensor_types([], ["media1"], ["appliance1"])
        assert prior._has_multiple_sensor_types(["motion1"], ["media1"], ["appliance1"])

        # No sensors
        assert not prior._has_multiple_sensor_types([], [], [])


class TestMethodFallbackLogic:
    """Test the fallback method selection logic."""

    async def test_method_selection_multi_motion_consensus(
        self, mock_coordinator: Mock
    ) -> None:
        """Test multi-motion consensus method is selected with ≥2 motion sensors."""
        mock_coordinator.config.history.enabled = True
        mock_coordinator.config.sensors.motion = ["sensor.motion1", "sensor.motion2"]
        mock_coordinator.config.sensors.media = []
        mock_coordinator.config.sensors.appliances = []
        mock_coordinator.config.sensors.doors = []
        mock_coordinator.config.sensors.windows = []
        mock_coordinator.config.sensors.lights = []
        mock_coordinator.config.sensors.illuminance = []
        mock_coordinator.config.sensors.humidity = []
        mock_coordinator.config.sensors.temperature = []
        mock_coordinator.config.sensors.primary_occupancy = None
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)

        with patch.object(
            prior, "_multi_motion_consensus", return_value=0.35
        ) as mock_method:
            result = await prior.calculate_area_baseline_prior()

            assert result == 0.35
            assert prior._method_used == "multi_motion_consensus"
            mock_method.assert_called_once()

    async def test_method_selection_confidence_weighted_fusion(
        self, mock_coordinator: Mock
    ) -> None:
        """Test confidence-weighted fusion method is selected with multiple sensor types."""
        mock_coordinator.config.history.enabled = True
        mock_coordinator.config.sensors.motion = [
            "sensor.motion1"
        ]  # Only 1 motion sensor
        mock_coordinator.config.sensors.media = ["media_player.tv"]
        mock_coordinator.config.sensors.appliances = []
        mock_coordinator.config.sensors.doors = []
        mock_coordinator.config.sensors.windows = []
        mock_coordinator.config.sensors.lights = []
        mock_coordinator.config.sensors.illuminance = []
        mock_coordinator.config.sensors.humidity = []
        mock_coordinator.config.sensors.temperature = []
        mock_coordinator.config.sensors.primary_occupancy = None
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)

        with patch.object(
            prior, "_confidence_weighted_fusion", return_value=0.40
        ) as mock_method:
            result = await prior.calculate_area_baseline_prior()

            assert result == 0.40
            assert prior._method_used == "confidence_weighted_fusion"
            mock_method.assert_called_once()

    async def test_method_selection_time_pattern_analysis(
        self, mock_coordinator: Mock
    ) -> None:
        """Test time pattern analysis method is selected with ≥2 total sensors."""
        mock_coordinator.config.history.enabled = True
        mock_coordinator.config.sensors.motion = ["sensor.motion1"]
        mock_coordinator.config.sensors.media = []
        mock_coordinator.config.sensors.appliances = []
        mock_coordinator.config.sensors.doors = [
            "binary_sensor.door"
        ]  # Total 2 sensors
        mock_coordinator.config.sensors.windows = []
        mock_coordinator.config.sensors.lights = []
        mock_coordinator.config.sensors.illuminance = []
        mock_coordinator.config.sensors.humidity = []
        mock_coordinator.config.sensors.temperature = []
        mock_coordinator.config.sensors.primary_occupancy = None
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)

        with patch.object(
            prior, "_time_pattern_analysis", return_value=0.30
        ) as mock_method:
            result = await prior.calculate_area_baseline_prior()

            assert result == 0.30
            assert prior._method_used == "time_pattern_analysis"
            mock_method.assert_called_once()

    async def test_method_selection_primary_with_margin(
        self, mock_coordinator: Mock
    ) -> None:
        """Test primary sensor with margin method is selected with single sensor."""
        mock_coordinator.config.history.enabled = True
        mock_coordinator.config.sensors.motion = []
        mock_coordinator.config.sensors.media = []
        mock_coordinator.config.sensors.appliances = []
        mock_coordinator.config.sensors.doors = []
        mock_coordinator.config.sensors.windows = []
        mock_coordinator.config.sensors.lights = []
        mock_coordinator.config.sensors.illuminance = []
        mock_coordinator.config.sensors.humidity = []
        mock_coordinator.config.sensors.temperature = []
        mock_coordinator.config.sensors.primary_occupancy = "binary_sensor.primary"
        mock_coordinator.config.wasp_in_box.enabled = False

        prior = Prior(mock_coordinator)

        with patch.object(
            prior, "_primary_sensor_with_margin", return_value=0.28
        ) as mock_method:
            result = await prior.calculate_area_baseline_prior()

            assert result == 0.28
            assert prior._method_used == "primary_with_margin"
            mock_method.assert_called_once()


class TestIndividualMethods:
    """Test individual calculation methods."""

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    @patch("custom_components.area_occupancy.data.prior.states_to_intervals")
    async def test_multi_motion_consensus_success(
        self,
        mock_intervals: AsyncMock,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful multi-motion consensus calculation."""
        now = dt_util.utcnow()
        start_time = now - timedelta(hours=2)
        end_time = now

        # Mock states
        mock_states = [
            State("sensor.motion1", STATE_ON, last_changed=now - timedelta(hours=2)),
            State("sensor.motion1", STATE_OFF, last_changed=now - timedelta(hours=1)),
        ]
        mock_get_states.return_value = mock_states

        # Mock intervals
        mock_intervals.return_value = [
            {
                "start": start_time,
                "end": start_time + timedelta(hours=1),
                "state": STATE_ON,
            },
            {
                "start": start_time + timedelta(hours=1),
                "end": end_time,
                "state": STATE_OFF,
            },
        ]

        prior = Prior(mock_coordinator)
        result = await prior._multi_motion_consensus(
            ["sensor.motion1", "sensor.motion2"], start_time, end_time
        )

        assert isinstance(result, float)
        assert 0 <= result <= 1

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    async def test_multi_motion_consensus_insufficient_data(
        self, mock_get_states: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test multi-motion consensus with insufficient sensor data."""
        mock_get_states.return_value = []  # No states

        prior = Prior(mock_coordinator)
        result = await prior._multi_motion_consensus(
            ["sensor.motion1", "sensor.motion2"],
            dt_util.utcnow() - timedelta(hours=2),
            dt_util.utcnow(),
        )

        assert result is None

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    @patch("custom_components.area_occupancy.data.prior.states_to_intervals")
    async def test_primary_sensor_with_margin_success(
        self,
        mock_intervals: AsyncMock,
        mock_get_states: AsyncMock,
        mock_coordinator: Mock,
    ) -> None:
        """Test successful primary sensor with margin calculation."""
        now = dt_util.utcnow()
        start_time = now - timedelta(hours=2)
        end_time = now

        # Mock states
        mock_states = [
            State(
                "binary_sensor.primary", STATE_ON, last_changed=now - timedelta(hours=2)
            ),
            State(
                "binary_sensor.primary",
                STATE_OFF,
                last_changed=now - timedelta(hours=1),
            ),
        ]
        mock_get_states.return_value = mock_states

        # Mock intervals - 50% occupancy
        mock_intervals.return_value = [
            {
                "start": start_time,
                "end": start_time + timedelta(hours=1),
                "state": STATE_ON,
            },
            {
                "start": start_time + timedelta(hours=1),
                "end": end_time,
                "state": STATE_OFF,
            },
        ]

        prior = Prior(mock_coordinator)
        result = await prior._primary_sensor_with_margin(
            "binary_sensor.primary", start_time, end_time
        )

        assert isinstance(result, float)
        assert 0 <= result <= 1
        # Should be adjusted from raw 0.5 toward 0.5 with 5% uncertainty
        # 0.5 * 0.95 + 0.5 * 0.05 = 0.5 (no change when raw rate is already 0.5)
        assert result == 0.5

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    async def test_primary_sensor_with_margin_no_data(
        self, mock_get_states: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test primary sensor with margin when no data available."""
        mock_get_states.return_value = []  # No states

        prior = Prior(mock_coordinator)
        result = await prior._primary_sensor_with_margin(
            "binary_sensor.primary",
            dt_util.utcnow() - timedelta(hours=2),
            dt_util.utcnow(),
        )

        assert result is None

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    async def test_time_pattern_analysis_success(
        self, mock_get_states: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test successful time pattern analysis calculation."""
        now = dt_util.utcnow()
        start_time = now - timedelta(hours=2)
        end_time = now

        # Mock states
        mock_states = [
            State("sensor.motion1", STATE_ON, last_changed=now - timedelta(hours=2)),
            State("sensor.motion1", STATE_OFF, last_changed=now - timedelta(hours=1)),
        ]
        mock_get_states.return_value = mock_states

        with patch(
            "custom_components.area_occupancy.data.prior.states_to_intervals"
        ) as mock_intervals:
            mock_intervals.return_value = [
                {
                    "start": start_time,
                    "end": start_time + timedelta(hours=1),
                    "state": STATE_ON,
                },
                {
                    "start": start_time + timedelta(hours=1),
                    "end": end_time,
                    "state": STATE_OFF,
                },
            ]

            prior = Prior(mock_coordinator)
            result = await prior._time_pattern_analysis(
                ["sensor.motion1", "binary_sensor.door"], start_time, end_time
            )

            assert isinstance(result, float)
            assert 0 <= result <= 1

    @patch("custom_components.area_occupancy.data.prior.get_states_from_recorder")
    async def test_time_pattern_analysis_no_data(
        self, mock_get_states: AsyncMock, mock_coordinator: Mock
    ) -> None:
        """Test time pattern analysis with no sensor data."""
        mock_get_states.return_value = []  # No states

        prior = Prior(mock_coordinator)
        result = await prior._time_pattern_analysis(
            ["sensor.motion1", "binary_sensor.door"],
            dt_util.utcnow() - timedelta(hours=2),
            dt_util.utcnow(),
        )

        assert result is None
