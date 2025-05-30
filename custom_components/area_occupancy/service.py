"""Service definitions for the Area Occupancy Detection integration."""

import logging

import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from .const import CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD, DOMAIN
from .exceptions import CalculationError

_LOGGER = logging.getLogger(__name__)


async def async_setup_services(hass: HomeAssistant):
    """Register custom services for area occupancy."""

    async def update_priors(call):
        """Manually trigger an update of learned priors."""
        entry_id = call.data["entry_id"]

        def raise_error(msg: str, err: Exception | None = None) -> None:
            """Raise appropriate error with consistent format."""
            error_msg = f"{msg}: {err}" if err else msg
            _LOGGER.error(
                "Failed to update priors for instance %s: %s", entry_id, error_msg
            )
            raise HomeAssistantError(error_msg)

        try:
            _LOGGER.debug("Updating priors for entry_id: %s", entry_id)

            if DOMAIN not in hass.data or entry_id not in hass.data[DOMAIN]:
                raise_error(f"Integration or instance {entry_id} not found")

            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]
            _LOGGER.debug("Found coordinator for entry_id %s", entry_id)

            # Get history period from service call, fallback to coordinator config, then to default
            history_period = call.data.get("history_period")
            if history_period is None:
                history_period = coordinator.config.get(
                    CONF_HISTORY_PERIOD, DEFAULT_HISTORY_PERIOD
                )

            _LOGGER.debug(
                "Using history period: %s (type: %s) for instance %s",
                history_period,
                type(history_period),
                entry_id,
            )

            # Always call update_learned_priors, passing the determined period
            _LOGGER.debug(
                "Calling update_learned_priors with period: %s for instance %s",
                history_period,
                entry_id,
            )
            try:
                await coordinator.update_learned_priors(history_period)
            except (CalculationError, HomeAssistantError) as err:
                _LOGGER.error(
                    "Error in update_learned_priors for instance %s: %s",
                    entry_id,
                    err,
                )
                raise_error("Failed to update learned priors", err)

            # Trigger a coordinator refresh
            _LOGGER.debug("Triggering coordinator refresh for instance %s", entry_id)
            await coordinator.async_refresh()
            _LOGGER.debug(
                "Prior update completed successfully for instance %s", entry_id
            )

        except KeyError as err:
            raise_error("Invalid entry_id or coordinator not found", err)
        except (HomeAssistantError, ValueError, RuntimeError) as err:
            raise_error("Failed to update priors", err)

    service_schema_update_priors = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional(
                "history_period",
            ): vol.All(int, vol.Range(min=1, max=90)),
        }
    )

    async def train_model(call):
        """Manually trigger ML model training."""
        entry_id = call.data["entry_id"]

        def raise_error(msg: str, err: Exception | None = None) -> None:
            """Raise appropriate error with consistent format."""
            error_msg = f"{msg}: {err}" if err else msg
            _LOGGER.error(
                "Failed to train ML model for instance %s: %s", entry_id, error_msg
            )
            raise HomeAssistantError(error_msg)

        try:
            _LOGGER.debug("Training ML model for entry_id: %s", entry_id)

            if DOMAIN not in hass.data or entry_id not in hass.data[DOMAIN]:
                raise_error(f"Integration or instance {entry_id} not found")

            coordinator = hass.data[DOMAIN][entry_id]["coordinator"]
            _LOGGER.debug("Found coordinator for entry_id %s", entry_id)

            # Check if ML is enabled and available
            if not coordinator.model_manager:
                raise_error("ML model manager not available")

            if not coordinator.model_manager.model_available:
                raise_error("ML dependencies not available")

            # Get history period from service call or use default
            history_period = call.data.get("history_period", 30)  # 30 days default

            _LOGGER.debug(
                "Using history period: %s days for ML training in instance %s",
                history_period,
                entry_id,
            )

            # Import here to avoid circular imports
            from datetime import timedelta

            from .ml_data_collector import TrainingSetBuilder

            # Get configured sensors from coordinator
            configured_sensors = coordinator.get_configured_sensors()
            if not configured_sensors:
                raise_error("No sensors configured for training")

            # Build sensor map using coordinator's probabilities and inputs
            sensor_map = {}

            # Map sensors to their entity types using coordinator's probabilities
            for sensor_id in configured_sensors:
                entity_type = coordinator.probabilities.entity_types.get(sensor_id)
                if entity_type:
                    sensor_map[sensor_id] = entity_type
                else:
                    _LOGGER.warning("No entity type found for sensor %s", sensor_id)

            if not sensor_map:
                raise_error("No valid sensors found for training")

            # Calculate date range
            from homeassistant.util import dt as dt_util

            end_time = dt_util.utcnow()
            start_time = end_time - timedelta(days=history_period)

            _LOGGER.debug(
                "Building training dataset from %s to %s for instance %s",
                start_time,
                end_time,
                entry_id,
            )

            # Build training dataset
            builder = TrainingSetBuilder(hass)
            primary_sensor = coordinator.inputs.primary_sensor

            df = await builder.snapshot(
                sensor_map=sensor_map,
                start=start_time,
                end=end_time,
                primary_sensor=primary_sensor,
            )

            if df is None or df.empty:
                raise_error("No training data available for the specified period")

            # Convert to ML training data format
            # At this point df is guaranteed to be non-None and non-empty
            assert df is not None  # Type narrowing for mypy

            # Debug: Log the DataFrame structure before ML preparation
            _LOGGER.debug(
                "DataFrame structure before ML preparation for instance %s: Shape=%s, Columns=%s",
                entry_id,
                df.shape,
                list(df.columns),
            )
            _LOGGER.debug(
                "DataFrame sample for instance %s:\n%s",
                entry_id,
                df.head().to_string() if not df.empty else "Empty DataFrame",
            )

            training_data = builder.prepare_ml_data(df)
            if training_data is None:
                raise_error("Failed to prepare training data")

            # At this point training_data is guaranteed to be non-None
            assert training_data is not None  # Type narrowing for mypy
            _LOGGER.debug(
                "Prepared training data with %d samples for instance %s",
                len(training_data.labels),
                entry_id,
            )

            # Debug: Log training data structure
            _LOGGER.debug(
                "Training data structure for instance %s: Features=%d, Labels=%d, Timestamps=%d",
                entry_id,
                len(training_data.features),
                len(training_data.labels),
                len(training_data.timestamps),
            )
            if training_data.features:
                sample_timestamp = next(iter(training_data.features.keys()))
                sample_features = training_data.features[sample_timestamp]
                _LOGGER.debug(
                    "Sample features for instance %s (timestamp=%s): %s",
                    entry_id,
                    sample_timestamp,
                    sample_features,
                )

            # Train the model
            model_meta = await coordinator.model_manager.async_train(training_data)

            if model_meta is None:
                raise_error("Model training failed")

            _LOGGER.info(
                "Successfully trained ML model for instance %s - Accuracy: %.3f",
                entry_id,
                model_meta.performance_metrics.get("accuracy", 0.0),
            )

        except KeyError as err:
            raise_error("Invalid entry_id or coordinator not found", err)
        except (HomeAssistantError, ValueError, RuntimeError) as err:
            raise_error("Failed to train ML model", err)

    service_schema_train_model = vol.Schema(
        {
            vol.Required("entry_id"): str,
            vol.Optional(
                "history_period",
                default=30,
            ): vol.All(int, vol.Range(min=7, max=90)),
        }
    )

    hass.services.async_register(
        DOMAIN,
        "update_priors",
        update_priors,
        schema=service_schema_update_priors,
    )

    hass.services.async_register(
        DOMAIN,
        "train_model",
        train_model,
        schema=service_schema_train_model,
    )
