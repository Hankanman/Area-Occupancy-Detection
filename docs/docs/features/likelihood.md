# Sensor Likelihoods

Each sensor has learned characteristics that describe how reliable it is as evidence of occupancy.

For non-motion sensors, the system learns the statistical distribution of the sensor's state (or value) when the room is occupied vs. unoccupied. This allows for a continuous, dynamic calculation of likelihood based on the exact current state.

See **[Sensor Correlation & Continuous Likelihood](sensor-correlation.md)** for detailed information on how this works.

## Motion Sensors

**Motion sensors do not have learned likelihoods.** Instead, they use **user-configurable likelihoods** that you can set per area during configuration:
- `P(Active | Occupied)`: Configurable (default: 0.95 or 95%)
- `P(Active | Not Occupied)`: Configurable (default: 0.02 or 2%)

Motion sensors are used as **ground truth** to determine when the area is occupied. Learning motion sensor likelihoods would create a circular dependency where motion sensors determine occupied intervals and then calculate their own likelihoods from those same intervals. Instead, you can configure these values based on your specific motion sensor setup and reliability.

You can adjust these values in the integration's configuration for each area. Higher values for `P(Active | Occupied)` mean the motion sensor is more reliable when the area is actually occupied. Lower values for `P(Active | Not Occupied)` mean fewer false positives.

## Other Sensors

For all other sensor types (media, appliances, doors, windows, environmental sensors), likelihoods are learned from historical data by correlating their activity with the occupied intervals determined by motion sensors.

This is done via the **[Unified Correlation Analysis](sensor-correlation.md)** process, which handles both numeric sensors (like temperature) and binary sensors (like media players) using the same statistical engine.

If history based learning is disabled, or insufficient history is available, default likelihoods from the integration are used instead.

You can manually refresh the stored likelihoods by calling the `area_occupancy.run_analysis` service.
