# Sensor Likelihoods

Each sensor has learned characteristics that describe how reliable it is as evidence of occupancy.

For non-motion sensors, the system learns the statistical distribution of the sensor's state (or value) when the room is occupied vs. unoccupied. This allows for a continuous, dynamic calculation of likelihood based on the exact current state.

See **[Sensor Correlation & Continuous Likelihood](sensor-correlation.md)** for detailed information on how this works.

## Motion Sensors

**Motion sensors do not have learned likelihoods.** Instead, they use **user-configurable likelihoods** that you can set per area during configuration:

- `P(Active | Occupied)`: Configurable (default: 0.95 or 95%)
- `P(Active | Not Occupied)`: Configurable (default: 0.005 or 0.5%)

Motion sensors are used as **ground truth** to determine when the area is occupied. Learning motion sensor likelihoods would create a circular dependency where motion sensors determine occupied intervals and then calculate their own likelihoods from those same intervals. Instead, you can configure these values based on your specific motion sensor setup and reliability.

You can adjust these values in the integration's configuration for each area. Higher values for `P(Active | Occupied)` mean the motion sensor is more reliable when the area is actually occupied. Lower values for `P(Active | Not Occupied)` mean fewer false positives.

## Other Sensors

For all other sensor types, likelihoods are learned from historical data by analyzing their activity relative to the occupied intervals determined by motion sensors. The analysis method depends on sensor type:

### Numeric Sensors (Temperature, Humidity, CO2, etc.)

Numeric sensors use **correlation analysis** to learn statistical distributions (Gaussian PDFs) that allow for dynamic likelihood calculation based on the exact sensor value.

See **[Sensor Correlation & Continuous Likelihood](sensor-correlation.md)** for detailed information.

### Binary Sensors (Media Players, Appliances, Doors, Windows)

Binary sensors use **duration-based analysis** to calculate static probabilities directly from how long they're active during occupied vs. unoccupied periods.

The system calculates:

- `P(Active | Occupied)`: Probability the sensor is active when the area is occupied
- `P(Active | Unoccupied)`: Probability the sensor is active when the area is unoccupied

These probabilities are stored as static values and used regardless of the current sensor state.

### Default Likelihoods

If history-based learning is disabled, insufficient history is available, or analysis fails, default likelihoods from the integration are used instead. These defaults come from the `EntityType` definition and vary by sensor type.

You can manually refresh the stored likelihoods by calling the `area_occupancy.run_analysis` service.
