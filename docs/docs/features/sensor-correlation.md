# Sensor Correlation & Continuous Likelihood

The Area Occupancy Detection integration uses advanced capability to learn occupancy patterns from all types of sensors (Temperature, Humidity, CO2, Media Players, etc.). Instead of a simple static probability for "Active/Inactive" states, the system calculates a continuous probability of occupancy based on the exact value of the sensor (or binary state) using **Gaussian Probability Density Functions (PDFs)**.

## Overview

While motion sensors directly indicate presence, other sensors often show correlation with occupancy. For example:

- **Numeric Sensors**:
  - **Temperature** might rise when people are in a room.
  - **CO2** levels often increase with occupancy.
  - **Humidity** might change when a shower is used.
- **Binary Sensors**:
  - **Media Players** might be playing (state "on" = 1.0) when occupied.
  - **Appliances** might be running (state "on" = 1.0) when occupied.
  - **Doors/Windows** might be open (state "on" = 1.0) more often when occupied.

Previously, binary sensors used a simple duration-weighted counting method to determine static probabilities. The new system unifies this by treating all sensors (except motion sensors) through the same correlation analysis engine.

## How It Works

The system performs a two-stage statistical analysis for all eligible sensors:

### 1. Correlation Check (Qualification)

Every hour as part of the analysis cycle, the system analyzes the relationship between the sensor's value and the area's occupancy state using the **Pearson correlation coefficient**.

- **Numeric Sensors**: Uses actual sensor values.
- **Binary Sensors**: Converts states to numeric values (0.0 for OFF, 1.0 for ON).

- **Positive Correlation**: Value increases when occupied.
- **Negative Correlation**: Value decreases when occupied.
- **No Correlation**: No clear pattern found.

If the correlation is too weak or the sample size is too small, the sensor is **rejected** for occupancy detection purposes to prevent false positives.

### 2. Learning Distributions

If a sensor qualifies, the system learns two statistical distributions:

1. **Occupied Distribution**: $(\mu_{occ}, \sigma_{occ})$ - The pattern when the room is _known_ to be occupied.
2. **Unoccupied Distribution**: $(\mu_{unocc}, \sigma_{unocc})$ - The pattern when the room is _known_ to be empty.

These parameters are stored in the `NumericCorrelations` database table.

### 3. Calculating Dynamic Likelihood

When the sensor reports a new value $x$, the system calculates two probability densities using the Gaussian PDF formula:

$$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} $$

1. **$P(x | Occupied)$**: Likelihood of $x$ given the "Occupied" distribution.
2. **$P(x | Unoccupied)$**: Likelihood of $x$ given the "Unoccupied" distribution.

**For Binary Sensors:**
- **State ON**: $x = 1.0$. Likelihood is calculated based on how close 1.0 is to the learned "Occupied" vs "Unoccupied" means.
- **State OFF**: $x = 0.0$. Likelihood is calculated based on how close 0.0 is to the learned means.

### 4. Bayesian Update

These densities are used directly in the Bayesian update formula.

- If $x$ is closer to the Occupied Mean, the likelihood ratio favors occupancy.
- If $x$ is closer to the Unoccupied Mean, it favors vacancy.
- The strength of the evidence scales with how extreme the value is relative to the distributions.

### Example

**Scenario**: Temperature Sensor
- **Unoccupied**: Mean = 20°C
- **Occupied**: Mean = 24°C

| Current Temp ($x$) | Result |
| :--- | :--- |
| **20°C** | **Strong Vacancy Evidence** (Matches Unoccupied mean) |
| **22°C** | **Neutral** (Ambiguous overlap) |
| **24°C** | **Strong Occupancy Evidence** (Matches Occupied mean) |

**Scenario**: Media Player (Binary)
- **Unoccupied**: Mean = 0.1 (Mostly OFF)
- **Occupied**: Mean = 0.9 (Mostly ON)

| Current State ($x$) | Result |
| :--- | :--- |
| **OFF (0.0)** | **Strong Vacancy Evidence** (Matches Unoccupied mean 0.1) |
| **ON (1.0)** | **Strong Occupancy Evidence** (Matches Occupied mean 0.9) |

## Benefits

1. **Unified Analysis**: Single consistent logic for all sensor types.
2. **No "Cliff Edge"**: Small changes in sensor values result in small changes in probability.
3. **True Evidence Weighting**: Extreme values provide stronger evidence.
4. **Automatic Calibration**: The system learns what is "normal" for each specific room.

## Data Flow

1. **Data Collection**:
   - **Numeric Sensors**: `NumericSamples` are recorded on sensor changes.
   - **Binary Sensors**: `Intervals` are recorded. During analysis, these are converted to samples (0.0/1.0).
   - `OccupiedIntervals` track occupancy (ground truth from motion sensors).
2. **Hourly Analysis**: The `analyze_correlation` job runs as part of the analysis cycle (every hour).
3. **Entity Update**: Live `Entity` objects are updated with `learned_gaussian_params`.
4. **Protection**: These learned parameters take precedence over standard defaults.

## Viewing Results

Call the `area_occupancy.run_analysis` service to view results:

```yaml
sensor.lounge_temperature:
  type: temperature
  is_active: false # (Binary abstraction for UI only)
  gaussian_params:
    mean_occupied: 24.0
    std_occupied: 1.0
    mean_unoccupied: 20.0
    std_unoccupied: 1.0
  analysis_error: null
```

If a sensor is not correlated, you might see:

```yaml
sensor.random_noise:
  analysis_error: "no_correlation"
  gaussian_params: null
```
