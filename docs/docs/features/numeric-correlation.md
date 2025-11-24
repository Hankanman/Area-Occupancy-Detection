# Numeric Sensor Correlation & Continuous Likelihood

The Area Occupancy Detection integration includes advanced capability to learn occupancy patterns from numeric sensors (Temperature, Humidity, CO2, etc.). Instead of a simple "Active/Inactive" switch, the system calculates a continuous probability of occupancy based on the exact value of the sensor using **Gaussian Probability Density Functions (PDFs)**.

## Overview

While motion sensors directly indicate presence, environmental sensors often show correlation with occupancy. For example:

- **Temperature** might rise when people are in a room.
- **CO2** levels often increase with occupancy.
- **Humidity** might change when a shower is used.

Previously, systems might convert these values into binary states (e.g., "Temperature > 22°C = Active"). This approach loses valuable information. The **Continuous Likelihood** system solves this by dynamically calculating probability based on how well the current value matches historical "Occupied" vs "Unoccupied" patterns.

## How It Works

The system performs a two-stage statistical analysis:

### 1. Correlation Check (Qualification)

Every hour as part of the analysis cycle, the system analyzes the relationship between the numeric sensor's value and the area's occupancy state using the **Pearson correlation coefficient**.

- **Positive Correlation**: Value increases when occupied.
- **Negative Correlation**: Value decreases when occupied.
- **No Correlation**: No clear pattern found.

If the correlation is too weak or the sample size is too small, the sensor is **rejected** for occupancy detection purposes to prevent false positives. You can see the `analysis_error` in the analysis output.

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

### 4. Bayesian Update

These densities are used directly in the Bayesian update formula.

- If $x$ is closer to the Occupied Mean, the likelihood ratio favors occupancy.
- If $x$ is closer to the Unoccupied Mean, it favors vacancy.
- The strength of the evidence scales with how extreme the value is relative to the distributions.

### Example

**Scenario**: Temperature Sensor

- **Unoccupied**: Mean = 20°C
- **Occupied**: Mean = 24°C

| Current Temp ($x$) | Result                                                |
| :----------------- | :---------------------------------------------------- |
| **20°C**           | **Strong Vacancy Evidence** (Matches Unoccupied mean) |
| **22°C**           | **Neutral** (Ambiguous overlap)                       |
| **24°C**           | **Strong Occupancy Evidence** (Matches Occupied mean) |

## Benefits

1. **No "Cliff Edge"**: Small changes in sensor values result in small changes in probability.
2. **True Evidence Weighting**: Extreme values provide stronger evidence.
3. **Automatic Calibration**: The system learns what is "normal" for each specific room.

## Data Flow

1. **Data Collection**: `NumericSamples` are recorded on sensor changes. `OccupiedIntervals` track occupancy.
2. **Hourly Analysis**: The `analyze_numeric_correlation` job runs as part of the analysis cycle (every hour).
3. **Entity Update**: Live `Entity` objects are updated with `learned_gaussian_params`.
4. **Protection**: These learned parameters take precedence over standard binary presence counting.

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
