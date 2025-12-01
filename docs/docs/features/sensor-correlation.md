# Sensor Correlation & Continuous Likelihood

The Area Occupancy Detection integration uses advanced statistical analysis to learn occupancy patterns from sensors. The system uses different analysis methods depending on sensor type:

- **Numeric Sensors** (Temperature, Humidity, CO2, etc.): Use correlation analysis with **Gaussian Probability Density Functions (PDFs)** to calculate continuous, dynamic likelihoods based on exact sensor values.
- **Binary Sensors** (Media Players, Appliances, Doors, Windows): Use duration-based analysis to calculate static probabilities from interval overlap durations.

## Overview

While motion sensors directly indicate presence, other sensors often show correlation with occupancy. For example:

- **Numeric Sensors**:
  - **Temperature** might rise when people are in a room.
  - **CO2** levels often increase with occupancy.
  - **Humidity** might change when a shower is used.
- **Binary Sensors**:
  - **Media Players** might be playing when occupied.
  - **Appliances** might be running when occupied.
  - **Doors/Windows** might be open more often when occupied.

The system analyzes these patterns differently:

- **Numeric sensors** use correlation analysis to learn statistical distributions and calculate dynamic likelihoods.
- **Binary sensors** use duration-based analysis to calculate static probabilities directly from how long they're active during occupied vs. unoccupied periods.

## How It Works

The system uses different analysis methods for numeric and binary sensors:

### Numeric Sensors: Correlation Analysis

#### 1. Correlation Check (Qualification)

Every hour as part of the analysis cycle, the system analyzes the relationship between the sensor's value and the area's occupancy state using the **Pearson correlation coefficient**.

The system classifies correlations into different types based on their strength:

- **Strong Positive Correlation** (≥ 0.4): Value increases significantly when occupied. Classified as `strong_positive`.
- **Strong Negative Correlation** (≤ -0.4): Value decreases significantly when occupied. Classified as `strong_negative`.
- **Weak Positive Correlation** (0.15 to 0.4): Value increases moderately when occupied. Classified as `positive`.
- **Weak Negative Correlation** (-0.4 to -0.15): Value decreases moderately when occupied. Classified as `negative`.
- **No Correlation** (< 0.15 absolute value): No meaningful pattern found. Classified as `none` with `analysis_error: "no_correlation"`.

**Thresholds:**

- **Weak Correlation Threshold**: 0.15 - Minimum correlation strength to be considered meaningful
- **Moderate Correlation Threshold**: 0.4 - Minimum correlation strength for strong correlations

Both strong and weak correlations are used for occupancy detection using the same Gaussian PDF approach. Only correlations below the weak threshold (< 0.15) are rejected to prevent false positives from noise.

#### 2. Learning Distributions

If a sensor qualifies, the system learns two statistical distributions:

1. **Occupied Distribution**: $(\mu_{occ}, \sigma_{occ})$ - The pattern when the room is _known_ to be occupied.
2. **Unoccupied Distribution**: $(\mu_{unocc}, \sigma_{unocc})$ - The pattern when the room is _known_ to be empty.

These parameters are stored in the `Correlations` database table.

#### 3. Calculating Dynamic Likelihood

When the sensor reports a new value $x$, the system calculates two probability densities using the Gaussian PDF formula:

$$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} $$

1. **$P(x | Occupied)$**: Likelihood of $x$ given the "Occupied" distribution.
2. **$P(x | Unoccupied)$**: Likelihood of $x$ given the "Unoccupied" distribution.

#### 4. Bayesian Update

These densities are used directly in the Bayesian update formula.

- If $x$ is closer to the Occupied Mean, the likelihood ratio favors occupancy.
- If $x$ is closer to the Unoccupied Mean, it favors vacancy.
- The strength of the evidence scales with how extreme the value is relative to the distributions.

### Binary Sensors: Duration-Based Analysis

#### 1. Interval Overlap Calculation

The system calculates how long each binary sensor interval overlaps with occupied vs. unoccupied periods:

- **Active Duration During Occupied**: Total seconds the sensor is active while the area is occupied.
- **Active Duration During Unoccupied**: Total seconds the sensor is active while the area is unoccupied.
- **Total Occupied Duration**: Total seconds the area is occupied.
- **Total Unoccupied Duration**: Total seconds the area is unoccupied.

#### 2. Static Probability Calculation

The system calculates two static probabilities:

1. **$P(Active | Occupied)$**: `active_duration_occupied / total_occupied_duration`
2. **$P(Active | Unoccupied)$**: `active_duration_unoccupied / total_unoccupied_duration`

These probabilities are clamped between 0.05 and 0.95 to avoid extreme values.

#### 3. Storage and Usage

These static probabilities are stored directly in the `Entities` table as `prob_given_true` and `prob_given_false`. They are used at runtime regardless of the current sensor state.

### Example

**Scenario**: Temperature Sensor

- **Unoccupied**: Mean = 20°C
- **Occupied**: Mean = 24°C

| Current Temp ($x$) | Result                                                |
| :----------------- | :---------------------------------------------------- |
| **20°C**           | **Strong Vacancy Evidence** (Matches Unoccupied mean) |
| **22°C**           | **Neutral** (Ambiguous overlap)                       |
| **24°C**           | **Strong Occupancy Evidence** (Matches Occupied mean) |

**Scenario**: Media Player (Binary)

- **$P(Active | Occupied)$**: 0.85 (85% chance it's playing when occupied)
- **$P(Active | Unoccupied)$**: 0.05 (5% chance it's playing when unoccupied)

| Current State | Occupied Probability Used | Unoccupied Probability Used | Notes                   |
| :------------ | :------------------------ | :------------------------- | :---------------------- |
| **OFF**       | 0.15                      | 0.95                       | Inverse probabilities   |
| **ON**        | 0.85                      | 0.05                       | Direct probabilities    |

## Benefits

1. **Appropriate Analysis Methods**: Numeric sensors use dynamic PDF calculation for continuous values, while binary sensors use simple duration-based probabilities.
2. **No "Cliff Edge" (Numeric)**: Small changes in sensor values result in small changes in probability.
3. **True Evidence Weighting (Numeric)**: Extreme values provide stronger evidence.
4. **Automatic Calibration**: The system learns what is "normal" for each specific room.
5. **Simple and Reliable (Binary)**: Duration-based analysis provides straightforward probabilities for binary states.

## Data Flow

1. **Data Collection**:
   - **Numeric Sensors**: `NumericSamples` are recorded on sensor changes.
   - **Binary Sensors**: `Intervals` are recorded (on/off periods with timestamps).
   - `OccupiedIntervalsCache` tracks occupancy (ground truth from motion sensors).
2. **Hourly Analysis**:
   - **Numeric Sensors**: `analyze_correlation()` runs correlation analysis and learns Gaussian parameters.
   - **Binary Sensors**: `analyze_binary_likelihoods()` calculates duration-based static probabilities.
3. **Entity Update**:
   - **Numeric Sensors**: Live `Entity` objects are updated with `learned_gaussian_params`.
   - **Binary Sensors**: Live `Entity` objects are updated with `prob_given_true` and `prob_given_false`.
4. **Runtime Usage**: Likelihoods are retrieved via `get_likelihoods()` which uses the appropriate method based on sensor type.

## Viewing Results

Call the `area_occupancy.run_analysis` service to view results:

**Numeric Sensor Example:**

```yaml
sensor.lounge_temperature:
  type: temperature
  prob_given_true: 0.75 # Runtime calculated from Gaussian PDF
  prob_given_false: 0.15
  active_range: [20.0, 24.0] # Learned active range
  analysis_data:
    mean_occupied: 24.0
    std_occupied: 1.0
    mean_unoccupied: 20.0
    std_unoccupied: 1.0
  analysis_error: null
```

**Binary Sensor Example:**

```yaml
light.study_bulb_1:
  type: appliance
  prob_given_true: 0.85 # Static probability from duration analysis
  prob_given_false: 0.10
  active_states: ["on", "standby"]
  analysis_data: null # Binary sensors don't store Gaussian params
  analysis_error: null
```

If a sensor analysis fails, you might see:

```yaml
sensor.random_noise:
  prob_given_true: 0.09 # Falls back to EntityType defaults
  prob_given_false: 0.01
  analysis_error: "no_correlation"
  analysis_data: null
```
