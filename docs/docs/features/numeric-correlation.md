# Numeric Sensor Correlation

The Area Occupancy Detection integration includes advanced capability to learn occupancy patterns from numeric sensors (Temperature, Humidity, CO2, etc.) that traditionally don't have binary "occupied/not occupied" states.

## Overview

While motion sensors directly indicate presence, environmental sensors often show correlation with occupancy. For example:

- **Temperature** might rise when people are in a room.
- **CO2** levels often increase with occupancy.
- **Humidity** might change when a shower is used.

This integration automatically analyzes historical data to find these patterns and uses them to improve occupancy detection accuracy.

## How It Works

The system performs a statistical analysis of your sensor data against known occupancy periods (determined by ground-truth sensors like motion detectors).

### 1. Correlation Analysis

Every 30 days (configurable), the system analyzes the relationship between the numeric sensor's value and the area's occupancy state using the **Pearson correlation coefficient**.

- **Positive Correlation**: Value increases when occupied (e.g., Temperature).
- **Negative Correlation**: Value decreases when occupied.
- **No Correlation**: No clear pattern found.

### 2. Learning Active Ranges

When a significant correlation is found (Confidence > 0), the system calculates a dynamic "Active Range" for that sensor.

It compares the statistics of the sensor when the room is **unoccupied** vs. when it is **occupied**.

#### For Positive Correlation (e.g., Temperature rises)

- **Lower Threshold**: `Mean(Unoccupied) + 2 * StdDev(Unoccupied)`
  - The value must be significantly higher than the normal unoccupied baseline to count as evidence.
- **Upper Threshold**: `Mean(Occupied) + 2 * StdDev(Occupied)`
  - If occupied statistics are available and distinct, a closed range is created.
  - If not, the upper bound is open-ended (infinity), meaning "anything above the threshold".

#### For Negative Correlation

- **Upper Threshold**: `Mean(Unoccupied) - 2 * StdDev(Unoccupied)`
  - The value must be significantly lower than the normal unoccupied baseline.
- **Lower Threshold**: `Mean(Occupied) - 2 * StdDev(Occupied)`
  - Similar logic for closed vs. open range.

### 3. Dynamic Likelihood Scaling

Instead of assigning a fixed probability (e.g., "Temperature active = 90% chance of occupancy"), the system scales the evidence based on the **Confidence** of the correlation.

The confidence score ($C$) is derived from the correlation strength and the sample size.

- **Probability given True ($P(E|H)$)**: $0.5 + (C \times 0.4)$
- **Probability given False ($P(E|\neg H)$)**: $0.5 - (C \times 0.4)$

**Examples:**

- **Weak Correlation ($C=0.2$)**:
  - $P(E|H) = 0.58$
  - $P(E|\neg H) = 0.42$
  - _Result_: Contributes a **tiny nudge** to the probability.
- **Strong Correlation ($C=0.8$)**:
  - $P(E|H) = 0.82$
  - $P(E|\neg H) = 0.18$
  - _Result_: Contributes **strong evidence** of occupancy.

### 4. Automatic Likelihood Assignment

For numeric sensors, likelihoods are **NOT** set by counting active/inactive intervals (standard learning), because "active" is a dynamic concept for these sensors.

Instead, they are **set automatically** based on the statistical confidence of the correlation.

- **Standard Method (Binary Sensors)**: "This sensor was ON for 80% of the time the room was occupied."
- **Correlation Method (Numeric Sensors)**: "We are 80% confident that Temperature correlates with Occupancy."

The system automatically applies the scaled likelihoods and **protects** them from being overwritten by the standard learning algorithm. This ensures that for numeric sensors, **Correlation Confidence** is the single source of truth for their reliability.

## Data Flow

1.  **Data Collection**: `NumericSamples` are recorded whenever a sensor state changes. `OccupiedIntervals` track when the area was occupied.
2.  **Nightly Analysis**: The `analyze_numeric_correlation` job runs (via `run_nightly_tasks` service).
3.  **Entity Update**: If a correlation is found, the live `Entity` object is updated with the new `learned_active_range` and scaled likelihood probabilities.
4.  **Protection**: These learned values are "protected" from the standard likelihood learning process, ensuring that statistical correlation takes precedence over simple frequency counting for these sensor types.

## Viewing Results

You can view the correlation results for your entities by calling the `area_occupancy.run_analysis` service. The output will show:

```yaml
sensor.lounge_temperature:
  type: temperature
  active_range:
    - 19.94 # Lower bound (Mean + 2σ Unoccupied)
    - 22.50 # Upper bound (Mean + 2σ Occupied)
  prob_given_true: 0.65 # Scaled by confidence
  prob_given_false: 0.35
```

If `active_range` shows `null` for one bound (e.g., `[19.94, null]`), it means the system is using an open-ended range (e.g., "Active if > 19.94").
