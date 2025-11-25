# Likelihood Calculation

This document explains how the system learns and uses likelihood probabilities for each sensor entity.

## Overview

Likelihoods represent how reliable each sensor is as evidence of occupancy. The system uses different analysis methods depending on sensor type:

- **Numeric Sensors** (temperature, humidity, CO2, etc.): Learn statistical distributions using correlation analysis with Gaussian PDFs.
- **Binary Sensors** (media players, appliances, doors, windows): Calculate static probabilities from duration-based interval overlap analysis.
- **Motion Sensors**: Use user-configurable static probabilities (not learned).

**Important:** Motion sensors do **not** have learned likelihoods. Instead, they use **user-configurable likelihoods** that can be set per area during configuration (defaults: `prob_given_true=0.95`, `prob_given_false=0.02`). This is because motion sensors are used as ground truth to determine occupied intervals. Learning motion sensor likelihoods would create a circular dependency.

## Likelihood Analysis Process

The analysis process runs periodically (typically hourly) as part of the analysis cycle.

See **[Sensor Correlation Analysis Chain](analysis-chain.md)** for the detailed flow.

### Analysis Methods by Sensor Type:

1. **Numeric Sensors**: Use correlation analysis to learn Gaussian distributions (Mean and Standard Deviation) for occupied and unoccupied states. Likelihoods are calculated dynamically at runtime using Gaussian PDFs.

2. **Binary Sensors**: Use duration-based analysis to calculate static probabilities directly from interval overlap durations. These are stored as `prob_given_true` and `prob_given_false` values.

3. **Motion Sensors**: Use configured static probabilities (not analyzed).

## How Likelihoods Are Used in Real-Time Calculation

During real-time probability calculation, likelihoods are retrieved differently based on sensor type.

**Code Reference:** `custom_components/area_occupancy/data/entity.py::get_likelihoods()`

### Numeric Sensors: Dynamic PDF Calculation

For numeric sensors with learned Gaussian parameters, the system calculates two probability densities dynamically:

$$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} $$

1. **$P(x | Occupied)$**: Likelihood of current value $x$ given the "Occupied" distribution.
2. **$P(x | Unoccupied)$**: Likelihood of current value $x$ given the "Unoccupied" distribution.

If the sensor state is unavailable (e.g., "unknown"), the system uses a representative value (average of occupied and unoccupied means) to calculate probabilities.

### Binary Sensors: Static Probabilities

For binary sensors (media, appliances, doors, windows), the system uses static probabilities calculated from duration-based analysis:

- **$P(Active | Occupied)$**: Probability that the sensor is active when the area is occupied.
- **$P(Active | Unoccupied)$**: Probability that the sensor is active when the area is unoccupied.

These values are stored directly and used regardless of the current sensor state.

### Motion Sensors: Configured Probabilities

Motion sensors use the user-configured static probabilities set during area configuration.

### Decay-Adjusted Likelihoods

When an entity is decaying (e.g., motion sensor that just turned off), its likelihoods are interpolated toward neutral (0.5).

**Code Reference:** `custom_components/area_occupancy/utils.py`

This gradually reduces the influence of stale evidence as decay progresses.

### Weight Application

The likelihoods are weighted by the entity's configured weight:

```
contribution_true = log(p_t) * entity.weight
contribution_false = log(p_f) * entity.weight
```

Higher weights mean the entity's evidence has more influence on the final probability.

## Database Schema

### NumericCorrelations Table

Stores learned Gaussian parameters for numeric sensors:

- `area_name`, `entity_id`
- `mean_occupied`, `std_occupied`
- `mean_unoccupied`, `std_unoccupied`
- `correlation_coefficient`, `confidence`

### Entities Table

Stores static probabilities for binary sensors:

- `prob_given_true`: Probability sensor is active when occupied (for binary sensors)
- `prob_given_false`: Probability sensor is active when unoccupied (for binary sensors)
- `learned_gaussian_params`: JSON field storing Gaussian parameters (for numeric sensors)
- `analysis_error`: Reason why analysis failed or was not performed

### NumericSamples Table

Stores raw numeric sensor values with timestamps.

### Intervals Table

Stores binary sensor state intervals (on/off periods) used for duration-based analysis.

## Default Likelihoods

If history-based learning is disabled, insufficient data is available, or correlation fails, the system uses default static likelihoods from the entity type definition:

**Code Reference:** `custom_components/area_occupancy/data/entity_type.py`

Default values vary by entity type:

- Motion sensors: High `P(Active | Occupied)`, low `P(Active | Not Occupied)` (configured per area)
- Media players: Medium `P(Active | Occupied)`, very low `P(Active | Not Occupied)`
- Environmental sensors: Low `P(Active | Occupied)`, low `P(Active | Not Occupied)`

**Note:** These defaults are used directly from `EntityType` when Gaussian parameters are unavailable. The system does not store fallback values in the database for non-motion sensors.

For non-motion sensors, these defaults are replaced by learned values once sufficient historical data is available.

## See Also

- [Sensor Correlation Analysis Chain](analysis-chain.md) - End-to-end process
- [Sensor Correlation Feature](../features/sensor-correlation.md) - User-facing documentation
- [Prior Calculation Deep Dive](prior-calculation.md) - Related learning process
- [Bayesian Calculation Deep Dive](bayesian-calculation.md) - How likelihoods are used
