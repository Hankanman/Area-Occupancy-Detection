# Likelihood Calculation

This document explains how the system learns and uses likelihood probabilities for each sensor entity, leveraging the Unified Correlation Analysis.

## Overview

Likelihoods represent how reliable each sensor is as evidence of occupancy. For each entity (both numeric and binary), the system learns statistical distributions:

- **Occupied Distribution**: $(\mu_{occ}, \sigma_{occ})$ - The pattern of the sensor state when the room is *known* to be occupied.
- **Unoccupied Distribution**: $(\mu_{unocc}, \sigma_{unocc})$ - The pattern of the sensor state when the room is *known* to be empty.

These values are learned from historical data by correlating each sensor's activity with the area's occupancy state (determined by motion sensors).

**Important:** Motion sensors do **not** have learned likelihoods. Instead, they use **user-configurable likelihoods** that can be set per area during configuration (defaults: `prob_given_true=0.95`, `prob_given_false=0.02`). This is because motion sensors are used as ground truth to determine occupied intervals. Learning motion sensor likelihoods would create a circular dependency.

## Likelihood Analysis Process

The analysis process runs periodically (typically hourly) as part of the Unified Correlation Analysis.

See **[Sensor Correlation Analysis Chain](analysis-chain.md)** for the detailed flow.

### Key differences from previous versions:

- **Binary Sensors**: Instead of duration-weighted counting, binary sensors are now analyzed using the same correlation engine as numeric sensors. Intervals are converted to numeric samples (0.0 for OFF, 1.0 for ON).
- **Statistical Learning**: The system learns Mean and Standard Deviation for both states (Occupied/Unoccupied).
- **Continuous Likelihood**: Likelihoods are no longer static values (e.g., 0.8) but are calculated dynamically based on the current sensor state using Gaussian PDFs.

## How Likelihoods Are Used in Real-Time Calculation

During real-time probability calculation, likelihoods are calculated dynamically.

**Code Reference:** `custom_components/area_occupancy/data/entity.py::get_likelihoods()`

### 1. Dynamic PDF Calculation

When an entity provides a state value (numeric or binary), the system calculates two probability densities:

$$ f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2} $$

1. **$P(x | Occupied)$**: Likelihood of current value $x$ given the "Occupied" distribution.
2. **$P(x | Unoccupied)$**: Likelihood of current value $x$ given the "Unoccupied" distribution.

For binary sensors:
- **ON**: $x = 1.0$
- **OFF**: $x = 0.0$

### 2. Bayesian Update

These densities are used directly in the Bayesian update formula.

### 3. Decay-Adjusted Likelihoods

When an entity is decaying (e.g., motion sensor that just turned off), its likelihoods are interpolated toward neutral (0.5).

**Code Reference:** `custom_components/area_occupancy/utils.py`

This gradually reduces the influence of stale evidence as decay progresses.

### 4. Weight Application

The likelihoods are weighted by the entity's configured weight:

```
contribution_true = log(p_t) * entity.weight
contribution_false = log(p_f) * entity.weight
```

Higher weights mean the entity's evidence has more influence on the final probability.

## Database Schema

### NumericCorrelations Table

Stores learned parameters for each entity:
- `area_name`, `entity_id`
- `mean_occupied`, `std_occupied`
- `mean_unoccupied`, `std_unoccupied`
- `correlation_coefficient`, `confidence`

### NumericSamples Table
Stores raw numeric values.

### Intervals Table
Stores binary sensor states (converted to samples during analysis).

## Default Likelihoods (Fallback)

If history-based learning is disabled, insufficient data is available, or correlation fails, the system falls back to default static likelihoods based on entity type:

**Code Reference:** `custom_components/area_occupancy/data/entity_type.py`

Default values vary by entity type:
- Motion sensors: High `P(Active | Occupied)`, low `P(Active | Not Occupied)`
- Media players: Medium `P(Active | Occupied)`, very low `P(Active | Not Occupied)`
- Environmental sensors: Low `P(Active | Occupied)`, low `P(Active | Not Occupied)`

For non-motion sensors, these defaults are replaced by learned values once sufficient historical data is available.

## See Also

- [Sensor Correlation Analysis Chain](analysis-chain.md) - End-to-end process
- [Sensor Correlation Feature](../features/sensor-correlation.md) - User-facing documentation
- [Prior Calculation Deep Dive](prior-calculation.md) - Related learning process
- [Bayesian Calculation Deep Dive](bayesian-calculation.md) - How likelihoods are used
