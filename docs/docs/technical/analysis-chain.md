# Sensor Correlation Analysis Chain

This document provides a comprehensive breakdown of the unified sensor correlation analysis chain, from data collection through likelihood calculation. It details how both numeric and binary sensors are processed using a single consistent statistical engine.

## Table of Contents

1. [Complete Flow Overview](#complete-flow-overview)
2. [Phase-by-Phase Breakdown](#phase-by-phase-breakdown)
3. [Method Reference](#method-reference)
4. [Unified Architecture](#unified-architecture)

## Complete Flow Overview

The analysis chain consists of four main phases:

1. **Data Collection** (Continuous) - Raw sensor data is continuously synced from Home Assistant
2. **Hourly Analysis Cycle** (Scheduled) - Periodic analysis runs every hour
3. **Correlation Analysis** (Within Analysis Cycle) - Calculates correlations for ALL sensors (numeric and binary)
4. **Likelihood Calculation** (Runtime) - Dynamic likelihood calculation using Gaussian PDFs

```mermaid
flowchart TD
    Start([Hourly Timer Fires]) --> Sync[Sync States from Recorder]
    Sync --> HealthCheck[Database Health Check]
    HealthCheck --> Prune[Prune Old Intervals]
    Prune --> Cache[Populate OccupiedIntervalsCache]
    Cache --> Aggregate[Interval Aggregation]
    Aggregate --> Prior[Prior Analysis]
    Prior --> CorrAnalysis[Correlation Analysis]
    CorrAnalysis --> Refresh[Refresh Coordinator]
    Refresh --> Save[Save to Database]
    Save --> End([Complete])

    CorrAnalysis --> CorrDetail[For Each Entity (Numeric & Binary)]
    CorrDetail --> CheckType{Is Binary?}
    CheckType -- Yes --> Convert[Convert Intervals to Samples]
    CheckType -- No --> GetSamples[Get NumericSamples]

    Convert --> GetIntervals
    GetSamples --> GetIntervals[Get OccupiedIntervalsCache]

    GetIntervals --> MapOccupancy[Map Samples to Occupancy]
    MapOccupancy --> CalcCorr[Calculate Pearson Correlation]
    CalcCorr --> CalcStats[Calculate Mean/Std]
    CalcStats --> SaveCorr[Save to NumericCorrelations]
    SaveCorr --> UpdateEntity[Update Entity with Gaussian Params]

    UpdateEntity --> Runtime[Runtime Likelihood Calculation]
    Runtime --> GetState[Get Current Sensor State]
    GetState --> CalcGaussian[Calculate Gaussian Densities]
    CalcGaussian --> Bayesian[Use in Bayesian Update]
```

## Phase-by-Phase Breakdown

### Phase 1: Data Collection (Continuous)

**Location**: `custom_components/area_occupancy/db/sync.py`

#### Step 1.1: Sync States from Recorder

**Method**: `sync_states()`

**What Happens**:

- Fetches recent state changes from Home Assistant recorder
- Converts numeric sensor states to `NumericSamples` records
- Converts binary sensor states to `Intervals` records
- Stores in database tables

**Data Stored**:

- `NumericSamples` table: Raw numeric sensor values with timestamps
- `Intervals` table: Binary sensor state intervals (on/off periods)

---

### Phase 2: Hourly Analysis Cycle (Scheduled)

**Location**: `custom_components/area_occupancy/coordinator.py::run_analysis()`

**Method**: `run_analysis()`

**Trigger**: Scheduled timer fires every hour

#### Step 2.1: Sync States

Imports latest data from Home Assistant recorder into local database.

#### Step 2.2: Database Health Check & Pruning

Ensures database integrity and removes old data beyond retention period.

#### Step 2.3: Populate OccupiedIntervalsCache

Calculates occupied intervals from motion sensors (ground truth) and caches them.

#### Step 2.4: Interval Aggregation

Aggregates raw intervals into daily/weekly/monthly aggregates for trend analysis.

#### Step 2.5: Prior Analysis

Calculates global prior probability and time-based priors for each area.

#### Step 2.6: Correlation Analysis

**Main analysis path** - Runs correlation analysis for all configured sensors (excluding motion sensors).

#### Step 2.7: Refresh & Save

Updates coordinator state and persists all changes to database.

---

### Phase 3: Correlation Analysis (Step 6 Detail)

**Location**: `custom_components/area_occupancy/coordinator.py::_run_correlation_analysis()`

#### Step 3.1: Get Correlatable Entities

**Method**: `_get_correlatable_entities_by_area()`

**What Happens**:

- Returns all configured sensors for the area (excluding motion sensors).
- Identifies if each entity is binary or numeric.

#### Step 3.2: Analyze Each Entity

For each entity:

1. **Call Analysis**:
   Calls `analyze_and_save_correlation()` with `is_binary` flag.

2. **Update Live Entity**:
   Updates the live entity object with `learned_gaussian_params`.

**Location**: `custom_components/area_occupancy/db/correlation.py`

#### Step 3.3: Analyze Correlation

**Method**: `analyze_correlation()`

**Parameters**:

- `is_binary`: Boolean flag indicating if interval conversion is needed.

**Process**:

1. **Data Retrieval**:

   - If `is_binary`: Calls `convert_intervals_to_samples()` to transform binary intervals into 0.0/1.0 samples.
   - If numeric: Queries `NumericSamples` directly.

2. **Map to Occupancy**:

   - Checks each sample timestamp against `OccupiedIntervalsCache`.
   - Creates parallel arrays of values and occupancy flags (0/1).

3. **Calculate Pearson Correlation**:

   - Determines relationship between value/state and occupancy.

4. **Calculate Statistics**:

   - Learns Mean/Std for Occupied state.
   - Learns Mean/Std for Unoccupied state.

5. **Save Result**:
   - Persists parameters to `NumericCorrelations` table.

---

### Phase 4: Likelihood Calculation (Runtime)

**Location**: `custom_components/area_occupancy/data/entity.py::get_likelihoods()`

This phase occurs at runtime whenever the Bayesian probability calculation needs likelihood values.

#### Step 4.1: Get Likelihoods

**Method**: `get_likelihoods()`

**What Happens**:

1. **Convert State**:

   - Numeric sensors: Use float value.
   - Binary sensors: Convert state "on" -> 1.0, "off" -> 0.0.

2. **Calculate Gaussian Densities**:

   - Calculates $P(value | Occupied)$ using learned Gaussian parameters.
   - Calculates $P(value | Unoccupied)$ using learned Gaussian parameters.

3. **Return Probabilities**:
   - Returns the two densities for Bayesian update.

## Unified Architecture

The system has been unified to eliminate the separate "Likelihood Analysis" path that existed for binary sensors.

### Key Changes:

1. **Single Analysis Path**: Both numeric and binary sensors go through `analyze_correlation()`.
2. **Data Transformation**: Binary intervals are converted to numeric samples (0.0/1.0) on the fly.
3. **Unified Runtime**: All sensors use PDF-based likelihood calculation.
4. **Deduplication**: Logic for occupied interval checking, validation, and error handling is now centralized.

### Benefits:

- **Consistency**: All sensors are treated with the same statistical rigor.
- **Maintainability**: Reduced code duplication and complexity.
- **Flexibility**: Easier to add new sensor types or analysis methods in the future.
