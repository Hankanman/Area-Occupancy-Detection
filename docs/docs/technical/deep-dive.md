# Area Occupancy Probability Calculation Explained

This document gives a high level overview of how the `area_occupancy` integration calculates occupancy probability and manages learned data.

## Core Concepts

- **Occupancy Probability:** Final output value (0.0–1.0) indicating the likelihood that the area is currently occupied.
- **Priors:** Historical probabilities learned from motion sensors and other configured entities.
- **Likelihoods:** For each entity, probabilities `P(Active | Occupied)` and `P(Active | Not Occupied)` learned from history.
- **Weights:** User-configured values (0.0–1.0) assigned per sensor type to influence their contribution.
- **Decay:** Exponential reduction in probability when no fresh evidence is present.
- **Threshold:** Probability level at which the binary occupancy sensor turns `on`.

## Complete Calculation Process

The area occupancy calculation operates in two main phases:

1. **Initialization & Learning Phase**: Sets up the system, loads historical data, and learns prior probabilities and likelihoods from sensor history
2. **Real-Time Calculation Phase**: Continuously monitors sensor states and calculates current occupancy probability using Bayesian inference

For a complete end-to-end explanation of the calculation process, see [Complete Calculation Flow](calculation-flow.md).

## Data Flow and Components

- **AreaOccupancyCoordinator (`coordinator.py`):** Central orchestrator that tracks entity states, schedules updates, handles decay and stores configuration.
- **EntityManager (`data/entity.py`):** Creates and maintains `Entity` objects with evidence, likelihoods and decay data.
- **Prior (`data/prior.py`):** Handles learning priors and likelihoods from historical recorder data and exposes time-based priors.
- **Bayesian Utilities (`utils.py`):** Provides `bayesian_probability` and helper functions for log-space probability calculations.
- **Database (`db.py`):** Stores historical state intervals used for learning.
- **Services (`service.py`):** Exposes services such as `run_analysis` and `get_area_status`.

For visual representations of data flow, see [Data Flow Diagrams](data-flow.md).

## Processing Steps

1. **Initialization:** Coordinator loads configuration, sets up entities and loads any stored priors from the database.
2. **State Updates:** When a monitored entity changes state, the coordinator updates the corresponding `Entity` object and triggers a probability recalculation.
3. **Probability Calculation:** `bayesian_probability` combines entity evidence with the area and time priors in log space, applying the configured weights.
4. **Decay Handling:** If probability decreases, entity decay objects gradually reduce their influence until new evidence appears.
5. **Learning Priors:** Periodically or via `run_analysis`, the `Prior` class analyses recorder history to update priors and likelihoods which are stored in the database.
6. **Outputs:** The coordinator updates Home Assistant entities (probability, status, priors, evidence, decay, threshold) with the latest values.

This architecture allows the integration to react quickly to new sensor data while continuously refining its understanding of each entity's reliability over time.

## Detailed Documentation

For in-depth explanations of specific aspects of the calculation:

### Core Calculation Process

- **[Complete Calculation Flow](calculation-flow.md)** - End-to-end process from initialization through real-time updates
- **[Bayesian Calculation Deep Dive](bayesian-calculation.md)** - Detailed mathematical explanation of the Bayesian probability calculation
- **[Calculation Feature Documentation](../features/calculation.md)** - User-facing documentation with examples

### Learning Processes

- **[Prior Calculation Deep Dive](prior-calculation.md)** - How global and time-based priors are calculated from historical data
- **[Likelihood Calculation Deep Dive](likelihood-calculation.md)** - How sensor reliability likelihoods are learned
- **[Prior Learning Feature](../features/prior-learning.md)** - User-facing prior learning documentation
- **[Likelihood Feature](../features/likelihood.md)** - User-facing likelihood documentation

### Evidence and State Management

- **[Entity Evidence Collection](entity-evidence.md)** - How evidence is collected from sensors and integrated with decay
- **[Decay Feature](../features/decay.md)** - User-facing decay documentation

### Visual Guides

- **[Data Flow Diagrams](data-flow.md)** - Visual flow diagrams using Mermaid syntax showing initialization, learning, real-time updates, and component interactions

## Key Functions and Files

### Initialization

- `coordinator.py:setup()` - Lines 241-321: Coordinator initialization and area setup
- `area/area.py:__init__()` - Lines 90-150: Area component initialization
- `data/analysis.py:start_prior_analysis()` - Lines 911-967: Prior learning orchestration

### Real-Time Calculation

- `coordinator.py:track_entity_state_changes()` - Lines 585-620: Entity state change detection
- `data/entity.py:Entity.has_new_evidence()` - Lines 175-220: Evidence transition detection
- `area/area.py:probability()` - Lines 188-201: Probability calculation entry point
- `utils.py:bayesian_probability()` - Lines 55-157: Core Bayesian calculation
- `utils.py:combine_priors()` - Lines 160-225: Prior combination in logit space

### Prior Calculation

- `data/analysis.py:PriorAnalyzer.analyze_area_prior()` - Lines 174-381: Global prior calculation
- `data/analysis.py:PriorAnalyzer.analyze_time_priors()` - Lines 383-545: Time-based prior calculation
- `data/prior.py:Prior.value` - Lines 79-116: Prior retrieval and combination

### Likelihood Calculation

- `db/correlation.py:analyze_correlation()` - Unified correlation analysis for all sensors (numeric and binary)
- `data/entity.py:Entity.get_likelihoods()` - Dynamic PDF-based likelihood calculation at runtime

### Decay

- `data/decay.py:Decay.decay_factor` - Lines 37-50: Decay factor calculation
- `data/entity.py:Entity.decay_factor` - Lines 151-160: Entity decay factor with evidence check
