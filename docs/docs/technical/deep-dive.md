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

The system consists of several conceptual components:

- **Coordinator:** Central orchestrator that tracks entity states, schedules updates, handles decay and stores configuration.
- **Entity Manager:** Creates and maintains entity objects with evidence, likelihoods and decay data.
- **Prior Manager:** Handles learning priors and likelihoods from historical recorder data and exposes time-based priors.
- **Bayesian Calculator:** Performs log-space probability calculations combining evidence with priors.
- **Database:** Stores historical state intervals used for learning.
- **Services:** Exposes services such as `run_analysis` for manual analysis triggers.

For visual representations of data flow, see [Data Flow Diagrams](data-flow.md).

## Processing Steps

1. **Initialization:** System loads configuration, sets up entities and loads any stored priors from the database.
2. **State Updates:** When a monitored entity changes state, the system updates the corresponding entity object and triggers a probability recalculation.
3. **Probability Calculation:** Bayesian probability calculation combines entity evidence with the area and time priors in log space, applying the configured weights.
4. **Decay Handling:** If probability decreases, entity decay gradually reduces their influence until new evidence appears.
5. **Learning Priors:** Periodically or via `run_analysis` service, the system analyses recorder history to update priors and likelihoods which are stored in the database.
6. **Outputs:** The system updates Home Assistant entities (probability, status, priors, evidence, decay, threshold) with the latest values.

This architecture allows the integration to react quickly to new sensor data while continuously refining its understanding of each entity's reliability over time.

## Detailed Documentation

For in-depth explanations of specific aspects of the calculation:

### Core Calculation Process

- **[Complete Calculation Flow](calculation-flow.md)** - End-to-end process from initialization through real-time updates
- **[Bayesian Calculation Deep Dive](bayesian-calculation.md)** - Detailed mathematical explanation of the Bayesian probability calculation
- **[Calculation Feature Documentation](../features/calculation.md)** - User-facing documentation with examples

### Learning Processes

- **[Prior Calculation Deep Dive](../features/prior-learning.md)** - How global and time-based priors are calculated from historical data
- **[Likelihood Calculation Deep Dive](likelihood-calculation.md)** - How sensor reliability likelihoods are learned
- **[Prior Learning Feature](../features/prior-learning.md)** - User-facing prior learning documentation
- **[Likelihood Feature](../features/likelihood.md)** - User-facing likelihood documentation

### Evidence and State Management

- **[Entity Evidence Collection](entity-evidence.md)** - How evidence is collected from sensors and integrated with decay
- **[Decay Feature](../features/decay.md)** - User-facing decay documentation

### Visual Guides

- **[Data Flow Diagrams](data-flow.md)** - Visual flow diagrams using Mermaid syntax showing initialization, learning, real-time updates, and component interactions

## Key Operations

### Initialization

- **Coordinator Setup:** Initializes all areas and loads configuration
- **Area Initialization:** Sets up area components (config, entities, priors, purpose)
- **Prior Learning Orchestration:** Coordinates the analysis of historical data to learn priors

### Real-Time Calculation

- **Entity State Change Detection:** Monitors sensor state changes and triggers recalculation
- **Evidence Transition Detection:** Identifies when entities transition between active and inactive states
- **Probability Calculation:** Core Bayesian calculation combining evidence with priors
- **Prior Combination:** Combines global and time-based priors in logit space

### Prior Calculation

- **Global Prior Calculation:** Analyses historical data to determine overall occupancy rate
- **Time-Based Prior Calculation:** Calculates occupancy probability for each day-of-week and time-slot combination
- **Prior Retrieval and Combination:** Retrieves and combines priors for real-time calculations

### Likelihood Calculation

- **Correlation Analysis:** Unified correlation analysis for all sensors (numeric and binary)
- **Dynamic Likelihood Calculation:** Calculates likelihoods at runtime based on current sensor state

### Decay

- **Decay Factor Calculation:** Calculates exponential decay factor based on time since evidence became inactive
- **Entity Decay Integration:** Integrates decay factor with entity evidence for probability calculation
