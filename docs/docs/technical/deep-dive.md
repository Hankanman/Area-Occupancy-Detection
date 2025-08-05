# Area Occupancy Probability Calculation Explained

This document gives a high level overview of how the `area_occupancy` integration calculates occupancy probability and manages learned data.

## Core Concepts

- **Occupancy Probability:** Final output value (0.0–1.0) indicating the likelihood that the area is currently occupied.
- **Priors:** Historical probabilities learned from the primary occupancy sensor and other configured entities.
- **Likelihoods:** For each entity, probabilities `P(Active | Occupied)` and `P(Active | Not Occupied)` learned from history.
- **Weights:** User-configured values (0.0–1.0) assigned per sensor type to influence their contribution.
- **Decay:** Exponential reduction in probability when no fresh evidence is present.
- **Threshold:** Probability level at which the binary occupancy sensor turns `on`.

## Data Flow and Components

- **AreaOccupancyCoordinator (`coordinator.py`):** Central orchestrator that tracks entity states, schedules updates, handles decay and stores configuration.
- **EntityManager (`data/entity.py`):** Creates and maintains `Entity` objects with evidence, likelihoods and decay data.
- **Prior (`data/prior.py`):** Handles learning priors and likelihoods from historical recorder data and exposes time-based priors.
- **Bayesian Utilities (`utils.py`):** Provides `bayesian_probability` and helper functions for log-space probability calculations.
- **Database (`db.py`):** Stores historical state intervals used for learning.
- **Services (`service.py`):** Exposes services such as `run_analysis` and `get_area_status`.

## Processing Steps

1. **Initialization:** Coordinator loads configuration, sets up entities and loads any stored priors from the database.
2. **State Updates:** When a monitored entity changes state, the coordinator updates the corresponding `Entity` object and triggers a probability recalculation.
3. **Probability Calculation:** `bayesian_probability` combines entity evidence with the area and time priors in log space, applying the configured weights.
4. **Decay Handling:** If probability decreases, entity decay objects gradually reduce their influence until new evidence appears.
5. **Learning Priors:** Periodically or via `run_analysis`, the `Prior` class analyses recorder history to update priors and likelihoods which are stored in the database.
6. **Outputs:** The coordinator updates Home Assistant entities (probability, status, priors, evidence, decay, threshold) with the latest values.

This architecture allows the integration to react quickly to new sensor data while continuously refining its understanding of each entity's reliability over time.
