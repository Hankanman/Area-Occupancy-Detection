# Complete Calculation Flow

This document provides a comprehensive end-to-end explanation of how area occupancy probability is calculated, from initial setup through real-time updates.

## Overview

The area occupancy calculation process operates in two main phases:

1. **Initialization & Learning Phase**: Sets up the system, loads historical data, and learns prior probabilities and likelihoods from sensor history
2. **Real-Time Calculation Phase**: Continuously monitors sensor states and calculates current occupancy probability using Bayesian inference

## Phase 1: Initialization & Learning

### Coordinator Setup

The process begins when Home Assistant loads the integration. The system initializes all areas and their components.

Key steps:

1. Load areas from configuration
2. Validate at least one area exists
3. Initialize each area's components
4. Load stored data from database
5. Track entity state changes
6. Start periodic timers (decay, save, analysis)

### Area Initialization

Each area is initialized with its configuration and components.

Components created:

- **Config**: Area-specific configuration (sensors, thresholds, weights)
- **EntityManager**: Manages all sensor entities for the area
- **Prior**: Handles prior probability calculations and caching
- **Purpose**: Manages area purpose and decay settings

### Database Loading

Historical data is loaded from the database to restore learned priors and likelihoods.

The database stores:

- Global priors for each area
- Time-based priors (day of week × time slot)
- Entity likelihoods (`P(Active|Occupied)` and `P(Active|Not Occupied)`)
- Entity configurations and states

### Prior Analysis

The system learns baseline occupancy probabilities from historical sensor data. This includes calculating both global priors (overall occupancy rate) and time-based priors (occupancy probability for each day-of-week and time-slot combination).

See [Prior Learning](../features/prior-learning.md) for detailed explanation of how priors are calculated and used.

### Likelihood Analysis

The system learns how reliable each sensor is as evidence of occupancy by analyzing sensor activity relative to occupied intervals determined from motion sensors.

See [Likelihood Calculation](likelihood-calculation.md) for detailed explanation of how likelihoods are learned for different sensor types.

## Phase 2: Real-Time Calculation

### Entity State Change Detection

When any monitored sensor changes state, the system detects the change and triggers recalculation.

Flow:

1. Home Assistant fires state change event
2. System receives the state change event
3. Finds which area(s) contain the changed entity
4. Checks if entity has new evidence
5. If evidence changed, triggers probability recalculation

### Evidence Collection

Each entity determines its current evidence state (active/inactive/unavailable) by checking its current state against configured active criteria. Entities that are decaying from recent activity may still provide evidence.

See [Entity Evidence Collection](entity-evidence.md) for detailed explanation of how evidence is collected and integrated with decay.

### Decay Calculation

When evidence transitions from active to inactive, decay gradually reduces its influence.

Process:

1. Decay starts when evidence transitions from active to inactive
2. Decay factor calculated using exponential decay: `0.5^(age/half_life)`
3. Decay stops when:
   - Evidence becomes active again
   - Decay factor drops below 5% (practical zero)

See [Decay Feature](../features/decay.md) for user-facing documentation.

### Prior Combination

The system combines the global prior with the time-based prior for the current time slot.

Process:

1. Gets global prior from database (learned from history)
2. Gets time-based prior for current day-of-week and time-slot
3. Combines using weighted average in logit space
4. Applies prior factor (1.05) to slightly increase baseline probability
5. Clamps to valid range [MIN_PROBABILITY, MAX_PROBABILITY]

The combination uses logit space for better interpolation:

- Converts probabilities to logits: `logit(p) = log(p / (1-p))`
- Weighted combination: `combined_logit = area_weight * area_logit + time_weight * time_logit`
- Converts back: `combined_prior = 1 / (1 + exp(-combined_logit))`

### Bayesian Probability Calculation

The core calculation combines all entity evidence with the prior using Bayesian inference.

Process:

1. **Entity Filtering**: Removes entities with zero weight or invalid likelihoods
2. **Prior Clamping**: Ensures prior is in valid range
3. **Log-Space Initialization**: Starts with log probabilities for occupied and not-occupied hypotheses
4. **Entity Processing**: For each entity:
   - Determines effective evidence (current or decaying)
   - Gets likelihoods based on evidence state:
     - **Active entities**: Uses `prob_given_true` and `prob_given_false` directly
     - **Inactive entities**: Uses inverse likelihoods (`1 - prob_given_true`, `1 - prob_given_false`)
   - Applies decay interpolation if entity is decaying
   - Calculates log contributions weighted by entity weight
   - Accumulates into log probabilities
5. **Normalization**: Converts log probabilities back to probability space
6. **Result**: Final probability between 0.0 and 1.0

See [Bayesian Calculation Deep Dive](bayesian-calculation.md) for detailed mathematical explanation.

### Final Probability Output

The calculated probability is exposed through Home Assistant sensors.

Outputs:

- **Occupancy Probability Sensor**: Shows the calculated probability (0.0-1.0)
- **Occupancy Status Binary Sensor**: `on` if probability >= threshold, `off` otherwise
- **Prior Probability Sensor**: Shows the combined prior value
- **Evidence Sensor**: Shows which entities are providing evidence
- **Decay Sensor**: Shows decay status

## Complete Flow Diagram

See [Data Flow Diagrams](data-flow.md) for visual representations of these processes.

## Key Concepts

### Log-Space Calculation

All probability calculations use log space for numerical stability. This prevents underflow/overflow when multiplying many small probabilities together.

### Decay Interpolation

When an entity is decaying, its likelihoods are interpolated between their learned values and neutral (0.5) based on the decay factor. This gradually reduces the influence of stale evidence.

### Entity Weights

Each entity type has a configured weight (0.0-1.0) that determines how much its evidence contributes to the final probability. Higher weights mean stronger influence.

### Prior Factor

The prior is multiplied by 1.05 before use, slightly increasing the baseline probability. This helps prevent the system from being too conservative.

## See Also

- [Prior Calculation Deep Dive](../features/prior-learning.md) - Detailed prior learning process
- [Likelihood Calculation Deep Dive](likelihood-calculation.md) - Detailed likelihood learning process
- [Entity Evidence Collection](entity-evidence.md) - How evidence is determined
- [Bayesian Calculation Deep Dive](bayesian-calculation.md) - Mathematical details
- [Data Flow Diagrams](data-flow.md) - Visual flow diagrams
- [Calculation Feature](../features/calculation.md) - User-facing documentation
- [Prior Learning Feature](../features/prior-learning.md) - Prior learning overview
- [Decay Feature](../features/decay.md) - Decay mechanism overview
