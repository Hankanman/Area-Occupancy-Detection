---
applyTo: '**'
---
# Bayesian Probability Calculations

## Prior Probability Calculation

The prior probabilities for each input entity are calculated in [custom_components/area_occupancy/calculate_prior.py](mdc:custom_components/area_occupancy/calculate_prior.py) using historical data:

1. For each input entity:
   - Query historical states using recorder component
   - Compare with primary occupancy indicator states
   - Calculate:
     - prob_given_true: P(Entity State | Area Occupied)
     - prob_given_false: P(Entity State | Area Not Occupied)
     - prior_probability: P(Entity State)

2. For environmental sensors:
   - Analyze historical correlations with occupancy patterns
   - Calculate baseline values and occupancy-induced changes
   - Determine sensor-specific thresholds and response patterns
   - Generate prior probabilities for environmental conditions

3. If insufficient history:
   - Fall back to defaults in [custom_components/area_occupancy/probabilities.py](mdc:custom_components/area_occupancy/probabilities.py)

## Composite Bayesian Calculation

The real-time occupancy probability is calculated in [custom_components/area_occupancy/calculate_prob.py](mdc:custom_components/area_occupancy/calculate_prob.py):

1. For each input entity:
   - Get current state
   - Use corresponding priors (calculated or default)
   - Apply Bayes' theorem:
     P(Occupied | Evidence) = P(Evidence | Occupied) * P(Occupied) / P(Evidence)

2. For environmental sensors:
   - Process environmental data through [environmental_analysis.py](mdc:custom_components/area_occupancy/environmental_analysis.py)
   - Generate environmental occupancy probability using ML or deterministic methods
   - Weight environmental probability by confidence score
   - Integrate with traditional sensor probabilities

3. Combine probabilities using:
   - For independent evidence: P = P1 * P2 * ... * Pn
   - For dependent evidence: Use appropriate weighting/correlation factors
   - For environmental evidence: Apply confidence weighting and temporal factors

4. Normalize final probability to 0-100% range

## Implementation Guidelines

### Prior Calculation
- Use significant state changes from recorder
- Consider time windows for correlation
- Handle missing or invalid data
- Cache results to avoid recalculation
- Update periodically (configurable interval)

### Real-time Calculation
- Update on any input entity state change
- Handle unavailable entities gracefully
- Apply confidence weighting
- Consider temporal factors
- Optimize for performance

### Data Flow
1. [coordinator.py](mdc:custom_components/area_occupancy/coordinator.py) triggers updates
2. [calculate_prior.py](mdc:custom_components/area_occupancy/calculate_prior.py) computes priors
3. [calculate_prob.py](mdc:custom_components/area_occupancy/calculate_prob.py) computes final probability
4. [decay_handler.py](mdc:custom_components/area_occupancy/decay_handler.py) applies probability decay
5. [storage.py](mdc:custom_components/area_occupancy/storage.py) persists state data
6. Results update sensors in [sensor.py](mdc:custom_components/area_occupancy/sensor.py) and [binary_sensor.py](mdc:custom_components/area_occupancy/binary_sensor.py)

## Probability Decay

The probability decay functionality is implemented in [custom_components/area_occupancy/decay_handler.py](mdc:custom_components/area_occupancy/decay_handler.py):

1. **Decay Triggers**:
   - Starts when occupancy probability drops below threshold
   - Continues until probability reaches minimum or area becomes occupied again

2. **Decay Algorithm**:
   - Exponential decay based on configurable decay rate
   - Time-based decay using configured intervals
   - Smooth transitions to avoid sudden probability jumps

3. **Configuration**:
   - Decay rate configurable via number entity
   - Decay start threshold configurable
   - Minimum decay probability configurable

## Data Persistence

State persistence is handled by [custom_components/area_occupancy/storage.py](mdc:custom_components/area_occupancy/storage.py):

1. **Stored Data**:
   - Current probability state
   - Prior calculation results
   - Decay state and parameters
   - Historical calculation metadata

2. **Storage Format**:
   - JSON serialization using type-safe conversion
   - Backward compatibility with version migrations
   - Error recovery for corrupted data

3. **Performance**:
   - Async write operations
   - Batched updates to minimize I/O
   - Lazy loading of stored data

## Virtual Sensors

Virtual sensor implementations in [custom_components/area_occupancy/virtual_sensor/](mdc:custom_components/area_occupancy/virtual_sensor/):

1. **Wasp in Box Algorithm**:
   - Implements probabilistic occupancy detection
   - Uses motion patterns and timing analysis
   - Provides confidence scoring for occupancy states

2. **Integration**:
   - Treated as standard input entities
   - Participate in Bayesian calculations
   - Configurable weighting and sensitivity

### Error Handling
- Validate probability ranges (0-1)
- Handle division by zero
- Log calculation steps at debug level
- Provide fallback values
- Report calculation errors
- Use custom exceptions from [exceptions.py](mdc:custom_components/area_occupancy/exceptions.py)
- Graceful degradation when components fail
- Recovery mechanisms for corrupted state data

## Type Safety

All probability calculations use type-safe data structures from [custom_components/area_occupancy/types.py](mdc:custom_components/area_occupancy/types.py):

1. **ProbabilityState**: Main state container with validation
2. **PriorData**: Prior probability data with range validation  
3. **DecayState**: Decay parameters and status tracking
4. **ConfigData**: Configuration validation and defaults

## State Mapping

Entity state mapping is handled by [custom_components/area_occupancy/state_mapping.py](mdc:custom_components/area_occupancy/state_mapping.py):

1. **State Normalization**:
   - Convert entity states to probability-relevant values
   - Handle different entity types (binary, sensor, etc.)
   - Map complex states to binary occupancy indicators

2. **State Validation**:
   - Validate entity states before processing
   - Handle unavailable or unknown states
   - Provide sensible defaults for missing data

