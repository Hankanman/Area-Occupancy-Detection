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

2. If insufficient history:
   - Fall back to defaults in [custom_components/area_occupancy/probabilities.py](mdc:custom_components/area_occupancy/probabilities.py)

## Composite Bayesian Calculation

The real-time occupancy probability is calculated in [custom_components/area_occupancy/calculate_prob.py](mdc:custom_components/area_occupancy/calculate_prob.py):

1. For each input entity:
   - Get current state
   - Use corresponding priors (calculated or default)
   - Apply Bayes' theorem:
     P(Occupied | Evidence) = P(Evidence | Occupied) * P(Occupied) / P(Evidence)

2. Combine probabilities using:
   - For independent evidence: P = P1 * P2 * ... * Pn
   - For dependent evidence: Use appropriate weighting/correlation factors

3. Normalize final probability to 0-100% range

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
4. Results update sensors in [sensor.py](mdc:custom_components/area_occupancy/sensor.py)

### Error Handling
- Validate probability ranges (0-1)
- Handle division by zero
- Log calculation steps at debug level
- Provide fallback values
- Report calculation errors

