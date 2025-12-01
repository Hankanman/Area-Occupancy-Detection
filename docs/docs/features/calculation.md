# Bayesian Probability Calculation

This integration uses Bayesian probability to determine the likelihood of an area being occupied based on the current states of configured sensors.

## Core Concept

Instead of a simple binary "motion detected = occupied" logic, this integration calculates a probability score (0% to 100%) representing the confidence that the area is occupied.

## Calculation Steps

### 1. Collect Evidence

Each configured entity reports whether it currently provides evidence of occupancy. Entities that are decaying after recent activity are also treated as evidence.

The evidence collection process:

- Retrieves current state from Home Assistant
- Determines if state indicates activity (based on `active_states` or `active_range`)
- Considers decay: if entity was recently active, decay may still provide evidence
- Returns `True` (active), `False` (inactive), or `None` (unavailable)

When evidence transitions from active to inactive, decay starts automatically.

### 2. Determine Priors

The integration combines the area prior (learned from history) with a time-based prior to form a baseline probability using a weighted average in logit space.

The prior combination process:

1. Gets global prior from database (learned from historical sensor data)
2. Gets time-based prior for current day-of-week and time-slot
3. Converts both to logit space: `logit(p) = log(p / (1-p))`
4. Combines using weighted average: `combined_logit = area_weight * area_logit + time_weight * time_logit`
5. Converts back to probability: `combined_prior = 1 / (1 + exp(-combined_logit))`
6. Applies prior factor (1.05) to slightly increase baseline
7. Clamps to valid range [MIN_PROBABILITY, MAX_PROBABILITY]

The default time weight is 0.2, meaning 20% of the prior comes from time-based patterns and 80% from the global area prior.

### 3. Adjust Likelihoods

For each entity, the learned likelihoods `P(Active | Occupied)` and `P(Active | Not Occupied)` are adjusted based on the entity's current state:

#### Active Entities

When an entity is active (or decaying), it uses the learned likelihoods directly:

- `p_t = prob_given_true`
- `p_f = prob_given_false`

If the entity is decaying, the likelihoods are interpolated between their learned values and neutral (0.5) based on the decay factor:

```
p_t_adjusted = 0.5 + (p_t_learned - 0.5) * decay_factor
p_f_adjusted = 0.5 + (p_f_learned - 0.5) * decay_factor
```

This gradually reduces the influence of stale evidence as decay progresses.

#### Inactive Entities

When an entity is inactive (not providing evidence), it uses **inverse likelihoods**:

```
p_t = 1.0 - prob_given_true  # P(Inactive | Occupied)
p_f = 1.0 - prob_given_false  # P(Inactive | Not Occupied)
```

This ensures inactive sensors provide proper negative evidence. For example, if a sensor is usually active when occupied (`prob_given_true = 0.8`), then when it's inactive, it suggests the area is likely not occupied (`p_t = 0.2`). This prevents inactive sensors from diluting the effect of active sensors.

The decay factor is calculated using exponential decay: `decay_factor = 0.5^(age / half_life)`, where `age` is the time since evidence became inactive.

### 4. Log-Space Combination

The calculation is performed in log space for numerical stability. For each entity the log probabilities for the "occupied" and "not occupied" hypotheses are accumulated and weighted according to the entity type's configured weight.

The process:

1. **Entity Filtering**: Removes entities with zero weight or invalid likelihoods
2. **Prior Clamping**: Ensures prior is in valid range [MIN_PROBABILITY, MAX_PROBABILITY]
3. **Log-Space Initialization**:

   ```
   log_true = log(prior)
   log_false = log(1 - prior)
   ```

4. **Entity Processing**: For each entity:
   - Determines effective evidence (current or decaying)
   - Gets adjusted likelihoods (with decay if applicable)
   - Clamps likelihoods to avoid log(0) or log(1)
   - Calculates weighted log contributions:

     ```
     contribution_true = log(p_t) * entity.weight
     contribution_false = log(p_f) * entity.weight
     ```

   - Accumulates into log probabilities:

     ```
     log_true += contribution_true
     log_false += contribution_false
     ```

### 5. Final Probability

The log probabilities are exponentiated and normalised to produce the final occupancy probability.

The normalization process:

1. Finds maximum log value: `max_log = max(log_true, log_false)`
2. Subtracts maximum to prevent overflow:

   ```
   true_prob = exp(log_true - max_log)
   false_prob = exp(log_false - max_log)
   ```

3. Normalizes: `probability = true_prob / (true_prob + false_prob)`
4. Handles edge case where both probabilities are zero (returns prior)

## Output

The result of this calculation is shown by the **Occupancy Probability** sensor, which displays the calculated probability as a percentage (0% to 100%).

The **Occupancy Status** binary sensor compares this probability to the configured **Occupancy Threshold** to determine its `on` or `off` state. When the probability equals or exceeds the threshold, the status sensor turns `on`.

## Mathematical Foundation

The calculation uses Bayes' theorem to combine evidence from multiple sensors. The system works in log space for numerical stability when combining many probabilities. For detailed mathematical explanations, see [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md).

## Entity Weight Application

Each entity type has a configured weight (0.0-1.0) that determines how much its evidence contributes to the final probability.

The weight is applied as a multiplier to the log probability contribution:

- Weight 1.0: Full contribution (entity fully influences the result)
- Weight 0.5: Half contribution (entity has moderate influence)
- Weight 0.0: No contribution (entity is excluded from calculation)

Default weights by entity type:

- Motion sensors: 0.85 (high reliability)
- Media players: 0.70 (medium-high reliability)
- Appliances: 0.40 (medium reliability)
- Doors/Windows: 0.20-0.30 (low reliability)
- Environmental sensors: 0.10 (very low reliability) - includes temperature, humidity, illuminance, CO2, sound pressure, atmospheric pressure, air quality, VOC, PM2.5, and PM10 sensors

## Decay Interpolation

When [Probability Decay](../features/decay.md) is active, likelihoods are interpolated between their learned values and neutral probabilities (0.5) based on the decay factor. As decay progresses, the likelihoods move toward neutral, gradually reducing their influence on the final probability. For the mathematical formula, see [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md#decay-interpolation).

## Edge Case Handling

The calculation handles several edge cases to ensure robust operation:

- **Unavailable Entities**: Entities with unavailable states are skipped unless they're decaying
- **Zero Weight Entities**: Entities with zero weight are excluded from the calculation
- **Invalid Likelihoods**: Entities with invalid likelihoods are excluded to prevent calculation errors
- **No Entities**: If no valid entities are available, the calculation returns the prior probability
- **Numerical Stability**: The system uses various techniques to prevent numerical overflow and underflow

For detailed technical information about edge case handling, see [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md#numerical-stability-techniques).

## Example Calculation

Consider an area with:

- Prior: 0.3 (30% baseline occupancy)
- Motion sensor: Active, weight 0.85, `P(Active|Occupied)=0.9`, `P(Active|Not Occupied)=0.1`
- Media player: Inactive, weight 0.70, `P(Active|Occupied)=0.6`, `P(Active|Not Occupied)=0.2`

Step 1: Initialize log probabilities

```
log_true = log(0.3) = -1.204
log_false = log(0.7) = -0.357
```

Step 2: Process motion sensor (active)

```
p_t = 0.9, p_f = 0.1
log_true += log(0.9) * 0.85 = -1.204 + (-0.105) * 0.85 = -1.293
log_false += log(0.1) * 0.85 = -0.357 + (-2.303) * 0.85 = -2.315
```

Step 3: Process media player (inactive)

For inactive entities, we use **inverse likelihoods**:

```
p_t = 1 - 0.6 = 0.4  # P(Inactive | Occupied) = 1 - P(Active | Occupied)
p_f = 1 - 0.2 = 0.8  # P(Inactive | Not Occupied) = 1 - P(Active | Not Occupied)

log_true += log(0.4) * 0.70 = -1.293 + (-0.916) * 0.70 = -1.934
log_false += log(0.8) * 0.70 = -2.315 + (-0.223) * 0.70 = -2.471
```

The inverse likelihoods provide negative evidence (the inactive media player suggests the area might not be occupied), but the motion sensor's strong positive evidence dominates the calculation.

Step 4: Process door sensor (active)

```
p_t = 0.4, p_f = 0.3
log_true += log(0.4) * 0.25 = -1.934 + (-0.916) * 0.25 = -2.163
log_false += log(0.3) * 0.25 = -2.471 + (-1.204) * 0.25 = -2.772
```

Step 5: Normalize

```
max_log = max(-2.163, -2.772) = -2.163
true_prob = exp(-2.163 - (-2.163)) = exp(0) = 1.0
false_prob = exp(-2.772 - (-2.163)) = exp(-0.609) = 0.544
probability = 1.0 / (1.0 + 0.544) = 0.648 (64.8%)
```

The motion sensor's strong positive evidence (active with high `P(Active|Occupied)`) significantly increases the probability from the 30% prior to 67.6%.

## See Also

- [Complete Calculation Flow](../technical/calculation-flow.md) - End-to-end process explanation
- [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md) - Detailed mathematical explanation
- [Prior Learning](../features/prior-learning.md) - How priors are learned from history
- [Likelihood Learning](../features/likelihood.md) - How likelihoods are learned
- [Decay Feature](../features/decay.md) - Decay mechanism overview
- [Entity Evidence Collection](../technical/entity-evidence.md) - How evidence is determined
