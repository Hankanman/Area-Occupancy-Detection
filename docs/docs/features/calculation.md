# Bayesian Probability Calculation

This integration uses Bayesian probability to determine the likelihood of an area being occupied based on the current states of configured sensors.

## Core Concept

Instead of a simple binary "motion detected = occupied" logic, this integration calculates a probability score (0% to 100%) representing the confidence that the area is occupied.

## Calculation Steps

### 1. Collect Evidence

Each configured entity reports whether it currently provides evidence of occupancy. Entities that are decaying after recent activity are also treated as evidence.

**Code Reference:** ```115:134:custom_components/area_occupancy/data/entity.py``` (Entity.evidence property)

The evidence collection process:
- Retrieves current state from Home Assistant
- Determines if state indicates activity (based on `active_states` or `active_range`)
- Considers decay: if entity was recently active, decay may still provide evidence
- Returns `True` (active), `False` (inactive), or `None` (unavailable)

**Code Reference:** ```175:212:custom_components/area_occupancy/data/entity.py``` (Entity.has_new_evidence)

When evidence transitions from active to inactive, decay starts automatically.

### 2. Determine Priors

The integration combines the area prior (learned from history) with a time-based prior to form a baseline probability using a weighted average in logit space.

**Code Reference:** ```79:116:custom_components/area_occupancy/data/prior.py``` (Prior.value property)

**Code Reference:** ```160:225:custom_components/area_occupancy/utils.py``` (combine_priors function)

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

**Code Reference:** ```118:132:custom_components/area_occupancy/utils.py```

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

**Code Reference:** ```37:50:custom_components/area_occupancy/data/decay.py``` (Decay.decay_factor)

The decay factor is calculated using exponential decay: `decay_factor = 0.5^(age / half_life)`, where `age` is the time since evidence became inactive.

### 4. Log-Space Combination

The calculation is performed in log space for numerical stability. For each entity the log probabilities for the "occupied" and "not occupied" hypotheses are accumulated and weighted according to the entity type's configured weight.

**Code Reference:** ```55:157:custom_components/area_occupancy/utils.py``` (bayesian_probability function)

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

**Code Reference:** ```146:157:custom_components/area_occupancy/utils.py```

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

The result of this calculation is shown by the **Occupancy Probability** sensor.

**Code Reference:** ```188:201:custom_components/area_occupancy/area/area.py``` (Area.probability)

The **Occupancy Status** binary sensor compares this probability to the configured **Occupancy Threshold** to determine its `on` or `off` state.

**Code Reference:** ```275:281:custom_components/area_occupancy/area/area.py``` (Area.occupied)

## The Maths

The algorithm operates in log space. For each entity, the effective likelihoods (`p_t` for occupied, `p_f` for not occupied) are clamped to avoid extremes. These values are logged and multiplied by the entity's weight:

\[
\log P(O|E) = \log P(O) + \sum (\log p_t \times w)
\]
\[
\log P(\neg O|E) = \log(1-P(O)) + \sum (\log p_f \times w)
\]

where \(w\) is the weight for the entity type. The final probability is then:

\[
P(O|E) = \frac{e^{\log P(O|E)}}{e^{\log P(O|E)} + e^{\log P(\neg O|E)}}
\]

### Why Log Space?

Log space is used for numerical stability. When multiplying many probabilities together (which can be very small), the result can underflow to zero. By working in log space and adding log probabilities, we avoid this issue. The final normalization step converts back to probability space safely.

## Entity Weight Application

Each entity type has a configured weight (0.0-1.0) that determines how much its evidence contributes to the final probability.

**Code Reference:** ```110:112:custom_components/area_occupancy/data/entity.py``` (Entity.weight property)

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

## Decay Interpolation Formula

When [Probability Decay](../features/decay.md) is active, likelihoods are interpolated between their learned values and neutral probabilities based on the decay factor.

**Code Reference:** ```123:128:custom_components/area_occupancy/utils.py```

The interpolation formula:

\[
p_{adjusted} = 0.5 + (p_{learned} - 0.5) \times decay\_factor
\]

Where:
- `p_learned` is the learned likelihood (either `prob_given_true` or `prob_given_false`)
- `decay_factor` ranges from 0.0 (fully decayed) to 1.0 (fresh evidence)
- 0.5 is the neutral probability (no evidence either way)

As decay progresses, the likelihoods move toward 0.5 (neutral), reducing their influence on the final probability.

## Edge Case Handling

The calculation handles several edge cases:

### Unavailable Entities

**Code Reference:** ```110:113:custom_components/area_occupancy/utils.py```

Entities with `None` evidence (unavailable) are skipped unless they're decaying. This prevents unavailable sensors from affecting the calculation.

### Zero Weight Entities

**Code Reference:** ```68:73:custom_components/area_occupancy/utils.py```

Entities with zero weight are excluded from the calculation entirely. They contribute nothing to the final probability.

### Invalid Likelihoods

**Code Reference:** ```75:93:custom_components/area_occupancy/utils.py```

Entities with invalid likelihoods (≤ 0.0 or ≥ 1.0) are excluded. Valid likelihoods must be strictly between 0.0 and 1.0 to avoid log(0) or log(1) errors.

### No Entities

**Code Reference:** ```64:66:custom_components/area_occupancy/utils.py```

If no entities are provided or all are filtered out, the function returns the prior probability (clamped to valid range).

### Numerical Overflow

**Code Reference:** ```151:156:custom_components/area_occupancy/utils.py```

If both probabilities become zero after exponentiation (shouldn't happen but handled defensively), the function returns the prior as a fallback.

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
