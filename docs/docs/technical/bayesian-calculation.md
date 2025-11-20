# Bayesian Calculation Deep Dive

This document provides a detailed mathematical and implementation explanation of the Bayesian probability calculation used for area occupancy detection.

## Mathematical Foundation

### Bayes' Theorem

The calculation is based on Bayes' theorem, which updates prior beliefs with new evidence:

\[
P(Occupied | Evidence) = \frac{P(Evidence | Occupied) \times P(Occupied)}{P(Evidence)}
\]

Where:

- `P(Occupied | Evidence)`: Posterior probability (what we want to calculate)
- `P(Evidence | Occupied)`: Likelihood (how likely is this evidence if occupied)
- `P(Occupied)`: Prior probability (baseline belief)
- `P(Evidence)`: Normalization constant

### Multiple Evidence Sources

With multiple sensors, we combine evidence using the assumption of conditional independence:

\[
P(Occupied | E*1, E_2, ..., E_n) \propto P(Occupied) \times \prod*{i=1}^{n} P(E_i | Occupied)
\]

This means we multiply the prior by the likelihoods from all sensors.

## Log-Space Implementation

### Why Log Space?

Working in log space provides numerical stability when multiplying many probabilities:

**Problems with probability space:**

- Multiplying many small probabilities can underflow to zero
- Very small probabilities lose precision
- Division by normalization constant can overflow

**Benefits of log space:**

- Addition instead of multiplication (more stable)
- Larger dynamic range
- Better precision for small values

### Log-Space Conversion

The calculation converts probabilities to log space:

\[
\log P(Occupied | Evidence) = \log P(Occupied) + \sum\_{i=1}^{n} \log P(E_i | Occupied)
\]

\[
\log P(Not Occupied | Evidence) = \log(1 - P(Occupied)) + \sum\_{i=1}^{n} \log P(E_i | Not Occupied)
\]

### Normalization

After accumulating log probabilities, we normalize back to probability space:

\[
P(Occupied | Evidence) = \frac{e^{\log P(Occupied | Evidence)}}{e^{\log P(Occupied | Evidence)} + e^{\log P(Not Occupied | Evidence)}}
\]

To prevent overflow, we subtract the maximum log value:

\[
\max_log = \max(\log P(Occupied), \log P(Not Occupied))
\]

\[
P(Occupied) = \frac{e^{\log P(Occupied) - \max_log}}{e^{\log P(Occupied) - \max_log} + e^{\log P(Not Occupied) - \max_log}}
\]

## Step-by-Step Calculation

**Code Reference:** `55:157:custom_components/area_occupancy/utils.py` (bayesian_probability function)

### Step 1: Entity Filtering

**Code Reference:** `64:93:custom_components/area_occupancy/utils.py`

Filter out entities that can't contribute:

1. **Empty entities dict**: Return prior if no entities
2. **Zero weight entities**: Exclude entities with `weight == 0.0`
3. **Invalid likelihoods**: Exclude entities where:
   - `prob_given_true <= 0.0` or `>= 1.0`
   - `prob_given_false <= 0.0` or `>= 1.0`

These would cause `log(0)` or `log(1)` errors.

### Step 2: Prior Clamping

**Code Reference:** `95:96:custom_components/area_occupancy/utils.py`

Ensure prior is in valid range:

```python
prior = clamp_probability(prior)  # Clamp to [MIN_PROBABILITY, MAX_PROBABILITY]
```

This prevents `log(0)` or `log(1)` when initializing log probabilities.

### Step 3: Log-Space Initialization

**Code Reference:** `98:100:custom_components/area_occupancy/utils.py`

Initialize log probabilities:

```python
log_true = math.log(prior)
log_false = math.log(1 - prior)
```

These represent the log probabilities for "occupied" and "not occupied" hypotheses before considering any entity evidence.

### Step 4: Entity Processing Loop

**Code Reference:** `102:144:custom_components/area_occupancy/utils.py`

For each entity:

#### 4a. Get Evidence and Decay

```python
value = entity.evidence  # Current evidence (True/False/None)
decay_factor = entity.decay.decay_factor
is_decaying = entity.decay.is_decaying
```

#### 4b. Skip Unavailable Entities

**Code Reference:** `110:113:custom_components/area_occupancy/utils.py`

```python
if value is None and not is_decaying:
    continue  # Skip unavailable entities unless decaying
```

#### 4c. Determine Effective Evidence

**Code Reference:** `115:116:custom_components/area_occupancy/utils.py`

```python
effective_evidence = value or is_decaying
```

Entity provides evidence if currently active OR decaying.

#### 4d. Get Likelihoods

**Code Reference:** `118:132:custom_components/area_occupancy/utils.py`

If effective evidence is present:

```python
p_t = entity.prob_given_true
p_f = entity.prob_given_false

# Apply decay interpolation if decaying
if is_decaying and decay_factor < 1.0:
    neutral_prob = 0.5
    p_t = neutral_prob + (p_t - neutral_prob) * decay_factor
    p_f = neutral_prob + (p_f - neutral_prob) * decay_factor
```

If no effective evidence (inactive sensor):

```python
# Use inverse likelihoods for inactive sensors
# P(Inactive | Occupied) = 1 - P(Active | Occupied)
# P(Inactive | Not Occupied) = 1 - P(Active | Not Occupied)
p_t = 1.0 - entity.prob_given_true
p_f = 1.0 - entity.prob_given_false
```

This ensures inactive sensors provide proper negative evidence. For example, if a sensor is usually active when occupied (`prob_given_true = 0.8`), then when it's inactive, it suggests the area is likely not occupied (`p_t = 0.2`).

#### 4e. Clamp Likelihoods

**Code Reference:** `134:136:custom_components/area_occupancy/utils.py`

```python
p_t = clamp_probability(p_t)  # Prevent log(0) or log(1)
p_f = clamp_probability(p_f)
```

#### 4f. Calculate Weighted Log Contributions

**Code Reference:** `138:144:custom_components/area_occupancy/utils.py`

```python
log_p_t = math.log(p_t)
log_p_f = math.log(p_f)
contribution_true = log_p_t * entity.weight
contribution_false = log_p_f * entity.weight

log_true += contribution_true
log_false += contribution_false
```

The weight multiplies the log contribution, so:

- Weight 1.0: Full contribution
- Weight 0.5: Half contribution
- Weight 0.0: No contribution (filtered out earlier)

### Step 5: Normalization

**Code Reference:** `146:157:custom_components/area_occupancy/utils.py`

Convert back to probability space:

```python
max_log = max(log_true, log_false)
true_prob = math.exp(log_true - max_log)
false_prob = math.exp(log_false - max_log)

total_prob = true_prob + false_prob
if total_prob == 0.0:
    return prior  # Fallback if both probabilities are zero

return true_prob / total_prob
```

The subtraction of `max_log` prevents overflow when exponentiating.

## Inverse Likelihoods for Inactive Sensors

When a sensor is inactive (not providing evidence), the system uses inverse likelihoods instead of neutral probabilities:

**Code Reference:** `129:132:custom_components/area_occupancy/utils.py`

### Why Inverse Likelihoods?

Inactive sensors should provide negative evidence based on their learned behavior:

- If a sensor is usually active when occupied (`prob_given_true` is high), then when it's inactive, it suggests the area is likely not occupied
- If a sensor is rarely active when not occupied (`prob_given_false` is low), then when it's inactive, it's consistent with the area being not occupied

Using neutral probabilities (0.5) for inactive sensors would:

- Dilute the effect of active sensors
- Ignore valuable negative evidence
- Cause incorrect probability calculations when multiple sensors are configured

### Inverse Likelihood Calculation

For inactive sensors:

```
p_t = 1.0 - prob_given_true  # P(Inactive | Occupied)
p_f = 1.0 - prob_given_false  # P(Inactive | Not Occupied)
```

Example:

- Sensor with `prob_given_true = 0.8, prob_given_false = 0.1`
- When active: uses `p_t = 0.8, p_f = 0.1` (strong positive evidence)
- When inactive: uses `p_t = 0.2, p_f = 0.9` (negative evidence, suggests not occupied)

This ensures inactive sensors contribute meaningful evidence to the calculation rather than being neutral.

## Decay Interpolation

When an entity is decaying, its likelihoods are interpolated toward neutral (0.5):

**Code Reference:** `123:128:custom_components/area_occupancy/utils.py`

### Interpolation Formula

\[
p*{adjusted} = 0.5 + (p*{learned} - 0.5) \times decay_factor
\]

Where:

- `p_learned`: The learned likelihood (either `prob_given_true` or `prob_given_false`)
- `decay_factor`: Ranges from 0.0 (fully decayed) to 1.0 (fresh)
- 0.5: Neutral probability (no evidence either way)

### Effect of Decay

As decay progresses:

- `decay_factor` decreases from 1.0 toward 0.0
- Likelihoods move from learned values toward 0.5 (neutral)
- Contribution to probability decreases
- Eventually becomes neutral (no influence)

### Example

Entity with `prob_given_true = 0.9` (very reliable):

- Fresh: `p_t = 0.9` (full strength)
- 50% decayed: `p_t = 0.5 + (0.9 - 0.5) * 0.5 = 0.7` (reduced)
- 90% decayed: `p_t = 0.5 + (0.9 - 0.5) * 0.1 = 0.54` (almost neutral)
- Expired: `p_t = 0.5` (no influence)

## Weight Application

Entity weights determine how much each entity's evidence contributes:

**Code Reference:** `140:141:custom_components/area_occupancy/utils.py`

### Weighted Log Contribution

\[
contribution = \log(p) \times weight
\]

Where:

- `p`: The likelihood probability
- `weight`: Entity weight (0.0-1.0)

### Weight Impact

- **Weight 1.0**: Full contribution (entity fully influences result)
- **Weight 0.5**: Half contribution (moderate influence)
- **Weight 0.0**: No contribution (excluded from calculation)

### Example

Two entities with same likelihoods but different weights:

- Entity A: `p_t = 0.8`, `weight = 1.0` → `contribution = log(0.8) * 1.0 = -0.223`
- Entity B: `p_t = 0.8`, `weight = 0.5` → `contribution = log(0.8) * 0.5 = -0.112`

Entity A has twice the influence of Entity B.

## Complete Example Calculation

Consider an area with:

- Prior: 0.3 (30%)
- Motion sensor: Active, weight 0.85, `p_t = 0.9`, `p_f = 0.1`
- Media player: Inactive, weight 0.70, `p_t = 0.6`, `p_f = 0.2`
- Door sensor: Active, weight 0.25, `p_t = 0.4`, `p_f = 0.3`

### Step 1: Initialize

```
log_true = log(0.3) = -1.204
log_false = log(0.7) = -0.357
```

### Step 2: Process Motion Sensor (Active)

```
p_t = 0.9, p_f = 0.1
log_true += log(0.9) * 0.85 = -1.204 + (-0.105) * 0.85 = -1.293
log_false += log(0.1) * 0.85 = -0.357 + (-2.303) * 0.85 = -2.315
```

### Step 3: Process Media Player (Inactive)

For inactive entities, we use inverse probabilities:

```
p_t = 1 - 0.6 = 0.4  # P(Inactive | Occupied)
p_f = 1 - 0.2 = 0.8  # P(Inactive | Not Occupied)

log_true += log(0.4) * 0.70 = -1.293 + (-0.916) * 0.70 = -1.934
log_false += log(0.8) * 0.70 = -2.315 + (-0.223) * 0.70 = -2.471
```

### Step 4: Process Door Sensor (Active)

```
p_t = 0.4, p_f = 0.3
log_true += log(0.4) * 0.25 = -1.934 + (-0.916) * 0.25 = -2.163
log_false += log(0.3) * 0.25 = -2.471 + (-1.204) * 0.25 = -2.772
```

### Step 5: Normalize

```
max_log = max(-2.163, -2.772) = -2.163
true_prob = exp(-2.163 - (-2.163)) = exp(0) = 1.0
false_prob = exp(-2.772 - (-2.163)) = exp(-0.609) = 0.544
probability = 1.0 / (1.0 + 0.544) = 0.648 (64.8%)
```

The motion sensor's strong positive evidence (active with high `p_t`) significantly increases the probability from 30% to 64.8%.

## Numerical Stability Techniques

### Clamping Probabilities

**Code Reference:** `49:51:custom_components/area_occupancy/utils.py` (clamp_probability)

All probabilities are clamped to `[MIN_PROBABILITY, MAX_PROBABILITY]` to prevent:

- `log(0)` errors (MIN_PROBABILITY > 0)
- `log(1)` errors (MAX_PROBABILITY < 1)

### Max Log Subtraction

**Code Reference:** `147:149:custom_components/area_occupancy/utils.py`

Subtracting the maximum log value before exponentiation prevents overflow:

```python
max_log = max(log_true, log_false)
true_prob = math.exp(log_true - max_log)
```

This ensures at least one exponentiated value is 1.0, preventing overflow.

### Edge Case Handling

**Code Reference:** `151:156:custom_components/area_occupancy/utils.py`

If both probabilities become zero after exponentiation (shouldn't happen but handled defensively):

```python
if total_prob == 0.0:
    return prior  # Fallback to prior
```

## Performance Considerations

### Log Space Efficiency

Log space calculations are computationally efficient:

- Addition instead of multiplication
- Single normalization step at the end
- No intermediate probability calculations

### Entity Filtering

Early filtering of invalid entities reduces computation:

- Zero-weight entities excluded before loop
- Invalid likelihoods excluded before loop
- Unavailable entities skipped in loop

### Caching

Prior values are cached to avoid repeated database queries:

- Time-based priors cached by (day, slot)
- Cache invalidated on update
- Reduces database load during real-time calculations

## See Also

- [Complete Calculation Flow](calculation-flow.md) - End-to-end process
- [Calculation Feature](../features/calculation.md) - User-facing documentation
- [Prior Calculation Deep Dive](prior-calculation.md) - How priors are calculated
- [Likelihood Calculation Deep Dive](likelihood-calculation.md) - How likelihoods are learned
- [Entity Evidence Collection](entity-evidence.md) - How evidence is determined
