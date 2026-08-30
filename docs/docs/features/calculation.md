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

### 3. Combine Evidence in Logit Space

Rather than adjusting per-entity likelihoods and running a Bayes'-theorem-style update, the integration builds one running total in **logit (log-odds) space**, starting from the prior and adding one weighted term per entity:

```
z = logit(prior)
for each entity with weight > 0:
    evidence = 1.0 if active
             else entity.decay_factor if decaying
             else 0.0   # inactive contributes nothing, not negative evidence
    strength = entity.prob_given_true * entity.type.strength_multiplier
    z += entity.effective_weight * evidence * correlation * strength

probability = sigmoid(z) = 1 / (1 + exp(-z))
```

Where `correlation` is a learned 0.0-1.0 multiplier from the correlation-analysis pipeline (defaults to `1.0` until enough data exists), and `strength_multiplier` is a per-`InputType` constant (`3.0` for motion, `2.0` for most other types) that gives ground-truth-quality sensors a stronger say.

An **inactive** entity contributes exactly `0.0` — it drops out of the sum entirely rather than pushing the probability down. This is a deliberate difference from a classic Bayesian likelihood-ratio update, where an inactive sensor that's usually active when occupied would actively count *against* occupancy.

A **decaying** entity's evidence value is the decay factor itself (`1.0` fresh, fading toward `0.0`), so its whole contribution shrinks smoothly toward zero as decay progresses — see [Decay Interpolation](#decay-interpolation) below.

### 4. Final Probability

`z` is passed through the sigmoid function and clamped to `[MIN_PROBABILITY, MAX_PROBABILITY]`:

```
probability = clamp(sigmoid(z))
```

Because contributions are summed additively before a single sigmoid call, there's no separate "occupied" vs. "not occupied" running total to normalize between — the sigmoid's output is already a valid probability in `(0, 1)`.

## Dual-Model Approach: Presence + Environmental

The integration splits sensor evidence into two independent models before combining them:

### Presence Confidence

Filters to **presence-related sensor types** — motion, media, appliance, door, window, cover, power, and sleep — and calculates a probability using the sigmoid model. This represents the "hard evidence" of someone being in the area.

The result is exposed as the **Presence Confidence** diagnostic sensor.

### Environmental Confidence

Filters to **environmental sensor types** (temperature, humidity, illuminance, CO2, etc.) and calculates a 0-1 confidence score. A value of **50% is neutral** (no environmental influence), values above 50% support occupancy, and values below 50% oppose it.

The result is exposed as the **Environmental Confidence** diagnostic sensor.

### Combined Probability (Occupancy Probability)

Environmental data is applied as an **additive update** to the presence estimate in **logit space**, damped to 20% of its own contribution:

```text
z_combined = logit(presence) + 0.2 × logit(environmental)
probability = sigmoid(z_combined)
```

Presence indicators set the level; environmental data nudges it. Because environmental contributions are non-negative by construction (a sensor is either inside its active range or contributes nothing), environmental confidence never falls below 50% in practice, so environmental data can only raise the probability or leave it unchanged.

An earlier version of this formula averaged the two channels (`0.8 × logit(presence) + 0.2 × logit(environmental)`). Averaging pulls the result toward whichever channel is less confident, and the environmental channel is structurally low-confidence, so adding a *supporting* environmental sensor could lower the probability — one active motion sensor gave 94.5%, and adding a CO2 sensor reading a normal 450 ppm dropped it to 90.8%. The additive form removes that.

## Output

The result of this calculation is shown by the **Occupancy Probability** sensor, which displays the calculated probability as a percentage (0% to 100%).

The **Occupancy Status** binary sensor compares this probability to the configured **Occupancy Threshold** to determine its `on` or `off` state. When the probability equals or exceeds the threshold, the status sensor turns `on`.

## Mathematical Foundation

The calculation combines a learned prior with sensor evidence in **logit (log-odds) space**: each entity adds one weighted, additive term to a running sum, and a single `sigmoid()` call converts that sum back to a probability at the end. This is not classic Bayes'-theorem likelihood-ratio multiplication — see [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md) for the full formula, derivation, and a worked example.

## Entity Weight Application

Each entity type has a configured weight (0.0-1.0, `effective_weight` after adjustment for learned information gain) that scales how much its evidence contributes to the running logit-space sum.

- Weight 1.0: Full contribution
- Weight 0.5: Half contribution
- Weight 0.0: No contribution (entity is excluded from calculation)

Default weights by entity type:

- Motion sensors: 1.00 (highest reliability, ground truth)
- Sleep: 0.90 (very high reliability)
- Media players: 0.85 (high reliability)
- Wasp in Box: 0.80 (high reliability)
- Wi-Fi client count: 0.35 (moderate reliability)
- Cover sensors: 0.50 (moderate reliability)
- Appliances: 0.40 (moderate reliability)
- Door sensors: 0.30 (lower reliability)
- Lock: 0.30 (lower reliability)
- Power sensors: 0.30 (lower reliability)
- Window sensors: 0.20 (low reliability)
- Environmental sensors: 0.10 (very low reliability) - includes temperature, humidity, illuminance, CO2, sound pressure, atmospheric pressure, air quality, VOC, PM2.5, and PM10 sensors

## Decay Interpolation

When [Probability Decay](../features/decay.md) is active, a decaying entity's evidence value is the decay factor itself (`1.0` fresh, fading toward `0.0`), so its whole logit-space contribution shrinks smoothly toward zero as decay progresses — not an interpolation of the likelihood toward a neutral value. For the mathematical formula, see [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md#decay-interpolation).

## Edge Case Handling

The calculation handles several edge cases to ensure robust operation:

- **Unavailable Entities**: Entities with unavailable states are skipped unless they're decaying
- **Zero Weight Entities**: Entities with zero weight are excluded from the calculation
- **No Entities**: If no entities are available, the calculation returns the prior probability
- **Numerical Stability**: The system uses various techniques to prevent numerical overflow and underflow

For detailed technical information about edge case handling, see [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md#numerical-stability-techniques).

## Example Calculation

Consider an area with:

- Prior: 0.3 (30% baseline occupancy)
- Motion sensor: Active, weight 1.0, `prob_given_true=0.95`, `strength_multiplier=3.0`
- Media player: Inactive (not decaying), weight 0.85, `prob_given_true=0.65`, `strength_multiplier=2.0`
- Door sensor: Active, weight 0.3, `prob_given_true=0.2`, `strength_multiplier=2.0`

Assume no learned correlations yet (multiplier `1.0` for all).

Step 1: Initialize from the prior

```
z = logit(0.3) = ln(0.3 / 0.7) ≈ -0.8473
```

Step 2: Motion sensor (active)

```
1.0 * 1.0 * 1.0 * (0.95 * 3.0) = 2.85
z ≈ 2.0027
```

Step 3: Media player (inactive, not decaying → evidence = 0)

```
0.85 * 0.0 * 1.0 * (0.65 * 2.0) = 0
z ≈ 2.0027 (unchanged) — the inactive sensor drops out entirely, it does not count against occupancy
```

Step 4: Door sensor (active)

```
0.3 * 1.0 * 1.0 * (0.2 * 2.0) = 0.12
z ≈ 2.1227
```

Step 5: Sigmoid

```
probability = sigmoid(2.1227) ≈ 0.8931 (89.3%)
```

The active, high-reliability motion sensor dominates (`+2.85` in logit space, vs. the door's `+0.12`), pulling the result from the 30% prior up to ~89%. The inactive media player contributes nothing to the result, rather than pulling it down.

## See Also

- [Complete Calculation Flow](../technical/calculation-flow.md) - End-to-end process explanation
- [Bayesian Calculation Deep Dive](../technical/bayesian-calculation.md) - Detailed mathematical explanation
- [Prior Learning](../features/prior-learning.md) - How priors are learned from history
- [Likelihood Learning](../features/likelihood.md) - How likelihoods are learned
- [Decay Feature](../features/decay.md) - Decay mechanism overview
- [Entity Evidence Collection](../technical/entity-evidence.md) - How evidence is determined
