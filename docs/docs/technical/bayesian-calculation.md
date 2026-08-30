# Bayesian Calculation Deep Dive

This document provides a detailed mathematical explanation of the probability calculation used for area occupancy detection: a **sigmoid/logit-space model** that combines a learned prior with per-sensor evidence. "Bayesian" here refers to the general shape of the problem — start from a prior belief, update it with evidence — not to classic Bayes'-theorem odds multiplication. The engine is `sigmoid_probability()` in `custom_components/area_occupancy/utils.py`, reached via `presence_probability()` and `environmental_confidence()`.

!!! note "Not naive Bayes"
This integration previously shipped a classic log-odds naive-Bayes accumulator (`bayesian_probability()`). It was superseded by the model below in PR #353 (2026.2.1) and removed entirely in PR #529 (2026.8.1) after being confirmed to have zero production call sites. If you're looking at older material (a cached doc, a stale comment) that describes multiplying likelihoods and renormalizing, it's describing the removed function, not current behavior.

## Mathematical Foundation

### The Prior as a Log-Odds Bias

The learned prior (global prior combined with time-of-day prior — see [Global Prior Flow](global-prior-flow.md) and [Time Prior Flow](time-prior-flow.md)) is converted to **logit space** and used as the starting point, or bias, for the calculation:

\[
bias = \text{logit}(prior) = \ln\left(\frac{prior}{1 - prior}\right)
\]

A prior of 0.5 (no lean either way) gives a bias of 0. A prior of 0.3 gives a bias of about -0.85 (below the midline, i.e. leaning "not occupied" before any sensor evidence is considered).

### Additive Evidence in Logit Space

Each sensor then adds a **weighted term** to that starting value:

\[
z = bias + \sum_{i=1}^{n} \left( w_i \times e_i \times c_i \times s_i \right)
\]

\[
P(Occupied) = \text{sigmoid}(z) = \frac{1}{1 + e^{-z}}
\]

Where, for entity \(i\):

- \(w_i\): **effective weight** (`entity.effective_weight`, falling back to `entity.weight`) — the entity's configured weight, reduced for sensors with low learned information gain
- \(e_i\): **evidence** — `1.0` if actively providing evidence, a fractional **decay factor** (0.0–1.0) if decaying from recent activity, or `0.0` if inactive and not decaying
- \(c_i\): **correlation** — a learned 0.0–1.0 multiplier from `db/correlation.py`'s statistical analysis (defaults to `1.0` if no correlation data exists yet)
- \(s_i\): **strength factor** — `prob_given_true × strength_multiplier`, i.e. the sensor's learned reliability (`prob_given_true`) scaled by a per-`InputType` constant (`3.0` for motion, `2.0` for most other types — see `DEFAULT_TYPES` in `data/entity_type.py`) that gives ground-truth-quality sensors a stronger say

## Why Logit Space?

Working in logit (log-odds) space instead of raw probability gives the same numerical-stability benefits classic Bayesian log-space combination has, plus a simpler combination rule:

- **Purely additive**: each sensor is one term added to a running sum — no renormalization step, no multiplying probabilities together
- **Order independence**: summation is commutative, so entity iteration order can never change the result
- **Unbounded contribution range**: a very strong or very weak sensor can push `z` arbitrarily far from 0 without the intermediate math overflowing, since the sigmoid squashes the *final* sum back into `(0, 1)`
- **No likelihood-ratio bookkeeping**: unlike naive Bayes, there's no need to track separate "occupied" and "not occupied" running totals and normalize between them at the end — one running sum, one sigmoid call

## Step-by-Step Calculation

### Step 1: Initialize from the Prior

\[
z = \text{logit}(prior)
\]

The prior is clamped to `[MIN_PROBABILITY, MAX_PROBABILITY]` (0.01–0.99) first, so `logit()` never sees exactly 0 or 1.

### Step 2: Entity Loop

For each entity with `weight > 0`:

1. **Determine evidence** (\(e_i\)):
   - Actively providing evidence (`entity.evidence is True`) → `1.0`
   - Not currently active, but decaying from recent activity (`entity.decay.is_decaying`) → `entity.decay_factor`, which fades from `1.0` toward `0.0` over the decay window
   - Neither → `0.0` — an inactive sensor contributes **nothing**, not negative evidence. This is a deliberate difference from a classic naive-Bayes formulation, where an inactive sensor with a high `prob_given_true` would actively push the posterior *down*. Here it simply drops out of the sum.
2. **Look up correlation** (\(c_i\)): the entity's learned correlation strength, or `1.0` if none is available yet.
3. **Compute the strength factor** (\(s_i\)): `entity.prob_given_true * entity.type.strength_multiplier`.
4. **Accumulate**: `z += effective_weight * evidence * correlation * strength_factor`.

### Step 3: Convert Back to Probability

\[
P(Occupied) = \text{clamp}\big(\text{sigmoid}(z)\big)
\]

`sigmoid()` branches on the sign of `z` internally (computing `1/(1+e^{-z})` for `z >= 0` and `e^{z}/(1+e^{z})` for `z < 0`) so it never evaluates `e^{x}` for a large positive `x`, avoiding overflow. The result is then clamped to `[MIN_PROBABILITY, MAX_PROBABILITY]`.

## Decay Interpolation

While an entity is decaying (recently active, now inactive, within its half-life window), its evidence value is the **decay factor itself** — not the learned likelihood interpolated toward neutral, and not the raw `prob_given_true`:

\[
e_i = decay\_factor \quad (\text{ranges } 1.0 \to 0.0 \text{ as decay progresses})
\]

So a decaying entity's whole contribution — `effective_weight × decay_factor × correlation × strength_factor` — shrinks toward zero as `decay_factor` shrinks toward zero. There's no intermediate "neutral 0.5" state to interpolate through; the contribution just fades out. See [`data/decay.py`](../features/decay.md) for how `decay_factor` itself is computed from elapsed time and half-life.

## Weight Application

`effective_weight` (falling back to plain `weight` when unset) scales each entity's whole contribution linearly:

- **Full weight** (1.0): the entity's evidence/correlation/strength product enters `z` unscaled
- **Half weight** (0.5): half the contribution
- **Zero weight**: the entity is skipped before evidence is even evaluated

Because contributions are additive in logit space rather than multiplicative in probability space, doubling a weight roughly doubles that entity's *logit-space* pull — the effect on the final `sigmoid()`-transformed probability is nonlinear (compressed near 0 and 1, closer to linear near 0.5).

## Complete Example Calculation

Consider an area with prior `0.3` and three sensors:

| Sensor | State | Weight | `prob_given_true` | `strength_multiplier` |
|---|---|---|---|---|
| Motion | Active | 1.0 | 0.95 | 3.0 |
| Media player | Inactive (not decaying) | 0.85 | 0.65 | 2.0 |
| Door | Active | 0.3 | 0.2 | 2.0 |

Assume no learned correlations yet (`c_i = 1.0` for all).

**Step 1 — bias from the prior:**

\[
z = \text{logit}(0.3) = \ln(0.3 / 0.7) \approx -0.8473
\]

**Step 2 — motion (active):**

\[
1.0 \times 1.0 \times 1.0 \times (0.95 \times 3.0) = 2.85 \qquad z \approx 2.0027
\]

**Step 3 — media player (inactive, not decaying → evidence = 0):**

\[
0.85 \times 0.0 \times 1.0 \times (0.65 \times 2.0) = 0 \qquad z \approx 2.0027 \text{ (unchanged)}
\]

**Step 4 — door (active):**

\[
0.3 \times 1.0 \times 1.0 \times (0.2 \times 2.0) = 0.12 \qquad z \approx 2.1227
\]

**Step 5 — sigmoid:**

\[
P(Occupied) = \text{sigmoid}(2.1227) \approx 0.8931 \; (89.3\%)
\]

The active, high-reliability motion sensor dominates (contributing `+2.85` in logit space, vs. the door's `+0.12`), pulling the result from a 30% prior up to ~89%. The inactive media player contributes nothing — it neither reinforces nor opposes the motion evidence.

## Numerical Stability Techniques

### Probability Clamping

Every probability that flows through `logit()` — the prior, and the final result — is clamped to `[MIN_PROBABILITY, MAX_PROBABILITY]` (`clamp_probability()`, default `0.01`–`0.99`) so `ln(p/(1-p))` never divides by zero or takes `ln(0)`.

### Sign-Branched Sigmoid

`sigmoid(z)` computes `1/(1+e^{-z})` when `z >= 0` and the algebraically equivalent `e^{z}/(1+e^{z})` when `z < 0`, so `math.exp()` is only ever called with a non-positive argument — no overflow risk regardless of how large `z` grows from a long list of strong sensors.

### Edge Case Handling

An empty `entities` dict, or every entity having `weight <= 0`, returns the (clamped) prior unchanged rather than dividing by zero or looping over nothing meaningfully.

## Performance Considerations

- **Single pass, O(n)**: one loop over entities accumulating a running sum — no separate "true" and "false" accumulators to maintain in parallel, and no final normalization/division step (`sigmoid()` is already bounded).
- **Correlation lookups**: correlation strengths are precomputed by the hourly analysis pipeline (`db/correlation.py`) and passed in as a plain dict, not queried per-entity during the hot path.
- **Prior caching**: time-based priors are cached by `(day_of_week, hour_slot)` and invalidated on update, so the real-time calculation never issues a database query.

## See Also

- [Complete Calculation Flow](calculation-flow.md) - End-to-end process
- [Calculation Feature](../features/calculation.md) - User-facing documentation
- [Prior Calculation Deep Dive](../features/prior-learning.md) - How priors are calculated
- [Likelihood Calculation Deep Dive](likelihood-calculation.md) - How likelihoods are learned
