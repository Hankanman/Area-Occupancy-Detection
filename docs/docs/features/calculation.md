# Bayesian Probability Calculation

This integration uses Bayesian probability to determine the likelihood of an area being occupied based on the current states of configured sensors.

## Core Concept

Instead of a simple binary "motion detected = occupied" logic, this integration calculates a probability score (0% to 100%) representing the confidence that the area is occupied.

## Calculation Steps

1. **Collect Evidence:** Each configured entity reports whether it currently provides evidence of occupancy. Entities that are decaying after recent activity are also treated as evidence.
2. **Determine Priors:** The integration combines the area prior (learned from history) with a time-based prior to form a baseline probability using a weighted average in logit space.
3. **Adjust Likelihoods:** For each entity, the learned likelihoods `P(Active | Occupied)` and `P(Active | Not Occupied)` are adjusted for any active decay to reduce the strength of stale evidence.
4. **Log-Space Combination:** The calculation is performed in log space for numerical stability. For each entity the log probabilities for the "occupied" and "not occupied" hypotheses are accumulated and weighted according to the entity type's configured weight.
5. **Final Probability:** The log probabilities are exponentiated and normalised to produce the final occupancy probability.

## Output

The result of this calculation is shown by the **Occupancy Probability** sensor.

The **Occupancy Status** binary sensor compares this probability to the configured **Occupancy Threshold** to determine its `on` or `off` state.

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

## Decay Calculation

When [Probability Decay](../features/decay.md) is active, likelihoods are interpolated between their learned values and neutral probabilities based on the decay factor. This reduces the influence of old evidence until it expires.
