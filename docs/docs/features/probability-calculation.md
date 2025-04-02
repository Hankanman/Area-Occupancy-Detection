# Probability Calculation

The Area Occupancy Detection integration uses Bayesian probability calculations to determine the likelihood of area occupancy. This page explains how these calculations work and what factors influence them.

## Basic Concepts

### Bayesian Probability

The integration uses Bayes' theorem to update the probability of occupancy based on sensor inputs:

\[P(O|E) = \frac{P(E|O) \times P(O)}{P(E)}\]

Where:
- \(P(O|E)\) is the probability of occupancy given the evidence
- \(P(E|O)\) is the probability of the evidence given occupancy
- \(P(O)\) is the prior probability of occupancy
- \(P(E)\) is the total probability of the evidence

### Probability Bounds

To prevent extreme certainty and maintain system responsiveness:
- Minimum probability: 1%
- Maximum probability: 99%

These bounds ensure the system can always adjust its estimate based on new evidence.

## Prior Probabilities

### Default Values

- Base prior probability: 17.13%
  - This represents a typical home occupancy pattern
  - Can be adjusted through historical learning
- Probability given true state: 50%
  - When a sensor indicates occupancy
- Probability given false state: 10%
  - When a sensor indicates no occupancy

### Historical Learning

The integration learns prior probabilities by:

1. Analyzing sensor history (1-30 days, default 7)
2. Comparing sensor states with primary occupancy sensor
3. Calculating correlation between states and known occupancy
4. Caching results for 6 hours to improve performance

## Time Decay

### Decay Function

The integration uses an exponential decay function:

\[P(t) = P_0 \times e^{-\lambda t}\]

Where:
- \(P(t)\) is the probability at time t
- \(P_0\) is the initial probability
- \(\lambda\) = 0.866433976 (decay constant)
- \(t\) is time since last sensor update

### Decay Behavior

- Starts after minimum delay (default: 60 seconds)
- Reduces to 25% at half of decay window
- Example: With 600s window
  - No decay 0-60s
  - At 300s: 25% of original probability
  - At 600s: ~6% of original probability

## Composite Calculation

The final probability is calculated by:

1. Starting with the prior probability
2. Applying Bayes' theorem for each sensor
3. Weighting results by sensor type
4. Applying time decay if enabled
5. Enforcing probability bounds
6. Updating dependent sensors

## Example Calculation

Consider a room with:

- Motion sensor (off)
- TV playing
- Light on
- 5 minutes since last motion

The calculation would:

1. Start with 17.13% prior
2. Adjust for motion (off): ~10%
3. Adjust for TV (playing): ~70%
4. Adjust for light (on): ~20%
5. Apply 5-minute decay
6. Bound result between 1-99%

The result provides a balanced probability based on all available evidence. 