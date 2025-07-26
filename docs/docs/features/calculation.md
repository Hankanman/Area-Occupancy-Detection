# Bayesian Probability Calculation

This integration uses Bayesian probability to determine the likelihood of an area being occupied based on the current states of configured sensors.

## Core Concept

Instead of a simple binary "motion detected = occupied" logic, this integration calculates a probability score (0% to 100%) representing the confidence that the area is occupied.

It relies on Bayes' Theorem, which mathematically describes how to update a belief (the probability of occupancy) given new evidence (sensor states).

## Calculation Steps

1.  **Identify Active Sensors:** The system first checks which of the configured sensors are currently in an "active" state (e.g., motion sensor `on`, media player `playing`, door sensor `open`, etc.). The definition of "active" is based on the integration's configuration and learned patterns (see [Prior Learning](../features/prior-learning.md)). Sensors that are unavailable or not in an active state are ignored for the current calculation cycle.
2.  **Retrieve Probabilities for Each Active Sensor:** For each active sensor, the system retrieves its associated probabilities, which are learned over time (or use defaults):
    - **Likelihood `P(Active | Occupied)`:** The probability the sensor is active _given_ the area is occupied (`prob_given_true`).
    - **Likelihood `P(Active | Not Occupied)`:** The probability the sensor is active _given_ the area is _not_ occupied (`prob_given_false`).
    - **Prior `P(Occupied)`:** The baseline probability of the area being occupied. This is determined from historical data analysis:
      - **Global Prior**: Calculated as the maximum of two historical analyses: primary sensors prior (motion sensors + wasp if enabled) and occupancy entity prior, then multiplied by a factor of 1.2
      - **Minimum Prior**: If no learned prior is available, use the minimum prior value (10%)
3.  **Calculate Unweighted Posterior Probability (per Sensor):** Using the probabilities above and Bayes' theorem, the system calculates the _posterior probability_ of occupancy suggested by _that single active sensor_, _before_ considering its weight. This answers: "Based only on this sensor being active and our prior belief, what is the new probability of occupancy?"
4.  **Apply Weight:** The unweighted posterior probability from step 3 is then multiplied by the configured _weight_ for its sensor type (e.g., `weight_motion`, `weight_media`). This scales the influence of each sensor type, resulting in a `weighted_probability` for that sensor's evidence.
5.  **Combine Weighted Probabilities (Complementary Method):** The `weighted_probability` values from all _active_ sensors are combined to get a single overall probability. This integration uses a complementary probability approach, assuming independence between the weighted sensor evidence:
    - For each active sensor, it calculates the probability that the area is _not_ occupied, given that sensor's weighted evidence: `P(Not Occupied | Sensor Active) = 1 - weighted_probability`.
    - It multiplies these individual "not occupied" probabilities together: `Combined P(Not Occupied) = Product(1 - weighted_probability)` for all active sensors.
    - The final result is the complement of this combined "not occupied" probability: `Final P(Occupied) = 1 - Combined P(Not Occupied)`. This yields the combined probability of the area being occupied, considering all active sensor evidence together.
6.  **Apply Bounds:** Individual probability calculations are bounded between 0.1% and 99.9% (0.001 to 0.999) during Bayesian updates to prevent extreme values and ensure numerical stability.

## Output

The result of this calculation (step 6) is the value shown by the **Occupancy Probability** sensor.

The **Occupancy Status** binary sensor compares this probability to the configured **Occupancy Threshold** to determine its `on` or `off` state.

## The Maths

### Bayesian Update (Single Sensor)

The integration uses Bayes' theorem within the `bayesian_update` function to calculate the unweighted posterior probability of Occupancy (`O`) given that a single sensor is Active (`A`):

\[P(O|A) = \frac{P(A|O) \times P(O)}{P(A)}\]

The denominator, \(P(A)\) (the overall probability of the sensor being active), is expanded using the law of total probability:

\[P(A) = P(A|O) P(O) + P(A|\neg O) P(\neg O)\]

Where:

- \(P(O|A)\) is the posterior probability of occupancy given the sensor is active (the result of `bayesian_update`).
- \(P(A|O)\) is the likelihood the sensor is active given occupancy (`prob_given_true`).
- \(P(O)\) is the prior probability of occupancy, determined by:
  - \(P(O)\_{global}\) = Global prior from historical analysis (multiplied by factor 1.2)
  - \(P(O)\_{min}\) = Minimum prior value (10%)
- \(P(A|\neg O)\) is the likelihood the sensor is active given the area is _not_ occupied (`prob_given_false`).
- \(P(\neg O)\) is the prior probability of the area _not_ being occupied (\(1 - P(O)\)).

Substituting the expanded \(P(A)\) into Bayes' theorem gives the formula used in the code:

\[
P(O|A) = \frac{P(A|O) P(O)}{P(A|O) P(O) + P(A|\neg O) (1 - P(O))}
\]

### Prior Selection Logic

The prior \(P(O)\) is calculated using this logic:

\[
P(O) = \begin{cases}
\min(\max(P(O)_{global} \times 1.2, P(O)_{min}), P(O)_{max}) & \text{if global prior available} \\
P(O)_{min} & \text{otherwise}
\end{cases}
\]

Where \(P(O)\_{global}\) is the maximum of:

- Primary sensors prior (calculated from motion sensors and wasp entity if enabled)
- Occupancy entity prior (calculated from the occupancy entity's historical data)

### Weighting

The result \(P(O|A)\) is then weighted **after** the Bayesian update:

\[
P*{weighted} = P(O|A) \times W*{type}\]

Where \(W\_{type}\) is the configured weight for the sensor's type. The
likelihood values themselves remain unweighted; each sensor's posterior is
scaled only at this stage.

### Combining Probabilities (Complementary Method)

To combine the weighted probabilities (\(P\_{weighted, i}\)) from multiple active sensors (\(i = 1 \text{ to } n\)), the complementary method is used:

1. Calculate the probability of _not_ being occupied suggested by each sensor: \(1 - P\_{weighted, i}\)
2. Multiply these probabilities together: \(\prod*{i=1}^{n} (1 - P*{weighted, i})\)
3. Subtract the result from 1 to get the final combined probability: \(P*{final} = 1 - \prod*{i=1}^{n} (1 - P\_{weighted, i})\)

### Decay Calculation

When [Probability Decay](../features/decay.md) is active, the probability \(P*{decayed}\) at a given time is calculated based on the probability when decay started (\(P*{start}\)) and the time elapsed since decay began (\(t\_{elapsed}\)).

The core decay factor (\(f\_{decay}\)) is calculated using an exponential function:

\[
f*{decay} = e^{ -\lambda \times \frac{t*{elapsed}}{T\_{window}} }
\]

Where:

- \(e\) is the base of the natural logarithm (Euler's number).
- \(\lambda\) (`DECAY_LAMBDA`) is a decay constant that influences the steepness of the decay curve.
- \(t\_{elapsed}\) is the time in seconds since the decay process started.
- \(T\_{window}\) (`decay_window`) is the configured time window in seconds over which the decay primarily occurs.

The decayed probability is then calculated by applying this factor to the starting probability, ensuring it stays within the defined bounds:

\[
P*{decayed} = \max( P*{min}, \min( P*{start} \times f*{decay}, P\_{max} ) )
\]

Where:

- \(P\_{start}\) is the probability value recorded just before decay began.
- \(P\_{min}\) is the minimum allowed probability (0.001).
- \(P\_{max}\) is the maximum allowed probability (0.999).

This ensures the probability decays exponentially but is clamped within reasonable limits.
