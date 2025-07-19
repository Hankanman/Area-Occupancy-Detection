# Bayesian Probability Calculation

This integration uses Bayesian probability to determine the likelihood of an area being occupied based on the current states of configured sensors.

## Core Concept

Instead of a simple binary "motion detected = occupied" logic, this integration calculates a probability score (0% to 100%) representing the confidence that the area is occupied.

It relies on Bayes' Theorem, which mathematically describes how to update a belief (the probability of occupancy) given new evidence (sensor states).

## Calculation Steps

1.  **Identify Active Sensors:** The system first checks which of the configured sensors are currently in an "active" state (e.g., motion sensor `on`, media player `playing`, door sensor `open`, etc.). The definition of "active" is based on the integration's configuration and learned patterns (see [Prior Learning](../features/prior-learning.md)). Sensors that are unavailable or not in an active state are ignored for the current calculation cycle.
2.  **Retrieve Probabilities for Each Active Sensor:** For each active sensor, the system retrieves its associated probabilities, which are learned over time (or use defaults):
    *   **Likelihood `P(Active | Occupied)`:** The probability the sensor is active *given* the area is occupied (`prob_given_true`).
    *   **Likelihood `P(Active | Not Occupied)`:** The probability the sensor is active *given* the area is *not* occupied (`prob_given_false`).
    *   **Prior `P(Occupied)`:** The baseline probability of the area being occupied. This is now determined using a **priority-based system**:
        *   **Time-Based Prior**: If [Time-Based Priors](../features/time-based-priors.md) are enabled and available for the current time slot, use the learned prior for that specific time
        *   **Global Prior**: If no time-based prior exists, fall back to the traditional global prior learned from historical data
        *   **Minimum Prior**: If no learned priors are available, use the minimum prior value (0.01%)
3.  **Calculate Unweighted Posterior Probability (per Sensor):** Using the probabilities above and Bayes' theorem, the system calculates the *posterior probability* of occupancy suggested by *that single active sensor*, *before* considering its weight. This answers: "Based only on this sensor being active and our prior belief, what is the new probability of occupancy?"
4.  **Apply Weight:** The unweighted posterior probability from step 3 is then multiplied by the configured *weight* for its sensor type (e.g., `weight_motion`, `weight_media`). This scales the influence of each sensor type, resulting in a `weighted_probability` for that sensor's evidence.
5.  **Combine Weighted Probabilities (Complementary Method):** The `weighted_probability` values from all *active* sensors are combined to get a single overall probability. This integration uses a complementary probability approach, assuming independence between the weighted sensor evidence:
    *   For each active sensor, it calculates the probability that the area is *not* occupied, given that sensor's weighted evidence: `P(Not Occupied | Sensor Active) = 1 - weighted_probability`.
    *   It multiplies these individual "not occupied" probabilities together: `Combined P(Not Occupied) = Product(1 - weighted_probability)` for all active sensors.
    *   The final result is the complement of this combined "not occupied" probability: `Final P(Occupied) = 1 - Combined P(Not Occupied)`. This yields the combined probability of the area being occupied, considering all active sensor evidence together.
6.  **Apply Bounds:** The final calculated probability is clamped between minimum (e.g., 1%) and maximum (e.g., 99%) values defined in the constants (`MIN_PROBABILITY`, `MAX_PROBABILITY`) to prevent extreme values.

## Output

The result of this calculation (step 6) is the value shown by the **Occupancy Probability** sensor.

The **Occupancy Status** binary sensor compares this probability to the configured **Occupancy Threshold** to determine its `on` or `off` state.

## Time-Based Prior Integration

The integration of [Time-Based Priors](../features/time-based-priors.md) enhances the calculation process by providing context-aware baseline probabilities:

### Time Slot Determination

During each calculation cycle, the system:

1. **Determines Current Time Slot**: Calculates the current day of week (0=Monday, 6=Sunday) and 30-minute time slot (0-47)
2. **Retrieves Time-Based Prior**: Looks up the learned prior probability for that specific time slot
3. **Fallback Strategy**: If no time-based prior exists, falls back to the global prior

### Example Time-Based Calculation

Consider a living room at 8:30 PM on a Friday:

1. **Time Slot**: Friday (4) at 20:30 = Slot 41 (20:30-20:59)
2. **Time-Based Prior**: System retrieves learned prior for Friday 20:30 (e.g., 0.75 or 75%)
3. **Calculation**: Uses this 75% prior instead of the global prior (e.g., 25%)
4. **Result**: Much more accurate baseline for evening occupancy detection

### Benefits

- **Context-Aware Detection**: Understands that occupancy patterns vary by time
- **Reduced False Positives**: Lower baseline probabilities during typically unoccupied periods
- **Enhanced Sensitivity**: Higher baseline probabilities during typically occupied periods
- **Automatic Adaptation**: Learns and adapts to your specific schedule patterns

## The Maths

### Bayesian Update (Single Sensor)

The integration uses Bayes' theorem within the `bayesian_update` function to calculate the unweighted posterior probability of Occupancy (`O`) given that a single sensor is Active (`A`):

\[P(O|A) = \frac{P(A|O) \times P(O)}{P(A)}\]

The denominator, \(P(A)\) (the overall probability of the sensor being active), is expanded using the law of total probability:

\[P(A) = P(A|O) P(O) + P(A|\neg O) P(\neg O)\]

Where:
- \(P(O|A)\) is the posterior probability of occupancy given the sensor is active (the result of `bayesian_update`).
- \(P(A|O)\) is the likelihood the sensor is active given occupancy (`prob_given_true`).
- \(P(O)\) is the prior probability of occupancy. This is now determined by the **priority system**:
  - \(P(O)_{time}\) = Time-based prior for current time slot (if available)
  - \(P(O)_{global}\) = Global prior from historical analysis
  - \(P(O)_{min}\) = Minimum prior value (0.01%)
- \(P(A|\neg O)\) is the likelihood the sensor is active given the area is *not* occupied (`prob_given_false`).
- \(P(\neg O)\) is the prior probability of the area *not* being occupied (\(1 - P(O)\)).

Substituting the expanded \(P(A)\) into Bayes' theorem gives the formula used in the code:
\[
P(O|A) = \frac{P(A|O) P(O)}{P(A|O) P(O) + P(A|\neg O) (1 - P(O))}
\]

### Prior Selection Logic

The prior \(P(O)\) is selected using this priority order:

\[
P(O) = \begin{cases}
P(O)_{time} & \text{if time-based prior available and } P(O)_{time} \geq P(O)_{min} \\
P(O)_{global} & \text{if global prior available and } P(O)_{global} \geq P(O)_{min} \\
P(O)_{min} & \text{otherwise}
\end{cases}
\]

### Weighting

The result \(P(O|A)\) is then weighted **after** the Bayesian update:
\[
P_{weighted} = P(O|A) \times W_{type}\]
Where \(W_{type}\) is the configured weight for the sensor's type. The
likelihood values themselves remain unweighted; each sensor's posterior is
scaled only at this stage.

### Combining Probabilities (Complementary Method)

To combine the weighted probabilities (\(P_{weighted, i}\)) from multiple active sensors (\(i = 1 \text{ to } n\)), the complementary method is used:

1. Calculate the probability of *not* being occupied suggested by each sensor: \(1 - P_{weighted, i}\)
2. Multiply these probabilities together: \(\prod_{i=1}^{n} (1 - P_{weighted, i})\)
3. Subtract the result from 1 to get the final combined probability:
\[
P_{final} = 1 - \prod_{i=1}^{n} (1 - P_{weighted, i})
\]

### Decay Calculation

When [Probability Decay](../features/decay.md) is active, the probability \(P_{decayed}\) at a given time is calculated based on the probability when decay started (\(P_{start}\)) and the time elapsed since decay began (\(t_{elapsed}\)).

The core decay factor (\(f_{decay}\)) is calculated using an exponential function:

\[
f_{decay} = e^{ -\lambda \times \frac{t_{elapsed}}{T_{window}} }
\]

Where:
- \(e\) is the base of the natural logarithm (Euler's number).
- \(\lambda\) (`DECAY_LAMBDA`) is a decay constant that influences the steepness of the decay curve.
- \(t_{elapsed}\) is the time in seconds since the decay process started.
- \(T_{window}\) (`decay_window`) is the configured time window in seconds over which the decay primarily occurs.

The decayed probability is then calculated by applying this factor to the starting probability, ensuring it stays within the defined bounds:
\[
P_{decayed} = \max( P_{min}, \min( P_{start} \times f_{decay}, P_{max} ) )
\]

Where:
- \(P_{start}\) is the probability value recorded just before decay began.
- \(P_{min}\) (`MIN_PROBABILITY`) is the minimum allowed probability (e.g., 0.01).
- \(P_{max}\) (`MAX_PROBABILITY`) is the maximum allowed probability (e.g., 0.99).

This ensures the probability decays exponentially but is clamped within reasonable limits.