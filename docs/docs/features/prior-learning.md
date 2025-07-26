# Prior Probability Learning

To make the Bayesian calculations more accurate and specific to your environment, the integration automatically learns key probability values from your sensor history. This process adapts the system to how *you* actually use the space.

## What is a Prior Probability?

In the context of Bayesian probability, the **prior probability** (often denoted as `P(Occupied)` or simply `prior`) represents our initial belief about the likelihood of an event *before* considering any new evidence. For this integration, it's the baseline probability that the area is occupied, independent of the *current* state of the sensors.

Think of it as the starting point for the calculation. When new sensor evidence comes in (e.g., motion is detected), Bayes' theorem uses this prior belief along with the sensor's learned likelihoods to calculate an updated belief (the posterior probability).

Initially, the integration uses default prior values. However, the **Prior Learning** process aims to calculate a more accurate prior based on the historical behavior of your **Primary Occupancy Sensor**, reflecting how often that area is typically occupied according to your chosen ground truth sensor.

## Types of Prior Learning

The integration now supports two types of prior learning:

### 1. Global Prior Learning (Traditional)

The traditional approach calculates a single baseline prior probability for the entire area, regardless of time. This is the foundation of the learning system and provides a good baseline for occupancy detection.

### 2. Time-Based Prior Learning (Advanced)

The new [Time-Based Priors](../features/time-based-priors.md) feature extends prior learning by calculating separate occupancy probabilities for specific time slots throughout the day and week. This provides much more accurate occupancy detection by understanding when areas are typically occupied during different periods.

**Key Differences:**
- **Global Prior**: Single probability for the entire area (e.g., "Living room is occupied 25% of the time")
- **Time-Based Priors**: 336 separate probabilities (7 days × 48 time slots) for specific times (e.g., "Living room is occupied 75% of the time on Monday evenings")

## Goal of Learning

The primary goal is to determine, for each configured sensor:

1.  **`P(Sensor Active | Area Occupied)` (Likelihood - True):** How likely is this sensor to be in its "active" state when the area is genuinely occupied?
2.  **`P(Sensor Active | Area Not Occupied)` (Likelihood - False):** How likely is this sensor to be in its "active" state when the area is *not* occupied? (This helps identify sensors that trigger falsely or independently of occupancy).
3.  **`P(Sensor Occupied)` (Prior):** What is the baseline probability that this sensor is active, based on its history? (This is particularly relevant for the primary sensor, which informs the overall prior).

## The Primary Sensor: Ground Truth

This learning process relies heavily on the **Primary Occupancy Sensor** you designate during configuration. This sensor (usually a reliable motion or dedicated occupancy sensor) is treated as the "ground truth" indicator of when the area was actually occupied during the historical analysis period.

## How Learning Works

### Global Prior Learning Process

1.  **Time Period:** The system looks back over the configured **History Period** (e.g., the last 7 days) using the Home Assistant recorder database.
2.  **Data Retrieval:** For the specified period, it fetches the state history for both the **Primary Occupancy Sensor** and the specific **sensor being analyzed**.
3.  **Interval Analysis:** The state histories are converted into time intervals, noting the start and end times for each state (e.g., when a sensor turned `on` and then `off`).
4.  **Correlation Calculation:** The system compares the time intervals of the sensor being analyzed with the intervals when the **Primary Occupancy Sensor** was `on` (considered occupied) and `off` (considered not occupied):
    *   **Calculate `P(Active | Occupied)`:** It measures the total duration the sensor was *active* during the times the primary sensor indicated the area was *occupied*. This duration is divided by the total time the area was considered *occupied*.
    *   **Calculate `P(Active | Not Occupied)`:** It measures the total duration the sensor was *active* during the times the primary sensor indicated the area was *not occupied*. This duration is divided by the total time the area was considered *not occupied*.
    *   **Calculate `P(Occupied)` (Prior):** For the primary sensor itself, its prior is calculated as the total time it was `on` divided by the total analysis period duration. For other sensors, their individual priors are also calculated based on their own total active time.
5.  **Storage:** These learned `prob_given_true`, `prob_given_false`, and `prior` values are stored persistently for each sensor. They override the default probabilities defined in the integration's constants.
6.  **Averaging for Type Priors:** The learned priors for individual entities are averaged to calculate priors for each *sensor type* (e.g., the average prior for all configured lights).
7.  **Overall Prior:** The overall prior probability for the area (used in the main Bayesian calculation and shown by the Prior Probability sensor) is calculated by averaging the priors of the *active* sensor types.

### Time-Based Prior Learning Process

Time-based prior learning follows a similar process but with additional steps:

1. **Time Slot Division:** The analysis period is divided into 30-minute time slots for each day of the week
2. **Slot-Specific Analysis:** For each time slot, the system analyzes occupancy patterns during that specific period
3. **Temporal Correlation:** Calculates occupancy probabilities for each of the 336 time slots (7 days × 48 slots)
4. **Database Storage:** Time-based priors are stored in a dedicated SQLite database with efficient indexing
5. **Real-Time Usage:** During calculations, the system determines the current time slot and uses the appropriate learned prior

## When Learning Occurs

### Global Prior Learning

*   **Automatically:** This learning process typically runs automatically in the background at a set interval (e.g., once per hour, though this might be configurable or dynamically adjusted).
*   **Manually:** You can trigger the learning process immediately using the `area_occupancy.update_priors` service call.
*   **Startup:** It may also run on Home Assistant startup or when the integration is first set up, especially if no prior learned data exists.

### Time-Based Prior Learning

*   **Automatically:** Time-based priors are calculated automatically on a configurable schedule (default: every 4 hours)
*   **Manually:** You can trigger time-based prior calculation using the `area_occupancy.run_analysis` service
*   **Startup:** Time-based prior calculations are deferred to background tasks to avoid blocking startup
*   **Frequency Control:** The frequency of both time-based prior and likelihood updates can be configured independently

## Using Learned Priors

Once calculated, these learned priors are used by the [Bayesian Probability Calculation](calculation.md) instead of the generic defaults, making the occupancy detection tailored to the specific behavior of sensors in that area.

### Priority Order

When calculating occupancy probability, the system uses priors in this order:

1. **Time-Based Prior**: If available for the current time slot, use the learned time-based prior
2. **Global Prior**: If no time-based prior exists, fall back to the traditional global prior
3. **Minimum Prior**: If no priors are available, use the minimum prior value (0.01%)

The **Prior Probability** sensor displays the calculated *overall* prior probability (`P(Occupied)` for the whole area), which may be either the time-based prior or global prior depending on availability.

## Configuration Options

Both types of prior learning can be configured through the integration options:

| Option | Description | Default | Impact |
|--------|-------------|---------|--------|
| **Historical Analysis Enabled** | Enable or disable all prior learning | `true` | Controls both global and time-based learning |
| **Time-Based Priors Enabled** | Enable or disable time-based priors specifically | `true` | Controls only time-based prior learning |
| **Time-Based Priors Frequency** | How often to recalculate time-based priors | `4` | Every 4 prior timer cycles (4 hours) |
| **Likelihood Updates Enabled** | Enable or disable automatic likelihood updates | `true` | Controls sensor likelihood learning |
| **Likelihood Updates Frequency** | How often to update sensor likelihoods | `2` | Every 2 prior timer cycles (2 hours) |

## Benefits of Combined Learning

The combination of global and time-based prior learning provides:

- **Baseline Accuracy**: Global priors provide a solid foundation for occupancy detection
- **Temporal Precision**: Time-based priors add context-aware accuracy for different times
- **Robust Fallback**: If time-based priors aren't available, the system gracefully falls back to global priors
- **Adaptive Learning**: Both systems continuously learn and adapt to changing patterns 