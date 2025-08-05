# Prior Probability Learning

To make the Bayesian calculations more accurate and specific to your environment, the integration automatically learns key probability values from your sensor history. This process adapts the system to how _you_ actually use the space.

## What is a Prior Probability?

In the context of Bayesian probability, the **prior probability** (often denoted as `P(Occupied)` or simply `prior`) represents our initial belief about the likelihood of an event _before_ considering any new evidence. For this integration, it's the baseline probability that the area is occupied, independent of the _current_ state of the sensors.

Think of it as the starting point for the calculation. When new sensor evidence comes in (e.g., motion is detected), Bayes' theorem uses this prior belief along with the sensor's learned likelihoods to calculate an updated belief (the posterior probability).

Initially, the integration uses default prior values. However, the **Prior Learning** process aims to calculate a more accurate prior based on the historical behavior of your **Primary Occupancy Sensor**, reflecting how often that area is typically occupied according to your chosen ground truth sensor.

## Global Prior Learning

The traditional approach calculates a single baseline prior probability for the entire area, regardless of time. This is the foundation of the learning system and provides a good baseline for occupancy detection.

## Goal of Learning

The primary goal is to determine, for each configured sensor:

1.  **`P(Sensor Active | Area Occupied)` (Likelihood - True):** How likely is this sensor to be in its "active" state when the area is genuinely occupied?
2.  **`P(Sensor Active | Area Not Occupied)` (Likelihood - False):** How likely is this sensor to be in its "active" state when the area is _not_ occupied? (This helps identify sensors that trigger falsely or independently of occupancy).
3.  **`P(Sensor Occupied)` (Prior):** What is the baseline probability that this sensor is active, based on its history? (This is particularly relevant for the primary sensor, which informs the overall prior).

## The Primary Sensor(s): Ground Truth

This learning process relies heavily on the **Primary Occupancy Sensor** you designate during configuration. This sensor (usually a reliable motion or dedicated occupancy sensor) is treated as the "ground truth" indicator of when the area was actually occupied during the historical analysis period.

## How Learning Works

### Global Prior Learning Process

1.  **Time Period:** The system looks back over the configured **History Period** (e.g., the last 7 days) using the Home Assistant recorder database.
2.  **Data Retrieval:** For the specified period, it fetches the state history for both the **Primary Occupancy Sensor** and the specific **sensor being analyzed**.
3.  **Interval Analysis:** The state histories are converted into time intervals, noting the start and end times for each state (e.g., when a sensor turned `on` and then `off`).
4.  **Correlation Calculation:** The system compares the time intervals of the sensor being analyzed with the intervals when the **Primary Occupancy Sensor** was `on` (considered occupied) and `off` (considered not occupied):
    - **Calculate `P(Active | Occupied)`:** It measures the total duration the sensor was _active_ during the times the primary sensor indicated the area was _occupied_. This duration is divided by the total time the area was considered _occupied_.
    - **Calculate `P(Active | Not Occupied)`:** It measures the total duration the sensor was _active_ during the times the primary sensor indicated the area was _not occupied_. This duration is divided by the total time the area was considered _not occupied_.
    - **Calculate `P(Occupied)` (Prior):** For the primary sensor itself, its prior is calculated as the total time it was `on` divided by the total analysis period duration. For other sensors, their individual priors are also calculated based on their own total active time.
5.  **Storage:** These learned `prob_given_true`, `prob_given_false`, and `prior` values are stored persistently for each sensor. They override the default probabilities defined in the integration's constants.
6.  **Averaging for Type Priors:** The learned priors for individual entities are averaged to calculate priors for each _sensor type_ (e.g., the average prior for all configured motion sensors).
7.  **Overall Prior:** The overall prior probability for the area (used in the main Bayesian calculation and shown by the Prior Probability sensor) is calculated by averaging the priors of the _active_ sensor types.

## When Learning Occurs

### Global Prior Learning

- **Automatically:** This learning process typically runs automatically in the background at a set interval (e.g., once per hour, though this might be configurable or dynamically adjusted).
- **Manually:** You can trigger the learning process immediately using the `area_occupancy.run_analysis` service call.
- **Startup:** It may also run on Home Assistant startup or when the integration is first set up, especially if no prior learned data exists.

## Using Learned Priors

Once calculated, these learned priors are used by the [Bayesian Probability Calculation](calculation.md) instead of the generic defaults, making the occupancy detection tailored to the specific behavior of sensors in that area.
