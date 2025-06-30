# Probability Decay

To prevent the occupancy status from flickering `off` the instant sensors become inactive (e.g., if you sit still for a moment), the integration includes a probability decay mechanism.

## Purpose

The decay feature provides a smoother transition from an occupied state to an unoccupied state. When sensor activity ceases or suggests the area might be becoming vacant, the calculated **Occupancy Probability** doesn't immediately drop. Instead, it gradually decreases over a configured time window.

## How Decay Works

1.  **Trigger Condition:** Decay starts *only* when the calculated occupancy probability (based on currently active sensors) *decreases* compared to the previous calculation cycle.
2.  **Decay Start:** When a decrease is detected:
    *   The system notes the **current time** (`decay_start_time`).
    *   It records the **probability value *before* the decrease** (`decay_start_probability`). This value serves as the starting point for the decay curve.
    *   The **Decay Status** sensor likely becomes active (showing > 0%).
3.  **Exponential Decay:** As time passes from `decay_start_time`, the system calculates a decay factor based on an exponential function. The rate of decay is determined by the configured **Decay Half Life** and an internal decay constant (`DECAY_LAMBDA`). A shorter half life results in faster decay.
4.  **Applying Decay:** The calculated decay factor is applied to the `decay_start_probability`. This results in a potentially lower probability value.
5.  **Floor Value:** Importantly, the decayed probability can **never go below** the probability currently being calculated based on any sensors that *are* still active. If, during decay, a sensor reactivates and pushes the calculated probability up, that new higher value becomes the floor.
6.  **Decay Stops When:**
    *   The calculated occupancy probability (based on sensor states) **increases** or stays the same. The decay state is reset.
    *   The decayed probability reaches the defined **minimum probability** (e.g., 1%). The probability stays at the minimum, and the decay state is reset.

## Configuration

*   **Decay Enabled:** A toggle to turn the decay feature on or off entirely.
*   **Decay Half Life (seconds):** The time for the probability to fall to half of its starting value when no new activity occurs. For example, a 300-second half life means the probability halves every 5 minutes (it may stop earlier if activity resumes).

Note: The previous *Decay Minimum Delay* option has been removed in version 9.2. Any existing configurations are updated automatically during migration.

## Output

*   The **Occupancy Probability** sensor reflects the decayed value when decay is active.
*   The **Decay Status** sensor indicates the progress of the decay, likely as a percentage (0% when not decaying, increasing towards 100% as decay progresses towards the minimum probability). *Note: The exact representation might vary; check the sensor's state in your HA instance.* 