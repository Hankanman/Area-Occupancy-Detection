# Entities

This integration creates several entities in Home Assistant to expose the calculated data and allow control.

## Primary Entities

*   **`binary_sensor.area_occupancy_status_<area_name>` (Occupancy Status)**
    *   **State:** `on` / `off`
    *   **Description:** This is the main occupancy output. It turns `on` if the **Occupancy Probability** is greater than or equal to the **Occupancy Threshold**, and `off` otherwise.
    *   **Icon:** Changes based on state (`mdi:home-account` for `on`, `mdi:home-outline` for `off`).
    *   **Device Class:** `occupancy`

*   **`sensor.area_occupancy_probability_<area_name>` (Occupancy Probability)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Shows the current calculated Bayesian probability that the area is occupied. This value incorporates sensor inputs, learned priors, weights, and decay (if active).
    *   **Device Class:** `power_factor` (used for % display)
    *   **State Class:** `measurement`
    *   **Attributes:**
        *   `type_probabilities`: Mapping of each sensor type to its individual occupancy probability contribution.

*   **`number.area_occupancy_threshold_<area_name>` (Occupancy Threshold)**
    *   **State:** Numeric value (configurable range, typically 1-99)
    *   **Unit:** `%`
    *   **Description:** Allows you to adjust the probability threshold required for the **Occupancy Status** binary sensor to turn `on`. Changes made here are reflected immediately in the binary sensor's state.
    *   **Mode:** `slider` or `box` (depends on HA frontend)

## Diagnostic Entities

These entities provide insight into the internal calculations and are useful for tuning and debugging.

*   **`sensor.area_prior_probability_<area_name>` (Prior Probability)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Shows the combined prior probability used for occupancy calculations.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`
    *   **Attributes:**
        *   `global_prior`: Baseline prior derived from historical analysis.
        *   `time_prior`: Time-based modifier applied to the prior.
        *   `day_of_week`: Day-of-week index used for time prior.
        *   `time_slot`: Time slot index used for time prior.

*   **`sensor.area_evidence_<area_name>` (Evidence)**
    *   **State:** Number of entities currently managed
    *   **Description:** Lists entities providing evidence and those that are inactive.
    *   **Entity Category:** `diagnostic`
    *   **Attributes:**
        *   `evidence`: Comma-separated list of active entity names.
        *   `no_evidence`: Comma-separated list of inactive entity names.
        *   `total`: Total number of entities.
        *   `details`: Detailed information for each entity including probabilities and decay status.

*   **`sensor.area_decay_status_<area_name>` (Decay Status)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Indicates the progress of the probability decay when active. `0.0` means decay is not active. A value increasing towards `100.0` suggests decay is in progress. (100% likely corresponds to reaching the minimum probability limit or the configured decay window duration).
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`
    *   **Attributes:** (May include details like `decay_start_time`, `decay_start_probability` - check the entity state in Developer Tools for specifics). 