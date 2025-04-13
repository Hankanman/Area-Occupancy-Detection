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
        *   `active_triggers`: A list of friendly names for sensors currently considered "active" and contributing to the probability.
        *   `sensor_probabilities`: A set of strings, each detailing an active sensor's contribution:
            *   `Friendly Name | W: [Weight] | P: [Raw Probability] | WP: [Weighted Probability]`
        *   `threshold`: The current threshold value (e.g., "50.0%").

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
    *   **Description:** Shows the *overall prior probability* (`P(Occupied)`) calculated for the area. This is typically the average of the priors for the sensor *types* that are currently configured and have learned/default priors.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`
    *   **Attributes:**
        *   `[sensor_type]`: Prior for each type (e.g., `motion`: "Prior: 35.0%"). Only shown if the type has sensors configured and a non-zero prior.
        *   `last_updated`: ISO timestamp string indicating when the priors were last calculated.
        *   `next_update`: ISO timestamp string indicating the next scheduled prior calculation time (or "Unknown").
        *   `total_period`: The history period used for learning (e.g., "7 days").
        *   `entity_count`: The number of individual entities for which specific priors have been learned and stored.
        *   `using_learned_priors`: Boolean (`true`/`false`) indicating if any learned entity-specific priors are currently being used.

*   **`sensor.area_decay_status_<area_name>` (Decay Status)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Indicates the progress of the probability decay when active. `0.0` means decay is not active. A value increasing towards `100.0` suggests decay is in progress. (100% likely corresponds to reaching the minimum probability limit or the configured decay window duration).
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`
    *   **Attributes:** (May include details like `decay_start_time`, `decay_start_probability` - check the entity state in Developer Tools for specifics). 