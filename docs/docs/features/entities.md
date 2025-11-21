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

## All Areas Aggregation Device

When you configure your first area, the integration automatically creates an **"All Areas"** device that aggregates occupancy data across all configured areas. This provides a unified view of occupancy across your entire home or selected areas.

### Device Information

*   **Device Name:** `All Areas`
*   **Device Identifier:** `all_areas`
*   **Created Automatically:** Yes, when the first area is configured

### Aggregation Entities

The "All Areas" device creates the following entities (using `all_areas` instead of an area name):

*   **`binary_sensor.area_occupancy_status_all_areas` (All Areas Occupancy Status)**
    *   **State:** `on` / `off`
    *   **Description:** Aggregated occupancy status using **OR logic**: turns `on` if **any** area is occupied, `off` only when **all** areas are unoccupied.
    *   **Icon:** Changes based on state (`mdi:home-account` for `on`, `mdi:home-outline` for `off`).
    *   **Device Class:** `occupancy`
    *   **Use Case:** Perfect for whole-home automations like "turn off all lights when no one is home" or "enable away mode when all areas are clear".

*   **`sensor.area_occupancy_probability_all_areas` (All Areas Occupancy Probability)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Average occupancy probability across all configured areas. Calculated as the mean of all individual area probabilities.
    *   **Device Class:** `power_factor` (used for % display)
    *   **State Class:** `measurement`
    *   **Use Case:** Provides a single metric representing overall occupancy likelihood across your home.

*   **`sensor.area_prior_probability_all_areas` (All Areas Prior Probability)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Average prior probability across all areas. This represents the baseline occupancy likelihood before considering current sensor evidence.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`
    *   **Use Case:** Useful for understanding overall occupancy patterns across your home.

*   **`sensor.area_decay_status_all_areas` (All Areas Decay Status)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Average decay status across all areas. Indicates the overall progress of probability decay when activity decreases.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`
    *   **Use Case:** Helps understand when overall occupancy is decreasing across multiple areas.

### How Aggregation Works

The "All Areas" device uses different aggregation strategies depending on the metric:

*   **Occupancy Status (Binary Sensor):** Uses **OR logic** - if any area is occupied, the aggregated status is `on`. Only when all areas are unoccupied does it show `off`.
*   **Probability:** Uses **average** - calculates the mean probability across all areas.
*   **Prior Probability:** Uses **average** - calculates the mean prior probability across all areas.
*   **Decay Status:** Uses **average** - calculates the mean decay status across all areas.

### Example Use Cases

1. **Whole-Home Occupancy Detection:**
   ```yaml
   automation:
     - alias: "Turn off all lights when no one is home"
       trigger:
         - platform: state
           entity_id: binary_sensor.area_occupancy_status_all_areas
           to: 'off'
       action:
         - service: light.turn_off
           target:
             entity_id: all
   ```

2. **Average Occupancy Monitoring:**
   ```yaml
   sensor:
     - platform: template
       sensors:
         overall_occupancy:
           value_template: "{{ states('sensor.area_occupancy_probability_all_areas') | float }}"
           unit_of_measurement: '%'
   ```

3. **Multi-Area Presence Detection:**
   Use the `binary_sensor.area_occupancy_status_all_areas` entity in presence detection automations that need to know if anyone is home, regardless of which specific area.

### Notes

*   The "All Areas" device is created automatically when you configure your first area.
*   The Evidence sensor is **not** created for "All Areas" as it would be too complex to aggregate evidence details across multiple areas.
*   All aggregation calculations are performed in real-time based on the current state of individual areas.
*   The device appears in Home Assistant's device registry under the name "All Areas".