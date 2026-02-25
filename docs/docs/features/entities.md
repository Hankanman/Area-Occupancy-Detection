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

*   **`sensor.area_presence_confidence_<area_name>` (Presence Confidence)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Shows the probability calculated from strong presence indicators only (motion, media, appliances, doors, windows, covers, power, sleep). This isolates the "hard evidence" of occupancy from environmental support.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`

*   **`sensor.area_environmental_confidence_<area_name>` (Environmental Confidence)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Shows the confidence from environmental sensors. **50% is neutral** (no environmental influence), values above 50% mean environmental data supports occupancy, and values below 50% mean environmental data opposes occupancy.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`

*   **`sensor.area_detected_activity_<area_name>` (Detected Activity)**
    *   **State:** One of: `showering`, `bathing`, `cooking`, `eating`, `watching_tv`, `listening_to_music`, `working`, `sleeping`, `idle`, `unoccupied`
    *   **Description:** Reports the currently detected activity in the area. Activities are constrained by the area's purpose (e.g., "showering" only appears in bathrooms). See [Activity Detection](activity-detection.md) for details.
    *   **Device Class:** `enum`
    *   **Attributes:**
        *   `confidence`: Confidence score as a percentage (0-100%).
        *   `matching_indicators`: List of entity IDs that matched the activity definition.

*   **`sensor.area_activity_confidence_<area_name>` (Activity Confidence)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Shows the confidence score of the detected activity as a percentage.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`

*   **`binary_sensor.<area_name>_sleeping` (Sleeping)**
    *   **State:** `on` / `off`
    *   **Description:** Turns `on` when one or more people assigned to this area are detected as sleeping. Supports multiple sleep sensors per person (both numeric and binary). Only created for areas that have people configured via the **Manage People** option. See [Sleep Presence](sleep-presence.md) for details.
    *   **Device Class:** `occupancy`
    *   **Icon:** `mdi:sleep`
    *   **Attributes:**
        *   `people_sleeping`: List of friendly names of people currently sleeping.
        *   `people`: Detailed list with person name, state, sleep sensor states, threshold, and sleeping status per person.

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

When you configure your first area, the integration automatically creates an **"All Areas"** device that aggregates occupancy data across all configured areas. This provides a unified view of occupancy across your entire home.

!!! info "Excluding areas from aggregation"
    Areas can be excluded from the All Areas aggregation using the **Exclude from All Areas** toggle in each area's Detection Behavior settings. This is useful for outdoor or utility areas (garages, driveways, patios) that you don't want contributing to the whole-home occupancy status. See [Configuration](../getting-started/configuration.md) for details.

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

*   **`sensor.area_presence_confidence_all_areas` (All Areas Presence Confidence)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Average presence confidence across all configured areas.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`

*   **`sensor.area_environmental_confidence_all_areas` (All Areas Environmental Confidence)**
    *   **State:** Numeric value (0.0 to 100.0)
    *   **Unit:** `%`
    *   **Description:** Average environmental confidence across all configured areas.
    *   **Device Class:** `power_factor`
    *   **State Class:** `measurement`
    *   **Entity Category:** `diagnostic`

!!! note
    **Detected Activity** and **Activity Confidence** are **not** aggregated across areas, as activities are specific to individual rooms.

### How Aggregation Works

The "All Areas" device uses different aggregation strategies depending on the metric:

*   **Occupancy Status (Binary Sensor):** Uses **OR logic** - if any area is occupied, the aggregated status is `on`. Only when all areas are unoccupied does it show `off`.
*   **Probability:** Uses **average** - calculates the mean probability across all areas.
*   **Prior Probability:** Uses **average** - calculates the mean prior probability across all areas.
*   **Decay Status:** Uses **average** - calculates the mean decay status across all areas.
*   **Presence Confidence:** Uses **average** - calculates the mean presence confidence across all areas.
*   **Environmental Confidence:** Uses **average** - calculates the mean environmental confidence across all areas.

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
*   Areas with **Exclude from All Areas** enabled are not included in the aggregation.
*   The Evidence sensor is **not** created for "All Areas" as it would be too complex to aggregate evidence details across multiple areas.
*   The Detected Activity and Activity Confidence sensors are **not** created for "All Areas" as activities are specific to individual rooms.
*   All aggregation calculations are performed in real-time based on the current state of individual areas.
*   The device appears in Home Assistant's device registry under the name "All Areas".

## Floor-Based Aggregation Devices

If your areas are assigned to [floors](https://www.home-assistant.io/docs/organizing/floors/) in Home Assistant, the integration automatically creates per-floor aggregation devices. These work identically to the All Areas device but only aggregate areas on the same floor.

### How It Works

At startup, the integration discovers which floors your configured areas belong to (via the Home Assistant floor registry) and creates a device for each floor. If you add or remove floors, the devices update on the next reload.

### Device Information

*   **Device Name:** `<Floor Name>` (e.g., "Upstairs", "Ground Floor")
*   **Created Automatically:** Yes, for each floor that has at least one configured area

### Floor Aggregation Entities

Each floor device creates the same set of entities as the All Areas device, scoped to that floor:

*   **`binary_sensor.area_occupancy_status_<floor_name>` (Floor Occupancy Status)**
    *   Uses **OR logic**: `on` if any area on the floor is occupied.

*   **`sensor.area_occupancy_probability_<floor_name>` (Floor Occupancy Probability)**
    *   Average probability across areas on this floor.

*   **`sensor.area_prior_probability_<floor_name>` (Floor Prior Probability)**
    *   Average prior probability across areas on this floor.
    *   **Entity Category:** `diagnostic`

*   **`sensor.area_decay_status_<floor_name>` (Floor Decay Status)**
    *   Average decay status across areas on this floor.
    *   **Entity Category:** `diagnostic`

*   **`sensor.area_presence_confidence_<floor_name>` (Floor Presence Confidence)**
    *   Average presence confidence across areas on this floor.
    *   **Entity Category:** `diagnostic`

*   **`sensor.area_environmental_confidence_<floor_name>` (Floor Environmental Confidence)**
    *   Average environmental confidence across areas on this floor.
    *   **Entity Category:** `diagnostic`

!!! note
    Floor aggregation respects the **Exclude from All Areas** toggle — excluded areas are also excluded from their floor's aggregation.

### Example Use Cases

1. **Floor-Based Lighting:**
   ```yaml
   automation:
     - alias: "Turn off upstairs lights when floor is clear"
       trigger:
         - platform: state
           entity_id: binary_sensor.area_occupancy_status_upstairs
           to: 'off'
           for:
             minutes: 10
       action:
         - service: light.turn_off
           target:
             area_id:
               - master_bedroom
               - guest_bedroom
               - upstairs_bathroom
   ```

2. **Floor-Level HVAC Control:**
   Use the floor occupancy probability to adjust heating/cooling zones per floor rather than per room.