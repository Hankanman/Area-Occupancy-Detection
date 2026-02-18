# Basic Usage

Once [configured](configuration.md), the integration creates entities for each area. The two you'll use most are **Occupancy Status** (a binary sensor for automations) and **Occupancy Probability** (the underlying confidence percentage). Everything else helps you understand and tune the system.

## What to Focus on First

After your first area is set up:

1. **Watch the Occupancy Probability graph** in the History panel for a few hours to see how it responds to your activity
2. **Adjust the Threshold** if needed — start at 50%, raise it if you get false positives, lower it for more sensitivity
3. **Let it learn** — the integration analyses your sensor history automatically and gets more accurate over time

![alt text](../images/lounge_occupancy_graph.png)

The graph above shows a 24-hour period for a lounge. The threshold is set at 75% because there are strong occupancy indicators (multiple motion sensors + TV as a media player). You can see clear periods of occupancy from the spikes in Occupancy Probability and the corresponding Occupancy Status turning on/off.

The prior probability (yellow) shows the baseline — around 20% of the time someone is in this room. It fluctuates by time-of-day and day-of-week, getting more accurate the longer the integration runs.

## Created Entities

### Primary Entities (Use in Automations)

These are the entities you'll use day to day:

**Occupancy Status** — Binary sensor showing whether the area is occupied.

| State             | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| **Occupied** (on) | Probability of occupancy is at or above the threshold. |
| **Clear** (off)   | Probability of occupancy is below the threshold. |

**Occupancy Probability** — The calculated probability (0-100%) based on all sensor inputs, learned patterns, and decay.

**Threshold** — Adjustable number entity (1-99%) controlling when the Occupancy Status turns on. Change this to tune sensitivity without reconfiguring the integration.

**Detected Activity** — An enum sensor showing what activity is happening: `showering`, `cooking`, `watching_tv`, `listening_to_music`, `working`, `eating`, `sleeping`, `idle`, or `unoccupied`. Activities are constrained by the area's [purpose](../features/purpose.md) — for example, "showering" only appears in bathrooms. See [Activity Detection](../features/activity-detection.md) for details.

### Diagnostic Entities (For Tuning and Understanding)

These help you understand *why* the system reached its conclusion. All are marked as diagnostic in Home Assistant.

**Presence Confidence** — Probability from strong presence indicators only (motion, media, appliances, doors, windows, covers, power, sleep). Shows the "hard evidence" side of the calculation.

**Environmental Confidence** — How much environmental sensors (temperature, humidity, CO2, etc.) support or oppose occupancy. 50% is neutral, above supports, below opposes.

**Prior Probability** — The baseline probability before any current evidence is considered. Useful for seeing learned time-of-day patterns.

| Attribute      | Description                                     |
| -------------- | ----------------------------------------------- |
| `global_prior` | Baseline prior derived from historical analysis |
| `time_prior`   | Time-based modifier applied to the prior        |
| `day_of_week`  | Day-of-week index used for time prior           |
| `time_slot`    | Time slot index used for time prior             |

**Evidence** — Lists which sensors are currently providing evidence and which are inactive.

| Attribute     | Description                                                                   |
| ------------- | ----------------------------------------------------------------------------- |
| `evidence`    | Comma-separated list of active entity names                                   |
| `no_evidence` | Comma-separated list of inactive entity names                                 |
| `total`       | Total number of entities                                                      |
| `details`     | Detailed information for each entity including probabilities and decay status |

**Decay Status** — Shows decay progress (0-100%) when probability is decreasing after activity stops.

**Activity Confidence** — Confidence (0-100%) in the currently detected activity.

### Optional Entities

**Sleeping** — A binary sensor that turns `on` when people assigned to this area are detected as sleeping. Only created when people are configured via **Manage People** in the integration options. See [Sleep Presence](../features/sleep-presence.md) for details.

## Basic Automations

### Occupancy-Based Lighting

```yaml
automation:
  - alias: "Turn on lights when area occupied"
    trigger:
      - platform: state
        entity_id: binary_sensor.living_room_occupancy_status
        to: "on"
    action:
      - service: light.turn_on
        target:
          entity_id: light.living_room

  - alias: "Turn off lights when area unoccupied"
    trigger:
      - platform: state
        entity_id: binary_sensor.living_room_occupancy_status
        to: "off"
        for:
          minutes: 5
    action:
      - service: light.turn_off
        target:
          entity_id: light.living_room
```

### High Probability Alert

```yaml
automation:
  - alias: "High Occupancy Probability Alert"
    trigger:
      - platform: numeric_state
        entity_id: sensor.living_room_occupancy_probability
        above: 90
    action:
      - service: notify.mobile_app
        data:
          title: "High Occupancy Probability"
          message: "Living Room occupancy probability is {{ states('sensor.living_room_occupancy_probability') }}%"
```

## Troubleshooting

| Problem | What to check |
| --- | --- |
| **Probability stuck low** | Verify sensors are reporting correctly in Developer Tools > States. Check the Evidence sensor to see which entities are active. |
| **Sensors missing from evidence** | Ensure the sensor is added in the area's configuration and its state matches the expected active states (e.g., `on` for motion, `playing` for media). |
| **Activity not detected** | Activity detection is purpose-aware — check the area's purpose matches the expected activity (e.g., "showering" requires Bathroom purpose). |
| **Too many false positives** | Raise the Occupancy Threshold or reduce weights for unreliable sensors. |
| **Too many false negatives** | Lower the threshold, add more sensors, or increase weights for reliable sensors. |

For detailed debugging, see the [Debugging guide](../technical/debug.md).

## Tips and Tricks

1. **Optimal Threshold**:

   - Start at 50%
   - Increase for fewer false positives
   - Decrease for higher sensitivity

2. **Sensor Weights**:

   - Adjust based on reliability
   - Higher weights = stronger influence
   - Balance multiple sensors

3. **Decay Settings**:

   - Match room usage patterns
   - Longer windows for less traffic
   - Shorter windows for high traffic

4. **Historical Learning**:
   - Allow time to accumulate data
   - Accuracy improves over the first few weeks
   - Use the `area_occupancy.run_analysis` service to trigger a manual refresh
