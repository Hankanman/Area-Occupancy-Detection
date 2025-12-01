# Basic Usage

This guide covers the basic usage of the Area Occupancy Detection integration after installation and configuration.

## Created Entities

After setup, the integration creates several entities:

### Occupancy Status

This entity shows the overall occupancy status, you can use it in automations such as turning on lights when the area is occupied

| State             | Description                                                         |
| ----------------- | ------------------------------------------------------------------- |
| **Occupied** (on) | This means the probability of occupancy is above the set threshold. |
| **Clear** (off)   | This means the probability of occupancy is below the set threshold. |

### Occupancy Probability

This entity shows the calculated probability of occupancy based on the sensors and their weights that are currently active based on your configuration.

### Prior Probability

The prior probability is the probability of occupancy before any sensors are active. It is used to provide a baseline probability of occupancy.

**Attributes:**

| Attribute      | Description                                     |
| -------------- | ----------------------------------------------- |
| `global_prior` | Baseline prior derived from historical analysis |
| `time_prior`   | Time-based modifier applied to the prior        |
| `day_of_week`  | Day-of-week index used for time prior           |
| `time_slot`    | Time slot index used for time prior             |

### Evidence

Lists active and inactive entities and provides detailed information about each entity.

**Attributes:**

| Attribute     | Description                                                                   |
| ------------- | ----------------------------------------------------------------------------- |
| `evidence`    | Comma-separated list of active entity names                                   |
| `no_evidence` | Comma-separated list of inactive entity names                                 |
| `total`       | Total number of entities                                                      |
| `details`     | Detailed information for each entity including probabilities and decay status |

### Decay Status

This entity shows the status of the decay process.

**State**: % of decay active (0-100)

### Threshold

This number entity shows the threshold for occupancy.

You can change the threshold using this entity and will be reflected in the occupancy status entity immediately. This is great for quickly testing different thresholds without having to reconfigure the integration.

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
      - Enable for better accuracy
      - Use longer periods when stable
      - Update regularly
