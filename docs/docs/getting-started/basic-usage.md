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

It has the following attributes:

| Attribute                | Description                                                       |
| ------------------------ | ----------------------------------------------------------------- |
| **Active Triggers**      | The triggers (sensors) that are currently active                  |
| **Sensor Probabilities** | The probabilities from each sensor, shown in format: `Sensor Name | W: Weight | P: Probability | WP: Weighted Probability`. Example: `Living Room Motion Sensor | W: 0.85 | P: 0.75 | WP: 0.64` |

### Prior Probability

The prior probability is the probability of occupancy before any sensors are active. It is used to provide a baseline probability of occupancy.

It has the following attributes:

| Attribute                | Description                                                                 |
| ------------------------ | --------------------------------------------------------------------------- |
| **Motion**               | The combined prior probability from all motion sensors                      |
| **Media**                | The combined prior probability from all media sensors                       |
| **Appliance**            | The combined prior probability from all appliance sensors                   |
| **Door**                 | The combined prior probability from all door sensors                        |
| **Window**               | The combined prior probability from all window sensors                      |
| **Light**                | The combined prior probability from all light sensors                       |
| **Environmental**        | The combined prior probability from all environmental sensors               |
| **Prior Probability**    | The probability of occupancy before any sensors are active                  |
| **Last Updated**         | The last time the prior probability was updated                             |
| **Total Period**         | The time over which the prior probability was calculated                    |
| **Entity Count**         | The number of entities used to calculate the prior probability              |
| **Using Learned Priors** | Whether the prior probability is being used (false if using default priors) |

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
