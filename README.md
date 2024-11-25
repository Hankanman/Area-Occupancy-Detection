# Area Occupancy Detection for Home Assistant

This integration provides intelligent room occupancy detection by combining multiple sensor inputs using Bayesian probability calculations. It can detect occupancy more accurately than single motion sensors by considering various environmental factors and device states.

![HACS Default][hacs-shield]
![Project Maintenance][maintenance-shield]
[![GitHub Release][release-shield]][release]

## Features

- **Intelligent Occupancy Detection**: Uses multiple sensors for more accurate presence detection
- **Probability-Based**: Shows both definitive occupancy state and probability percentage
- **Multi-Sensor Support**:
  - Motion sensors (primary detection)
  - Illuminance sensors (light level changes)
  - Temperature sensors (environmental changes)
  - Humidity sensors (environmental changes)
  - Device states (TV, game consoles, etc.)
- **Adaptive Learning**: Uses historical data for improved accuracy
- **Configurable Settings**: Customize thresholds, decay times, and sensor weights
- **Graceful Degradation**: Continues functioning even if some sensors are unavailable
- **Real-Time Updates**: Immediate response to sensor changes

## Installation

### Option 1: HACS Installation (Recommended)

1. Ensure [HACS](https://hacs.xyz/) is installed
2. Search for "Area Occupancy Detection" in HACS
3. Click Install
4. Restart Home Assistant

### Option 2: Manual Installation

1. Download the latest release
2. Copy the `custom_components/area_occupancy` folder to your `config/custom_components` directory
3. Restart Home Assistant

## Configuration

### Initial Setup

1. Go to Settings → Devices & Services
2. Click "+ Add Integration"
3. Search for "Area Occupancy Detection"
4. Follow the configuration flow

### Configuration Options

| Option | Description | Required |
|--------|-------------|----------|
| Area Name | Name for the monitored area | Yes |
| Motion Sensors | One or more motion sensors | Yes |
| Illuminance Sensors | Light level sensors | No |
| Humidity Sensors | Humidity sensors | No |
| Temperature Sensors | Temperature sensors | No |
| Device States | Media players, TVs, etc. | No |
| Threshold | Occupancy probability threshold (0.0-1.0) | No |
| History Period | Days of historical data to use | No |
| Decay Enabled | Enable sensor reading decay | No |
| Decay Window | How long before sensor readings decay | No |
| Decay Type | Linear or exponential decay curve | No |

### Entities Created

The integration creates two entities for each configured area:

1. **Binary Sensor** (`binary_sensor.{area_name}_occupancy_status`)
   - State: `on` (occupied) or `off` (not occupied)
   - Indicates definitive occupancy based on probability threshold

2. **Probability Sensor** (`sensor.{area_name}_occupancy_probability`)
   - State: 0-100 percentage
   - Shows the calculated likelihood of room occupancy

### Entity Attributes

Both entities provide detailed attributes:

- `probability`: Current calculated probability (0-100%)
- `prior_probability`: Previous probability calculation
- `active_triggers`: Currently active sensors/triggers
- `sensor_probabilities`: Individual probability per sensor
- `decay_status`: Current decay values for sensors
- `confidence_score`: Reliability of the calculation
- `sensor_availability`: Status of each configured sensor
- `last_occupied`: Last time area was occupied
- `state_duration`: Time in current state
- `occupancy_rate`: Percentage of time area is occupied
- `moving_average`: Average probability over time
- `rate_of_change`: How quickly probability is changing

## Usage Examples

### Basic Setup with Motion Only

```yaml
# Example configuration with just motion sensors
Area Name: Living Room
Motion Sensors:
  - binary_sensor.living_room_motion
  - binary_sensor.living_room_corner
Threshold: 0.5
```

### Advanced Multi-Sensor Setup

```yaml
# Example configuration using all sensor types
Area Name: Home Office
Motion Sensors:
  - binary_sensor.office_motion
  - binary_sensor.desk_motion
Illuminance Sensors:
  - sensor.office_light_level
Temperature Sensors:
  - sensor.office_temperature
Device States:
  - media_player.office_tv
  - binary_sensor.computer_power
Threshold: 0.6
Decay Window: 300
Decay Type: exponential
```

## Automation Examples

### Turn Off Lights When Area Empty

```yaml
automation:
  - alias: "Turn off lights when office empty"
    trigger:
      - platform: state
        entity_id: binary_sensor.office_occupancy_status
        to: "off"
        for:
          minutes: 5
    action:
      - service: light.turn_off
        target:
          entity_id: light.office_lights
```

### Adjust Based on Probability

```yaml
automation:
  - alias: "Dim lights on low occupancy probability"
    trigger:
      - platform: numeric_state
        entity_id: sensor.office_occupancy_probability
        below: 30
    action:
      - service: light.turn_on
        target:
          entity_id: light.office_lights
        data:
          brightness_pct: 50
```

## Troubleshooting

### Occupancy Not Detecting Correctly

1. Check sensor states in Developer Tools
2. Verify motion sensors are responding
3. Lower threshold value if detection is too strict
4. Increase decay window if state changes too quickly

### Probability Always Low

1. Verify sensor connections and availability
2. Check sensor placement and coverage
3. Ensure device states are updating
4. Review historical data period setting

### Delayed Response

1. Reduce decay window setting
2. Check sensor update intervals
3. Verify automation trigger conditions

### General Issues

1. Enable debug logging:
   ```yaml
   logger:
     default: info
     logs:
       custom_components.area_occupancy: debug
   ```
2. Review Home Assistant logs
3. Check sensor availability attributes
4. Verify sensor connections and placement

## Support

- Report issues on [GitHub][issues]
- Join our [Community Discussion][community]
- View [Release Notes][releases]

[hacs-shield]: https://img.shields.io/badge/HACS-Default-orange.svg
[maintenance-shield]: https://img.shields.io/badge/maintainer-Seb%20Burrell-blue.svg
[release-shield]: https://img.shields.io/github/release/Hankanman/Area-Occupancy-Detection.svg
[release]: https://github.com/Hankanman/Area-Occupancy-Detection/releases
[issues]: https://github.com/Hankanman/Area-Occupancy-Detection/issues
[community]: https://github.com/Hankanman/Area-Occupancy-Detection/discussions
[releases]: https://github.com/Hankanman/Area-Occupancy-Detection/releases
