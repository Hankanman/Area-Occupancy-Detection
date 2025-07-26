# Area Occupancy Detection for Home Assistant

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/validate.yml?branch=main)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/test.yml?label=tests)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/docs.yml?branch=main&label=docs)
![Downloads](https://img.shields.io/badge/Downloads-265-blue)



This integration provides advanced room occupancy detection by combining multiple sensors through Bayesian probability calculations. It aims to improve occupancy accuracy beyond single motion detectors by considering various environmental factors, device states, and historical data.

## Features

- **Intelligent Occupancy Detection**: Uses multiple sensor inputs and Bayesian statistics
- **Multiple Sensor Support**:
  - Motion/Occupancy Sensors: Primary input for detecting presence
  - Media Devices: TV, media players, and similar devices used as activity indicators
  - Appliances: Sensors or switches representing devices like fans, PCs, or other appliances
  - Environmental Sensors: Illuminance, humidity, and temperature sensors contribute subtle occupancy clues
  - Doors, Windows, and Lights: These can influence or correlate with presence
- **Probability-Based Output**: Provides an occupancy probability (1-99%) and a binary occupancy status based on a configurable threshold
- **Adaptive Historical Analysis**: Learns sensor priors over time, improving accuracy as it gathers data
- **Configurable Time Decay**: Gradually reduces occupancy probability if no new triggers occur
- **Real-Time Threshold Adjustment**: Modify the occupancy threshold without reconfiguration
- **Weighted Sensor Contributions**: Fine-tune how much each sensor type influences the final probability
- **Sensor Likelihoods**: Learns how reliable each sensor is and lets you recalculate those values via service call
- **Purpose-Based Defaults**: Selecting a room purpose automatically sets a sensible decay half life

## Documentation

All documentation is available at the [Documentation Site](https://hankanman.github.io/Area-Occupancy-Detection/).

## Installation

### Option 1: HACS (Recommended)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Hankanman&repository=Area-Occupancy-Detection&category=integration)

### Option 2: Manual

1. Download the latest release from GitHub
2. Place the `custom_components/area_occupancy` directory in your `config/custom_components` folder
3. Restart Home Assistant

## Requirements

- Home Assistant with the following required integrations:
  - recorder
  - sensor
  - binary_sensor
  - number

## Configuration

1. Go to **Settings → Devices & Services**
2. Click **+ Add Integration** and search for **Area Occupancy Detection**
3. Follow the setup wizard to configure your sensors and parameters

### Configuration Steps

The setup wizard will guide you through:

1. **Basic Setup**:
   - Name: Label for your monitored area
   - Primary Occupancy Sensor: Main occupancy sensor for learning (optional but recommended)
   - Motion Sensors: Select one or more motion sensors (required)

2. **Device Configuration**:
   - Media Players: Entertainment devices (active when "playing" or "paused")
   - Appliances: Various device sensors (active when "on" or "standby")
   - Lights: Light entities
   - Window Sensors: Window contact sensors (active when "open")
   - Door Sensors: Door contact sensors (active when "closed")

3. **Environmental Sensors**:
   - Illuminance Sensors
   - Humidity Sensors
   - Temperature Sensors

4. **Weights Configuration**:
   - Motion Weight (default: 0.85)
   - Media Weight (default: 0.70)
   - Appliance Weight (default: 0.30)
   - Door Weight (default: 0.30)
   - Window Weight (default: 0.20)
   - Light Weight (default: 0.20)
   - Environmental Weight (default: 0.10)

5. **Parameters**:
   - Occupancy Threshold: Percentage at which binary sensor indicates occupancy (default: 50%)
   - History Period: Days of history to analyze for learning (default: 7 days)
   - Enable Time Decay: Whether probability should decay over time
   - Decay Half Life: Time for probability to halve when no activity is detected (default: 600 seconds)
   - Enable Historical Analysis: Whether to learn from historical data

   *Note: The Decay Minimum Delay parameter has been removed as of version 9.2. Existing configurations are migrated automatically.*

## Technical Details

### Probability Calculation

- Probability values are bounded between 1% and 99% to prevent extreme certainty
- Default prior probability is 17.13% based on typical home occupancy patterns
- Probability given true state: 50%
- Probability given false state: 10%

### Time Decay

- Uses an exponential decay function with λ = 0.866433976
- The probability drops to 25% at one decay half life
- Example: With the default 600 s half life the probability reduces to 25% after 300 s

### Historical Analysis

- Prior probabilities are learned from historical data
- Analysis results are cached for 6 hours to improve performance
- Historical data lookback period: 1-30 days (default: 7 days)

## Entities Created

The integration creates several entities, all prefixed with your configured area name:

### Binary Sensor
- `binary_sensor.[name]_occupancy_status`: Indicates if area is occupied based on probability threshold
  - State: `on` (occupied) / `off` (not occupied)
  - Device Class: `occupancy`

### Sensors
- `sensor.[name]_occupancy_probability`: Current calculated occupancy probability
  - Value: 1-99%
  - Attributes:
    - `active_triggers`: List of sensors currently indicating activity
    - `sensor_probabilities`: Individual probability details for each sensor
    - `threshold`: Current threshold setting
    - `decay_active`: Whether decay is currently active
    - `last_trigger_time`: Timestamp of last sensor trigger

- `sensor.[name]_prior_probability`: Learned prior probabilities from historical analysis
  - Value: 1-99% (average of all sensor type priors)
  - Attributes:
    - `motion_prior`: Prior probability for motion sensors
    - `media_prior`: Prior probability for media devices
    - `appliance_prior`: Prior probability for appliances
    - `door_prior`: Prior probability for doors
    - `window_prior`: Prior probability for windows
    - `light_prior`: Prior probability for lights
    - `environmental_prior`: Prior probability for environmental sensors
    - `last_update`: Timestamp of last prior update
    - `analysis_period`: Days of history analyzed

- `sensor.[name]_decay_status`: Current decay influence on probability
  - Value: 0-100% (amount of decay applied)
  - Attributes:
    - `decay_start_time`: When decay began
    - `original_probability`: Probability before decay started
    - `decay_half_life`: Current decay half life setting

### Number
- `number.[name]_occupancy_threshold`: Adjustable threshold for occupancy determination
  - Range: 1-99%
  - Default: 50%

## Services

### area_occupancy.run_analysis

Run the historical analysis process for an Area Occupancy instance. This imports recent state data from the recorder, updates priors and likelihoods and refreshes the entities.

Service Data:
```yaml
entry_id: "<config_entry_id>"  # Required: ID of the Area Occupancy instance
```

## Example Automations

**Turn Off Lights When Area Empty**:
```yaml
automation:
  - alias: "Turn off lights when room empty"
    trigger:
      - platform: state
        entity_id: binary_sensor.living_room_occupancy_status
        to: "off"
        for: "00:05:00"
    action:
      - service: light.turn_off
        target:
          area: living_room
```

**Adjust Based on Probability Level**:
```yaml
automation:
  - alias: "Dim lights on low occupancy probability"
    trigger:
      - platform: numeric_state
        entity_id: sensor.living_room_occupancy_probability
        below: 30
    action:
      - service: light.turn_on
        data:
          brightness_pct: 50
        target:
          area: living_room
```

**Update Priors Weekly**:
```yaml
automation:
  - alias: "Weekly prior probability update"
    trigger:
      - platform: time
        at: "03:00:00"
    condition:
      - condition: time
        weekday:
          - mon
    action:
      - service: area_occupancy.run_analysis
        data:
          entry_id: !input config_entry_id
```

## Troubleshooting

### Common Issues

1. **No Occupancy Detection**:
   - Verify motion sensors are working correctly
   - Check if threshold is too high
   - Ensure sensors are properly configured and available
   - Update the weights of the sensors to better favour your available sensors

2. **False Positives**:
   - Lower weights for less reliable sensors
   - Increase the occupancy threshold
   - Adjust decay settings to clear occupancy faster

3. **False Negatives**:
   - Increase weights for reliable sensors
   - Lower the occupancy threshold
   - Add additional relevant sensors

### Debugging

Enable debug logging in `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.area_occupancy: debug
```

Key things to check in the logs:
- Sensor state changes
- Probability calculations
- Prior probability updates
- Decay calculations

## Screenshots

![Probability Cards](docs/docs/images/probability-cards.png)

## Support & Feedback

- **Issues**: [GitHub Issues][issues]
- **Discussions**: [Community Discussion][community]
- **Releases & Changelog**: [GitHub Releases][releases]

If you enjoy the integration please consider buying me a coffee!
[![Buy me a coffee][buymeacoffee-shield]][buymeacoffee]

[issues]: https://github.com/Hankanman/Area-Occupancy-Detection/issues
[community]: https://github.com/Hankanman/Area-Occupancy-Detection/discussions
[releases]: https://github.com/Hankanman/Area-Occupancy-Detection/releases
[buymeacoffee-shield]: https://www.buymeacoffee.com/assets/img/guidelines/download-assets-sm-2.svg
[buymeacoffee]: https://buymeacoffee.com/sebburrell
