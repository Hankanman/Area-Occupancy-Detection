# Area Occupancy Detection for Home Assistant

This integration provides advanced room occupancy detection by combining multiple sensors through Bayesian probability calculations. It aims to improve occupancy accuracy beyond single motion detectors by considering various environmental factors, device states, and historical data.

![version_badge](https://img.shields.io/badge/Minimum%20HA%20version-2024.11-red)

[![Buy me a coffee][buymeacoffee-shield]][buymeacoffee]

## Features

- **Intelligent Occupancy Detection**: Uses multiple sensor inputs and Bayesian statistics.
- **Multiple Sensor Support**:
  - **Motion Sensors**: Primary input for detecting presence.
  - **Media Devices**: TV, media players, and similar devices used as activity indicators.
  - **Appliances**: Sensors or switches representing devices like fans, PCs, or other appliances.
  - **Environmental Sensors**: Illuminance, humidity, and temperature sensors contribute subtle occupancy clues.
  - **Doors, Windows, and Lights**: These can influence or correlate with presence.
- **Probability-Based Output**: Provides an occupancy probability (0-100%) and a binary occupancy status based on a configurable threshold.
- **Adaptive Historical Analysis**: Learns sensor priors over time, improving accuracy as it gathers data.
- **Decay Mechanism**: Gradually reduces occupancy probability if no new triggers occur.
- **Real-Time & Historical Insights**:
  - On-demand historical analysis.
  - Exportable calculations and historical data for external review.
- **Adjustable Parameters**: Fine-tune thresholds, decay windows, and time periods for historical learning.
- **Friendly Configuration Flow**: Easy setup and subsequent option adjustments via the UI.

## Installation

### Option 1: HACS (Recommended)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Hankanman&repository=Area-Occupancy-Detection&category=integration)

### Option 2: Manual

1. Download the latest release from GitHub.
2. Place the `custom_components/area_occupancy` directory in your `config/custom_components` folder.
3. Restart Home Assistant.

## Configuration

1. Go to **Settings → Devices & Services**.
2. Click **+ Add Integration** and search for **Area Occupancy Detection**.
3. Follow the setup wizard to select your sensors and configure parameters.

### Key Configuration Options

- **Name & Area ID**: Label your monitored area (e.g., "Living Room").
- **Motion Sensors**: At least one required. Consider multiple sensors for better accuracy.
- **Additional Sensors**: Add media players, appliances, doors, windows, lights, or environmental sensors.
- **Threshold**: Percentage at which the binary sensor indicates occupancy (`on` if probability ≥ threshold).
- **History Period**: Number of days to consider for learning priors.
- **Decay Settings**: Enable and configure how the probability decays over time without triggers.
- **Historical Analysis**: Enable storing and using historical data to refine calculations.

## Entities Provided

| **Entity**                                   | **Description**                                                                                    |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `binary_sensor.{area_name}_occupancy_status` | **State**: `on` (occupied) / `off` (not occupied). Derived from probability and threshold.         |
| `sensor.{area_name}_occupancy_probability`   | Shows the current occupancy probability (0-100%).                                                  |
| `sensor.{area_name}_motion_prior`            | Displays learned priors from historical data for motion sensors.                                   |
| `sensor.{area_name}_media_prior`             | Displays learned priors from historical data for media devices.                                    |
| `sensor.{area_name}_appliance_prior`         | Displays learned priors from historical data for appliances.                                       |
| `sensor.{area_name}_door_prior`              | Displays learned priors from historical data for doors.                                            |
| `sensor.{area_name}_light_prior`             | Displays learned priors from historical data for lights.                                           |
| `sensor.{area_name}_occupancy_prior`         | Displays learned priors from historical data for overall occupancy.                                |
| `sensor.{area_name}_decay_status`            | Indicates how much the probability has decayed since last trigger.                                 |
| `number.{area_name}_occupancy_threshold`     | Adjust the occupancy threshold in real-time. Useful for fine-tuning without editing configuration. |

## Attributes

Entity attributes may include:

- `active_triggers`: Sensors currently influencing occupancy.
- `sensor_probabilities`: Calculated probabilities per sensor.
- `decay_status`: Amount of decay applied.
- `learned_prior_sensors_count`: How many sensors have learned priors.
- `configured_sensors`: Which sensors are involved.
- `history_period`: How many days of history are considered.
- `last_occupied`: Timestamp of last confirmed occupancy.

## Services

The integration provides services to export data for analysis:

- **`area_occupancy.export_calculations`**
  Export the current probability calculations and sensor states to a JSON file for external review.

  **Service Data:**

  - `entry_id`: Integration instance ID.
  - `start_time`/`end_time`: Time window for the export (optional).
  - `output_file`: File path to save the output.

- **`area_occupancy.export_historical_analysis`**
  Export learned timeslot-based historical analysis.

  **Service Data:**

  - `entry_id`: Integration instance ID.
  - `days`: Number of days for historical data.
  - `output_file`: File path to save the output.

These services allow you to review and debug the integration's behavior, analyze sensor performance over time, and tweak settings accordingly.

## Example Automations

**Turn Off Lights When Empty**:

```yaml
automation:
  - alias: "Turn off lights when office empty"
    trigger:
      - platform: state
        entity_id: binary_sensor.office_occupancy_status
        to: "off"
        for: "00:05:00"
    action:
      - service: light.turn_off
        target:
          entity_id: light.office_lights
```

**Adjust Based on Probability**:

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

## Troubleshooting & Tips

- **Check Sensor States**: Validate in Developer Tools that sensors are reporting correctly.
- **Threshold Tuning**: Adjust the threshold using the threshold number entity to refine occupancy detection.
- **Decay Settings**: If occupancy lingers too long after triggers stop, shorten the decay window or adjust `decay_min_delay`.
- **Historical Learning**: Ensure `historical_analysis_enabled` is on and `history_period` is sufficient for stable priors.
- **Logging**: Enable debug logging in `configuration.yaml` for deeper troubleshooting:

  ```yaml
  logger:
    default: info
    logs:
      custom_components.area_occupancy: debug
  ```

## Support & Feedback

- **Issues**: [GitHub Issues][issues]
- **Discussions**: [Community Discussion][community]
- **Releases & Changelog**: [GitHub Releases][releases]

[issues]: https://github.com/Hankanman/Area-Occupancy-Detection/issues
[community]: https://github.com/Hankanman/Area-Occupancy-Detection/discussions
[releases]: https://github.com/Hankanman/Area-Occupancy-Detection/releases
[buymeacoffee-shield]: https://www.buymeacoffee.com/assets/img/guidelines/download-assets-sm-2.svg
[buymeacoffee]: https://buymeacoffee.com/sebburrell