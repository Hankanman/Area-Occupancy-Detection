# Area Occupancy Detection for Home Assistant

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/validate.yml?style=for-the-badge&branch=main)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/test.yml?style=for-the-badge&label=tests)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/docs.yml?style=for-the-badge&branch=main&label=docs)

[![version](https://shields.io/github/v/release/Hankanman/Area-Occupancy-Detection?style=for-the-badge)](https://github.com/Hankanman/Area-Occupancy-Detection/releases)
[![Latest Release](https://img.shields.io/badge/dynamic/json?style=for-the-badge&color=41BDF5&logo=home-assistant&label=installs&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.area_occupancy.total)](https://analytics.home-assistant.io/custom_integrations.json)

This integration provides advanced area occupancy detection by combining multiple sensors through probability calculations. It aims to improve occupancy accuracy beyond single motion detectors by considering various environmental factors, device states, historical data, and learned time-based occupancy patterns.

## Documentation

All documentation is available at the [Documentation Site](https://hankanman.github.io/Area-Occupancy-Detection/).

## Why?

Most presence detection in Home Assistant is based on a single binary motion sensor. While simple, it’s often unreliable —
walk into a room and stand still, and the sensor “forgets” you’re there. Watch TV, and you’ll suddenly be marked as “away”. This leads to frustration, false triggers, and automations that don’t feel smart.

**Area Occupancy Detection** solves this by thinking more like a human:

- It **combines multiple sources of information** — motion, doors, windows, appliances, media players, lights, and environmental sensors — to build a picture of whether the room is truly occupied.
- It uses **Bayesian probability** to weigh and balance all these inputs, meaning a single missed detection won’t ruin the accuracy.
- It **learns your patterns** with **time-based priors**, adjusting its confidence depending on the day of the week and time of day.

  - Example: It knows you’re more likely to be in the kitchen on Sunday mornings than on Tuesday afternoons.

- It **adapts automatically** as your routines change, improving the longer it runs.
- It offers **fine-grained control** with weights, thresholds, motion timeouts, and decay settings so you can tune it to your environment.
- It **runs locally** in Home Assistant — no cloud services, no delays, full privacy.

The result?
Lights that stay on while you’re still in the room. Heating that only runs when someone’s actually home.
Automations that feel **predictive, not reactive** — making your smart home feel truly smart.

![Probability Cards](docs/docs/images/probability-cards.png)

## Features

- **Intelligent Occupancy Detection**: Uses multiple sensor inputs and Bayesian statistics.
- **Multiple Sensor Support**:

  - **Motion/Occupancy Sensors**: Primary input for detecting presence.
  - **Media Devices**: TV, media players, and similar devices used as activity indicators.
  - **Appliances**: Sensors or switches representing devices like fans, PCs, or other appliances.
  - **Environmental Sensors**: Illuminance, humidity, and temperature sensors contribute subtle occupancy clues.
  - **Doors, Windows, and Lights**: These can influence or correlate with presence.

- **Probability-Based Output**: Provides an occupancy probability (1-99%) and a binary occupancy status based on a configurable threshold.
- **Time-Based Priors**: Learns occupancy patterns by **day of week** and **time of day**, adjusting probability dynamically based on historical usage.
- **Adaptive Historical Analysis**: Learns sensor priors over time, improving accuracy as it gathers data.
- **Configurable Motion Timeout**: Specify how long motion remains active after last detection (default: 5 minutes).
- **Configurable Time Decay**: Gradually reduces occupancy probability if no new triggers occur.
- **Per-Type Probabilities**: Calculates and exposes occupancy probabilities for each input type (motion, media, appliance, door, window, illuminance, humidity, temperature).
- **Real-Time Threshold Adjustment**: Modify the occupancy threshold without reconfiguration.
- **Weighted Sensor Contributions**: Fine-tune how much each sensor type influences the final probability.
- **Purpose-Based Defaults**: Selecting a room purpose automatically sets a sensible decay half-life.

## Planned Features

- **Activity Detection**: Based on history and current state, predict the current activity like Cooking, Working, Sleeping etc.
- **Machine Learning Model**: Train a true neural network based on history to predict occupancy, may be needed for the activity detection.
- **Auto-Adjusting Threshold**: Based on the history of itself and predictions, automatically adjust the threshold for occupancy status so the user doesn't need to.
- **Numeric Sensor Analysis**: Currently numeric sensors (like environmental ones) are just triggered based on a fixed set of value ranges. This feature will correctly analyse the history and set correct ranges and detect trends.
- **Location Aware**: Leveraging BLE, WiFi, GPS
- **Weather-Aware**: If it’s cold and rainy, you might be more likely to be indoors — integrate weather data into priors to influence probabilities.
- **Occupancy Zone Hierarchies**:
  - **Parent-Child Area Relationships**: If the kitchen is occupied and the dining room is adjacent, allow probabilities to influence each other (shared likelihood factors).
  - **Multi-Room Tracking**: Track movement across rooms for more continuous occupancy detection — useful for corridor lighting.
  - **Adjacent Area Detection**: If motion is detected in a hallway before the living room, begin pre-heating or turning on devices as the probability rises.

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
3. Follow the setup wizard to configure your sensors and parameters.

## Entities Created

The integration creates several entities, all prefixed with your configured area name.

### Binary Sensor

- `binary_sensor.[name]_occupancy_status`: Indicates if area is occupied based on probability threshold.

  - State: `on` (occupied) / `off` (not occupied).
  - Device Class: `occupancy`.

### Sensors

- `sensor.[name]_occupancy_probability`: Current calculated occupancy probability.

  - Value: 1-99%.
  - Attributes:

    - `active_triggers`: List of sensors currently indicating activity.
    - `sensor_probabilities`: Individual probability details for each sensor.
    - `time_prior`: Prior probability for the current day/time slot.
    - `threshold`: Current threshold setting.
    - `decay_active`: Whether decay is currently active.
    - `last_trigger_time`: Timestamp of last sensor trigger.

- `sensor.[name]_prior_probability`: Learned prior probabilities from historical analysis.

  - Value: 1-99% (average of all priors).
  - Attributes:

    - `motion_prior`, `media_prior`, `appliance_prior`, `door_prior`, `window_prior`, `environmental_prior`.
    - `time_priors`: Learned priors for each day/time slot.
    - `last_update`: Timestamp of last prior update.
    - `analysis_period`: Days of history analyzed.

- `sensor.[name]_decay_status`: Current decay influence on probability.

  - Value: 0-100% (amount of decay applied).
  - Attributes:

    - `decay_start_time`: When decay began.
    - `decay_half_life`: Current decay half-life setting.

### Number

- `number.[name]_occupancy_threshold`: Adjustable threshold for occupancy determination.

  - Range: 1-99%.
  - Default: 50%.

## Debugging

Enable debug logging in `configuration.yaml`:

```yaml
logger:
  default: info
  logs:
    custom_components.area_occupancy: debug
```

Key things to check in the logs:

- Sensor state changes.
- Probability calculations.
- Prior probability updates (including time-based priors).
- Decay calculations.

## Support & Feedback

- **Issues**: [GitHub Issues][issues]
- **Discussions**: [Community Discussion][community]
- **Releases & Changelog**: [GitHub Releases][releases]

If you enjoy the integration, please consider buying me a coffee!
[![Buy me a coffee][buymeacoffee-shield]][buymeacoffee]

[issues]: https://github.com/Hankanman/Area-Occupancy-Detection/issues
[community]: https://github.com/Hankanman/Area-Occupancy-Detection/discussions
[releases]: https://github.com/Hankanman/Area-Occupancy-Detection/releases
[buymeacoffee-shield]: https://www.buymeacoffee.com/assets/img/guidelines/download-assets-sm-2.svg
[buymeacoffee]: https://buymeacoffee.com/sebburrell
