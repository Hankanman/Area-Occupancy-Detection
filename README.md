# Area Occupancy Detection for Home Assistant

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/validate.yml?style=for-the-badge&branch=main)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/test.yml?style=for-the-badge&label=tests)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/docs.yml?style=for-the-badge&branch=main&label=docs)

[![version](https://shields.io/github/v/release/Hankanman/Area-Occupancy-Detection?style=for-the-badge)](https://github.com/Hankanman/Area-Occupancy-Detection/releases)
[![Latest Release](https://img.shields.io/badge/dynamic/json?style=for-the-badge&color=41BDF5&logo=home-assistant&label=installs&cacheSeconds=15600&url=https://analytics.home-assistant.io/custom_integrations.json&query=$.area_occupancy.total)](https://analytics.home-assistant.io/custom_integrations.json)
![GitHub Repo stars](https://img.shields.io/github/stars/Hankanman/Area-Occupancy-Detection?style=for-the-badge)

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ff813f?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/sebburrell)

Have you ever had your lights turn off while you're still in the room? Or watched your smart home mark you as "away" while you're sitting perfectly still, watching TV? These frustrating experiences happen because most occupancy detection relies on simple motion sensors that can't understand context.

**Area Occupancy Detection** solves these real-world problems by thinking more intelligently about what "occupied" really means. Instead of just checking if motion was detected, it combines multiple clues, learns from your patterns, and calculates the probability that someone is actually there.

**Area Occupancy Detection doesn't automate anything for you**—it provides the intelligent occupancy information you need to create reliable automations. AOD creates sensors that your automations can use to control lights, heating, and other devices. Think of AOD as the "smart sensor" that gives your automations better data to work with.

## The Quick Answer

**Here's why AOD is different:**

**HA**: "Motion detected? Occupied. Motion stopped? Not occupied."

🎯 **AOD**: "Let me check motion, TV, doors, appliances, learned patterns, and time of day... 75% confident someone is there."

**HA**: You configure everything manually. It never learns.

🧠 **AOD**: Learns from your history automatically. Gets smarter over time. Knows you're usually in the kitchen Sunday mornings.

**Core HA**: One sensor fails → wrong answer.

🔀 **AOD**: Combines multiple sensors intelligently. If motion misses you, TV being on maintains occupancy probability → your automations keep lights on.

**Core HA**: Motion stops → occupancy sensor turns off → automations turn lights off immediately.

⏱️ **AOD**: Motion stops → probability gradually decreases → occupancy sensor stays on longer → your automations keep lights on while you sit still.

**Core HA**: Basic features only.

✨ **AOD**: Special features like "Wasp in Box" (for bathrooms), whole-home aggregation, purpose-based defaults.

**The bottom line:** AOD provides intelligent occupancy sensors that your automations can use. It learns, adapts, and understands context—so when you build automations that respond to occupancy, they work reliably instead of turning lights off while you're still in the room.

## Creating Automations with AOD

Here's how AOD fits into your automation workflow:

### The Workflow

1. **AOD analyzes your sensors** → Motion, TV, doors, appliances, learned patterns
2. **AOD calculates probability** → Combines all inputs using Bayesian inference
3. **AOD creates occupancy sensors** → Binary occupancy status and probability sensors
4. **Your automations use these sensors** → Trigger actions based on occupancy state or probability
5. **AOD learns and adapts** → Gets smarter over time, improving your automations automatically

### What AOD Provides

AOD creates sensors that your automations can use:

- **Occupancy Status**: Binary sensor (`on` = occupied, `off` = clear) - use this in most automations
- **Occupancy Probability**: Percentage (0-100%) - use this for conditional or gradual actions
- **Prior Probability**: Baseline from learned patterns - useful for monitoring and debugging
- **Threshold**: Adjustable setting - fine-tune without reconfiguration

### How You Use It

You create automations that respond to AOD's sensors. For example:

- **Turn lights on** when occupancy status turns `on`
- **Turn lights off** when occupancy status turns `off` (with a delay to prevent flickering)
- **Adjust heating** based on occupancy probability
- **Dim lights gradually** as probability decreases

The key difference: AOD provides intelligent occupancy data. You decide what actions to take based on that data.

![Probability Cards](docs/docs/images/probability-cards.png)

## Documentation

AOD is extensively documented [here](https://hankanman.github.io/Area-Occupancy-Detection/).

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

## HACS

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Hankanman&repository=Area-Occupancy-Detection&category=integration)

1. **Ensure HACS is installed:** If you don't have the [Home Assistant Community Store (HACS)](https://hacs.xyz/) installed, follow their instructions to set it up first.
2. **Navigate to HACS:** Open your Home Assistant frontend and go to HACS in the sidebar.
3. **Search for Area Occupancy Detection:** Search for "Area Occupancy Detection" and select then **Download**.
4. **Restart Home Assistant:** After the download is complete, restart your Home Assistant instance

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
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ff813f?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/sebburrell)

[issues]: https://github.com/Hankanman/Area-Occupancy-Detection/issues
[community]: https://github.com/Hankanman/Area-Occupancy-Detection/discussions
[releases]: https://github.com/Hankanman/Area-Occupancy-Detection/releases
