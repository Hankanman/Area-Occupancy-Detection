# Area Occupancy Detection Documentation

Area Occupancy Detection aims to improve occupancy accuracy beyond single motion detectors by considering various environmental factors, device states, and historical data. It uses Bayesian probability to calculate the likelihood of an area being occupied based on multiple sensor inputs and learned patterns.

![Probability Cards](images/probability-cards.png)

## Features

### Core Features

- **[Bayesian Probability Calculation](features/calculation.md)**: Uses learned sensor reliability to calculate occupancy probability
- **[Historical Learning](features/prior-learning.md)**: Automatically learns from your sensor history to improve accuracy
- **[Probability Decay](features/decay.md)**: Gradually reduces probability when no activity is detected
- **[Multiple Sensor Types](features/entities.md)**: Supports motion, media, door, window, light, appliance, and environmental sensors
- **[Wasp in Box](features/wasp-in-box.md)**: Special logic for rooms with single entry/exit points

### Advanced Features

- **[Sensor Likelihoods](features/likelihood.md)**: Learns how reliable each sensor is for occupancy detection
- **[Purpose-Based Configuration](features/purpose.md)**: Automatic configuration based on room purpose

### User Interface

- **[Services](features/services.md)**: Available service calls for automation and control
- **[Entities](features/entities.md)**: Created entities and their attributes

## Getting Started

### Installation

- **[Installation Guide](getting-started/installation.md)**: How to install the integration via HACS or manual installation

### Configuration

- **[Configuration Guide](getting-started/configuration.md)**: Step-by-step configuration instructions
- **[Basic Usage](getting-started/basic-usage.md)**: How to use the integration after setup

## Key Concepts

### Bayesian Probability

The integration uses Bayes' theorem to update occupancy probability based on sensor evidence:

1. **Prior Belief**: Baseline probability of occupancy (learned from history)
2. **Sensor Evidence**: Current state of configured sensors
3. **Likelihood**: How reliable each sensor is (learned from history)
4. **Posterior**: Updated probability after considering new evidence

### Sensor Types

The integration supports multiple sensor types with different default weights:

- **Motion Sensors** (0.85): High reliability for occupancy detection
- **Media Devices** (0.70): Good indicator of active use
- **Appliances** (0.40): Moderate reliability
- **Door Sensors** (0.30): Lower reliability, but useful for entry/exit
- **Window Sensors** (0.20): Minimal influence
- **Lights** (0.20): Can indicate activity
- **Environmental** (0.10): Very low influence

## Example Use Cases

### Living Room

- **Sensors**: Motion sensors, TV, lights, door sensors
- **Patterns**: High occupancy in evenings, low during work hours
- **Automation**: Turn on lights when occupied, dim when low probability

### Office

- **Sensors**: Motion sensors, computer, lights, door sensors
- **Patterns**: High occupancy during work hours, low on weekends
- **Automation**: Turn on/off lights based on occupancy, adjust HVAC

### Bathroom

- **Sensors**: Motion sensors, door sensors, lights
- **Wasp in Box**: Maintain occupancy when door is closed
- **Automation**: Turn on lights when occupied, turn off when unoccupied

## Challenges with Basic Motion Sensors

- **Lights turning off while you're still:** If you sit still for too long, a basic motion sensor assumes the room is empty.
- **False triggers:** A pet walking through might trigger occupancy.
- **Limited context:** Simple motion doesn't know if you're watching TV, working at a desk, or just passing through.

## How Area Occupancy Detection Helps

This integration provides enhanced room occupancy detection for Home Assistant by intelligently combining data from multiple sensor inputs. Unlike simple motion sensors, it leverages Bayesian probability calculations to factor in various environmental cues and device states, leading to more accurate and resilient occupancy detection.

- **Increased Accuracy:** By fusing data from multiple sensor types (motion, doors, lights, media, etc.), the system gains a much richer understanding of the area's status.
- **Probabilistic Approach:** Instead of a simple ON/OFF state, it calculates an _occupancy probability_. You decide how certain the system must be before declaring occupancy.
- **Adaptability:** The prior probability learning feature analyses how _your_ sensors correlate with actual occupancy, learning which sensors are reliable indicators.
- **Reduced False Negatives/Positives:** The combination of multi-sensor input, learned probabilities and decay logic significantly reduces incorrect occupancy states.

## Key Features

- **Multi-Sensor Fusion:** Combines inputs from motion/occupancy sensors, media players, lights, doors, windows, appliances and environmental sensors (temperature, humidity, illuminance).
- **Bayesian Inference:** Calculates the probability of occupancy based on the current state of configured sensors and their individual learned likelihoods.
- **Prior Probability Learning:** Automatically learns how sensor states relate to actual occupancy (using a primary sensor as ground truth) over a configurable history period.
- **Configurable Weights:** Assign weights to different sensor _types_ to influence their impact on the overall probability.
- **Probability Decay:** Gradually decreases the occupancy probability when sensors indicate inactivity, providing a natural transition to "unoccupied".
- **Purpose-Based Decay:** Choosing a room purpose automatically sets a decay half life suited to the space.
- **Configurable Threshold:** Define the probability percentage required to consider the area "occupied".
- **Exposed Entities:**
  - Occupancy Probability Sensor (%)
  - Occupancy Status Binary Sensor (on/off)
  - Prior Probability Sensor (%)
  - Decay Status Sensor (%)
  - Occupancy Threshold Number Input
- **UI Configuration:** Easy setup and management through the Home Assistant UI.
- **Manual Prior Update Service:** Trigger the prior learning process on demand.

## How It Works

1.  **Configuration:** You select various sensors associated with an area (motion, doors, lights, media players, etc.) and configure parameters like weights and the history period for learning.
2.  **Prior Learning:** The integration analyses the history of your selected sensors against a designated primary motion/occupancy sensor. It calculates:
    - **P(Sensor Active | Area Occupied):** How likely is a sensor to be active when the area is truly occupied?
    - **P(Sensor Active | Area Not Occupied):** How likely is a sensor to be active when the area is _not_ occupied?
    - **P(Area Occupied):** The baseline (prior) probability of the area being occupied, derived from the primary sensor's history.
      These learned probabilities (or defaults if history is insufficient) are stored and used in calculations.
3.  **Real-time Calculation:** As your sensor states change, the integration uses Bayes' theorem. For each _active_ sensor, it updates the probability of occupancy based on its learned likelihoods and the overall prior probability.
4.  **Weighted Combination:** The contributions from individual active sensors are combined using a complementary probability approach, factoring in their configured weights.
5.  **Output:** The final calculated probability is exposed. If it crosses the configured threshold, the Occupancy Status sensor turns "on".
6.  **Decay:** If the probability starts decreasing (fewer active sensors), an exponential decay function gradually lowers the probability over a configured time window, unless new sensor activity pushes it back up.

## Common Issues

1. **No Occupancy Detection**:

   - Verify sensors are working correctly
   - Check threshold setting
   - Ensure sensors are properly configured
   - Adjust sensor weights

2. **False Positives**:

   - Lower weights for less reliable sensors
   - Increase occupancy threshold
   - Adjust decay settings
   - Review time-based priors

3. **False Negatives**:
   - Increase weights for reliable sensors
   - Lower occupancy threshold
   - Add additional sensors
   - Check time-based prior patterns

## Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/Hankanman/Area-Occupancy-Detection/issues)
- **Community Discussion**: [Ask questions and share experiences](https://github.com/Hankanman/Area-Occupancy-Detection/discussions)
- **GitHub Releases**: [Check for updates and changelog](https://github.com/Hankanman/Area-Occupancy-Detection/releases)
