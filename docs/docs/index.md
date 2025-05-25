# Home Assistant - Area Occupancy Detection

![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/validate.yml?branch=main)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/test.yml?label=tests)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Hankanman/Area-Occupancy-Detection/docs.yml?branch=main&label=docs)

**Intelligent room occupancy detection using Bayesian probability.**

This integration provides enhanced room occupancy detection for Home Assistant by intelligently combining data from multiple sensor inputs. Unlike simple motion sensors, it leverages Bayesian probability calculations to factor in various environmental cues and device states, leading to more accurate and resilient occupancy detection.

## Key Features

*   **Multi-Sensor Fusion:** Combines inputs from motion/occupancy sensors, media players, lights, doors, windows, appliances, and environmental sensors (temperature, humidity, illuminance).
*   **Bayesian Inference:** Calculates the probability of occupancy based on the current state of configured sensors and their individual learned likelihoods.
*   **Prior Probability Learning:** Automatically learns the typical correlation between sensor states and actual occupancy (using a primary motion/occupancy sensor as ground truth) over a configurable history period (e.g., last 7 days). This adapts the system to specific room usage patterns.
*   **Configurable Weights:** Allows assigning weights to different sensor *types* to influence their impact on the overall probability.
*   **Probability Decay:** Gradually decreases the occupancy probability when sensors indicate inactivity, providing a more natural transition to 'unoccupied' state.
*   **Configurable Threshold:** Define the probability percentage required to consider the area 'occupied'.
*   **Exposed Entities:**
    *   Occupancy Probability Sensor (%)
    *   Occupancy Status Binary Sensor (on/off)
    *   Prior Probability Sensor (%)
    *   Decay Status Sensor (%)
    *   Occupancy Threshold Number Input
*   **UI Configuration:** Easy setup and management through the Home Assistant UI.
*   **Manual Prior Update Service:** Trigger the prior learning process on demand.

## Why Use Area Occupancy Detection?

Traditional occupancy detection often relies on a single motion sensor, leading to common frustrations:

*   **Lights turning off while you're still:** If you sit still for too long, a basic motion sensor assumes the room is empty.
*   **False triggers:** A pet walking through might trigger occupancy.
*   **Limited context:** Simple motion doesn't know if you're watching TV, working at a desk, or just passing through.

Area Occupancy Detection addresses these issues by providing:

*   **Increased Accuracy:** By fusing data from multiple sensor types (motion, doors, lights, media, etc.), the system gains a much richer understanding of the area's status. It's less likely to be fooled by the limitations of a single sensor.
*   **Probabilistic Approach:** Instead of a simple ON/OFF state, it calculates an *occupancy probability*. This gives a more nuanced view and allows fine-tuning through the threshold. You can decide how certain the system needs to be before declaring occupancy.
*   **Adaptability:** The prior probability learning feature automatically analyzes how *your* sensors correlate with actual occupancy in *your* space over time. It learns which sensors are reliable indicators and adjusts its calculations accordingly, adapting to your specific usage patterns.
*   **Reduced False Negatives/Positives:** The combination of multi-sensor input, learned probabilities, and decay logic significantly reduces instances where the room is incorrectly marked as empty (false negative) or occupied (false positive).

This leads to more reliable and intelligent automations based on true room presence.

## Screenshots

![Probability Cards](images/probability-cards.png)

## How It Works

1.  **Configuration:** You select various sensors associated with an area (motion, doors, lights, media players, etc.) and configure parameters like weights and the history period for learning.
2.  **Prior Learning:** The integration analyzes the history of your selected sensors against a designated primary motion/occupancy sensor. It calculates:
    *   **P(Sensor Active | Area Occupied):** How likely is a sensor to be active when the area is truly occupied?
    *   **P(Sensor Active | Area Not Occupied):** How likely is a sensor to be active when the area is *not* occupied?
    *   **P(Area Occupied):** The baseline (prior) probability of the area being occupied, derived from the primary sensor's history.
    These learned probabilities (or defaults if history is insufficient) are stored and used in calculations.
3.  **Real-time Calculation:** As your sensor states change, the integration uses Bayes' theorem. For each *active* sensor, it updates the probability of occupancy based on its learned likelihoods (P(Active|Occupied) and P(Active|Not Occupied)) and the overall prior probability.
4.  **Weighted Combination:** The contributions from individual active sensors are combined using a complementary probability approach, factoring in their configured weights.
5.  **Output:** The final calculated probability is exposed. If it crosses the configured threshold, the Occupancy Status sensor turns 'on'.
6.  **Decay:** If the probability starts decreasing (fewer active sensors), an exponential decay function gradually lowers the probability over a configured time window, unless new sensor activity pushes it back up.

## Getting Started

See the [Installation](getting-started/installation.md) and [Configuration](getting-started/configuration.md) guides.
