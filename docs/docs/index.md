# Area Occupancy Detection

Welcome to the Area Occupancy Detection integration documentation. This integration provides intelligent room occupancy detection by combining multiple sensor inputs using Bayesian probability calculations.

## Overview

The Area Occupancy Detection integration enhances traditional motion-based occupancy detection by:

1. **Combining Multiple Data Sources**

      - Motion and occupancy sensors
      - Media device states
      - Appliance usage
      - Environmental sensors
      - Door and window states
      - Light states

2. **Using Bayesian Probability**

      - Calculates occupancy probability based on sensor states
      - Learns from historical data
      - Adapts to your living patterns
      - Provides confidence levels for occupancy detection

3. **Smart Features**

      - Time-based decay of probability
      - Historical analysis for improved accuracy
      - Configurable weights for different sensor types
      - Real-time threshold adjustments

## How it Works


Here's how the Area Occupancy sensor figures out if someone is in the room:

Think of the Area Occupancy sensor as a detective trying to figure out if someone's in a room based on various clues (your sensors).

1.  **Sensor Clues:** It constantly watches the sensors you've told it about (motion detectors, door sensors, lights, media players, etc.).
2.  **Adding Up Evidence:** When a sensor becomes "active" (like motion is detected, a door opens, or music starts playing), it increases the "chance of occupancy" score.
    *   **Weights Matter:** You tell the sensor how important each *type* of clue is by setting "weights". Motion sensors usually have a high weight, while a light turning on might have a lower weight.
3.  **Learning from the Past (The Smart Part):**
    *   The sensor looks back at history (e.g. the last week – you configure how long).
    *   It pays close attention to your "Primary Sensor" (usually the main motion sensor you trust most).
    *   It learns patterns: "How often was the *light* on when the *primary motion sensor* also saw movement?" or "How often was the *media player* playing when the *primary motion sensor* saw no movement?"
    *   This helps it figure out which sensors are *actually* good indicators of occupancy *in that specific room*. A sensor that's often active even when the room is empty won't be trusted as much over time. This learning happens automatically in the background (usually hourly).
4.  **The "Decay" Feature (Preventing Flickering):**
    *   When *all* sensors become inactive, the chance of occupancy doesn't drop to zero immediately.
    *   Instead, it slowly fades or "decays" over a time window you set (e.g., 5 minutes/300 seconds).
    *   This is helpful so the sensor doesn't think the room is empty just because you sat perfectly still for 30 seconds.
5.  **The Threshold:**
    *   You set a "Threshold" percentage (e.g. 50%).
    *   The sensor compares its final calculated "chance of occupancy" score (after considering all clues, learning, and decay) to this threshold.
    *   If the score is above the threshold, the main Occupancy Sensor turns ON. If it's below, it turns OFF.

**In short:** It combines real-time sensor activity (weighted by importance) with learned historical patterns to make an educated guess about occupancy, using a decay feature to smooth things out and a threshold to make the final occupied/unoccupied decision.


## Quick Start

1. Install the integration through HACS or manually
2. Configure at least one motion sensor
3. Add additional sensors to improve accuracy
4. Adjust weights and thresholds as needed

## Next Steps

- Read the [Installation Guide](getting-started/installation.md) for detailed setup instructions
- Learn about [Configuration Options](getting-started/configuration.md)
- Understand [How It Works](features/probability-calculation.md)
