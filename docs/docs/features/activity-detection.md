# Activity Detection

Activity detection identifies *what* activity is happening in an area, not just *whether* the area is occupied. It provides context-aware information about how a room is being used.

## Overview

The system evaluates a set of predefined activities against the current sensor state and area purpose. Each activity has a definition composed of weighted indicators — specific sensor types and conditions that suggest the activity is occurring. The activity with the highest score is reported.

## Available Activities

| Activity | Description | Allowed Room Purposes |
| --- | --- | --- |
| **Showering** | Hot water running in a bathroom | Bathroom |
| **Bathing** | Extended bathroom use with door closed | Bathroom |
| **Cooking** | Active kitchen use with appliances and environmental changes | Kitchen |
| **Watching TV** | Media playback on a TV or receiver | Living Room, Media Room, Bedroom |
| **Listening to Music** | Audio playback on a speaker | Living Room, Media Room, Office |
| **Working** | Desk work with computer/appliance use | Office |
| **Eating** | Seated meal time with environmental cues | Dining Room |
| **Sleeping** | Sleep detected via sleep presence sensor | Bedroom |
| **Idle** | Area is occupied but no specific activity matches |  |
| **Unoccupied** | Area is not occupied |  |

Activities are **purpose-aware**: an activity can only be detected in rooms whose purpose matches. For example, "Showering" will only appear in areas with a Bathroom purpose, and "Cooking" only in Kitchen areas.

## How Scoring Works

Each activity definition contains a list of **indicators** — sensor types with associated weights that signal the activity. When evaluating an activity:

1. **Filter by purpose:** Only activities matching the area's purpose are considered.
2. **Check each indicator:** For each indicator in the activity definition, the system checks whether a matching sensor exists in the area and whether it meets the condition (active state, elevated/suppressed environmental reading).
3. **Score calculation:** Matched indicator weights are summed. The score is then **normalized by the total definition weight** — meaning if some sensors in the definition are not configured in the area, the maximum achievable confidence is reduced proportionally. This prevents activities from scoring artificially high when key sensors are missing.
4. **Minimum threshold:** An activity must reach a minimum matched weight (default: 0.3) to qualify as a candidate.
5. **Tiebreaking:** When multiple activities qualify, ties are broken by:
      1. Highest confidence score
      2. Purpose specificity (fewer allowed purposes = more specific)
      3. Highest matched evidence weight

### Environmental Indicators

Some indicators check whether an environmental sensor reading is **elevated** or **suppressed** relative to learned Gaussian parameters, rather than using hardcoded thresholds. For example, the "Showering" activity checks for elevated humidity — what counts as "elevated" is learned from historical data specific to your home.

## Created Entities

Activity detection creates two entities per area:

*   **`sensor.<area>_detected_activity` (Detected Activity)**
    *   **Type:** Enum sensor
    *   **Description:** Reports the currently detected activity as one of the values listed above.
    *   **Attributes:**

| Attribute | Description |
| --- | --- |
| `confidence` | Confidence score as a percentage (0-100%) |
| `matching_indicators` | List of entity IDs that matched the activity definition |

*   **`sensor.<area>_activity_confidence` (Activity Confidence)**
    *   **Type:** Diagnostic sensor (%)
    *   **Description:** Reports the confidence score of the detected activity as a percentage.
    *   **Entity Category:** `diagnostic`

These entities are **per-area only** and are not included in the All Areas aggregation, since activities are specific to individual rooms.

## Example

Consider a **Bathroom** area with a humidity sensor, temperature sensor, motion sensor, and door sensor:

1. After someone starts a shower, humidity rises and the door closes.
2. The "Showering" definition matches: humidity elevated (weight 0.5) + temperature elevated (0.2) + motion active (0.15) + door closed (0.15) = total weight 1.0.
3. All indicators match, so the confidence is 100%.
4. The detected activity sensor reports `showering` with confidence `100.0`.
5. When the shower ends and humidity drops, the activity transitions to `idle` and eventually `unoccupied`.
