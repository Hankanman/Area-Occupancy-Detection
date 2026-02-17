# Sleep Presence Detection

Sleep presence detection identifies when people are sleeping in an area, using Home Assistant Person entities combined with phone-reported sleep confidence from the [Companion App](https://companion.home-assistant.io/).

## How It Works

The system pairs two pieces of information per person:

1. **Person entity** (`person.<name>`): Tracks whether the person is `home` or `not_home`.
2. **Sleep confidence sensor** (`sensor.<name>_sleep_confidence`): A numeric sensor (0-100) reported by the HA Companion App indicating how confident the phone is that the user is sleeping.

When both conditions are met — the person is **home** and their sleep confidence is **at or above the configured threshold** — the sleep presence sensor for the assigned area turns **on**.

## Configuration

Sleep presence is configured through the **Manage People** option in the integration's main menu:

1. Navigate to **Settings** > **Devices & Services** > **Integrations** > **Area Occupancy Detection** > **Configure**.
2. Select **Manage People** from the main menu.
3. For each person, configure:

| Setting | Description |
| --- | --- |
| **Person Entity** | The `person.<name>` entity to track |
| **Sleep Confidence Sensor** | The Companion App sleep confidence sensor (`sensor.<name>_sleep_confidence`) |
| **Sleep Area** | Which area this person sleeps in |
| **Confidence Threshold** | Minimum sleep confidence (0-100) required to consider the person sleeping. Default: **50** |

You can add multiple people, each assigned to a different (or the same) sleep area.

## Created Entity

For each area that has at least one person assigned, a binary sensor is created:

*   **`binary_sensor.<area>_sleeping` (Sleeping)**
    *   **State:** `on` / `off`
    *   **Description:** Turns `on` when one or more assigned people are detected as sleeping in this area.
    *   **Device Class:** `occupancy`
    *   **Icon:** `mdi:sleep`
    *   **Attributes:**

| Attribute | Description |
| --- | --- |
| `people_sleeping` | List of friendly names of people currently sleeping |
| `people` | Detailed list with `person_name`, `person_state`, `sleep_confidence`, `sleep_threshold`, and `sleeping` (bool) per person |

## Impact on Occupancy

The sleep presence sensor feeds into the occupancy calculation as an `InputType.SLEEP` entity:

- **Weight:** 0.90 (very strong influence)
- **P(Active \| Occupied):** 0.95
- **P(Active \| Not Occupied):** 0.02
- **Decay half-life:** 2 hours (7200 seconds)

This means that when sleep is detected, the area will be held at a high occupancy probability. When sleep ends, the probability decays gradually over roughly 2 hours rather than dropping instantly.

## Requirements

- Home Assistant Companion App installed on the person's phone (provides the sleep confidence sensor)
- Person entities configured in Home Assistant
- At least one area configured in the integration

## Tips

- **Threshold tuning:** Start with the default threshold of 50. If the sensor triggers too easily (e.g., while watching TV in bed), increase it. If it doesn't trigger reliably, decrease it.
- **Multiple bedrooms:** Each person can be assigned to a different sleep area, making this work well for households with multiple bedrooms.
- **Interaction with Bedroom purpose:** Sleep presence works alongside the [Bedroom purpose](purpose.md) decay settings. During sleep, the combination of long bedroom decay and the sleep sensor's own 2-hour decay half-life provides very stable occupancy.
