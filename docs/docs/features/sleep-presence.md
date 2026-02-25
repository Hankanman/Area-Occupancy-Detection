# Sleep Presence Detection

Sleep presence detection identifies when people are sleeping in an area, using Home Assistant Person entities combined with one or more sleep detection sensors. This supports a wide range of devices including the [Companion App](https://companion.home-assistant.io/) sleep confidence, iOS Sleep Focus, Withings sleep mats, and any other binary or numeric sleep sensor.

## How It Works

The system pairs two pieces of information per person:

1. **Person entity** (`person.<name>`): Tracks whether the person is `home` or `not_home`. If a **device tracker** override is configured, that entity's state is used instead.
2. **Sleep sensors**: One or more sensors that indicate sleep. Two types are supported:
    - **Numeric sensors** (e.g., `sensor.<name>_sleep_confidence`): A 0-100 value where the person is considered sleeping when the value is at or above the configured confidence threshold.
    - **Binary sensors** (e.g., `binary_sensor.<name>_in_bed`, `binary_sensor.<name>_sleep_focus`): The person is considered sleeping when the sensor is `on`.

When both conditions are met — the person is **home** (per the person entity or the overriding device tracker) and **any** of their sleep sensors is active — the sleep presence sensor for the assigned area turns **on**.

!!! tip "When to use the device tracker override"
    HA's Person entity aggregates all associated device trackers and reports `home` when *any* tracker is home. If you have multiple trackers (e.g., phone GPS, router presence) and one is more reliable for sleep detection, set it as the override so the sleep check uses that specific tracker instead of the aggregated result.

## Configuration

Sleep presence is configured through the **Manage People** option in the integration's main menu:

1. Navigate to **Settings** > **Devices & Services** > **Integrations** > **Area Occupancy Detection** > **Configure**.
2. Select **Manage People** from the main menu.
3. For each person, configure:

| Setting | Description |
| --- | --- |
| **Person Entity** | The `person.<name>` entity to track |
| **Sleep Sensors** | One or more sleep detection sensors. Supports numeric sensors (e.g., sleep confidence 0-100%) and binary sensors (e.g., in-bed sensor, Sleep Focus). Any active sensor means the person is sleeping. |
| **Sleep Area** | Which area this person sleeps in |
| **Confidence Threshold** | Minimum value (0-100) for **numeric** sleep sensors to consider the person sleeping. Does not apply to binary sensors. Default: **75** |
| **Device Tracker** *(optional)* | A specific `device_tracker` entity to use for home/away detection instead of the person entity. When set, this tracker's state is checked instead of `person.<name>`. Leave empty to use the person entity (default). |

You can add multiple people, each assigned to a different (or the same) sleep area.

!!! example "Supported sleep sensors"
    - **Android Companion App**: `sensor.<name>_sleep_confidence` (numeric, 0-100)
    - **iOS Sleep Focus**: `binary_sensor.<name>_sleep_focus` (binary)
    - **Withings Sleep Mat**: `binary_sensor.<name>_in_bed` (binary)
    - **Any binary sensor** that turns `on` when sleeping

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

- Person entities configured in Home Assistant
- At least one sleep detection sensor per person (Companion App, in-bed sensor, Sleep Focus, etc.)
- At least one area configured in the integration

## Tips

- **Threshold tuning:** Start with the default threshold of 75 for numeric sensors. If the sensor doesn't trigger reliably, decrease it. If it triggers too easily (e.g., while watching TV in bed), increase it. Binary sensors ignore the threshold entirely.
- **Multiple sensors per person:** You can combine multiple sleep sensors for better reliability. For example, use both a phone sleep confidence sensor and a Withings sleep mat. If **any** sensor is active, the person is considered sleeping.
- **Multiple bedrooms:** Each person can be assigned to a different sleep area, making this work well for households with multiple bedrooms.
- **Interaction with Bedroom purpose:** Sleep presence works alongside the [Bedroom purpose](purpose.md) decay settings. During sleep, the combination of long bedroom decay and the sleep sensor's own 2-hour decay half-life provides very stable occupancy.

!!! note "Migration from single sensor"
    If you previously configured a single sleep confidence sensor per person, the integration automatically migrates this to the new multi-sensor format. Your existing configuration will continue to work without any changes.
