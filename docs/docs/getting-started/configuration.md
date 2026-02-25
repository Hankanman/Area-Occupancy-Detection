# Configuration

Area Occupancy Detection is built for simplicity: just specify the sensors available in each area and a purpose, the integration will then intelligently handle occupancy detection for you. While some customization options are available, the system’s powerful history-based intelligence learns your habits over time—automatically refining its accuracy without constant manual tweaking.

### Before You Start

Almost every option in the config is optional, sensible defaults are available for everything. The minimum configuration for an area is:

- A Home Assistant Area. Must exist in Home Assistant first, [see here to set up areas](https://www.home-assistant.io/docs/organizing/areas/)
- A Purpose. What the room is used for, [see more about purposes here](../features/purpose.md)
- 1 Motion sensor. A physical device in the area like PIR, mmWave

The integration will work with just these configured. Everything else can be added as you get new devices. However the more you add in, the more accurate the predictions will be.

## Configuration Wizard

Area configuration uses a multi-step wizard that walks you through setup one section at a time:

### Step 1: Area Basics

Select the Home Assistant area and its primary [Purpose](../features/purpose.md). The purpose sets a sensible default for the [decay](../features/decay.md) half-life used when probability decreases.

The following purposes are available (in order of decay time, shortest to longest):

- Passageway
- Driveway
- Utility
- Garage
- Kitchen
- Garden
- Bathroom
- Dining Room
- Living Room
- Office
- Media Room
- Bedroom

You can override the resulting half-life in the Detection Behavior step if needed.

**Global Sleep Schedule:**
For areas with the `Bedroom` purpose, the half-life dynamically adjusts based on your configured sleep schedule. You can set your household’s `Sleep Start` and `Sleep End` times in the **Global Settings** menu (accessible via the main integration options). Outside of sleep hours, `Bedroom` areas behave like `Living Room` areas.

### Step 2: Motion Sensors

Configure motion and presence sensors for the area. At least one motion sensor is required. You can also adjust:

- **Motion Weight**: How much influence motion sensors have on the calculation
- **Motion Timeout**: Additional timeout applied to historical motion data (configured using a duration picker)

### Step 3: Additional Sensors

Configure optional sensors grouped into collapsible sections. You only need to add [sensors](../features/sensors.md) that are relevant to the area:

- **Windows, Doors & Covers**: Door sensors, window sensors, cover entities with active state configuration
- **Media Players**: Media devices with active state selection
- **Appliances**: Switches and appliances with active state selection
- **Environmental Sensors**: Illuminance, temperature, humidity, CO2, CO, sound, pressure, air quality, VOC, PM2.5, PM10
- **Power Sensors**: Power consumption monitoring

More information on the [Sensors](../features/sensors.md) page.

### Step 4: Detection Behavior

Configure how occupancy is detected and reported:

| Parameter                   | Description                                                                                                                                 | Default          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| Occupancy Threshold (%)     | The probability percentage required for the main **Occupancy Status** binary sensor to turn `on`                                            | 50%              |
| Enable Time Decay           | Toggle whether to enable the [Probability Decay](../features/decay.md) feature                                                              | Enabled          |
| Decay Half-Life             | How long it takes for probability to reduce by half after activity stops (duration picker)                                                  | Based on purpose |
| Minimum Prior Override      | The minimum prior probability to use, overriding learned patterns. Set to 0 to disable.                                                    | 0 (disabled)     |
| Exclude from All Areas      | When enabled, this area won’t contribute to the [All Areas](../features/entities.md#all-areas-aggregation-device) or floor aggregate sensors. Useful for garages, driveways, and outdoor areas. | Disabled |

This step also includes the [Wasp in Box](../features/wasp-in-box.md) configuration section for single-entry rooms.

## People Management

You can configure sleep presence detection through the **Manage People** option in the integration’s main menu. This allows the integration to detect when people are sleeping and maintain high occupancy probability in bedrooms overnight.

For each person, you configure:

- **Person Entity**: The `person.<name>` entity to track
- **Sleep Sensors**: One or more sleep detection sensors (supports both numeric sensors like Companion App sleep confidence and binary sensors like in-bed sensors or iOS Sleep Focus)
- **Sleep Area**: Which area this person sleeps in
- **Confidence Threshold**: Minimum value for numeric sleep sensors to consider the person sleeping (default: 75). Does not apply to binary sensors.
- **Device Tracker** *(optional)*: A specific device tracker for home/away detection instead of the person entity

When configured, a **Sleeping** binary sensor is created for each assigned area. See [Sleep Presence](../features/sleep-presence.md) for details.

## Next Steps

After configuration:

1. Monitor the created entities to ensure they reflect actual occupancy
2. Adjust the threshold if needed
3. Review the [Basic Usage](basic-usage.md) guide
