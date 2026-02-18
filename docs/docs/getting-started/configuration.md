# Configuration

Area Occupancy Detection is built for simplicity: just specify the sensors available in each area and a purpose, the integration will then intelligently handle occupancy detection for you. While some customization options are available, the system’s powerful history-based intelligence learns your habits over time—automatically refining its accuracy without constant manual tweaking.

## Purpose

The first step after naming the area is choosing its [Purpose](../features/purpose.md). This sets a sensible default for the [decay](../features/decay.md) half life used when probability decreases. [Decay](../features/decay.md) half life affects how quickly the system reduces the probability of occupancy after activity stops. This essentially works like a gradual timeout for the area.

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

You can override the resulting half life in the parameters section if needed.

**Global Sleep Schedule:**
For areas with the `Bedroom` purpose, the half life dynamically adjusts based on your configured sleep schedule. You can set your household's `Sleep Start` and `Sleep End` times in the **Global Settings** menu (accessible via the main integration options). Outside of sleep hours, `Bedroom` areas behave like `Living Room` areas.

## Sensor Selection

You will be prompted to select entities for various categories. You only need to select [Sensors](../features/sensors.md) relevant to the specific area you are configuring.

- Motion Sensors (active, inactive)
- Door Sensors (open, closed)
- Window Sensors (open, closed)
- Cover Sensors (opening, closing)
- Media Devices (playing, paused, idle, off)
- Appliances/Switches (on, off, standby)
- Illuminance (lx)
- Temperature (°C/F)
- Humidity Sensors (%)
- CO2 (ppm)
- Sound (dB)
- Atmospheric Pressure (hPa)
- Air Quality Index (AQI)
- VOC (ppb)
- Particulate matter (µg/m³)
- Power (W/kW)

More information on the [Sensors](../features/sensors.md) page.

## People Management

You can configure sleep presence detection through the **Manage People** option in the integration's main menu. This allows the integration to detect when people are sleeping and maintain high occupancy probability in bedrooms overnight.

For each person, you configure:

- **Person Entity**: The `person.<name>` entity to track
- **Sleep Confidence Sensor**: The Companion App sleep confidence sensor
- **Sleep Area**: Which area this person sleeps in
- **Confidence Threshold**: Minimum confidence to consider the person sleeping (default: 75)

When configured, a **Sleeping** binary sensor is created for each assigned area. See [Sleep Presence](../features/sleep-presence.md) for details.

## Wasp in Box

The [Wasp in Box](../features/wasp-in-box.md) is a virtual sensor that combines motion and door sensor data to create a more reliable occupancy indicator. It's particularly useful for areas where there is a single entry/exit point. This is disabled by default. When enabled it will add a new binary sensor entity to the area. More information on the [Wasp in Box](../features/wasp-in-box.md) page.

### Parameters

| Parameter                 | Description                                                                                                                                 | Range      | Default          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------- |
| Occupancy Threshold (%)   | The probability percentage required for the main **Occupancy Status** binary sensor to turn `on`                                            | 1-99       | 50%              |
| Decay Enabled             | Toggle whether to enable the [Probability Decay](../features/decay.md) feature                                                              | True/False | Enabled          |
| Decay Half Life (Seconds) | When decay is enabled this defines how long it takes for the occupancy probability to reduce by half after activity stops                   | 10-3600    | Based on purpose |
| Minimum Prior Override    | The minimum prior probability to use when calculating the occupancy probability. This overrides the learned prior probability for the area. | 0.0-1.0    | 0 (disabled)     |

## Next Steps

After configuration:

1. Monitor the created entities to ensure they reflect actual occupancy
2. Adjust the threshold if needed
3. Review the [Basic Usage](basic-usage.md) guide
