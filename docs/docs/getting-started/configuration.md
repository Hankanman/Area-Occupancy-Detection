# Configuration

Area Occupancy Detection is configured entirely through the Home Assistant user interface.

## Adding a New Area

1.  **Navigate to Integrations:** Go to **Configuration** -> **Devices & Services** -> **Integrations**.
2.  **Add Integration:** Click the **+ Add Integration** button in the bottom right.
3.  **Search:** Search for "Area Occupancy Detection" and select it.
4.  **Configure Area Name:**
    - Enter a descriptive **Name** for the area this instance will monitor (e.g., "Living Room", "Office"). This name will be used in entity IDs.

## Configuration Options

### Area Purpose

The first step after naming the area is choosing its **purpose**. This sets a sensible default for the decay half life used when probability decreases. The purpose selection affects how quickly the system "forgets" about occupancy after activity stops.

| Purpose                | Description                                                                                                                                                                           | Default Half-Life        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **Passageway**         | Quick walk-through: halls, stair landings, entry vestibules. Motion evidence should disappear almost immediately after the last footstep.                                             | Very short (~45 sec)     |
| **Utility**            | Laundry room, pantry, boot room. Short functional visits (grab the detergent, put on shoes) with little lingering.                                                                    | Short (~90 sec)          |
| **Food-Prep**          | Kitchen work zone around the hob or countertop. Residents step away to the fridge or sink and return; a few minutes of memory prevents flicker.                                       | Moderate (~4 min)        |
| **Bathroom**           | Showers, baths, getting ready. Motion can be obstructed or minimal; a moderate memory prevents darkness during a shower.                                                              | Moderate-long (~7.5 min) |
| **Eating**             | Dining table, breakfast bar. Family members usually stay seated 10-20 minutes but may be fairly still between bites.                                                                  | Moderate-long (~8 min)   |
| **Working / Studying** | Home office, homework desk. Long seated sessions with occasional trips for coffee or printer; ten-minute half-life avoids premature "vacant".                                         | Long (~10 min)           |
| **Social**             | Living room, play zone, game area. Conversations or board games create sporadic motion; evidence fades gently to ride out quiet pauses.                                               | Long (~8 minutes)        |
| **Relaxing**           | TV lounge, reading nook, music corner. People can remain very still while watching or reading; a quarter-hour memory keeps the room "occupied" through stretches of calm.             | Long (~10 minutes)       |
| **Sleeping**           | Bedrooms, nap pods. Motion is scarce; a long half-life prevents false vacancy during deep sleep yet lets the house revert to "empty" within a couple of hours after everyone gets up. | Very long (~20 min)      |

You can override the resulting half life in the parameters section if needed.

**Global Sleep Schedule:**
For areas with the `Sleeping` purpose, the half-life dynamically adjusts based on your configured sleep schedule. You can set your household's `Sleep Start` and `Sleep End` times in the **Global Settings** menu (accessible via the main integration options). Outside of sleep hours, `Sleeping` areas behave like `Relaxing` areas.

After providing the name and purpose, you'll be guided through selecting sensors and configuring parameters. You can also reconfigure these later by clicking **Configure** on the integration card.

### Sensor Selection

You will be prompted to select entities for various categories. You only need to select sensors relevant to the specific area you are configuring.

| Sensor Type                  | Entity Type                         | Description                                                          | Default States/Range |
| ---------------------------- | ----------------------------------- | -------------------------------------------------------------------- | -------------------- |
| Motion Sensors               | `binary_sensor`                     | Additional motion sensors in the area such as PIR or mmWave sensors. | `on`                 |
| Door Sensors                 | `binary_sensor`                     | Relevant door sensors.                                               | `Closed`             |
| Window Sensors               | `binary_sensor`                     | Relevant window sensors.                                             | `Open`               |
| Media Devices                | `media_player`                      | Relevant media players.                                              | `playing`, `paused`  |
| Appliances                   | `switch`, `binary_sensor`, `sensor` | Relevant switch or sensor entities representing appliances.          | `on`, `standby`      |
| Illuminance Sensors          | `sensor`                            | Illuminance sensors measuring light levels (lux)                     | `30.0 - 100000.0`    |
| Temperature Sensors          | `sensor`                            | Temperature sensors measuring temperature                            | `18.0 - 24.0`        |
| Humidity Sensors             | `sensor`                            | Humidity sensors measuring humidity                                  | `70.0 - 100.0`       |
| CO2 Sensors                  | `sensor`                            | Carbon dioxide sensors measuring CO2 levels (ppm)                    | `400.0 - 1200.0`     |
| Sound Pressure Sensors       | `sensor`                            | Sound pressure sensors measuring noise levels in decibels (dB)       | `40.0 - 80.0`        |
| Atmospheric Pressure Sensors | `sensor`                            | Atmospheric pressure sensors measuring air pressure (hPa)            | `980.0 - 1050.0`     |
| Air Quality Index Sensors    | `sensor`                            | Air quality index sensors measuring overall air quality              | `50.0 - 150.0`       |
| VOC Sensors                  | `sensor`                            | Volatile organic compound sensors measuring VOC levels (ppb)         | `200.0 - 1000.0`     |
| PM2.5 Sensors                | `sensor`                            | Particulate matter sensors measuring PM2.5 levels (µg/m³)            | `12.0 - 55.0`        |
| PM10 Sensors                 | `sensor`                            | Particulate matter sensors measuring PM10 levels (µg/m³)             | `55.0 - 155.0`       |
| Energy Sensors               | `sensor`                            | Energy sensors measuring power consumption (kWh)                     | `0.1 - 10.0`         |

### Available States

Different sensor types support different state configurations:

#### Door Sensors

- **Open** - Door is open
- **Closed** - Door is closed (default)

#### Window Sensors

- **Open** - Window is open (default)
- **Closed** - Window is closed

#### Media Devices

- **Playing** - Media is actively playing (default)
- **Paused** - Media is paused
- **Idle** - Media device is idle
- **Off** - Media device is off

#### Appliances

- **On** - Appliance is active (default)
- **Off** - Appliance is off
- **Standby** - Appliance is in standby mode

#### Motion Sensors

- **Active** - Motion detected (default)
- **Inactive** - No motion detected

### Parameters

| Parameter                 | Description                                                                                                               | Range      | Default          |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------- | ---------------- |
| Occupancy Threshold (%)   | The probability percentage required for the main **Occupancy Status** binary sensor to turn `on`                          | 1-99       | 50               |
| History Period (Days)     | The number of past days to analyze when performing [Prior Probability Learning](../features/prior-learning.md)            | 1-90       | 7                |
| Decay Enabled             | Toggle whether to enable the [Probability Decay](../features/decay.md) feature                                            | True/False | Enabled          |
| Decay Half Life (Seconds) | When decay is enabled this defines how long it takes for the occupancy probability to reduce by half after activity stops | 10-3600    | Based on purpose |
| Motion Timeout (Seconds)  | How long motion sensors remain "active" after detecting motion before automatically resetting to inactive                 | 0-3600     | 300 (5 minutes)  |

### Wasp in Box Feature

The "Wasp in Box" is a virtual sensor that combines motion and door sensor data to create a more reliable occupancy indicator. It's particularly useful for areas where people might be still for extended periods.

| Parameter               | Description                                                            | Range           | Default               |
| ----------------------- | ---------------------------------------------------------------------- | --------------- | --------------------- |
| **Wasp in Box Enabled** | Toggle the Wasp in Box virtual sensor                                  | True/False      | Disabled              |
| **Motion Timeout**      | How long motion sensors remain active after detecting motion           | 0-3600 seconds  | 300 seconds           |
| **Weight**              | Influence of the Wasp in Box sensor on final probability calculation   | 0.0-1.0         | 0.8                   |
| **Max Duration**        | Maximum time the Wasp in Box can indicate occupancy without new motion | 0-86400 seconds | 3600 seconds (1 hour) |

### Sensor Weights

Adjust the influence of different _types_ of sensors on the final probability calculation. Weights range from 0.0 (no influence) to 1.0 (maximum influence). Default values are provided based on typical sensor reliability for occupancy.

| Sensor Type          | Default Weight |
| -------------------- | -------------- |
| Motion Sensor        | 1.00           |
| Media Device         | 0.70           |
| Appliance            | 0.40           |
| Door Sensor          | 0.30           |
| Energy Sensor        | 0.30           |
| Window Sensor        | 0.20           |
| Environmental Sensor | 0.10           |
| Wasp in Box          | 0.80           |

## Reconfiguring an Existing Area

1.  Go to **Configuration** -> **Devices & Services** -> **Integrations**.
2.  Find the Area Occupancy Detection integration card for the area you want to change.
3.  Click **Configure**.
4.  You can then step through and modify any of the sensor selections, parameters, or weights.

Click **Submit** on each step to save changes.

## Automatic Learning

The integration automatically:

- Analyzes historical data to determine sensor reliability
- Calculates correlation between sensor states and occupancy
- Adjusts probabilities based on learned patterns
- Updates calculations in real-time

## Created Entities

After configuration, the integration creates:

1. **Occupancy Probability Sensor** – Shows the calculated probability as a percentage.
2. **Occupancy Status Binary Sensor** – Indicates if the area is occupied based on the threshold.
3. **Prior Probability Sensor** – Displays the combined prior used for calculations.
4. **Evidence Sensor** – Lists entities providing evidence and those that are inactive.
5. **Decay Status Sensor** – Indicates progress of probability decay.
6. **Occupancy Threshold Number** – Allows adjusting the threshold used by the binary sensor.
7. **Wasp in Box Sensor** – Virtual sensor combining motion and door data (if enabled).

## Adjusting Configuration

You can adjust settings anytime through the integration options:

1. Go to Settings → Devices & Services
2. Find Area Occupancy Detection
3. Click "Configure"
4. Modify settings as needed

## Next Steps

After configuration:

1. Monitor the created entities to ensure they reflect actual occupancy
2. Adjust the threshold if needed
3. Review the [Basic Usage](basic-usage.md) guide
