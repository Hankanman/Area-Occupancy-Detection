# Configuration

Area Occupancy Detection is configured entirely through the Home Assistant user interface.

## Adding a New Area

1.  **Navigate to Integrations:** Go to **Configuration** -> **Devices & Services** -> **Integrations**.
2.  **Add Integration:** Click the **+ Add Integration** button in the bottom right.
3.  **Search:** Search for "Area Occupancy Detection" and select it.
4.  **Configure Area Name:**
    *   Enter a descriptive **Name** for the area this instance will monitor (e.g., "Living Room", "Office"). This name will be used in entity IDs.

## Configuration Options

### Area Purpose

The first step after naming the area is choosing its **purpose**. This sets a sensible default for the decay half life used when probability decreases. Purposes include options like *Passageway*, *Utility*, *Social*, and *Sleeping*. You can override the resulting half life in the options if needed.

After providing the name, you'll be guided through selecting sensors and configuring parameters. You can also reconfigure these later by clicking **Configure** on the integration card.

### Sensor Selection

You will be prompted to select entities for various categories. You only need to select sensors relevant to the specific area you are configuring.

| Sensor Type | Entity Type | Description | Active States |
|-------------|-------------|-------------|---------------|
| Primary Occupancy Sensor (Required) | `binary_sensor` | One reliable motion or occupancy sensor. Crucial as ground truth for [Prior Probability Learning](../features/prior-learning.md). | `on` |
| Motion Sensors | `binary_sensor` | Additional motion sensors in the area. | `on` |
| Door Sensors | `binary_sensor` | Relevant door sensors. | Default: `Closed` |
| Window Sensors | `binary_sensor` | Relevant window sensors. | Default: `Open` |
| Media Devices | `media_player` | Relevant media players. | Default: `playing`, `paused` |
| Lights | `light` | Relevant light entities. | `on` |
| Appliances | `switch`, `binary_sensor`, `sensor` | Relevant switch or sensor entities representing appliances. | Default: `on`, `standby` |
| Environmental Sensors (Optional) | `sensor` | - Illuminance sensors measuring light levels (lux)<br>- Temperature sensors measuring temperature<br>- Humidity sensors measuring humidity<br>*(Note: Environmental sensors typically have a lower default weight and may require more history for their priors to become meaningful)* | N/A |

### Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|--------|---------|
| Occupancy Threshold (%) | The probability percentage required for the main **Occupancy Status** binary sensor to turn `on` | 1-99 | 50 |
| History Period (Days) | The number of past days to analyze when performing [Prior Probability Learning](../features/prior-learning.md) | 1-90 | 7 |
| Decay Enabled | Toggle whether to enable the [Probability Decay](../features/decay.md) feature | True/False | Enabled |
| Decay Half Life (Seconds) | When decay is enabled this defines how long it takes for the occupancy probability to reduce by half after activity stops | 10-3600 | 300 (5 minutes) |

### Sensor Weights

Adjust the influence of different *types* of sensors on the final probability calculation. Weights range from 0.0 (no influence) to 1.0 (maximum influence). Default values are provided based on typical sensor reliability for occupancy.

| Sensor Type | Default Weight |
|-------------|---------------|
| Motion Sensor | 0.85 |
| Media Device | 0.70 |
| Appliance | 0.40 |
| Door Sensor | 0.30 |
| Window Sensor | 0.20 |
| Light | 0.20 |
| Environmental Sensor | 0.10 |

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

1. **Occupancy Probability Sensor**

      - Shows the calculated probability as a percentage
      - Updates in real-time based on sensor states

2. **Occupancy Status Binary Sensor**

      - ON when probability exceeds threshold
      - OFF when probability is below threshold

3. **Individual Prior Sensors**

      - One for each sensor category
      - Shows contribution to overall probability

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
