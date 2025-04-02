# Configuration

This guide will help you configure the Area Occupancy Detection integration through the Home Assistant UI.

## Initial Setup

1. Navigate to Settings → Devices & Services
2. Click "+ Add Integration"
3. Search for "Area Occupancy Detection"
4. Click on the integration to begin setup

## Configuration Steps

### 1. Area Selection

- Enter a name for the area you want to monitor (e.g., "Living Room")
- This will be used to identify all sensors created by the integration

### 2. Primary Motion Sensor

- Select a primary motion or occupancy sensor
- This should be your most reliable sensor in the area
- The integration will use this sensor's history to learn occupancy patterns

### 3. Additional Sensors

#### Motion Sensors

- Add additional motion or occupancy sensors in the area
- Each sensor contributes to the overall occupancy calculation
- The integration automatically determines sensor reliability based on historical data

#### Media Devices

- Select media players in the area (TVs, speakers, etc.)
- The integration automatically detects relevant states (playing, paused, etc.)
- Historical correlation with occupancy is automatically calculated

#### Appliances

- Add smart plugs, switches, or other appliances
- The integration learns which states indicate occupancy
- Common examples: fans, air purifiers, game consoles

#### Doors and Windows

- Select door and window sensors
- The integration learns patterns of use
- Helps distinguish between passing through and occupancy

#### Lights

- Add light entities in the area
- The integration considers both state and brightness
- Patterns are learned from historical usage

#### Environmental Sensors

- Add temperature, humidity, or illuminance sensors
- These provide additional context for occupancy
- Changes in readings can indicate human presence

### 4. Advanced Settings

#### Threshold Adjustment

- Set the probability threshold for occupancy detection (default: 50%)
- Higher values require more certainty before reporting occupancy
- Adjustable through the integration options after setup

#### Historical Analysis

- Choose how many days of history to analyze (default: 7 days)
- Longer periods provide better learning but require more resources
- Can be adjusted based on your Home Assistant's capabilities

#### Time Decay

- Enable/disable time-based probability decay
- Adjust decay window (how long before probabilities start decreasing)
- Set minimum delay between probability updates

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
