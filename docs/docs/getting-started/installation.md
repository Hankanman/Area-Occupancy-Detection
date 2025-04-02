# Installation

The Area Occupancy Detection integration can be installed either through HACS (recommended) or manually.

## Prerequisites

Before installing, ensure your Home Assistant installation has:

1. Required Integrations:

      - recorder (for historical data analysis)
      - sensor (for probability sensors)
      - binary_sensor (for occupancy status)
      - number (for threshold adjustment)

2. Recommended Sensors:

      - At least one motion sensor
      - Additional sensors for better accuracy:
        - Media players
        - Appliances
        - Door/window sensors
        - Light entities
        - Environmental sensors

## HACS Installation (Recommended)

1. Open Home Assistant
2. Go to HACS (Home Assistant Community Store)
3. Click the "Integrations" section
4. Click the "+" button in the bottom right
5. Search for "Area Occupancy Detection"
6. Click "Download"
7. Restart Home Assistant

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Hankanman&repository=Area-Occupancy-Detection&category=integration)

## Manual Installation

1. Download the latest release from the [GitHub repository](https://github.com/Hankanman/Area-Occupancy-Detection)
2. Extract the `custom_components/area_occupancy` directory
3. Copy it to your Home Assistant's `config/custom_components` directory
4. Restart Home Assistant

## Verification

After installation:

1. Restart Home Assistant
2. Go to Settings → Devices & Services
3. Click "+ Add Integration"
4. Search for "Area Occupancy Detection"
   - If it appears, installation was successful
   - If not, check Home Assistant logs for errors

### Getting Help

If you encounter issues:

1. Search [GitHub Issues](https://github.com/Hankanman/Area-Occupancy-Detection/issues)
2. Join the [Discussion](https://github.com/Hankanman/Area-Occupancy-Detection/discussions)

## Next Steps

After successful installation:

1. Continue to [Configuration](configuration.md)
2. Review [Basic Usage](basic-usage.md)
3. Learn about [Sensor Types](../features/sensor-types.md)
