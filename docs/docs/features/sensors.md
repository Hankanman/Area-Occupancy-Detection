# Sensors

## Sensor Selection

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
| CO Sensors                   | `sensor`                            | Carbon monoxide sensors measuring CO levels (ppm)                    | `5.0 - 50.0`         |
| Sound Pressure Sensors       | `sensor`                            | Sound pressure sensors measuring noise levels in decibels (dB)       | `40.0 - 80.0`        |
| Atmospheric Pressure Sensors | `sensor`                            | Atmospheric pressure sensors measuring air pressure (hPa)            | `980.0 - 1050.0`     |
| Air Quality Index Sensors    | `sensor`                            | Air quality index sensors measuring overall air quality              | `50.0 - 150.0`       |
| VOC Sensors                  | `sensor`                            | Volatile organic compound sensors measuring VOC levels (ppb)         | `200.0 - 1000.0`     |
| PM2.5 Sensors                | `sensor`                            | Particulate matter sensors measuring PM2.5 levels (µg/m³)            | `12.0 - 55.0`        |
| PM10 Sensors                 | `sensor`                            | Particulate matter sensors measuring PM10 levels (µg/m³)             | `55.0 - 155.0`       |
| Power Sensors                | `sensor`                            | Power sensors measuring power consumption (W/kW)                     | `0.1 - 10.0`         |

## Sensor Weights

Weights allow you to adjust the influence of different _types_ of sensors on the final probability calculation. Weights range from 0.0 (no influence) to 1.0 (maximum influence). Default values are provided based on typical sensor reliability for occupancy. You can override the default weights in the configuration menu for each sensor type.

| Sensor Type          | Default Weight |
| -------------------- | -------------- |
| Motion Sensor        | 1.00           |
| Wasp in Box          | 0.80           |
| Media Device         | 0.70           |
| Appliance            | 0.40           |
| Door Sensor          | 0.30           |
| Power Sensor         | 0.30           |
| Window Sensor        | 0.20           |
| Environmental Sensor | 0.10           |
