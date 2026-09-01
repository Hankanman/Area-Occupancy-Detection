# Sensors

## Sensor Selection

You will be prompted to select entities for various categories. You only need to select sensors relevant to the specific area you are configuring.

| Sensor Type                  | Entity Type                         | Description                                                          | Default States/Range |
| ---------------------------- | ----------------------------------- | -------------------------------------------------------------------- | -------------------- |
| Motion Sensors               | `binary_sensor`                     | Additional motion sensors in the area such as PIR or mmWave sensors. | `on`                 |
| Door Sensors                 | `binary_sensor`                     | Relevant door sensors.                                               | `Closed`             |
| Lock Sensors                 | `lock`                              | Smart locks (e.g. Nuki) — unlocking is treated as activity evidence. | `Unlocked`            |
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
| Cover Sensors                | `cover`                             | Blinds, shades, shutters, garage doors being operated                | `opening`, `closing`  |
| Power Sensors                | `sensor`                            | Power sensors measuring power consumption (W/kW)                     | `0.1 - 10.0`         |
| Wi-Fi Client Sensors          | `sensor`                            | Sensors reporting connected Wi-Fi client counts (e.g. UniFi Network's connected-clients sensors) for an SSID/AP | `1+` (no upper bound) |
| Custom Binary Sensors         | any                                 | Any entity with no domain or device_class filter, for sensors the sections above reject (e.g. an MQTT/HASS.Agent sensor with a custom on/off-style state). | user-configured (default `on`) |
| Custom Numeric Sensors        | any                                 | Any numeric entity with no domain or device_class filter, for sensors the sections above reject. | user-configured (default `1.0+`) |

!!! note
    Wi-Fi client count is treated as a **presence-channel** signal (like motion, media, or power), not an environmental one — a device joining a network is a much stronger, more direct sign of a person than an ambient reading like temperature. The default active range (`1` and up, unbounded) treats any nonzero client count as evidence of presence, but the useful range varies enormously between networks (a guest SSID might swing 0-1 clients while a busy office SSID swings 5-45), so you will likely want to be selective about which sensors you include per area rather than relying on the raw count alone.

!!! note
    Custom sensors have **no domain or device_class filter at all** — every other section restricts you to entities HA can identify as door, motion, power, etc. Use custom sensors when your entity doesn't fit any of the typed sections above, such as an MQTT or [HASS.Agent](https://hassagent.spablo.com) sensor with unique state semantics. Since the active states/range can't be inferred, you must configure them yourself in the Custom Sensors section — there's no reliable default.

## Sensor Weights

Weights allow you to adjust the influence of different _types_ of sensors on the final probability calculation. Weights range from 0.0 (no influence) to 1.0 (maximum influence). Default values are provided based on typical sensor reliability for occupancy. You can override the default weights in the configuration menu for each sensor type.

| Sensor Type          | Default Weight |
| -------------------- | -------------- |
| Motion Sensor        | 1.00           |
| Sleep                | 0.90           |
| Media Device         | 0.85           |
| Wasp in Box          | 0.80           |
| Cover Sensor         | 0.50           |
| Appliance            | 0.40           |
| Wi-Fi Client Sensor  | 0.35           |
| Custom Binary Sensor | 0.40           |
| Door Sensor          | 0.30           |
| Lock Sensor          | 0.30           |
| Power Sensor         | 0.30           |
| Custom Numeric Sensor | 0.30          |
| Window Sensor        | 0.20           |
| Environmental Sensor | 0.10           |

!!! note
    The **Sleep** weight is not user-configurable per sensor. It is automatically applied when sleep presence is detected via [People Management](sleep-presence.md).
