# Lovelace Card

This integration includes a simple custom card to display the occupancy probability in the Home Assistant dashboard.

## Installation

1. Ensure the integration is installed through HACS or manually.
2. Copy `area-occupancy-card.js` from the `www` folder of this repository into your Home Assistant `config/www` directory if not using HACS.
3. Add the card as a resource in **Settings → Dashboards → Resources** with the URL `/local/area-occupancy-card.js` and type `JavaScript Module`.

## Usage

Add the card to a dashboard using YAML mode or the UI editor.
Example YAML:

```yaml
type: custom:area-occupancy-card
entity: sensor.living_room_occupancy_probability
status_entity: binary_sensor.living_room_occupancy_status
name: Living Room
```

The card shows the current probability, the occupancy status, and a progress bar visualising the probability level.
