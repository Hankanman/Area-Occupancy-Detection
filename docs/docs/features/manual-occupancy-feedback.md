# Manual occupancy references and room tiles

Area Occupancy can record what you observe in a room at this moment. A
reference is diagnostic data: it does not replace the live occupancy state and
does not feed online learning.

Call `area_occupancy.set_manual_occupancy` with:

| Field | Values |
|---|---|
| `area_id` | The Home Assistant area ID |
| `state` | `occupied` or `unoccupied` |

The service stores the UTC timestamp, area ID, confirmed state, and the
automatic probability captured at that moment when available. It does not
change the occupancy binary sensor, probability sensor, recorder data, priors,
or learned likelihoods. The latest reference is exposed separately through
`manual_reference_*` attributes.

## Room tiles

The bundled frontend module registers
`custom:area-occupancy-dashboard-card`:

```yaml
type: custom:area-occupancy-dashboard-card
title: Rooms
areas:
  - name: Living room
    area_id: living_room
    occupancy_entity: binary_sensor.living_room_occupancy_status
    probability_entity: sensor.living_room_occupancy_probability
  - name: Kitchen
    area_id: kitchen
    occupancy_entity: binary_sensor.kitchen_occupancy_status
    probability_entity: sensor.kitchen_occupancy_probability
```

Each tile shows the current automatic state, probability, uncertainty
indicators, and the latest reference. The `Now occupied` and `Now clear`
buttons record a point-in-time reference; they do not create a timed override.
Entity IDs are explicit so renamed areas do not silently control another room.
Missing entities are shown as unavailable.

## Retention and failure behavior

References are stored in Home Assistant's `.storage` area. Records older than
365 days, records for removed areas, and malformed records are discarded. At
most 1,000 records are retained. Repeating the same state for the same area
within five seconds is rejected as a duplicate. A storage failure is returned
to the service caller and cannot change the live occupancy state.
