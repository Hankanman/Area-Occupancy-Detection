# Manual Controls and Room Tiles

Area Occupancy exposes a temporary manual state for cases where the sensors
cannot see the room. The override changes the visible occupancy binary sensor,
but it does **not** change the probability, priors, likelihoods, or recorder
data used for learning.

Call `area_occupancy.set_manual_occupancy` with:

| Field | Values |
|---|---|
| `area_id` | The Home Assistant area ID |
| `state` | `occupied`, `unoccupied`, or `clear` |
| `duration` | Seconds, from 1 second to 7 days |

`clear` returns to automatic detection immediately. A timed override also
returns to automatic detection when it expires, including after a restart.
Malformed or expired stored overrides are discarded. Missing sensor values do
not extend an override and, after expiry, the normal automatic result is used.

The occupancy binary sensor exposes `state_source: manual` while an override
is active, together with its expiry timestamp. The probability sensor remains
the automatic probability and exposes active and fading evidence so a
dashboard can show uncertainty honestly.

## Room tiles

The bundled frontend module also registers
`custom:area-occupancy-dashboard-card`. It uses existing entities and offers
one-hour occupied, unoccupied, and automatic buttons:

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

The entity IDs are explicit so renamed areas do not silently control the
wrong room. The tile shows the automatic probability, the state source, and
whether evidence is active or fading.
