# Manual occupancy references

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
or learned likelihoods. The latest reference is exposed separately through `manual_reference_*`
attributes. The same attributes include the number of references with a
captured automatic classification, the number of agreements, and the
agreement rate for that area. Once 20 usable references exist, a
calibration-only shadow probability is also exposed. It does not replace the
live probability.

These references are kept independent from the existing daily accuracy
tallies. They provide persisted human-vs-automatic observations for evaluating
the detector, priors, sensor evidence, and shadow models without changing the
live detector or its learning inputs. References without an automatic
probability are retained but are not included in calibration or the agreement
rate.

The feedback also snapshots each configured entity's active state. This
produces per-sensor balanced-accuracy values in
`manual_sensor_trust_shadow`, with Laplace smoothing. These values are
diagnostic candidates for future sensor-weight learning; they do not change
sensor weights yet.

## Retention and failure behavior

References are stored in Home Assistant's `.storage` area. Records older than
365 days, records for removed areas, and malformed records are discarded. At
most 1,000 records are retained. Repeating the same state for the same area
within five seconds is rejected as a duplicate. A storage failure is returned
to the service caller and cannot change the live occupancy state.
