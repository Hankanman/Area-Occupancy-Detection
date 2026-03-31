# Sensor Health Monitoring

The integration continuously monitors the health of all configured sensors and alerts you through Home Assistant's **Repairs** system when issues are detected. This helps you identify degraded sensors that may be reducing occupancy detection accuracy.

## How It Works

During the hourly analysis cycle, the integration checks each sensor for:

- **Stuck active** — a binary sensor has been continuously active for longer than expected
- **Stuck inactive** — a binary sensor hasn't changed state for an unusually long time
- **Unavailable** — a sensor has been offline for more than 1 hour
- **Never triggered** — a sensor has never been active since it was added (checked after 7 days)

When an issue is detected, a repair entry appears in **Settings → System → Repairs** with a description of the problem and actionable troubleshooting steps. Repairs are automatically dismissed when the sensor recovers.

## Detection Thresholds

Thresholds are tuned per sensor type to avoid false positives:

### Stuck Active (sensor continuously "on")

| Sensor Type | Threshold | Rationale |
|------------|-----------|-----------|
| Motion | 2 hours | PIR sensors should cycle on/off frequently |
| Media | 12 hours | TVs/speakers may run for hours but not overnight |
| Appliance | 24 hours | Ovens/washers can run for hours but not a full day |
| Door | 48 hours | Doors can legitimately stay open for a day or two |
| Window | 72 hours | Windows may stay open for days in warm weather |
| Cover | 24 hours | Covers shouldn't stay in transition for a full day |

### Stuck Inactive (sensor never changes state)

| Sensor Type | Threshold | Rationale |
|------------|-----------|-----------|
| Motion | 7 days | A motion sensor should detect something within a week |
| Media | 14 days | Media devices should be used within two weeks |
| Appliance | 28 days | Some appliances are used infrequently |
| Door | 14 days | Doors should open/close within two weeks |
| Window | 14 days | Windows should be opened within two weeks |
| Cover | 14 days | Covers should be used within two weeks |
| Power | 14 days | Power sensors should show variation within two weeks |

### Other Checks

| Check | Threshold | Description |
|-------|-----------|-------------|
| Unavailable | 1 hour | Sensor has been offline (dead battery, connectivity loss) |
| Never triggered | 7 days | Sensor has never been active since it was added to the integration |

## Excluded Sensors

The following sensors are excluded from health checks:

- **Sleep sensors** — virtual sensors that are managed by the integration
- **Wasp in Box sensors** — virtual sensors for bathroom occupancy
- **Environmental sensors** (temperature, humidity, illuminance, etc.) — only checked for unavailability, not for "stuck" states, since their values change continuously

## Repair Issues

Repair entries appear in **Settings → System → Repairs** and include:

- The affected sensor's entity ID and area
- How long the issue has persisted
- The sensor type and applicable threshold
- Specific troubleshooting guidance

### Severity Levels

| Issue Type | Severity | Description |
|-----------|----------|-------------|
| Stuck active | Error | Sensor is likely malfunctioning |
| Unavailable | Error | Sensor is offline — occupancy detection is degraded |
| Stuck inactive | Warning | Sensor may be dead or misconfigured |
| Never triggered | Warning | Sensor may be misconfigured or unnecessary |

### Auto-Resolution

Repair entries are **automatically deleted** when the issue resolves:

- A stuck sensor changes state → repair disappears
- An unavailable sensor comes back online → repair disappears
- A never-triggered sensor fires for the first time → repair disappears

You can also manually dismiss repairs in the HA UI if you've investigated and determined the issue is expected.

## Sensor Health Entity

Each area gets a diagnostic sensor entity that exposes health status:

**`sensor.<area_name>_sensor_health`**

- **State:** Number of current health issues (0 = all healthy)
- **Icon:** `mdi:heart-pulse` when healthy, `mdi:alert-circle` when issues exist
- **Entity Category:** Diagnostic

**Attributes:**

| Attribute | Description |
|-----------|-------------|
| `issues` | List of issue details (entity_id, type, duration, description) |
| `healthy_count` | Number of healthy sensors |
| `total_count` | Total number of sensors in the area |
| `last_check` | Timestamp of the last health check |

The Evidence sensor also includes a `health_status` field in its `details` attribute for each entity, showing one of: `healthy`, `stuck_active`, `stuck_inactive`, `unavailable`, `never_triggered`, or `excluded`.

## Common Scenarios

### Dead Battery

A Zigbee motion sensor runs out of battery and goes `unavailable`. After 1 hour, a repair entry appears:

> **binary_sensor.bathroom_motion_1 is offline in Bathroom**
>
> The motion sensor has been unavailable for 5 hours. The integration is currently falling back to learned priors for this area.

### Stuck Sensor

A PIR sensor gets stuck in the "on" state (hardware malfunction). After 2 hours:

> **binary_sensor.toilet_motion_1 appears stuck in Toilet**
>
> The motion sensor has been continuously active for 3 hours, which exceeds the 2-hour threshold.

### Misconfigured Sensor

A kitchen oven binary sensor is configured but the power threshold is too high, so it never triggers. After 7 days:

> **binary_sensor.kitchen_oven may be misconfigured in Kitchen**
>
> The appliance sensor has never been active in 7 days of monitoring.

## Tips

- Check the **Sensor Health** diagnostic entity in your dashboards to see at-a-glance health across areas
- Use the **Evidence** sensor's `details` attribute to see per-entity health status alongside probability contributions
- If a sensor is intentionally unused, consider removing it from the area configuration rather than ignoring the repair
- The health check runs hourly — new issues may take up to 1 hour to appear after a sensor degrades
