# Services

The Area Occupancy Detection integration provides services that can be called from automations or scripts.

## `area_occupancy.run_analysis`

Runs the historical analysis process for all areas in the Area Occupancy instance. This imports recent state data from the recorder, updates priors and likelihoods, and refreshes the coordinator.

**Example:**

```yaml
service: area_occupancy.run_analysis
```

**Returns:**

- `areas`: Dictionary mapping area names to their analysis data. Each area contains:
  - `area_name`: Name of the area
  - `current_prior`: Current global prior probability
  - `global_prior`: Global prior after analysis
  - `time_prior`: Time-based prior used in calculations
  - `prior_entity_ids`: List of entities included in analysis
  - `total_entities`: Total number of entities
  - `entity_states`: Current states of all entities
  - `likelihoods`: Updated likelihood data per entity. Each entity in `likelihoods` contains:
    - `analysis_error`: Error code indicating why analysis failed or was not performed (see below)
- `update_timestamp`: ISO timestamp of when the analysis completed
- `analysis_time_ms`: Total time taken for the full analysis in milliseconds (float)
- `device_sw_version`: Integration software version string

### `analysis_error` Values

The `analysis_error` field in each entity's likelihood data indicates why correlation analysis failed or was not performed. Possible values:

- `null` - Analysis completed successfully with no errors
- `"not_analyzed"` - Entity has not been analyzed yet (default state for non-motion sensors before first analysis)
- `"motion_sensor_excluded"` - Motion sensors are excluded from correlation analysis by design, as they are used to determine occupancy rather than correlate with it
- `"no_occupied_intervals"` - No occupied time intervals were found in the analysis period, so correlation cannot be calculated
- `"no_occupied_time"` - The total occupied time in the analysis period is zero or negative, indicating insufficient occupancy data
- `"no_unoccupied_time"` - The total unoccupied time in the analysis period is zero or negative, indicating the area was occupied for the entire period
- `"no_sensor_data"` - No sensor interval data was found for this entity in the analysis period
- `"no_occupied_samples"` - No sensor samples were found when the area was occupied, preventing correlation calculation
- `"no_unoccupied_samples"` - No sensor samples were found when the area was unoccupied, preventing correlation calculation
- `"no_correlation"` - The correlation coefficient is below the moderate threshold, indicating no meaningful correlation between the sensor and occupancy
- `"too_few_samples"` - Insufficient samples collected for reliable correlation analysis (below minimum threshold)
- `"too_few_samples_after_filtering"` - After filtering samples, there are insufficient samples remaining for correlation analysis
- `"no_occupancy_data"` - No occupied intervals were found for analysis validation

**Notes:**

- This service always runs analysis for all configured areas.
- Services that query historical data can be resource-intensive.
- Analysis results (including `analysis_error` values) are persisted to the database and will be restored when entities are reloaded. This ensures that `analysis_error` values are preserved across Home Assistant restarts and entity reloads.

## `area_occupancy.get_time_priors`

Returns the learned **weekly occupancy forecast** for every area: 7 days × 24
hourly slots (168 per area). Unlike `occupancy_probability`, which estimates the
present, this can be queried for *future* slots — so a climate controller can
pre-heat a room before its habitual occupancy instead of reacting to it.

This is a response-only service (`SupportsResponse.ONLY`): it reads state and
never modifies it.

**Example:**

```yaml
action: area_occupancy.get_time_priors
data:
  area_id: studio   # optional; omit for every area
response_variable: forecast
```

**Returns:**

- `slot_minutes`: Slot resolution in minutes (60)
- `areas`: Dictionary mapping area name to:
    - `area_id`: The area's Home Assistant area id
    - `global_prior`: The area's learned area-wide prior (absent on aggregate zones)
    - `slot_minutes`: Slot resolution, echoed per area
    - `current_slot`: the `"day,slot"` key covering *now* — the anchor that
      aligns the live estimate to the weekly grid
    - `threshold`: the area's occupancy threshold
    - `tau_slots`: how many slots the current evidence keeps influencing the
      forecast, derived from the area's purpose
    - `slots`: `"day,slot"` → **evidence-conditioned** forecast. `day` is
      0=Monday…6=Sunday, `slot` is the hour index
    - `slots_baseline`: `"day,slot"` → the same forecast *without* live evidence
    - `slots_raw`: `"day,slot"` → the learned time prior itself, uncombined
    - `data_points`: `"day,slot"` → weeks of observation behind the slot
    - `aggregate`, `members`, `name`: present only on aggregate zones
      (*All Areas*, per-floor devices), listing the member area ids

**Which map should I use?**

| Goal | Field |
|---|---|
| Act now — heat if the room is, or is about to be, occupied | `slots` |
| Program a thermostat schedule that must not move between polls | `slots_baseline` |
| Rank hours, audit the learned shape | `slots_raw` |
| Decide whether to trust a slot at all | `data_points` |

`slots` folds in what the area knows *right now*: at `current_slot` it equals the
area's `occupancy_probability` exactly, and the lift relaxes back to habit over the
next few slots. That makes it a forecast **issued at the moment of the call** —
two calls a minute apart can differ, by design.

`slots_baseline` is the stable weekly matrix: it only changes when the hourly
analysis relearns, which is what an old-fashioned weekly programme needs.

`slots_raw` is the learned time prior alone. Both `slots` and `slots_baseline` are
blended with the area's global prior, which keeps 60% of the weight, so their
dynamic range is compressed; `slots_raw` is not. See
[Occupancy Forecast](../technical/occupancy-forecast.md#dynamic-range).

!!! warning "Always check `data_points`"
    A slot with `data_points: 0` was never observed. Its prior is a neutral
    placeholder, not a measurement — skip it rather than acting on it. Freshly
    installed areas, and hours the analysis window never covered, report zero.

**Example automation** — pre-heat when the next hour is habitually occupied:

```yaml
triggers:
  - trigger: time_pattern
    minutes: "0"
actions:
  - action: area_occupancy.get_time_priors
    data:
      area_id: studio
    response_variable: forecast
  - variables:
      slot: >-
        {{ (now() + timedelta(hours=1)).weekday() }},{{ (now() + timedelta(hours=1)).hour }}
      area: "{{ forecast.areas['Studio'] }}"
  - condition: template
    value_template: >-
      {{ area.data_points[slot] | int > 0 and area.slots[slot] | float > 0.5 }}
  - action: climate.set_temperature
    target:
      entity_id: climate.studio
    data:
      temperature: 21
```

**Notes:**

- `slots_baseline`, `slots_raw` and `data_points` are refreshed by the hourly
  analysis; calling more often than that returns the same values for them.
  `slots` changes continuously, because it tracks the live estimate.
- A companion Lovelace card visualises the matrix — see `lovelace/README.md`.

## `area_occupancy.export_config`

Exports the complete integration configuration as YAML. This is useful for debugging, sharing your setup when reporting issues, or backing up your configuration.

**Example:**

```yaml
service: area_occupancy.export_config
```

**Returns:**

The full merged configuration (config entry data + options) as a dictionary. Each area configuration is reordered so that `area_id` appears first for readability. The response includes:

- All area configurations (sensors, weights, thresholds, decay settings, purpose, etc.)
- People configurations (person entities, sleep sensors, sleep areas)
- Global settings (sleep schedule)

!!! tip "Viewing the output"
    Call this service from **Developer Tools > Services** in Home Assistant. The response is rendered as YAML directly in the UI, making it easy to review or copy your full configuration.

## `area_occupancy.purge_area_history`

Deletes **all learned history** for a single configured area without removing the area itself. This clears the area's intervals, priors, correlations, aggregates, and cached occupied intervals from the database, then reloads and refreshes the coordinator so the UI immediately reflects the purge.

Use this when a room's learned behaviour is no longer accurate — for example after significantly changing the sensor layout, swapping hardware, or repurposing a room — and you want the integration to re-learn from scratch.

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `area_id` | Yes | The Home Assistant `area_id` whose learned history should be purged. Use the area selector in the UI or the raw `area_id` string in YAML. |

**Example:**

```yaml
service: area_occupancy.purge_area_history
data:
  area_id: living_room
```

**Returns:**

| Key | Description |
|-----|-------------|
| `area_id` | The area_id that was purged |
| `area_name` | The area's display name |
| `entities_deleted` | Number of entity rows removed from the database for this area |
| `shell_repersisted` | `true` if the empty area shell was successfully re-saved, `false` on a non-fatal re-persist failure (the purge itself still succeeds; the shell is recreated on the next save cycle) |
| `purged_at` | ISO timestamp of when the purge completed |

**Errors:**

- Calling the service with an unknown `area_id` raises a `ServiceValidationError` listing the currently configured area_ids.

!!! warning "This is destructive"
    All learned priors, correlations, intervals, and aggregates for the selected area are permanently deleted. The integration will start re-learning from scratch on the next analysis cycle (hourly by default). Other areas are unaffected.

!!! tip "Resetting everything"
    To wipe learned history for *every* area, remove the integration entirely (which now also deletes the database file — see the [2026.4.1 release notes](https://github.com/Hankanman/Area-Occupancy-Detection/releases/tag/2026.4.1)) and reinstall. Use `purge_area_history` when you only want to reset one area.

### Resetting from the UI

The same purge is available without writing a service call. Open the integration's options (**Settings → Devices & Services → Area Occupancy Detection → Configure**), pick **Manage Areas**, choose the area, and click **Reset Learning**. A yes/no confirmation appears before anything is deleted, and you'll land back on the area's management menu when it's done. Same destructive behaviour — same trade-offs as the service call above.
