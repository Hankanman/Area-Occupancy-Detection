# Services
The Area Occupancy Detection integration provides services that can be called from automations or scripts.

## `area_occupancy.run_analysis`

Runs the historical analysis process for an Area Occupancy instance. This imports recent state data from the recorder, updates priors and likelihoods, cleans up old records and then refreshes the coordinator.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.run_analysis
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `area_name`: Name of the area
- `current_prior`: Current global prior probability
- `global_prior`: Global prior after analysis
- `occupancy_prior`: Prior used for occupancy calculations
- `prior_entity_ids`: List of entities included in analysis
- `total_entities`: Total number of entities
- `import_stats`: Number of state intervals imported per entity
- `total_imported`: Total intervals imported
- `total_intervals`: Total intervals stored
- `entity_states`: Current states of all entities
- `likelihoods`: Updated likelihood data per entity
- `update_timestamp`: ISO timestamp of when the analysis completed



## `area_occupancy.reset_entities`

Resets all entity probabilities and learned data for a specific Area Occupancy instance. This will clear all calculated probabilities and return entities to their initial state.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |
| `clear_storage` | No | Whether to also clear stored data from disk. Defaults to `false`. | `true` |

**Example:**
```yaml
service: area_occupancy.reset_entities
data:
  entry_id: your_config_entry_id_here
  clear_storage: true
```

## `area_occupancy.get_entity_metrics`

Returns basic metrics about entities in the Area Occupancy instance. This service returns data and can be used for monitoring and diagnostics.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.get_entity_metrics
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `total_entities`: Total number of entities
- `active_entities`: Number of entities currently providing evidence
- `available_entities`: Number of available entities
- `unavailable_entities`: Number of unavailable entities
- `decaying_entities`: Number of entities currently in decay state

## `area_occupancy.get_problematic_entities`

Identifies entities that may need attention, such as unavailable entities or those with stale updates. This service returns data for troubleshooting purposes.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.get_problematic_entities
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `unavailable`: List of entity IDs that are currently unavailable
- `stale_updates`: List of entity IDs that haven't been updated in over an hour

## `area_occupancy.get_entity_details`

Returns detailed information about specific entities or all entities if none are specified. This service provides comprehensive data about entity states, probabilities, and configurations.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |
| `entity_ids` | No | List of specific entity IDs to get details for. If empty, returns details for all entities. | `["binary_sensor.motion_sensor_1"]` |

**Example:**
```yaml
service: area_occupancy.get_entity_details
data:
  entry_id: your_config_entry_id_here
  entity_ids:
    - binary_sensor.motion_sensor_1
    - binary_sensor.door_sensor
```

**Returns detailed information including:**
- Entity state and evidence
- Availability and last updated timestamp
- Current probability and decay status
- Entity type configuration (weight, probabilities, active states)
- Prior probability values

data:
  entry_id: your_config_entry_id_here
  entity_ids:
    - binary_sensor.motion_sensor_1
```

**Returns:**
- `updated_entities`: Number of entities that were updated

## `area_occupancy.get_area_status`

Returns the current occupancy status and confidence level for the area, along with entity metrics for context.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.get_area_status
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `area_name`: Name of the area
- `occupied`: Boolean indicating if area is currently occupied
- `occupancy_probability`: Current probability of occupancy (0.0-1.0)
- `area_baseline_prior`: The baseline prior probability
- `confidence_level`: Text description of confidence (high/medium/low/unknown)
- Entity metrics (total, active, available, unavailable, decaying counts)

## `area_occupancy.get_entity_type_learned_data`

Returns the learned data for all entity types, including probabilities and configuration values that have been learned from historical data.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.get_entity_type_learned_data
data:
  entry_id: your_config_entry_id_here
```

**Returns learned data for each entity type:**
- `prior`: Prior probability for the entity type
- `prob_true`: Probability when condition is true
- `prob_false`: Probability when condition is false
- `weight`: Weight factor for the entity type
- `active_states`: States considered as "active" for this entity type
- `active_range`: Range of values considered "active" (for numeric entities)

## Debug Services

## Debug Services

### `area_occupancy.debug_database_state`

Check current simplified database state including intervals count, sample data, database statistics and schema information.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.debug_database_state
data:
  entry_id: your_config_entry_id_here
```

### `area_occupancy.purge_intervals`

Purge the stored state intervals that are older than a retention period. Optionally filter by entity IDs.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |
| `retention_days` | No | Number of days to retain intervals. Older intervals may be removed. | `365` |
| `entity_ids` | No | List of entity IDs to purge. If empty, all configured entities are considered. | `["binary_sensor.motion"]` |

**Example:**
```yaml
service: area_occupancy.purge_intervals
data:
  entry_id: your_config_entry_id_here
  retention_days: 180
```

**Notes:**

- All services except `reset_entities` return response data that can be used in automations or scripts.
- Services that query historical data (`run_analysis`) can be resource-intensive.
- The `entry_id` can be found in Home Assistant under Settings → Devices & Services → Area Occupancy Detection → (click on an instance).
