# Services

The Area Occupancy Detection integration provides services that can be called from automations or scripts.

## `area_occupancy.update_area_prior`

Manually triggers the [Prior Probability Learning](../features/prior-learning.md) process for a specific Area Occupancy instance.

This is useful if you want to force the system to re-analyze historical data immediately, for example, after making significant changes to sensor configurations or room usage patterns, rather than waiting for the next scheduled automatic update.

| Parameter        | Required | Description                                                                                                                                | Example Value          |
| :--------------- | :------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :--------------------- |
| `entry_id`       | Yes      | The configuration entry ID for the Area Occupancy instance you want to update. You can find this in the Home Assistant UI under the integration details. | `a1b2c3d4e5f6...`      |
| `history_period` | No       | The number of past days to analyze. If omitted, uses the value configured for the instance in its options, otherwise defaults to 7 days. | `14`                   |

**Example Service Call (YAML):**

```yaml
service: area_occupancy.update_area_prior
data:
  entry_id: your_config_entry_id_here # Replace with the actual ID
  # Optional: Specify a different history period for this run
  # history_period: 10
```

**Returns:**
- `area_prior`: The calculated baseline prior probability for the area
- `history_period`: Number of days of history that were analyzed
- `update_timestamp`: ISO timestamp of when the update was performed
- `calculation_details`: Detailed breakdown of the calculation including:
  - `motion_sensors`: List of sensor IDs used in the calculation
  - `sensor_count`: Number of sensors analyzed
  - `calculation_method`: Description of the calculation approach
  - `sensor_details`: Per-sensor breakdown of occupancy ratios and filtering statistics
  - `raw_average_ratio`: The raw average occupancy ratio before buffer
  - `buffer_multiplier`: The buffer multiplier applied (typically 1.05)
  - `final_prior`: The final calculated prior value
  - `calculation`: Text description of the calculation formula
  - `filtering_summary`: Statistics about interval filtering (removes false triggers and stuck sensors)

**Notes:**

*   Running this service can be resource-intensive as it queries the recorder database.
*   After the area prior is updated, the coordinator will automatically refresh, potentially updating the **Prior Probability** sensor and influencing future **Occupancy Probability** calculations.
*   The service analyzes motion sensor states over the specified history period and calculates an average occupancy ratio with a 5% buffer to establish the area's baseline occupancy probability.

## `area_occupancy.update_likelihoods`

Recalculates the sensor likelihood values used in the Bayesian calculation. This is similar to updating priors but focuses on the per-sensor probabilities.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |
| `history_period` | No | Number of days of history to analyse. Defaults to the configured history period. | `7` |

**Example:**
```yaml
service: area_occupancy.update_likelihoods
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `updated_entities`: Number of entities that had their likelihoods updated
- `history_period`: Number of days of history that were analyzed
- `total_entities`: Total number of entities in the system
- `update_timestamp`: ISO timestamp of when the update was performed
- `prior`: The current area baseline prior probability
- `likelihoods`: Detailed likelihood data for each entity including:
  - `type`: Entity input type (binary_sensor, sensor, etc.)
  - `weight`: Weight factor for this entity type
  - `prob_given_true`: Probability of entity state when area is occupied
  - `prob_given_false`: Probability of entity state when area is unoccupied
  - `prob_given_true_raw`: Raw probability before smoothing
  - `prob_given_false_raw`: Raw probability before smoothing
  - Filtering statistics (intervals processed, filtered, stuck sensor analysis)
- `likelihood_filtering_summary`: Overall filtering statistics across all entities

## `area_occupancy.update_time_based_priors`

Manually triggers a recalculation of [Time-Based Priors](../features/time-based-priors.md) for a specific Area Occupancy instance.

This service runs the time-based prior calculation in the background, analyzing historical data to learn occupancy patterns for different times of day and days of the week.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.update_time_based_priors
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `status`: "started" (calculation runs in background)
- `message`: Confirmation message with entry ID
- `history_period_days`: Number of days that will be analyzed
- `start_timestamp`: ISO timestamp when calculation began
- `note`: Information about background processing

**Notes:**
- This is a background operation that can take several minutes to complete
- The service returns immediately while calculation continues in the background
- Check the Home Assistant logs for completion status and any errors
- Time-based priors are automatically recalculated on a schedule (default: every 4 hours)

## `area_occupancy.get_time_based_priors`

Retrieves current time-based priors in a human-readable format, showing learned occupancy patterns for different times of day and days of the week.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.get_time_based_priors
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `area_name`: Name of the area
- `current_time_slot`: Current time slot (e.g., "Monday 14:00-14:30")
- `current_prior`: Current time-based prior value
- `time_prior`: Time-based prior for current slot
- `global_prior`: Fallback global prior value
- `total_time_slots_available`: Number of time slots with data
- `daily_summaries`: Prior values organized by day and time
- `key_periods`: Average priors for common time periods (Early Morning, Morning, Afternoon, Evening, Night)
- `note`: Explanation of time-based priors

**Example Output:**
```json
{
  "area_name": "Living Room",
  "current_time_slot": "Monday 14:00-14:30",
  "current_prior": 0.2345,
  "time_prior": 0.1567,
  "global_prior": 0.2345,
  "total_time_slots_available": 312,
  "daily_summaries": {
    "Monday": {
      "08:00": 0.4567,
      "08:30": 0.4789,
      "18:00": 0.8234,
      "18:30": 0.8456
    },
    "Saturday": {
      "10:00": 0.3456,
      "20:00": 0.9123
    }
  },
  "key_periods": {
    "Morning (08:00-12:00)": [
      {"day": "Monday", "average": 0.4567}
    ],
    "Evening (17:00-21:00)": [
      {"day": "Monday", "average": 0.8234},
      {"day": "Saturday", "average": 0.9123}
    ]
  }
}
```

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

## `area_occupancy.force_entity_update`

Forces an immediate update of specific entities or all entities if none are specified. This can be useful for testing or when you need to refresh entity states immediately.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |
| `entity_ids` | No | List of specific entity IDs to update. If empty, updates all entities. | `["binary_sensor.motion_sensor_1"]` |

**Example:**
```yaml
service: area_occupancy.force_entity_update
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

### `area_occupancy.debug_import_intervals`

Manually triggers state intervals import from recorder with detailed debug logging. Use this to diagnose why state intervals aren't being populated.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |
| `days` | No | Number of days of historical data to import from recorder. Defaults to 10. | `10` |

**Example:**
```yaml
service: area_occupancy.debug_import_intervals
data:
  entry_id: your_config_entry_id_here
  days: 14
```

### `area_occupancy.debug_database_state`

Check current simplified database state including intervals count, sample data, database statistics, and schema information.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `entry_id` | Yes | The configuration entry ID for the Area Occupancy instance. | `a1b2c3d4e5f6...` |

**Example:**
```yaml
service: area_occupancy.debug_database_state
data:
  entry_id: your_config_entry_id_here
```

**Notes:**

- All services except `reset_entities` return response data that can be used in automations or scripts
- Services that query historical data (`update_area_prior`, `update_likelihoods`, `update_time_based_priors`) can be resource-intensive
- The `entry_id` can be found in Home Assistant under Settings → Devices & Services → Area Occupancy Detection → (click on an instance)
- Time-based prior services are part of the new advanced features and require historical analysis to be enabled

