# Services
The Area Occupancy Detection integration provides services that can be called from automations or scripts.

## `area_occupancy.run_analysis`

Runs the historical analysis process for an Area Occupancy instance. This imports recent state data from the recorder, updates priors and likelihoods, and refreshes the coordinator.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `area_name` | Yes | The name of the area to run analysis for, or "all" for all areas. | `"Living Room"` or `"all"` |

**Example:**
```yaml
service: area_occupancy.run_analysis
data:
  area_name: "Living Room"
```

**Example for all areas:**
```yaml
service: area_occupancy.run_analysis
data:
  area_name: "all"
```

**Returns:**
- `area_name`: Name of the area
- `current_prior`: Current global prior probability
- `global_prior`: Global prior after analysis
- `time_prior`: Time-based prior used in calculations
- `prior_entity_ids`: List of entities included in analysis
- `total_entities`: Total number of entities
- `entity_states`: Current states of all entities
- `likelihoods`: Updated likelihood data per entity
- `update_timestamp`: ISO timestamp of when the analysis completed

## `area_occupancy.reset_entities`

Resets all entity probabilities and learned data for a specific area.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `area_name` | Yes | The name of the area to reset, or "all" for all areas. | `"Kitchen"` or `"all"` |

**Example:**
```yaml
service: area_occupancy.reset_entities
data:
  area_name: "Kitchen"
```

**Example for all areas:**
```yaml
service: area_occupancy.reset_entities
data:
  area_name: "all"
```

## `area_occupancy.get_entity_metrics`

Returns basic metrics about entities in the specified area. This service returns data and can be used for monitoring and diagnostics.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `area_name` | Yes | The name of the area to get metrics for, or "all" for all areas. | `"Bedroom"` or `"all"` |

**Example:**
```yaml
service: area_occupancy.get_entity_metrics
data:
  area_name: "Bedroom"
```

**Example for all areas:**
```yaml
service: area_occupancy.get_entity_metrics
data:
  area_name: "all"
```

**Returns:**
- `metrics`: Object containing:
  - `total_entities`: Total number of entities
  - `active_entities`: Number of entities currently providing evidence
  - `available_entities`: Number of available entities
  - `unavailable_entities`: Number of unavailable entities
  - `decaying_entities`: Number of entities currently in decay state
  - `availability_percentage`: Percentage of entities available
  - `activity_percentage`: Percentage of entities active
  - `summary`: Human readable summary of metrics

## `area_occupancy.get_problematic_entities`

Identifies entities that may need attention, such as unavailable entities or those with stale updates. This service returns data for troubleshooting purposes.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `area_name` | Yes | The name of the area to check for problems, or "all" for all areas. | `"Office"` or `"all"` |

**Example:**
```yaml
service: area_occupancy.get_problematic_entities
data:
  area_name: "Office"
```

**Example for all areas:**
```yaml
service: area_occupancy.get_problematic_entities
data:
  area_name: "all"
```

**Returns:**
- `problems`: Object containing:
  - `unavailable`: List of entity IDs that are currently unavailable
  - `stale_updates`: List of entity IDs that haven't been updated in over an hour
  - `total_problems`: Count of unavailable and stale entities
  - `summary`: Human readable summary of issues

## `area_occupancy.get_area_status`

Returns the current occupancy status and confidence level for the area, along with entity metrics for context.

| Parameter | Required | Description | Example Value |
|-----------|---------|-------------|---------------|
| `area_name` | Yes | The name of the area to get status for, or "all" for all areas. | `"Living Room"` or `"all"` |

**Example:**
```yaml
service: area_occupancy.get_area_status
data:
  area_name: "Living Room"
```

**Example for all areas:**
```yaml
service: area_occupancy.get_area_status
data:
  area_name: "all"
```

**Returns:**
- `area_status`: Object containing:
  - `area_name`: Name of the area
  - `occupied`: Boolean indicating if area is currently occupied
  - `occupancy_probability`: Current probability of occupancy (0.0-1.0)
  - `area_baseline_prior`: The baseline prior probability
  - `confidence_level`: Text description of confidence
  - `confidence_description`: Detailed description of confidence level
  - `entity_summary`: Counts of total, active, available, unavailable and decaying entities
  - `status_summary`: Human readable summary of the area state

**Notes:**
- Services that return response data: `run_analysis`, `get_entity_metrics`, `get_problematic_entities`, and `get_area_status` all return nested objects (see Returns sections above for structure).
- The `reset_entities` service does not return response data.
- Services that query historical data (`run_analysis`) can be resource-intensive.
- When `area_name` is set to `"all"`, the service operates on all configured areas and returns aggregated data.
- The area names available in the dropdown are automatically populated from your configured areas.
- The deprecated `entry_id` parameter is still supported for backward compatibility but will be removed in a future version.
