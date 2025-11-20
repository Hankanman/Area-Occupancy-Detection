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
  - `likelihoods`: Updated likelihood data per entity
- `update_timestamp`: ISO timestamp of when the analysis completed

**Notes:**
- This service always runs analysis for all configured areas.
- Services that query historical data can be resource-intensive.
