# Services

The Area Occupancy Detection integration provides services that can be called from automations or scripts.

## `area_occupancy.update_priors`

Manually triggers the [Prior Probability Learning](../features/prior-learning.md) process for a specific Area Occupancy instance.

This is useful if you want to force the system to re-analyze historical data immediately, for example, after making significant changes to sensor configurations or room usage patterns, rather than waiting for the next scheduled automatic update.

| Parameter        | Required | Description                                                                                                                                | Example Value          |
| :--------------- | :------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :--------------------- |
| `entry_id`       | Yes      | The configuration entry ID for the Area Occupancy instance you want to update. You can find this in the Home Assistant UI under the integration details. | `a1b2c3d4e5f6...`      |
| `history_period` | No       | The number of past days to analyze. If omitted, uses the value configured for the instance in its options, otherwise defaults to 7 days. | `14`                   |

**Example Service Call (YAML):**

```yaml
service: area_occupancy.update_priors
data:
  entry_id: your_config_entry_id_here # Replace with the actual ID
  # Optional: Specify a different history period for this run
  # history_period: 10
```

**Notes:**

*   Running this service can be resource-intensive as it queries the recorder database.
*   After the priors are updated, the coordinator will automatically refresh, potentially updating the **Prior Probability** sensor and influencing future **Occupancy Probability** calculations. 