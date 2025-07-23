# Time-Based Priors

The Area Occupancy Detection integration now includes **Time-Based Priors**, an advanced feature that learns occupancy patterns based on the time of day and day of the week. This provides much more accurate occupancy detection by understanding when areas are typically occupied during different periods.

## What Are Time-Based Priors?

Time-based priors enhance the traditional prior probability learning by calculating separate occupancy probabilities for specific time slots throughout the day and week. Instead of using a single global prior probability, the system learns that:

- Your living room might have a 15% occupancy probability at 2 AM on weekdays
- But a 75% occupancy probability at 8 PM on weekends
- Your office might have a 90% occupancy probability at 10 AM on weekdays
- But only a 5% occupancy probability on weekends

This temporal awareness significantly improves the accuracy of occupancy detection by providing context-aware baseline probabilities.

## How Time-Based Priors Work

### Time Slot Granularity

The system divides each day into **48 time slots** of 30 minutes each:
- 00:00-00:29 = Slot 0
- 00:30-00:59 = Slot 1
- 01:00-01:29 = Slot 2
- ... and so on until 23:30-23:59 = Slot 47

Each day of the week (Monday=0 through Sunday=6) has its own set of 48 time slots, resulting in **336 unique time-based priors** per area (7 days × 48 slots).

### Learning Process

1. **Historical Analysis**: The system analyzes historical data from your configured sensors over the specified history period (default: 7 days)

2. **Time Slot Calculation**: For each time slot, it calculates:
   - How often the area was occupied during that specific time period
   - The total duration of occupancy vs. total time available
   - The resulting occupancy probability for that time slot

3. **Database Storage**: Time-based priors are stored in a dedicated SQLite database table with efficient indexing for quick retrieval

4. **Real-Time Usage**: During occupancy calculations, the system:
   - Determines the current time slot (day of week + 30-minute period)
   - Retrieves the learned prior for that specific time slot
   - Uses it as the baseline probability in the Bayesian calculation

### Fallback Strategy

The system includes a robust fallback strategy:

1. **Time-Based Prior**: First, try to use the learned prior for the current time slot
2. **Global Prior**: If no time-based prior exists, fall back to the traditional global prior
3. **Minimum Prior**: If no priors are available, use the minimum prior value (0.01%)

## Configuration Options

Time-based priors can be configured through the integration options:

| Option | Description | Default | Range |
|--------|-------------|---------|-------|
| **Time-Based Priors Enabled** | Enable or disable the time-based priors feature | `true` | `true`/`false` |
| **Time-Based Priors Frequency** | How often to recalculate time-based priors (in prior timer cycles) | `4` | `1-24` |
| **Likelihood Updates Enabled** | Enable or disable automatic likelihood updates | `true` | `true`/`false` |
| **Likelihood Updates Frequency** | How often to update sensor likelihoods (in prior timer cycles) | `2` | `1-24` |

### Frequency Explanation

- **Time-Based Priors Frequency**: Since prior calculations run every hour by default, a frequency of `4` means time-based priors are recalculated every 4 hours
- **Likelihood Updates Frequency**: A frequency of `2` means sensor likelihoods are updated every 2 hours

## Performance Considerations

### Background Processing

Time-based prior calculations are computationally intensive, so the system:

- **Deferred Startup**: Calculations are deferred to background tasks to avoid blocking Home Assistant startup
- **Chunked Processing**: Time slots are processed in chunks of 12 (6 hours worth) with periodic yields to the event loop
- **Batch Database Operations**: Database writes are performed in batches of 50 records for efficiency
- **Caching**: Results are cached for 30 minutes to reduce repeated calculations

### Database Storage

Time-based priors are stored in a dedicated SQLite database with:

- **Efficient Indexing**: Indexes on `(entry_id, day_of_week, time_slot)` for fast lookups
- **Data Retention**: 365-day retention policy for historical data
- **Integrity Checks**: Built-in database integrity validation and corruption recovery

## Services

### Update Time-Based Priors

Manually trigger a recalculation of time-based priors:

```yaml
service: area_occupancy.update_time_based_priors
data:
  entry_id: your_config_entry_id_here
```

**Returns:**
- `status`: "started" (calculation runs in background)
- `message`: Confirmation message
- `history_period_days`: Number of days analyzed
- `start_timestamp`: When calculation began

### Get Time-Based Priors

Retrieve current time-based priors in a human-readable format:

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

## Example Output

The `get_time_based_priors` service provides rich, human-readable output:

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
      "09:00": 0.5123,
      "18:00": 0.8234,
      "18:30": 0.8456,
      "19:00": 0.8678
    },
    "Saturday": {
      "10:00": 0.3456,
      "10:30": 0.3678,
      "20:00": 0.9123,
      "20:30": 0.9234
    }
  },
  "key_periods": {
    "Morning (08:00-12:00)": [
      {"day": "Monday", "average": 0.4567},
      {"day": "Tuesday", "average": 0.4234}
    ],
    "Evening (17:00-21:00)": [
      {"day": "Monday", "average": 0.8234},
      {"day": "Saturday", "average": 0.9123}
    ]
  }
}
```

## Benefits

### Improved Accuracy

- **Context-Aware Detection**: Understands that occupancy patterns vary by time
- **Reduced False Positives**: Lower baseline probabilities during typically unoccupied periods
- **Enhanced Sensitivity**: Higher baseline probabilities during typically occupied periods

### Better Automation

- **Time-Based Rules**: Create automations that respond differently based on time of day
- **Pattern Recognition**: Identify unusual occupancy patterns
- **Energy Efficiency**: More precise control based on learned patterns

### User Experience

- **Faster Learning**: Adapts to your schedule more quickly
- **Reduced Tuning**: Less manual configuration required
- **Transparency**: Clear visibility into learned patterns through services

## Troubleshooting

### Common Issues

1. **No Time-Based Priors Available**:
   - Ensure historical analysis is enabled
   - Check that sufficient historical data exists
   - Verify time-based priors are enabled in configuration

2. **Slow Performance**:
   - Increase the calculation frequency (less frequent updates)
   - Reduce the history period for faster calculations
   - Monitor database size and cleanup old data if needed

3. **Inaccurate Patterns**:
   - Increase the history period for more data
   - Manually trigger recalculation after significant schedule changes
   - Review the learned patterns using the `get_time_based_priors` service

### Debug Services

Two additional debug services are available:

- **`debug_import_intervals`**: Manually trigger state intervals import from recorder
- **`debug_database_state`**: Check database state and statistics

## Integration with Existing Features

Time-based priors work seamlessly with all existing features:

- **Probability Decay**: Decay calculations use time-based priors as the baseline
- **Sensor Weights**: Individual sensor weights continue to apply
- **Historical Learning**: Complements traditional prior learning
- **Wasp in Box**: Works alongside the Wasp in Box feature

The time-based priors feature represents a significant advancement in occupancy detection accuracy, providing context-aware probabilities that adapt to your specific usage patterns and schedule. 