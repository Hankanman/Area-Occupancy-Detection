# Wasp-in-the-Box Logic

## Overview

The "wasp-in-the-box" logic is a feature in the Area Occupancy Detection integration that improves occupancy detection by implementing a "sticky" occupancy state. The name comes from the analogy of a wasp trapped in a box - if you saw a wasp enter a box and haven't seen it leave, there's a high probability it's still there.

**Available from version 2025.4.4+**

## How It Works

### Basic Principle

When enabled, the integration creates a virtual sensor that contributes to the occupancy probability calculation. This virtual sensor:

1. Becomes "active" when the area's occupancy probability exceeds the configured threshold
2. Stays active as long as the area is considered occupied
3. Has high confidence values and weight to maintain occupancy state

### Technical Implementation

#### Configuration
- Toggle: `wasp_in_box_enabled` in configuration/options flow
- Default: Disabled
- Location: Advanced Parameters section

#### Probability Values
- `WASP_PROB_GIVEN_TRUE = 0.95` - 95% confidence when area is occupied
- `WASP_PROB_GIVEN_FALSE = 0.05` - 5% probability of false positive
- `WASP_DEFAULT_PRIOR = 0.4` - 40% base probability
- `WASP_WEIGHT = 0.9` - 90% weight in calculations

#### Virtual Sensor
The integration creates a virtual sensor (`wasp.virtual`) that:
- State is `ON` when area probability ≥ threshold
- State is `OFF` when area probability < threshold
- Always reports as available
- Updates timestamp on each state change
- Is not a real Home Assistant entity, but is shown in the probability/prior sensor attributes as 'Wasp-in-the-Box'

### Integration with Bayesian Calculation

1. **Sensor Type**
   - Registered as `EntityType.WASP`
   - Handled like other sensor types in probability calculations
   - Contributes to overall prior probability

2. **State Processing**
   ```python
   wasp_state = {
       "state": "on" if area_probability >= threshold else "off",
       "availability": True,
       "last_changed": current_timestamp
   }
   ```

3. **Probability Impact**
   - When active (ON):
     - Contributes high probability (95%) weighted at 90%
     - Helps maintain occupancy state even if other sensors become inactive
   - When inactive (OFF):
     - Contributes low probability (5%)
     - Allows other sensors to more easily trigger occupancy

### Use Cases

1. **Rooms with Limited Sensor Coverage**
   - Helps maintain occupancy state in areas where sensors might not cover all spots
   - Reduces false negatives from motion sensor blind spots

2. **Quiet Activities**
   - Better detection when occupants are still (reading, watching TV, etc.)
   - Prevents premature "unoccupied" states

3. **Sensor Transition Gaps**
   - Smooths transitions between sensor activations
   - Reduces flickering of occupancy state

### Example Scenario

```
Initial State:
- Motion sensor detects movement (probability rises to 85%)
- Area becomes "occupied" (above threshold)
- Wasp sensor activates

After Motion Stops:
- Motion sensor becomes inactive
- Wasp sensor stays active
- High wasp probability maintains occupancy
- Overall probability stays elevated

New Motion:
- Motion sensor activates again
- Confirms continued occupancy
- Wasp sensor remains active
```

## Configuration

Enable through the integration's configuration or options flow:

1. Go to Configuration → Integrations
2. Find Area Occupancy Detection
3. Click Options
4. Navigate to Advanced Parameters
5. Enable "Wasp-in-the-Box Logic"

## Attributes and UI

The wasp-in-the-box state can be monitored through:

1. **Probability Sensor**
   - Shows wasp contribution in the `sensor_probabilities` attribute
   - Format: `"Wasp-in-the-Box | W: 0.9 | P: 0.95 | WP: 0.855"`
   - The label 'Wasp-in-the-Box' is used for clarity in the UI

2. **Prior Sensor**
   - Shows wasp prior in attributes
   - Format: `"wasp": "Prior: 40.0%"`

### How to Interpret the Wasp Entry
- **Wasp-in-the-Box | W: 0.9 | P: 0.95 | WP: 0.855**
  - **Wasp-in-the-Box**: The virtual sensor's label
  - **W**: Weight (0.9)
  - **P**: Probability given current state (0.95 if ON, 0.05 if OFF)
  - **WP**: Weighted probability (W × P)

### Notes
- The wasp state is not persisted between restarts (it is recalculated on startup)
- Priors for the wasp sensor are stored and loaded with other sensor data
- The wasp sensor is fully integrated into the Bayesian calculation and all sensor attributes
- The feature is visible in the UI as 'Wasp-in-the-Box' in the probability/prior sensor attributes

## Best Practices

1. **When to Enable**
   - Areas with intermittent sensor coverage
   - Spaces where people remain still for long periods
   - Rooms where false negatives are more problematic than false positives

2. **When to Disable**
   - Areas requiring immediate response to vacancy
   - Spaces with complete, reliable sensor coverage
   - Locations where false positives are more problematic

3. **Tuning**
   - Adjust the occupancy threshold to balance sensitivity
   - Use in conjunction with decay settings for optimal results
   - Monitor the probability sensor attributes to understand impact

## Technical Notes

1. **Storage**
   - Wasp state is not persisted between restarts
   - Priors are stored and loaded with other sensor data

2. **Performance**
   - Minimal impact on system resources
   - No additional API calls or external dependencies
   - Lightweight virtual sensor implementation

3. **Integration**
   - Fully integrated with existing Bayesian calculation
   - Respects all core integration features (decay, thresholds, etc.)
   - Compatible with all sensor types 