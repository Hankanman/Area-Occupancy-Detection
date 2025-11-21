# Prior Probability Calculation

This document explains in detail how prior probabilities are calculated and combined for area occupancy detection.

## Overview

Prior probabilities represent the baseline likelihood that an area is occupied, learned from historical sensor data. The system calculates two types of priors:

1. **Global Prior**: Overall occupancy rate for the area (regardless of time)
2. **Time-Based Prior**: Occupancy rate for specific day-of-week and time-slot combinations

These are combined to form the final prior used in Bayesian calculations.

## Global Prior Calculation

The global prior represents the overall probability that the area is occupied, calculated from historical sensor data.

**Code Reference:** `174:381:custom_components/area_occupancy/data/analysis.py` (PriorAnalyzer.analyze_area_prior)

### Step 1: Motion-Only Calculation

The system first calculates the prior using only motion sensors:

1. **Get Total Occupied Time**: Calculates total seconds the area was occupied based on motion sensor intervals

   - Uses motion timeout to extend intervals after last motion
   - Aggregates overlapping intervals into continuous occupied periods

2. **Get Time Bounds**: Determines the earliest and latest timestamps from motion sensor history

   - Used to calculate the total time period for analysis

3. **Calculate Motion Prior**:
   ```
   motion_prior = total_occupied_seconds / total_seconds
   ```

**Code Reference:** `547:563:custom_components/area_occupancy/data/analysis.py` (get_total_occupied_seconds)

**Code Reference:** `589:598:custom_components/area_occupancy/data/analysis.py` (get_time_bounds)

### Step 2: Media/Appliance Supplementation

If the motion-only prior is below 0.10 (10%), the system supplements with media players and appliances:

**Code Reference:** `251:317:custom_components/area_occupancy/data/analysis.py`

Logic:

- If `motion_prior >= 0.10`: Use motion-only prior
- If `motion_prior < 0.10`: Supplement with media/appliance sensors
  - Includes media players in "playing" state
  - Includes appliances in "on" state
  - Recalculates total occupied time including these sensors
  - Uses same time bounds from motion sensors

This supplementation helps in areas with low motion activity but high media/appliance usage (e.g., home theaters, kitchens).

### Step 3: Minimum Prior Override

If configured, a minimum prior override is applied to ensure the prior never drops below a configured minimum, useful for areas that should always have some baseline occupancy probability.

**Important:** The override is **only** applied at runtime, **not** during the learning phase. This ensures that the database stores the actual calculated prior from historical data, allowing for accurate analysis and potential future adjustments to the override value.

The override is applied at runtime:

**Runtime Phase** (when calculating Prior.value property):

   **Code Reference:** `116:127:custom_components/area_occupancy/data/prior.py`

   Applied after combining global_prior with time_prior and applying PRIOR_FACTOR, ensuring the final runtime prior never drops below the configured minimum regardless of time-based adjustments or scaling factors.

   This ensures that even if the combined prior (global_prior + time_prior) or the final prior (after PRIOR_FACTOR) would be below the minimum, the override takes effect at runtime.

   The actual calculated prior (without override) is stored in the database during the learning phase, allowing the override to be adjusted without needing to recalculate historical priors.

### Step 4: Database Storage

The calculated global prior is stored in the database:

**Code Reference:** `332:368:custom_components/area_occupancy/data/analysis.py`

Stored metadata includes:

- Prior value
- Data period (start and end times)
- Total occupied seconds
- Total period seconds
- Interval count
- Calculation method
- Data source (motion-only or merged)

## Time-Based Prior Calculation

Time-based priors calculate occupancy probability for specific day-of-week and time-slot combinations, capturing patterns like "kitchen is usually occupied at 7am on weekdays."

**Code Reference:** `383:545:custom_components/area_occupancy/data/analysis.py` (PriorAnalyzer.analyze_time_priors)

### Time Slot Structure

The system divides each day into time slots:

- Default slot duration: 60 minutes
- Slots per day: 24 (one per hour)
- Days per week: 7 (Monday through Sunday)

**Code Reference:** `52:56:custom_components/area_occupancy/data/analysis.py` (time slot constants)

### Calculation Process

1. **Get Interval Aggregates**: Aggregates occupied intervals by day-of-week and time-slot

   - Groups intervals by the day and hour they occurred
   - Sums total occupied seconds for each (day, slot) combination

2. **Calculate Time Bounds**: Gets earliest and latest timestamps from sensor history

   - Used to determine how many days of data are available

3. **Calculate Priors for Each Slot**: For each day-of-week (0-6) and time-slot (0-23):

   ```
   total_slot_seconds = days * slot_duration_seconds
   occupied_slot_seconds = sum of occupied seconds for this (day, slot)
   prior = occupied_slot_seconds / total_slot_seconds
   ```

4. **Store in Database**: Each (day, slot, prior) combination is stored in the `Priors` table

**Code Reference:** `600:613:custom_components/area_occupancy/data/analysis.py` (get_interval_aggregates)

### Day-of-Week Conversion

The system uses Python's weekday format (0=Monday, 6=Sunday) internally, but converts from SQLite's format (0=Sunday, 6=Saturday) when reading from the database.

**Code Reference:** `58:61:custom_components/area_occupancy/data/analysis.py` (weekday conversion constants)

## Prior Retrieval

When calculating occupancy probability, the system retrieves the appropriate prior:

**Code Reference:** `79:116:custom_components/area_occupancy/data/prior.py` (Prior.value property)

1. **Get Global Prior**: Retrieved from database (stored in `GlobalPriors` table)
2. **Get Time-Based Prior**: Retrieved for current day-of-week and time-slot

   - Uses caching to avoid repeated database queries
   - Cache invalidated when prior is updated

3. **Combine Priors**: Uses `combine_priors()` function to merge global and time-based priors

## Prior Combination

The global prior and time-based prior are combined using weighted averaging in logit space.

**Code Reference:** `160:225:custom_components/area_occupancy/utils.py` (combine_priors)

### Logit Space Interpolation

Logit space provides better interpolation than linear space for probabilities:

1. **Convert to Logits**:

   ```
   logit(p) = log(p / (1 - p))
   ```

2. **Weighted Combination**:

   ```
   area_weight = 1.0 - time_weight
   combined_logit = area_weight * area_logit + time_weight * time_logit
   ```

3. **Convert Back to Probability**:
   ```
   combined_prior = 1 / (1 + exp(-combined_logit))
   ```

### Edge Case Handling

**Code Reference:** `174:202:custom_components/area_occupancy/utils.py`

- **Zero Time Weight**: Returns area prior directly
- **Full Time Weight**: Returns time prior directly
- **Extreme Values**: Converts 0.0 to MIN_PROBABILITY and 1.0 to MAX_PROBABILITY to avoid logit issues
- **Identical Priors**: Returns the common value directly (avoids unnecessary calculation)

### Default Time Weight

The default time weight is 0.2, meaning:

- 80% of the prior comes from the global area prior
- 20% comes from the time-based prior

This balance ensures time patterns influence the result without overwhelming the overall area characteristics.

## Prior Factor Application

After combining priors, a factor of 1.05 is applied:

**Code Reference:** `105:106:custom_components/area_occupancy/data/prior.py`

```
adjusted_prior = combined_prior * 1.05
```

This slightly increases the baseline probability, helping prevent the system from being too conservative. The factor is clamped to ensure the result stays within [MIN_PRIOR, MAX_PRIOR].

## Caching

Time-based priors are cached to avoid repeated database queries:

**Code Reference:** `72:137:custom_components/area_occupancy/data/prior.py`

- Cache key: (day_of_week, time_slot)
- Cache invalidated when:
  - Prior is updated via `set_global_prior()`
  - Cache is explicitly cleared

This improves performance since time-based priors are queried frequently during real-time calculations.

## Example Calculation

Consider an area with the following history:

- Global prior: 0.25 (25% of time area is occupied)
- Time-based prior for Monday 7am: 0.60 (60% of Monday mornings at 7am)
- Time weight: 0.2

Step 1: Convert to logits

```
area_logit = log(0.25 / 0.75) = log(0.333) = -1.099
time_logit = log(0.60 / 0.40) = log(1.5) = 0.405
```

Step 2: Weighted combination

```
area_weight = 1.0 - 0.2 = 0.8
time_weight = 0.2
combined_logit = 0.8 * (-1.099) + 0.2 * 0.405 = -0.879 + 0.081 = -0.798
```

Step 3: Convert back to probability

```
combined_prior = 1 / (1 + exp(-(-0.798))) = 1 / (1 + exp(0.798)) = 1 / (1 + 2.223) = 0.310
```

Step 4: Apply prior factor

```
final_prior = 0.310 * 1.05 = 0.326 (32.6%)
```

The time-based prior (60% for Monday 7am) increases the combined prior from 25% to 32.6%, reflecting that this time slot is more likely to be occupied than average.

## Database Schema

### GlobalPriors Table

Stores global priors for each area:

- `area_name`: Area identifier
- `prior_value`: Calculated prior probability
- `data_period_start`: Start of analysis period
- `data_period_end`: End of analysis period
- `total_occupied_seconds`: Total time area was occupied
- `total_period_seconds`: Total time period analyzed
- `interval_count`: Number of occupied intervals
- `calculation_method`: Method used (e.g., "interval_analysis")
- `last_updated`: Timestamp of last update

### Priors Table

Stores time-based priors for each (day, slot) combination:

- `area_name`: Area identifier
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `time_slot`: Time slot (0-23 for hourly slots)
- `prior_value`: Calculated prior for this (day, slot)
- `data_points`: Number of data points used
- `last_updated`: Timestamp of last update

## See Also

- [Complete Calculation Flow](calculation-flow.md) - End-to-end process
- [Prior Learning Feature](../features/prior-learning.md) - User-facing documentation
- [Bayesian Calculation Deep Dive](bayesian-calculation.md) - How priors are used in calculations
- [Likelihood Calculation Deep Dive](likelihood-calculation.md) - Related learning process
