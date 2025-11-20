# Likelihood Calculation

This document explains how the system learns and uses likelihood probabilities for each sensor entity.

## Overview

Likelihoods represent how reliable each sensor is as evidence of occupancy. For each entity, the system learns:

- **`P(Active | Occupied)`**: How often the sensor is active when the area is genuinely occupied
- **`P(Active | Not Occupied)`**: How often the sensor is active when the area is not occupied

These values are learned from historical data by correlating each sensor's activity with the area's occupancy state (determined by motion sensors).

**Important:** Motion sensors do **not** have learned likelihoods. Instead, they use **user-configurable likelihoods** that can be set per area during configuration (defaults: `prob_given_true=0.95`, `prob_given_false=0.02`). This is because motion sensors are used as ground truth to determine occupied intervals. Learning motion sensor likelihoods would create a circular dependency where motion sensors determine occupied intervals and then calculate their own likelihoods from those same intervals.

Motion sensor likelihoods are configured per area in the integration's configuration flow, allowing you to fine-tune the reliability values based on your specific motion sensor setup.

## Likelihood Analysis Process

The likelihood analysis process runs periodically (typically hourly) or on-demand via the `run_analysis` service.

**Code Reference:** ```970:1060:custom_components/area_occupancy/data/analysis.py``` (start_likelihood_analysis)

### Step 1: Get Occupied Intervals

The system first determines when the area was occupied based on motion sensor history:

**Code Reference:** ```1004:1004:custom_components/area_occupancy/data/analysis.py```

```python
occupied_times = area.prior.get_occupied_intervals()
```

This returns a list of `(start_time, end_time)` tuples representing periods when motion sensors indicated the area was occupied.

**Code Reference:** ```201:251:custom_components/area_occupancy/data/prior.py``` (Prior.get_occupied_intervals)

The occupied intervals are calculated using:
- Motion sensor state changes
- Motion timeout to extend intervals after last motion
- Optional media/appliance sensors for supplementation

### Step 2: Get Entity Intervals

For each entity being analyzed, the system retrieves its activity intervals from the database:

**Code Reference:** ```759:789:custom_components/area_occupancy/data/analysis.py``` (_get_intervals_by_entity)

The intervals are stored in the `Intervals` table, which contains:
- `entity_id`: The sensor entity
- `start_time`: When the interval started
- `duration_seconds`: How long the interval lasted
- `state`: The state value during this interval

Intervals are grouped by entity_id for efficient processing.

### Step 3: Analyze Each Entity

For each entity, the system correlates its activity intervals with the occupied intervals:

**Code Reference:** ```791:848:custom_components/area_occupancy/data/analysis.py``` (_analyze_entity_likelihood)

The analysis process:

1. **Get Entity Intervals**: Retrieves all intervals for this entity from the grouped data

2. **Count Overlaps**: For each entity interval, determines if it overlaps with any occupied interval:
   - If interval overlaps with occupied time: counts toward `true_occ` or `true_empty`
   - If interval doesn't overlap: counts toward `false_occ` or `false_empty`

3. **Determine Activity**: Checks if the interval state indicates activity:
   - For binary sensors: checks if state is in `active_states` list
   - For numeric sensors: checks if value is in `active_range`

4. **Accumulate Durations**: Sums the duration (in seconds) for each category:
   - `true_occ`: Entity active AND area occupied
   - `false_occ`: Entity inactive AND area occupied
   - `true_empty`: Entity active AND area not occupied
   - `false_empty`: Entity inactive AND area not occupied

### Step 4: Calculate Probabilities

From the accumulated durations, the system calculates the likelihoods:

**Code Reference:** ```838:848:custom_components/area_occupancy/data/analysis.py```

```
P(Active | Occupied) = true_occ / (true_occ + false_occ)
P(Active | Not Occupied) = true_empty / (true_empty + false_empty)
```

If either denominator is zero (no data for that condition), the probability defaults to 0.5 (neutral).

### Step 5: Update Database and Memory

The calculated likelihoods are stored in the database and updated in memory:

**Code Reference:** ```1020:1043:custom_components/area_occupancy/data/analysis.py```

1. **Database Update**: Updates `Entities` table with new `prob_given_true` and `prob_given_false` values
2. **Memory Update**: Updates the corresponding `Entity` object in the `EntityManager`

**Code Reference:** ```871:908:custom_components/area_occupancy/data/analysis.py``` (_update_likelihoods_in_db)

## How Likelihoods Are Used in Real-Time Calculation

During real-time probability calculation, likelihoods are used to update the probability based on current sensor evidence.

**Code Reference:** ```118:132:custom_components/area_occupancy/utils.py```

### Evidence Evaluation

When an entity provides evidence (is active or decaying), the system uses the appropriate likelihood:

- If entity is active: Uses `prob_given_true` and `prob_given_false` directly
- If entity is inactive: Uses inverse probabilities (1 - prob_given_true, 1 - prob_given_false)
- If entity is unavailable: Skips the entity (unless decaying)

### Decay-Adjusted Likelihoods

When an entity is decaying, its likelihoods are interpolated toward neutral (0.5):

**Code Reference:** ```123:128:custom_components/area_occupancy/utils.py```

```
p_t_adjusted = 0.5 + (p_t_learned - 0.5) * decay_factor
p_f_adjusted = 0.5 + (p_f_learned - 0.5) * decay_factor
```

This gradually reduces the influence of stale evidence as decay progresses.

### Weight Application

The likelihoods are weighted by the entity's configured weight:

**Code Reference:** ```140:144:custom_components/area_occupancy/utils.py```

```
contribution_true = log(p_t) * entity.weight
contribution_false = log(p_f) * entity.weight
```

Higher weights mean the entity's evidence has more influence on the final probability.

## Example Calculation

Consider a media player entity with the following historical data:

### Historical Analysis Period

- Total time analyzed: 7 days = 604,800 seconds
- Occupied time: 100,800 seconds (16.7% of time)
- Not occupied time: 504,000 seconds (83.3% of time)

### Entity Activity

- Media player was "playing" for 50,400 seconds total
- During occupied time: 40,320 seconds playing, 60,480 seconds not playing
- During not occupied time: 10,080 seconds playing, 493,920 seconds not playing

### Likelihood Calculation

```
P(Active | Occupied) = 40,320 / (40,320 + 60,480) = 40,320 / 100,800 = 0.40 (40%)
P(Active | Not Occupied) = 10,080 / (10,080 + 493,920) = 10,080 / 504,000 = 0.02 (2%)
```

Interpretation:
- When the area is occupied, the media player is playing 40% of the time
- When the area is not occupied, the media player is playing only 2% of the time

This indicates the media player is a good indicator of occupancy (much more likely to be active when occupied).

### Real-Time Usage

When the media player is currently playing:

1. Entity evidence: `True` (active)
2. Likelihoods used: `p_t = 0.40`, `p_f = 0.02`
3. Log contributions (weight = 0.70):
   ```
   contribution_true = log(0.40) * 0.70 = -0.916 * 0.70 = -0.641
   contribution_false = log(0.02) * 0.70 = -3.912 * 0.70 = -2.738
   ```

The large negative contribution to `log_false` (compared to `log_true`) means this evidence strongly supports the "occupied" hypothesis.

## Database Schema

### Entities Table

Stores likelihoods for each entity:
- `entity_id`: The sensor entity ID
- `area_name`: Area this entity belongs to
- `prob_given_true`: `P(Active | Occupied)`
- `prob_given_false`: `P(Active | Not Occupied)`
- `last_updated`: Timestamp of last likelihood update

### Intervals Table

Stores activity intervals for each entity:
- `entity_id`: The sensor entity ID
- `start_time`: When the interval started
- `duration_seconds`: How long the interval lasted
- `state`: The state value during this interval

Intervals are created by the database sync process, which imports state history from Home Assistant's recorder.

## Likelihood Quality Indicators

The quality of learned likelihoods depends on:

1. **Data Volume**: More historical data provides more reliable statistics
2. **Occupied Time Coverage**: Need sufficient occupied and not-occupied periods
3. **Entity Activity**: Entities that are rarely active may have less reliable likelihoods
4. **Sensor Reliability**: Sensors with frequent false positives/negatives will have less useful likelihoods

The system handles edge cases:
- **No data for condition**: Defaults to 0.5 (neutral)
- **All active during condition**: `P(Active | Condition) = 1.0` (clamped to MAX_PROBABILITY)
- **Never active during condition**: `P(Active | Condition) = 0.0` (clamped to MIN_PROBABILITY)

## Default Likelihoods

If history-based learning is disabled or insufficient data is available, the system uses default likelihoods based on entity type:

**Code Reference:** ```custom_components/area_occupancy/data/entity_type.py```

Default values vary by entity type:
- Motion sensors: High `P(Active | Occupied)`, low `P(Active | Not Occupied)`
- Media players: Medium `P(Active | Occupied)`, very low `P(Active | Not Occupied)`
- Environmental sensors: Low `P(Active | Occupied)`, low `P(Active | Not Occupied)`

For non-motion sensors, these defaults are replaced by learned values once sufficient historical data is available.

## See Also

- [Complete Calculation Flow](calculation-flow.md) - End-to-end process
- [Likelihood Feature](../features/likelihood.md) - User-facing documentation
- [Prior Calculation Deep Dive](prior-calculation.md) - Related learning process
- [Bayesian Calculation Deep Dive](bayesian-calculation.md) - How likelihoods are used
- [Entity Evidence Collection](entity-evidence.md) - How evidence is determined

