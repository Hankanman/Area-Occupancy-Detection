# Entity Evidence Collection

This document explains how the system collects and processes evidence from sensor entities for occupancy calculation.

## Overview

Evidence collection is the process of determining whether each configured sensor entity currently provides evidence of occupancy. The system tracks entity states, determines activity, and integrates decay to provide a complete picture of current evidence.

## Entity State Tracking

Each entity tracks its current state and evidence.

### State Retrieval

Entities retrieve their current state from Home Assistant.

The state retrieval process:
1. Gets state from Home Assistant's state registry (`hass.states.get(entity_id)`)
2. Handles unavailable states: Returns `None` for `"unknown"`, `"unavailable"`, `None`, `""`, or `"NaN"`
3. Returns the state value (string, float, or bool depending on entity type)

### Evidence Determination

The system determines if an entity's current state indicates activity.

Process:
1. **Check Availability**: If state is `None` (unavailable), returns `None`
2. **Binary Sensors**: Checks if state is in `active_states` list
   - Returns `True` if state matches an active state
   - Returns `False` if state doesn't match
3. **Numeric Sensors**: Checks if state value is in `active_range`
   - Converts state to float
   - Returns `True` if `min_val <= state <= max_val`
   - Returns `False` otherwise
4. **Default**: Returns `None` if no active criteria are defined

### Active State Calculation

The active state combines current evidence with decay state:

\[
active = evidence \lor decay\_is\_active
\]

This means an entity is considered "active" if:
- It currently has evidence (`evidence == True`), OR
- It is decaying (was recently active but is now inactive)

## Decay Integration

Decay allows entities to continue providing evidence for a period after they become inactive, preventing rapid flickering of occupancy status.

### Decay Start/Stop Triggers

Decay is managed automatically based on evidence transitions.

The system detects evidence transitions:
1. Gets current evidence from state
2. Compares with previous evidence
3. Detects transitions:
   - **FALSE → TRUE**: Stops decay (if active), evidence is now present
   - **TRUE → FALSE**: Starts decay, evidence was lost
4. Updates previous evidence state for next comparison
5. Triggers recalculation if transition occurred

### Decay Factor Calculation

The decay factor represents how "fresh" the evidence is, ranging from 1.0 (fresh) to 0.0 (expired).

The decay factor is calculated using exponential decay:
```
age = current_time - decay_start_time
decay_factor = 0.5^(age / half_life)
```

The decay factor:
- Starts at 1.0 when decay begins
- Decreases exponentially over time
- Stops automatically when it drops below 0.05 (5%, practical zero)

The entity's decay factor returns 1.0 if evidence is currently `True`, preventing decay from being applied when the entity is actively providing evidence.

### Effective Evidence Calculation

During probability calculation, the system determines "effective evidence" - whether the entity should be treated as providing evidence:

\[
effective\_evidence = current\_evidence \lor decay\_is\_active
\]

Where:
- `current_evidence` is the current evidence state (True/False/None)
- `decay_is_active` is whether the entity's decay is currently active

This means an entity provides effective evidence if:
- It currently has evidence, OR
- It is decaying

## Entity Filtering

Before entities are used in probability calculation, they are filtered to exclude invalid or non-contributing entities.

### Zero Weight Exclusion

Entities with zero weight are excluded from the calculation. Entities with `weight == 0.0` contribute nothing to the calculation, so they are filtered out early.

### Invalid Likelihood Exclusion

Entities with invalid likelihoods are excluded. Invalid likelihoods are:
- `prob_given_true <= 0.0` or `>= 1.0`
- `prob_given_false <= 0.0` or `>= 1.0`

These would cause `log(0)` or `log(1)` errors in log-space calculations, so they are excluded.

### Unavailable Entity Handling

Entities with unavailable states are handled specially. Unavailable entities (`evidence == None`) are skipped unless they are decaying. This prevents unavailable sensors from affecting the calculation while still allowing decaying evidence to contribute.

## Evidence State Machine

The entity evidence system operates as a state machine:

```
┌─────────────┐
│   INACTIVE  │ (evidence = False, decay not active)
└──────┬──────┘
       │ evidence becomes True
       ▼
┌─────────────┐
│   ACTIVE    │ (evidence = True, decay stopped)
└──────┬──────┘
       │ evidence becomes False
       ▼
┌─────────────┐
│   DECAYING  │ (evidence = False, decay active)
└──────┬──────┘
       │ decay expires OR evidence becomes True
       ▼
┌─────────────┐
│   INACTIVE  │
└─────────────┘
```

### State Transitions

1. **INACTIVE → ACTIVE**: Evidence transitions from `False` to `True`
   - Decay is stopped (if active)
   - Entity immediately provides full evidence

2. **ACTIVE → DECAYING**: Evidence transitions from `True` to `False`
   - Decay is started
   - Entity continues to provide evidence (with decreasing strength)

3. **DECAYING → ACTIVE**: Evidence becomes `True` while decaying
   - Decay is stopped
   - Entity immediately provides full evidence

4. **DECAYING → INACTIVE**: Decay factor drops below 5%
   - Decay stops automatically
   - Entity no longer provides evidence

## Example: Motion Sensor Evidence

Consider a motion sensor entity:

### Initial State
- State: `"off"`
- Evidence: `False`
- Decay: Not active
- Effective evidence: `False`

### Motion Detected
- State: `"on"`
- Evidence: `True` (state is in `active_states = ["on"]`)
- Decay: Stopped (if was active)
- Effective evidence: `True`

### Motion Stops
- State: `"off"`
- Evidence: `False`
- Decay: Started (transition detected)
- Effective evidence: `True` (decay provides evidence)

### Decay Progress
- State: `"off"`
- Evidence: `False`
- Decay: Active, factor = 0.7 (30% decayed)
- Effective evidence: `True` (decay still provides evidence, but with reduced strength)

### Decay Expires
- State: `"off"`
- Evidence: `False`
- Decay: Stopped (factor < 0.05)
- Effective evidence: `False`

## Example: Numeric Sensor Evidence

Consider a temperature sensor with `active_range = (20.0, 25.0)`:

### Normal Temperature
- State: `22.5`
- Evidence: `True` (22.5 is in range [20.0, 25.0])
- Effective evidence: `True`

### Temperature Drops
- State: `18.0`
- Evidence: `False` (18.0 is below range)
- Decay: Started (if was previously in range)
- Effective evidence: `True` (decay provides evidence)

### Temperature Rises Back
- State: `23.0`
- Evidence: `True` (23.0 is in range)
- Decay: Stopped
- Effective evidence: `True`

## Entity Manager

The Entity Manager manages all entities for an area.

Responsibilities:
- Creates entities from configuration
- Loads entities from database
- Tracks entity state changes
- Provides access to entities by ID or type
- Manages entity lifecycle

Entities are accessed via:
- `entities.entities`: Dictionary of all entities by entity_id
- `entities.get_entity(entity_id)`: Get specific entity
- `entities.get_entities_by_input_type(type)`: Get entities of specific type

## See Also

- [Complete Calculation Flow](calculation-flow.md) - How evidence is used in calculations
- [Bayesian Calculation Deep Dive](bayesian-calculation.md) - How evidence affects probability
- [Decay Feature](../features/decay.md) - User-facing decay documentation
- [Calculation Feature](../features/calculation.md) - User-facing calculation documentation

