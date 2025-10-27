---
description: Detect occupancy in rooms with a single entry/exit point
---

# Wasp in Box

The "Wasp in Box" feature provides enhanced occupancy detection for rooms with a single entry/exit point (like bathrooms, closets, or small offices). It uses a simple but effective principle: if someone enters a room and the door closes, they remain in that room until the door opens again.

## How It Works

The feature is named after the concept of a wasp trapped in a box - once inside, it remains there until an opening appears. Similarly, once a person enters a closed space, they must be considered "present" until they exit through the door.

### With Door and Motion Sensors

When both door and motion sensors are available:

1. **Entry Detection** (Bidirectional):
   - Pattern A: Door closes → Motion detected → Room becomes OCCUPIED
   - Pattern B: Motion detected → Door closes (within timeout window) → Room becomes OCCUPIED
   - The state persists even if motion stops
   - Works regardless of door opening direction

2. **Exit Detection**:
   - Door opens → Room becomes UNOCCUPIED
   - Any subsequent motion requires a new entry cycle

### With Door Sensor Only

When only a door sensor is available:

1. **Entry Detection**:
   - Door closes while the room is unoccupied → Room becomes OCCUPIED

2. **Exit Detection**:
   - Door opens while the room is occupied → Room becomes UNOCCUPIED

## Benefits

- **Solves the "bathroom problem"**: Traditional motion sensors often time out while someone is in the bathroom, leading to lights turning off at inconvenient moments
- **Works with minimal sensors**: Can function with just a door sensor if needed
- **Complements motion detection**: Fills the gaps where motion detection fails
- **Improves automation reliability**: More accurately maintains occupancy state for rooms where people may be stationary

## Configuration

The Wasp in Box feature can be enabled in the integration configuration UI:

1. Navigate to the integration configuration 
2. Expand the "Wasp in Box Settings" section
3. Enable the feature and configure the options:

| Option             | Description                                                            | Default         |
| ------------------ | ---------------------------------------------------------------------- | --------------- |
| Enable Wasp in Box | Activates the feature                                                  | Disabled        |
| Motion Timeout     | Duration in seconds that motion remains valid after detection          | 300 (5 minutes) |
| Sensor Weight      | How heavily this sensor influences the overall probability calculation | 0.8             |
| Verification Delay | Delay before re-checking motion to verify occupancy (0 = disabled)    | 0 (disabled)    |

## Integration

The feature creates a binary sensor that becomes part of your Area Occupancy Detection system:

- **Entity ID**: `binary_sensor.[area_name]_wasp_in_box`
- **States**: `on` (occupied) / `off` (unoccupied)
- **Attributes**:
  - `door_state`: Current state of the door sensor
  - `last_door_time`: Timestamp of last door state change
  - `motion_state`: Current motion state (if using motion sensors)
  - `last_motion_time`: Timestamp of last motion detection (if using motion sensors)
  - `motion_timeout`: Configured timeout value
  - `verification_delay`: Configured verification delay (seconds)
  - `verification_pending`: Whether a verification check is scheduled

The sensor's state is considered alongside other sensors in the Bayesian probability calculation, with its influence determined by the configured weight.

## Motion Re-verification

The sensor includes an optional motion re-verification feature to prevent false positives caused by motion sensor cooldown periods:

- When enabled, the sensor marks the room as occupied immediately when triggered
- After the configured delay (in seconds), it re-checks motion sensors to verify occupancy
- If motion is still detected: occupancy is maintained
- If no motion is detected: occupancy is automatically cleared (false positive detected)
- Set to 0 to disable this feature

This is particularly useful in scenarios where someone might quickly enter and exit a room before motion sensors clear their cooldown period.

## Use Cases

- **Bathrooms**: Maintain occupancy even when a person is showering or otherwise stationary
- **Closets**: Detect occupancy in walk-in closets with minimal sensor requirements
- **Small Offices**: Maintain occupancy state when people are sitting still at a desk
- **Storage Rooms**: Track when people are retrieving items from storage
- **Laundry Rooms**: Detect presence during laundry activities with minimal motion
- **Rooms with inward-opening doors**: Bidirectional detection works regardless of door direction

## Technical Details

The Wasp in Box sensor tracks state transitions in a finite state machine:

1. **UNKNOWN** → Initial state before any data is collected
2. **UNOCCUPIED** → No one is in the room
3. **OCCUPIED** → Someone is in the room

The sensor uses Home Assistant's state tracking to monitor door and motion entities, processing their state changes to update its internal state based on the logic described above.

# Wasp in Box Logic

The "Wasp in Box" sensor is a virtual binary sensor that helps determine if a space is occupied based on door states and motion detection. The concept derives its name from the idea of a wasp trapped in a box: once the door (lid) is closed with the wasp inside, the room is considered occupied until a door opens, indicating the wasp has left.

## How It Works

The sensor monitors one or more door sensors and optional motion sensors, supporting **bidirectional entry detection**:

1. When a door closes and recent motion was detected, the room is marked as occupied.
2. When a door closes and there are no motion sensors, the room is assumed to be occupied.
3. When a door opens while the room is occupied, the room is marked as unoccupied.
4. When motion is detected while all doors are closed, the room is marked as occupied.

**Bidirectional Entry Patterns:**
- Pattern A: Door closes → Motion detected → Occupied
- Pattern B: Motion detected (within timeout) → Door closes → Occupied

This ensures reliable detection regardless of whether the door opens inward or outward.

The sensor retains its state between Home Assistant restarts, making it reliable for long-term occupancy tracking.

## Maximum Duration Feature

The sensor can be configured with a maximum occupancy duration. This addresses scenarios where a space might be incorrectly marked as occupied for extended periods:

1. When enabled, the sensor will automatically reset to unoccupied after the specified duration.
2. Set to 0 (default) to disable this feature and maintain the traditional Wasp in Box behavior.
3. Useful in environments where someone might leave through an unmonitored exit.

## Configuration

The Wasp in Box sensor can be configured in the integration settings:

- **Enable Wasp in Box**: Turn this virtual sensor on or off
- **Motion Timeout**: How long motion events are considered recent (in seconds)
- **Wasp Weight**: The weight factor for this sensor in probability calculations (0.1-1.0)
- **Maximum Occupied Duration**: Maximum time (in seconds) a space can be marked as occupied before automatically resetting (0 = no limit)
- **Verification Delay**: Delay before re-checking motion to verify occupancy, in seconds (0-120, 0 = disabled)

### Motion Re-verification

The verification delay feature helps prevent false positives when people enter and exit quickly:

- **How it works**: When the room is marked as occupied, wait the specified delay, then re-check motion sensors
- **If motion present**: Keep the room occupied (genuine occupancy)
- **If no motion**: Clear occupancy (false positive from sensor cooldown)
- **Recommendation**: Start with 15-30 seconds for most setups; adjust based on your motion sensor cooldown periods

## Example Use Cases

- **Small rooms with single entry/exit points**: The sensor excels at tracking occupancy in bathrooms, closets, or offices with one door.
- **Spaces with incomplete sensor coverage**: If you don't have motion sensors that cover the entire room, the Wasp in Box can still provide reliable occupancy detection.
- **Persistent occupancy status**: Unlike regular motion sensors, the Wasp in Box maintains its state even during periods of inactivity, as long as doors remain closed.

## Attributes

The sensor provides these attributes:

- `door_state`: Current state of the monitored door(s)
- `last_door_time`: Timestamp of the last door state change
- `motion_state`: Current state of motion sensors (if applicable)
- `last_motion_time`: Timestamp of the last motion detection (if applicable)
- `motion_timeout`: Current motion timeout setting
- `max_duration`: Maximum time in seconds the space can be marked as occupied (0 = no limit)
- `last_occupied_time`: Timestamp when the space was last marked as occupied
- `verification_delay`: Configured verification delay in seconds (0 = disabled)
- `verification_pending`: Whether a verification check is currently scheduled (boolean) 