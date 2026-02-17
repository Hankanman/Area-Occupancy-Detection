---
description: Why choose Area Occupancy Detection over core Home Assistant features?
---

# Why Area Occupancy Detection?

Have you ever had your lights turn off while you're still in the room? Or watched your smart home mark you as "away" while you're sitting perfectly still, watching TV? These frustrating experiences happen because most occupancy detection relies on simple motion sensors that can't understand context.

**Area Occupancy Detection** solves these real-world problems by thinking more intelligently about what "occupied" really means. Instead of just checking if motion was detected, it combines multiple clues, learns from your patterns, and calculates the probability that someone is actually there.

**Area Occupancy Detection doesn't automate anything for you**—it provides the intelligent occupancy information you need to create reliable automations. AOD creates sensors that your automations can use to control lights, heating, and other devices. Think of AOD as the "smart sensor" that gives your automations better data to work with.

## The Quick Answer

**Here's why AOD is different:**

**HA**: "Motion detected? Occupied. Motion stopped? Not occupied."

🎯 **AOD**: "Let me check motion, TV, doors, appliances, learned patterns, and time of day... 75% confident someone is there."

**HA**: You configure everything manually. It never learns.

🧠 **AOD**: Learns from your history automatically. Gets smarter over time. Knows you're usually in the kitchen Sunday mornings.

**Core HA**: One sensor fails → wrong answer.

🔀 **AOD**: Combines multiple sensors intelligently. If motion misses you, TV being on maintains occupancy probability → your automations keep lights on.

**Core HA**: Motion stops → occupancy sensor turns off → automations turn lights off immediately.

⏱️ **AOD**: Motion stops → probability gradually decreases → occupancy sensor stays on longer → your automations keep lights on while you sit still.

**Core HA**: Basic features only.

✨ **AOD**: Activity detection (what's happening, not just who's there), sleep presence tracking, "Wasp in Box" for bathrooms, whole-home aggregation, purpose-based defaults.

**The bottom line:** AOD provides intelligent occupancy sensors that your automations can use. It learns, adapts, and understands context—so when you build automations that respond to occupancy, they work reliably instead of turning lights off while you're still in the room.

## Creating Automations with AOD

Here's how AOD fits into your automation workflow:

### The Workflow

1. **AOD analyzes your sensors** → Motion, TV, doors, appliances, learned patterns
2. **AOD calculates probability** → Combines all inputs using Bayesian inference
3. **AOD creates occupancy sensors** → Binary occupancy status and probability sensors
4. **Your automations use these sensors** → Trigger actions based on occupancy state or probability
5. **AOD learns and adapts** → Gets smarter over time, improving your automations automatically

### What AOD Provides

AOD creates sensors that your automations can use:

- **Occupancy Status**: Binary sensor (`on` = occupied, `off` = clear) - use this in most automations
- **Occupancy Probability**: Percentage (0-100%) - use this for conditional or gradual actions
- **Detected Activity**: What's happening in the room (showering, cooking, watching TV, sleeping, etc.) - use this for context-aware automations
- **Sleeping**: Whether people are sleeping in the area - use for overnight occupancy
- **Presence Confidence / Environmental Confidence**: Split view of what's driving the probability
- **Prior Probability**: Baseline from learned patterns - useful for monitoring and debugging
- **Threshold**: Adjustable setting - fine-tune without reconfiguration

### How You Use It

You create automations that respond to AOD's sensors. For example:

- **Turn lights on** when occupancy status turns `on`
- **Turn lights off** when occupancy status turns `off` (with a delay to prevent flickering)
- **Adjust heating** based on occupancy probability
- **Dim lights gradually** as probability decreases

The key difference: AOD provides intelligent occupancy data. You decide what actions to take based on that data.

For automation examples, see the [Basic Usage Guide](basic-usage.md).

## What Core Home Assistant Can Do

Home Assistant provides several built-in ways to detect occupancy, each with its own strengths and limitations:

### Binary Motion Sensors

The simplest approach: a motion sensor reports `on` when motion is detected, `off` when it's not. This is straightforward, but has significant limitations:

- **No context**: Can't tell the difference between someone sitting still and an empty room
- **Instant state changes**: Motion stops → immediately marked as unoccupied
- **Single point of failure**: One sensor determines everything
- **No learning**: Doesn't adapt to your patterns

### Template Sensors

You can manually combine multiple sensors using YAML templates, but this requires YAML knowledge and manual configuration for every combination. No learning, no probability—just binary results.

### Automation Conditions

You can use AND/OR logic in automations to combine sensors, but each automation requires manual configuration. No learning, no probability, and you must update everything when you add sensors.

### History and Statistics

Home Assistant can analyze historical data, but you must manually query APIs, write scripts, and configure everything yourself. No automatic learning.

### Summary of Core HA Limitations

| Feature                  | Core HA                      | AOD                                    |
| ------------------------ | ---------------------------- | -------------------------------------- |
| **Learning**             | Manual configuration only    | Automatic learning from history        |
| **Probability**          | Binary on/off only           | Probability percentage (0-100%)        |
| **Adaptation**           | Static configuration         | Adapts to your patterns                |
| **Sensor Reliability**   | You must configure manually  | Learns automatically                   |
| **Multi-Sensor Fusion**  | Manual templates/automations | Automatic with learned weights         |
| **Time-Based Patterns**  | Manual configuration         | Learns day/time patterns automatically |
| **Activity Detection**   | None                         | Detects what's happening (cooking, showering, etc.) |
| **Sleep Tracking**       | None                         | Detects sleep via phone companion app  |
| **Specialized Features** | None                         | Wasp in Box, whole-home aggregation, purpose-based config |

## What Makes AOD Different

### 1. Intelligent Probability vs. Binary Logic

**Core HA Approach:**

- Binary state: `occupied` or `not occupied`
- Instant decisions based on current sensor states
- No nuance or confidence level

**AOD Approach:**

- Probability percentage: 0% to 100% confidence
- Configurable threshold (e.g., 75% = occupied)
- More nuanced understanding of occupancy

**Example:**

- **Core HA**: Motion detected → `occupied = true` (even if it's just a pet)
- **AOD**: Motion detected + TV off + door closed + learned patterns → `probability = 45%` → below threshold → `occupied = false`

**Benefit:** Fewer false positives and false negatives. The system understands that not all motion means occupancy.

### 2. Automatic Learning vs. Manual Configuration

**Core HA Approach:**

- You configure everything manually
- You decide which sensors to trust
- You set up time-based patterns yourself
- Static configuration that doesn't improve

**AOD Approach:**

- **Prior Learning**: Automatically learns baseline occupancy patterns (global and time-based). See [Prior Learning](../features/prior-learning.md) for details
- **Likelihood Learning**: Learns how reliable each sensor is. See [Sensor Likelihoods](../features/likelihood.md) for details
- **Time-based Patterns**: Learns day-of-week and time-of-day patterns automatically

**Benefit:** Gets smarter over time. The longer it runs, the more accurate it becomes. No manual tuning required.

### 3. Multi-Sensor Fusion vs. Manual Combination

**Core HA Approach:**

- You manually combine sensors with templates or automation conditions
- You decide how to weight each sensor
- Requires YAML knowledge or complex automations
- Must update configuration when you add sensors

**AOD Approach:**

- Automatically combines motion, media, appliances, doors, windows, covers, power sensors, sleep presence, and environmental sensors
- Separates presence indicators (80% weight) from environmental support (20% weight) for more accurate results
- Each sensor type has learned reliability with weighted contributions
- See [Bayesian Calculation](../features/calculation.md) for details

**Example:**

- **Core HA**: You write a template: `motion OR tv_playing OR computer_on`
- **AOD**: Automatically considers all sensors, weighs them by learned reliability, and calculates probability

**Benefit:** Richer context and more accurate detection. The system understands that multiple weak signals can be as strong as one strong signal.

### 4. Probability Decay vs. Instant On/Off

**Core HA Approach:**

- Motion stops → occupancy sensor immediately turns off
- No grace period for sitting still
- Your automations turn lights off even if you're still in the room

**AOD Approach:**

- Gradual probability decay when activity stops
- Occupancy sensor stays on longer, giving your automations better data
- See [Probability Decay](../features/decay.md) for details

**Example:**

- **Core HA**: You sit still for 1 minute → motion stops → occupancy sensor turns off → your automation turns lights off
- **AOD**: You sit still → motion stops → probability gradually decreases over 5-10 minutes → occupancy sensor stays on → your automation keeps lights on until probability drops below threshold

**Benefit:** Prevents lights from turning off when you're still in the room. More natural, less frustrating behavior.

### 5. Specialized Features

AOD includes features not available in core Home Assistant:

#### Activity Detection

Identifies *what* is happening in a room — not just that someone is there. The system can detect activities like showering, cooking, watching TV, working, sleeping, and more. Activities are tied to room purpose, so "showering" only appears in bathrooms and "cooking" only in kitchens. See [Activity Detection](../features/activity-detection.md) for details.

**Example:**

- **Core HA**: Motion detected in kitchen → you know someone is there, but not what they're doing
- **AOD**: Appliance on + humidity elevated + temperature elevated = `cooking` with 85% confidence → your automation can turn on the extractor fan

#### Sleep Presence Detection

Detects when people are sleeping by combining Home Assistant Person entities with phone-reported sleep confidence from the Companion App. When sleep is detected, the area maintains high occupancy probability overnight. See [Sleep Presence](../features/sleep-presence.md) for details.

**Example:**

- **Core HA**: No motion in bedroom overnight → occupancy turns off → heating turns off while you sleep
- **AOD**: Person home + phone reports sleeping → Sleeping sensor on → occupancy probability stays high → your automation keeps heating on

#### Wasp in Box

Special logic for rooms with a single entry/exit point (bathrooms, closets, small offices). If someone enters and the door closes, they remain until the door opens again. See [Wasp in Box](../features/wasp-in-box.md) for details.

**Example:**

- **Core HA**: Motion stops in bathroom → occupancy sensor turns off → your automation turns lights off (even if door is closed)
- **AOD**: Door closed + recent motion → maintains occupancy probability → occupancy sensor stays on → your automation keeps lights on

#### All Areas Aggregation

Automatically creates aggregated entities across all configured areas for whole-home occupancy detection. No manual configuration required.

#### Purpose-Based Configuration

Selecting a room purpose (Living Room, Bedroom, Kitchen, etc.) automatically sets sensible defaults for decay half-life and other parameters. See [Purpose-Based Configuration](../features/purpose.md) for details.

## Real-World Scenarios

### Scenario 1: Watching TV

**The Problem:**
You're watching TV in the living room. You sit still for 10 minutes. The motion sensor stops detecting movement.

**Core HA Solution:**

- Motion sensor → `off`
- Occupancy sensor → `off`
- Your automation turns lights off
- You're sitting in the dark, frustrated

**AOD Solution:**

- Motion sensor → inactive
- TV → `playing`
- Learned pattern: "Evening + TV playing = likely occupied"
- Probability: 85% (above threshold)
- Occupancy sensor → `on`
- Your automation keeps lights on
- You continue watching comfortably

### Scenario 2: Working at Desk

**The Problem:**
You're working at your desk. No motion for 15 minutes while you read or type.

**Core HA Solution:**

- Motion sensor → `off` after timeout
- Occupancy sensor → `off`
- Your automations turn lights/heating off

**AOD Solution:**

- Motion sensor → inactive
- Computer/appliance → `on`
- Learned pattern: "Work hours + computer on = likely occupied"
- Probability: 70% (above threshold)
- Occupancy sensor → `on`
- Your automations keep lights/heating on
- You work comfortably

### Scenario 3: Pet Walking Through

**The Problem:**
Your pet walks through the room, triggering the motion sensor.

**Core HA Solution:**

- Motion detected → occupancy sensor → `on`
- Your automation turns lights on
- False positive

**AOD Solution:**

- Motion detected → weak signal
- No other sensors active
- Learned pattern: "Single brief motion = likely not occupied"
- Probability: 25% (below threshold)
- Occupancy sensor → `off`
- Your automation doesn't turn lights on
- False positive avoided

### Scenario 4: Bathroom (Wasp in Box)

**The Problem:**
You're in the bathroom. Motion stops, but the door is closed.

**Core HA Solution:**

- Motion stops → occupancy sensor → `off`
- Your automation turns lights off
- You're in the dark

**AOD Solution (with Wasp in Box enabled):**

- Motion stops → but door is closed
- Wasp in Box logic: "Door closed + recent motion = occupied"
- Probability: 80% (above threshold)
- Occupancy sensor → `on`
- Your automation keeps lights on
- You're comfortable

## When to Use AOD vs. Core HA

### Use AOD When:

- ✅ You want intelligent, adaptive occupancy detection
- ✅ You have multiple sensors per area (motion, media, doors, etc.)
- ✅ You want automatic learning from your patterns
- ✅ You need probability-based detection (not just binary)
- ✅ You want to know *what* is happening in a room (activity detection)
- ✅ You need reliable overnight bedroom occupancy (sleep presence)
- ✅ You want specialized features (Wasp in Box, whole-home aggregation)
- ✅ You want less maintenance (system learns and adapts)
- ✅ You're frustrated with automations turning lights off while you're still in the room
- ✅ You want your smart home to feel truly smart

### Use Core HA When:

- ✅ Simple binary motion detection is sufficient for your needs
- ✅ You prefer manual control over automatic learning
- ✅ You only have a single motion sensor per area
- ✅ You want to learn YAML/automation configuration
- ✅ You need very simple, predictable behavior
- ✅ You enjoy manually configuring and maintaining templates

## Key Benefits Summary

1. **Accuracy**: Multi-sensor fusion + learning = fewer false positives/negatives
2. **Adaptability**: Learns your patterns automatically, gets smarter over time
3. **Intelligence**: Bayesian probability vs. simple binary logic
4. **Context**: Activity detection tells you *what* is happening, not just *if* someone is there
5. **Overnight**: Sleep presence keeps bedrooms occupied while you sleep
6. **Convenience**: UI-based configuration, automatic learning, purpose-based defaults
7. **Specialized Features**: Wasp in Box, whole-home aggregation, dual presence/environmental model
8. **Privacy**: Runs locally, no cloud services, full control

## Getting Started

Ready to try Area Occupancy Detection? See the [Installation Guide](installation.md) and [Configuration Guide](configuration.md) to get started. The integration creates occupancy sensors that you can use in your automations. It starts learning from your sensor history immediately and gets smarter over time.

## Learn More

- **[Bayesian Calculation](../features/calculation.md)**: How probability is calculated
- **[Prior Learning](../features/prior-learning.md)**: How the system learns from history
- **[Activity Detection](../features/activity-detection.md)**: How activity detection works
- **[Sleep Presence](../features/sleep-presence.md)**: How sleep detection works
- **[Probability Decay](../features/decay.md)**: How decay prevents false negatives
- **[Wasp in Box](../features/wasp-in-box.md)**: Special logic for single-entry rooms
- **[Sensor Likelihoods](../features/likelihood.md)**: How sensor reliability is learned
