# Prior Probability Learning

To make the Bayesian calculations more accurate and specific to your environment, the integration automatically learns key probability values from your sensor history. This process adapts the system to how _you_ actually use the space.

## What is a Prior Probability?

Simply put the prior is the probability that the area is occupied at any given time. For example "I spend about 30% of my time in the living room" so the prior for the living room is 0.30. This sets the "ground truth" for the living room.

Initially, the integration uses default prior values. However, the **Prior Learning** process aims to calculate a more accurate prior based on the historical behavior of your **motion sensors**, reflecting how often that area is typically occupied. This is done by analyzing the history of your sensors over a lookback period (default: 90 days).

In the context of Bayesian probability, the **prior probability** (often denoted as `P(Occupied)` or simply `prior`) represents our initial belief about the likelihood of an event _before_ considering any new evidence. For this integration, it's the baseline probability that the area is occupied, independent of the _current_ state of the sensors.

## How Priors Are Calculated

The system calculates two types of priors to provide a robust baseline:

1.  **Global Prior**: A single value representing the overall "busyness" of a room (e.g., a living room is occupied 30% of the time, a guest room 1% of the time).
2.  **Time-Based Prior**: A time-specific probability based on the day of week and time of day (e.g., "Mondays at 09:00").

The final prior used for real-time detection is a combination of these two, giving weight to the specific time of day while using the global prior as a stable anchor.

### 1. Global Prior Calculation

The Global Prior is determined by analyzing the history of your sensors over a lookback period (default: 90 days).

- **Motion Analysis**: First, it calculates the percentage of time motion sensors were active.
- **Fallback for Low Activity**: If motion sensors show very low activity (less than 10% of the time), the system assumes people might be sitting still (e.g., watching TV or working). It then checks **Media Players** and **Appliances** to supplement the occupancy data.
- **Result**: This produces a single probability value (e.g., 0.35 or 35%) that serves as the general baseline for the room.

### 2. Time-Based Prior Calculation

The Time-Based Prior provides granularity by breaking the week into 1-hour slots (e.g., "Monday 09:00-10:00", "Tuesday 14:00-15:00").

- **Grid**: It creates a schedule of 168 slots (24 hours $\times$ 7 days).
- **History Analysis**: For each slot, it looks at historical data to see how often the room was occupied during that specific hour on that day of the week.
- **Result**: This produces a unique probability for every hour of the week, allowing the system to "expect" occupancy at usual times (like evening TV time) and "expect" vacancy at others (like work hours).

## Ground Truth: Defining "Occupancy"

To learn from history, the system needs to know when the room was _actually_ occupied. It uses **Motion Sensors** as the "Ground Truth" or Gold Standard.

- **Why Motion?**: Motion sensors are the most direct indicator of human presence. They rarely generate false positives (indicating presence when no one is there).
- **Motion Timeout**: The system automatically accounts for the time after motion stops. If your motion sensors have a timeout (e.g., 5 minutes), this "cooldown" period is counted as occupied time, ensuring consistent logic across all calculations.

## Real-Time Probability Adjustment

When the integration runs in real-time, it determines the current Prior Probability dynamically:

1.  **Combine Priors**: It starts by mixing the **Global Prior** and the specific **Time-Based Prior** for the current hour. This balances the general busyness of the room with the specific expectation for the current time.
2.  **Bias Towards Safety**: The system slightly biases the prior towards assuming occupancy (multiplying by a small factor). This is a "better safe than sorry" heuristic to prevent lights from turning off on you.
3.  **Apply Minimum Floor**: Finally, it applies the **Minimum Prior Override** (if configured). This ensures that even in very rarely used rooms, the probability never drops so low that the system becomes unresponsive to new sensor activity.

## Handling Special Situations

The system is designed to handle various edge cases robustly:

- **New Installations**: If there is no history yet, the system uses a neutral **Default Prior** (50%) to avoid making assumptions until data is available.
- **Missing Data**: If specific time slots have no history (e.g., a power outage occurred every Tuesday at 2 AM), it falls back to safe defaults.
- **Sensor Dropouts**: If a sensor becomes unavailable, it is temporarily excluded from the calculation to prevent skewing the data.

## Minimum Prior Override

You can configure a `min_prior_override` for each area.

- **Purpose**: This setting prevents the calculated prior from dropping too low in rarely-used rooms (like a guest room or attic).
- **Effect**: If the historical probability is extremely low (e.g., 0.01%), the system might require overwhelming evidence to switch to "Occupied." Setting a minimum override (e.g., 0.10) ensures the system remains responsive to new motion or sensor activity, even in "quiet" rooms.

## Default Behaviors

When data is missing or insufficient, the system uses these defaults:

- **Neutral (0.5)**: Used when historical data is completely missing or invalid (e.g., during initial setup).
- **Time Slot Default (0.5)**: Used for specific time slots that have no recorded history.
- **Empty Bias (0.1)**: Before the very first analysis runs (typically 5 minutes after startup), the system assumes a low probability to avoid false triggers.
