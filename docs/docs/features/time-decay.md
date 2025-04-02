# Time Decay

The Area Occupancy Detection integration includes a sophisticated time decay system that gradually reduces the occupancy probability when no new sensor triggers occur. This feature helps prevent false positives from lingering sensor states.

## How Time Decay Works

### Basic Concept

Time decay gradually reduces the occupancy probability over time when:

- No new sensor triggers occur
- The minimum delay period has passed
- Decay is enabled in configuration

### Why is it needed?

- To prevent false positives from lingering sensor states
- To provide a more natural-feeling decay
- To balance responsiveness with stability
- Provides a more accurate representation of occupancy patterns (e.g. when someone is in the room but not moving or just wandered out of the room but just for a moment)

### How Can I Use It?

You could tie the brightness of a light to the occupancy probability, so the light slowly dims when the room is empty!

### Decay Function

The integration uses an exponential decay function:

\[P(t) = P_0 \times e^{-\lambda t}\]

Where:
- \(P(t)\) is the probability at time t
- \(P_0\) is the initial probability
- \(\lambda\) = 0.866433976 (decay constant)
- \(t\) is time since last sensor update

### Decay Constant

The decay constant (λ = 0.866433976) is carefully chosen to:
- Reduce probability to 25% at half of decay window
- Provide smooth, natural-feeling decay
- Balance responsiveness with stability

## Configuration Options

### Decay Enable/Disable

- Can be enabled/disabled during setup
- Configurable per area
- Can be changed without recreation

### Timing Parameters

1. **Decay Window**

      - Default: 600 seconds (10 minutes)
      - Total time for full decay effect
      - Customizable per area needs
      - Affects decay rate calculation

## Interaction with Sensors

### Trigger Behavior

- Any sensor trigger resets decay
- New probability calculated first
- Decay timer restarted

## Monitoring Decay

### Decay Status Sensor

- Entity: `sensor.[name]_decay_status`
- Shows current decay percentage
- Updates in real-time
- Useful for troubleshooting

### Attributes

- `decay_start_time`: When decay began
- `original_probability`: Pre-decay probability
- `decay_window`: Current setting

## Best Practices

### Setting Decay Parameters

1. **Decay Window**

      - Match to room usage
      - Consider automation needs
      - Balance with sensor coverage
      - Test different values

### Optimization Tips

1. **High Traffic Areas**

      - Shorter minimum delay
      - Shorter decay window
      - More frequent updates

2. **Quiet Areas**

      - Longer minimum delay
      - Extended decay window
      - Stable probability

3. **Mixed Use**

      - Balanced settings
      - Regular monitoring
      - Adjust based on patterns

## Troubleshooting

### Common Issues

1. **Too Fast Decay**

      - Extend decay window
      - Check sensor coverage
      - Verify trigger handling

2. **Too Slow Decay**

      - Shorten decay window
      - Review sensor weights
      - Check for stuck sensors
