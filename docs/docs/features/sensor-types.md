# Sensor Types

The Area Occupancy Detection integration supports various sensor types, each contributing differently to the occupancy calculation. This page details each sensor type and how they are used.

## Primary Sensors

| Sensor Type              | Weight | Active States             | Examples                                                             | Usage                                                                                                                                                               |
| ------------------------ | ------ | ------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Primary Occupancy Sensor | N/A    | `on`                       | - Reliable occupancy sensor                                          | - Used to train other sensor correlations<br>- Should be a reliable occupancy sensor<br>- Historical data source for learning priors  |
| Motion/Occupancy Sensors | 0.85   | `on`                      | - Motion sensors<br>- Occupancy sensors                              | - Primary indicator of occupancy<br>- Fast response to presence<br>- Can be combined with multiple motion sensors<br>- Used as ground truth for historical learning |
| Media Devices            | 0.70   | `playing`<br>`paused`     | - TVs<br>- Media players<br>- Game consoles                          | - Strong indicator of presence<br>- Particularly useful for sedentary activities<br>- Can help detect presence when motion is minimal                               |
| Appliances               | 0.30   | `on`<br>`standby`     | - Computers<br>- Fans<br>- Kitchen appliances                        | - Moderate indicator of presence<br>- Helps detect stationary activities<br>- Can indicate recent presence                                                          |
| Doors                    | 0.30   | `closed`                  | - Door sensors                                                       | - Indicates potential room usage<br>- Can help with entry/exit detection<br>- More reliable in private rooms                                                        |
| Windows                  | 0.20   | `open`                    | - Window sensors                                                     | - Indirect indicator of presence<br>- Can suggest room ventilation<br>- Less reliable in common areas                                                               |
| Lights                   | 0.20   | `on`                      | - Light entities                                                     | - Basic presence indicator<br>- Less reliable if automated<br>- Better indicator at night<br>- Can be weighted differently based on time                            |
| Environmental Sensors    | 0.10   | N/A                       | - Illuminance sensors<br>- Humidity sensors<br>- Temperature sensors | - Subtle indicators of presence<br>- Changes can suggest activity<br>- Most useful when combined<br>- Better for long-term patterns                                 |

## Sensor State Management

### Active States
- Each sensor type has defined active states
- States can indicate different levels of certainty
- Multiple states can be considered active
- Custom states can be configured

### Historical Analysis
- Sensor states are compared with known occupancy
- Correlation strength affects probability
- Analysis cached for 6 hours
- 1-30 days of history analyzed

### Reliability Considerations

1. **Direct Detection**

      - Motion & Presence sensors most reliable
      - Media devices highly reliable when active
      - Other sensors provide supporting evidence

2. **False Positives**

      - Automated lights can trigger falsely
      - Open windows might not indicate presence
      - Some appliances run automatically

3. **False Negatives**

      - PIR Motion sensors might miss stillness
      - Environmental changes can be subtle
      - Device states might not update immediately

## Best Practices

1. **Sensor Selection**

      - Use multiple sensor types
      - Prioritize reliable sensors
      - Consider room usage patterns
      - Include complementary sensors

2. **Configuration**

      - Adjust weights based on reliability
      - Set appropriate active states
      - Use historical analysis
      - Regular prior updates

3. **Optimization**

      - Monitor false positives/negatives
      - Adjust weights as needed
      - Consider automation impacts
      - Regular performance review 