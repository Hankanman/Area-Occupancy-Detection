# Historical Analysis

The Area Occupancy Detection integration uses historical data analysis to improve its accuracy over time. This feature learns from past sensor states and their correlation with known occupancy patterns.

## Overview

Historical analysis serves several purposes:

1. Learning sensor behavior patterns
2. Calculating prior probabilities
3. Improving accuracy over time
4. Adapting to usage patterns

## How It Works

### Data Collection

The integration analyzes:
- Sensor state history
- Primary occupancy sensor data
- State correlations
- Temporal patterns

### Analysis Period

- Default: 7 days
- Configurable: 1-30 days
- Longer periods:
    - More stable results
    - Better pattern recognition
    - Higher memory usage
- Shorter periods:
    - Faster adaptation
    - Less memory usage
    - More volatile results

## Prior Probability Calculation

### Process

1. **Data Gathering**

      - Collect sensor states
      - Record occupancy states
      - Store timestamps
      - Track correlations

2. **Correlation Analysis**

      - Compare sensor states with occupancy
      - Calculate coincidence rates
      - Determine reliability scores
      - Weight by confidence

3. **Prior Generation**

      - Calculate base probabilities
      - Apply sensor weights
      - Consider temporal factors
      - Generate final priors

### Caching

- Results cached for 6 hours
- Reduces database load
- Maintains responsiveness
- Automatic refresh on:
    - Cache expiry
    - Configuration changes
    - Manual updates from service

## Sensor Learning

### Primary Occupancy Sensor

- Used as ground truth
- Trains other sensors
- Establishes baselines
- Validates patterns

### Sensor Types

| Sensor Type    | Learning Characteristics                   | Pattern Analysis                               | Reliability |
| -------------- | ------------------------------------------ | ---------------------------------------------- | ----------- |
| Motion Sensors | - Direct correlation<br>- High confidence  | - Quick learning<br>- Reliable patterns        | Very High   |
| Media Devices  | - State correlation<br>- Usage patterns    | - Activity duration<br>- Occupancy overlap     | High        |
| Appliances     | - Usage correlation<br>- Activity patterns | - Operation cycles<br>- User interaction       | Medium      |
| Environmental  | - Subtle changes<br>- Long-term patterns   | - Indirect indicators<br>- Background learning | Low         |

## Configuration

### Settings

1. **History Period**
   ```yaml
   history_period: 7  # days
   ```
      - Affects analysis depth
      - Balances accuracy/resources
      - Customizable per area

2. **Historical Analysis**
   ```yaml
   historical_analysis_enabled: true
   ```
      - Enable/disable learning
      - Per-area setting
      - Runtime configurable

### Manual Updates

Use the service:
```yaml
service: area_occupancy.update_priors
data:
  entry_id: "<config_entry_id>"
  history_period: 14  # optional
```
