# Features Overview

The Area Occupancy Detection integration combines multiple sensor inputs with Bayesian probability calculations to provide intelligent and accurate room occupancy detection. This page provides an overview of all major features.

## Core Features

### Multi-Sensor Integration

The integration supports various sensor types, each contributing to the overall occupancy calculation:

| Sensor Type | Examples | Description |
|------------|------------|-------------|
| Primary Sensors | - Motion/occupancy sensors<br>- Primary occupancy sensor for learning<br>- Direct presence detection | Main sensors used for occupancy detection |
| Media & Entertainment | - TVs and media players<br>- Gaming consoles<br>- Active states: playing, paused | Entertainment devices that indicate presence |
| Appliances & Devices | - Computers and electronics<br>- Home appliances<br>- Active states: on, standby | Devices that show room usage |
| Environmental Sensors | - Illuminance sensors<br>- Temperature sensors<br>- Humidity sensors | Environmental indicators of occupancy |
| Access Points | - Door sensors (active when closed)<br>- Window sensors (active when open)<br>- Light entities | Entry/exit points and lighting |

## Intelligent Detection

### Bayesian Probability

- Combines multiple data sources
- Weighted sensor contributions
- Real-time probability updates
- Bounded results (1-99%)
- Default priors based on typical patterns

### Time Decay

- Gradual probability reduction
- Configurable decay window
- Minimum delay period
- Exponential decay function
- Automatic trigger reset

### Historical Learning

- Analyzes past sensor states
- Learns from occupancy patterns
- 6-hour result caching
- Configurable history period (1-30 days)
- Automatic correlation discovery

## Customization

### Sensor Weights

Sensor weights are used to adjust the contribution of each sensor to the overall occupancy calculation. You can change the weights in the configuration page.

Default weights reflect reliability:

| Sensor Type | Default Weight |
|-------------|---------------|
| Motion sensors | 0.85 |
| Media devices | 0.70 |
| Appliances | 0.30 |
| Doors | 0.30 |
| Windows | 0.20 |
| Lights | 0.20 |
| Environmental | 0.10 |

### Configuration Options

- Adjustable threshold (1-99%)
- History period selection
- Decay window customization
- State mapping configuration
- Per-area settings

To learn more about specific features:

- [Sensor Types](sensor-types.md) - Detailed sensor information
- [Probability Calculation](probability-calculation.md) - How probabilities are computed
- [Time Decay](time-decay.md) - Decay system details
- [Historical Analysis](historical-analysis.md) - Learning system documentation 