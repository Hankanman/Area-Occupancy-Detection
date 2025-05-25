# Environmental Sensor Analysis

The Area Occupancy Detection integration includes advanced environmental sensor analysis capabilities that can enhance occupancy detection accuracy by analyzing patterns in environmental data such as temperature, humidity, and illuminance sensors.

## Overview

Environmental sensor analysis works as an optional enhancement to the main Bayesian probability calculation. It analyzes environmental sensor readings to provide additional occupancy probability insights that are then combined with the primary sensor-based calculations.

## Supported Environmental Sensors

The integration automatically detects and can utilize the following types of environmental sensors from your Home Assistant setup:

- **Temperature sensors** (`sensor` domain with `temperature` device class)
- **Humidity sensors** (`sensor` domain with `humidity` device class)  
- **Illuminance/Light sensors** (`sensor` domain with `illuminance` device class)

## How It Works

### 1. Automatic Detection
During configuration, the integration automatically detects available environmental sensors in the selected area and includes them in the analysis if they are configured in the environmental sensors section.

### 2. Analysis Methods
The system supports multiple analysis methods:

- **Deterministic Analysis**: Uses statistical methods like z-score analysis, variance calculations, and correlation analysis to determine occupancy likelihood
- **Machine Learning Analysis**: (Future enhancement) Uses trained models to predict occupancy based on environmental patterns
- **Hybrid Analysis**: (Future enhancement) Combines both deterministic and ML approaches

### 3. Integration with Main Calculation
Environmental analysis contributes up to 20% adjustment to the final occupancy probability based on:
- Environmental probability (how likely occupancy is based on environmental readings)
- Confidence level (how confident the system is in its environmental analysis)
- Contribution weight (currently set to 20% maximum impact)

## Configuration

Environmental sensors are configured in the main Area Occupancy setup flow:

1. **Navigate to Settings → Devices & Services → Area Occupancy**
2. **Select your area configuration**
3. **In the Environmental section**, select the sensors you want to include:
   - Illuminance Sensors
   - Humidity Sensors  
   - Temperature Sensors
4. **Set the Environmental Weight** (determines the influence of environmental analysis on final probability)

## Analysis Features

### Statistical Analysis
- **Z-score calculation**: Determines how far current readings deviate from historical averages
- **Variance analysis**: Measures the variability in sensor readings
- **Rate of change**: Analyzes how quickly environmental values are changing
- **Cross-correlation**: Examines relationships between different environmental sensors

### Data Management
- **Historical data storage**: Maintains environmental sensor reading history for analysis
- **Automatic cleanup**: Removes old data to prevent storage bloat
- **Data validation**: Ensures sensor readings are valid and within expected ranges

## Benefits

### Enhanced Accuracy
Environmental analysis can help detect occupancy patterns that motion sensors might miss, such as:
- Gradual temperature increases from body heat
- Humidity changes from breathing and activity
- Light level changes from device usage or window adjustments

### Reduced False Positives/Negatives
By analyzing multiple environmental factors together, the system can:
- Better distinguish between true occupancy and sensor anomalies
- Detect occupancy during periods of low motion (e.g., sleeping, reading)
- Provide more stable probability calculations

## Technical Details

### Storage
Environmental data is stored in JSON format in the Home Assistant configuration directory:
```
config/custom_components/area_occupancy/environmental_data/
```

### Update Frequency
Environmental analysis runs every 5 minutes by default, providing regular updates without overwhelming the system.

### Error Handling
The environmental analysis system includes robust error handling:
- Graceful degradation if environmental sensors become unavailable
- Fallback to main sensor analysis if environmental analysis fails
- Logging of issues for troubleshooting

## Troubleshooting

### No Environmental Analysis
If environmental analysis isn't working:
1. Check that environmental sensors are properly configured
2. Verify sensors are reporting valid numeric values
3. Check the Home Assistant logs for environmental analysis errors
4. Ensure the environmental weight is set above 0

### Inconsistent Results
If environmental analysis seems erratic:
1. Allow time for the system to collect sufficient historical data
2. Check that environmental sensors are stable and not malfunctioning
3. Verify sensors are actually in the area being monitored

## Future Enhancements

Planned improvements to environmental sensor analysis include:

- **Machine Learning Models**: Advanced ML models trained on your specific environmental patterns
- **Additional Sensor Types**: Support for CO2, sound level, and pressure sensors
- **Predictive Analysis**: Ability to predict occupancy changes based on environmental trends
- **Custom Thresholds**: User-configurable sensitivity settings for environmental analysis

## Related Documentation

- [Bayesian Probability Calculation](calculation.md) - Core probability calculation system
- [Configuration Guide](../getting-started/configuration.md) - General setup instructions
- [Troubleshooting](../getting-started/troubleshooting.md) - Common issues and solutions
