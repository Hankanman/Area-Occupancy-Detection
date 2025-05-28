---
applyTo: '**'
---
# Environmental Sensor Analysis

## Overview

Environmental sensors provide valuable occupancy indicators by detecting environmental changes caused by human presence. This feature extends the Bayesian probability calculations to include environmental factors that correlate with occupancy patterns.

## Supported Environmental Sensors

### Primary Environmental Sensors
- **Carbon Dioxide (CO2)**: Levels increase with human respiration
- **Temperature**: Changes from body heat and device usage
- **Humidity**: Affected by breathing and human activity
- **Luminance/Light**: Changes from lighting controls and natural light management
- **Sound Level (Decibels)**: Noise from human activity
- **Atmospheric Pressure**: Subtle changes from HVAC and human presence

### Secondary Environmental Indicators
- **Air Quality Index**: General air quality changes
- **VOC (Volatile Organic Compounds)**: Chemical signatures of human presence
- **Motion-triggered Temperature**: Temperature spikes from motion-activated devices

## Implementation Architecture

### Core Components

#### Environmental Analysis Module
```python
# custom_components/area_occupancy/environmental_analysis.py
```
- Environmental data collection and preprocessing
- Pattern recognition for occupancy correlations
- Fallback deterministic analysis

#### Environmental Data Storage
```python
# custom_components/area_occupancy/environmental_storage.py
```
- Historical environmental data management
- Training data preparation and caching
- Data cleanup and retention policies

#### ML Model Manager
- See [machine_learning.instructions.md](./.github/instructions/machine_learning.instructions.md) for universal ML requirements and architecture.

### Configuration Integration

#### Config Flow Extensions
Update [config_flow.py](mdc:custom_components/area_occupancy/config_flow.py) to include:
- Environmental sensor selection interface
- Analysis method configuration (ML/Deterministic/Hybrid, universally for all sensors)
- Training data requirements and thresholds
- Model performance settings

#### Constants and Types
Extend [const.py](mdc:custom_components/area_occupancy/const.py) and [types.py](mdc:custom_components/area_occupancy/types.py):
```python
# Environmental sensor types
CONF_ENVIRONMENTAL_SENSORS = "environmental_sensors"
CONF_CO2_SENSORS = "co2_sensors"
CONF_TEMPERATURE_SENSORS = "temperature_sensors"
CONF_HUMIDITY_SENSORS = "humidity_sensors"
CONF_LUMINANCE_SENSORS = "luminance_sensors"
CONF_SOUND_SENSORS = "sound_sensors"
CONF_PRESSURE_SENSORS = "pressure_sensors"

# Analysis configuration
CONF_ANALYSIS_METHOD = "analysis_method"
CONF_ML_ENABLED = "ml_enabled"
CONF_TRAINING_PERIOD_DAYS = "training_period_days"
CONF_MODEL_UPDATE_INTERVAL = "model_update_interval"
CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"

# Environmental data types
@dataclass
class EnvironmentalData:

@dataclass
class EnvironmentalAnalysisResult:
    model_version: str | None
```

## Data Collection and Preprocessing

### Historical Data Analysis
1. **Data Collection**:
   - Query historical environmental sensor data using Home Assistant's recorder
   - Correlate with primary occupancy sensor states
   - Extract features for training and analysis

2. **Feature Engineering**:
   - Time-based features (hour of day, day of week, season)
   - Rate of change calculations (temperature delta, CO2 rise rate)
   - Moving averages and trend analysis
   - Cross-sensor correlations

3. **Data Quality Validation**:
   - Sensor availability and reliability checks
   - Outlier detection and handling
   - Missing data imputation strategies
   - Temporal consistency validation

### Real-time Data Processing
```python
class EnvironmentalDataProcessor:
    """Process real-time environmental sensor data."""
    
    async def process_sensor_update(
        self, 
        sensor_id: str, 
        new_value: float, 
        timestamp: datetime
    ) -> EnvironmentalData:
        """Process new sensor reading."""
        
    async def calculate_features(
        self, 
        sensor_data: list[EnvironmentalData]
    ) -> dict[str, float]:
        """Calculate analysis features from sensor data."""
        
    async def validate_data_quality(
        self, 
        data: EnvironmentalData
    ) -> float:
        """Assess data quality and return confidence score."""
```

## Machine Learning Implementation

### Model Architecture
1. **Primary Model**: Gradient Boosting (XGBoost/LightGBM)
   - Handles mixed data types well
   - Provides feature importance
   - Robust to missing data
   - Good performance on tabular data

2. **Alternative Models**:
   - Random Forest for interpretability
   - Neural Networks for complex patterns
   - Time Series models for temporal dependencies

### Feature Engineering
1. **Temporal Features**:
   - Hour of day, day of week, month
   - Time since last occupancy
   - Duration of current state

2. **Sensor-Specific Features**:
   - **CO2**: Rate of change, absolute level, time to peak
   - **Temperature**: Delta from baseline, rate of change, thermal mass effects
   - **Humidity**: Absolute and relative changes, interaction with temperature
   - **Light**: Sudden changes, gradual changes, baseline variations
   - **Sound**: Peak levels, sustained levels, frequency of events
   - **Pressure**: Barometric trends, HVAC correlations

3. **Cross-Sensor Features**:
   - Temperature-humidity correlation
   - CO2-occupancy lag relationships
   - Light-activity correlations
   - Multi-sensor confidence scores

## Deterministic Analysis (Fallback)

### Rule-Based Analysis
For systems with insufficient data or computational constraints:

```python
class DeterministicEnvironmentalAnalysis:
    """Rule-based environmental analysis."""
    
    def __init__(self):
        self.rules = {
            "co2": {
                "baseline_threshold": 400,  # ppm
                "occupancy_increase": 200,  # ppm above baseline
                "response_time": 300,  # seconds
            },
            "temperature": {
                "human_heat_signature": 2.0,  # degrees increase
                "device_heat_delay": 600,  # seconds
                "thermal_mass_factor": 0.1,  # per minute
            },
            "luminance": {
                "sudden_change_threshold": 50,  # lux
                "occupancy_pattern": "increase",
                "time_of_day_factor": True,
            }
        }
    
    async def analyze_co2_pattern(
        self, 
        current_co2: float, 
        baseline_co2: float, 
        time_series: list[float]
    ) -> float:
        """Analyze CO2 patterns for occupancy indicators."""
        
    async def analyze_temperature_pattern(
        self, 
        current_temp: float, 
        baseline_temp: float, 
        rate_of_change: float
    ) -> float:
        """Analyze temperature patterns for occupancy indicators."""
```

### Threshold-Based Rules
1. **CO2 Analysis**:
   - Baseline establishment (400-450 ppm typically)
   - Occupancy detection threshold (600+ ppm)
   - Rate of change analysis (rapid vs. gradual increases)

2. **Temperature Analysis**:
   - Human heat signature detection (1-3°C increase)
   - Device usage correlation (appliance heat generation)
   - Thermal lag compensation

3. **Humidity Analysis**:
   - Breathing signature (gradual increase)
   - Activity correlation (showering, cooking)
   - Seasonal baseline adjustment

## Hybrid Analysis Approach

### Adaptive Model Selection
```python
class HybridEnvironmentalAnalysis:
    """Combines ML and deterministic approaches."""
    
    async def analyze_environmental_data(
        self, 
        sensor_data: dict[str, EnvironmentalData]
    ) -> EnvironmentalAnalysisResult:
        """Perform hybrid analysis using best available method."""
        
        # Assess data quality and availability
        data_quality = await self._assess_data_quality(sensor_data)
        
        # Choose analysis method based on conditions
        if self._ml_model_available() and data_quality > 0.8:
            return await self._ml_analysis(sensor_data)
        elif data_quality > 0.5:
            return await self._deterministic_analysis(sensor_data)
        else:
            return await self._fallback_analysis(sensor_data)
    
    async def _assess_data_quality(
        self, 
        sensor_data: dict[str, EnvironmentalData]
    ) -> float:
        """Assess overall data quality for analysis method selection."""
```

## Integration with Bayesian Calculations

### Probability Integration
Update [calculate_prob.py](mdc:custom_components/area_occupancy/calculate_prob.py):

```python
async def calculate_environmental_probability(
    environmental_result: EnvironmentalAnalysisResult,
    sensor_priors: dict[str, PriorData]
) -> float:
    """Integrate environmental analysis into Bayesian calculation."""
    
    # Weight environmental probability by confidence
    weighted_prob = (
        environmental_result.occupancy_probability * 
        environmental_result.confidence
    )
    
    # Combine with sensor-based probabilities using Bayesian inference
    return bayesian_update(weighted_prob, sensor_priors)
```

### Prior Calculation Updates
Extend [calculate_prior.py](mdc:custom_components/area_occupancy/calculate_prior.py):

```python
async def calculate_environmental_priors(
    environmental_sensors: list[str],
    primary_occupancy_sensor: str,
    history_days: int
) -> dict[str, PriorData]:
    """Calculate prior probabilities for environmental sensors."""
    
    # Analyze historical correlations
    # Generate sensor-specific priors
    # Handle seasonal and temporal variations
```

## Performance and Optimization

### Computational Efficiency
1. **Model Optimization**:
   - Model quantization for reduced memory usage
   - Feature selection to reduce computation
   - Caching of intermediate results
   - Asynchronous inference pipeline

2. **Data Management**:
   - Efficient data storage and retrieval
   - Data compression for historical storage
   - Intelligent data retention policies
   - Streaming data processing

### Memory Management
```python
class EnvironmentalDataManager:
    """Manage environmental data efficiently."""
    
    def __init__(self, max_memory_mb: int = 100):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.data_cache = {}
        self.model_cache = {}
    
    async def optimize_memory_usage(self) -> None:
        """Optimize memory usage by cleaning old data."""
        
    async def get_historical_data(
        self, 
        sensor_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> list[EnvironmentalData]:
        """Efficiently retrieve historical data."""
```

## Testing Requirements

### Unit Tests
- Environmental data processing accuracy
- ML model training and prediction
- Deterministic rule validation
- Data quality assessment
- Feature engineering correctness

### Integration Tests
- End-to-end environmental analysis pipeline
- Bayesian integration accuracy
- Performance under various data conditions
- Model update and versioning
- Hybrid method switching

### Performance Tests
- Memory usage under load
- Inference time benchmarks
- Training time optimization
- Data processing throughput
- Concurrent analysis handling

## Configuration and User Interface

### Config Flow Updates
1. **Environmental Sensor Selection**:
   - Multi-select interface for sensor types
   - Individual sensor entity selection
   - Sensor quality validation

2. **Analysis Method Configuration**:
   - ML vs. Deterministic vs. Hybrid selection
   - Training data requirements
   - Model performance thresholds

3. **Advanced Settings**:
   - Feature engineering parameters
   - Model update intervals
   - Memory usage limits
   - Data retention policies

### Service Definitions
Extend [services.yaml](mdc:custom_components/area_occupancy/services.yaml):

```yaml
train_environmental_model:
  description: Train or retrain the environmental analysis model
  fields:
    force_retrain:
      description: Force complete model retraining
      example: false

analyze_environmental_patterns:
  description: Analyze environmental sensor patterns for debugging
  fields:
    sensor_types:
      description: List of sensor types to analyze
      example: ["co2", "temperature"]

export_environmental_data:
  description: Export environmental analysis data for external analysis
  fields:
    start_date:
      description: Start date for data export
      example: "2025-01-01"
```

## Monitoring and Diagnostics

### Performance Metrics
1. **Model Performance**:
   - Prediction accuracy
   - False positive/negative rates
   - Confidence score distributions
   - Feature importance tracking

2. **System Performance**:
   - Inference latency
   - Memory usage trends
   - Data processing throughput
   - Error rates and recovery

### Diagnostic Tools
```python
class EnvironmentalDiagnostics:
    """Diagnostic tools for environmental analysis."""
    
    async def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        
    async def analyze_sensor_correlations(self) -> dict[str, float]:
        """Analyze correlations between environmental sensors."""
        
    async def validate_model_performance(self) -> dict[str, float]:
        """Validate current model performance metrics."""
```

## Migration and Backwards Compatibility

### Configuration Migration
Update [migrations.py](mdc:custom_components/area_occupancy/migrations.py):

```python
async def migrate_to_environmental_sensors(config_data: dict) -> dict:
    """Migrate configuration to support environmental sensors."""
    
    # Add environmental sensor defaults
    # Migrate existing sensor configurations
    # Set up initial analysis method preferences
```

### Data Migration
- Historical data format updates
- Model compatibility handling
- Feature schema evolution
- Graceful degradation for missing sensors

## Documentation Requirements

### Technical Documentation
- Environmental sensor correlation analysis methodology
- Machine learning model architecture and rationale
- Feature engineering documentation
- Performance optimization strategies

### User Documentation
- Environmental sensor setup guide
- Troubleshooting common issues
- Performance tuning recommendations
- Privacy and data handling information

### API Documentation
- Environmental analysis API reference
- Configuration options documentation
- Service call examples
- Integration patterns and best practices
