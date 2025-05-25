---
applyTo: '**'
---
# Area Occupancy Detection Integration

## Project Structure

The integration is organized in the `custom_components/area_occupancy` directory with the following key files:

### Core Integration Files
- [custom_components/area_occupancy/__init__.py](mdc:custom_components/area_occupancy/__init__.py): Integration setup and platform registration
- [custom_components/area_occupancy/binary_sensor.py](mdc:custom_components/area_occupancy/binary_sensor.py): Occupancy Status binary sensor
- [custom_components/area_occupancy/calculate_prob.py](mdc:custom_components/area_occupancy/calculate_prob.py): Bayesian probability calculations
- [custom_components/area_occupancy/calculate_prior.py](mdc:custom_components/area_occupancy/calculate_prior.py): Prior probability calculations
- [custom_components/area_occupancy/config_flow.py](mdc:custom_components/area_occupancy/config_flow.py): Configuration UI flow
- [custom_components/area_occupancy/const.py](mdc:custom_components/area_occupancy/const.py): Constants and defaults
- [custom_components/area_occupancy/coordinator.py](mdc:custom_components/area_occupancy/coordinator.py): Data update coordination
- [custom_components/area_occupancy/manifest.json](mdc:custom_components/area_occupancy/manifest.json): Integration metadata
- [custom_components/area_occupancy/probabilities.py](mdc:custom_components/area_occupancy/probabilities.py): Default probability values
- [custom_components/area_occupancy/sensor.py](mdc:custom_components/area_occupancy/sensor.py): Probability and Prior sensors
- [custom_components/area_occupancy/number.py](mdc:custom_components/area_occupancy/number.py): Number entities for configuration
- [custom_components/area_occupancy/service.py](mdc:custom_components/area_occupancy/service.py): Integration services
- [custom_components/area_occupancy/services.yaml](mdc:custom_components/area_occupancy/services.yaml): Service definitions
- [custom_components/area_occupancy/types.py](mdc:custom_components/area_occupancy/types.py): Type definitions

### Additional Components
- [custom_components/area_occupancy/decay_handler.py](mdc:custom_components/area_occupancy/decay_handler.py): Probability decay functionality
- [custom_components/area_occupancy/exceptions.py](mdc:custom_components/area_occupancy/exceptions.py): Custom exception definitions
- [custom_components/area_occupancy/migrations.py](mdc:custom_components/area_occupancy/migrations.py): Configuration migration handling
- [custom_components/area_occupancy/state_mapping.py](mdc:custom_components/area_occupancy/state_mapping.py): Entity state mapping utilities
- [custom_components/area_occupancy/storage.py](mdc:custom_components/area_occupancy/storage.py): Data persistence and storage

### Environmental Analysis Components (In Development)
- [custom_components/area_occupancy/environmental_analysis.py](mdc:custom_components/area_occupancy/environmental_analysis.py): Environmental sensor data analysis
- [custom_components/area_occupancy/environmental_storage.py](mdc:custom_components/area_occupancy/environmental_storage.py): Environmental data persistence
- [custom_components/area_occupancy/ml_models.py](mdc:custom_components/area_occupancy/ml_models.py): Machine learning model management
- [custom_components/area_occupancy/strings.json](mdc:custom_components/area_occupancy/strings.json): UI strings and translations
- [custom_components/area_occupancy/virtual_sensor/](mdc:custom_components/area_occupancy/virtual_sensor/): Virtual sensor implementations
- [custom_components/area_occupancy/translations/](mdc:custom_components/area_occupancy/translations/): Translation files

## Development Guidelines

### Code Organization
- All constants must be defined in `const.py`
- All custom types must be defined in `types.py`
- All custom exceptions must be defined in `exceptions.py`
- Configuration flow logic must be in `config_flow.py`
- Configuration migrations must be in `migrations.py`
- Service definitions must be in `services.yaml` with implementation in `service.py`
- Sensor entities must be defined in `sensor.py`
- Binary sensor entities must be defined in `binary_sensor.py`
- Number entities must be defined in `number.py`
- Core Bayesian calculation logic must be in `calculate_prob.py`
- Prior probability calculation logic must be in `calculate_prior.py`
- Default probability values must be defined in `probabilities.py`
- Coordinator logic must be in `coordinator.py`
- Probability decay handling must be in `decay_handler.py`
- Data persistence logic must be in `storage.py`
- Entity state mapping utilities must be in `state_mapping.py`
- Environmental sensor analysis must be in `environmental_analysis.py`
- Environmental data storage must be in `environmental_storage.py`
- Machine learning models must be in `ml_models.py`
- Virtual sensor implementations must be in `virtual_sensor/` directory
- UI strings and translations must be in `strings.json` and `translations/` directory
- Probability decay handling must be in `decay_handler.py`
- Data persistence logic must be in `storage.py`
- Entity state mapping utilities must be in `state_mapping.py`
- Virtual sensor implementations must be in `virtual_sensor/` directory
- UI strings and translations must be in `strings.json` and `translations/` directory

### Coding Standards
- Use snake_case for all file names, variables, and functions
- Follow PEP8 standards strictly
- Use specific exceptions (not general Exception)
- Include stack traces in debug logs with exc_info=True
- Use f-strings for log messages
- Run ruff check and format before commits

### Testing Requirements
- Write unit tests in `tests/` directory using pytest
- Cover core logic: coordinator updates, calculations, config flow, entity states
- Achieve minimum 90% test coverage
- Test edge cases: sensor unavailability, invalid configs, zero history
- Use Home Assistant test harness (hass, mock_config_entry, etc.)

### Integration Features
- Occupancy Probability Sensor: Shows Bayesian probability as percentage
- Occupancy Status Sensor: Binary sensor based on probability threshold
- Occupancy Prior Sensor: Shows overall prior probability
- Individual Prior Sensors: Show priors for each input entity
- Probability Threshold Number: Configurable threshold for binary sensor
- Decay Rate Number: Configurable probability decay rate
- Dynamic prior calculation based on historical data
- Probability decay functionality for unoccupied states
- Data persistence and storage across restarts
- Configuration migration support
- Virtual sensor implementations (e.g., Wasp in Box)
- Environmental sensor analysis (CO2, temperature, humidity, light, sound, pressure)
- Machine learning and deterministic analysis methods
- Hybrid analysis approach for optimal performance
- Fallback to default probabilities when history insufficient
- Data persistence and storage across restarts
- Configuration migration support
- Virtual sensor implementations (e.g., Wasp in Box)
- Fallback to default probabilities when history insufficient

### Data Handling
- Use DataUpdateCoordinator for state management
- Calculate priors using historical data from recorder
- Handle missing data gracefully with defaults
- Efficient state listeners to minimize system load
- Store calculated priors in coordinator data

### Configuration
- Allow selection of input entities (motion, devices, etc.)
- Configure probability threshold for binary sensor
- Set history period for prior calculations
- Select primary occupancy indicator
- Provide options for reconfiguration

### Documentation
- Include docstrings for all public interfaces
- Document Bayesian calculation methodology
- Provide clear error messages and logging
- Maintain README with setup and usage instructions