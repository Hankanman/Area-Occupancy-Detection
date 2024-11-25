# Area Occupancy Integration - Developer Guide

This document contains technical information for developers who want to contribute to or modify the Area Occupancy Detection integration.

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Home Assistant development environment
- Git
- Visual Studio Code (recommended)
- Docker (optional, for container-based development)

### Local Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate   # Windows
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.dev.txt
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Project Structure

```
custom_components/area_occupancy/
├── __init__.py           # Integration initialization
├── manifest.json         # Integration metadata
├── const.py             # Constants and configuration
├── config_flow.py       # Configuration UI
├── coordinator.py       # Data update coordinator
├── calculations.py      # Bayesian calculations
├── base.py             # Base entity classes
├── sensor.py           # Probability sensor
├── binary_sensor.py    # Occupancy sensor
├── strings.json        # String resources
└── translations/       # Localization
    └── en.json        # English translations
```

## Core Components

### Data Flow

1. `coordinator.py`: Manages sensor updates and scheduling
2. `calculations.py`: Handles Bayesian probability calculations
3. `base.py`: Provides shared entity functionality
4. `sensor.py`/`binary_sensor.py`: Entity implementations

### Key Classes

- `AreaOccupancyCoordinator`: Manages data updates and sensor state tracking
- `ProbabilityCalculator`: Implements Bayesian probability calculations
- `AreaOccupancySensorBase`: Base class for sensor entities
- `AreaOccupancyConfigFlow`: Handles UI configuration

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sensor.py

# Run with coverage
pytest --cov=custom_components.area_occupancy
```

### Test Structure

```
tests/
├── conftest.py           # Test fixtures
├── test_init.py         # Integration tests
├── test_config_flow.py  # Configuration tests
├── test_coordinator.py  # Coordinator tests
└── test_sensor.py      # Entity tests
```

## Contributing Guidelines

### Code Style

- Follow [Home Assistant's style guide](https://developers.home-assistant.io/docs/development_guidelines)
- Use `black` for code formatting
- Use `isort` for import sorting
- Maintain pylint score above 9/10

### Pull Request Process

1. Fork the repository
2. Create feature branch from `dev`
3. Write tests for new functionality
4. Update documentation
5. Submit PR against `dev` branch
6. Ensure all checks pass

## Debugging

### Enable Debug Logging

```yaml
logger:
  default: info
  logs:
    custom_components.area_occupancy: debug
```

### Development Container

1. Open in VS Code with Remote Containers extension
2. Select "Reopen in Container"
3. Use provided debug configurations

## Technical Documentation

### Probability Calculation

The integration uses Bayesian probability with:
- Prior probability based on historical data
- Likelihood calculations for each sensor type
- Weighted sensor contributions
- Time-based decay of sensor readings

### State Management

- Coordinator manages sensor state updates
- Historical data stored in rolling windows
- Decay implemented using configurable time windows
- Sensor availability tracking and fallback

### Configuration System

- UI-based configuration using config flows
- Dynamic validation of sensor entities
- Runtime configuration updates
- Options flow for post-install modification

## Performance Considerations

### Memory Usage

- Limited historical data storage
- Efficient state tracking
- Cleanup of unused data

### CPU Usage

- Optimized calculation frequency
- Caching of intermediate results
- Batched sensor updates

## Release Process

1. Version Management
   - Update `manifest.json` version
   - Update `CHANGELOG.md`
   - Create release tag

2. Testing
   - Run full test suite
   - Perform manual testing
   - Test upgrade path

3. Documentation
   - Update documentation
   - Update example configurations

4. Release
   - Create GitHub release
   - Update HACS repository

## Security Notes

- Input validation on all configuration
- Sensor state verification
- Proper error handling
- Secure default values

## Additional Resources

- [Home Assistant Development](https://developers.home-assistant.io/)
- [Integration Reference](https://developers.home-assistant.io/docs/creating_integration_manifest)
- [Testing Documentation](https://developers.home-assistant.io/docs/development_testing)
- [Code Review Guidelines](https://developers.home-assistant.io/docs/development_code_review)
