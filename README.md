# Room Occupancy Integration - Developer Guide

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Installation for Development](#installation-for-development)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing Guidelines](#contributing-guidelines)
- [Debugging](#debugging)
- [Configuration Reference](#configuration-reference)
- [Common Issues](#common-issues)

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Home Assistant development environment
- Git
- Visual Studio Code (recommended)
- Docker (optional, for container-based development)

### Setting Up Local Development Environment

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows
```

2. Install development dependencies:

```bash
pip install -r requirements_dev.txt
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

## Installation for Development

### Method 1: Direct Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ha-room-occupancy.git
```

2. Create symbolic link:

```bash
# Linux/macOS
ln -s /path/to/ha-room-occupancy/custom_components/room_occupancy /path/to/homeassistant/config/custom_components/room_occupancy

# Windows (as administrator)
mklink /D C:\path\to\homeassistant\config\custom_components\room_occupancy C:\path\to\ha-room-occupancy\custom_components\room_occupancy
```

3. Restart Home Assistant

### Method 2: Using HACS for Development

1. Add repository to HACS as custom repository:

   - Category: Integration
   - URL: Your repository URL
   - Branch: dev (or your development branch)

2. Install through HACS interface

3. Restart Home Assistant

## Project Structure

```bash
custom_components/room_occupancy/
├── __init__.py           # Integration initialization
├── manifest.json         # Integration metadata
├── const.py             # Constants and configuration
├── config_flow.py       # Configuration UI
├── coordinator.py       # Data update coordinator
├── probability.py       # Bayesian calculations
├── sensor.py           # Probability sensor
├── binary_sensor.py    # Occupancy sensor
├── strings.json        # String resources
└── translations/       # Localization
    └── en.json        # English translations
```

## Testing

### Setting Up Test Environment

1. Install test dependencies:

```bash
pip install -r requirements_test.txt
```

2. Install pytest-homeassistant-custom-component:

```bash
pip install pytest-homeassistant-custom-component
```

### Running Tests

1. Run all tests:

```bash
pytest tests/
```

2. Run specific test file:

```bash
pytest tests/test_sensor.py
```

3. Run with coverage:

```bash
pytest tests/ --cov=custom_components.room_occupancy
```

### Test Structure

```bash
tests/
├── conftest.py                  # Test fixtures
├── test_init.py                # Integration tests
├── test_config_flow.py         # Configuration tests
├── test_coordinator.py         # Coordinator tests
├── test_probability.py         # Probability calculation tests
├── test_sensor.py             # Probability sensor tests
└── test_binary_sensor.py      # Binary sensor tests
```

### Writing Tests

Example test case:

```python
async def test_sensor_probability_calculation(hass):
    """Test probability calculation with multiple sensors."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            "name": "Test Room",
            "motion_sensors": ["binary_sensor.motion"],
            "threshold": 0.5,
        },
    )

    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()

    # Test sensor states and calculations
    hass.states.async_set("binary_sensor.motion", "on")
    await hass.async_block_till_done()

    state = hass.states.get("sensor.room_occupancy_probability")
    assert state
    assert float(state.state) > 90.0  # High probability when motion detected
```

## Contributing Guidelines

### Code Style

- Follow Home Assistant's code style guidelines
- Use black for code formatting
- Use isort for import sorting
- Maintain pylint score above 9/10

### Pull Request Process

1. Create feature branch from dev
2. Write tests for new functionality
3. Update documentation
4. Submit PR against dev branch
5. Ensure all checks pass
6. Request review

## Debugging

### Enable Debug Logging

Add to configuration.yaml:

```yaml
logger:
  default: info
  logs:
    custom_components.room_occupancy: debug
```

### Common Debug Points

1. Probability Calculation:

```python
_LOGGER.debug(
    "Calculating probability - sensors: %s, values: %s",
    sensor_probabilities.keys(),
    sensor_probabilities.values()
)
```

2. Sensor Updates:

```python
_LOGGER.debug(
    "Sensor update - entity: %s, state: %s",
    entity_id,
    new_state.state
)
```

### Using Remote Debugger

1. Install debugpy:

```bash
pip install debugpy
```

2. Add breakpoint in code:

```python
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

3. Connect using VS Code debug configuration

## Configuration Reference

### Configuration Options

| Option              | Type   | Default  | Description            |
| ------------------- | ------ | -------- | ---------------------- |
| name                | string | Required | Room name              |
| motion_sensors      | list   | Required | Motion sensor entities |
| illuminance_sensors | list   | Optional | Light level sensors    |
| humidity_sensors    | list   | Optional | Humidity sensors       |
| temperature_sensors | list   | Optional | Temperature sensors    |
| device_states       | list   | Optional | Device state entities  |
| threshold           | float  | 0.5      | Occupancy threshold    |
| history_period      | int    | 7        | Days of history        |
| decay_enabled       | bool   | true     | Enable sensor decay    |
| decay_window        | int    | 600      | Decay window (seconds) |
| decay_type          | string | "linear" | Decay calculation type |

### Entity Attributes

```python
attributes = {
    "probability": 0.85,          # Current probability
    "prior_probability": 0.5,     # Prior probability
    "active_triggers": ["sensor.motion_1"],  # Active sensors
    "sensor_probabilities": {     # Individual probabilities
        "sensor.motion_1": 0.95,
        "sensor.illuminance_1": 0.7
    },
    "decay_status": {            # Decay values
        "sensor.motion_1": 0.8
    },
    "confidence_score": 0.9,     # Calculation confidence
    "sensor_availability": {     # Sensor status
        "sensor.motion_1": true
    }
}
```

## Common Issues

### Installation Issues

1. **Integration Not Appearing**

   - Check custom_components folder structure
   - Verify manifest.json contents
   - Clear browser cache
   - Restart Home Assistant

2. **Configuration Fails**
   - Verify sensor entity IDs exist
   - Check sensor permissions
   - Review Home Assistant logs

### Runtime Issues

1. **High CPU Usage**

   - Increase update interval
   - Reduce number of sensors
   - Check sensor update frequency

2. **Incorrect Probability**
   - Verify sensor states
   - Check decay settings
   - Review probability calculations
   - Enable debug logging

### Troubleshooting Steps

1. Enable debug logging
2. Check sensor states
3. Verify configuration
4. Review system resources
5. Check entity availability
6. Analyze probability calculations
7. Test with minimal configuration

---

## Release Process

1. Version Bump

   - Update manifest.json version
   - Update CHANGELOG.md
   - Create release tag

2. Testing

   - Run full test suite
   - Perform manual testing
   - Test upgrade path

3. Documentation

   - Update README.md
   - Update integration documentation
   - Update example configurations

4. Release
   - Create GitHub release
   - Update HACS repository
   - Notify users of update

## Performance Optimization

### Memory Usage

- Limit historical data storage
- Clean up unused state data
- Optimize data structures

### CPU Usage

- Implement caching where appropriate
- Optimize calculation frequency
- Use efficient algorithms

### Network Usage

- Batch sensor updates
- Implement rate limiting
- Optimize update intervals

## Security Considerations

1. Data Handling

   - Sanitize user inputs
   - Validate configuration data
   - Handle sensitive data appropriately

2. Integration Security

   - Verify sensor permissions
   - Validate entity access
   - Handle errors securely

3. Best Practices
   - Follow OWASP guidelines
   - Implement proper error handling
   - Use secure default values
