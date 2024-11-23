# Testing Guide for Room Occupancy Integration

## Local Testing Setup

### 1. Create Testing Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Install development dependencies
pip install -r requirements_dev.txt
```

### 2. Directory Structure for Testing

Create the following directory structure in your project:

```
room_occupancy/
├── custom_components/
│   └── room_occupancy/
│       ├── __init__.py
│       ├── manifest.json
│       └── ... (other component files)
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── ... (test files)
├── requirements.txt
└── requirements_dev.txt
```

### 3. Configure pytest

Create `pytest.ini` in the root directory:

```ini
[pytest]
testpaths = tests
norecursedirs = .git custom_components
asyncio_mode = auto
```

### 4. Running Tests Without Home Assistant

The `pytest-homeassistant-custom-component` package provides a mock Home Assistant environment for testing. Here's how to use it:

#### Basic Test Running

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_sensor.py

# Run with coverage report
pytest --cov=custom_components.room_occupancy
```

#### Test Configuration

In your `conftest.py`, add these fixtures for mock Home Assistant setup:

```python
import pytest
from unittest.mock import patch
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

@pytest.fixture
def hass(loop):
    """Fixture to provide a test instance of Home Assistant."""
    hass = HomeAssistant()
    loop.run_until_complete(async_setup_component(hass, "homeassistant", {}))
    return hass

@pytest.fixture
def mock_hass_config():
    """Fixture to mock Home Assistant configuration."""
    return {
        "homeassistant": {
            "name": "Test Home",
            "latitude": 0,
            "longitude": 0,
            "elevation": 0,
            "unit_system": "metric",
            "time_zone": "UTC",
        }
    }
```

### 5. Writing Tests with Mock Components

Example of testing with mock entities:

```python
async def test_sensor_behavior(hass):
    """Test sensor behavior with mock entities."""
    # Set up mock motion sensor
    hass.states.async_set("binary_sensor.motion", "off")
    
    # Configure integration
    entry = MockConfigEntry(
        domain="room_occupancy",
        data={
            "name": "Test Room",
            "motion_sensors": ["binary_sensor.motion"]
        }
    )
    entry.add_to_hass(hass)
    
    # Initialize integration
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    
    # Test sensor behavior
    hass.states.async_set("binary_sensor.motion", "on")
    await hass.async_block_till_done()
    
    state = hass.states.get("sensor.test_room_occupancy_probability")
    assert state is not None
    assert float(state.state) > 50  # Probability should be high with motion
```

### 6. Mocking External Dependencies

Example of mocking coordinator updates:

```python
from unittest.mock import patch

async def test_coordinator_update(hass):
    """Test coordinator update with mocked data."""
    with patch(
        "custom_components.room_occupancy.coordinator.RoomOccupancyCoordinator._async_update_data",
        return_value={"probability": 0.75}
    ):
        # Test code here
        pass
```

### 7. Testing Config Flow

Example of testing configuration flow:

```python
async def test_config_flow(hass):
    """Test the config flow."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    
    assert result["type"] == "form"
    assert result["errors"] == {}
    
    # Test form submission
    result2 = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Test Room",
            "motion_sensors": ["binary_sensor.motion"]
        }
    )
    
    assert result2["type"] == "create_entry"
```

## Common Testing Patterns

### 1. Testing State Changes

```python
async def test_state_changes(hass):
    """Test response to state changes."""
    # Initial setup
    await setup_integration(hass)
    
    # Change states
    hass.states.async_set("binary_sensor.motion", "on")
    await hass.async_block_till_done()
    
    # Assert expected behavior
    state = hass.states.get("sensor.room_occupancy_probability")
    assert state is not None
```

### 2. Testing Time-Based Behavior

```python
from freezegun import freeze_time

async def test_decay(hass):
    """Test time-based decay."""
    await setup_integration(hass)
    
    with freeze_time("2024-01-01 12:00:00"):
        # Initial state
        hass.states.async_set("binary_sensor.motion", "on")
        await hass.async_block_till_done()
        initial_state = hass.states.get("sensor.room_occupancy_probability")
        
    with freeze_time("2024-01-01 12:05:00"):
        # 5 minutes later
        await hass.async_block_till_done()
        later_state = hass.states.get("sensor.room_occupancy_probability")
        
        assert float(later_state.state) < float(initial_state.state)
```

### 3. Testing Error Conditions

```python
async def test_error_handling(hass):
    """Test error handling."""
    with patch(
        "custom_components.room_occupancy.coordinator.RoomOccupancyCoordinator._async_update_data",
        side_effect=Exception("Test error")
    ):
        # Test error handling code
        pass
```

## Tips for Testing

1. Use `async_block_till_done()` after state changes
2. Mock time-dependent functions using `freeze_time`
3. Test both success and failure paths
4. Use coverage reports to identify untested code
5. Test edge cases and error conditions
6. Mock external dependencies consistently

## Running Tests in CI

Add this GitHub Actions workflow for automated testing:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements_dev.txt
    - name: Run tests
      run: |
        pytest --cov=custom_components.room_occupancy
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```
