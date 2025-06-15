# Centralized Mock Infrastructure for Area Occupancy Detection Tests

This document describes the centralized mock infrastructure available in `tests/conftest.py` for consistent and efficient testing across the Area Occupancy Detection integration.

## Overview

The centralized mock system eliminates code duplication by providing reusable fixtures that can be injected into any test class or function. This ensures consistent behavior across all tests and makes maintenance easier.

## Core Fixtures

### Home Assistant Infrastructure

#### `mock_hass`

Comprehensive Home Assistant instance mock with:

- Event loop management with automatic setup/cleanup
- Config entries, states, and entity registry
- Storage system with version 9.1 support
- Event system and service registration
- Async method support

```python
def test_example(self, mock_hass: Mock) -> None:
    """Test using mock Home Assistant instance."""
    assert mock_hass.config.config_dir
    mock_hass.states.get.return_value = Mock(state="on")
```

#### `mock_config_entry`

Complete configuration entry with:

- All configuration options with defaults
- Version settings (9.1)
- Runtime data and lifecycle methods
- Entry ID: "test_entry_id"

```python
def test_example(self, mock_config_entry: Mock) -> None:
    """Test using mock config entry."""
    assert mock_config_entry.entry_id == "test_entry_id"
    assert mock_config_entry.data[CONF_NAME] == "Test Area"
```

### Coordinator Fixtures

#### `mock_coordinator`

Base coordinator with:

- Mock Home Assistant and config entry references
- Entity manager with coordinator reference
- Storage system and async methods
- Basic occupancy state (probability: 0.5, threshold: 0.5)

```python
def test_example(self, mock_coordinator: Mock) -> None:
    """Test using base coordinator."""
    assert mock_coordinator.probability == 0.5
    assert mock_coordinator.entity_manager is not None
```

#### `mock_coordinator_with_threshold`

Coordinator specialized for threshold/number entity tests:

- Threshold value: 0.6
- `async_update_threshold` method

```python
def test_threshold(self, mock_coordinator_with_threshold: Mock) -> None:
    """Test threshold functionality."""
    assert mock_coordinator_with_threshold.threshold == 0.6
    await mock_coordinator_with_threshold.async_update_threshold(0.8)
```

#### `mock_coordinator_with_sensors`

Coordinator with comprehensive sensor data:

- Prior: 0.35, Probability: 0.65, Decay: 0.8
- Entity manager with 4 mock entities (motion, light, media)
- Each entity has realistic probability and state data

```python
def test_sensors(self, mock_coordinator_with_sensors: Mock) -> None:
    """Test sensor functionality."""
    entities = mock_coordinator_with_sensors.entity_manager.entities
    assert len(entities) == 4
    assert "binary_sensor.motion1" in entities
```

### Entity Component Fixtures

#### `mock_entity_type`

Comprehensive entity type mock:

- Input type: MOTION
- Weight: 0.8, Prior: 0.35
- Active states: [STATE_ON]
- `is_active()` method returns True

#### `mock_prior`

Prior probability mock with:

- Prior: 0.35, prob_given_true: 0.8, prob_given_false: 0.1
- Timestamp and serialization support

#### `mock_decay`

Decay system mock with:

- Decay enabled, window: 300s, factor: 1.0
- Not currently decaying
- All decay control methods

#### `mock_comprehensive_entity`

Complete entity mock combining all components:

- Uses `mock_entity_type`, `mock_prior`, `mock_decay`
- Full method suite (update_probability, decay timers, etc.)
- Serialization support

```python
def test_entity(self, mock_comprehensive_entity: Mock) -> None:
    """Test entity functionality."""
    assert mock_comprehensive_entity.entity_id == "binary_sensor.test_motion"
    assert mock_comprehensive_entity.probability == 0.5
    mock_comprehensive_entity.update_probability()
```

#### `mock_comprehensive_entity_manager`

Entity manager with comprehensive entity:

- Contains one `mock_comprehensive_entity`
- All manager methods (add, remove, reset, etc.)
- State tracking and serialization

### Service Testing Fixtures

#### `mock_service_call`

Basic service call mock:

- Data: `{"entry_id": "test_entry_id"}`
- `return_response`: True

#### `mock_service_call_with_entity`

Service call with entity specification:

- Data: `{"entry_id": "test_entry_id", "entity_id": "binary_sensor.test_motion"}`
- `return_response`: True

```python
async def test_service(self, mock_hass: Mock, mock_service_call: Mock) -> None:
    """Test service functionality."""
    await some_service_function(mock_hass, mock_service_call)
    assert mock_service_call.data["entry_id"] == "test_entry_id"
```

### Utility Fixtures

#### `mock_entity_registry`

Entity registry with iteration support:

- Empty entities container by default
- Registry methods (async_get_entity_id, async_update_entity)

#### `mock_device_info`

Standard device info dictionary:

- Identifiers, name, manufacturer, model, version
- Ready for entity device_info properties

#### `mock_storage_manager_patches`

Common patches for StorageManager tests:

- Store initialization, async_load, async_save
- Event tracking patches

## Data Fixtures

#### `valid_entity_data`

Valid entity data structure for testing:

- Complete entity with all required fields
- Realistic probability and state values
- Proper timestamp formatting

#### `valid_storage_data`

Valid storage data with current format (v9.1):

- Proper nested structure: `data.instances.{entry_id}.entities`
- Contains one test entity with complete data

## Migration Guide

### From Local to Centralized Fixtures

**Before (Local Fixture):**

```python
class TestMyComponent:
    @pytest.fixture
    def mock_coordinator(self) -> Mock:
        coordinator = Mock()
        coordinator.threshold = 0.6
        coordinator.available = True
        return coordinator

    def test_something(self, mock_coordinator: Mock) -> None:
        # Test code
```

**After (Centralized Fixture):**

```python
class TestMyComponent:
    def test_something(self, mock_coordinator_with_threshold: Mock) -> None:
        # Test code - fixture automatically injected
```

### Customizing Centralized Fixtures

If you need to customize a centralized fixture:

```python
class TestMyComponent:
    @pytest.fixture
    def custom_coordinator(self, mock_coordinator: Mock) -> Mock:
        """Customize the base coordinator for specific tests."""
        mock_coordinator.special_attribute = "custom_value"
        return mock_coordinator

    def test_with_custom(self, custom_coordinator: Mock) -> None:
        assert custom_coordinator.special_attribute == "custom_value"
```

## Best Practices

### 1. Use Appropriate Fixture Level

- Use `mock_coordinator` for basic tests
- Use `mock_coordinator_with_threshold` for number entity tests
- Use `mock_coordinator_with_sensors` for sensor entity tests

### 2. Fixture Dependencies

Fixtures are designed to work together:

```python
# These fixtures automatically use each other
def test_comprehensive(
    self,
    mock_comprehensive_entity: Mock,  # Uses mock_entity_type, mock_prior, mock_decay
    mock_comprehensive_entity_manager: Mock  # Uses mock_coordinator, mock_comprehensive_entity
) -> None:
    # All components work together seamlessly
```

### 3. Extending Fixtures

Create specialized fixtures that build on centralized ones:

```python
@pytest.fixture
def wasp_coordinator(self, mock_coordinator: Mock) -> Mock:
    """Coordinator with wasp-specific configuration."""
    mock_coordinator.config.wasp_in_box.enabled = True
    return mock_coordinator
```

### 4. Parallel Tool Calls

When gathering information, use parallel tool calls for efficiency:

```python
# Good - parallel execution
@pytest.mark.asyncio
async def test_multiple_operations(self, mock_coordinator: Mock) -> None:
    results = await asyncio.gather(
        mock_coordinator.async_method1(),
        mock_coordinator.async_method2(),
        mock_coordinator.async_method3()
    )
```

## Fixture Dependency Chain

```
mock_hass
├── mock_config_entry
├── mock_entity_manager
│   └── mock_coordinator
│       ├── mock_coordinator_with_threshold
│       └── mock_coordinator_with_sensors
└── mock_entity_type
    ├── mock_prior
    ├── mock_decay
    └── mock_comprehensive_entity
        └── mock_comprehensive_entity_manager
```

## Future Enhancements

### Planned Additions

- Mock migration system fixtures
- Mock historical data fixtures
- Mock state mapping fixtures
- Performance testing fixtures

### Contributing New Fixtures

When adding new centralized fixtures:

1. Follow naming convention: `mock_{component}_{specialization}`
2. Include comprehensive docstring with usage example
3. Update this documentation
4. Ensure fixture works with existing dependency chain
5. Add to appropriate test file for validation

## Testing the Fixtures

To verify fixtures work correctly:

```bash
# Test specific fixture usage
python -m pytest tests/test_binary_sensor.py::TestOccupancy::test_initialization -v

# Test all centralized fixture usage
python -m pytest tests/ -k "mock_coordinator" -v
```

This centralized approach has reduced test code duplication by ~200+ lines while ensuring consistent mock behavior across the entire test suite.
