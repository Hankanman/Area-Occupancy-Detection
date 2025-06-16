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

## Refactored Test Files

### `test_number.py` Refactoring

**Initial Issues**:
- Local fixtures duplicating centralized functionality (`mock_coordinator`, `mock_hass`, `mock_config_entry`)
- Linter error: Tests accessing non-existent `_entry_id` attribute
- Test failures: 13 failed, 8 passed

**Key Implementation Misunderstandings Discovered**:
- **Critical Issue**: Tests expected wrong parameter format for `async_update_threshold`:
  - Tests expected: decimal values (0.75)
  - Actual implementation expects: percentage values (75.0)
  - Coordinator documentation confirms: "value: The new threshold value as a percentage (1-99)"
- **Other Implementation Gaps**:
  - Tests expected `entity.mode == "slider"` but actual uses `NumberMode.BOX`
  - Tests expected `entity.icon == "mdi:percent"` but entity has no icon property
  - Error message patterns didn't match actual implementation
  - Device info hardcoded instead of using coordinator's device_info
  - Available property uses `coordinator.last_update_success`, not `coordinator.available`

**Code Investigation Results**:
- **Number Entity Implementation**:
  - `Threshold` entity inherits from `CoordinatorEntity` and `NumberEntity`
  - `unique_id` format: `f"{entry_id}_{NAME_THRESHOLD_NUMBER.lower().replace(' ', '_')}"`
  - No `_entry_id` attribute stored
  - No icon property defined
- **Coordinator async_update_threshold Method**:
  - Expects percentage values (1-99)
  - Converts internally: `value / 100.0`
  - Updates config with decimal value

**Changes Made**:
- **Removed Local Fixtures**: Eliminated all local `@pytest.fixture` definitions in all test classes
- **Fixed Implementation Expectations**:
  - Updated all `async_update_threshold` calls to expect percentage values (75.0 instead of 0.75)
  - Fixed `entity.mode` assertion to `NumberMode.BOX`
  - Removed `test_icon_property` test since entity has no icon
  - Fixed `unique_id` expectations: `"test_entry_occupancy_threshold"`
  - Updated error message regex patterns to match actual implementation
  - Fixed entity category to use `EntityCategory.CONFIG`
  - Fixed available property test to use `coordinator.last_update_success`
- **Updated Test Logic**:
  - Fixed attribute access errors (removed `_entry_id` references)
  - Updated device_info test to use coordinator's device_info property
  - Fixed available property test expectations
  - Enhanced precision tests for percentage-to-percentage conversion
  - Updated all integration test workflows
  - Fixed runtime_data setup in async_setup_entry tests

**Number Entity Testing Patterns**:
```python
# Number entity test with centralized fixtures
class TestThreshold:
    def test_threshold_functionality(self, mock_coordinator_with_threshold: Mock) -> None:
        """Test threshold number entity."""
        entity = Threshold(mock_coordinator_with_threshold, "test_entry")

        # Available property uses coordinator.last_update_success
        mock_coordinator_with_threshold.last_update_success = True
        assert entity.available is True

        # Native value converts coordinator decimal to percentage
        mock_coordinator_with_threshold.threshold = 0.75
        assert entity.native_value == 75.0

# Async setup entry test pattern
async def test_async_setup_entry_success(
    self, mock_hass: Mock, mock_config_entry: Mock
) -> None:
    """Test successful setup entry."""
    mock_async_add_entities = Mock()

    # Must set runtime_data for coordinator
    mock_coordinator = Mock()
    mock_coordinator.device_info = {"test": "device_info"}
    mock_config_entry.runtime_data = mock_coordinator

    await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

    # Verify entity creation
    entities = mock_async_add_entities.call_args[0][0]
    assert len(entities) == 1
    assert isinstance(entities[0], Threshold)

# Percentage value testing
async def test_percentage_values(self, mock_coordinator_with_threshold: Mock) -> None:
    """Test percentage value handling."""
    entity = Threshold(mock_coordinator_with_threshold, "test_entry")

    # Coordinator expects percentage values (not decimals)
    await entity.async_set_native_value(75.0)
    mock_coordinator_with_threshold.async_update_threshold.assert_called_with(75.0)
```

**Results**: 13 failed tests → 20 passed (systematic fixes applied to all identified issues)

### `test_sensor.py` Refactoring

**Initial Issues**:
- Local fixtures duplicating centralized functionality (`mock_coordinator`, `mock_hass`, `mock_config_entry`)
- Implementation misunderstandings about sensor properties and behavior
- Test failures: 29 failed, 5 passed

**Key Implementation Misunderstandings Discovered**:
- **Unique ID Formats**: Tests expected wrong formats:
  - Tests expected: `"test_entry_priors"` vs actual: `"test_entry_prior_probability"`
  - Tests expected: `"test_entry_probability"` vs actual: `"test_entry_occupancy_probability"`
  - Tests expected: `"test_entry_decay"` vs actual: `"test_entry_decay_status"`
- **Entity Properties Missing**:
  - Tests expected `icon` properties that don't exist on sensor classes
  - Tests expected `_entry_id` attribute but base class doesn't store it
  - Tests expected `_attr_entity_registry_enabled_default` to be set initially but base class only sets it via method
- **Available Property**: Uses `coordinator.last_update_success`, not `coordinator.available`
- **Entity Manager Interface**: Tests expected different interface than actual:
  - Actual: `coordinator.entities.entities` (dict)
  - Actual: `coordinator.entities.active_entities` / `coordinator.entities.inactive_entities` (lists)
- **Sensor Value Calculations**:
  - DecaySensor returns `(1 - coordinator.decay) * 100`, not `coordinator.decay * 100`
  - Sensors don't check availability in `native_value` - they always return calculated values
- **Extra State Attributes**: Actual implementation structure differs from test expectations

**Code Investigation Results**:
- **Sensor Base Class**:
  - No `_entry_id` storage, just used for subclass unique_id generation
  - No default `_attr_entity_registry_enabled_default` - set via `set_enabled_default` method
  - Uses `coordinator.device_info` for device info
- **Sensor Name Patterns**:
  - PriorsSensor: "Prior Probability"
  - ProbabilitySensor: "Occupancy Probability"
  - EntitiesSensor: "Entities"
  - DecaySensor: "Decay Status"
- **Entity Categories**: Diagnostic sensors use `EntityCategory.DIAGNOSTIC` for enabled defaults

**Changes Made**:
- **Removed Local Fixtures**: Eliminated all local `@pytest.fixture` definitions across all test classes
- **Fixed Implementation Expectations**:
  - Updated unique_id assertions to match actual formats
  - Removed icon property tests where sensors don't have icons
  - Fixed available property tests to use `coordinator.last_update_success`
  - Updated entity manager interface expectations
  - Fixed DecaySensor calculation expectations: `(1 - decay) * 100`
  - Updated entity category expectations to use `EntityCategory.DIAGNOSTIC`
- **Updated Test Logic**:
  - Fixed attribute access errors (removed `_entry_id` and initial enabled default expectations)
  - Updated entity manager mock structure to match coordinator interface
  - Enhanced extra_state_attributes tests to match actual implementation
  - Fixed runtime_data setup in async_setup_entry tests
  - Updated sensor availability behavior expectations

**Sensor Testing Patterns**:
```python
# Sensor test with centralized fixtures
class TestPriorsSensor:
    def test_sensor_functionality(self, mock_coordinator: Mock) -> None:
        """Test sensor with centralized coordinator."""
        sensor = PriorsSensor(mock_coordinator, "test_entry")

        # Available property uses coordinator.last_update_success
        mock_coordinator.last_update_success = True
        assert sensor.available is True

        # Actual unique_id format from implementation
        assert sensor.unique_id == "test_entry_prior_probability"
        assert sensor.name == "Prior Probability"

        # Native value converts coordinator decimal to percentage
        mock_coordinator.prior = 0.35
        assert sensor.native_value == 35.0

# Entities sensor test with proper entity manager interface
def test_entities_sensor(self, mock_coordinator_with_sensors: Mock) -> None:
    """Test entities sensor with proper coordinator structure."""
    sensor = EntitiesSensor(mock_coordinator_with_sensors, "test_entry")

    # Uses coordinator.entities.entities dict
    assert sensor.native_value == 4  # From mock_coordinator_with_sensors

    # Extra state attributes require coordinator.data and entities lists
    mock_coordinator_with_sensors.data = {"test": "data"}
    mock_coordinator_with_sensors.entities.active_entities = []
    mock_coordinator_with_sensors.entities.inactive_entities = []
    attributes = sensor.extra_state_attributes
    assert "active" in attributes

# Decay sensor with corrected calculation
def test_decay_sensor(self, mock_coordinator: Mock) -> None:
    """Test decay sensor calculation."""
    sensor = DecaySensor(mock_coordinator, "test_entry")

    # DecaySensor returns (1 - coordinator.decay) * 100
    mock_coordinator.decay = 0.85
    assert sensor.native_value == 15.0  # (1 - 0.85) * 100

# Async setup entry test pattern
async def test_async_setup_entry_success(
    self, mock_hass: Mock, mock_config_entry: Mock
) -> None:
    """Test successful setup entry."""
    mock_async_add_entities = Mock()

    # Must set runtime_data for coordinator
    mock_coordinator = Mock()
    mock_coordinator.device_info = {"test": "device_info"}
    mock_config_entry.runtime_data = mock_coordinator

    await async_setup_entry(mock_hass, mock_config_entry, mock_async_add_entities)

    # Verify all 4 sensor entities created
    entities = mock_async_add_entities.call_args[0][0]
    assert len(entities) == 4
    entity_types = [type(entity).__name__ for entity in entities]
    assert "PriorsSensor" in entity_types
    assert "ProbabilitySensor" in entity_types
    assert "EntitiesSensor" in entity_types
    assert "DecaySensor" in entity_types
```

**Results**: 29 failed tests → 34 passed (comprehensive implementation understanding and systematic fixes applied)

### `test_migrations.py` Refactoring

**Initial Issues**:
- Local `mock_hass` fixtures in multiple test classes duplicating centralized functionality
- Local `mock_config_entry` fixtures not using centralized versions
- Storage manager mocking issues causing `AttributeError: 'dict' object has no attribute 'async_fetch'`
- Incorrect test expectations for future version behavior and error handling

**Changes Made**:
- **Removed Local Fixtures**: Eliminated all local `@pytest.fixture` definitions for `mock_hass` and basic `mock_config_entry`
- **Updated Version Logic**: Fixed version constants imports (`CONF_VERSION`, `CONF_VERSION_MINOR`) for proper test setup
- **Fixed Storage Mocking**: Added proper `Store` class patches to avoid storage manager dependency issues:
  ```python
  with (
      patch("homeassistant.helpers.storage.Store.async_load", new_callable=AsyncMock, return_value=None),
      patch("homeassistant.helpers.storage.Store.async_save", new_callable=AsyncMock),
  ):
  ```
- **Corrected Test Expectations**:
  - Future version tests now expect `True` (treated as already current)
  - Migration error tests properly expect exceptions to propagate
- **Enhanced Config Entry Fixtures**: Created specialized fixtures (`mock_config_entry_v1_0`, `mock_config_entry_current`) that extend centralized base
- **Improved Mock Validation**: Added proper assertions for `config_entries.async_update_entry` calls

**Migration-Specific Patterns**:
```python
# Version-specific config entry
@pytest.fixture
def mock_config_entry_v1_0(self, mock_config_entry: Mock) -> Mock:
    """Create a mock config entry at version 1.0."""
    entry = Mock(spec=ConfigEntry)
    entry.version = 1
    entry.minor_version = 0
    entry.entry_id = mock_config_entry.entry_id
    entry.data = {CONF_MOTION_SENSORS: ["binary_sensor.motion1"]}
    entry.options = {}
    return entry

# Storage migration test pattern
async def test_storage_migration(self, mock_hass: Mock) -> None:
    """Test storage migration functionality."""
    with (
        patch("homeassistant.helpers.storage.Store.async_load",
              new_callable=AsyncMock, return_value={"data": "test"}),
        patch("homeassistant.helpers.storage.Store.async_save",
              new_callable=AsyncMock),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.unlink") as mock_unlink,
    ):
        await async_migrate_storage(mock_hass, "test_entry_id")
        mock_unlink.assert_called_once()
```

**Results**: 23 tests → 23 passed (fixed through multiple iterations)

### `test_service.py` Refactoring

**Initial Issues**:
- Local fixtures duplicating centralized functionality (`mock_call`, `mock_hass`, etc.)
- Implementation misunderstandings about service interfaces and error handling
- Test failures: 31 failed, 0 passed

**Key Implementation Misunderstandings Discovered**:
- **Service Coordinator Access**: Tests expected `hass.data[DOMAIN][entry_id]` access pattern but actual implementation uses:
  - `hass.config_entries.async_entries(DOMAIN)` to find config entries
  - `entry.runtime_data` to get coordinator from config entry
- **Error Types**: Tests expected `ServiceValidationError` but actual implementation raises `HomeAssistantError`
- **Missing Parameters**: Tests expected specific error messages for missing parameters, but actual implementation raises `KeyError` for missing dictionary keys
- **Return Value Structures**: Tests expected wrong return formats:
  - `_get_entity_metrics` returns `{"metrics": metrics}` not `{"entities": entities}`
  - `_get_problematic_entities` returns `{"problems": problems}` not `{"issues": issues}`
  - `_get_area_status` returns `{"area_status": status}` not flat status dict
  - `_get_entity_details` returns `{"entity_details": details}` not `{"entity": entity}`
  - `_get_entity_type_learned_data` returns `{"entity_types": learned_data}` not flat dict
- **Service Entity Interface**: Tests expected `entity_manager` but actual uses `entities` directly on coordinator
- **Entity Methods**: Tests expected `force_update()` but actual uses `async_update()`

**Code Investigation Results**:
- **_get_coordinator Function**:
  - Iterates through `hass.config_entries.async_entries(DOMAIN)`
  - Returns `entry.runtime_data` for matching `entry_id`
  - Raises `HomeAssistantError` for not found, not `ServiceValidationError`
- **Service Parameter Handling**:
  - Uses `call.data["entry_id"]` directly (raises `KeyError` if missing)
  - Optional parameters use `call.data.get("param", default)`
  - No explicit validation for required parameters
- **Entity Manager Interface**:
  - Services use `coordinator.entities.entities` (dict of entities)
  - Services use `coordinator.entities.get_entity(entity_id)` for single entity access
  - Entity updates use `entity.async_update()` not `force_update()`
- **Return Value Patterns**:
  - All services return structured dictionaries with descriptive root keys
  - Error handling wraps exceptions in `HomeAssistantError` with descriptive messages
  - Services that support entity lists handle `ValueError` from `get_entity()` gracefully

**Changes Made**:
- **Removed Local Fixtures**: Eliminated all local `@pytest.fixture` definitions across all test classes
- **Fixed Service Access Pattern**:
  - Updated coordinator retrieval to use `mock_hass.config_entries.async_entries.return_value = [mock_config_entry]`
  - Set `mock_config_entry.runtime_data = mock_coordinator` instead of `hass.data[DOMAIN]`
- **Updated Error Expectations**:
  - Changed `ServiceValidationError` to `HomeAssistantError` for coordinator errors
  - Changed parameter validation errors to expect `KeyError` for missing keys
- **Fixed Return Value Expectations**:
  - Updated all assertions to match actual return structure (e.g., `result["metrics"]` not `result["entities"]`)
  - Fixed entity metrics tests to expect summary counts, not individual entity data
  - Updated problematic entities tests to expect `{"problems": {"unavailable": [], "stale_updates": []}}`
- **Updated Entity Interface**:
  - Changed `coordinator.entity_manager` to `coordinator.entities`
  - Updated entity method calls from `force_update()` to `async_update()`
  - Fixed entity access patterns to use proper coordinator interface
- **Enhanced Test Logic**:
  - Added proper datetime handling for stale update testing
  - Fixed entity details tests to handle complete entity data structure
  - Updated area status tests to handle occupancy state mocking
  - Enhanced entity type tests to use proper `InputType` enum values

**Service Testing Patterns**:
```python
# Service test with centralized fixtures
class TestServiceFunction:
    async def test_service_success(
        self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
    ) -> None:
        """Test service functionality."""
        mock_coordinator = Mock()
        mock_coordinator.entities.entities = {"entity1": mock_entity}

        # Setup config entry with runtime_data
        mock_config_entry.runtime_data = mock_coordinator
        mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
        mock_service_call.data = {"entry_id": "test_entry_id"}

        result = await service_function(mock_hass, mock_service_call)

        # Check actual return structure
        assert "expected_root_key" in result
        assert result["expected_root_key"]["field"] == expected_value

# Coordinator retrieval pattern
def test_coordinator_access(self, mock_hass: Mock, mock_config_entry: Mock) -> None:
    """Test coordinator access pattern."""
    mock_coordinator = Mock()
    mock_config_entry.runtime_data = mock_coordinator
    mock_hass.config_entries.async_entries.return_value = [mock_config_entry]

    result = _get_coordinator(mock_hass, "test_entry_id")
    assert result == mock_coordinator

# Error handling pattern
async def test_service_error_handling(
    self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
) -> None:
    """Test service error handling."""
    mock_coordinator = Mock()
    mock_coordinator.some_operation.side_effect = Exception("Test error")

    mock_config_entry.runtime_data = mock_coordinator
    mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
    mock_service_call.data = {"entry_id": "test_entry_id"}

    with pytest.raises(HomeAssistantError, match="Failed to .* for test_entry_id: Test error"):
        await service_function(mock_hass, mock_service_call)

# Entity details test pattern
async def test_entity_details(
    self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call_with_entity: Mock
) -> None:
    """Test entity details service."""
    mock_coordinator = Mock()
    mock_entities = Mock()

    # Mock complete entity structure
    mock_entity = Mock()
    mock_entity.state = "on"
    mock_entity.probability = 0.75
    mock_entity.decay.decay_factor = 1.0
    mock_entity.type.input_type.value = "motion"
    mock_entity.prior.prior = 0.35

    mock_entities.get_entity.return_value = mock_entity
    mock_coordinator.entities = mock_entities

    mock_config_entry.runtime_data = mock_coordinator
    mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
    mock_service_call_with_entity.data = {
        "entry_id": "test_entry_id",
        "entity_ids": ["binary_sensor.motion1"],
    }

    result = await _get_entity_details(mock_hass, mock_service_call_with_entity)

    assert "entity_details" in result
    assert "binary_sensor.motion1" in result["entity_details"]
    entity_detail = result["entity_details"]["binary_sensor.motion1"]
    assert entity_detail["state"] == "on"
    assert entity_detail["probability"] == 0.75

# Area status test pattern with state mocking
async def test_area_status(
    self, mock_hass: Mock, mock_config_entry: Mock, mock_service_call: Mock
) -> None:
    """Test area status service."""
    mock_coordinator = Mock()
    mock_coordinator.config.name = "Test Area"
    mock_coordinator.entities.entities = {}

    mock_config_entry.runtime_data = mock_coordinator
    mock_hass.config_entries.async_entries.return_value = [mock_config_entry]
    mock_service_call.data = {"entry_id": "test_entry_id"}

    # Mock occupancy state in Home Assistant
    mock_state = Mock()
    mock_state.state = "on"
    mock_state.attributes = {"probability": 0.8}
    mock_state.last_updated.isoformat.return_value = "2024-01-01T00:00:00"
    mock_hass.states.get.return_value = mock_state

    result = await _get_area_status(mock_hass, mock_service_call)

    assert "area_status" in result
    status = result["area_status"]
    assert status["area_name"] == "Test Area"
    assert status["is_occupied"] is True
    assert status["occupancy_probability"] == 0.8
```

**Results**: 31 failed tests → Comprehensive refactoring completed (eliminated ~200 lines of duplicate fixture code, aligned with actual service implementation patterns)

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

### Migration Test Patterns

For tests requiring Home Assistant storage functionality:

```python
class TestStorageComponent:
    async def test_storage_operation(self, mock_hass: Mock) -> None:
        """Test component using storage."""
        with (
            patch("homeassistant.helpers.storage.Store.async_load",
                  new_callable=AsyncMock, return_value=None),
            patch("homeassistant.helpers.storage.Store.async_save",
                  new_callable=AsyncMock),
        ):
            # Test code that uses storage
            result = await component_function(mock_hass)
            assert result is True
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

### 5. Storage and Migration Patterns

For migration tests, always patch the Storage class:

```python
# Required for migration tests to avoid storage manager issues
with (
    patch("homeassistant.helpers.storage.Store.async_load", new_callable=AsyncMock),
    patch("homeassistant.helpers.storage.Store.async_save", new_callable=AsyncMock),
):
    # Migration test code
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

# Test migration functionality
python -m pytest tests/test_migrations.py -v
```

This centralized approach has reduced test code duplication by ~200+ lines while ensuring consistent mock behavior across the entire test suite. The migration test refactoring eliminated an additional ~50 lines of duplicate fixtures while solving complex storage mocking issues.
