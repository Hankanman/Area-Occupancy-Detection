---
applyTo: '**'
---
# Advanced Features and Components

## Decay Handler

The decay handler in [custom_components/area_occupancy/decay_handler.py](mdc:custom_components/area_occupancy/decay_handler.py) provides sophisticated probability decay functionality:

### Implementation Requirements
- Exponential decay algorithm with configurable rate
- Time-based decay using datetime calculations
- Smooth probability transitions
- State persistence across restarts
- Integration with coordinator for updates

### Configuration
- Decay rate: Configurable via number entity (0.1-1.0 range)
- Decay threshold: Probability level that triggers decay
- Minimum decay probability: Lower bound for decay
- Decay interval: Time between decay calculations

### Error Handling
- Validate decay parameters
- Handle timer edge cases
- Recovery from invalid decay states
- Logging for decay transitions

## Storage System

The storage system in [custom_components/area_occupancy/storage.py](mdc:custom_components/area_occupancy/storage.py) handles data persistence:

### Data Management
- Async file I/O operations
- JSON serialization with type validation
- Backup and recovery mechanisms
- Data migration support
- Performance optimization

### Stored Data Types
- ProbabilityState: Current calculation state
- PriorData: Historical prior calculations
- DecayState: Decay parameters and status
- Configuration: User settings and entity mappings

### Storage Guidelines
- Use Home Assistant's storage API
- Implement proper error handling
- Validate data on load
- Provide migration paths for schema changes
- Minimize storage footprint

## Virtual Sensors

Virtual sensors in [custom_components/area_occupancy/virtual_sensor/](mdc:custom_components/area_occupancy/virtual_sensor/) provide advanced detection algorithms:

### Wasp in Box Algorithm
- Probabilistic motion detection
- Pattern recognition for occupancy
- Confidence scoring system
- Temporal analysis of motion events
- False positive reduction

### Integration Requirements
- Implement standard entity interface
- Participate in Bayesian calculations
- Provide configuration options
- Support enable/disable functionality
- Log algorithm decisions

### Development Guidelines
- Each virtual sensor in separate module
- Use abstract base class for common functionality
- Implement proper state management
- Provide comprehensive testing
- Document algorithm methodology

## Exception Handling

Custom exceptions in [custom_components/area_occupancy/exceptions.py](mdc:custom_components/area_occupancy/exceptions.py):

### Exception Types
- CalculationError: Bayesian calculation failures
- StorageError: Data persistence issues
- ConfigurationError: Invalid configuration
- DecayError: Decay handler problems
- ValidationError: Data validation failures

### Usage Guidelines
- Use specific exceptions over generic Exception
- Include meaningful error messages
- Log exceptions with stack traces
- Provide recovery mechanisms where possible
- Document exception conditions

## Migration System

Configuration migrations in [custom_components/area_occupancy/migrations.py](mdc:custom_components/area_occupancy/migrations.py):

### Migration Requirements
- Version-based migration system
- Backward compatibility preservation
- Data validation during migration
- Rollback capabilities
- Migration logging

### Implementation
- Detect configuration version
- Apply incremental migrations
- Validate migrated data
- Update version markers
- Handle migration failures gracefully

## State Mapping

Entity state mapping in [custom_components/area_occupancy/state_mapping.py](mdc:custom_components/area_occupancy/state_mapping.py):

### Mapping Functions
- Binary states (on/off, open/closed)
- Numeric ranges to probability ranges
- Enum states to occupancy indicators
- Device-specific state handling
- Custom mapping configurations

### Implementation Guidelines
- Support all Home Assistant entity types
- Provide sensible defaults
- Allow custom mapping overrides
- Handle edge cases gracefully
- Validate mapped values

## Number Entities

Number entities in [custom_components/area_occupancy/number.py](mdc:custom_components/area_occupancy/number.py):

### Entity Types
- Probability Threshold: Binary sensor threshold (0-1)
- Decay Rate: Probability decay rate (0.1-1.0)
- History Period: Days of history for prior calculation
- Update Interval: Coordinator update frequency

### Requirements
- Real-time updates to coordinator
- Validation of input ranges
- Persistence of user settings
- Integration with config flow
- Proper entity lifecycle management

## Services

Service implementations in [custom_components/area_occupancy/service.py](mdc:custom_components/area_occupancy/service.py):

### Available Services
- Recalculate Priors: Force prior recalculation
- Reset Decay: Reset decay state
- Update Probability: Manual probability override
- Export Data: Export state for debugging
- Import Configuration: Bulk configuration import

### Service Guidelines
- Validate service parameters
- Provide meaningful responses
- Log service calls
- Handle errors gracefully
- Support async operations

## Internationalization

Translation support in [custom_components/area_occupancy/translations/](mdc:custom_components/area_occupancy/translations/) and [custom_components/area_occupancy/strings.json](mdc:custom_components/area_occupancy/strings.json):

### Translation Requirements
- Support multiple languages
- Translate all user-facing strings
- Use Home Assistant translation system
- Provide fallbacks for missing translations
- Maintain translation consistency

### Implementation
- Define strings in strings.json
- Provide language-specific files
- Use translation keys in UI components
- Test translations in different locales
- Update translations with new features
