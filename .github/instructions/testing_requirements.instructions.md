---
applyTo: 'tests/*'
---
# Testing Requirements

## Test Structure

Tests are organized in the `tests` directory following Home Assistant conventions:

- [tests/test_integration.py](mdc:tests/test_integration.py): Integration setup tests
- [tests/test_config_flow.py](mdc:tests/test_config_flow.py): Configuration flow tests
- [tests/test_coordinator.py](mdc:tests/test_coordinator.py): Data coordinator tests
- [tests/test_calculate_prior.py](mdc:tests/test_calculate_prior.py): Prior calculation tests
- [tests/test_calculate_prob.py](mdc:tests/test_calculate_prob.py): Probability calculation tests
- [tests/test_decay_handler.py](mdc:tests/test_decay_handler.py): Decay functionality tests
- [tests/test_storage.py](mdc:tests/test_storage.py): Data persistence tests
- [tests/test_service.py](mdc:tests/test_service.py): Service implementation tests
- [tests/test_probabilities.py](mdc:tests/test_probabilities.py): Default probability tests
- [tests/test_types.py](mdc:tests/test_types.py): Type validation tests
- [tests/test_virtual_sensor_wasp_in_box.py](mdc:tests/test_virtual_sensor_wasp_in_box.py): Virtual sensor tests
- [tests/test_environmental_analysis.py](mdc:tests/test_environmental_analysis.py): Environmental sensor analysis tests
- [tests/test_environmental_storage.py](mdc:tests/test_environmental_storage.py): Environmental data storage tests
- [tests/test_ml_models.py](mdc:tests/test_ml_models.py): Machine learning model tests
- [tests/conftest.py](mdc:tests/conftest.py): Pytest fixtures and configuration

## Test Coverage Requirements

### Core Components (100% coverage)
- Prior probability calculations
- Bayesian probability calculations
- Probability decay functionality
- Configuration validation
- Entity state management
- Service handlers
- Data persistence and storage
- Type validation and conversion
- Environmental sensor analysis algorithms
- Machine learning model training and inference
- Environmental data processing and feature engineering

### Integration Components (90% coverage)
- Setup and initialization
- State updates and listeners
- Historical data fetching
- Error handling paths
- Coordinator updates
- Configuration migrations
- Virtual sensor implementations
- Environmental sensor integration
- ML model lifecycle management
- Hybrid analysis method switching

## Test Categories

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies
- Verify calculation accuracy
- Test edge cases and error conditions
- Validate type hints and interfaces

### Integration Tests
- Test component interactions
- Verify state propagation
- Test configuration flows
- Validate service calls
- Test update coordination

### Mocking Guidelines
- Use `pytest-mock` for mocking
- Mock Home Assistant core services
- Mock recorder/history data
- Mock entity states and updates
- Provide realistic test data

## Test Cases

### Prior Calculation Tests
- Calculate priors with sufficient history
- Handle insufficient history data
- Test correlation calculations
- Verify default fallbacks
- Test time window handling

### Probability Calculation Tests
- Test Bayesian formula implementation
- Verify probability combinations
- Test normalization
- Handle missing/invalid inputs
- Test threshold calculations

### Decay Handler Tests
- Test probability decay functionality
- Verify decay rate calculations
- Test decay start/stop conditions
- Handle edge cases in decay logic
- Validate decay state persistence

### Environmental Analysis Tests
- Test environmental data processing accuracy
- Verify feature engineering correctness
- Test ML model training and prediction
- Validate deterministic rule accuracy
- Test hybrid method selection
- Verify sensor correlation analysis
- Test data quality assessment
- Validate environmental prior calculations

### Machine Learning Tests
- Test model training with various data sizes
- Verify prediction accuracy and confidence
- Test model serialization and loading
- Validate incremental learning
- Test model performance monitoring
- Verify feature importance calculation
- Test overfitting prevention
- Validate cross-validation results

### Storage Tests
- Test data persistence across restarts
- Verify state serialization/deserialization
- Test storage error handling
- Validate data migration
- Test storage cleanup
- Test environmental data storage
- Verify model persistence and versioning

### Configuration Tests
- Validate entity selections
- Test option validation
- Verify schema updates
- Test error handling
- Validate migrations
- Test environmental sensor configuration
- Verify analysis method selection

### Entity Tests
- Test sensor creation and updates
- Test binary sensor thresholds
- Test number entity configurations
- Verify state updates and attributes
- Validate availability and cleanup

### Virtual Sensor Tests
- Test virtual sensor implementations
- Verify Wasp in Box algorithm
- Test virtual sensor state updates
- Validate virtual sensor configurations
- Test virtual sensor error handling

### Service Tests
- Test service call implementations
- Verify service parameter validation
- Test service error handling
- Validate service state changes
- Test service permissions

## Test Execution

### Setup
- Use pytest
- Enable coverage reporting
- Configure test fixtures
- Set up mocking utilities
- Prepare test data

### Running Tests
- Run with `pytest tests/`
- Generate coverage report
- Verify minimum coverage
- Check test performance
- Validate clean teardown

### CI Integration
- Run tests in GitHub Actions
- Enforce coverage thresholds
- Report test results
- Block merges on failures
- Archive test artifacts

