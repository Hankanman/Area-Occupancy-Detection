---
applyTo: 'tests/*'
---
# Testing Requirements

## Test Structure

Tests are organized in the `tests` directory following Home Assistant conventions:

- [tests/test_init.py](mdc:tests/test_init.py): Integration setup tests
- [tests/test_config_flow.py](mdc:tests/test_config_flow.py): Configuration flow tests
- [tests/test_coordinator.py](mdc:tests/test_coordinator.py): Data coordinator tests
- [tests/test_calculate_prior.py](mdc:tests/test_calculate_prior.py): Prior calculation tests
- [tests/test_calculate_prob.py](mdc:tests/test_calculate_prob.py): Probability calculation tests
- [tests/test_sensor.py](mdc:tests/test_sensor.py): Sensor entity tests
- [tests/test_binary_sensor.py](mdc:tests/test_binary_sensor.py): Binary sensor tests
- [tests/conftest.py](mdc:tests/conftest.py): Pytest fixtures and configuration

## Test Coverage Requirements

### Core Components (100% coverage)
- Prior probability calculations
- Bayesian probability calculations
- Configuration validation
- Entity state management
- Service handlers

### Integration Components (90% coverage)
- Setup and initialization
- State updates and listeners
- Historical data fetching
- Error handling paths
- Coordinator updates

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

### Configuration Tests
- Validate entity selections
- Test option validation
- Verify schema updates
- Test error handling
- Validate migrations

### Entity Tests
- Test sensor creation
- Verify state updates
- Test attribute handling
- Validate availability
- Test cleanup

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

