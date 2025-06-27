# Development Environment Setup

## Ensuring Local/CI Environment Parity

This guide explains how to maintain consistency between your local development environment and the CI environment to prevent test failures.

## Development Container Setup (Recommended)

The project includes a configured devcontainer that automatically sets up a consistent development environment:

1. **Open in Dev Container** (VS Code):
   - Install the "Dev Containers" extension
   - Open the project in VS Code
   - Click "Reopen in Container" when prompted
   - The container will automatically:
     - Create a Python 3.13 environment
     - Set up a virtual environment (`.venv`)
     - Install all dependencies
     - Validate the environment

2. **Manual Setup** (if not using devcontainer):
   ```bash
   # Create and activate virtual environment
   python3.13 -m venv .venv
   source .venv/bin/activate

   # Run setup script
   ./scripts/setup
   ```

## Quick Validation

After setup, validate your environment matches CI:

```bash
# In devcontainer, venv is auto-activated
./scripts/validate-env

# For manual setup, ensure venv is activated first
source .venv/bin/activate
./scripts/validate-env
```

## Environment Consistency Benefits

The devcontainer approach ensures:
- **Identical Python version** (3.13.x) as CI
- **Isolated dependencies** via virtual environment
- **Consistent package versions** via pinned requirements
- **Same OS environment** (Ubuntu-based) as CI
- **Automated validation** on container creation

## Running Tests Like CI

```bash
# Run complete CI-like test suite
./scripts/test-ci-local

# Run specific failing tests
pytest tests/test_binary_sensor.py::TestWaspInBoxSensor::test_set_state_to_occupied -v
```

## Environment Validation Tools

### `scripts/validate-env`
Checks that your local environment matches CI expectations:
- Python version (3.13.x)
- Key package versions
- pytest configuration
- Required files

### `scripts/test-ci-local`
Runs the complete CI test suite locally:
- Environment validation
- Dependency installation
- Linting checks
- Full test suite with CI flags

## Pre-commit Hooks

Install pre-commit hooks to catch issues early:
```bash
pip install pre-commit
pre-commit install
```

This will automatically run on each commit:
- Environment validation
- Code formatting (ruff)
- Linting checks
- Core tests

## Key Dependencies to Monitor

These packages are critical for test consistency:

| Package | Version | Why Critical |
|---------|---------|--------------|
| `pytest` | 8.4.0 | Test runner behavior |
| `pytest-asyncio` | 1.0.0 | Event loop management |
| `pytest-homeassistant-custom-component` | 0.13.256 | HA-specific test fixtures |

## Common Environment Issues

### 1. **Asyncio/Timer Issues**
**Symptoms:** Tests pass locally but fail in CI with "lingering timer" errors

**Causes:**
- Different pytest-asyncio versions
- Different event loop cleanup behavior
- Mock timer objects not properly cancellable

**Solutions:**
- Use devcontainer for identical environment
- Pin exact pytest-asyncio version
- Use cancellable timer mocks in fixtures

### 2. **Version Drift**
**Symptoms:** Different test behavior between local and CI

**Causes:**
- Unpinned dependency versions
- Local cache with old versions
- Different Python versions

**Solutions:**
- Use devcontainer with pinned versions
- Virtual environment isolation
- Regular `./scripts/validate-env` checks

### 3. **Platform Differences**
**Symptoms:** Tests behave differently on different OS

**Causes:**
- Path handling differences
- File permission differences
- Timing differences

**Solutions:**
- Use devcontainer (Linux-based like CI)
- Platform-independent code patterns
- Mock platform-specific behavior

## Debugging Environment Issues

### 1. **Compare Package Versions**
```bash
# In devcontainer
pip list | grep pytest

# Check CI logs for version output
```

### 2. **Run Specific Failing Test**
```bash
# Run the exact test that failed in CI
pytest tests/test_specific.py::TestClass::test_method -v
```

### 3. **Enable Debug Output**
```bash
# Run with maximum verbosity
pytest --tb=long -vv -s
```

### 4. **Test Timer Cleanup**
```bash
# Test with strict cleanup checking
pytest -W error::RuntimeWarning
```

## Development Workflow

1. **Open devcontainer** (VS Code) or activate venv manually
2. **Validate environment:**
   ```bash
   ./scripts/validate-env
   ```
3. **Make changes** with confidence in consistent environment
4. **Before committing:**
   ```bash
   ./scripts/test-ci-local
   ```
5. **If tests fail in CI but pass locally:**
   - Environment is already matched via devcontainer
   - Check for recent dependency updates
   - Review CI logs for any infrastructure changes

## Container Features

The devcontainer includes:
- **Python 3.13** with virtual environment
- **Node.js** for documentation tools
- **GitHub CLI** for workflow management
- **Docker-in-Docker** for container testing
- **VS Code extensions** for optimal development

## Continuous Monitoring

The CI workflow includes:
- Environment validation step
- Package version logging
- Consistent test flags
- Enhanced error reporting

This ensures any environment drift is caught early and provides debugging information when issues occur.

## Benefits of This Approach

1. **True Environment Parity**: Devcontainer matches CI exactly
2. **Isolated Dependencies**: Virtual environment prevents conflicts
3. **Automated Setup**: No manual environment configuration
4. **Consistent Tooling**: Same VS Code extensions and settings
5. **Easy Onboarding**: New developers get identical setup
6. **Reproducible Issues**: Bugs are consistent across environments

## Future Improvements

Consider these additional measures:

1. **Docker Development Environment**
   - Use the same container as CI for local development
   - Ensures identical environment

2. **Automated Dependency Updates**
   - Use Dependabot or similar for controlled updates
   - Test in CI before merging

3. **Environment Snapshots**
   - Generate `requirements-lock.txt` files
   - Pin transitive dependencies too