# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Area Occupancy Detection is a Home Assistant custom integration that uses Bayesian probability to intelligently detect room occupancy. It combines multiple sensor inputs (motion, media devices, appliances, environmental sensors) with learned historical patterns to provide probabilistic occupancy detection.

The integration learns from your patterns over time, uses decay functions to handle stationary occupancy, and provides both binary occupancy status and probability sensors that automations can use.

## Development Commands

### Cloning the Repository

**Important:** Use the optimized clone command to avoid downloading bloated PR history:

```bash
# Efficient clone (recommended - downloads only ~6MB)
git clone --single-branch --branch main https://github.com/Hankanman/Area-Occupancy-Detection.git

# Regular clone downloads ~445MB due to historical PR refs
# If you already cloned normally, your local repo works fine but is larger than necessary
```

### Environment Setup

```bash
# Bootstrap environment (installs uv, creates venv, installs dependencies, sets up pre-commit)
scripts/bootstrap

# Manual setup if not using devcontainer
# 1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run: scripts/bootstrap
# 3. Activate: source .venv/bin/activate
```

### Code Quality

```bash
# Lint and format code (runs ruff format and ruff check --fix)
scripts/lint

# Manual linting
uv run ruff format .
uv run ruff check . --fix
```

### Testing

```bash
# Run all tests with coverage report
scripts/test

# Run specific test file
uv run pytest tests/test_area_area.py

# Run with verbose output
uv run pytest -v

# Run specific test within a file
uv run pytest tests/test_area_area.py::test_area_initialization -v
```

### Development Environment

This project uses a **devcontainer** that provides a standalone Home Assistant instance. When opening in VS Code, accept the devcontainer prompt to get:
- Pre-configured development environment
- Home Assistant running with `config/configuration.yaml`
- All dependencies installed
- Pre-commit hooks configured

## Architecture

### High-Level Structure

The integration uses a **single-instance coordinator architecture** that manages multiple areas:

```
AreaOccupancyCoordinator (global singleton)
├── AreaOccupancyDB (SQLite database with SQLAlchemy)
├── IntegrationConfig (global settings)
└── areas: dict[str, Area]
    └── Area (per-room instance)
        ├── AreaConfig (sensors, weights, thresholds)
        ├── EntityManager (tracks sensor states and evidence)
        ├── Prior (learned probabilities)
        └── Purpose (room type and decay settings)
```

### Key Concepts

**Multi-Area Management**: A single coordinator manages all configured areas. Each area has its own device in Home Assistant with associated entities (sensors, binary sensors, numbers).

**Entity Types**: Sensors are classified by `InputType` (MOTION, MEDIA, APPLIANCE, DOOR, WINDOW, ENVIRONMENTAL, POWER, WASP). Each type has different probability contributions and weights.

**Bayesian Calculation**: The core algorithm combines:
- **Prior probabilities**: Learned from historical patterns (time-of-day, day-of-week)
- **Sensor evidence**: Current state of all sensors with type-specific weights
- **Decay**: Gradual probability reduction when no new evidence arrives
- Result: A probability (1-99%) that updates continuously

**Database Architecture**: SQLite with SQLAlchemy ORM, organized in modules:
- `db/core.py`: Database initialization and session management
- `db/schema.py`: SQLAlchemy table definitions (Areas, Entities, Intervals, Aggregates, etc.)
- `db/operations.py`: CRUD operations for entities and intervals
- `db/aggregation.py`: Time-series aggregation (hourly, daily, weekly, monthly)
- `db/correlation.py`: Statistical correlation analysis between sensors and occupancy
- `db/queries.py`: Complex queries for occupied intervals and cache management
- `db/sync.py`: Import entity states from Home Assistant recorder
- `db/maintenance.py`: Health checks, pruning, backups

**Analysis Pipeline**: Every hour (configurable), the coordinator runs:
1. Sync states from recorder → import recent entity state changes
2. Health check and pruning → validate database integrity, remove old data
3. Populate occupied intervals cache → identify when areas were occupied
4. Run aggregations → hourly/daily/weekly/monthly summaries
5. Recalculate priors → update learned probabilities from historical data
6. Correlation analysis → identify sensor relationships with occupancy
7. Save and refresh → persist changes, update entities

### Critical Files

- `coordinator.py`: Main coordinator managing lifecycle, timers, and multi-area orchestration
- `area/area.py`: Per-area logic, encapsulates configuration, entities, priors, and calculations
- `data/entity.py`: Entity tracking, state management, evidence detection (380+ lines)
- `data/analysis.py`: Orchestrates the full analysis pipeline
- `data/prior.py`: Prior probability calculations from historical patterns
- `data/config.py`: Configuration management for both integration-level and area-level settings
- `data/decay.py`: Time-based probability decay implementation
- `data/purpose.py`: Room purpose definitions (social, work, sleep, etc.) with default decay settings
- `db/core.py`: Database initialization, connection management
- `db/correlation.py`: Statistical analysis of sensor-occupancy relationships (660+ lines)
- `utils.py`: Bayesian probability calculations, state mapping utilities

### Configuration Flow

The integration uses Home Assistant's config flow with a **list-based multi-area architecture**:
- Configuration stored in `config_entry.data[CONF_AREAS]` as a list of area configurations
- Each area config contains: `area_id`, sensor lists, weights, thresholds, decay settings
- Options flow allows adding/editing/removing areas
- Changes trigger `async_update_options()` which handles area lifecycle (create/update/delete)

### State Management

**Entity State Tracking**:
- Single state listener for all entities across all areas (`_area_state_listeners`)
- On state change, finds affected areas and checks if entity has new evidence
- Triggers coordinator refresh only if setup is complete and evidence detected

**Timers**:
- **Decay timer**: Every 10 seconds, refreshes if any area has decay enabled
- **Analysis timer**: Every hour, runs full analysis pipeline (sync, aggregate, correlate, learn)
- **Save timer**: Every 10 minutes, persists data to database

### Testing Architecture

Tests use `pytest-homeassistant-custom-component` with extensive mocking:
- `tests/conftest.py`: Shared fixtures for Home Assistant, coordinator, database
- 85%+ coverage requirement (90% for core calculations)
- Tests organized by component: area, coordinator, db, entities, config flow, etc.
- Mock Home Assistant services, entity states, recorder data
- Use `pytest-cov` for coverage reporting

## Important Development Notes

### Database Operations

All database operations must be run in executor to avoid blocking the event loop:
```python
await self.hass.async_add_executor_job(self.db.save_data)
```

Database sessions are managed with context managers. Never hold sessions across async boundaries.

### Configuration Migrations

When updating `CONF_VERSION`, implement migration in `migrations.py`. Migrations must:
- Handle both data and options dicts
- Preserve all user settings
- Log migration steps
- Be idempotent (safe to run multiple times)

### Entity Evidence

Entity evidence is detected by comparing current state to previous state and checking against active states:
```python
entity.has_new_evidence()  # Returns True if entity state changed and is now "active"
```

This prevents unnecessary coordinator refreshes and ensures decay works correctly.

### Time Handling

The integration uses timezone-aware datetimes throughout:
- Store UTC in database: `to_utc(dt)` from `time_utils.py`
- Convert to local for bucketing/display: `to_local(dt, timezone)`
- Time priors are bucketed by local time (day-of-week, hour) for accurate pattern learning

### Virtual Sensors

"Wasp in Box" is a virtual sensor for bathrooms: when someone enters (door closes) with no motion, it maintains high occupancy probability. Useful for rooms where motion sensors can't see the entire space.

### Performance Considerations

- Occupied intervals cache is validated before queries (hourly refresh)
- Aggregations only process new data since last run
- Database retention policies automatically prune old data
- Correlation analysis requires minimum 50 samples for reliability

### Branch and Release Strategy

- Development happens on `dev` branch
- PR from `dev` to `preview` for prereleases
- PR from `preview` to `main` for full releases
- Use semantic versioning (MAJOR.MINOR.PATCH)
- Version stored in `pyproject.toml`, `const.py`, and `manifest.json`

### Pre-commit Hooks

Pre-commit runs ruff formatting and linting automatically. If it fails:
1. Review the changes it made
2. Stage the changes: `git add -u`
3. Commit again

### Code Style

- Use ruff for formatting and linting (matches Home Assistant core standards)
- Full type annotations required (Python 3.13+)
- Google-style docstrings for public APIs
- Log at appropriate levels: debug for internals, info for user-visible events, warning/error for issues
- Use `_LOGGER` from module-level `logging.getLogger(__name__)`

## Common Workflows

### Adding a New Sensor Type

1. Add `InputType` enum value to `data/entity_type.py`
2. Add default probabilities to `const.py`
3. Update `EntityFactory` in `data/entity.py` to handle new type
4. Add configuration keys to `const.py` (e.g., `CONF_NEW_TYPE_SENSORS`)
5. Update config flow in `config_flow.py` to allow sensor selection
6. Add weight configuration (e.g., `CONF_WEIGHT_NEW_TYPE`)
7. Update database schema if needed (add correlation tracking)
8. Write tests

### Modifying Bayesian Calculation

1. Core calculation is in `utils.py::bayesian_probability()`
2. Prior calculation in `data/prior.py`
3. Decay implementation in `data/decay.py`
4. Evidence detection in `data/entity.py::Entity.has_new_evidence()`
5. Update tests in `tests/test_calculate_prob.py` or similar
6. Ensure 100% coverage for calculation changes

### Debugging Database Issues

Enable debug logging in Home Assistant's `configuration.yaml`:
```yaml
logger:
  default: info
  logs:
    custom_components.area_occupancy: debug
    custom_components.area_occupancy.db: debug
```

Check for:
- Failed integrity checks (indicates schema issues)
- Long-running queries (check indexes)
- Pruning failures (retention policy issues)
- Correlation analysis errors (insufficient data)
