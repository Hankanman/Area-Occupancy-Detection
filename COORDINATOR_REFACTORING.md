# Coordinator Refactoring Plan

## Problem
The `AreaOccupancyCoordinator` has grown to over 1200 lines and violates the Single Responsibility Principle by handling too many concerns.

## Current Responsibilities (10+)
1. **Component Initialization** - Setting up 8+ components
2. **State Management** - Managing current/previous sensor states  
3. **Event Handling** - Setting up and managing state change listeners
4. **Storage Operations** - Loading/saving prior state data
5. **Decay Management** - Timer handling and decay calculations
6. **Prior Updates** - Scheduling and executing learned prior calculations
7. **Configuration Management** - Handling config updates and validation
8. **Data Updates** - Coordinating probability calculations
9. **ML Integration** - Managing ML model components
10. **Error Handling** - Try/catch for all operations

## Revised Proposed Architecture

### 1. **StateManager** (Rename & Enhance EntityManager)
**Responsibility**: Complete state management for the integration
```python
class StateManager:
    """Manages all state for the Area Occupancy integration."""
    
    # Existing entity functionality (renamed from EntityManager)
    def get_entity_type(self, entity_id: str) -> EntityType
    def is_entity_active(self, entity_id: str, state: str) -> bool
    def get_entity_state(self, entity_id: str) -> SensorInfo
    def validate_entity(self, entity_id: str) -> EntityValidationResult
    def get_entity_statistics(self) -> dict[str, Any]
    
    # New state tracking functionality (moved from coordinator)
    async def initialize_states(self, sensor_ids: list[str]) -> None
    def setup_state_tracking(self, hass: HomeAssistant, callback: Callable) -> None
    async def update_state(self, entity_id: str, state_info: SensorInfo) -> None
    def get_current_states(self) -> dict[str, SensorInfo]
    def get_previous_states(self) -> dict[str, SensorInfo]
    def get_tracked_entities(self) -> set[str]
    def stop_state_tracking(self) -> None
```

### 2. **PriorManager** (New)
**Responsibility**: Prior probability management and scheduling
```python
class PriorManager:
    """Manages prior probability calculations and scheduling."""
    
    async def load_stored_priors(self) -> None
    async def update_learned_priors(self, history_period: int = None) -> None
    async def schedule_next_update(self) -> None
    async def save_prior_state(self) -> None
```

### 3. **DecayManager** (Enhanced DecayHandler)
**Responsibility**: Probability decay logic and timer management
```python
class DecayManager:
    """Manages probability decay calculations and timers."""
    
    def start_decay_updates(self) -> None
    def stop_decay_updates(self) -> None
    async def handle_decay_update(self) -> None
    def calculate_decay(self, ...) -> tuple[float, ...]
```

### 4. **ConfigurationManager** (New)
**Responsibility**: Configuration updates and component reinitialization
```python
class ConfigurationManager:
    """Manages configuration updates and component lifecycle."""
    
    async def update_options(self, new_config: dict) -> None
    async def update_threshold(self, value: float) -> None
    def reinitialize_components(self) -> None
```

### 5. **Simplified Coordinator**
**Responsibility**: Orchestration only - delegates to managers
```python
class AreaOccupancyCoordinator(DataUpdateCoordinator[ProbabilityState]):
    """Orchestrates area occupancy detection by coordinating managers."""
    
    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry):
        # Initialize core components
        self.state_manager = StateManager(...)  # Renamed from EntityManager, enhanced with state tracking
        self.prior_manager = PriorManager(...)
        self.decay_manager = DecayManager(...)
        self.config_manager = ConfigurationManager(...)
        
    async def _async_update_data(self) -> ProbabilityState:
        # Get states from state_manager, delegate to calculator
```

## Benefits

1. **Single Responsibility** - Each class has one clear purpose
2. **Testability** - Smaller classes are easier to unit test
3. **Maintainability** - Changes to one concern don't affect others
4. **Reusability** - Managers could be reused in other contexts
5. **Clarity** - Much easier to understand what each component does

## Revised Migration Strategy

### ✅ **Phase 1: Enhance StateManager with State Tracking (COMPLETED)**

**Completed Actions:**
- ✅ Renamed EntityManager to StateManager
- ✅ Added state tracking functionality to StateManager:
  - `async_initialize_states()` - Initialize and track sensor states
  - `setup_state_tracking()` - Set up state change listeners
  - `get_current_states()` / `get_previous_states()` - State access methods
  - `stop_state_tracking()` - Cleanup method
- ✅ Updated coordinator to delegate state management to StateManager
- ✅ Removed duplicate state management code from coordinator:
  - Removed `_setup_entity_tracking()` method
  - Updated `async_initialize_states()` to use StateManager
  - Updated `async_shutdown()` to use StateManager cleanup
- ✅ Fixed all import references and parameter names

**Result:** StateManager now handles all state management for the integration, eliminating duplication and providing a clean separation of concerns.

### ✅ **Phase 2: Extract PriorManager (COMPLETED)**

**Completed Actions:**
- ✅ Created new `PriorManager` class to handle all prior-related functionality:
  - `async_setup()` - Setup and determine if initial prior calculation needed
  - `async_shutdown()` - Cleanup and cancel scheduled tasks
  - `load_stored_priors()` - Load prior state data from storage
  - `save_prior_state()` - Save prior state data to storage
  - `update_learned_priors()` - Execute prior calculations using historical data
  - `schedule_next_update()` - Schedule periodic prior updates
- ✅ Moved all prior logic from coordinator to PriorManager:
  - Prior state management and initialization
  - Prior calculation scheduling and execution
  - Storage operations for prior data
  - Type prior aggregation from entity priors
  - Startup logic for determining when prior updates are needed
- ✅ Updated coordinator to delegate to PriorManager:
  - Removed ~400 lines of prior-related code from coordinator
  - Added PriorManager initialization in coordinator constructor
  - Updated public methods to delegate to PriorManager
  - Updated async_setup to use PriorManager.async_setup()
- ✅ Fixed all linter errors and type annotations:
  - Proper type annotation for `prior_manager: PriorManager`
  - Fixed delegation to pass `prior_state` instead of `prior_manager` to calculators
  - Removed duplicate prior scheduling methods from coordinator
  - Fixed broken storage loading and saving methods
- ✅ Created comprehensive tests for PriorManager functionality with proper cleanup

**Result:** PriorManager now handles all prior probability management, removing substantial complexity from the coordinator. The coordinator went from ~1200 lines to ~687 lines while maintaining all functionality and passing all tests.

### **Phase 3**: Enhance DecayManager (move timer logic from coordinator) - PENDING

### **Phase 4**: Extract ConfigurationManager (config update logic) - PENDING

### **Phase 5**: Simplify Coordinator (remove extracted logic) - PENDING

## Implementation Notes

- Maintain existing public API for backward compatibility
- Use dependency injection to pass managers to coordinator
- Keep error handling at the coordinator level initially
- Ensure all managers are properly async/await compatible 