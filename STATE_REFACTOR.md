I'll create a comprehensive refactoring plan document that outlines the changes needed to improve state management.

# Area Occupancy Detection - State Management Refactoring Plan

## 1. Current Issues

### 1.1 State Management Fragmentation
- State logic spread across multiple modules (`types.py`, `probabilities.py`, `coordinator.py`)
- Duplicate validation logic
- Mixed concerns in state classes
- Unclear state update paths
- Complex state dependencies

### 1.2 Specific Problems
- `ProbabilityState` and `PriorState` contain logic
- `Probabilities` class mixes configuration and state management
- Validation scattered across different classes
- State updates lack clear boundaries
- Serialization/deserialization mixed with state logic

## 2. Proposed Architecture

### 2.1 Core Components
```
state/
├── __init__.py
├── containers.py      # Pure data containers
├── validation.py      # State validation logic
├── updates.py         # State update handlers
├── serialization.py   # State serialization/deserialization
└── config.py          # Configuration management
```

### 2.2 Logic
```
logic/
├── __init__.py
├── prior_calculator.py    # Prior probability calculations
├── probability_calculator.py  # Probability calculations
└── decay_handler.py       # Decay management
```

## 3. Implementation Plan

### Phase 1: Core State Management
1. Create new state management structure
2. Implement pure data containers
3. Add validation system
4. Create state update handlers
5. Implement serialization

### Phase 2: Logic Separation
1. Extract logic from state classes
2. Create dedicated calculator classes
3. Implement configuration management
4. Add state update coordination

### Phase 3: Integration
1. Update coordinator to use new system
2. Migrate existing state handling
3. Update dependent modules
4. Add migration support

## 4. Detailed Changes

### 4.1 State Containers (`containers.py`)
```python
@dataclass
class ProbabilityState:
    """Pure data container for probability state."""
    probability: float
    previous_probability: float
    threshold: float
    prior_probability: float
    sensor_probabilities: dict[str, SensorProbability]
    decay_status: float
    current_states: dict[str, SensorInfo]
    previous_states: dict[str, SensorInfo]
    is_occupied: bool
    decaying: bool
    decay_start_time: datetime | None
    decay_start_probability: float | None

@dataclass
class PriorState:
    """Pure data container for prior state."""
    overall_prior: float
    motion_prior: float
    media_prior: float
    appliance_prior: float
    door_prior: float
    window_prior: float
    light_prior: float
    environmental_prior: float
    wasp_in_box_prior: float
    entity_priors: dict[str, PriorData]
    type_priors: dict[str, PriorData]
    analysis_period: int
```

### 4.2 State Validation (`validation.py`)
```python
class StateValidator:
    """Centralized state validation."""
    
    @staticmethod
    def validate_probability(value: float, name: str) -> None:
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    
    @staticmethod
    def validate_prior_state(state: PriorState) -> None:
        for field in [
            "overall_prior", "motion_prior", "media_prior",
            "appliance_prior", "door_prior", "window_prior",
            "light_prior", "environmental_prior", "wasp_in_box_prior"
        ]:
            StateValidator.validate_probability(getattr(state, field), field)
```

### 4.3 State Updates (`updates.py`)
```python
class StateUpdater:
    """Handles state updates with validation."""
    
    def __init__(self, validator: StateValidator):
        self._validator = validator
    
    def update_probability_state(
        self, 
        state: ProbabilityState,
        updates: dict[str, Any]
    ) -> ProbabilityState:
        """Update probability state with validation."""
        for key, value in updates.items():
            if hasattr(state, key):
                if key in ["probability", "previous_probability", "threshold", "prior_probability"]:
                    self._validator.validate_probability(value, key)
                setattr(state, key, value)
        return state
```

### 4.4 Configuration Management (`config.py`)
```python
class SensorConfiguration:
    """Manages sensor configuration and type mapping."""
    
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._entity_types: dict[str, EntityType] = {}
        self._sensor_weights: dict[str, float] = {}
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize configuration from raw config."""
        self._map_entity_types()
        self._calculate_weights()
```

## 5. Migration Strategy

### 5.1 Step-by-Step Migration
1. Create new state management structure
2. Add new components alongside existing code
3. Gradually migrate state handling to new system
4. Update dependent modules one at a time
5. Remove old state management code

### 5.2 Module Migration Order
1. `types.py` → `state/containers.py`
2. `probabilities.py` → `state/config.py`
3. `calculate_prior.py` → `logic/prior_calculator.py`
4. `calculate_prob.py` → `logic/probability_calculator.py`
5. `decay_handler.py` → `logic/decay_handler.py`
6. Update `coordinator.py` to use new system

### 5.3 Testing Strategy
1. Unit tests for new components
2. Integration tests for state management
3. Migration tests for data conversion
4. End-to-end tests for full functionality

## 6. Benefits

### 6.1 Improved Maintainability
- Clear separation of concerns
- Centralized validation
- Explicit state updates
- Isolated configuration
- Separated logic

### 6.2 Better Extensibility
- Easy to add new state fields
- Simple to add validation rules
- Clear patterns for updates
- Flexible configuration
- Modular logic

### 6.3 Enhanced Debugging
- Clear state update paths
- Centralized validation
- Explicit state changes
- Better error messages
- Easier to track issues

## 7. Risks and Mitigation

### 7.1 Potential Risks
- Breaking changes to existing functionality
- Performance impact of new validation
- Migration complexity
- Integration issues

### 7.2 Mitigation Strategies
- Comprehensive testing
- Gradual migration
- Performance profiling
- Clear documentation
- Fallback mechanisms

## 8. Timeline

### 8.1 Phase 1 (Week 1-2)
- Core state management implementation
- Basic validation system
- Initial testing

### 8.2 Phase 2 (Week 3-4)
- Logic separation
- Configuration management
- Integration testing

### 8.3 Phase 3 (Week 5-6)
- Full migration
- Performance optimization
- Documentation
- Final testing

Would you like me to elaborate on any part of this plan or provide more specific details for any section?
