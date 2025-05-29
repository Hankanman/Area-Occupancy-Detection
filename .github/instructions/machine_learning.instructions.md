# Machine Learning Requirements (Universal)

## Overview

Machine learning (ML) models are used to enhance occupancy detection by learning complex patterns from all available sensor data, not just environmental sensors. ML is a universal feature and should be considered for any sensor type where sufficient data is available.

## Core ML Components

### ML Model Manager
- Model training and validation for all sensor types
- Feature engineering for all available data
- Model selection and hyperparameter tuning
- Performance monitoring and drift detection
- Model persistence and versioning

### Training Pipeline
- Should be generalized to support all sensor types
- Handles data preparation, training, validation, and inference

### Feature Engineering
- Temporal features (time of day, day of week, etc.)
- Sensor-specific features (raw values, deltas, rates of change)
- Cross-sensor features (correlations, combined states)

### Model Architecture
- **Primary Model**: Gradient Boosting (XGBoost/LightGBM) or similar
- **Alternative Models**: Other ML models as appropriate for the data

### Configuration Integration
- ML enable/disable option in config flow
- Training data requirements and thresholds
- Model performance settings (confidence threshold, update interval, etc.)
- Analysis method selection (ML/Deterministic/Hybrid)

### Data Collection and Preprocessing
- Historical data collection for all sensor types
- Data quality validation
- Training data preparation and caching
- Data cleanup and retention policies

### Real-time Inference
- ML model inference used in real-time probability calculations when enabled and sufficient data is available
- Fallback to deterministic analysis if ML is not available or not confident

### Error Handling
- Validate input data and model outputs
- Handle missing or invalid data gracefully
- Log model training and inference steps at debug level
- Provide fallback values and recovery mechanisms

### Testing Requirements
- Test model training with various data sizes
- Verify prediction accuracy and confidence
- Test model serialization and loading
- Validate incremental learning
- Test model performance monitoring
- Verify feature importance calculation
- Test overfitting prevention


## Implementation Plan

**Implementation roadmap for an ML “bolt-on” to *Area Occupancy Detection***

---

### 1  High-level architecture

| Layer                         | Responsibility                                                                                                                          | New / existing modules                                                     | Key interactions                                                                                                                  |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Data acquisition**          | Extract raw state & statistics rows from Recorder / Statistics API, window them and label with ground–truth occupancy (primary sensor). | `ml_data_collector.py` **(new)**                                           | Called by the coordinator on a scheduled job; re-uses the async Recorder queries already in `calculate_prior.py` for consistency  |
| **Feature engineering**       | Transform raw rows into feature matrix (temporal, deltas, cross-sensor, environmental enrichment).                                      | `ml_feature_engineering.py` **(new)** + extend `environmental_analysis.py` | Imported by both the trainer and real-time inference path; re-uses environmental rules                                            |
| **Model manager**             | Train, validate, persist, load, and drift-check ML models (primary LightGBM; fall back models supported).                               | `ml_models.py` *(file already scaffolded)*                                 | Exposes async API: `async_train`, `async_predict`, `async_check_drift`. Requirements match ML spec                                |
| **Hybrid probability engine** | Fuse Bayesian result from `ProbabilityCalculator`  with ML confidence, using configurable weighting / switch-over logic.                | `calculate_prob.py` (extend) + `ml_hybrid.py` **(new)**                    | If ML confidence ≥ threshold → trust ML; else blend ML & Bayes.                                                                   |
| **Storage & versioning**      | Persist model binaries, metadata, and feature schema in HA storage dir.                                                                 | extend `storage.py`                                                        | Uses same JSON helpers and migration hooks already present                                                                        |
| **Config / services**         | UI toggles for “Use ML”, “Retrain interval”, “Confidence threshold”; service `area_occupancy.train_model`.                              | extend `config_flow.py`, `services.yaml`, `service.py`                     | Mirrors pattern used for “Update Priors” service                                                                                  |
| **Tests**                     | 90 %+ coverage, incl. ML life-cycle & hybrid switching.                                                                                 | `tests/test_ml_models.py`, `test_hybrid.py`, etc.                          | Aligns with test-suite requirements                                                                                               |

---

### 2  Project structure additions

```
custom_components/area_occupancy/
    ml_data_collector.py        # historical extract, incremental windowing
    ml_feature_engineering.py   # pure functions -> features DataFrame
    ml_models.py                # already listed; fill out Manager class
    ml_hybrid.py                # Bayesian-ML fusion helpers
    environmental_analysis.py   # keep deterministic path, add shared transforms
```

Each file keeps a single responsibility and full typing, following the repository’s coding standards .

---

### 3  Module outlines (concise)

1. **`ml_data_collector.py`**

   ```python
   class TrainingSetBuilder:
       async def snapshot(
           hass: HomeAssistant,
           sensor_map: dict[str, EntityType],
           start: datetime,
           end: datetime,
       ) -> pandas.DataFrame: ...
   ```

   *Pulls recorder states, statistics, aligns to 1-minute buckets, labels with primary sensor state.*

2. **`ml_feature_engineering.py`**
   *Stateless helpers (`build_feature_matrix(df)`) that add:*

   * Δt-derivatives, rolling means, hour-of-day, weekday, seasonal flags.
   * Cross-sensor interactions (e.g. “motion active AND light on”).

3. **`ml_models.py`**

   ```python
   class ModelManager:
       async def async_train(self, X, y) -> ModelMeta: ...
       async def async_predict(self, X) -> tuple[np.ndarray, np.ndarray]: ...
       async def async_load(self) -> ModelMeta | None: ...
       async def async_check_drift(self, X_new) -> bool: ...
   ```

   *Uses LightGBM for tabular efficiency; persists via joblib into HA storage.*

4. **`ml_hybrid.py`**

   ```python
   def combine(bayes_prob: float, ml_prob: float, conf: float, mode: str) -> float
   ```

   *Implements “ML-only”, “Weighted”, “Bayes-only” strategies selected in options.*

5. **`calculate_prob.py` (patch)**

   * Call `ModelManager.async_predict()` inside `ProbabilityCalculator.calculate_occupancy_probability`.
   * Feed same `current_states` after feature transform.
   * Return both raw and hybrid probabilities.

---

### 4  Coordinator & scheduling

*In `coordinator.py`:*

```python
if self._ml_enabled and self._should_retrain():
    await self._model_manager.async_train(*await TrainingSetBuilder.snapshot(...))
```

* Retraining cadence configured (e.g. weekly) or on-demand via service.
* Inference runs every coordinator update; falls back gracefully when model unavailable or `confidence < threshold` .

---

### 5  Configuration & migrations

| Option                    | Type                                  | Default | Purpose                                      |
| ------------------------- | ------------------------------------- | ------- | -------------------------------------------- |
| `ml_enabled`              | bool                                  | false   | Toggle ML path                               |
| `ml_confidence_threshold` | float (0–1)                           | 0.7     | Minimum model confidence to trust prediction |
| `ml_retrain_interval`     | int (days)                            | 7       | Auto-retrain cadence                         |
| `analysis_method`         | enum(`deterministic`, `ml`, `hybrid`) | hybrid  | Runtime fusion mode                          |

Migration step adds these keys with sane defaults; uses `migrations.py` increment.

---

### 6  Testing strategy

* **Unit:** deterministic feature transforms, model training on synthetic data, drift detection paths.
* **Integration:** end-to-end prediction against mocked Recorder data.
* **Performance:** ensure trainer completes < 60 s on 30 days of 1-min data (approx. 40 k rows).
* **CI:** extend coverage target to include new ML modules; maintain ≥ 90 % global, 100 % on critical maths paths as mandated .

---

### 7  “Bolt-on” design principles

1. **No breaking changes:** Existing Bayesian flow continues to work when `ml_enabled=False`.
2. **Shared contracts:** Re-use `SensorInfo`, `PriorState`, and constants from `types.py`/`const.py` to avoid duplication.
3. **Loose coupling:** ML components interact through the coordinator and a minimal `ModelManager` API; core calculation modules stay testable without heavy ML dependencies.
4. **Progressive rollout:** Ship feature flagged; metrics logged at debug until accuracy verified.
5. **Graceful degradation:** Any ML exception triggers automatic fallback to Bayesian probability and outputs warning via custom `CalculationError`.

---

### 8  Next steps checklist

1. [ ] Draft `ml_data_collector.py` and write unit tests with fixture data.
2. [ ] Implement `ml_feature_engineering.py`; validate against test vectors.
3. [ ] Flesh out `ModelManager` with LightGBM backend; include joblib persistence.
4. [ ] Patch `calculate_prob.py` & add `ml_hybrid.py`; update sensor entities to expose hybrid probability.
5. [ ] Extend `config_flow.py` with new options, plus translation strings.
6. [ ] Add `train_model` service and retrain scheduler.
7. [ ] Write integration & performance tests; update CI config.
8. [ ] Update documentation and architecture diagrams.

