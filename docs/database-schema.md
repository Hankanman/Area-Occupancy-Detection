# Database Schema Documentation

## Overview

The Area Occupancy Detection integration uses a SQLite database to store sensor data, occupancy intervals, priors, and analytical data. The database is designed to support a single integration instance with multiple areas, enabling efficient cross-area analysis and long-term trend analysis.

**Database Version:** 5
**Database File:** `area_occupancy.db` (stored in `.storage/` directory)

## Architecture Principles

1. **Single Integration, Multiple Areas**: All areas share the same `entry_id`, but each area has a unique `area_name` as its primary identifier.
2. **Tiered Aggregation**: Raw data is kept for a short period, then aggregated into daily, weekly, and monthly summaries to prevent database bloat.
3. **Long-term Retention**: Aggregated data is retained for years to enable seasonal trend analysis.
4. **Cross-Area Support**: Relationships between areas and shared sensors are tracked for advanced probability calculations.

## Core Tables

### `areas`

Stores area configuration and metadata.

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | String | Integration entry ID (same for all areas) |
| `area_name` | String (PK) | Unique area identifier |
| `area_id` | String | Home Assistant area ID |
| `purpose` | String | Area purpose (e.g., "social", "work", "sleep") |
| `threshold` | Float | Occupancy probability threshold (0.0-1.0) |
| `adjacent_areas` | JSON | Array of adjacent area names |
| `created_at` | DateTime | Creation timestamp |
| `updated_at` | DateTime | Last update timestamp |

**Indexes:**
- Primary key on `area_name`
- Index on `entry_id`

**Relationships:**
- One-to-many with `entities`
- One-to-many with `priors`

### `entities`

Stores entity (sensor) configuration and Bayesian parameters.

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | String | Integration entry ID |
| `area_name` | String (PK, FK) | Area this entity belongs to |
| `entity_id` | String (PK) | Home Assistant entity ID |
| `entity_type` | String | Type (motion, media, appliance, etc.) |
| `weight` | Float | Bayesian weight (0.0-1.0) |
| `prob_given_true` | Float | P(entity active \| area occupied) |
| `prob_given_false` | Float | P(entity active \| area unoccupied) |
| `is_shared` | Boolean | Whether entity is shared across areas |
| `shared_with_areas` | JSON | Array of area names this entity is shared with |
| `last_updated` | DateTime | Last update timestamp |
| `created_at` | DateTime | Creation timestamp |
| `is_decaying` | Boolean | Whether probability is currently decaying |
| `decay_start` | DateTime | When decay started |
| `evidence` | Boolean | Current evidence state |

**Indexes:**
- Composite primary key on `(area_name, entity_id)`
- Indexes on `entry_id`, `area_name`, `entity_type`, `is_shared`
- Composite index on `(entry_id, area_name, entity_type)`

**Relationships:**
- Many-to-one with `areas`
- One-to-many with `intervals`

### `intervals`

Stores state change intervals for all sensors.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Area this interval belongs to |
| `entity_id` | String (FK) | Entity ID |
| `state` | String | Sensor state (e.g., "on", "off") |
| `start_time` | DateTime | Interval start time |
| `end_time` | DateTime | Interval end time |
| `duration_seconds` | Float | Interval duration |
| `aggregation_level` | String | Aggregation level: "raw", "daily", "weekly", "monthly" |
| `created_at` | DateTime | Creation timestamp |

**Indexes:**
- Primary key on `id`
- Unique constraint on `(entity_id, start_time, end_time, aggregation_level)`
- Indexes on `entry_id`, `area_name`, `entity_id`, `start_time`, `end_time`, `aggregation_level`
- Composite indexes for common query patterns

**Relationships:**
- Many-to-one with `entities`

**Retention Policy:**
- Raw intervals: 30 days
- Daily aggregates: 90 days
- Weekly aggregates: 365 days
- Monthly aggregates: 5 years

### `priors`

Stores time-slot priors (day of week × time slot).

| Column | Type | Description |
|--------|------|-------------|
| `entry_id` | String | Integration entry ID |
| `area_name` | String (PK, FK) | Area name |
| `day_of_week` | Integer (PK) | Day of week (0=Monday, 6=Sunday) |
| `time_slot` | Integer (PK) | Time slot (0-23 for hourly slots) |
| `prior_value` | Float | Prior probability for this slot |
| `data_points` | Integer | Number of data points used |
| `confidence` | Float | Confidence in the calculation (0.0-1.0) |
| `last_calculation_date` | DateTime | When prior was last calculated |
| `sample_period_start` | DateTime | Start of data period used |
| `sample_period_end` | DateTime | End of data period used |
| `calculation_method` | String | Method used (e.g., "interval_analysis") |
| `last_updated` | DateTime | Last update timestamp |

**Indexes:**
- Composite primary key on `(area_name, day_of_week, time_slot)`
- Indexes on `entry_id`, `area_name`, `(day_of_week, time_slot)`

**Relationships:**
- Many-to-one with `areas`

## New Tables for Advanced Features

### `interval_aggregates`

Stores aggregated interval statistics for efficient querying.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Area name |
| `entity_id` | String (FK) | Entity ID |
| `aggregation_period` | String | Period: "daily", "weekly", "monthly", "yearly" |
| `period_start` | DateTime | Period start time |
| `period_end` | DateTime | Period end time |
| `state` | String | Sensor state |
| `interval_count` | Integer | Number of intervals in period |
| `total_duration_seconds` | Float | Total duration of all intervals |
| `min_duration_seconds` | Float | Minimum interval duration |
| `max_duration_seconds` | Float | Maximum interval duration |
| `avg_duration_seconds` | Float | Average interval duration |
| `first_occurrence` | DateTime | First interval start in period |
| `last_occurrence` | DateTime | Last interval end in period |
| `created_at` | DateTime | Creation timestamp |

**Indexes:**
- Unique constraint on `(entity_id, aggregation_period, period_start, state)`
- Composite indexes for area-based and entity-based queries

**Purpose:** Enables fast queries for prior calculations and trend analysis without scanning raw intervals.

### `occupied_intervals_cache`

Stores precomputed occupied intervals for fast prior calculations.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Area name |
| `start_time` | DateTime | Interval start time |
| `end_time` | DateTime | Interval end time |
| `duration_seconds` | Float | Interval duration |
| `calculation_date` | DateTime | When interval was calculated |
| `data_source` | String | Source: "motion_sensors", "merged" |
| `created_at` | DateTime | Creation timestamp |

**Indexes:**
- Unique constraint on `(area_name, start_time, end_time)`
- Composite indexes for time-range queries

**Purpose:** Precomputed occupied intervals eliminate the need to recalculate from raw sensor data for each prior calculation.

### `global_priors`

Stores global prior values with calculation metadata and history. **This is the only source of truth for global priors** - the `areas` table no longer contains an `area_prior` field.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String (Unique) | Area name |
| `prior_value` | Float | Global prior probability |
| `calculation_date` | DateTime | When prior was calculated |
| `data_period_start` | DateTime | Start of data period used |
| `data_period_end` | DateTime | End of data period used |
| `total_occupied_seconds` | Float | Total occupied time in period |
| `total_period_seconds` | Float | Total period duration |
| `interval_count` | Integer | Number of intervals used |
| `confidence` | Float | Confidence in calculation (0.0-1.0) |
| `calculation_method` | String | Method used |
| `underlying_data_hash` | String | Hash of underlying data (for validation) |
| `created_at` | DateTime | Creation timestamp |
| `updated_at` | DateTime | Last update timestamp |

**Indexes:**
- Unique constraint on `area_name`
- Index on `calculation_date`

**Purpose:** Stores the global prior for each area with full history and metadata. Only the most recent calculation is kept per area (older calculations are pruned).

**Retention:** Last 15 calculations per area are retained.

### `numeric_samples`

Stores raw numeric sensor samples for correlation analysis.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Area name |
| `entity_id` | String (FK) | Entity ID |
| `timestamp` | DateTime | Sample timestamp |
| `value` | Float | Numeric value |
| `unit_of_measurement` | String | Unit (e.g., "°C", "%") |
| `state` | String | Associated state (if any) |
| `created_at` | DateTime | Creation timestamp |

**Indexes:**
- Unique constraint on `(entity_id, timestamp)`
- Composite indexes for time-range queries

**Retention:** 14 days of raw samples.

**Purpose:** Raw samples are used to calculate correlations with occupancy and then aggregated.

### `numeric_aggregates`

Stores aggregated numeric sensor data for trend analysis.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Area name |
| `entity_id` | String (FK) | Entity ID |
| `aggregation_period` | String | Period: "hourly", "daily", "weekly", "monthly", "yearly" |
| `period_start` | DateTime | Period start time |
| `period_end` | DateTime | Period end time |
| `min_value` | Float | Minimum value in period |
| `max_value` | Float | Maximum value in period |
| `avg_value` | Float | Average value in period |
| `median_value` | Float | Median value in period |
| `sample_count` | Integer | Number of samples in period |
| `first_value` | Float | First value in period |
| `last_value` | Float | Last value in period |
| `std_deviation` | Float | Standard deviation |
| `created_at` | DateTime | Creation timestamp |

**Indexes:**
- Unique constraint on `(entity_id, aggregation_period, period_start)`
- Composite indexes for area-based and entity-based queries

**Retention:**
- Hourly aggregates: 30 days
- Weekly aggregates: 3 years (for seasonal analysis)

**Purpose:** Enables trend analysis across seasons (e.g., temperature differences between winter and summer).

### `numeric_correlations`

Stores calculated correlations between numeric sensor values and occupancy.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Area name |
| `entity_id` | String (FK) | Entity ID |
| `correlation_coefficient` | Float | Pearson correlation (-1.0 to 1.0) |
| `correlation_type` | String | Type: "occupancy_positive", "occupancy_negative", "none" |
| `analysis_period_start` | DateTime | Start of analysis period |
| `analysis_period_end` | DateTime | End of analysis period |
| `sample_count` | Integer | Number of samples used |
| `confidence` | Float | Confidence in correlation (0.0-1.0) |
| `mean_value_when_occupied` | Float | Mean value when area is occupied |
| `mean_value_when_unoccupied` | Float | Mean value when area is unoccupied |
| `std_dev_when_occupied` | Float | Standard deviation when occupied |
| `std_dev_when_unoccupied` | Float | Standard deviation when unoccupied |
| `threshold_active` | Float | Threshold for active state |
| `threshold_inactive` | Float | Threshold for inactive state |
| `calculation_date` | DateTime | When correlation was calculated |
| `created_at` | DateTime | Creation timestamp |
| `updated_at` | DateTime | Last update timestamp |

**Indexes:**
- Unique constraint on `(area_name, entity_id, analysis_period_start)`
- Composite indexes for querying by correlation type and confidence

**Retention:** Last 10 correlation analyses per sensor are retained.

**Purpose:** Identifies which numeric sensors (temperature, humidity, illuminance, CO2, sound pressure, atmospheric pressure, air quality, VOC, PM2.5, PM10, energy, etc.) correlate with occupancy, enabling them to be used as occupancy indicators.

### `entity_statistics`

Stores per-entity operational and Bayesian statistics.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Area name |
| `entity_id` | String (FK) | Entity ID |
| `statistic_type` | String | Type: "operational" or "bayesian" |
| `statistic_name` | String | Name (e.g., "total_activations", "prob_given_true") |
| `statistic_value` | Float | Statistic value |
| `period_start` | DateTime | Period start time |
| `period_end` | DateTime | Period end time |
| `updated_at` | DateTime | Last update timestamp |

**Indexes:**
- Unique constraint on `(entity_id, statistic_type, statistic_name, period_start)`
- Composite indexes for area-based and entity-based queries

**Purpose:** Tracks operational statistics (counts, durations, frequencies) and Bayesian parameters (probabilities, weights) over time.

### `area_relationships`

Defines and tracks relationships between areas.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `area_name` | String | Source area name |
| `related_area_name` | String | Related/adjacent area name |
| `relationship_type` | String | Type: "adjacent", "shared_wall", "shared_entrance", etc. |
| `influence_weight` | Float | Influence weight (0.0-1.0) |
| `distance` | Float | Physical distance (if applicable) |
| `created_at` | DateTime | Creation timestamp |
| `updated_at` | DateTime | Last update timestamp |

**Indexes:**
- Unique constraint on `(area_name, related_area_name)`
- Indexes for bidirectional queries

**Purpose:** Tracks which areas are adjacent or related, enabling cross-area probability adjustments. The `influence_weight` determines how much one area's occupancy affects another's probability.

### `cross_area_stats`

Stores aggregated statistics that span multiple areas.

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer (PK) | Auto-increment primary key |
| `entry_id` | String | Integration entry ID |
| `statistic_type` | String | Type (e.g., "combined_occupancy", "shared_sensor_active") |
| `statistic_name` | String | Statistic name |
| `involved_areas` | JSON | Array of area names involved |
| `aggregation_period` | String | Period: "hourly", "daily", "weekly", "monthly" |
| `period_start` | DateTime | Period start time |
| `period_end` | DateTime | Period end time |
| `statistic_value` | Float | Statistic value |
| `extra_metadata` | JSON | Additional metadata |
| `created_at` | DateTime | Creation timestamp |

**Indexes:**
- Unique constraint on `(statistic_type, statistic_name, aggregation_period, period_start)`
- Composite indexes for type and period queries

**Purpose:** Enables analysis of patterns across multiple areas, such as combined occupancy or shared sensor activity.

### `metadata`

Stores database metadata (version, last prune time, etc.).

| Column | Type | Description |
|--------|------|-------------|
| `key` | String (PK) | Metadata key |
| `value` | String | Metadata value |

**Common Keys:**
- `db_version`: Database schema version
- `last_prune_time`: Timestamp of last interval prune operation

## Data Flow

### Interval Processing

1. **Raw Intervals**: State changes from Home Assistant recorder are converted to intervals and stored in `intervals` table with `aggregation_level="raw"`.
2. **Aggregation**: Periodically, raw intervals are aggregated:
   - Raw → Daily: After 30 days
   - Daily → Weekly: After 90 days
   - Weekly → Monthly: After 365 days
   - Monthly aggregates are retained indefinitely
3. **Occupied Intervals Cache**: Raw intervals are processed to create precomputed occupied intervals stored in `occupied_intervals_cache`.

### Prior Calculation

1. **Time-Slot Priors**: Calculated from `occupied_intervals_cache` and stored in `priors` table.
2. **Global Priors**: Calculated from total occupied time and stored in `global_priors` table with full metadata.

### Correlation Analysis

1. **Numeric Samples**: Raw numeric sensor values are stored in `numeric_samples`.
2. **Aggregation**: Samples are aggregated into `numeric_aggregates` for trend analysis.
3. **Correlation**: Samples are correlated with occupancy intervals to calculate correlations stored in `numeric_correlations`.

## Retention Policies

| Data Type | Retention Period | Aggregation |
|-----------|------------------|-------------|
| Raw intervals | 30 days | None |
| Raw numeric samples | 14 days | None |
| Daily interval aggregates | 90 days | From raw |
| Weekly interval aggregates | 365 days | From daily |
| Monthly interval aggregates | 5 years | From weekly |
| Hourly numeric aggregates | 30 days | From raw samples |
| Weekly numeric aggregates | 3 years | From hourly |
| Global priors | Last 15 calculations | N/A |
| Numeric correlations | Last 10 per sensor | N/A |

## Indexes and Performance

The database uses extensive indexing to optimize common query patterns:

- **Time-range queries**: Indexes on `start_time`, `end_time`, `timestamp`
- **Area-based queries**: Indexes on `area_name` in all relevant tables
- **Entity-based queries**: Indexes on `entity_id` in all relevant tables
- **Composite indexes**: Optimize multi-column queries (e.g., area + time range)

## Migration Strategy

When the database schema version changes:

1. Database version is checked on startup
2. If version mismatch detected, database is deleted and recreated with new schema
3. All previous data is cleared (no migration scripts for major version changes)

This approach is used for DB_VERSION 5+ due to the fundamental architectural change from multiple integrations to a single integration with multiple areas.

