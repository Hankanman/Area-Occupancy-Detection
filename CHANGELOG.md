# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]


## [2025.12.4] - 2025-12-29

This version stabilizes the timezone normalization changes introduced in the pre-releases and includes several important improvements:

- **Improved timezone handling** ensures accurate occupancy detection across different time zones, fixing issues with timestamp normalization that could affect data accuracy. All datetime operations now consistently use UTC internally with proper conversion to local time for display and analysis. Fixes [#308](https://github.com/Hankanman/Area-Occupancy-Detection/issues/308), [#301](https://github.com/Hankanman/Area-Occupancy-Detection/issues/301)
- **Enhanced migration logic** prevents unnecessary migrations for fresh installations, making setup smoother for new users. The migration system now intelligently detects fresh entries with the new format (CONF_AREAS) and skips migration steps that aren't needed, preventing false migration triggers. Fixes [#306](https://github.com/Hankanman/Area-Occupancy-Detection/discussions/306)
- **Added support for additional sensor types** including VOLATILE_ORGANIC_COMPOUNDS_PARTS, expanding the range of environmental sensors that can be used for occupancy detection. Fixes [#305](https://github.com/Hankanman/Area-Occupancy-Detection/issues/305)
- **Improved interval caching** with validation and deduplication for better performance and reliability. Invalid intervals (where start time is greater than end time) are now automatically skipped, and duplicate intervals are prevented from being stored. Fixes [#301](https://github.com/Hankanman/Area-Occupancy-Detection/issues/301)
- **Updated user interface text** and descriptions for clearer configuration options, making it easier to understand door and wasp-in-box sensor settings. Thank you @simon5738
- **Updated state monitoring** to allow unavailable or unknown entities to be picked up when they become available. Fixes [#310](https://github.com/Hankanman/Area-Occupancy-Detection/issues/310), [#285](https://github.com/Hankanman/Area-Occupancy-Detection/issues/285)
- **Refactored binary sensor state mapping** to improve consistency and maintainability across the integration. The mapping logic has been extracted to a centralized utility function (`map_binary_state_to_semantic`) that converts binary sensor states ('on'/'off') to semantic states ('open'/'closed') based on active states. [#299](https://github.com/Hankanman/Area-Occupancy-Detection/issues/299)

## ‼️Potential Breaking Changes

The following applies _**if**_ you are upgrading from a version _**before**_ 2025.12.4.

> [!CAUTION]
> **Database Reset Required**: This version includes fundamental changes to how timestamps are handled throughout the integration. The database schema version has been incremented (CONF_VERSION 15 → 16) to support timezone normalization and local bucketing. **Your `area_occupancy.db` file will be automatically deleted and recreated with the new schema.** All historical data (intervals, priors, correlations) will be cleared and will need to be rebuilt from Home Assistant's Recorder history.

> [!NOTE]
> The database reset is necessary because the new timezone normalization system requires consistent UTC storage with proper timezone-aware datetime handling. Existing data stored with the old timestamp format cannot be safely migrated. The integration will automatically rebuild historical data from your Home Assistant Recorder over the next few hours/days.

> [!WARNING]
> If you rely on historical occupancy data for automations or analysis, be aware that this data will be lost during the upgrade. The integration will rebuild this data automatically, but it may take some time depending on how much history is available in your Recorder.

## What's Changed
* Implement timezone normalization and local bucketing utilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/304
* Update version to 2025.12.4-pre2 and dependencies in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/314
* Improving user interface text by @simon5738 in https://github.com/Hankanman/Area-Occupancy-Detection/pull/307
* Add validation and deduplication for occupied interval caching by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/316
* Extract binary sensor state mapping utility to utils.py by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/317
* Bump pre-commit from 4.5.0 to 4.5.1 in the all-pip-dependencies group across 1 directory by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/315
* Bump actions/upload-artifact from 5 to 6 in the all-github-actions group by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/309
* Add support for VOLATILE_ORGANIC_COMPOUNDS_PARTS by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/318
* Fix issue where new entries named wrong by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/320
* Update versioning for Area Occupancy Detection integration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/321
* Refactor timezone handling with UTC storage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/322

## New Contributors
* @simon5738 made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/307

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.12.3...2025.12.4


## [2025.12.4-pre2] - 2025-12-23

Release to test fix and accuracy of a sweeping change to how we handle timestamps across the integration in an attempt to correctly normalise

> [!CAUTION]
> This pre-release will delete you area_occupancy.db file and re-create it, test with caution.

> [!NOTE]
> You can help by testing this [pre-release](https://hankanman.github.io/Area-Occupancy-Detection/technical/prerelease/) and submitting [issues](https://github.com/Hankanman/Area-Occupancy-Detection/issues), please include [debug logs](https://hankanman.github.io/Area-Occupancy-Detection/technical/debug/)

> [!WARNING]
> This is a pre-release and may include unknown breaking changes!

## What's Changed
* Implement timezone normalization and local bucketing utilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/304
* Update version to 2025.12.4-pre2 and dependencies in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/314
* Improving user interface text by @simon5738 in https://github.com/Hankanman/Area-Occupancy-Detection/pull/307
* Add validation and deduplication for occupied interval caching by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/316
* Extract binary sensor state mapping utility to utils.py by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/317
* Bump pre-commit from 4.5.0 to 4.5.1 in the all-pip-dependencies group across 1 directory by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/315
* Bump actions/upload-artifact from 5 to 6 in the all-github-actions group by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/309
* Add support for VOLATILE_ORGANIC_COMPOUNDS_PARTS by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/318

## New Contributors
* @simon5738 made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/307

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.12.3...2025.12.4-pre2


## [2025.12.4-pre1] - 2025-12-12

Release to test fix and accuracy of a sweeping change to how we handle timestamps across the integration in an attempt to correctly normalise

> [!CAUTION]
> This pre-release will delete you area_occupancy.db file and re-create it, test with caution.

> [!NOTE]
> You can help by testing this [pre-release](https://hankanman.github.io/Area-Occupancy-Detection/technical/prerelease/) and submitting [issues](https://github.com/Hankanman/Area-Occupancy-Detection/issues), please include [debug logs](https://hankanman.github.io/Area-Occupancy-Detection/technical/debug/)

> [!WARNING]
> This is a pre-release and may include unknown breaking changes!

## What's Changed
* Implement timezone normalization and local bucketing utilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/304


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.12.3...2025.12.4-pre1


## [2025.12.3] - 2025-12-12

This version adds:
- Carbon Monoxide sensors to the config
- New Area Purposes: Garden, Garage, Driveway
- Renamed frontend strings for some purposes so they better align with room type decriptions
- Fixed a divide by zero bug

Thank you to @Canis-L-Sapien for the first pull-request contributing to the project!

## ‼️Potential Breaking Changes

The following applies _**if**_ you are upgrading from a version _**before**_ 2025.12.0.

> [!WARNING]
> There is a lot of core changes, everything __should__ migrate automatically, please read the below before updating. Submit an [issue](https://github.com/Hankanman/Area-Occupancy-Detection/issues) if you have problems.

> [!CAUTION]
> If you have renamed entities and rely on them for automations, this update will break those automations, this is due to a change in how Unique IDs are generated for the entities to align with Home Assistant best practice.

> [!CAUTION]
> When updating AOD will attempt to match existing configured AOD areas to real HA Areas, If you do not have Home Assistant Areas that match names of the AOD areas, new Home Assistant areas will be created for you to allow the migration to complete. You can correct the AOD areas following migration if need be reconfiguring AOD areas and assigning the correct HA Area

## What's Changed
* Refactor dependency management and update workflows by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/290
* Update Atmospheric pressure configuration to include SensorDeviceClass.ATMOSPHERIC_PRESSURE by @Canis-L-Sapien in https://github.com/Hankanman/Area-Occupancy-Detection/pull/289
* Migrate to Ruff linting and consolidate documentation by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/293
* Add CO sensor support and update related configurations by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/294
* Update version to 2025.12.3 in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/295
* Bump astral-sh/setup-uv from 5 to 7 in the all-github-actions group by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/291
* Bump pytest-homeassistant-custom-component from 0.13.299 to 0.13.300 in the all-pip-dependencies group by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/292
* Update dependencies and add build configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/297
* 2025.12.3 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/296
* Bump actions/cache from 4 to 5 in the all-github-actions group by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/302
* Fix divide by zero and improve DB tests by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/303

## New Contributors
* @Canis-L-Sapien made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/289

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.12.2...2025.12.3


## [2025.12.2] - 2025-12-07

Minor release to address Energy vs Power oversight as noted in issue [#279](https://github.com/Hankanman/Area-Occupancy-Detection/issues/279)

## ‼️Potential Breaking Changes

The following applies _**if**_ you are upgrading from a version _**before**_ 2025.12.0.

> [!WARNING]
> There is a lot of core changes, everything __should__ migrate automatically, please read the below before updating. Submit an [issue](https://github.com/Hankanman/Area-Occupancy-Detection/issues) if you have problems.

> [!CAUTION]
> If you have renamed entities and rely on them for automations, this update will break those automations, this is due to a change in how Unique IDs are generated for the entities to align with Home Assistant best practice.

> [!CAUTION]
> When updating AOD will attempt to match existing configured AOD areas to real HA Areas, If you do not have Home Assistant Areas that match names of the AOD areas, new Home Assistant areas will be created for you to allow the migration to complete. You can correct the AOD areas following migration if need be reconfiguring AOD areas and assigning the correct HA Area

## What's Changed
* Update README.md by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/281
* Update version to 2025.12.2 and migrate energy sensors to power sensors by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/282
* 2025.12.2 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/286


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.12.1...2025.12.2


## [2025.12.1] - 2025-12-06

Minor update to add some new purposes to match more areas, updated the documentation to include clearer setup and config instructions with images.

## 🎉 New & Updated Features

* Added three area purposes (Garage, Garden, Driveway) and renamed several purposes for clearer labels
* Decaying-entity listings now expose a half-life value
* Analysis service responses now include execution time (ms) and integration software version
* New Purpose and Sensors pages; streamlined Installation and Configuration guides; updated usage and examples

## ‼️Potential Breaking Changes

The following applies **if** you are upgrading from a version **before** 2025.12.0.

> [!WARNING]
> There is a lot of core changes, everything __should__ migrate automatically, please read the below before updating. Submit an [issue](https://github.com/Hankanman/Area-Occupancy-Detection/issues) if you have problems.

> [!CAUTION]
> If you have renamed entities and rely on them for automations, this update will break those automations, this is due to a change in how Unique IDs are generated for the entities to align with Home Assistant best practice.

> [!CAUTION]
> When updating AOD will attempt to match existing configured AOD areas to real HA Areas, If you do not have Home Assistant Areas that match names of the AOD areas, new Home Assistant areas will be created for you to allow the migration to complete. You can correct the AOD areas following migration if need be reconfiguring AOD areas and assigning the correct HA Area

## 📄 What’s Changed
* Update dependencies and improve installation scripts by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/272
* Expand AreaPurpose enum and update to human-readable names by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/273
* Remove markdownlint and refactor tests with registry area IDs by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/274
* Add half-life attribute to decay entities in AreaOccupancySensor by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/275
* Enhance area occupancy analysis service with performance metrics by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/276
* Update version numbers to 2025.12.1 in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/277
* 2025.12.1 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/278


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.12.0...2025.12.1


## [2025.12.0] - 2025-12-03

The **biggest release** of the integration to date.
This version introduces a brand-new architecture for areas, a real statistical engine for numeric sensors, a redesigned configuration experience, and a fully interactive **probability simulator** to help users understand exactly how the system behaves.

Thank you again to everyone starring, sharing, and testing the project — the growth has been incredible since the last release we somehow have almost doubled our stars! This release reflects weeks of work driven directly by your feedback.

---

# ‼️Potential Breaking Changes

The following applies if you are upgrading from a version before 2025.12.0.

> [!WARNING]
> There is a lot of core changes, everything __should__ migrate automatically, please read the below before updating. Submit an [issue](https://github.com/Hankanman/Area-Occupancy-Detection/issues) if you have problems.

> [!CAUTION]
> If you have renamed entities and rely on them for automations, this update will break those automations, this is due to a change in how Unique IDs are generated for the entities to align with Home Assistant best practice.

> [!CAUTION]
> When updating AOD will attempt to match existing configured AOD areas to real HA Areas, If you do not have Home Assistant Areas that match names of the AOD areas, new Home Assistant areas will be created for you to allow the migration to complete. You can correct the AOD areas following migration if need be reconfiguring AOD areas and assigning the correct HA Area

---

# 🎉 New & Updated Features

## 🎚️ **Interactive Probability Simulator**

A brand-new **visual simulator** is now available on the documentation site.

[Try it out here](https://hankanman.github.io/Area-Occupancy-Detection/simulator/)

It allows you to flip your entities on and off to see the effect without having to run around you house to trigger things

You can now experiment with:

* Priors
* Half-life decay
* Motion P(True|Occupied) settings
* Weights
* Continuous likelihood curves
* Combined probability output

The simulator uses the **real integration code** and therefore behaves exactly like the real integration. I built it to help myself better visually see what is going on and i think it will be of great help to you guys too!

It's designed to help you understand *why* a room is being marked occupied or not - without needing to guess or dig through logs.
A huge step forward for transparency and ease of tuning.

> [!NOTE]
> A server runs to allow the real code to be executed. You therefore send data to that server to process, Nothing is stored, just calculated and sent back. The data you send is from the `Run Analysis` service of the integration, you can inspect exactly what you are sending before you load the simulation.

---

## 🌐 **Multi-Area Architecture**

The integration now supports **multiple areas under a single configuration entry**, each with its own sensors, priors, decay, and analysis.

* Independent occupancy entities per area
* Native Home Assistant **Area Registry** support
* New *Area Device Handle* system for stable unique IDs

This is a foundational redesign that enables _future_ features like room-to-room influence, directional flows, and graph-based occupancy models.

---

## 📈 **Real Numeric Sensor Intelligence**

Numeric sensors are no longer static inputs — they now participate in *real statistical analysis*.

### New analytic capabilities:

* **Pearson correlation analysis** against motion ground-truth
* **Gaussian continuous likelihoods** (full PDF-based probability)
* **Active/inactive range learning**
* **Rejection reasons** explaining why a sensor is excluded
* **Environmental pattern learning** across days, weeks, and months

Supported inputs now include:

* Temperature
* Humidity
* CO₂ | [#51](https://github.com/Hankanman/Area-Occupancy-Detection/issues/51)
* VOC
* PM2.5 / PM10
* Barometric pressure
* Energy consumption | [#41](https://github.com/Hankanman/Area-Occupancy-Detection/issues/41)
* Sound pressure | [#52](https://github.com/Hankanman/Area-Occupancy-Detection/issues/52)
* Air quality

This dramatically improves detection accuracy in low-motion environments such as bedrooms, offices, and lounges.

---

## ⚙️ **Redesigned Configuration Flow**

The setup UI has been rebuilt from scratch:

* Add/manage/remove areas in a guided multi-step flow
* New hybrid menu-based interaction
* Per-area configuration for motion, numeric, doors/windows, appliances
* Minimum-prior override
* Time-of-day prior bounds
* Global sleep schedule | [#83](https://github.com/Hankanman/Area-Occupancy-Detection/issues/83)
* Combined windows/doors configuration
* Automatic preservation of existing global options

The new flow is faster, clearer, and much easier to use.

---

## 🚪 **Wasp-in-Box: Improved Accuracy & New Features**

The Wasp in Box system now benefits from the new multi-area model and includes improved handling of:

* Immediate vacancy for WASP entities
* Cleaner decay logic
* Better restoration and state transitions
* Defensive cleanup for shutdown/reloads

The docs now include updated diagrams and examples aligned with the new architecture.

---

# 📊 Performance & Database Improvements

## 🔄 New Aggregation & Correlation Engine

The analysis pipeline has been restructured:

* Tiered aggregation of raw → daily → weekly → monthly data
* Faster correlation calculations
* Interval caching for huge performance gains
* Batch inserts and chunked DB access
* Safer retention & pruning across all data types

Combined with smarter session handling and async offloading, the integration now places *considerably less* load on Home Assistant.

---

## 🧠 Smarter Priors & Time Modelling

* New **TIME_PRIOR_MIN_BOUND** and **TIME_PRIOR_MAX_BOUND**
* New **DEFAULT_TIME_PRIOR**
* Support for **minimum prior override**
* Updated half-life defaults for all area purposes
* Fully UTC-consistent time handling

These improvements stabilise behaviour when data is sparse and eliminate spikes or drops due to edge-case timestamps.

---

## 🧹 Entity, Device & Migration Improvements

* Automatic backup of older DB versions before migration
* Full migration path from single → multi-area
* Normalisation of area names & unique IDs
* Orphaned entity and interval cleanup
* Improved cross-area isolation in SQL
* Lowercase unique ID conversion
* Robust recovery logic for corrupted data

This release significantly reduces “mysterious states” or stale entries during updates.

---

# 🐞 Bug Fixes (Highlights)

* Hundreds of corrections across interval boundaries, decays, ranges, and timezone edge cases
* Improved error handling and rollback conditions
* Fixes for duplicated or missing numeric correlation samples
* More resilient handling of invalid or NaN sensor values
* Correct behaviour when no sensors exist for an area
* Fixes for early returns in state sync
* Dozens of safety guards added across DB and analysis flows

A huge amount of stability work has gone into this version.

---

# 📄 What’s Changed

* Multi-Device Architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/189
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/206
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/210
* Next by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/190
* Setup standard Area Class and simplify Helper Classes by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/212
* Move heavy analysis functions to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/213
* Add testing guide and refactor test setup for multi-area architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/214
* Restructure Area device properties to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/215
* Create All Areas Device by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/216
* Enhance area management in config flow by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/217
* Add Home Asssitant Area Association by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/218
* Refactor cleanup processes to enhance memory management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/219
* Refactor entity type filtering in AreaOccupancyDB and PriorAnalyzer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/220
* Fix prior persistence semantics by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/221
* Refactor DB with new schema by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/222
* Remove legacy support for area configurations and enhance error handling by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/223
* Refactor area occupancy coordinator and service logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/224
* Bump uv from 0.9.6 to 0.9.7 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/203
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/202
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/201
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/200
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/199
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/229
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/228
* Bump uv from 0.9.6 to 0.9.10 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/227
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/225
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/226
* 2025.11.3 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/211
* Consolidate database aggregation to SQL-only path by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/231
* Configure motion sensor probabilities and refine Bayesian logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/232
* Replace occupancy sensor with motion sensors and consolidate configs by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/239
* Add environmental sensor support to area occupancy component by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/240
* Add energy sensors to configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/241
* Add migration logic for multi-area entry conversion by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/242
* Update version to 2025.11.3-pre5 in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/243
* Refactor database reset logic in migration process by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/244
* Update version to 2025.11.3-pre7 and enhance migration logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/245
* Refactor time-prior calculation to use Python instead of SQL by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/246
* Bump version to 2025.11.3-pre8 and refactor test infrastructure by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/247
* Add time-of-day-aware decay and sleep schedule configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/248
* Add numeric sensor correlation analysis for occupancy detection by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/249
* Refactor analysis workflow and rename error fields by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/250
* Simplify prior calculation to motion/presence sensors only by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/251
* Refactor analysis: remove likelihood, add PriorAnalyzer class by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/252
* Add binary sensor analysis with duration-based probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/253
* Remove filelock and refactor database abstraction layer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/255
* Refactor correlations subsystem with binary analysis support by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/256
* Bump version and refactor area validation with decay enhancements by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/257
* Release 2025.11.3: Refactor decay defaults and correlations by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/258
* Refactor simulator UI with async operations and multi-area support by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/260
* Bump version to 2025.12.0 and refactor correlation analysis by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/261
* 2025.12.0 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/254
* Bump pre-commit from 4.3.0 to 4.5.0 in the all-pip-dependencies group by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/262
* Bump actions/checkout from 5 to 6 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/233
* Add time-based prior learning for occupancy detection by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/264
* Add numeric aggregation pathway and extend correlation analysis by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/265
* Restructure prior calculation flow with immediate database persistence by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/266
* Add Docker containerization infrastructure by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/267
* Add aggregation pipelines and Docker containerization for simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/268
* Refactor database API layer and expand test coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/269
* Bug Fixes and Test Updates by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/270


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.12.0


## [2025.11.3-pre10] - 2025-11-27

## What's Changed
* Multi-Device Architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/189
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/206
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/210
* Next by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/190
* Setup standard Area Class and simplify Helper Classes by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/212
* Move heavy analysis functions to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/213
* Add testing guide and refactor test setup for multi-area architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/214
* Restructure Area device properties to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/215
* Create All Areas Device by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/216
* Enhance area management in config flow by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/217
* Add Home Asssitant Area Association by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/218
* Refactor cleanup processes to enhance memory management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/219
* Refactor entity type filtering in AreaOccupancyDB and PriorAnalyzer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/220
* Fix prior persistence semantics by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/221
* Refactor DB with new schema by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/222
* Remove legacy support for area configurations and enhance error handling by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/223
* Refactor area occupancy coordinator and service logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/224
* Bump uv from 0.9.6 to 0.9.7 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/203
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/202
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/201
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/200
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/199
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/229
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/228
* Bump uv from 0.9.6 to 0.9.10 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/227
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/225
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/226
* 2025.11.3 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/211
* Consolidate database aggregation to SQL-only path by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/231
* Configure motion sensor probabilities and refine Bayesian logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/232
* Replace occupancy sensor with motion sensors and consolidate configs by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/239
* Add environmental sensor support to area occupancy component by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/240
* Add energy sensors to configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/241
* Add migration logic for multi-area entry conversion by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/242
* Update version to 2025.11.3-pre5 in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/243
* Refactor database reset logic in migration process by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/244
* Update version to 2025.11.3-pre7 and enhance migration logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/245
* Refactor time-prior calculation to use Python instead of SQL by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/246
* Bump version to 2025.11.3-pre8 and refactor test infrastructure by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/247
* Add time-of-day-aware decay and sleep schedule configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/248
* Add numeric sensor correlation analysis for occupancy detection by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/249
* Refactor analysis workflow and rename error fields by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/250
* Simplify prior calculation to motion/presence sensors only by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/251
* Refactor analysis: remove likelihood, add PriorAnalyzer class by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/252
* Add binary sensor analysis with duration-based probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/253
* Remove filelock and refactor database abstraction layer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/255
* Refactor correlations subsystem with binary analysis support by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/256
* Bump version and refactor area validation with decay enhancements by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/257


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.11.3-pre10


## [2025.11.3-pre9] - 2025-11-25

## What's Changed
* Multi-Device Architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/189
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/206
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/210
* Next by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/190
* Setup standard Area Class and simplify Helper Classes by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/212
* Move heavy analysis functions to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/213
* Add testing guide and refactor test setup for multi-area architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/214
* Restructure Area device properties to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/215
* Create All Areas Device by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/216
* Enhance area management in config flow by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/217
* Add Home Asssitant Area Association by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/218
* Refactor cleanup processes to enhance memory management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/219
* Refactor entity type filtering in AreaOccupancyDB and PriorAnalyzer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/220
* Fix prior persistence semantics by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/221
* Refactor DB with new schema by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/222
* Remove legacy support for area configurations and enhance error handling by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/223
* Refactor area occupancy coordinator and service logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/224
* Bump uv from 0.9.6 to 0.9.7 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/203
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/202
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/201
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/200
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/199
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/229
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/228
* Bump uv from 0.9.6 to 0.9.10 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/227
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/225
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/226
* 2025.11.3 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/211
* Consolidate database aggregation to SQL-only path by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/231
* Configure motion sensor probabilities and refine Bayesian logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/232
* Replace occupancy sensor with motion sensors and consolidate configs by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/239
* Add environmental sensor support to area occupancy component by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/240
* Add energy sensors to configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/241
* Add migration logic for multi-area entry conversion by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/242
* Update version to 2025.11.3-pre5 in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/243
* Refactor database reset logic in migration process by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/244
* Update version to 2025.11.3-pre7 and enhance migration logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/245
* Refactor time-prior calculation to use Python instead of SQL by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/246
* Bump version to 2025.11.3-pre8 and refactor test infrastructure by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/247
* Add time-of-day-aware decay and sleep schedule configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/248
* Add numeric sensor correlation analysis for occupancy detection by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/249
* Refactor analysis workflow and rename error fields by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/250
* Simplify prior calculation to motion/presence sensors only by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/251
* Refactor analysis: remove likelihood, add PriorAnalyzer class by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/252
* Add binary sensor analysis with duration-based probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/253


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.11.3-pre9


## [2025.11.3-pre8] - 2025-11-22

## What's Changed
* Multi-Device Architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/189
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/206
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/210
* Next by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/190
* Setup standard Area Class and simplify Helper Classes by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/212
* Move heavy analysis functions to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/213
* Add testing guide and refactor test setup for multi-area architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/214
* Restructure Area device properties to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/215
* Create All Areas Device by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/216
* Enhance area management in config flow by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/217
* Add Home Asssitant Area Association by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/218
* Refactor cleanup processes to enhance memory management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/219
* Refactor entity type filtering in AreaOccupancyDB and PriorAnalyzer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/220
* Fix prior persistence semantics by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/221
* Refactor DB with new schema by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/222
* Remove legacy support for area configurations and enhance error handling by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/223
* Refactor area occupancy coordinator and service logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/224
* Bump uv from 0.9.6 to 0.9.7 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/203
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/202
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/201
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/200
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/199
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/229
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/228
* Bump uv from 0.9.6 to 0.9.10 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/227
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/225
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/226
* 2025.11.3 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/211
* Consolidate database aggregation to SQL-only path by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/231
* Configure motion sensor probabilities and refine Bayesian logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/232
* Replace occupancy sensor with motion sensors and consolidate configs by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/239
* Add environmental sensor support to area occupancy component by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/240
* Add energy sensors to configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/241
* Add migration logic for multi-area entry conversion by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/242
* Update version to 2025.11.3-pre5 in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/243
* Refactor database reset logic in migration process by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/244
* Update version to 2025.11.3-pre7 and enhance migration logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/245
* Refactor time-prior calculation to use Python instead of SQL by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/246
* Bump version to 2025.11.3-pre8 and refactor test infrastructure by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/247


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.11.3-pre8


## [2025.11.3-pre7] - 2025-11-21

## What's Changed
* Update version to 2025.11.3-pre7 and enhance migration logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/245


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.3-pre6...2025.11.3-pre7


## [2025.11.3-pre6] - 2025-11-21

## What's Changed
* Refactor database reset logic in migration process by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/244


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.3-pre5...2025.11.3-pre6


## [2025.11.3-pre5] - 2025-11-21

## What's Changed
* Multi-Device Architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/189
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/206
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/210
* Next by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/190
* Setup standard Area Class and simplify Helper Classes by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/212
* Move heavy analysis functions to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/213
* Add testing guide and refactor test setup for multi-area architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/214
* Restructure Area device properties to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/215
* Create All Areas Device by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/216
* Enhance area management in config flow by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/217
* Add Home Asssitant Area Association by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/218
* Refactor cleanup processes to enhance memory management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/219
* Refactor entity type filtering in AreaOccupancyDB and PriorAnalyzer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/220
* Fix prior persistence semantics by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/221
* Refactor DB with new schema by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/222
* Remove legacy support for area configurations and enhance error handling by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/223
* Refactor area occupancy coordinator and service logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/224
* Bump uv from 0.9.6 to 0.9.7 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/203
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/202
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/201
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/200
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/199
* Bump hass-nabucasa from 1.4.0 to 1.5.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/229
* Bump audioop-lts from 0.2.1 to 0.2.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/228
* Bump uv from 0.9.6 to 0.9.10 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/227
* Bump cryptography from 46.0.2 to 46.0.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/225
* Bump orjson from 3.11.3 to 3.11.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/226
* 2025.11.3 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/211
* Consolidate database aggregation to SQL-only path by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/231
* Configure motion sensor probabilities and refine Bayesian logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/232
* Replace occupancy sensor with motion sensors and consolidate configs by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/239
* Add environmental sensor support to area occupancy component by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/240
* Add energy sensors to configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/241
* Add migration logic for multi-area entry conversion by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/242
* Update version to 2025.11.3-pre5 in project files by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/243


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.11.3-pre5


## [2025.11.3-pre4] - 2025-11-16

## What's Changed
* Multi-Device Architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/189
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/206
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/210
* Next by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/190
* Setup standard Area Class and simplify Helper Classes by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/212
* Move heavy analysis functions to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/213
* Add testing guide and refactor test setup for multi-area architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/214
* Restructure Area device properties to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/215
* Create All Areas Device by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/216
* Enhance area management in config flow by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/217
* Add Home Asssitant Area Association by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/218
* Refactor cleanup processes to enhance memory management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/219
* Refactor entity type filtering in AreaOccupancyDB and PriorAnalyzer by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/220


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.11.3-pre4


## [2025.11.3-pre2] - 2025-11-14

## What's Changed
* Multi-Device Architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/189
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/206
* Simulator by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/210
* Next by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/190
* Setup standard Area Class and simplify Helper Classes by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/212
* Move heavy analysis functions to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/213
* Add testing guide and refactor test setup for multi-area architecture by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/214
* Restructure Area device properties to dedicated module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/215
* Create All Areas Device by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/216
* Enhance area management in config flow by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/217
* Add Home Asssitant Area Association by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/218


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.11.3-pre2


## [2025.11.3-pre1] - 2025-11-06

## What's Changed
* Fix incorrect prior being used resulting in low probabilities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/205


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.2...2025.11.3-pre1


## [2025.11.2] - 2025-11-03

## What's Changed
* Bugfix: Database Pool Error by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/197
* Bump awesomeversion from 25.5.0 to 25.8.0 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/196
* Bump ruff from 0.14.2 to 0.14.3 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/195
* Bump home-assistant-bluetooth from 1.13.1 to 2.0.0 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/194
* Bump setuptools from 78.1.1 to 80.9.0 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/193
* Bump lru-dict from 1.3.0 to 1.4.1 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/192
* 2025.11.2 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/198


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.11.1...2025.11.2


## [2025.11.1] - 2025-11-02

## Hotfix Release

This is a minor release to revert the MIN_PRIOR to 10% as it was causing lower than expected probabilities. A more comprehensive fix for this is coming in a future release. Discussed here: #188

Also addresses #177, #174 

## What's Changed
* Bump actions/setup-python from 5 to 6 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/148
* Enhance database integrity checks and initialization logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/185
* Refactor motion sensor probability logging and enhance test coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/186
* 2025.11.1 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/187
* Bump uv from 0.9.5 to 0.9.6 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/183
* Bump cronsim from 2.6 to 2.7 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/182
* Bump astral from 2.2 to 3.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/181
* Bump aiohttp from 3.13.1 to 3.13.2 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/180
* Bump sqlalchemy from 2.0.41 to 2.0.44 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/179
* Bump voluptuous-openapi from 0.1.0 to 0.2.0 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/178


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.10.1...2025.11.1


## [2025.11.1-pre1] - 2025-11-01

## What's Changed
* Bump actions/setup-python from 5 to 6 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/148
* Enhance database integrity checks and initialization logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/185
* Refactor motion sensor probability logging and enhance test coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/186


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.10.1...2025.11.1-pre1


## [2025.10.1] - 2025-10-28

The 💯⭐ release! Thank you to all who share and support the project, having that star number go up satisfies my craving for internet points 😁.

Now we have it running on several hundred system according to HA telemetry some of the at scale issues reared their head, so I have addressed as much as possible bug-wise in this release. as well as some much requested wasp updates. Please don't be afraid to add anything to [issues](https://github.com/Hankanman/Area-Occupancy-Detection/issues) or [discussions](https://github.com/Hankanman/Area-Occupancy-Detection/discussions). I read them all and features suggested in there are coming!

## 🎉 New & Updated Features

### Enhanced Wasp in Box Sensor
The Wasp in a Box feature is the most frequently discussed so I did a bit of an overhaul in this version, you can find out exactly how it works and the logic it follows now with a fancy diagram and table of scenarios here: [Wasp in a Box Docs](https://hankanman.github.io/Area-Occupancy-Detection/features/wasp-in-box/).

There were several discussions and issues that I now consider resolved, bar a single feature (adjecent room detection) which will come in a future release. #102 #55 #150 

#### Aggregate State Calculation
The Wasp in Box sensor now intelligently handles multiple sensors:

  - **Multi-Door Support**: Monitors ALL configured door sensors - any open door indicates potential exit
  - **Multi-Motion Support**: Monitors ALL configured motion sensors - any active sensor indicates presence
  - **Smart State Logic**: 
    - Room is occupied when ALL doors closed AND ANY motion detected
    - Room becomes unoccupied when ANY door opens while occupied
    - Maintains state when motion stops (until door opens)

#### Verification Delay Feature
New optional verification mechanism to reduce false positives:

  - **Configurable Delay**: Set a delay (0-120 seconds) before confirming occupancy
  - **Motion Re-Check**: After the delay, system verifies motion is still present
  - **False Positive Reduction**: Automatically clears occupancy if motion was transient
  - **Door-Only Support**: Verification is skipped when no motion sensors are configured
  - **Default Disabled**: Set to 0 by default to maintain backward compatibility

### Entity Management
- **Stale Entity Cleanup**: Automatic removal of entities no longer in configuration, fixes [#156](https://github.com/Hankanman/Area-Occupancy-Detection/issues/156)
- **Orphan Detection**: Identifies and removes entities with no parent configuration

### Configuration & Setup
- **Appliance Detection**: Light entities now included in appliance detection, fixes [#146](https://github.com/Hankanman/Area-Occupancy-Detection/issues/146)

### Documentation Updates

Available on the [Documentation Site](https://hankanman.github.io/Area-Occupancy-Detection)

## 📊 Performance Improvements

### Startup Performance
  - **~60% faster startup**: Heavy operations deferred to background tasks
  - **Parallel initialization**: Multiple instances no longer block each other during startup
  - **Reduced blocking**: Async database operations prevent event loop blocking

### Runtime Performance
  - **SQL aggregation**: 10-100x faster time slot aggregation depending on data volume
  - **Cached intervals**: Reduces repeated database queries by up to 90%
  - **Debounced saves**: Reduces database writes by 80-95% during active periods
  - **Filtered queries**: Lookback windows reduce query processing time

### Database Size Management
  - **Automatic pruning**: Prevents unbounded database growth
  - **Optimized indexes**: Better query performance over time
  - **Periodic maintenance**: ANALYZE and OPTIMIZE commands run during health checks

## 🤓 Technical Feature Updates

### Multi-Instance Coordination System
The integration now supports running multiple instances simultaneously with intelligent coordination:

  - **Master Election**: Automatic election of a master instance to handle database writes and maintenance. This will allow for more advanced coordination features like multi-area aggregation and adjacent room influence
  - **Heartbeat System**: Master instances broadcast heartbeats; non-master instances monitor health and trigger re-election if needed
  - **Staggered Analysis**: Analysis runs are distributed across instances to reduce system load (2-minute intervals per instance)
  - **Coordinated Database Access**: File-based locking prevents race conditions and data corruption
  - **Automatic Failover**: If master becomes unresponsive, a new master is automatically elected

### Fast Startup Mode
Significantly improved Home Assistant startup times:

  - **Deferred Operations**: Heavy database operations (integrity checks, historical analysis) moved to background tasks
  - **Parallel Loading**: Multiple instances can load data in parallel without blocking each other
  - **Quick Validation**: Basic database validation only during startup, full checks happen after HA is ready
  - **Non-Blocking Initialization**: Database initialization runs asynchronously to prevent event loop blocking

### Database Resilience & Recovery

#### Automatic Health Monitoring
  - **Periodic Health Checks**: Master instance runs integrity checks during analysis cycles
  - **Corruption Detection**: Identifies common database corruption patterns
  - **Automatic Recovery**: Attempts multiple recovery strategies when corruption detected
  - **Backup & Restore**: Automatic backups every 24 hours with restore capability

#### Recovery Strategies
  1. **WAL Mode Recovery**: Enable Write-Ahead Logging and checkpoint
  2. **Backup Restoration**: Restore from most recent backup if available
  3. **Database Rebuild**: Complete rebuild as last resort

#### File-Based Locking
  - **Race Condition Prevention**: Uses `filelock` package for robust cross-process locking
  - **Timeout Handling**: Configurable timeouts with graceful failure
  - **Stale Lock Detection**: Automatic cleanup of stale locks
  - **Parallel Read Safety**: Two-phase loading allows parallel reads, locks only for writes

## 🐞 Bug Fixes

  - HA locks up for 3 minutes right after unified occupancy calculation #163
  - Removed sensors still appear in diagnostics and affect occupancy detection #156
  - Failed to setup coordinator #154
  - Missing Lights #146

**Database Issues**
  - Fixed UNIQUE constraint violations during interval syncing by checking for existing intervals
  - Resolved race conditions in database access during parallel startup
  - Fixed database initialization in test environments
  - Corrected session handling to prevent nested transaction issues

**Wasp in Box Issues**
  - Fixed aggregate state calculation for multiple sensors
  - Corrected motion timestamp updates to prevent premature state changes
  - Fixed verification timer cancellation during state transitions
  - Resolved issue where door-only setups would incorrectly verify motion

**Entity Processing**
  - Fixed entity creation from database when coordinator types don't match
  - Corrected decay factor clamping to prevent invalid values
  - Fixed likelihood calculation for entities with insufficient data
  - Resolved type conversion issues in entity data loading

**Coordinator Issues**
  - Fixed debouncer conflicts during setup by marking setup complete before initial refresh
  - Corrected analysis timer scheduling to maintain proper stagger intervals
  - Fixed refresh requests during configuration updates to prevent blocking
  - Resolved timer cleanup issues during shutdown


## 📄 What's Changed
  - Bump actions/checkout from 4 to 5 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/143
  - Bump ruff from 0.12.7 to 0.12.8 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/139
  - Database corruption by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/153
  - Bug/interval retention caching by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/165
  - Add cleanup functionality for orphaned entities in AreaOccupancyDB by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/166
  - Enhance AreaOccupancyDB to delete stale entities from the database by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/167
  - Implement asynchronous database initialization in AreaOccupancy by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/168
  - Bug/startup performance by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/169
  - Add debounced database save logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/171
  - Feat/master delegation by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/172
  - Update appliance detection to include light entities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/173
  - Add verification delay feature to Wasp in Box sensor by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/175
  - Enhance WASP logic and documentation by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/176
  - 2025.10.1 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/164


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.8.2...2025.10.1


## [2025.10.1-pre6] - 2025-10-27

## What's Changed
* Bump actions/checkout from 4 to 5 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/143
* Bump ruff from 0.12.7 to 0.12.8 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/139
* Database corruption by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/153
* Bug/interval retention caching by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/165
* Add cleanup functionality for orphaned entities in AreaOccupancyDB by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/166
* Enhance AreaOccupancyDB to delete stale entities from the database by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/167
* Implement asynchronous database initialization in AreaOccupancy by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/168
* Bug/startup performance by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/169
* Add debounced database save logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/171
* Feat/master delegation by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/172
* Update appliance detection to include light entities by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/173
* Add verification delay feature to Wasp in Box sensor by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/175


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.8.2...2025.10.1-pre6


## [2025.8.2] - 2025-08-11

## What's Changed
* Refactor issue templates for bug reports and feature requests by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/141
* Fix door sensors attribute naming by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/142

Fixes #140 

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.8.1...2025.8.2


## [2025.8.1] - 2025-08-09

## 🚀 New Features

* **Time-Based Priors for Smarter Occupancy Detection**

  * Addresses the request in: https://github.com/Hankanman/Area-Occupancy-Detection/issues/24
  * Occupancy probability now adjusts dynamically based on **day of week** and **time of day**.
  * Learns from historical patterns so your system can, for example, expect higher occupancy on weekday evenings or weekend mornings.
  * Improves accuracy by combining **area priors** with **time-slot-specific priors** during Bayesian probability calculations.

* **Motion Timeout Configuration**

  * Added a new `motion_timeout` setting for motion sensors, allowing you to define how long motion is considered active after the last detection.
  * Default timeout is **5 minutes**, adjustable from **0 to 3600 seconds** in 5-second steps.
  * Improves occupancy accuracy by merging and extending motion intervals.

* **Per-Type Occupancy Probabilities**

  * Sensors now calculate and expose separate occupancy probabilities for each input type (motion, media, appliance, door, window, illuminance, humidity, temperature).
  * Enhances insight into which sensor categories are contributing to occupancy detection.

* **Extra State Attributes on Sensors**

  * Occupancy probability sensors now include additional attributes such as:

    * `prob_given_true` / `prob_given_false` (likelihood values)
    * `decay_factor` (decay state of the sensor)
    * `time_prior` and other context data for advanced analysis.

## Bug Fixes

  * #135
  * #133 

## 🔧 Changes & Improvements

* **Database-Backed Storage**

  * Replaced legacy file-based storage with a new `AreaOccupancyDB` system for more robust, reliable, and performant data handling.
  * Historical data imports, state tracking, and priors are now fully database-driven.

* **Improved Occupancy Probability Calculations**

  * Updated Bayesian probability functions for better numerical stability and clamping of extreme values.
  * Adjusted min/max probability bounds to `0.01`–`0.99` to avoid unrealistic certainty.

* **Streamlined Entity Management**

  * Removed the `Likelihood` class; entities now store and update their own probability attributes.
  * New `EntityFactory` automatically builds sensors from configuration or database data.
  * Simplified entity type mapping and validation.

* **Configuration Handling Overhaul**

  * Configuration is now loaded directly from the coordinator for faster and more reliable updates.
  * Improved validation of sensor lists, weights, and thresholds.
  * Default active state for motion sensors set to `on`.

* **Decay Model Improvements**

  * Decay timers now use precise UTC datetime tracking rather than floating-point timestamps.
  * Added more explicit methods for starting and stopping decay, improving accuracy.

* **Prior Calculations**

  * Introduced caching for time-based priors to improve performance.
  * More accurate calculation of area priors using historical motion data.

## 🗑 Removed / Deprecated

* **Removed Obsolete Services**

  * `get_entity_details` and `get_entity_type_learned_data` services removed.
  * Legacy debug and purge services also removed.

* **Removed Legacy Storage Layer**

  * The old `AreaOccupancyStorage` and related file-based state handling are no longer used.


## What's Changed
* Refactor AreaOccupancyStorage and Entity management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/130
* Update DB Schema and Calculation Methods by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/132
* docs: sync docs with current implementation by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/137
* Add pre-release install docs by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/138
* Update DB Schema and Calculation Methods by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/136


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.5...2025.8.1


## [2025.8.1-pre2] - 2025-08-05

## What's Changed
* Refactor AreaOccupancyStorage and Entity management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/130
* Update DB Schema and Calculation Methods by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/132


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.5...2025.8.1-pre2


## [2025.8.1-pre1] - 2025-08-02

Addresses: #133 

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.5...2025.8.1-pre1


## [2025.7.5] - 2025-07-30

This is a minor release to prep the storage module for upcoming features.

## What's Changed
* Switch storage module to orm methods by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/126
* Bump ruff from 0.12.5 to 0.12.7 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/128
* 2025.7.5 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/127


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.4...2025.7.5


## [2025.7.5-pre1] - 2025-07-28

## What's Changed
* Switch storage module to orm methods by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/126


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.4...2025.7.5-pre1


## [2025.7.4] - 2025-07-28

This is a hotfix release to fix two issues:

* Global prior and likelihoods were not correctly stored and restored in the database.
* Analysis timer was not correctly fired due to a missing parameter.

## What's Changed
* 2025.7.4 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/125


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.3...2025.7.4


## [2025.7.3] - 2025-07-27

## Highlights

* **Major Storage Overhaul:** Migrated from JSON storage to SQLite for better performance and historical analysis.
* **Automated Historical Analysis:** Integration now imports the last **10 days** of past state history from Recorder and updates priors hourly.
* **Improved Bayesian Calculations:** Priors and likelihoods recalculated using optimized interval merging and chunked async processing.
* **Removed Legacy Config:** Eliminated `history_period` and `historical_analysis_enabled` options (analysis now fully automatic).
* **New DB Schema:** Supports efficient queries and future Machine Learning enhancements.
* **Enhanced Stability:** Better error handling during setup and cleanup.

## Breaking Changes

* Local `.storage` JSON files are no longer used; a SQLite database handles all persistence. These will still be in your .storage directory in Home Assistant (will be cleaned up automatically in future release)
* Deprecated config options for historical analysis removed. (Migrated automatically)

## After Upgrade

* Initial setup may take several minutes as historical data is imported and priors are recalculated.
* Expect **1-hour cycles** for continuous historical updates.
* No manual config changes are needed for new historical features.

## Details
* Bump ruff from 0.12.1 to 0.12.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/100
* Raise coverage past 85% by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/103
* Improve tests to reach 85% coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/104
* Improve storage durability and indexing by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/106
* Improve sqlite storage test coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/107
* Fix lingering timer test failures by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/108
* Time based prior by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/90
* Fix prior calculation bug for multiple sensors by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/110
* Refactor SQLite storage table and index creation in Area Occupancy component by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/111
* Extract DB to Recorder operations to new module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/113
* Add purge_intervals service to manage state intervals retention by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/114
* Enhance prior calculation logic in the Prior class by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/115
* Simplify Coordinator setup procedure by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/117
* Removal of "force" and "history_period" Parameters by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/118
* Remove time slot analysis by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/119
* Bump ruff from 0.12.4 to 0.12.5 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/116
* Enhance SQLite storage handling by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/120
* Enhancements to Prior Probability Calculations by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/121
* Update docs for release by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/122
* SQL schema updates by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/123
* 2025.7.3 by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/112
* Update Area Occupancy component version and manifest formatting by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/124

## New Contributors
* @dependabot[bot] made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/100

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.2...2025.7.3


## [2025.7.3-pre6] - 2025-07-27

## What's Changed
* Update docs for release by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/122
* SQL schema updates by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/123


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.3-pre5...2025.7.3-pre6


## [2025.7.3-pre5] - 2025-07-26

## What's Changed
* Enhancements to Prior Probability Calculations by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/121


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.3-pre4...2025.7.3-pre5


## [2025.7.3-pre4] - 2025-07-26

## What's Changed
* Improve Reliability of Weights and Overall Probability by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/101
* Bump ruff from 0.12.1 to 0.12.4 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/100
* Simplify Coordinator setup procedure by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/117
* Removal of "force" and "history_period" Parameters by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/118
* Remove time slot analysis by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/119
* Bump ruff from 0.12.4 to 0.12.5 by @dependabot[bot] in https://github.com/Hankanman/Area-Occupancy-Detection/pull/116
* Enhance SQLite storage handling by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/120

## New Contributors
* @dependabot[bot] made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/100

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.3-pre3...2025.7.3-pre4


## [2025.7.3-pre3] - 2025-07-24

## What's Changed
* Extract DB to Recorder operations to new module by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/113
* Add purge_intervals service to manage state intervals retention by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/114
* Enhance prior calculation logic in the Prior class by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/115


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.3-pre2...2025.7.3-pre3


## [2025.7.3-pre2] - 2025-07-24

## What's Changed
* Raise coverage past 85% by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/103
* Improve tests to reach 85% coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/104
* Improve storage durability and indexing by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/106
* Improve sqlite storage test coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/107
* Fix lingering timer test failures by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/108
* Time based prior by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/90
* Fix prior calculation bug for multiple sensors by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/110
* Refactor SQLite storage table and index creation in Area Occupancy component by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/111


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.2...2025.7.3-pre2


## [2025.7.3-pre1] - 2025-07-23

## WARNING
If you choose to test this pre-release there is a risk of data loss as we switch storage methods

## What's Changed
* Raise coverage past 85% by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/103
* Improve tests to reach 85% coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/104
* Improve storage durability and indexing by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/106
* Improve sqlite storage test coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/107
* Fix lingering timer test failures by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/108
* Time based prior by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/90


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.2...2025.7.3-pre1


## [2025.7.2] - 2025-07-19

You should notice the weights you configure are more obviously impactful following this release, the probability will be slightly different from earlier releases, but not different enough that I expect your threshold will need to be updated, please monitor and submit an issue if there is anything unexpected.

## What's Changed
* Improve Reliability of Weights and Overall Probability by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/101


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.1...2025.7.2


## [2025.7.1] - 2025-07-14

This is a smaller hotfix release primarily to address issue #92 as i have some temporary code that code shipped, meaning environmental sensors were either always or never active. Better defaults have been added in this release. In a later release these values will be calcualted automatically.

## Probability Calculations
Added three new probability calculation methods (complementary_probability, conditional_probability, and conditional_sorted_probability) to provide more accurate occupancy estimations. The main method has not changed (yet) and the other methods are available as attributes of the probability entity. Feedback on these methods is much appreciated.

## Sensor Types and Environmental Inputs
Split the ENVIRONMENTAL sensor type into individual types (TEMPERATURE, HUMIDITY, ILLUMINANCE) and updated their respective configurations, active ranges, and probabilities. [[1]](https://github.com/Hankanman/Area-Occupancy-Detection/pull/95/files#diff-f6c771f31b60a615b67e168e2f2c989f2baaba47323507e34b7dac06f9f08cd9R30-R32) [[2]](https://github.com/Hankanman/Area-Occupancy-Detection/pull/95/files#diff-f6c771f31b60a615b67e168e2f2c989f2baaba47323507e34b7dac06f9f08cd9R259-R282) [[3]](https://github.com/Hankanman/Area-Occupancy-Detection/pull/95/files#diff-bf51304cf2e96e4371cdb63d35175f6fef3f589db16cce4c113c27cd33a227a4L392-R418)

## Entity Fixes and Enhancements
Introduced a decay_factor property in custom_components/area_occupancy/data/entity.py to prevent inconsistent states where evidence is True but decay is applied.
Fixed inconsistent states in has_new_evidence by stopping decay when evidence is True.

## Miscellaneous Updates
Increased the minimum prior value in custom_components/area_occupancy/data/prior.py from 0.01 to 0.1 to avoid division by zero in calculations.


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.6.1...2025.7.1


## [2025.7.1-pre2] - 2025-07-04

- Fix bug where conditional probability methods where not considering decay now calculates effective evidence, considering both evidence and decay state, before calculation.
- Fix bug where decay was not correctly set when restoring from storage

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.7.1-pre1...2025.7.1-pre2


## [2025.7.1-pre1] - 2025-07-04

## What's Changed
Added temporary attributes to the probability sensor displaying different methods of fusing probability:

Complementary Probability. (same as release 2025.6.1)

This function computes the probability that at least ONE entity provides evidence for occupancy, using the complement rule:
P(at least one) = 1 - product(P(not each)). For each contributing entity, a Bayesian update is performed assuming evidence is True (or decaying), and the complement of the posterior is multiplied across all such entities.
Is not affected by the order of the entities.
Does not consider negative evidence.

Conditional Probability. (Most mathematically accurate)

Sequentially update the prior probability by applying Bayes' theorem for each entity, using the entity's evidence and likelihoods. The posterior from each step becomes the prior for the next entity. This method reflects the effect of each entity's evidence (and decay, if applicable) on the overall probability.

Conditional Sorted Probability. (Mathematically accurate, with higher weighted and true evidence entities considered first in the for loop)

Sequentially update the prior probability by applying Bayes' theorem for each entity, using the entity's evidence and likelihoods. The posterior from each step becomes the prior for the next entity. This method reflects the effect of each entity's evidence (and decay, if applicable) on the overall probability. The entities are sorted by evidence status (active first) and then by weight (highest weight first) to ensure that the most relevant entities are considered first.

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.6.1...2025.7.1-pre1


## [2025.6.1] - 2025-06-30

### ‼️IMPORTANT
There are new options you should adjust following this upgrade. You should also potentially adjust your thresholds.

### 🧠 New
* **Area Purpose**
  This is a dropdown with options like _Passageway_, _Utility_, _Social_, and _Sleeping_. This should be chosen based on what best describes the area's purpose. If there is something missing that should be added as your are doesn't fit into any of these categories, please add to the thread here: [Area Purpose](https://github.com/Hankanman/Area-Occupancy-Detection/discussions/80)

* **Decay Half Life**
  A lower Half Life means it will decay faster, something like a hallway works well on a Half Life as low as 10 seconds, where somewhere like a living room where you may sit still for a while may be better with a 300 second Half Life.

  This has been added to better represent the decay lifecycle of the probability for an area. This is a time in seconds over which the probability will drop by half. This provides a smoother drop off. After extensive testing in my own set up it has resulted in some very accurate representation of occupancy.

* **Evidence Sensor**
  A new senor has been added to the diagnostic area, this shows the number of entities being tracked as its main state, but has a number of useful attributes that tell you what is affecting the area occupancy at any given time. You can get a friendly list of what entities are causing occupancy to be detected using the following jinja syntax:
  ```
  {{ state_attr('sensor.study_evidence', 'evidence') }}
  ```

* **Services**
  A number of new services have been added to help you test as well as see what has been learned. The majority of services now return values, for example `Update Sensor Likelihoods` will perform the history calculations for all configured sensors on demand, and then return all the info learned by the integration. This includes things like if we excluded some time periods from the calculation because they looked like anomalies.
  
  You can find out more about the new services here: [Services](https://hankanman.github.io/Area-Occupancy-Detection/features/services/)

### 🧪 Upcoming Features
A lot of the work done for this release was foundational for the next few features on the roadmap.

The refactor lets me roll out big features as incremental updates instead of disruptive rewrites. First up is numeric-sensor historical analysis (temperature, humidity, CO₂, etc.) — already underway and targeting a July beta. Next will follow time-of-day probability adjustments and activity detection (“in shower”, “cooking”, “sleeping”), all powered by the same learning engine.

### 🗑️ Removed Options
* **Decay Window**
  Replaced entirely by Decay Half Life
* **Decay Min Delay**
  Replaced entirely by Decay Half Life
* **Lights as Inputs**
  Light entitles are typically what you are trying to control rather than a reliable occupancy indicator, as a result, you would end up with contradictions quite often, so these can been removed.

### 📚 Docs
New pages available on the [📘 docs site](https://hankanman.github.io/Area-Occupancy-Detection/features/) for the new features.

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.5.1...2025.6.1


## [2025.6.1-pre6] - 2025-06-27

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.6.1-pre5...2025.6.1-pre6


## [2025.6.1-pre5] - 2025-06-24

## What's Changed
* Refactor overall_probability by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/71
* Ensure interval coverage for entire range by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/72
* Remove deprecated light sensor configuration by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/75
* Bayesian logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/76


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.6.1-pre4...2025.6.1-pre5


## [2025.6.1-pre4] - 2025-06-23

## What's Changed
* Add area primary purpose dropdown by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/65
* Refactor Area Occupancy probability calculations and utility functions by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/66
* Refactor Area Occupancy service and storage components by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/67
* Bayesian logic by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/70


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.6.1-pre3...2025.6.1-pre4


## [2025.6.1-pre3] - 2025-06-19

## What's Changed
* Reduce test clutter and hit 90% coverage by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/62
* Update setup script by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/63
* Remove unused decay delay option by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/64


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.6.1-pre2...2025.6.1-pre3


## [2025.6.1-pre2] - 2025-06-18

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.6.1-pre1...2025.6.1-pre2


## [2025.6.1-pre1] - 2025-06-16

## What's Changed
* State management by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/58


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.5.1...2025.6.1-pre1


## [2025.5.1] - 2025-05-25

This release introduces the highly-requested **Wasp in a Box** feature, along with a major internal refactor to support virtual sensors, simplify configuration, and improve performance and reliability across the board.

Now on HACS! The component has officially been accepted to the HACS store and is even easier to find and install!

### ✅ Bugs Fixed

[#47 - Can't start due to migration error](https://github.com/Hankanman/Area-Occupancy-Detection/issues/47)

---

### 🧠 New: Virtual Sensor Framework

* **Virtual Sensors Are Here**
  You can now create occupancy logic based on sensor combinations and rules — like motion after a door opens — using the new virtual sensor system.

* **Wasp in a Box**
  First out of the gate: a virtual sensor that detects occupancy when someone enters a room (door opens), and then moves (motion detected), even if no motion is detected afterward.
  Perfect for rooms like bathrooms or studies where people might sit still for a while.

* **Highly Configurable**
  Set timeouts, weights, and max durations. Works alongside your regular sensors without replacing them.

---

### 🔧 Smarter Configuration

* **Refactored Configuration Flow**
  The UI for selecting sensors is now cleaner, more modular, and easier to extend.
  Strings and validation are also improved — fewer errors, clearer labels.

* **Auto-Include Primary Sensors**
  Your `primary_occupancy_sensor` is automatically added to the motion sensors if needed, reducing setup friction.

---

### 📚 Docs + Screenshots

* **Wasp in Box Documentation**
  A new page on the [📘 docs site](https://hankanman.github.io/Area-Occupancy-Detection/features/wasp-in-box/) explains the concept and setup of the new virtual sensor.

* **Updated Overview & Index**
  The index now includes GitHub Action badges and links to new technical explanations and UI screenshots.

---

### 🧪 Testing & Developer UX

* CI workflows added for testing, linting, and validation via GitHub Actions
* Types and constants refined for clarity and maintainability
* Linter compliance and test coverage improved

---

This update sets the foundation for more powerful sensor logic, smarter detection, and easier configuration — all without compromising performance or stability.

As always, find full setup guides and technical docs at:
🔗 [https://hankanman.github.io/Area-Occupancy-Detection/](https://hankanman.github.io/Area-Occupancy-Detection/)
📦 [https://github.com/Hankanman/Area-Occupancy-Detection](https://github.com/Hankanman/Area-Occupancy-Detection)

Let me know what you build with Wasp in a Box!
There is a discussion thread here: 🐝📦[Wasp in a Box](https://github.com/Hankanman/Area-Occupancy-Detection/discussions/21)

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.3...2025.5.1


## [2025.4.4-pre2] - 2025-05-01

Added much more comprehensive implementation for wasp in a box and additional config options

Addresses #27 

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.3...2025.4.4-pre2


## [2025.4.4-pre1] - 2025-04-20

Introduces the "Wasp-in-the-Box" logic, a virtual sensor that maintains occupancy state when the area was recently occupied, even if all other sensors are inactive. The feature includes configuration options, default probability values, and updates to the Bayesian calculation to integrate the new sensor type. Additionally, documentation is added to explain the functionality and usage of the Wasp-in-the-Box logic, along with corresponding tests to ensure its reliability and performance within the Area Occupancy Detection integration.

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.3...2025.4.4-pre1


## [2025.4.3] - 2025-04-20

This release primarily cleans up a lot of code and improves maintainability going forward as well as adding a full raft of automated tests. There are no new features here, but gets the integration to a high quality standard and level of stability so that new features can be built on top.

### ✅ Bugs Fixed

[#34 - Door Sensors status not being reflected in the integration.](https://github.com/Hankanman/Area-Occupancy-Detection/issues/34)
(this one wasn't actually fixed in last release)

### 🧠 Accuracy & Performance

- **Refined Bayesian Calculation:** Composite probability now uses freshly computed priors if available, with fallbacks to well-defined defaults.
- **Optimized Sensor State Fetching:** Improves performance and resilience when querying historical data, including timeout handling and caching.
- **Enhanced Decay & Weighting:** Appliance weight increased from 0.3 to 0.4; decay window adjusted from 10 to 5 minutes for faster responsiveness.

### 🛠 Configuration & UX

- **Improved Logging & Error Handling:**
  - Logs now differentiate between transient issues and actual misconfigurations.
  - All prior update and calculation errors now propagate meaningful messages.

### 🧪 Developer & Testing Improvements

- **Increased Test Coverage:**
  - Extensive tests for the `DecayHandler`, `Probabilities`, `PriorCalculator`, and coordinator logic.
- **Strict Linting and Type Safety:** Codebase refactored to meet PEP8/257, use full type hints, and follow Home Assistant best practices with `ruff` linting.
- **Automated CI via GitHub Actions:** Full integration tests, coverage reporting, and linting now run automatically for each change.

### 🗃 Internal Refactors

- `PriorCalculator`, `ProbabilityCalculator`, and `AreaOccupancyCoordinator` refactored for better state isolation, testability, and clarity.
- `AreaOccupancyStore` replaces older storage class, improving maintainability and naming consistency.

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.2...2025.4.3


## [2025.4.3-pre-3] - 2025-04-18

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.3-pre-2...2025.4.3-pre-3


## [2025.4.3-pre-2] - 2025-04-13

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.3-pre-1...2025.4.3-pre-2


## [2025.4.3-pre-1] - 2025-04-12

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.2...2025.4.3-pre-1


## [2025.4.2] - 2025-04-11

This patch builds on the recent storage and prior probability overhaul with further enhancements to performance, reliability, and configuration clarity.

---

### 🔧 Smarter Setup & Auto Configuration

- **Primary Sensor Auto-Inclusion**:  
  The primary occupancy sensor is now **automatically added to your motion sensor list** if you forgot to include it. This makes setup easier and ensures accurate prior learning without requiring manual tweaks.

---

### 🧠 Improved Prior State Management

- **More Accurate Scheduling**:  
  Prior updates are now triggered based on **actual age and completeness** of stored data, not just a timer. If your data is stale or incomplete, it’ll be updated automatically on startup.

- **Better Caching**:  
  Prior calculations now align cache expiration with update schedules, reducing unnecessary recalculations and improving performance.

---

### 💾 Storage Cleanup & Persistence

- **Orphaned Entry Cleanup**:  
  When an integration instance is removed, its storage data is now cleaned up automatically.

- **Improved Migration Handling**:  
  Safer loading of prior data with fallbacks and error handling for better robustness during setup or restarts.

---

### 🔍 Logging & Debugging Improvements

- **Cleaner Logs**:  
  Refined debug logging across the codebase—less spam, more clarity.

- **Traceable Calculations**:  
  All major prior and probability calculations now include instance IDs and detailed metrics in the logs for easier debugging.

---

### ✅ Bugs Fixed

[#34 - Door Sensors status not being reflected in the integration.](https://github.com/Hankanman/Area-Occupancy-Detection/issues/34)
[#33 - Config settings not being used.](https://github.com/Hankanman/Area-Occupancy-Detection/issues/33)
[#32 - mmwave as primary motion sensor?](https://github.com/Hankanman/Area-Occupancy-Detection/issues/32)

This release keeps things smooth under the hood and brings us closer to a stable, robust, and hands-off experience.  
Thanks for using Area Occupancy Detection!


**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.1...2025.4.2


## [2025.4.2-pre-5] - 2025-04-11

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.2-pre-4...2025.4.2-pre-5


## [2025.4.2-pre-4] - 2025-04-11

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.2-pre-3...2025.4.2-pre-4


## [2025.4.2-pre-3] - 2025-04-11

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.2-pre-2...2025.4.2-pre-3


## [2025.4.2-pre-2] - 2025-04-03

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.2-pre-1...2025.4.2-pre-2


## [2025.4.2-pre-1] - 2025-04-03

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.1...2025.4.2-pre-1


## [2025.4.1] - 2025-04-02

## 🚀 Area Occupancy Integration – New Release Highlights

### 🔍 Smarter Occupancy Detection with Bayesian Inference  
We've fundamentally reimagined how occupancy is detected:

- **Bayesian Inference Engine**  
  The new probability engine now uses **Bayesian inference** to combine data from multiple sensors intelligently. This allows the integration to better reason about occupancy even when sensor signals are noisy or partial.

- **Learned Priors from Sensor History**  
  A new **learning system** continuously evaluates historical data from your devices to derive **per-sensor** and **sensor-type priors**. These values are persisted across restarts and improve over time.

---

### 📘 New Documentation Site Now Live!

We’re proud to launch the official documentation site for this integration:

👉 **[https://hankanman.github.io/Area-Occupancy-Detection/](https://hankanman.github.io/Area-Occupancy-Detection/)**

It includes full setup guides, configuration references, architectural overviews, and troubleshooting tips — with more content coming soon.

---

### 🧠 Introducing the `primary_occupancy_sensor`  

To support this learning system, we’ve added a new configuration option:  
> **Primary Occupancy Sensor**  

This acts as the **ground truth reference** for historical learning.  
It should be your **most reliable sensor** for detecting occupancy — typically a motion, presence, or occupancy sensor that best reflects real-world activity in the space.

By analysing how other sensors behave when the primary sensor is active or inactive, the system learns which signals actually indicate someone is present — and which do not.

---

### 🔄 Decay Logic Revamped

- **Smooth Probability Decay**  
  Previously, occupancy probability dropped off in abrupt steps. Now, it **decays smoothly and continuously** over time, making transitions between states feel more natural and realistic.

- **No More 1-Second Polling**  
  We've **eliminated the 1-second update interval** in favour of:
  - **Event-driven state tracking**, and
  - A **scheduled decay timer** that ticks only when needed.

This change drastically reduces CPU usage and improves responsiveness, especially on resource-constrained devices.

---

### ⚙️ Configuration Flow & Validation Upgrades

- **Enhanced Input Validation**  
  The setup UI now performs comprehensive checks to help you avoid misconfiguration — including validation of motion sensors, weights, active states, and decay timing.

- **Centralized Slider Configuration**  
  All configuration sliders (weights, thresholds, etc.) now use centralized settings for consistent UX and easier future tuning.

---

### 🧱 Structural and Developer Improvements

- Refactored key modules (`calculate_prior`, `calculate_prob`, `coordinator`, etc.) into clean, modular components.
- Replaced ad-hoc sensor logic with well-defined data models (`SensorInputs`, `PriorState`, `ProbabilityState`).
- Enhanced logging and error handling with more descriptive messages and robust fallback mechanisms.
- Added detailed inline documentation to help new contributors get started faster.




**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.3...2025.4.1


## [2025.4.1-pre-3] - 2025-04-02

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.1-pre-2...2025.4.1-pre-3


## [2025.4.1-pre-2] - 2025-04-02

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.4.1-pre-1...2025.4.1-pre-2


## [2025.4.1-pre-1] - 2025-04-02

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2025.2.1-pre-3...2025.4.1-pre-1


## [2025.2.1-pre-3] - 2025-02-02

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.3...2025.2.1-pre-3


## [2025.2.1-pre-2] - 2025-02-01

Consolidated config flow and added ability to configure active states

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.3...2025.2.1-pre-1

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.3...2025.2.1-pre-2


## [2025.2.1-pre-1] - 2025-02-01

This release improves statement management and reliability of the decay calculation

Modularize state and decay management with new utility classes
This commit introduces two new utility classes to improve the Area Occupancy integration's architecture:

- `StateManagement`: A dedicated class for managing sensor states with improved concurrency and state tracking
- `DecayHandler`: A specialized class to handle probability decay calculations with enhanced configuration management

The changes include:
- Extracting state management logic from the coordinator
- Implementing thread-safe state tracking
- Separating decay calculation concerns
- Simplifying probability calculation methods
- Improving code modularity and maintainability

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.3...2025.2.1-pre-1


## [2025.3.1] - 2024-12-19

## 🛠️ Area Occupancy Integration – Release 2025.3.1

### ✨ New Features
- **Configurable State Options for Sensors and Devices**  
  Introduced a new `state_mapping.py` module, allowing users to define which states are considered "active" for devices like doors, windows, media players, appliances, lights, and motion sensors.  
  This provides greater flexibility and accuracy in occupancy detection based on user-defined state configurations.

### 🧰 Refactoring & Architectural Improvements
- **Modular State and Decay Management**  
  - Introduced `StateManagement` and `DecayHandler` utility classes to decouple concerns and improve concurrency handling.
  - Enhanced thread safety, separation of logic, and overall modularity in state tracking and probability decay calculations.

- **Streamlined Configuration Flow**  
  - Unified schema generation through a single `create_schema` method.
  - Refined options and configuration flows for reduced redundancy and improved maintainability.

- **Improved Configuration UX**  
  - Implemented sectioned and collapsible configuration forms with descriptive labels.
  - Created a more intuitive and organized setup process for users.

- **Optimized Coordinator and Service Logic**  
  - Refactored coordinator initialization and service handling for cleaner code structure and better default behavior.

### 🧪 Testing Enhancements
- **Expanded Test Coverage**  
  - Comprehensive tests added for:
    - Probability calculation logic
    - State decay and threshold handling
    - Service layer
    - Coordinator behavior
    - Configuration flow and integration scenarios
  - Enhanced test reliability with better async handling, mock fixtures, and recorder simulation.


## [2024.12.3] - 2024-12-19

## Fixes

Key fixes include resolving issues such as malfunctioning decay, and the threshold not being respected from the sensor.

## New Features

- You can now configure weights for the different types of sensors. This will allow you to favour certain sensors over others. You can access the settings on the last page of the setup. (You can reconfigure this at any time)

### Default Weights

- Motion Weight (default: 0.85) when [on]
- Media Weight (default: 0.70) when [playing or paused]
- Appliance Weight (default: 0.30) when [on]
- Door Weight (default: 0.30) when [off orclosed]
- Window Weight (default: 0.20) when [on or open]
- Light Weight (default: 0.20) when [on]
- Environmental Weight (default: 0.10) when [more than 5% outside the average]

I plan to make the triggers more flexible in the future, so you can configure them to your liking.

## Refactors

- Renamed calculation files for better clarity (‘calculations.py’ to ‘calculate_prob.py’ and introduced ‘calculate_prior.py’).
- Simplified binary sensor implementation by removing unused attributes and methods.
- Enhanced decay mechanisms to manage probabilities over time with improved logging and streamlined logic.
- Improved logging throughout the integration for better traceability and debugging.

<a id='issues'></a>

## <img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/issue-icon.svg" width="20" height="20" alt="Issue Icon" style="vertical-align: middle; margin-right: 5px;"> Issues

<div style='margin-left:1em'>

<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#18](https://github.com/Hankanman/Area-Occupancy-Detection/issues/18) **Decay not working.**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#17](https://github.com/Hankanman/Area-Occupancy-Detection/issues/17) **Threshold is only respected from the configuration and not the sensor** 
</div>

<a id='commits'></a>

## <img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="20" height="20" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> Commits

<div style='margin-left:1em'>

<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5e8225a](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5e8225a932e45625ddddca6f7532bf60ac0dc3b0) fix(area_occupancy): correct prior calculation logic and improve formatting
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [105e906](https://github.com/Hankanman/Area-Occupancy-Detection/commit/105e906183c1d9cb062d8d41351b3f4c2c711fe8) fix(area_occupancy): correct method name and enhance prior update scheduling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e1525ce](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e1525cef3e47fa17daab4084c62a0c9c57ea2a33) refactor(area_occupancy): rename calculation files and improve documentation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [81d42ba](https://github.com/Hankanman/Area-Occupancy-Detection/commit/81d42ba5cf39331a7179ec3fcea69eeab926a512) feat(area_occupancy): enhance prior calculations and integrate type priors management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [7e9934c](https://github.com/Hankanman/Area-Occupancy-Detection/commit/7e9934c4a6cb85cc1b5e163a0236d174840957bb) refactor(area_occupancy): streamline device info handling and improve threshold calculations, clean up coordinator
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [1b0b4c5](https://github.com/Hankanman/Area-Occupancy-Detection/commit/1b0b4c5da7a82b2499cb148336f48dac84f16e95) refactor(area_occupancy): improve data loading and storage handling in coordinator
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [6166f01](https://github.com/Hankanman/Area-Occupancy-Detection/commit/6166f015d09c0acde496f53ba5b93c7be3e1f58f) feat(area_occupancy): enhance probability calculations and refactor prior handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [66ec133](https://github.com/Hankanman/Area-Occupancy-Detection/commit/66ec133cca1b1049dfba37d294aa22829f45efc7) feat(area_occupancy): enhance prior probability calculations and integrate Home Assistant instance
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [3c14a4c](https://github.com/Hankanman/Area-Occupancy-Detection/commit/3c14a4c150cbfaf0d1bfdae71c1c0f38c6c3bbb0) refactor(area_occupancy): integrate probabilities handling and enhance calculator functionality
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8067619](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8067619d18fd9266b30fcda579d770bcd15a527c) feat(area_occupancy): add weights configuration for occupancy detection
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e35854f](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e35854f6121debac9e6aff81cb21fa42c532c3c1) refactor(area_occupancy): enhance sensor probability calculations and streamline entity management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ea497dc](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ea497dc31e874291b6b70e60083586d7e60769c2) feat(area_occupancy): introduce prior probability calculation and enhance coordinator functionality
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [b2b6a22](https://github.com/Hankanman/Area-Occupancy-Detection/commit/b2b6a2221e380c25f23af4984bb4eba6d6a18c22) refactor(area_occupancy): remove unused sensor attributes and streamline helper functions
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [4ca1c72](https://github.com/Hankanman/Area-Occupancy-Detection/commit/4ca1c72f298f08b53f1e3ff281e0fcf897406a03) refactor(area_occupancy): streamline sensor methods and improve unique ID formatting
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [490b88c](https://github.com/Hankanman/Area-Occupancy-Detection/commit/490b88cf1effc80e5009ad0c71431763b6cb5fb2) refactor(area_occupancy): clean up binary sensor code and remove unused attributes
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [de009b7](https://github.com/Hankanman/Area-Occupancy-Detection/commit/de009b7f74d371bea57854afb489e762e68c167d) refactor(area_occupancy): enhance decay logic and improve probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [064b31c](https://github.com/Hankanman/Area-Occupancy-Detection/commit/064b31c3eed1a33aea0b7df58babee25974d3d98) refactor(area_occupancy): enhance logging, update probability calculations, and introduce threshold management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [0a3890c](https://github.com/Hankanman/Area-Occupancy-Detection/commit/0a3890c57cb2c7dc6252dca73bf08c21e2b1310d) refactor(area_occupancy): update migration logic, enhance configuration handling, and introduce combined priors sensor

</div>



## [2024.12.3-pre-5] - 2024-12-19

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.2...2024.12.3-pre-5


## [2024.12.3-pre-4] - 2024-12-19

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.2...2024.12.3-pre-4


## [2024.12.3-pre-2] - 2024-12-18

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.2...2024.12.3-pre-2


## [2024.12.3-pre-1] - 2024-12-18

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.2...2024.12.3-pre-1


## [2024.12.2] - 2024-12-17

Retooled a few things to make the integration more robust and reliable. Updated the calculation logic, a probability will now decay over time as intended and the integration will now handle the maximum limits more gracefully. Also, the setup process has been streamlined and the configuration handling has been unified.

Note: This release removes the previously added services as the timeslot calculations have been removed for now. The services will be added back in a future release.

## Features

- Added a new service to calculate the prior probabilities on demand. (By default they will be calculated every 6 hours to reduce impact on the databse), You may want to run this service manually after changing the configuration.

<a id='issues'></a>

## <img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/issue-icon.svg" width="20" height="20" alt="Issue Icon" style="vertical-align: middle; margin-right: 5px;"> Issues

<div style='margin-left:1em'>

<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#15](https://github.com/Hankanman/Area-Occupancy-Detection/issues/15) **Can't add switch as a light**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#14](https://github.com/Hankanman/Area-Occupancy-Detection/issues/14) **Motion Sensor entities cannot be changed**

</div>

<a id='commits'></a>

## <img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="20" height="20" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> Commits

<div style='margin-left:1em'>

<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [886e249](https://github.com/Hankanman/Area-Occupancy-Detection/commit/886e24901f0899c9e1a8e939939f03808f3fa46e) chore: update bug report template and increment integration version
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [581e9b2](https://github.com/Hankanman/Area-Occupancy-Detection/commit/581e9b26d1d008bdd275c2475d0b6a53135d0550) enhance probability calculation and decay logic
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [1039c2b](https://github.com/Hankanman/Area-Occupancy-Detection/commit/1039c2bfa052f80bf1131878a1e3405b4f1b4a55) update configuration for mock entities and enhance logging
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [4828e18](https://github.com/Hankanman/Area-Occupancy-Detection/commit/4828e18c12ea41814da46d71fcb8727531b17aaa) simplify type definitions and remove unused classes
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [0d706f6](https://github.com/Hankanman/Area-Occupancy-Detection/commit/0d706f6b2aab3cf9304672000481eadbdd2a4a4b) enhance probability calculation logic and ensure maximum limits
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [cd4a2bf](https://github.com/Hankanman/Area-Occupancy-Detection/commit/cd4a2bfc12a06e87e9acde92a13f8ea1238e7823) streamline configuration flow and enhance error handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8af60fd](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8af60fdea9e0f5722f2b9e06d8772a27f025bed4) improve migration logic and update storage version
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [31ef9f3](https://github.com/Hankanman/Area-Occupancy-Detection/commit/31ef9f37115337e7ff1add6d8901cd5000d1cf19) standardize unique ID generation for sensors and thresholds
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [89c1e83](https://github.com/Hankanman/Area-Occupancy-Detection/commit/89c1e83eac8c27a2fcb0da2f64b49ea040751875) improve migration handling and streamline setup process
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [58cf3fb](https://github.com/Hankanman/Area-Occupancy-Detection/commit/58cf3fb5e07e43a2b845734cb9c69523e1398e3b) enhance learned priors handling with age consideration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [12073e5](https://github.com/Hankanman/Area-Occupancy-Detection/commit/12073e55ad07ffa6b42be95db74066975a702da7) enhance learned priors handling and setup process
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [521f993](https://github.com/Hankanman/Area-Occupancy-Detection/commit/521f993c7c25747893a304f76ababdbbeb2fbdfd) add sensor prior calculation and update service
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [2c9d192](https://github.com/Hankanman/Area-Occupancy-Detection/commit/2c9d192bdeefa32c5c56a597137156bc6884c8dc) implement unique ID migration for sensors and thresholds
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [aef5bdc](https://github.com/Hankanman/Area-Occupancy-Detection/commit/aef5bdcb078261c7dc9a9e240718f5319ab13216) unify configuration handling and enhance documentation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [2b8fe51](https://github.com/Hankanman/Area-Occupancy-Detection/commit/2b8fe5130b90bcd75defaf8c68cd71c897ebe506) add .cursorignore and enhance project instructions
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5adc865](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5adc865b0472f5815e6cddcae6c1c9c89fd5bd76) enhance configuration update handling and probability calculation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [09f46c9](https://github.com/Hankanman/Area-Occupancy-Detection/commit/09f46c9e9275c95821efb239bd637af8d620d7ae) unify configuration handling and streamline initialization
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [c5c1073](https://github.com/Hankanman/Area-Occupancy-Detection/commit/c5c1073b7061415329829656897154ef49435065) enhance configuration handling and storage migration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e574331](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e574331bb5f6e7326f93d5353ac0d0117de670b9) unify synchronous and asynchronous trigger handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [bebf4c3](https://github.com/Hankanman/Area-Occupancy-Detection/commit/bebf4c377dd64ae2a12e8e419dbd3b3f9dad5469) streamline probability calculation and decay handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [a9e73f3](https://github.com/Hankanman/Area-Occupancy-Detection/commit/a9e73f360bb0695a1ea99f29a738be6193c008f0) remove redundant user input handling in options flow
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8e5ddcf](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8e5ddcfaca8f4ee0233277954fd6741d6a0e7a83) implement motion sensor migration and update configuration handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [794c3be](https://github.com/Hankanman/Area-Occupancy-Detection/commit/794c3be16dfe4209c4db241f3570db2bc843f836) update string keys for clarity in configuration and translations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [6272dad](https://github.com/Hankanman/Area-Occupancy-Detection/commit/6272daddb1536289ef0857792f3fc0d3a49f89f6) enhance coordinator initialization and sensor setup
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [818fef2](https://github.com/Hankanman/Area-Occupancy-Detection/commit/818fef2cb1c2e944820410beb172c1af04d9cc97) streamline configuration and enhance logging in area occupancy integration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ac19727](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ac19727d4374e35c48bde70501193a82d3002e0d) Prevent startup blocking, enhance logging and initialization flow in coordinator
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [88f9168](https://github.com/Hankanman/Area-Occupancy-Detection/commit/88f9168b63331b10016c5a7c7c92345030d0c428) enhance state initialization and debouncing in coordinator
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ae8e76b](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ae8e76b99d76d642d3095adb4ca9de3ede3ffcb8) implement an ordered cache for baseline calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [7a83fbf](https://github.com/Hankanman/Area-Occupancy-Detection/commit/7a83fbfa90aa2510fbfbde49706c9dab2ad860f7) improve state tracking and debounce handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [c85114a](https://github.com/Hankanman/Area-Occupancy-Detection/commit/c85114a966cc45e09f7a53052c052aa4a01ff5d3) Update README.md

</div>

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2


## [2024.12.2-pre-7] - 2024-12-17

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2-pre-7


## [2024.12.2-pre-6] - 2024-12-17

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2-pre-6


## [2024.12.2-pre-5] - 2024-12-17

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2-pre-5


## [2024.12.2-pre-4] - 2024-12-17

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2-pre-4


## [2024.12.2-pre-3] - 2024-12-17

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2-pre-3


## [2024.12.2-pre-2] - 2024-12-17

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2-pre-2


## [2024.12.2-pre-1] - 2024-12-16

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.12.1...2024.12.2-pre-1


## [2024.12.1] - 2024-12-15

# A Thank You

The last few days have been a whirlwind, this integration is something i always wanted to build and finally found the time, and the motivation to do so. I am grateful for the support so far and a big thank you to BeardedTinker for featuring the integration before it even got into the HACS store! He has some great videos on Home Assistant.

- Featured here: [More of the HACS - Bluesky, Area Occupancy and Better moments](https://youtu.be/yzwa4Mmzvcc?t=320&si=ngnuKESfD-_23Yny#)
- Sub to [BeardedTinker](https://www.youtube.com/watch?v=yzwa4Mmzvcc)!

Okay on to the release notes, this is a big one!

## New Features

### All the Devices

- All motion and occupancy sensors are now supported. (PIR and mmWave or anything else with a device class of motion or occupancy or presence)
- Door and Window sensors are now supported. (This is experimental as some may prefer the door being open to mean present and others closed, for now it is closed, but this will be configurable in the future, if you are getting false positives, I suggest excluding doors and windows from your config for now)
- Lights are now supported. (If the light is on there is probably someone in the room)
- The appliances section now supports a much broader filter essentially anything that is a fan, switch or binary sensor.
- Media players are now supported. (If the media player is playing there is probably someone in the room)

### Added Services

- **Export calculations** - This service exports the current probability calculations to a json file in the config directory. This is useful for debugging and understanding how the integration is working.
- **Export historical analysis** - This service exports the historical data to a json file in the config directory. This is useful for debugging and understanding how the integration is working.
- **Run Historical Analysis** - This service runs the historical analysis on the current data. This is useful for ensuring your data is up to date before exporting it.

### Added Intelligence

- **Decay** - The decay of the probability calculations is now configurable. This is how quickly the probability of someone being in a room decays over time.
- **Decay Window** - The decay window is how long the decay is applied for. This is useful for ensuring that the probability of someone being in a room decays over time, but not too quickly.
- **Delay Before Starting Decay** - This is the delay before the decay starts. This is useful for ensuring that the probability of someone being in a room does not decay too quickly after they leave.
- **ALL THE MATH** - The probability calculations have been enhanced to be more accurate and to take into account the decay settings. Calculated priors and probabilities, optimised defaults and much much more. I have spent a lot of time on this and I am very happy with the results so far.

## BREAKING CHANGES

I have attempted my best efforst at including full migration between version but a huge amount of the code changed, this will be much less likely in the future. If you are upgrading from a previous version, please ensure you have a backup of your configuration and that you have read the new documentation on the wiki, it is possible you will need to delete your configurations and restart Home Assistant to get the new version working. I am sorry for the inconvenience but this is a major update for the better i promise!!!

## New Documentation

You can read all about how everything works and more on the [Wiki](https://github.com/Hankanman/Area-Occupancy-Detection/wiki)! I hope it is helpful, there were a lot of people wondering whats the point of it or how it works so i hope this provides clarity!

<a id='issues'></a>

## <img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/issue-icon.svg" width="20" height="20" alt="Issue Icon" style="vertical-align: middle; margin-right: 5px;"> Issues

<div style='margin-left:1em'>

<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#9](https://github.com/Hankanman/Area-Occupancy-Detection/issues/9) **Bad link in the README.md**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#8](https://github.com/Hankanman/Area-Occupancy-Detection/issues/8) **Door "closed" treated as active trigger**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#7](https://github.com/Hankanman/Area-Occupancy-Detection/issues/7) **Add device class Occupancy**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#6](https://github.com/Hankanman/Area-Occupancy-Detection/issues/6) **Aqara FP1 and FP2 sensors not working also EPL.**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#5](https://github.com/Hankanman/Area-Occupancy-Detection/issues/5) **Motion Category should include Occupancy type**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#4](https://github.com/Hankanman/Area-Occupancy-Detection/pull/4) **Fix My Home Assistant link**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#3](https://github.com/Hankanman/Area-Occupancy-Detection/issues/3) **Area Occupancy Detection - Device selection dropdown not properly populated**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#2](https://github.com/Hankanman/Area-Occupancy-Detection/pull/2) **main to dev**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#1](https://github.com/Hankanman/Area-Occupancy-Detection/pull/1) **chore(deps-dev): bump black from 23.9.1 to 24.3.0**
</div>

<a id='pull-requests'></a>

## <img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/pull-request-icon.svg" width="20" height="20" alt="Pull Request Icon" style="vertical-align: middle; margin-right: 5px;"> Pull Requests

<div style='margin-left:1em'>

<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#4](https://github.com/Hankanman/Area-Occupancy-Detection/pull/4) **Fix My Home Assistant link**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#2](https://github.com/Hankanman/Area-Occupancy-Detection/pull/2) **main to dev**
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/comment-icon.svg" width="16" height="16" alt="PullRequest Icon" style="vertical-align: middle; margin-right: 5px;"> [#1](https://github.com/Hankanman/Area-Occupancy-Detection/pull/1) **chore(deps-dev): bump black from 23.9.1 to 24.3.0**
</div>

<a id='commits'></a>

## <img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="20" height="20" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> Commits

<div style='margin-left:1em'>

<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [44fdf79](https://github.com/Hankanman/Area-Occupancy-Detection/commit/44fdf79f1804126ae3aba96628c3fabff1b344b2) refactor: implement periodic historical analysis and enhance probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [f62860b](https://github.com/Hankanman/Area-Occupancy-Detection/commit/f62860bb2037e6032a4add3cfaf88936a977ad10) refactor: enhance probability calculations and update ignore files
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [4251d45](https://github.com/Hankanman/Area-Occupancy-Detection/commit/4251d455efa92cb44ffadec8e3dfbe400f6e9b44) refactor: streamline sensor classes and introduce helper functions
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5951d2e](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5951d2ec5b4b9bac8d0d8da10fef2cff5f6bc474) refactor: enhance sensor classes and introduce window sensor support
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [14259fd](https://github.com/Hankanman/Area-Occupancy-Detection/commit/14259fdfc9b4ee1324e5b38c0e77e85be41cb8d2) refactor: enhance door and window sensor state handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [7d140c2](https://github.com/Hankanman/Area-Occupancy-Detection/commit/7d140c24f7a3b7536312dbe98a614f31a5ef11a0) refactor: enhance device configuration and update decay settings
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [03c1685](https://github.com/Hankanman/Area-Occupancy-Detection/commit/03c168510892b8ad1bcfa5ee733c2940df2216bd) refactor: update README and remove testing guide
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [6c88cd6](https://github.com/Hankanman/Area-Occupancy-Detection/commit/6c88cd6b5036ba4a16a725755e513fc6827773a6) refactor: add window sensor support and enhance configuration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [4b244dd](https://github.com/Hankanman/Area-Occupancy-Detection/commit/4b244ddabc3aa7af0365e875bc2e71ea7adeea72) refactor: enhance sensor configuration and probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [55495d1](https://github.com/Hankanman/Area-Occupancy-Detection/commit/55495d1f497cdf4950bf69cd23822091a69e1a91) refactor: improve probability calculations and update configuration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [b436147](https://github.com/Hankanman/Area-Occupancy-Detection/commit/b436147584d8b4b2d93ab0b7f3b029537872df7c) refactor: remove decay type configuration and enhance probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8d5a4b5](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8d5a4b547cb867a2b0e541024750476bed0f57a3) refactor: update Python version and enhance probability calculation logic
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [eb3ec49](https://github.com/Hankanman/Area-Occupancy-Detection/commit/eb3ec498c98f06188f32f1425a718354f69ea4aa) refactor: enhance binary sensor and sensor classes for clarity and functionality
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [58358a1](https://github.com/Hankanman/Area-Occupancy-Detection/commit/58358a1f4a27ef584cd87ad26f6f8200337d37bd) refactor: change logging level to debug for probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [c8032a8](https://github.com/Hankanman/Area-Occupancy-Detection/commit/c8032a893e7a42fc8838f6a138d8f70f89e1c1b5) refactor: update output path handling and enhance export logging
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [d987092](https://github.com/Hankanman/Area-Occupancy-Detection/commit/d9870920025972f8c35abd4f48aba4191353dc80) refactor: enhance occupancy detection and probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [d57d45f](https://github.com/Hankanman/Area-Occupancy-Detection/commit/d57d45f5d1c199c73afe3d6e1652c602c254e311) refactor: enhance probability calculations and remove historical analysis
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5b72cc5](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5b72cc5919e587dea16c9736ce01a9f8e9f623fa) refactor: enhance probability calculations and add export services
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [c324cf8](https://github.com/Hankanman/Area-Occupancy-Detection/commit/c324cf830481dfab18e2172bdaf43116b19d7da8) refactor: remove service implementations and related configuration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8c5c062](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8c5c06231c73c1563f4f6d9ba6f40028bb23a869) refactor: enhance historical analysis logic and error handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [bc89f39](https://github.com/Hankanman/Area-Occupancy-Detection/commit/bc89f39194a47066a8cef660cdcab800a762e657) refactor: streamline historical analysis and enhance state validation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [b11a243](https://github.com/Hankanman/Area-Occupancy-Detection/commit/b11a243cc176156ff3923de98e042107b398e70a) refactor: simplify occupancy detection and streamline probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [81dd0b7](https://github.com/Hankanman/Area-Occupancy-Detection/commit/81dd0b7c55f1174cb033ba7add2a40f746b32178) refactor: remove minimum confidence parameter and streamline calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ad0c286](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ad0c286f591f58c387fc56e68964fac61cfa1e57) refactor: update threshold handling and improve configuration validation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [c2c4013](https://github.com/Hankanman/Area-Occupancy-Detection/commit/c2c4013b80dd6d587a071504163f8fb0de148ae1) refactor: enhance occupancy threshold management and introduce number platform
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [f4e75dd](https://github.com/Hankanman/Area-Occupancy-Detection/commit/f4e75dd6398bb51ab5c30f0130cdfb188c4284c6) refactor: remove base.py and consolidate sensor logic
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [0e03d5a](https://github.com/Hankanman/Area-Occupancy-Detection/commit/0e03d5ab6e833d262fc180816f49a3ffb1c33a28) refactor: simplify binary sensor setup and improve entity description creation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5865f1e](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5865f1e98843dc100b6cab75aa1f2d9e2e78761d) refactor: remove prior sensors and consolidate logic
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [f37f344](https://github.com/Hankanman/Area-Occupancy-Detection/commit/f37f3447ffd6809efb9cfa0575254090cfe20ba0) refactor: remove debug logging for minor probability changes
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [189a901](https://github.com/Hankanman/Area-Occupancy-Detection/commit/189a901a9b9ee5b29124d7e93d8f8cc0fcb6d3b0) refactor: implement debouncing for state updates and enhance logging
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e87f9e0](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e87f9e0774ca760b2faf9302107f0394f53ff127) refactor: enhance configuration flow and validation logic
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [a5f2e12](https://github.com/Hankanman/Area-Occupancy-Detection/commit/a5f2e12578ff0f43208ee36e16bd82977010cfcb) refactor: consolidate configuration validation and improve error handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [2960c68](https://github.com/Hankanman/Area-Occupancy-Detection/commit/2960c68c58f92328539f048bc19d4550cb0f497c) refactor: streamline coordinator logic and improve state tracking
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [7e2a3f9](https://github.com/Hankanman/Area-Occupancy-Detection/commit/7e2a3f92f97a1af9ec14e490550e894de18f3bdf) refactor: optimize coordinator logic and enhance state tracking
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [92dc117](https://github.com/Hankanman/Area-Occupancy-Detection/commit/92dc1176babd3947a9e1361d1b2964891a3ace6f) refactor: update default configuration values and enhance historical data handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [031687f](https://github.com/Hankanman/Area-Occupancy-Detection/commit/031687f4216e70c9c690c4c1867ee158ac1cab55) refactor: enhance storage management and error handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [1eb6b30](https://github.com/Hankanman/Area-Occupancy-Detection/commit/1eb6b30cbc170a8afb916ec39d5a71950572b83b) refactor: optimize data storage and improve entity availability checks
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [b7d7e16](https://github.com/Hankanman/Area-Occupancy-Detection/commit/b7d7e16bc4fb34241a3dc1ac5583e1c2921ea381) refactor: enhance configuration and add new services
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ccd98ce](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ccd98ce9b3bc2993499cef3010d781dc45a4be75) refactor: enhance configuration migration and validation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [495c03d](https://github.com/Hankanman/Area-Occupancy-Detection/commit/495c03df095bc9c73d1a27cf5db08f066842bdd1) refactor: streamline setup processes and enhance background operations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [0b7338d](https://github.com/Hankanman/Area-Occupancy-Detection/commit/0b7338d7b3488cc3df9e141c5f71361026d5e057) refactor: enhance probability calculations and entity state management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [cb12d84](https://github.com/Hankanman/Area-Occupancy-Detection/commit/cb12d84ce451527df540d52d990b5d3ffc9fb8d8) refactor: improve configuration validation and entity management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [2b94e88](https://github.com/Hankanman/Area-Occupancy-Detection/commit/2b94e88598983fa543f0eeabc7d7180a7921106d) refactor: reorganize integration structure and enhance component definitions
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [879eb2f](https://github.com/Hankanman/Area-Occupancy-Detection/commit/879eb2fd60b12cc22ab6670eb69c17f7a0b0eeb9) refactor: enhance coordinator and sensor state management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [84c8b7b](https://github.com/Hankanman/Area-Occupancy-Detection/commit/84c8b7ba0cdd76b2f76ba21e05016f6aaf8290db) chore(devcontainer): update configuration for improved development environment
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [f26ae9d](https://github.com/Hankanman/Area-Occupancy-Detection/commit/f26ae9d9aace77fd08a85685610cae1616dba7ea) feat: add project guidelines and devcontainer configuration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [63eb441](https://github.com/Hankanman/Area-Occupancy-Detection/commit/63eb44143560ac5d991d8db3167f73dbb526020b) refactor: replace pending task updates with debouncer
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [d9daf65](https://github.com/Hankanman/Area-Occupancy-Detection/commit/d9daf655f3582c0fd4584322d33d080b3b4909b5) refactor: optimize sensor state management and update handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [139ab01](https://github.com/Hankanman/Area-Occupancy-Detection/commit/139ab01b72267d7d6d8e50aa5cfbe643fca02b93) refactor: enhance error handling by adding pylint disable comments for broad exceptions
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e298b97](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e298b97f6addbb849e808381de372ba4734ae440) feat: add Example_Slots.json for time slot probability data
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5bfd1e2](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5bfd1e270abd6f1830029f943c74ec8cb453e906) refactor: reorganize type definitions and enhance clarity for historical analysis and probability calculations
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8875f18](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8875f181d2fc92bba842f53bb35d5d79e201ff6c) feat: add service implementations for calculating prior probabilities and timeslot analysis
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ce18187](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ce181870d49c1bce3448dd46ecd6f611c9ca6e1a) refactor: remove EnvironmentalPriorSensor integration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [26c342a](https://github.com/Hankanman/Area-Occupancy-Detection/commit/26c342aa3d098a9d504fb51b1edcc19dd991fc48) refactor: simplify prior probability calculation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [01446b0](https://github.com/Hankanman/Area-Occupancy-Detection/commit/01446b089be891f9ca7da87341944f600785460e) refactor: simplify historical analysis
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [f59b464](https://github.com/Hankanman/Area-Occupancy-Detection/commit/f59b4645840cddcf41c517281cc7e459bcfba09e) refactor: optimize data handling and improve performance metrics
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [656d985](https://github.com/Hankanman/Area-Occupancy-Detection/commit/656d98588a07b631defdd7b9343ec31abfbb25dc) refactor: add service-related constants for time and output management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [05aac8d](https://github.com/Hankanman/Area-Occupancy-Detection/commit/05aac8dd55c46454adf1ee22d27a2d6e30f8f7e6) refactor: enhance service setup and unload logic
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e737481](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e737481626338c27e3c2e9728373540bf9806ff6) refactor: streamline probability calculation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [eb801f3](https://github.com/Hankanman/Area-Occupancy-Detection/commit/eb801f31c5c13b5933cee7efc58f4ead7d0e885b) refactor(environmental_detection): remove obsolete environmental detection module
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [1e2ad65](https://github.com/Hankanman/Area-Occupancy-Detection/commit/1e2ad654f1263a214e2e6cd76d4b225a7bd2b482) refactor: improve probability calculation logic and structure
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [6d84257](https://github.com/Hankanman/Area-Occupancy-Detection/commit/6d84257e99af93d79416c5af4cc86676fbf4f9b4) refactor: improve sensor attributes and add detailed device info
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [49b3f8e](https://github.com/Hankanman/Area-Occupancy-Detection/commit/49b3f8efef4d2bc6072a021f7a162ba6700bc8b1) feat: add prior probability sensors
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [eb70cc5](https://github.com/Hankanman/Area-Occupancy-Detection/commit/eb70cc5e8d07ffb7d53d1bc63bdaec7cde82e81c) feat: enhance processing of occupancy analysis results
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [3627e77](https://github.com/Hankanman/Area-Occupancy-Detection/commit/3627e775df3ec2eec0ca34f1607fbaaa20f611b2) refactor: use TypedDict and update config flow for Home Assistant
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [9d66f0e](https://github.com/Hankanman/Area-Occupancy-Detection/commit/9d66f0ea441059197b1200412213328e6f3f34fc) refactor: centralise type management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ccea3c2](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ccea3c28cca3f6fb19366dffb51527834b377432) refactor: remove YAML config in favor of constants file
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5e4b799](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5e4b7995c9aa8ef3ed0c5eaf12fadab693dfa944) refactor: introduce area_id for unique identification and configuration management
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [71b209d](https://github.com/Hankanman/Area-Occupancy-Detection/commit/71b209dae090edc83797819edcf3525e4f7935c0) refactor: add description placeholders for form steps
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [7abe16f](https://github.com/Hankanman/Area-Occupancy-Detection/commit/7abe16ff079aee771e24038ae99f9b1b8bc1c924) refactor: update device name handling and sensor naming
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [75c57b4](https://github.com/Hankanman/Area-Occupancy-Detection/commit/75c57b439f7a63dbf852b2fe7e93a827ac143a9c) refactor: consolidate motion sensor config to core
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [9b09e71](https://github.com/Hankanman/Area-Occupancy-Detection/commit/9b09e71eb4c586a5896a0a18a27ed9baf7876d6c) refactor(config_flow): streamline configuration steps and enhance options handling
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [d46f1e3](https://github.com/Hankanman/Area-Occupancy-Detection/commit/d46f1e39282abbef1cad9c88774823bb56593831) refactor: streamline configuration and validation process
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e139846](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e139846d3e44dae92e06391f1e15824d7d274f39) refactor: improve config validations and logging
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8ff0b8a](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8ff0b8a6668766cd38a3312022ba8e74dbf5b160) feat: enhance configuration and pattern analysis
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [2a6963d](https://github.com/Hankanman/Area-Occupancy-Detection/commit/2a6963d91d01a66adbd09c4ee6e1678b71299a0d) feat: enhance area occupancy detection with new probabilistic model and sensor support
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [3c6797c](https://github.com/Hankanman/Area-Occupancy-Detection/commit/3c6797c2fa3ab45ccec71a7f1a5290b397780772) refactor: reorganize component structure for improved maintainability
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8c68c1a](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8c68c1adff3485cc41915ce49c8e90f3f70d56f0) feat: add pattern-based probability adjustment with safety bounds
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [23d7911](https://github.com/Hankanman/Area-Occupancy-Detection/commit/23d79118ae6b28175831e25c2d1ff56537a937a1) Update manifest.json
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [2c5cbca](https://github.com/Hankanman/Area-Occupancy-Detection/commit/2c5cbca30f1b3a2f5b37ebc54d662d69d84999d3) feat: integrate historical data into occupancy calculation
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [abc0935](https://github.com/Hankanman/Area-Occupancy-Detection/commit/abc093578f3496ec357c481e1ca3f439193721e9) feat: enhance area occupancy detection setup with media devices and appliances support
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [a876926](https://github.com/Hankanman/Area-Occupancy-Detection/commit/a87692657b9edfc44326cccb048a1372459e0d84) feat: add default probabilities configuration for area occupancy detection
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [7870cfd](https://github.com/Hankanman/Area-Occupancy-Detection/commit/7870cfd9b985f73e94227f21c1d40d1398f82064) feat: improve area occupancy configuration with additional media device attributes
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5bc4d73](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5bc4d73e3856e21184449f20a50efe7175328bc0) feat: add media devices and appliances support in area occupancy configuration
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [94a242b](https://github.com/Hankanman/Area-Occupancy-Detection/commit/94a242bcebe0cbf56150e93994c121e24ec4e677) feat: enhance configuration flow to support media devices and appliances
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [58e211c](https://github.com/Hankanman/Area-Occupancy-Detection/commit/58e211cc361b98afa58634223b18dbd1020696fd) feat: implement new configuration options for enhanced user experience
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [ad32ec0](https://github.com/Hankanman/Area-Occupancy-Detection/commit/ad32ec0bf7ef3964e2fe68b55fcac7be0a4dd48f) feat: add media and appliance states attributes to AreaOccupancySensorBase
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [e35db1a](https://github.com/Hankanman/Area-Occupancy-Detection/commit/e35db1a4573406e2928f61540c2348f50f3026a2) feat: enhance Area Occupancy Detection with base probability configuration and migration of device states
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [8e105ab](https://github.com/Hankanman/Area-Occupancy-Detection/commit/8e105ab3868f6c8fe54bc876258f7c40c61997d4) fix: update YAML file association in VSCode settings
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [bc205e9](https://github.com/Hankanman/Area-Occupancy-Detection/commit/bc205e9ff60f64bbd5bc634519a01edfe0681214) Merge pull request #2 from Hankanman/main
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [d68076b](https://github.com/Hankanman/Area-Occupancy-Detection/commit/d68076b4d4bbf3e42c93094e0bf085b8b3ee6f34) feat: add user guide for Smart Lighting Control blueprint
<img src="https://raw.githubusercontent.com/Hankanman/Changelog-Weaver/refs/heads/main/assets/commit-icon.svg" width="16" height="16" alt="Commit Icon" style="vertical-align: middle; margin-right: 5px;"> [5b8f387](https://github.com/Hankanman/Area-Occupancy-Detection/commit/5b8f38734d24651124674b8ce9a134e3e1ced0aa) feat: add smart lighting control blueprint for automated room lighting management

</div>



## [2024.12.1-pre-5] - 2024-12-15

## What's Changed
* main to dev by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

## New Contributors
* @Hankanman made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.11.2...2024.12.1-pre-5


## [2024.12.1-pre-4] - 2024-12-14

## What's Changed
* main to dev by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

## New Contributors
* @Hankanman made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.11.2...2024.12.1-pre-4


## [2024.12.1-pre-3] - 2024-12-14

## What's Changed
* main to dev by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

## New Contributors
* @Hankanman made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.11.2...2024.12.1-pre-3


## [2024.12.1-pre-2] - 2024-12-14

## What's Changed
* main to dev by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

## New Contributors
* @Hankanman made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.11.2...2024.12.1-pre-2


## [2024.12.1-pre-1] - 2024-12-13

## What's Changed
* main to dev by @Hankanman in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

## New Contributors
* @Hankanman made their first contribution in https://github.com/Hankanman/Area-Occupancy-Detection/pull/2

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.11.2...2024.12.1-pre-1


## [2024.11.2] - 2024-11-25

**Full Changelog**: https://github.com/Hankanman/Area-Occupancy-Detection/compare/2024.11.1...2024.11.2


## [2024.11.2-pre-7] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/compare/2024.11.1...2024.11.2-pre-7


## [2024.11.2-pre-6] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/compare/2024.11.1...2024.11.2-pre-6


## [2024.11.2-pre-5] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/compare/2024.11.1...2024.11.2-pre-5


## [2024.11.2-pre-4] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/compare/2024.11.1...2024.11.2-pre-4


## [2024.11.2-pre-3] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/compare/2024.11.1...2024.11.2-pre-3


## [2024.11.2-pre-2] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/compare/2024.11.1...2024.11.2-pre-2


## [2024.11.2-pre] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/compare/2024.11.1...2024.11.2-pre


## [2024.11.1-pre] - 2024-11-24

_No release notes available._


## [2024.11.1] - 2024-11-24

**Full Changelog**: https://github.com/Hankanman/Room-Occupancy-Detection/commits/2024.11.1

