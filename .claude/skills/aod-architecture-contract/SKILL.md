---
name: aod-architecture-contract
description: Use when you need to understand WHY Area Occupancy Detection is built the way it is before changing coordinator.py, area/area.py, utils.py, data/analysis.py, data/decay.py, data/entity.py, or anything under db/ — the single-coordinator-many-areas design, the exact probability pipeline order, the three timers, DB session/executor rules, or the list of invariants a change must not violate. Load this before any PR that touches Bayesian math, timers, DB schema, or entity evidence, and whenever asked "why does this work this way" or "is it safe to change X".
---

# AOD Architecture Contract

## What this covers

The load-bearing design decisions in Area Occupancy Detection (AOD) and why they exist: the single-coordinator/many-areas shape, the exact probability pipeline, the three background timers, entity/evidence semantics, the database layout and its concurrency rules, and the hard invariants a change must never break. It also states known weak points plainly so you don't mistake "it's unmerged/untested" for "it's proven."

## When NOT to use this

- The math itself (formulas, worked examples, why sigmoid/logit) → `bayesian-occupancy-reference`
- How to change something safely (process, review, versioning rules) → `aod-change-control`
- Reproducing/fixing a specific bug → `aod-debugging-playbook` or `aod-failure-archaeology`
- Config keys, defaults, options-flow mechanics → `aod-config-and-flags`
- Prior/likelihood accuracy work specifically → `aod-learning-accuracy-campaign`

## Status: PR #454 (adjacent-areas) merged 2026-07-06

PR #454 merged into `main` on 2026-07-06 (main HEAD `17b71d2`); the working tree is now on `main`. Adjacency (phase-3 boost + decay-modifier, the `area_transitions` table, and the 13-step analysis pipeline) is **current on `main`**, not a pending branch — described as such throughout this file. It remains functionally unvalidated against real-home data (see "Known weak points" below), which is a maturity caveat, not a merge-status one. Re-verify with `gh pr view 454 --json state,mergedAt` if in doubt.

## The core design: one coordinator, many areas

```
AreaOccupancyCoordinator (global singleton, one per config entry)
├── AreaOccupancyDB (one SQLite DB, shared by all areas)
├── IntegrationConfig (global settings)
└── areas: dict[str, Area]        # coordinator.py:62
    └── Area (per-room instance)
```

Why a single coordinator instead of one per area (the pattern most HA integrations use):

- **One state listener for all entities.** `coordinator.py` registers exactly one `async_track_state_change_event` call for the union of every configured entity across every area, stored under the sentinel key `"_all"` in `_area_state_listeners`. The inline comment says it outright: *"Create single listener for all entities (more efficient than per-area listeners)"* (`coordinator.py` ~line 1192). On each firing, the callback maps the changed entity to whichever area(s) contain it and only refreshes those areas' evidence — one listener, N-area fan-out, not N listeners.
- **One shared database.** All areas write into the same `AreaOccupancyDB` instance/session factory — no per-area DB files, no per-area connection pools.
- **One analysis pipeline.** The hourly analysis pipeline (below) runs once per coordinator tick and iterates areas internally, rather than each area independently scheduling its own sync/aggregate/learn cycle.

Consequence: a bug in the shared listener, shared DB session handling, or shared timer affects every configured area simultaneously — there is no per-area blast-radius containment. Treat coordinator-level code with more caution than area-level code for exactly this reason.

## The probability pipeline (verified on `main`)

Two-phase calculation in `area/area.py`, all math in `utils.py`:

1. **`_base_probability()`** — sensor-only, logit-space weighted evidence, no activity/adjacency:
   - `presence = presence_probability()` — combines MOTION/MEDIA/APPLIANCE/DOOR/WINDOW/COVER/POWER/SLEEP evidence in logit space, each entity's contribution scaled by `effective_weight (weight × information_gain) × strength_multiplier`.
   - `env = environmental_confidence()` — same mechanism over environmental sensor types; returns exactly `0.5` (neutral) when the area has zero environmental sensors configured.
   - If `env == 0.5` exactly, `_base_probability()` returns `presence` directly — the 80/20 logit blend (`combined_probability`, weights presence 0.8 / env 0.2) is **skipped** so a no-env-sensor area is never compressed toward neutral (`area/area.py::_base_probability`, comment explains this explicitly).
2. **`probability()`** — activity boost:
   - `base = _base_probability()`; `is_occupied = base >= config.threshold`.
   - `activity = detect_activity(self, base_probability=base, is_occupied=is_occupied)`.
   - If `activity.activity_id` is `UNOCCUPIED` or `IDLE`, return `base` unchanged.
   - Otherwise `apply_activity_boost(base, activity.occupancy_boost, activity.confidence)` — a logit-space additive boost, `sigmoid(logit(base) + boost*confidence)`.
3. **`occupied()`** — `return self.probability() >= self.config.threshold` (`area/area.py::occupied`). Threshold default is 50.0/100 (0.50).

**Load-bearing decision (PR #486, merged 2026-07-06):** occupancy threshold comparison and every downstream decision (wasp-in-box, activity scoring) operate on the **internal unrounded float** probability. The user-facing "Sensor state precision" setting (0–2 decimals, default 2) only affects *published sensor state formatting* (`format_float` in `sensor.py`) — never the value fed into `occupied()`. PR #486's own description states this as the reason the change was "functionally safe": *"All decision logic — `area.occupied()`, wasp-in-box, activity scoring, thresholds — operates on internal unrounded floats. The sensor states are publication-only formatting."* If you ever see code comparing a *rounded/published* value to the threshold, that is a regression of this decision — flag it.

Every probability value is clamped through `clamp_probability()` (`utils.py::clamp_probability`) to `[MIN_PROBABILITY=0.01, MAX_PROBABILITY=0.99]` before use; `logit()` clamps its input first so `logit(0)`/`logit(1)` never raise. NaN clamps to MAX (with a warning log), ±inf clamps to MAX/MIN respectively.

### Dead code trap: `utils.py::bayesian_probability()` is NOT the live calculation

`utils.py` still defines a classic naive-Bayes log-odds function `bayesian_probability()`, and CLAUDE.md's own "Modifying Bayesian Calculation" workflow points at it — but it has **zero production call sites**. It was superseded by the sigmoid/logit pipeline above (introduced in PR #353, "Add sigmoid-based occupancy detection framework"). It survives only because ~25 unit tests in `tests/test_utils.py` still exercise it directly. If you're asked to "modify the Bayesian calculation," the real entry points are `presence_probability()`, `environmental_confidence()`, `combined_probability()`, `apply_activity_boost()` in `utils.py`, plus `Area._base_probability()`/`Area.probability()` in `area/area.py` — not `bayesian_probability()`.

### Phase 3 (merged PR #454, 2026-07-06): adjacency boost

After activity boost, a **phase 3** now runs on `main`: `boost = coordinator.adjacency_boost_for(area)`; if present, `result = sigmoid(logit(result) + boost.logit_contribution)`. Its invariants are covered in their own subsection below.

## The three timers

| Timer | Interval | Const | Why this cadence |
|---|---|---|---|
| Decay | 10s | `DECAY_INTERVAL` (`const.py:342`, main) | Fast enough that probability decay feels continuous to automations without re-running the full analysis pipeline; only refreshes the coordinator if at least one area has `decay.enabled`. |
| Analysis | 3600s (1h) | `ANALYSIS_INTERVAL` (`const.py:343`) | Expensive (DB sync, aggregation, prior/correlation recompute) — hourly balances freshness against DB/CPU load. First run is deliberately deferred via `homeassistant.helpers.start.async_at_started` plus an additional 5-minute delay, specifically so it never competes with HA's own startup. |
| Save | 600s (10min) | `SAVE_INTERVAL` (`const.py:344`) | Periodic persistence so a crash between saves loses at most 10 minutes of learned state, without writing to SQLite continuously. |

All three timers check `self._stop_requested` **both before doing work and before rearming** (`coordinator.py::_handle_decay_timer` / `_handle_save_timer` / analysis equivalent), and `EVENT_HOMEASSISTANT_STOP` synchronously cancels all three registered timer handles. This exists to close a race where a timer that had already fired (but not yet run its callback body) could otherwise still kick off executor work after shutdown began — read the inline comments in `_handle_decay_timer`/`_handle_save_timer` before touching this logic; they document why the check appears twice, not once.

Analysis timer retry-on-failure backoff is **15 minutes**, not the normal hourly cadence (`coordinator.py`, analysis timer handler: `if _failed: next_update = _now + timedelta(minutes=15)`), so a transient failure (recorder purge collision, momentary DB lock) doesn't wait a full hour to retry.

### The hourly analysis pipeline — exact order (verified on `main`: **13 steps**, since PR #454 merged 2026-07-06)

`data/analysis.py::run_full_analysis()`, `total_steps = 13`:

1. `sync_states` — import recent entity state changes from the HA recorder
2. `health_check_and_prune` — DB integrity check (`PRAGMA integrity_check`) + prune intervals older than `RETENTION_DAYS`
3. `sensor_health_check` — per-entity anomaly detection → HA repairs
4. `populate_occupied_intervals_cache` — motion-sensor-only ground truth, only if cache invalid/stale
5. `interval_aggregation` — raw → daily/weekly/monthly rollups
6. `numeric_aggregation` — raw numeric samples → hourly/weekly (feeds Gaussian correlation)
7. `recalculate_priors` — per-area global prior + 168 time-priors
8. `correlation_analysis` — sensor/occupancy statistical correlation (needs ≥50 samples, see invariants)
9. `transition_learning` — learns area-to-area adjacency transitions feeding phase-3 boost (PR #454)
10. `pipeline_health_check` — area-scope anomalies (stale cache, slow analysis, insufficient priors, correlation failure ratio)
11. `save_data_before_refresh`
12. `refresh_coordinator` — recompute `probability()` for every area
13. `save_data_after_refresh`

Each step is wrapped by an internal `_run_step()` helper that times it independently and catches all exceptions into a `failed_steps` list — **a failing step does not abort the run; all 13 steps always attempt to execute.** Only after all steps run does a non-empty `failed_steps` raise `HomeAssistantError` (triggering the 15-minute retry backoff above). If `EVENT_HOMEASSISTANT_STOP` fires mid-run, remaining steps are skipped, the run is marked cancelled, and `_last_analysis_duration_ms` is deliberately **not** written — so a fast, aborted partial run can never mask a genuinely slow prior run in the `SLOW_ANALYSIS` health check.

## Entity / evidence semantics

`Entity.evidence` (`data/entity.py::evidence`) is a tri-state property: `None` when the raw HA state is unavailable/unknown/empty/NaN, else `True`/`False` from `active_states` (semantically mapped on/off ↔ open/closed) or `active_range` (numeric bounds, overridden by `learned_active_range` when correlation analysis has produced one).

`Entity.has_new_evidence()` (`data/entity.py::has_new_evidence`, ~line 555) is the **single gate** that starts/stops decay and decides whether a state change is worth a coordinator refresh:
- unavailable-with-prior-`True` → starts decay
- becoming-available-with-`True` → stops decay, **returns `True`** (forces refresh — the motivating case is a Zigbee2MQTT/HA-startup `unknown → active` transition)
- `True` while already decaying → auto-corrected (decay stopped; defensive fix for an inconsistent state)
- `False → True` stops decay; `True → False` starts decay

**Gotcha, easy to misread as backwards:** `InputType.DOOR`'s default `active_states` is `[STATE_CLOSED]`, not open. A *closed* door is the evidence-supporting ("active") state for occupancy purposes — this is Wasp-in-Box semantics (someone closed the door behind them). Verify: `data/entity_type.py`, `InputType.DOOR` entry, `active_states`.

`effective_weight = weight × information_gain` is used everywhere in place of raw config weight (`data/entity.py::information_gain`, `utils.py`). `information_gain` measures `|prob_given_true - prob_given_false| / max(pgt, pgf, 0.01)`, clamped to `[0,1]`; entities whose correlation analysis produced a real failure are force-clamped to `information_gain = 0.1` regardless of their configured likelihoods. This is the automatic mechanism by which an uninformative sensor gets down-weighted without the user's configured `weight` ever changing — useful to know when a user asks "why isn't my sensor influencing the probability like I configured it to."

## Database layout and the executor rule

Modules (`db/`), each with one job:

| Module | Responsibility |
|---|---|
| `core.py` | Session management (`get_session()` context manager), path setup, delegated-method wiring |
| `schema.py` | SQLAlchemy declarative table definitions |
| `operations.py` | CRUD for entities/intervals, `prune_old_intervals` |
| `aggregation.py` | Time-series rollups (hourly/daily/weekly/monthly) |
| `correlation.py` | Sensor↔occupancy statistical correlation |
| `queries.py` | Complex queries, occupied-intervals cache validity |
| `sync.py` | Global-watermark recorder import (`sync_states`) |
| `maintenance.py` | Integrity check, corruption recovery, backups, pruning |

**Verified table count on `main`: 15** (since PR #454 merged 2026-07-06) — `areas, entities, priors, intervals, metadata, interval_aggregates, occupied_intervals_cache, global_priors, numeric_samples, numeric_aggregates, correlations, entity_statistics, area_relationships, area_transitions, cross_area_stats` (`db/schema.py`, grep `__tablename__`). `area_relationships`/`cross_area_stats` were relationship-storage scaffolding (`db/relationships.py`) added ahead of the feature; `area_transitions` is PR #454's own table. The producer/consumer wiring (transition learning step 9, phase-3 boost) is now live on `main` — see the pipeline and invariants sections above/below.

**Executor rule (non-negotiable, matches CLAUDE.md):** `AreaOccupancyDB.get_session()` (`db/core.py::get_session`) is a synchronous `@contextmanager` — every `with self.db.get_session() as session:` block opens, uses, and closes the session entirely inside a function run via `hass.async_add_executor_job(...)`. **Sessions never cross an `await` boundary.** All DB entry points delegate through the executor from `coordinator.py`. If you write new DB-calling code, wrap it in `async_add_executor_job` the same way — do not call session-opening DB methods directly from async code, even "just to read one row."

Schema-version mismatch handling is destructive by design: `_ensure_schema_up_to_date` (`db/maintenance.py`) deletes and recreates the **entire DB** on any mismatch — there is no migration-script path for the DB schema (unlike `migrations.py`'s config-entry `CONF_VERSION` ladder, which is additive/idempotent). This is why the adjacent-areas feature deliberately did **not** bump `CONF_VERSION` for its purely-additive schema change — bumping it would trigger the destructive reset path and wipe every user's learned priors/history. Any DB schema change must ask: does this need `Base.metadata.create_all(checkfirst=True)` (additive, safe) or does it require a real version bump (destructive, wipes history)?

## Invariants (verified on `main`)

| Invariant | Where enforced |
|---|---|
| Probability and prior always clamped to `[0.01, 0.99]` | `utils.py::clamp_probability`; `const.py: MIN_PROBABILITY, MAX_PROBABILITY, MIN_PRIOR, MAX_PRIOR` |
| Time-priors bucketed tighter: `[0.03, 0.9]` | `const.py: TIME_PRIOR_MIN_BOUND, TIME_PRIOR_MAX_BOUND` |
| UTC stored in DB; local wall-clock used for the 168 (day-of-week × hour) time-prior buckets, DST-safe by walking hour-by-hour in UTC and deriving bucket keys from local time | `data/analysis.py::calculate_time_priors` — this exact bug class (timezone/DST) is one of the two costliest historical failure modes; see `aod-failure-archaeology` |
| Correlation analysis requires ≥50 samples before it's trusted | `const.py: MIN_CORRELATION_SAMPLES = 50`; used in 3 places in `db/correlation.py` — sample-count gate, confidence discount, staleness re-check on reload |
| Occupied-intervals cache is validated before being queried | `db/queries.py::is_occupied_intervals_cache_valid`; rebuilt hourly, health-checked stale at 25h |
| A configured/purpose prior floor can never by itself push probability above the occupancy threshold | `data/prior.py`; floor capped at `max(MIN_PRIOR, threshold - 0.01)` — fix for issue #435, don't regress it |
| Purpose half-life matching only compares against the **selected** purpose's own default, never "any purpose whose default happens to match" | `data/purpose.py::Purpose.is_purpose_half_life` — fix for issue #439/#440; the same bug class recurred for Bedroom/SLEEPING (#481, fixed in PR #493, merged 2026-07-06: `data/decay.py::_resolve_purpose_half_life()` now guards `_base_half_life != purpose.half_life → return base` before applying the adjacency modifier) |
| Decay-modifier is clamped to `≥1.0` — adjacency silence can only *slow* decay, never speed it up (merged PR #454, 2026-07-06) | `data/decay.py::set_modifier_factor` — `max(1.0, float(factor))`, current on `main` |
| Adjacency math reads the *previous* tick's lagged probability snapshot, never the in-flight recompute, to avoid a same-tick feedback loop between neighbouring areas (merged PR #454, 2026-07-06) | `coordinator.py::_lagged_probabilities` — current on `main`; verify with `grep -n lagged custom_components/area_occupancy/coordinator.py` |

## Purpose system as the config-surface strategy

`AreaPurpose` (`data/purpose.py`) is the project's deliberate answer to "how do we expose per-room-type tuning without a sprawling per-field config surface": instead of exposing raw half-life/min-prior knobs per area by default, each purpose (`PASSAGEWAY`, `DRIVEWAY`, `UTILITY`, `GARAGE`, `FOOD_PREP`, `GARDEN`, `BATHROOM`, `EATING`, `SOCIAL`, `WORKING`, `RELAXING`, `SLEEPING`) carries a curated default `half_life` (45s–1200s), an optional `min_prior` floor (only transit-type purposes: `PASSAGEWAY=0.1`, `DRIVEWAY=0.05`), and — uniquely for `SLEEPING` — an `awake_half_life` (620s) used outside the configured sleep window so a bedroom clears faster once everyone's up. Users can still override with a custom half-life per area; the purpose default is a sentinel (`0`) resolved at entity-creation time, not a hard floor.

This is directly relevant to "config surface is sacred" (an unwritten law per project convention): adding a new purpose is safe/additive (new enum value + `PURPOSE_DEFINITIONS` entry), but changing what an *existing* purpose's default half-life or min_prior means is a silent behavior change for every user who left that field on default — treat it with the same care as a math change, not a config tweak.

## Known weak points (stated plainly, don't oversell)

- **Adjacency (merged PR #454) remains functionally unvalidated on real homes.** Its tunables (`ADJACENCY_BOOST_GAIN`, `ADJACENCY_DECAY_MODIFIER_GAIN`, `ADJACENCY_DECAY_MODIFIER_MAX`, the `ADJACENCY_N_*` sample-count thresholds, all in `const.py`) are explicit first-pass guesses — the code comment above them still reads "First-pass values; tune from real data once Phase 3 is collecting transitions." Merged status is not the same as validated status: no commit or test exercises them against real HA recorder data (only synthetic/mocked entities in the 4 adjacency test files). Treat any adjacency boost/decay-modifier number as a hypothesis, not a tuned constant, until real-world data says otherwise.
- **Simulator (`simulator/`) has zero project-level tests.** It's a Flask app that imports the real `EntityType`/`Entity` classes and recomputes probability in-process — useful for manual verification, but has no automated test suite of its own (only vendored library tests exist under `simulator/.venv/`).
- **Simulator deploy is fully manual, no CI.** `simulator/README.md` documents a step-by-step IBM Cloud Container Registry docker build/tag/push sequence run by hand (`ibmcloud cr login`, `docker build`, `docker push`); there is no `.github/workflows/*simulator*` automation and nothing ties the deployed image's version to the integration's version.
- **SETTLED (2026-07-06, PR #496): the Python 3.13-local vs 3.14-CI interpreter skew is resolved.** Historically, CI ran Python 3.14 while the documented/devcontainer dev environment was 3.13, with no `.python-version` file to pin either and `pyproject.toml` only setting a floor — a bug reproducing on one minor version could pass locally and fail in CI or vice versa. PR #496 bumped the toolchain in lockstep: `requires-python = ">=3.14.2"`, `.python-version = 3.14`, devcontainer image `python:3.14`, so CI and local now run the same interpreter. Keep this as a dated cautionary story for the "silent version skew" failure class — the trap itself no longer exists.
- **SETTLED (2026-07-06, PR #496): ruff's three-way version pin skew is resolved.** Historically `pyproject.toml` floor `>=0.13.0`, `.pre-commit-config.yaml` pin `v0.14.2`, and `uv.lock`'s resolved `0.15.2` could silently disagree, letting pre-commit and CI/local diverge on lint behavior after a ruff release. Both `pyproject.toml` (`ruff==0.15.2` dev dependency) and `.pre-commit-config.yaml` (`rev: v0.15.2`) are now pinned to the same exact version. Keep this as a dated cautionary story; re-check both files together on any future ruff bump.
- **Coverage gate comment is fixed.** `pyproject.toml`'s `[tool.coverage.report]` line now reads `fail_under = 85 # Enforced global minimum; aim for 90%+ on core calculation modules (CLAUDE.md)` — the enforced number (85) and the comment now agree, and it explicitly frames 90% as an aspiration for core modules rather than a second gate. This matches CLAUDE.md's "90% for core calculations" language, which is still not separately enforced by any tool config in the repo.
- **Single maintainer, thin merge gating.** Classic branch protection is absent (`gh api .../branches/main/protection` → 404), but an active repository ruleset ("Main") requires PRs and blocks deletion/force-push on `main` — with an always-on admin bypass, and no required status checks. So CI remains advisory and the maintainer can push directly; combined with the single-maintainer bus factor, be conservative with anything that touches `main` directly. Full details in `aod-change-control`.

## Provenance and maintenance

Verified 2026-07-06 against integration version 2026.5.17 (`pyproject.toml`, `manifest.json`, `const.py::DEVICE_SW_VERSION`) — note none of the 2026-07-06 merge wave (including PR #454) is in a tagged release yet; the version number itself hasn't moved. Working tree at verification time was on `main`, HEAD `17b71d2`, which already includes PR #454 (adjacent-areas, merged 2026-07-06) — all claims above were verified directly against this checked-out tree.

Re-verification commands, by volatile fact:

| Fact | Re-check with |
|---|---|
| Current branch you're actually looking at | `git branch --show-current` |
| PR #454 (adjacency) merge status | `gh pr view 454 --json state,mergedAt` |
| PR #486 (raw-float threshold decision) merge status | `gh pr view 486 --json state,mergedAt` |
| Single state listener | `grep -n "_area_state_listeners\[.\"_all\"" custom_components/area_occupancy/coordinator.py` |
| Probability pipeline order | `sed -n '1,300p' custom_components/area_occupancy/area/area.py` on `main` |
| `occupied()` uses raw float | `grep -n "def occupied" -A3 custom_components/area_occupancy/area/area.py` |
| 13-step analysis pipeline (includes `transition_learning` from PR #454) | `grep -n "total_steps\|_run_step(" custom_components/area_occupancy/data/analysis.py` |
| Timer intervals | `grep -n "DECAY_INTERVAL\|ANALYSIS_INTERVAL\|SAVE_INTERVAL" custom_components/area_occupancy/const.py` |
| Analysis retry backoff | `grep -n "minutes=15" custom_components/area_occupancy/coordinator.py` |
| DOOR active_states gotcha | `grep -n "InputType.DOOR" -A6 custom_components/area_occupancy/data/entity_type.py` |
| DB table count (15, includes `area_transitions` from PR #454) | `grep -c "__tablename__" custom_components/area_occupancy/db/schema.py` on `main` |
| Decay-modifier clamp / lagged-probability read (both current on `main`) | `grep -n modifier_factor custom_components/area_occupancy/data/decay.py; grep -n lagged custom_components/area_occupancy/coordinator.py` |
| Correlation min-sample threshold | `grep -n MIN_CORRELATION_SAMPLES custom_components/area_occupancy/const.py` |
| Probability clamp bounds | `grep -n "MIN_PROBABILITY\|MAX_PROBABILITY" custom_components/area_occupancy/const.py` |
| Purpose half-life table | `sed -n '1,240p' custom_components/area_occupancy/data/purpose.py` |
| Coverage gate vs comment (now in agreement) | `grep -n fail_under pyproject.toml` |
| Ruff version pin (now matched at `0.15.2` in both places, since PR #496) | `grep "ruff==" pyproject.toml; grep rev: .pre-commit-config.yaml` |
| CI vs local Python version (now matched at 3.14 everywhere, since PR #496) | `cat .python-version; grep requires-python pyproject.toml` |
| Branch protection status | `gh api repos/Hankanman/Area-Occupancy-Detection/branches/main/protection` |
| Simulator test coverage | `find simulator -maxdepth 1 -iname '*test*'` (excluding `.venv`) |
