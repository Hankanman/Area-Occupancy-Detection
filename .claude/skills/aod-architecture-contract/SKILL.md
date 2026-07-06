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

## IMPORTANT: this skill describes `main`, not the checked-out branch

At the time this skill was written, the working tree may be checked out to `feat/adjacent-areas` (open PR #454) rather than `main`. **Do not trust `git show HEAD:...` for architecture facts — check `git branch --show-current` first, and read `main` explicitly (`git show main:<path>`) when verifying anything in this file.** PR #454 adds an entire fourth pipeline phase (adjacency boost + decay-modifier) that does not exist on `main` yet. It is called out separately below, clearly labeled, per instructions — do not describe it as merged. Re-check before relying on it: `gh pr view 454 --json state,mergedAt`.

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

### Pending (PR #454, unmerged as of 2026-07-06): a third pipeline phase

`feat/adjacent-areas` adds a **phase 3** after activity boost: `boost = coordinator.adjacency_boost_for(area)`; if present, `result = sigmoid(logit(result) + boost.logit_contribution)`. This does not exist on `main` — verify before relying on it: `gh pr view 454 --json state,mergedAt`. Its invariants are covered in their own subsection below, clearly marked pending.

## The three timers

| Timer | Interval | Const | Why this cadence |
|---|---|---|---|
| Decay | 10s | `DECAY_INTERVAL` (`const.py:305`, main) | Fast enough that probability decay feels continuous to automations without re-running the full analysis pipeline; only refreshes the coordinator if at least one area has `decay.enabled`. |
| Analysis | 3600s (1h) | `ANALYSIS_INTERVAL` (`const.py:306`) | Expensive (DB sync, aggregation, prior/correlation recompute) — hourly balances freshness against DB/CPU load. First run is deliberately deferred via `homeassistant.helpers.start.async_at_started` plus an additional 5-minute delay, specifically so it never competes with HA's own startup. |
| Save | 600s (10min) | `SAVE_INTERVAL` (`const.py:307`) | Periodic persistence so a crash between saves loses at most 10 minutes of learned state, without writing to SQLite continuously. |

All three timers check `self._stop_requested` **both before doing work and before rearming** (`coordinator.py::_handle_decay_timer` / `_handle_save_timer` / analysis equivalent), and `EVENT_HOMEASSISTANT_STOP` synchronously cancels all three registered timer handles. This exists to close a race where a timer that had already fired (but not yet run its callback body) could otherwise still kick off executor work after shutdown began — read the inline comments in `_handle_decay_timer`/`_handle_save_timer` before touching this logic; they document why the check appears twice, not once.

Analysis timer retry-on-failure backoff is **15 minutes**, not the normal hourly cadence (`coordinator.py`, analysis timer handler: `if _failed: next_update = _now + timedelta(minutes=15)`), so a transient failure (recorder purge collision, momentary DB lock) doesn't wait a full hour to retry.

### The hourly analysis pipeline — exact order (verified on `main`: **12 steps**, not 13)

`data/analysis.py::run_full_analysis()`, `total_steps = 12`:

1. `sync_states` — import recent entity state changes from the HA recorder
2. `health_check_and_prune` — DB integrity check (`PRAGMA integrity_check`) + prune intervals older than `RETENTION_DAYS`
3. `sensor_health_check` — per-entity anomaly detection → HA repairs
4. `populate_occupied_intervals_cache` — motion-sensor-only ground truth, only if cache invalid/stale
5. `interval_aggregation` — raw → daily/weekly/monthly rollups
6. `numeric_aggregation` — raw numeric samples → hourly/weekly (feeds Gaussian correlation)
7. `recalculate_priors` — per-area global prior + 168 time-priors
8. `correlation_analysis` — sensor/occupancy statistical correlation (needs ≥50 samples, see invariants)
9. `pipeline_health_check` — area-scope anomalies (stale cache, slow analysis, insufficient priors, correlation failure ratio)
10. `save_data_before_refresh`
11. `refresh_coordinator` — recompute `probability()` for every area
12. `save_data_after_refresh`

Each step is wrapped by an internal `_run_step()` helper that times it independently and catches all exceptions into a `failed_steps` list — **a failing step does not abort the run; all 12 steps always attempt to execute.** Only after all steps run does a non-empty `failed_steps` raise `HomeAssistantError` (triggering the 15-minute retry backoff above). If `EVENT_HOMEASSISTANT_STOP` fires mid-run, remaining steps are skipped, the run is marked cancelled, and `_last_analysis_duration_ms` is deliberately **not** written — so a fast, aborted partial run can never mask a genuinely slow prior run in the `SLOW_ANALYSIS` health check.

**Pending (PR #454):** adds a 13th step, `transition_learning`, inserted between `correlation_analysis` and `pipeline_health_check` — do not assume it exists until #454 merges.

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

**Verified table count on `main`: 14**, not 15 — `areas, entities, priors, intervals, metadata, interval_aggregates, occupied_intervals_cache, global_priors, numeric_samples, numeric_aggregates, correlations, entity_statistics, area_relationships, cross_area_stats` (`db/schema.py`, grep `__tablename__`). `area_relationships`/`cross_area_stats` already exist on `main` as relationship-storage scaffolding (`db/relationships.py`), but nothing on `main` currently produces or consumes adjacency data through them — **pending PR #454 adds a 15th table, `area_transitions`**, plus the actual Bayesian producer/consumer wiring. Don't assume `area_relationships` being populated means the adjacency feature is live; check whether anything reads it.

**Executor rule (non-negotiable, matches CLAUDE.md):** `AreaOccupancyDB.get_session()` (`db/core.py::get_session`) is a synchronous `@contextmanager` — every `with self.db.get_session() as session:` block opens, uses, and closes the session entirely inside a function run via `hass.async_add_executor_job(...)`. **Sessions never cross an `await` boundary.** All DB entry points delegate through the executor from `coordinator.py`. If you write new DB-calling code, wrap it in `async_add_executor_job` the same way — do not call session-opening DB methods directly from async code, even "just to read one row."

Schema-version mismatch handling is destructive by design: `_ensure_schema_up_to_date` (`db/maintenance.py`) deletes and recreates the **entire DB** on any mismatch — there is no migration-script path for the DB schema (unlike `migrations.py`'s config-entry `CONF_VERSION` ladder, which is additive/idempotent). This is why the adjacent-areas feature deliberately did **not** bump `CONF_VERSION` for its purely-additive schema change — bumping it would trigger the destructive reset path and wipe every user's learned priors/history. Any DB schema change must ask: does this need `Base.metadata.create_all(checkfirst=True)` (additive, safe) or does it require a real version bump (destructive, wipes history)?

## Invariants (verified on `main` unless marked pending)

| Invariant | Where enforced |
|---|---|
| Probability and prior always clamped to `[0.01, 0.99]` | `utils.py::clamp_probability`; `const.py: MIN_PROBABILITY, MAX_PROBABILITY, MIN_PRIOR, MAX_PRIOR` |
| Time-priors bucketed tighter: `[0.03, 0.9]` | `const.py: TIME_PRIOR_MIN_BOUND, TIME_PRIOR_MAX_BOUND` |
| UTC stored in DB; local wall-clock used for the 168 (day-of-week × hour) time-prior buckets, DST-safe by walking hour-by-hour in UTC and deriving bucket keys from local time | `data/analysis.py::calculate_time_priors` — this exact bug class (timezone/DST) is one of the two costliest historical failure modes; see `aod-failure-archaeology` |
| Correlation analysis requires ≥50 samples before it's trusted | `const.py: MIN_CORRELATION_SAMPLES = 50`; used in 3 places in `db/correlation.py` — sample-count gate, confidence discount, staleness re-check on reload |
| Occupied-intervals cache is validated before being queried | `db/queries.py::is_occupied_intervals_cache_valid`; rebuilt hourly, health-checked stale at 25h |
| A configured/purpose prior floor can never by itself push probability above the occupancy threshold | `data/prior.py`; floor capped at `max(MIN_PRIOR, threshold - 0.01)` — fix for issue #435, don't regress it |
| Purpose half-life matching only compares against the **selected** purpose's own default, never "any purpose whose default happens to match" | `data/purpose.py::Purpose.is_purpose_half_life` — fix for issue #439/#440; the same bug class recurred for Bedroom/SLEEPING (#481, fix in PR #493, open as of 2026-07-06) |
| **Pending (PR #454):** decay-modifier is clamped to `≥1.0` — adjacency silence can only *slow* decay, never speed it up | `data/decay.py::set_modifier_factor` on `feat/adjacent-areas` only — **absent on `main`**, verify with `git show main:custom_components/area_occupancy/data/decay.py \| grep modifier_factor` before citing as current |
| **Pending (PR #454):** adjacency math reads the *previous* tick's lagged probability snapshot, never the in-flight recompute, to avoid a same-tick feedback loop between neighbouring areas | `coordinator.py::update()` lagged-snapshot mechanism on `feat/adjacent-areas` only — **absent on `main`**, verify with `git show main:custom_components/area_occupancy/coordinator.py \| grep -c lagged` (expect 0) |

## Purpose system as the config-surface strategy

`AreaPurpose` (`data/purpose.py`) is the project's deliberate answer to "how do we expose per-room-type tuning without a sprawling per-field config surface": instead of exposing raw half-life/min-prior knobs per area by default, each purpose (`PASSAGEWAY`, `DRIVEWAY`, `UTILITY`, `GARAGE`, `FOOD_PREP`, `GARDEN`, `BATHROOM`, `EATING`, `SOCIAL`, `WORKING`, `RELAXING`, `SLEEPING`) carries a curated default `half_life` (45s–1200s), an optional `min_prior` floor (only transit-type purposes: `PASSAGEWAY=0.1`, `DRIVEWAY=0.05`), and — uniquely for `SLEEPING` — an `awake_half_life` (620s) used outside the configured sleep window so a bedroom clears faster once everyone's up. Users can still override with a custom half-life per area; the purpose default is a sentinel (`0`) resolved at entity-creation time, not a hard floor.

This is directly relevant to "config surface is sacred" (an unwritten law per project convention): adding a new purpose is safe/additive (new enum value + `PURPOSE_DEFINITIONS` entry), but changing what an *existing* purpose's default half-life or min_prior means is a silent behavior change for every user who left that field on default — treat it with the same care as a math change, not a config tweak.

## Known weak points (stated plainly, don't oversell)

- **Adjacency tunables (PR #454) are explicit first-pass guesses**, not validated against real data — `const.py`'s own comment on that branch says "tune from real data once Phase 3 is collecting transitions." No commit or test exercises them against real HA recorder data (only synthetic/mocked entities in tests). Treat any adjacency boost/decay-modifier number as a hypothesis, not a tuned constant, once #454 merges.
- **Simulator (`simulator/`) has zero project-level tests.** It's a Flask app that imports the real `EntityType`/`Entity` classes and recomputes probability in-process — useful for manual verification, but has no automated test suite of its own (only vendored library tests exist under `simulator/.venv/`).
- **Simulator deploy is fully manual, no CI.** `simulator/README.md` documents a step-by-step IBM Cloud Container Registry docker build/tag/push sequence run by hand (`ibmcloud cr login`, `docker build`, `docker push`); there is no `.github/workflows/*simulator*` automation and nothing ties the deployed image's version to the integration's version.
- **CI tests on Python 3.14; the documented/devcontainer dev environment is 3.13.** No `.python-version` file exists anywhere in the repo to pin this. `pyproject.toml` only sets a floor (`>=3.13.2`), so `uv sync` in CI resolves the newest available interpreter. A bug that only reproduces on one minor version can pass locally and fail in CI or vice versa. See `aod-build-and-env` for the detection commands.
- **Ruff version is pinned inconsistently in three places**: `pyproject.toml` floor `>=0.13.0`, `.pre-commit-config.yaml` pins `v0.14.2`, `uv.lock` currently resolves `0.15.2`. Pre-commit and CI/local can silently disagree on lint behavior after a ruff release. See `aod-build-and-env`.
- **Coverage gate comment is stale.** `pyproject.toml`'s `[tool.coverage.report]` line reads `fail_under = 85 # Enforce 90% coverage minimum` — the enforced number is 85, the comment says 90. CLAUDE.md's "90% for core calculations" is not separately enforced by any tool config found in the repo; treat it as an aspiration, not a gate.
- **Single maintainer, thin merge gating.** Classic branch protection is absent (`gh api .../branches/main/protection` → 404), but an active repository ruleset ("Main") requires PRs and blocks deletion/force-push on `main` — with an always-on admin bypass, and no required status checks. So CI remains advisory and the maintainer can push directly; combined with the single-maintainer bus factor, be conservative with anything that touches `main` directly. Full details in `aod-change-control`.

## Provenance and maintenance

Verified 2026-07-06 against integration version 2026.5.17 (`pyproject.toml`, `manifest.json`, `const.py::DEVICE_SW_VERSION`). Working tree at verification time was checked out to `feat/adjacent-areas`; all `main`-branch claims above were independently re-verified via `git show main:<path>` rather than trusted from the checked-out tree — see the individual file/line citations inline.

Re-verification commands, by volatile fact:

| Fact | Re-check with |
|---|---|
| Current branch you're actually looking at | `git branch --show-current` |
| PR #454 (adjacency) merge status | `gh pr view 454 --json state,mergedAt` |
| PR #486 (raw-float threshold decision) merge status | `gh pr view 486 --json state,mergedAt` |
| Single state listener | `grep -n "_area_state_listeners\[.\"_all\"" custom_components/area_occupancy/coordinator.py` |
| Probability pipeline order | `sed -n '1,300p' custom_components/area_occupancy/area/area.py` on `main` |
| `occupied()` uses raw float | `grep -n "def occupied" -A3 custom_components/area_occupancy/area/area.py` |
| 12-step (main) vs 13-step (PR #454) analysis pipeline | `grep -n "total_steps\|_run_step(" custom_components/area_occupancy/data/analysis.py` |
| Timer intervals | `grep -n "DECAY_INTERVAL\|ANALYSIS_INTERVAL\|SAVE_INTERVAL" custom_components/area_occupancy/const.py` |
| Analysis retry backoff | `grep -n "minutes=15" custom_components/area_occupancy/coordinator.py` |
| DOOR active_states gotcha | `grep -n "InputType.DOOR" -A6 custom_components/area_occupancy/data/entity_type.py` |
| DB table count | `grep -c "__tablename__" custom_components/area_occupancy/db/schema.py` on `main` |
| Decay-modifier / lagged-read existence on `main` (should be absent) | `git show main:custom_components/area_occupancy/data/decay.py \| grep -c modifier_factor` |
| Correlation min-sample threshold | `grep -n MIN_CORRELATION_SAMPLES custom_components/area_occupancy/const.py` |
| Probability clamp bounds | `grep -n "MIN_PROBABILITY\|MAX_PROBABILITY" custom_components/area_occupancy/const.py` |
| Purpose half-life table | `sed -n '1,240p' custom_components/area_occupancy/data/purpose.py` |
| Coverage gate vs comment | `grep -n fail_under pyproject.toml` |
| Ruff version skew | `grep required-version pyproject.toml; grep rev: .pre-commit-config.yaml; uv run ruff --version` |
| CI vs local Python version | see `aod-build-and-env` |
| Branch protection status | `gh api repos/Hankanman/Area-Occupancy-Detection/branches/main/protection` |
| Simulator test coverage | `find simulator -maxdepth 1 -iname '*test*'` (excluding `.venv`) |
