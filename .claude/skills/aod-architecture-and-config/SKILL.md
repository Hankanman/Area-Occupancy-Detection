---
name: aod-architecture-and-config
description: Use when you need to understand WHY Area Occupancy Detection is built the way it is before changing coordinator.py, area/area.py, utils.py, data/analysis.py, data/decay.py, data/entity.py, or anything under db/ — the single-coordinator-many-areas design, the exact probability pipeline order, the three timers, DB session/executor rules, or the list of invariants a change must not violate. Load this before any PR that touches Bayesian math, timers, DB schema, or entity evidence, and whenever asked "why does this work this way" or "is it safe to change X". Also use when adding, renaming, removing, or debugging any CONF_* / DEFAULT_* configuration key, config_flow.py schema, strings.json/translations entry, migrations.py version bump, or sensor default in Area Occupancy Detection; when a user reports a setting "not sticking", reverting, or silently overwritten (e.g. decay half-life, threshold, sensor precision); or when deciding whether a change needs a CONF_VERSION bump. Covers the full CONF_* catalog with defaults/ranges/stability, the decay half-life 0-sentinel, the #440 normalisation rule, and the add-a-new-option checklist.
---

# AOD Architecture and Config

## What this covers

The load-bearing design decisions in Area Occupancy Detection (AOD) and why they exist: the single-coordinator/many-areas shape, the exact probability pipeline, the three background timers, entity/evidence semantics, the database layout and its concurrency rules, and the hard invariants a change must never break. It also states known weak points plainly so you don't mistake "it's unmerged/untested" for "it's proven."

The complete configuration surface of Area Occupancy Detection: every `CONF_*`/`DEFAULT_*` key in `const.py`, the `IntegrationConfig` vs `AreaConfig` split, config-flow/options-flow schema wiring, the decay-half-life sentinel and its #440 anti-clobber rule, `migrations.py`'s version ladder, and the checklist (with re-verification commands) for adding a new option safely. This is the "what does this setting do, what's its default/range, and how do I add one without breaking existing users" skill.

## When NOT to use this

- The math itself (formulas, worked examples, why sigmoid/logit) → `aod-math-reference`
- Reproducing/fixing a specific bug, or prior/likelihood accuracy work → `aod-debugging-and-history`
- Whether a change is safe to ship / needs a version bump philosophy and review process → `aod-change-and-validation`
- Debugging a *running* instance's config (inspecting a live DB/entry) → `aod-diagnostics-and-tooling` or `aod-debugging-and-history`
- Adjacent-areas' Bayesian boost/decay-modifier math itself (not the config keys) → `aod-research-frontier` (feature merged 2026-07-06, PR #454; still unvalidated on real homes)

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
   - Otherwise `combined_probability(presence, env)` applies env as an **additive** logit-space update damped to 20%: `sigmoid(logit(presence) + 0.2×logit(env))`. It is *not* an average — averaging (the pre-2026-07 `0.8×logit(presence) + 0.2×logit(env)`) let supporting environmental data lower a motion-confirmed probability. If `env == 0.5` exactly, `_base_probability()` short-circuits and returns `presence`, which is what the formula yields anyway (`logit(0.5) == 0`).
2. **`probability()`** — activity boost:
   - `base = _base_probability()`; `is_occupied = base >= config.threshold`.
   - `activity = detect_activity(self, base_probability=base, is_occupied=is_occupied)`.
   - If `activity.activity_id` is `UNOCCUPIED` or `IDLE`, return `base` unchanged.
   - Otherwise `apply_activity_boost(base, activity.occupancy_boost, activity.confidence)` — a logit-space additive boost, `sigmoid(logit(base) + boost*confidence)`.
3. **`occupied()`** — `return self.probability() >= self.config.threshold` (`area/area.py::occupied`). Threshold default is 50.0/100 (0.50).

**Load-bearing decision (PR #486, merged 2026-07-06):** occupancy threshold comparison and every downstream decision (wasp-in-box, activity scoring) operate on the **internal unrounded float** probability. The user-facing "Sensor state precision" setting (0–2 decimals, default 2) only affects *published sensor state formatting* (`format_float` in `sensor.py`) — never the value fed into `occupied()`. PR #486's own description states this as the reason the change was "functionally safe": *"All decision logic — `area.occupied()`, wasp-in-box, activity scoring, thresholds — operates on internal unrounded floats. The sensor states are publication-only formatting."* If you ever see code comparing a *rounded/published* value to the threshold, that is a regression of this decision — flag it.

Every probability value is clamped through `clamp_probability()` (`utils.py::clamp_probability`) to `[MIN_PROBABILITY=0.01, MAX_PROBABILITY=0.99]` before use; `logit()` clamps its input first so `logit(0)`/`logit(1)` never raise. NaN clamps to MAX (with a warning log), ±inf clamps to MAX/MIN respectively.

### Resolved dead-code trap: `bayesian_probability()` is gone (PR #529, 2026.8.1)

`utils.py` used to also define a classic naive-Bayes log-odds function, `bayesian_probability()`, left in place as a revert safety net after the sigmoid/logit pipeline above superseded it (PR #353, "Add sigmoid-based occupancy detection framework"). It had zero production call sites for that entire window and was deleted in PR #529 (2026.8.1) along with its three private helpers, once the deprecation window the maintainer asked for had elapsed. AGENTS.md's "Modifying Probability Calculation" workflow now points at the real entry points: `presence_probability()`, `environmental_confidence()`, `combined_probability()`, `apply_activity_boost()` in `utils.py`, plus `Area._base_probability()`/`Area.probability()` in `area/area.py`. If you're reading old context (a stale docstring, an old PR, a previous session's notes) that references `bayesian_probability()`, treat it as historical — the function no longer exists.

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

**Executor rule (non-negotiable, matches AGENTS.md):** `AreaOccupancyDB.get_session()` (`db/core.py::get_session`) is a synchronous `@contextmanager` — every `with self.db.get_session() as session:` block opens, uses, and closes the session entirely inside a function run via `hass.async_add_executor_job(...)`. **Sessions never cross an `await` boundary.** All DB entry points delegate through the executor from `coordinator.py`. If you write new DB-calling code, wrap it in `async_add_executor_job` the same way — do not call session-opening DB methods directly from async code, even "just to read one row."

Schema-version mismatch handling is destructive by design: `_ensure_schema_up_to_date` (`db/maintenance.py`) deletes and recreates the **entire DB** on any mismatch — there is no migration-script path for the DB schema (unlike `migrations.py`'s config-entry `CONF_VERSION` ladder, which is additive/idempotent). This is why the adjacent-areas feature deliberately did **not** bump `CONF_VERSION` for its purely-additive schema change — bumping it would trigger the destructive reset path and wipe every user's learned priors/history. Any DB schema change must ask: does this need `Base.metadata.create_all(checkfirst=True)` (additive, safe) or does it require a real version bump (destructive, wipes history)?

## Invariants (verified on `main`)

| Invariant | Where enforced |
|---|---|
| Probability and prior always clamped to `[0.01, 0.99]` | `utils.py::clamp_probability`; `const.py: MIN_PROBABILITY, MAX_PROBABILITY, MIN_PRIOR, MAX_PRIOR` |
| Time-priors bucketed tighter: `[0.03, 0.9]` | `const.py: TIME_PRIOR_MIN_BOUND, TIME_PRIOR_MAX_BOUND` |
| UTC stored in DB; local wall-clock used for the 168 (day-of-week × hour) time-prior buckets, DST-safe by walking hour-by-hour in UTC and deriving bucket keys from local time | `data/analysis.py::calculate_time_priors` — this exact bug class (timezone/DST) is one of the two costliest historical failure modes; see `aod-debugging-and-history` |
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
- **Coverage gate comment is fixed.** `pyproject.toml`'s `[tool.coverage.report]` line now reads `fail_under = 85 # Enforced global minimum; aim for 90%+ on core calculation modules (AGENTS.md)` — the enforced number (85) and the comment now agree, and it explicitly frames 90% as an aspiration for core modules rather than a second gate. This matches AGENTS.md's "90% for core calculations" language, which is still not separately enforced by any tool config in the repo.
- **Single maintainer, thin merge gating.** Classic branch protection is absent (`gh api .../branches/main/protection` → 404), but an active repository ruleset ("Main") requires PRs and blocks deletion/force-push on `main` — with an always-on admin bypass, and no required status checks. So CI remains advisory and the maintainer can push directly; combined with the single-maintainer bus factor, be conservative with anything that touches `main` directly. Full details in `aod-change-and-validation`.

## Provenance and maintenance (architecture)

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

## The IntegrationConfig vs AreaConfig split

Two config classes, both in `custom_components/area_occupancy/data/config.py`, backed by the **same** `ConfigEntry` object but reading different scopes:

- **`IntegrationConfig`** (`data/config.py:132-273`) — entry-wide, global settings. Properties read `config_entry.options` live (no caching) on every access: `sleep_start`/`sleep_end`, `health_enabled`, `sensor_precision` (clamped 0-2), `people` (parses `CONF_PEOPLE` into `PersonConfig` dataclasses with legacy single-sensor fallback). `analysis_interval`/`decay_interval` are set once from `const.py` constants and are **not** user-configurable (comment in code: "could be made configurable in the future").
- **`AreaConfig`** (`data/config.py:421-823`) — one instance per area, constructed by matching an entry in the `CONF_AREAS` list via the area's HA-registry `area_id` (not by name string). Holds `sensors`, `sensor_states`, `weights`, `decay`, `wasp_in_box`, `min_prior_override`, `exclude_from_all_areas`, `threshold`, `purpose`, `adjacent_areas`. `AreaConfig.update_config()` persists by rewriting that one area's dict inside `CONF_AREAS` (writing to `config_entry.options` if `CONF_AREAS` already lives there, else `config_entry.data`), then requests a coordinator refresh only if `coordinator.setup_complete`.

Rule of thumb: if the setting applies once to the whole integration (sleep window, health toggle, recorder-write precision, people list), it belongs on `IntegrationConfig`. If it varies per room, it belongs on `AreaConfig`.

## The CONF_* catalog

Stability labels: **production** = shipped, stable, exercised by real installs. **recent** = merged within the last few weeks, expect edge cases. **experimental** = hardcoded/unvalidated on real data, or living on an unmerged branch — do not present as shipped behavior.

### Sensors (per-area entity lists — all default `[]`, all production)

| CONF_* key | const.py value | Notes |
|---|---|---|
| `CONF_MOTION_SENSORS` | `motion_sensors` | Ground-truth signal; `strength_multiplier=3.0` in `data/entity_type.py`, not user-configurable |
| `CONF_MEDIA_DEVICES` | `media_devices` | |
| `CONF_APPLIANCES` | `appliances` | |
| `CONF_ILLUMINANCE_SENSORS`, `CONF_HUMIDITY_SENSORS`, `CONF_TEMPERATURE_SENSORS`, `CONF_CO2_SENSORS`, `CONF_CO_SENSORS`, `CONF_SOUND_PRESSURE_SENSORS`, `CONF_PRESSURE_SENSORS`, `CONF_AIR_QUALITY_SENSORS`, `CONF_VOC_SENSORS`, `CONF_PM25_SENSORS`, `CONF_PM10_SENSORS` | (environmental group) | Feed `ENVIRONMENTAL_INPUT_TYPES`; share `CONF_WEIGHT_ENVIRONMENTAL` |
| `CONF_POWER_SENSORS` | `power_sensors` | |
| `CONF_DOOR_SENSORS` | `door_sensors` | Active state default is `STATE_CLOSED` — a closed door is "evidence of occupancy" (wasp-in-box semantics), easy to misread as backwards |
| `CONF_WINDOW_SENSORS` | `window_sensors` | Active state default `STATE_OPEN` |
| `CONF_COVER_SENSORS` | `cover_sensors` | |

Per-type numeric `active_range` overrides on `AreaConfig` (e.g. a hypothetical `temperature_active_range`) are **dead code**: `data/entity.py`'s `create_from_config_spec()` does `getattr(self.config, f"{input_type}_active_range", None)` but `AreaConfig` never defines any such attribute for any type, so this always returns `None` and falls back to `data/entity_type.py` `DEFAULT_TYPES`' hardcoded range. The only thing that adapts numeric ranges at runtime is the separate `learned_active_range` mechanism populated by correlation analysis, not by config. (Verified: `grep -rn '_active_range' custom_components/area_occupancy/data/config.py` → no matches.)

### Sensor active-states / likelihoods

| CONF_* key | Default | Range/options | Stability |
|---|---|---|---|
| `CONF_MOTION_PROB_GIVEN_TRUE` | `DEFAULT_MOTION_PROB_GIVEN_TRUE`=0.95 | must exceed prob_given_false (`_validate_config`) | production |
| `CONF_MOTION_PROB_GIVEN_FALSE` | `DEFAULT_MOTION_PROB_GIVEN_FALSE`=0.005 | | production |
| `CONF_DOOR_ACTIVE_STATE` | `STATE_CLOSED` | one of `DOOR_STATES` options | production |
| `CONF_WINDOW_ACTIVE_STATE` | `STATE_OPEN` | one of `WINDOW_STATES` | production |
| `CONF_COVER_ACTIVE_STATES` | `[OPENING, CLOSING]` | subset of `COVER_STATES` | production |
| `CONF_APPLIANCE_ACTIVE_STATES` | `[ON, STANDBY]` | subset of `APPLIANCE_STATES` | production |
| `CONF_MEDIA_ACTIVE_STATES` | `[PLAYING, PAUSED]` | subset of `MEDIA_STATES` | production |

Motion is the **only** InputType with a user-configurable `prob_given_true`/`prob_given_false`. All other types' likelihoods are fixed at `data/entity_type.py` `DEFAULT_TYPES` values (not exposed in config_flow). `const.py`'s per-type `*_PROB_GIVEN_TRUE`/`*_PROB_GIVEN_FALSE`/`*_DEFAULT_PRIOR` constants (`MEDIA_PROB_GIVEN_TRUE`, `APPLIANCE_DEFAULT_PRIOR`, etc., lines 238-266) are **dead code** — referenced only inside `const.py` itself. The real, live defaults are `data/entity_type.py`'s `DEFAULT_TYPES` dict, and the two tables have drifted apart (e.g. `const.py` `MEDIA_PROB_GIVEN_TRUE=0.25` vs `entity_type.py` `InputType.MEDIA.prob_given_true=0.65`). **Do not add new sensor-type defaults to `const.py`'s dead table — put them in `data/entity_type.py` `DEFAULT_TYPES`.**

### Weights (per-area, `AreaConfig.weights`)

| CONF_* key | Default (`DEFAULT_WEIGHT_*`) | Config-flow range | Stability |
|---|---|---|---|
| `CONF_WEIGHT_MOTION` | 1.0 | 0-1 (`WEIGHT_MIN`/`WEIGHT_MAX` in config_flow.py) | production |
| `CONF_WEIGHT_MEDIA` | 0.7 | 0-1 | production |
| `CONF_WEIGHT_APPLIANCE` | 0.4 | 0-1 | production |
| `CONF_WEIGHT_DOOR` | 0.3 | 0-1 | production |
| `CONF_WEIGHT_WINDOW` | 0.2 | 0-1 | production |
| `CONF_WEIGHT_COVER` | 0.5 | 0-1 | production |
| `CONF_WEIGHT_ENVIRONMENTAL` | 0.1 | 0-1 | production |
| `CONF_WEIGHT_POWER` | 0.3 | 0-1 | production |
| `CONF_WASP_WEIGHT` (not `CONF_WEIGHT_WASP`!) | `DEFAULT_WASP_WEIGHT`=0.8 | 0-1 | production |

`CONF_WEIGHT_WASP` ("weight_wasp") and `CONF_WEIGHT_SLEEP` ("weight_sleep") are declared in `const.py` (lines 118, 109) but **never referenced anywhere else** — dead consts. The wasp weight actually used is `CONF_WASP_WEIGHT` ("wasp_weight", defined in the Virtual Sensor section of `const.py`), and it feeds **two** dataclass fields from one config key: `Weights.wasp` and `WaspInBox.weight` both read `data.get(CONF_WASP_WEIGHT, DEFAULT_WASP_WEIGHT)` (`data/config.py:570,588`). Don't be fooled into wiring the dead `CONF_WEIGHT_WASP`/`CONF_WEIGHT_SLEEP` keys when extending this area.

Separately, `const.py` also defines `MIN_WEIGHT=0.01`/`MAX_WEIGHT=0.99` — these are **not** the config-flow validation bounds; they're used only in `data/entity.py:731` as a sanity clamp when loading a weight value back out of the database.

### Thresholds / priors

| CONF_* key | Default | Range | Stability | Notes |
|---|---|---|---|---|
| `CONF_THRESHOLD` | `DEFAULT_THRESHOLD`=50.0 (UI, 0-100) | config-flow validates 1-100 (`_validate_config`, `invalid_threshold`); **but** the live `number.<area>_threshold` entity clamps 1.0-99.0 and does not re-run `_validate_config` at all | production, but 3 independent write paths with inconsistent bounds — see below | Stored on `AreaConfig.threshold` as `value/100.0` (0.0-1.0 float) |
| `CONF_MIN_PRIOR_OVERRIDE` | `DEFAULT_MIN_PRIOR_OVERRIDE`=0.0 (disabled) | NumberSelector 0.0-1.0 step 0.01; not re-validated server-side | production | Capped at runtime by `PRIOR_FLOOR_THRESHOLD_MARGIN`=0.01 below `threshold` — see #435 below |

`CONF_THRESHOLD` has three write paths: (1) config-flow wizard's parameters section, (2) the options-flow equivalent, (3) the live `number.py::Threshold` entity (`async_set_native_value` → `area.config.update_config({CONF_THRESHOLD: value})`), which bypasses `_validate_config` entirely and uses its own `native_min_value=1.0`/`native_max_value=99.0`. If you touch threshold validation, you must update all three call sites or you reintroduce this inconsistency (verify: `grep -n 'CONF_THRESHOLD' custom_components/area_occupancy/number.py custom_components/area_occupancy/config_flow.py`).

`MIN_PRIOR`/`MAX_PRIOR`/`MIN_PROBABILITY`/`MAX_PROBABILITY` = 0.01/0.99 (const.py:173-176) are the hard safety clamp everywhere in the Bayesian pipeline — see `aod-math-reference` for how `clamp_probability()` uses them. `TIME_PRIOR_MIN_BOUND`/`MAX_BOUND` = 0.03/0.9 (const.py:186-187) are a separate, tighter bound applied only to time-of-day priors.

Prior floor mechanism (issue #435, fixed): `Prior.value` (`data/prior.py:78-140`) applies `area.purpose.min_prior` and `config.min_prior_override` as floors, but both are capped at `floor_cap = max(MIN_PRIOR, config.threshold - PRIOR_FLOOR_THRESHOLD_MARGIN)`. A configured/purpose floor alone can never push a stale, no-evidence area's prior up to or above its own occupancy threshold — only a genuinely learned prior can cross the threshold.

### Decay

| CONF_* key | Default | Range | Stability |
|---|---|---|---|
| `CONF_DECAY_ENABLED` | `DEFAULT_DECAY_ENABLED`=True | bool | production |
| `CONF_DECAY_HALF_LIFE` | `DEFAULT_DECAY_HALF_LIFE`=0 (sentinel, see below) | 0, or 10-3600s (`_validate_config`, `invalid_decay_half_life`) | production — see PR #493 below, merged 2026-07-06, for the settled #481 bug history |

See "The sentinels" section below — this is the highest-blast-radius axis in this catalog per the maintainer's stated costliest-failure list (decay half-life config bugs).

### Wasp-in-box (per-area, `AreaConfig.wasp_in_box`)

| CONF_* key | Default | Stability |
|---|---|---|
| `CONF_WASP_ENABLED` | False | production |
| `CONF_WASP_MOTION_TIMEOUT` | `DEFAULT_WASP_MOTION_TIMEOUT`=300s | production |
| `CONF_WASP_WEIGHT` | `DEFAULT_WASP_WEIGHT`=0.8 | production (also feeds `Weights.wasp`, see above) |
| `CONF_WASP_MAX_DURATION` | `DEFAULT_WASP_MAX_DURATION`=3600s | production |
| `CONF_WASP_VERIFICATION_DELAY` | `DEFAULT_WASP_VERIFICATION_DELAY`=0 (disabled) | production |

### Purpose

| CONF_* key | Default | Options | Stability |
|---|---|---|---|
| `CONF_PURPOSE` | `DEFAULT_PURPOSE`="social" | 12 `AreaPurpose` values (`data/purpose.py` `PURPOSE_DEFINITIONS`) | production |

Purpose drives the decay half-life default (45s Passageway → 1200s Sleeping) and, for `SLEEPING` only, an `awake_half_life`=620s used outside the configured sleep window — see the sentinel section, and PR #493 below (merged 2026-07-06) for the now-fixed bug in that switch.

### Health

| CONF_* key | Default | Scope | Stability |
|---|---|---|---|
| `CONF_HEALTH_ENABLED` | `DEFAULT_HEALTH_ENABLED`=True | `IntegrationConfig` (global, not per-area) | production (landed PR #472) |

When False, `data/analysis.py` short-circuits both sensor-health and pipeline-health checks and calls `area.health_monitor.clear_all_issues()`, but leaves the in-memory `_unavailable_since` clock intact so re-enabling doesn't instantly trip every currently-unavailable sensor.

### Adjacency (config keys) — still experimental pending real-home validation

Merge status (PR #454, merged 2026-07-06) is stated once, in "Status: PR #454 (adjacent-areas) merged 2026-07-06" above — refer to that section for the merge fact; this section covers only the config keys themselves.

| Key | Default | Stability |
|---|---|---|
| `CONF_ADJACENT_AREAS` ("adjacent_areas") | `[]` | On `main` since PR #454 (see Status section above). Confirmed present: `grep -c CONF_ADJACENT_AREAS custom_components/area_occupancy/const.py` → 1. Behavior itself is still a candidate — unvalidated on real homes, not just unmerged code. |
| `ADJACENCY_TRANSITION_WINDOW_S` | 60 | experimental, hardcoded (not a `CONF_*`, no UI) |
| `ADJACENCY_RECENCY_HALF_LIFE_DAYS` | 30 | experimental, hardcoded |
| `ADJACENCY_TRAJECTORY_WINDOW_S` | 300 | experimental, hardcoded |
| `ADJACENCY_BOOST_GAIN` | 0.5 | experimental, hardcoded |
| `ADJACENCY_DECAY_MODIFIER_GAIN` | 0.75 | experimental, hardcoded |
| `ADJACENCY_DECAY_MODIFIER_MAX` | 1.75 | experimental, hardcoded |
| `ADJACENCY_N_SPECIFIC` / `ADJACENCY_N_HOUR` / `ADJACENCY_N_CHAIN` / `ADJACENCY_N_PAIR` | 5 / 20 / 50 / 20 | experimental, hardcoded (min-observation thresholds for a 6-level smoothing fallback) |

All ten `ADJACENCY_*` constants carry the in-code comment "First-pass values; tune from real data once Phase 3 is collecting transitions" (`const.py:189-221` on `main`) — there is still no real-recorder validation. None are user-configurable; only `CONF_ADJACENT_AREAS` (the neighbour list itself) has a config-flow UI. Treat any specific numeric claim about adjacency behavior as provisional — the code is now on `main`, but the tuning is not validated — see `aod-research-frontier` for the candidate-status tracking.

**adjacent_areas symmetric-write**: the config flow enforces mutual adjacency as a pure-function transform over the flat `CONF_AREAS` list (`config_flow.py:1614-1780`, on `main` since PR #454). `_normalize_adjacent_areas()` coerces any stored shape (None/str/list/tuple/set/other) to `list[str]`, defensively handling hand-edited storage JSON. `_apply_symmetric_adjacency(areas, updated_area)`: when area A saves neighbours `[B, C]`, it rewrites A's own row (self-reference stripped, sorted), adds A to B's and C's lists if missing, and removes A from any other area that used to list A but no longer should. `_strip_adjacency_references()` removes a deleted area's id from every surviving area when that area is removed. This mirroring is pure Python at config-flow save time — `AreaConfig._load_config` (`data/config.py:485-493`) just reads and string-filters the list; it does **not** itself enforce symmetry, so any other write path (e.g. a future service call) that touches `adjacent_areas` directly must replicate the mirror logic or symmetry will silently drift.

### Global settings (`IntegrationConfig`, entry-wide)

| CONF_* key | Default | Range | Stability |
|---|---|---|---|
| `CONF_SLEEP_START` | `DEFAULT_SLEEP_START`="23:00:00" | `TimeSelector` | production |
| `CONF_SLEEP_END` | `DEFAULT_SLEEP_END`="07:00:00" | `TimeSelector` | production |
| `CONF_HEALTH_ENABLED` | True | bool | production |
| `CONF_SENSOR_PRECISION` | `DEFAULT_SENSOR_PRECISION`=`ROUNDING_PRECISION`=2 | `NumberSelector` 0-2 int, clamped again in `IntegrationConfig.sensor_precision` (`max(0, min(2, precision))`, catches `ValueError`/`TypeError`/`OverflowError` → falls back to default) | production — merged 2026-07-06, PR #486 |
| `CONF_PEOPLE` | `[]` | list of person dicts, see below | production |

`CONF_SENSOR_PRECISION` controls the decimal precision that diagnostic sensors write to the HA recorder (0 decimals = whole percent). It was added specifically to cut recorder write volume (issue #467): measured 55% fewer rows at precision 0 vs the old unconditional 2-decimal writes. This is the canonical worked example for "how to add a global setting" — see the checklist below, which is verified against this exact PR.

### People (nested under `CONF_PEOPLE`, parsed into `PersonConfig`)

| Key | Default | Notes |
|---|---|---|
| `CONF_PERSON_ENTITY` | required | e.g. `person.seb` |
| `CONF_PERSON_SLEEP_SENSORS` | `[]` | current list key |
| `CONF_PERSON_SLEEP_SENSOR` | — | legacy single-sensor string key; migrated to `CONF_PERSON_SLEEP_SENSORS` by the v16→v17 migration, but `IntegrationConfig.people` *also* still reads it live as a fallback for any config that skipped migration |
| `CONF_PERSON_SLEEP_AREA` | required | HA area id |
| `CONF_PERSON_CONFIDENCE_THRESHOLD` | `DEFAULT_SLEEP_CONFIDENCE_THRESHOLD`=75 | int, parse errors fall back to default with a warning log |
| `CONF_PERSON_DEVICE_TRACKER` | `None` | optional override for home/away state |

### Misc per-area

| CONF_* key | Default | Stability |
|---|---|---|
| `CONF_MOTION_TIMEOUT` | `DEFAULT_MOTION_TIMEOUT`=300s | production |
| `CONF_EXCLUDE_FROM_ALL_AREAS` | `DEFAULT_EXCLUDE_FROM_ALL_AREAS`=False | production, added in the v17→v18 migration (see below) |

## The sentinels

### Decay half-life 0 = "use purpose default"

`DEFAULT_DECAY_HALF_LIFE = 0` (`const.py:124`). Resolution happens once, at load time, in `AreaConfig._load_config` (`data/config.py:573-577`):

```python
half_life_value = int(data.get(CONF_DECAY_HALF_LIFE, DEFAULT_DECAY_HALF_LIFE))
if half_life_value == 0:
    half_life_value = int(get_default_decay_half_life(self.purpose))
```

`get_default_decay_half_life()` (`data/purpose.py:247-263`) looks up `PURPOSE_DEFINITIONS[purpose].half_life`. The 12 purpose defaults range from 45s (Passageway) to 1200s (Sleeping, with an `awake_half_life`=620s used outside the sleep window). Config-flow validation (`_validate_config`, `config_flow.py:2105-2113`) allows exactly `0` or `10 <= value <= 3600` — note Sleeping's own default (1200s) is **outside** that 3600s ceiling, so it's only reachable via the 0-sentinel auto-path, never as an explicit typed value.

### The #440 normalisation rule

Issue #439 (2026-04-17): a user's custom half-life appeared to save but reverted on reopen, because `Purpose.is_purpose_half_life()` used to return True whenever the entered value matched **any** purpose's built-in default (12 round values), silently normalising, e.g., a Living Room user's `600s` (= Office's default) back to the `0` sentinel. Fixed same-day by PR #440: the comparison is now scoped to only the **currently-selected** purpose's default (`data/purpose.py:125-153`):

```python
@staticmethod
def is_purpose_half_life(value: float, purpose: str | None = None) -> bool:
    if value == 0:
        return True
    if purpose is None:
        return False
    return PURPOSE_DEFINITIONS[AreaPurpose(purpose)].half_life == value
```

This is called from `config_flow.py::_apply_purpose_based_decay_default` (`config_flow.py:1545-1565`) at save time: **if the user's entered value equals the selected purpose's own default (or is empty), normalise to 0** so the value stays purpose-driven across a later purpose change; any other custom value is preserved untouched. **Rule for anyone touching this code: always pass the currently-selected purpose, never compare against all purposes' defaults.**

This exact bug class recurred in a different code path: issue #481 (a Bedroom/SLEEPING area's custom 10s half-life was overridden by the purpose's `awake_half_life`=620s outside the sleep window, because the sleep/awake switch in `Decay._resolve_purpose_half_life()` — `data/decay.py:81-124` — applied unconditionally instead of only when the half-life still equalled the purpose default). **Settled**: PR #493 merged 2026-07-06, closing #481. `_resolve_purpose_half_life()` now has an explicit guard (`if self._base_half_life != self._purpose.half_life: return self._base_half_life`) before the sleep/awake switch runs, plus the separate adjacency `modifier_factor` multiplying on top of the resolved base in `half_life` (`data/decay.py:76-79`). Its PR body explicitly says it "mirrors the custom-vs-default semantics established for #440." Verify current code with `sed -n '81,124p' custom_components/area_occupancy/data/decay.py`.

## How missing keys default, and why no CONF_VERSION bump is needed for additive keys

Every read of a `CONF_*` key in `AreaConfig._load_config` and `IntegrationConfig` properties goes through `data.get(CONF_X, DEFAULT_X)` (or `config_entry.options.get(...)`). A config entry saved before a key existed simply doesn't have it in its dict — `.get()` returns the default, no migration required, no `CONF_VERSION` bump required, **as long as the default produces correct/safe behavior for pre-existing configs.**

Two real precedents, both confirmed in the repo:

1. **PR #486** (`CONF_SENSOR_PRECISION`, merged 2026-07-06, commit `7e3a856`) added a brand-new global option with zero `CONF_VERSION` involvement — it lives in `config_entry.options`, read live via `.get()` with a clamped default. No migration, no version touch. `git show 7e3a856 -- custom_components/area_occupancy/const.py | grep CONF_VERSION` → no output.
2. **v17→v18** (`CONF_EXCLUDE_FROM_ALL_AREAS`, `migrations.py:571-581`) *did* bump `CONF_VERSION`, but the migration itself does no data mutation — the comment says so explicitly: "No data changes needed — missing key handled by `AreaConfig._load_config()`." The bump here was belt-and-suspenders (a version marker for tooling/tests), not a technical requirement of the additive change itself.

Rule of thumb: **a purely additive, `.get()`-defaulted key never needs a `CONF_VERSION` bump.** Only bump when you need to (a) mutate/rename/restructure existing stored data, or (b) force a one-time side effect (e.g. a DB reset) on upgrade. Bumping `CONF_VERSION` is not free: any mismatch versus the DB's stored schema version triggers `db/maintenance.py`'s `_ensure_schema_up_to_date`, which **deletes and recreates the entire SQLite database** (wipes all learned priors/history) — see the DB layout section above and `aod-debugging-and-history` for that mechanism. This is exactly why the adjacent-areas feature's own `AreaTransitions` table and `adjacent_areas` column were deliberately kept out of the version bump (`migrations.py:583-588`, merged to `main` 2026-07-06 as part of PR #454, `CONF_VERSION` still 18): it's additive (new column defaults on load, new `AreaTransitions` table — one of 15 tables in `db/schema.py` now — created via `Base.metadata.create_all(checkfirst=True)`), so bumping would have wiped every user's learned history for no reason.

## migrations.py rules

`async_migrate_entry` (`migrations.py:515` on) runs under a module-level `asyncio.Lock` to prevent concurrent migrations. Current ladder (`CONF_VERSION=18`, `CONF_VERSION_MINOR=0`, `const.py:33-34`):

| From → To | What happens | Idempotent? |
|---|---|---|
| `< 14` | `async_reset_database_if_needed()` deletes `.storage/area_occupancy.db` (+`-wal`/`-shm`) — breaking schema change from v13 | yes, guarded by file-exists checks |
| `13 <= v < 15` | `_migrate_energy_to_power()` strips legacy `energy_sensors`/`weight_energy` keys; unconditionally bumps to 16 even if nothing was found | yes (gated by version range) |
| `== 16` | `_migrate_sleep_sensor_to_list()` converts `CONF_PERSON_SLEEP_SENSOR` (str) → `CONF_PERSON_SLEEP_SENSORS` (list) in both `data` and `options`; bumps to 17 | yes |
| `== 17` | Pure version bump to 18 for `CONF_EXCLUDE_FROM_ALL_AREAS` — no data mutation | yes |
| `< 13` (true legacy single-area entries) | `_combine_config_entries()` merges every such entry into one target entry's `CONF_AREAS` list (deterministic target = lowest `entry_id`); invalid areas dropped; old entries marked deleted, registries cleaned | yes (only entries still `< 13` are touched) |

All numeric-version branches are gated with `if config_entry.version == N` (or a range check), so re-running the whole function on an already-migrated entry is a no-op — this is the idempotency guarantee AGENTS.md requires. When you add a new migration step: gate it the same way, mutate both `data` and `options` dicts if the key could live in either, and log what you did.

## Checklist: adding a new config option

Verified end-to-end against how PR #486 added `CONF_SENSOR_PRECISION` (a global setting) — for a per-area/per-sensor-type option, steps 2-4 target `AreaConfig`/`Sensors`/`Weights` instead of `IntegrationConfig`.

1. **`const.py`**: add `CONF_<NAME>: Final = "<snake_case_key>"` and `DEFAULT_<NAME>: Final = <value>`. Group it near its section (weights, decay, global settings, etc.) — don't scatter.
2. **Schema section in `config_flow.py`**: add a `vol.Required`/`vol.Optional` entry with an appropriate selector (`NumberSelector`, `BooleanSelector`, `TimeSelector`, `DurationSelector`, …) and range/step matching the intended domain. For a global setting this goes in `_create_global_settings_schema` (`config_flow.py:1919-1950`); for a per-area setting, into the relevant `_create_*_section_schema` **and** wired into `_nest_config_for_sections()` — skip that second step and "suggested values" won't repopulate on edit.
3. **`strings.json` AND `translations/en.json` — BOTH files, every time.** PR #486 added the `sensor_precision` label+description to both in the same commit (`git show 7e3a856 -- custom_components/area_occupancy/strings.json custom_components/area_occupancy/translations/en.json`). The project has an existing, unfixed drift where `strings.json` is missing 10 keys that `en.json` has (the whole `services.*` block plus `person_already_configured` under both `config.error` and `options.error`) — confirmed via a JSON key-diff (`python3` flatten-and-diff, 0 keys unique to `strings.json`, 10 unique to `en.json`). Do not repeat that mistake; `strings.json` is HA's canonical source that hassfest (`validate.yml`) checks and other locales derive from.
4. **Parsing with clamp**: add the property/field read via `.get(CONF_X, DEFAULT_X)`. If the value has a valid range, clamp defensively at the read site (see `IntegrationConfig.sensor_precision`'s `max(0, min(2, precision))` inside a `try/except (ValueError, TypeError, OverflowError)`) — never trust that the selector's client-side bounds were actually respected (hand-edited YAML/JSON, old snapshots, API calls all bypass the selector).
5. **Server-side validation** in `_validate_config` if the config-flow UI doesn't already fully constrain it (compare: `CONF_SENSOR_PRECISION` relies solely on the `NumberSelector` + read-site clamp and has no `_validate_config` entry — acceptable because the read-site clamp is the true backstop; `CONF_THRESHOLD` and `CONF_DECAY_HALF_LIFE` do have explicit `_validate_config` checks because they gate deeper pipeline behavior).
6. **Tests**: `tests/test_config_flow.py` (schema/flow) and `tests/test_data_config.py` (parsing/clamping) — PR #486 touched exactly these two files for its non-UI logic (`git show 7e3a856 --stat`).
7. **Docs**: `docs/docs/getting-started/configuration.md` (or the relevant `features/*.md` page) — PR #486 added 3 lines there. See `aod-docs-and-positioning` for house style.
8. **No `CONF_VERSION` bump** unless the new key requires migrating *existing* stored data (see previous section) — a purely additive `.get()`-defaulted key does not need one.

## Entity-registry enabled-default for diagnostic sensors (#488)

PR #488 (merged, commit `2c28849`) added `set_enabled_default(False)` (`sensor.py:90-92`, sets `self._attr_entity_registry_enabled_default`) to 7 diagnostic sensor classes: `PriorsSensor`, `EvidenceSensor`, `DecaySensor`, `PresenceProbabilitySensor`, `EnvironmentalConfidenceSensor`, `ActivityConfidenceSensor`, `SensorHealthSensor` (verify: `grep -n set_enabled_default custom_components/area_occupancy/sensor.py`). `ProbabilitySensor` and `DetectedActivitySensor` remain enabled by default. Rationale: these sensors update on the 10s decay timer and were measured writing ~16k recorder rows in 3 hours on a 6-area install (issue #467).

**Restore caveat, load-bearing for anyone touching entity registration**: this only applies at **first registration**. HA's `entity_registry.async_get_or_create()` on an *already-existing* registry entry routes to `_async_update_entity`, which has no `disabled_by` parameter and structurally cannot touch it — so existing installs upgrading through this change keep whatever enabled/disabled state they already had (verified by a regression test that seeds an "existing install" registry entry as enabled and asserts it stays enabled). **Deleting and re-adding an area counts as a fresh registration** — diagnostics come back disabled in that case, they are not "restored" the way an in-place reload/upgrade preserves them. If a user reports "my diagnostics sensors disappeared after I removed and re-added the area," this is why, not a bug.

## Re-verification one-liners (config)

Run these to regenerate this catalog's facts against current `main` (the working tree is on `main` post-merge; `git branch --show-current` should show `main`, not `feat/adjacent-areas`):

```bash
# Full CONF_*/DEFAULT_* symbol list with line numbers
grep -n '^CONF_\|^DEFAULT_' custom_components/area_occupancy/const.py

# Confirm adjacency is merged (expect >=1 on main as of 2026-07-06)
grep -c ADJACENCY_ custom_components/area_occupancy/const.py
gh pr view 454 --json state,mergeStateStatus,mergeable

# Confirm CONF_SENSOR_PRECISION / diagnostic-disable status (expect MERGED)
gh pr view 486 --json state,mergedAt
gh pr view 488 --json state,mergedAt

# Find any const.py key referenced nowhere else (candidate dead code)
for sym in $(grep -oP '^(CONF|DEFAULT)_\w+(?=: Final)' custom_components/area_occupancy/const.py); do
  n=$(grep -rl "$sym" custom_components/area_occupancy --include='*.py' | grep -v '/const.py$' | wc -l)
  [ "$n" -eq 0 ] && echo "dead: $sym"
done

# strings.json vs translations/en.json key-diff (expect 10 keys only in en.json, 0 only in strings.json, as of 2026-07-06)
python3 - <<'EOF'
import json
def flatten(d, p=''):
    out={}
    for k,v in d.items():
        key=f'{p}.{k}' if p else k
        out.update(flatten(v,key) if isinstance(v,dict) else {key:v})
    return out
s=flatten(json.load(open('custom_components/area_occupancy/strings.json')))
e=flatten(json.load(open('custom_components/area_occupancy/translations/en.json')))
print('only in en.json:', len(set(e)-set(s)))
print('only in strings.json:', len(set(s)-set(e)))
EOF

# Current CONF_VERSION / migration ladder
grep -n 'CONF_VERSION\|CONF_VERSION_MINOR' custom_components/area_occupancy/const.py
grep -n 'config_entry.version ==\|config_entry.version <\|config_entry.version >=' custom_components/area_occupancy/migrations.py

# Confirm the #440 rule and decay-half-life validation bounds are unchanged
sed -n '124,155p' custom_components/area_occupancy/data/purpose.py
grep -n 'invalid_decay_half_life' -A3 -B3 custom_components/area_occupancy/config_flow.py

# entity_registry_enabled_default call sites (expect 7)
grep -n 'set_enabled_default(False)' custom_components/area_occupancy/sensor.py | wc -l

# Fixed-bug status for the #440-recurrence (PR #493) — confirm merged
gh pr view 493 --json state,mergedAt
```

## Provenance and maintenance (config)

Date-stamped 2026-07-06 (post-merge sweep), integration version still 2026.5.17 — none of the 2026-07-06 merge wave is in a tagged release yet (main branch HEAD `17b71d2`; HEAD drifts — re-derive with `git log -1 --oneline origin/main`). Facts in this skill were verified directly against the repo unless noted otherwise:

- Directly verified by reading source: full `const.py` (all `CONF_*`/`DEFAULT_*`/`ADJACENCY_*` lines and values), `data/config.py` (`IntegrationConfig`, `AreaConfig`, `Sensors`/`Weights`/`DecayConfig`/`WaspInBox` dataclasses, `_load_config`, `update_config`), `data/purpose.py` (`is_purpose_half_life`, `PURPOSE_DEFINITIONS`, `get_default_decay_half_life`), `migrations.py` (`async_migrate_entry` full ladder, the adjacency no-version-bump comment at `migrations.py:583-588`), `config_flow.py` (`_create_global_settings_schema`, `_apply_purpose_based_decay_default`, `_validate_config`, `_normalize_adjacent_areas`/`_apply_symmetric_adjacency`/`_strip_adjacency_references`), `sensor.py` (`set_enabled_default` call sites), `data/entity.py:731` (`MIN_WEIGHT`/`MAX_WEIGHT` usage), `number.py` (`Threshold` entity bounds), `data/decay.py` (`_resolve_purpose_half_life()`'s #481 guard and the `modifier_factor` multiply in `half_life`), `db/schema.py` (`AreaTransitions`, now one of 15 tables).
- Directly verified by command: `git log -1 --oneline` (confirmed working tree is on `main` at `17b71d2`, post-merge); `grep -c ADJACENCY_ const.py` → non-zero (adjacency merged); `gh pr view 454/486/488/491/492/493/494/495/496/472` all report `MERGED`; `git show 7e3a856 --stat` (PR #486's file list) and `git show 7e3a856 -- const.py | grep CONF_VERSION` → empty; the Python key-diff script for `strings.json` vs `en.json` (10 keys only in en.json, 0 only in strings.json); dead-const grep for `CONF_WEIGHT_SLEEP`, `CONF_WEIGHT_WASP`, `DEFAULT_SLEEP_WEIGHT`, and the per-type `const.py` probability constants.
- Taken from the discovery dossier and spot-checked (not independently re-derived line-by-line): the exact historical narrative timing of issues #439/#481/#467/#435 and PR #440's commit hash (`68d576b`) — the causal chain and current code state were verified directly, but the dossier's issue-comment quotes were not re-fetched via `gh issue view` in this session.
- PRs #491, #492, #493, #494, #454 (and #486, #488, #495, #496) are **merged to `main`** — all confirmed `MERGED` via `gh pr view <n>` as of 2026-07-06. None have shipped in a tagged release yet (integration version is still 2026.5.17); phrase user-facing claims accordingly.
