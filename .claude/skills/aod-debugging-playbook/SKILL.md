---
name: aod-debugging-playbook
description: Use when triaging a live Area Occupancy Detection symptom report — occupancy stuck on/won't clear, occupancy won't turn on, probability pinned at 0.99 or 0.01, wrong or spammy repair issues, a config setting that seems to be "ignored" (especially decay half-life or min_prior_override), the database growing/slow, config-flow errors, or entities showing 100% occupied right after a restart. Also load this before touching timezone/DST datetime handling, decay half-life resolution, or prior/global_prior calculation code — these are the project's three most expensive historical bug classes and this skill has the exact traps.
---

# AOD debugging playbook

## What this covers

A symptom-to-fix runbook for Area Occupancy Detection's real, recurring failure modes — the ones that have burned real debugging time across this project's history. Each symptom below gives first-checks, a discriminating experiment (an exact query/log/diagnostics field to look at, not a guess), and the fix pattern. Three specific bug classes get their own "burned lesson" trap boxes with the full incident story because they recurred multiple times: timezone/DST handling, decay half-life config normalization, and global-prior inflation.

## When NOT to use this

- You need the *why* behind a historical incident in more narrative depth (full commit-by-commit archaeology, dead branches, abandoned drafts) → `aod-failure-archaeology`.
- You're deciding whether a proposed code change is safe to ship / needs a migration / could break configs → `aod-change-control`.
- You need the Bayesian formula itself (sigmoid/logit pipeline, exact constants, clamping) rather than "why is my number wrong" → `bayesian-occupancy-reference`.
- You need the full config-surface reference (every `CONF_*` key, defaults, what's user-facing vs internal) rather than "this one setting looks ignored" → `aod-config-and-flags`.
- You're improving prior/likelihood accuracy as a project, not debugging one report → `aod-learning-accuracy-campaign`.
- You need diagnostics/tooling mechanics (how the simulator works, `visualize_distributions.py` internals, diagnostics JSON schema in full) beyond what's quoted here → `aod-diagnostics-and-tooling`.

## Debug logging recipe

Add to `configuration.yaml`, then restart Home Assistant:

```yaml
logger:
  logs:
    custom_components.area_occupancy: debug
```

Logs land in **Settings → System → Logs** and, in the devcontainer, `config/home-assistant.log`. This is step 3 of the project's own debugging order — do this *last*, not first (see next section).

Verified: `docs/docs/technical/debug.md` lines 11-20; `config/configuration.yaml` already ships this block for the dev sensor rig.

## Order of operations: diagnostics → repairs → logs

This is the project's own documented convention, not something inferred — follow it in this order:

1. **Download diagnostics first.** Integration card (Settings → Devices & Services → Area Occupancy Detection) → **⋮** menu → **Download diagnostics**. No config change needed; captures every prior/weight/evidence/decay/correlation/health value in one JSON file.
2. **Check Settings → System → Repairs** for `sensor_health_*` / `pipeline_health_*` issues. A stuck/unavailable sensor or stale cache is a common root cause and is surfaced automatically.
3. **Only then** enable debug logging and reproduce live.

Verified: `docs/docs/technical/debug.md` lines 1-9.

### Reading the diagnostics export

Top-level shape: `{"integration": {...}, "areas": [...], "database": {...}}`. Each `areas[]` entry has `current` (live snapshot: `probability`, `occupied`, `decay_factor`, `active_entity_count`, `decaying_entity_count`, optional `adjacency`), `prior` (includes `global_prior`, `min_prior_floor_applied`), `config` (weights, decay, wasp/sleep settings), `entities[]` (per-sensor `weight`, `prob_given_true/false`, `evidence`, `previous_evidence`, `last_updated`, `analysis_error`, `correlation_strength`), and `health`. Every subsection is wrapped in try/except so one failure surfaces as a sibling `<section>_error` key instead of nuking the whole export.

Cheat-sheet (from the project's own diagnostics doc):

| Question | Look at |
|---|---|
| Why is the area stuck occupied? | `current.probability`, `current.decay_factor`, any `entities[]` with unexpected `evidence: true`, a high `correlation_strength` paired with a frozen `last_updated`, `prior.min_prior_floor_applied` (non-`none` = a purpose/override floor is holding the value up) |
| Has learning finished? | `prior.global_prior` (null until learned); compare `database.prior_count` to `area_count × 168` |
| Why is correlation broken for sensor X? | `entities[].analysis_error` |
| Is health monitoring stale? | `health.last_check` |
| Is the occupied-intervals cache stale? | `database.occupied_intervals_cache.<area>.valid` |

Verified: `custom_components/area_occupancy/diagnostics.py` (`_area_snapshot`, `_collect_db_stats`, `async_get_config_entry_diagnostics`); `docs/docs/technical/diagnostics.md` lines 1-116.

You can also query the SQLite DB directly (`config/.storage/area_occupancy.db` — a normal SQLite3 file) or run `python scripts/visualize_distributions.py "<Area>" <entity_id>` to plot a sensor's learned occupied-vs-unoccupied distribution against its raw data. See `aod-diagnostics-and-tooling` for the full toolset.

---

## Symptom → triage table

### 1. Occupancy stuck on / won't clear

| First checks | Discriminating experiment |
|---|---|
| Is `decay.enabled` true for this area? | Diagnostics `areas[].config.decay.enabled` |
| What half-life is actually in effect? Is `decay_half_life` the `0` sentinel ("use purpose default")? | Diagnostics `entities[].decay` vs the area's `purpose` default (see Purpose table below) — if stored value is `0`, the *effective* half-life is `get_default_decay_half_life(purpose)`, not literally instant |
| Is this a Wasp-in-Box or Sleep-presence entity? Both bypass purpose/sleep semantics and have their own fixed half-life | Wasp entities: `half_life = 0.1s` (should clear in well under a second — if one looks stuck, the wasp door/motion wiring is wrong, not decay). Sleep-presence virtual entities: `SLEEP_PRESENCE_HALF_LIFE = 7200s` (2h) is *intentional* persistent presence, not a bug |
| Bedroom (SLEEPING purpose) outside the configured sleep window: is a **custom** half-life being silently replaced by the purpose's `awake_half_life` (620s)? | See burned lesson #2 below (issue #481, fixed by PR #493, merged 2026-07-06) — compare the diagnostics `config.decay` stored half-life against `Purpose.awake_half_life`; if they still differ and the room still takes ~620s (10 min) to clear, the fix has regressed |
| Is a purpose/override prior floor holding probability near/above threshold? | Diagnostics `prior.min_prior_floor_applied` (`none`/`purpose`/`override`) — note the floor is capped at `threshold - 0.01` so it alone can never push probability *above* threshold, but it can make an area look "always borderline occupied" |
| Is an adjacent area's silence extending this area's decay? (adjacent-areas feature, merged 2026-07-06 via #454) | Diagnostics `current.adjacency.decay_modifier` — capped at 1.75× base half-life. Still unvalidated on real homes — treat unexpected values as a candidate-feature edge case, not necessarily user error |

Fix pattern: identify which of the above five mechanisms (decay disabled, sentinel resolution, wasp/sleep bypass, purpose-vs-custom half-life bug, prior floor) is actually in play before touching code — they produce visually identical "won't clear" symptoms but have completely different fixes.

Verified: `custom_components/area_occupancy/data/decay.py` (half-life resolution, `decay_factor` floors to 0.0 below a 5% threshold); `data/entity.py` lines ~763-774 and ~859-870 (wasp/sleep half-life overrides, `SLEEP_PRESENCE_HALF_LIFE` import); `data/prior.py` (floor capped at `threshold - PRIOR_FLOOR_THRESHOLD_MARGIN`, `const.py:180` `PRIOR_FLOOR_THRESHOLD_MARGIN = 0.01`).

### 2. Occupancy won't turn on

| First checks | Discriminating experiment |
|---|---|
| Is the sensor's `evidence` actually `True`? Evidence semantics are **not** "on = active" for every type | Diagnostics `entities[].evidence` — note `InputType.DOOR`'s default `active_states = [STATE_CLOSED]`: a **closed** door is the active/evidence-supporting state (wasp-in-box semantics — someone closed the door behind them). This is the single easiest thing to misread as backwards |
| Is `entity.state` unavailable/unknown? | `evidence` returns `None` (not `False`) for unavailable/unknown/empty states — check the raw HA entity state, not just AOD's evidence field |
| Is `effective_weight` (= `weight × information_gain`) much lower than the configured `weight`? | Diagnostics `entities[].analysis_error` — if it's anything other than `NOT_ANALYZED`/`MOTION_EXCLUDED`, `information_gain` is force-clamped to `0.1` regardless of the sensor's real likelihoods, silently muting a "failed" correlation |
| Is the area `threshold` set unexpectedly high? | Diagnostics `areas[].threshold` (UI shows 0-100%, internal comparison is against `/100`; default is `50.0`) |
| Strength multiplier: motion/sleep evidence pushes ~1.5× harder per unit of likelihood than other types, and this is **not** configurable via the options flow | `data/entity_type.py` `DEFAULT_TYPES[...]["strength_multiplier"]` (3.0 for MOTION/SLEEP, 2.0 for everything else) — if a non-motion sensor "feels weak" compared to motion, this is why, and it's by design |

Fix pattern: reproduce with diagnostics, not logs — the `evidence`/`analysis_error`/`effective_weight` chain explains almost every "sensor is on but occupancy isn't" report without needing debug logging.

Verified: `custom_components/area_occupancy/data/entity.py` lines ~335-360 (`evidence` property), ~309-332 (`information_gain`/`effective_weight`); `data/entity_type.py` (`DOOR` `active_states=[STATE_CLOSED]`, `strength_multiplier` table); `const.py` `DEFAULT_THRESHOLD = 50.0`.

### 3. Probability pinned at 0.99 or 0.01

| First checks | Discriminating experiment |
|---|---|
| Both are hard clamps, not calculation artifacts: `MIN_PROBABILITY=0.01`, `MAX_PROBABILITY=0.99` (same bounds for `MIN_PRIOR`/`MAX_PRIOR`) | `const.py` lines 170-173 — these are floors/ceilings applied by `clamp_probability()`, so "pinned at 0.99" always means *something upstream* pushed the logit very high, not that 0.99 is a calculated value in its own right |
| Is `0.01` actually a **safe-fallback** value rather than a learned prior? | `data/analysis.py` sets `global_prior = 0.01` explicitly on invalid interval bounds or clock-skew guards, logging `"fallback due to invalid interval bounds"` / `"fallback due to clock skew"` — check the log line, don't assume it was learned |
| Is `0.99` from **global-prior denominator inflation** during a long quiet stretch? | See burned lesson #3 below (issue #483, fixed by PR #491, merged 2026-07-06) — `actual_period_end` is now always `now`, so this should no longer reproduce; if `global_prior` still keeps climbing every hourly recalculation during quiet periods, the fix has regressed |

Fix pattern: don't chase weight/threshold config first — pinned values almost always trace to the prior-calculation denominator or a fallback path, both in `data/analysis.py`, not to sensor configuration.

Verified: `const.py` lines 170-176 (`MIN/MAX_PROBABILITY`, `MIN/MAX_PRIOR`); `data/analysis.py` (`calculate_and_update_prior`, fallback `set_global_prior(0.01)` call sites).

### 4. Repair issues that are wrong (false positives / spam / won't stay dismissed)

| First checks | Discriminating experiment |
|---|---|
| Is the integration-level kill switch off? | `CONF_HEALTH_ENABLED` (`const.py:93`), default `True` (`const.py:138`) — if `False`, `area.health_monitor.clear_all_issues()` runs instead of checks (added PR #472 to close issue #463's "40+ issues every morning") |
| Is the stuck-active threshold purpose-appropriate? | Check the effective threshold = base × purpose multiplier (e.g. motion in a bedroom: 8h × 6 = 48h). The canonical threshold/multiplier/exemption table lives in `aod-diagnostics-and-tooling` §2. Historical context: motion's base was 2h before PR #474, which caused the original false-positive wave (issues #465, #468) |
| Is it a `media_player.*` "unavailable" complaint? | `_UNAVAILABLE_EXEMPT_PREFIXES = ("media_player.",)` — TVs/speakers going unavailable when powered off is exempted entirely from the unavailable check (PR #474, closing issue #466's "TV off overnight" spam) |
| Did a dismissed/ignored issue "come back"? | Check `_is_ignored()` — it reads `ir.async_get(hass).async_get_issue(...)`'s `dismissed_version` (must be a `str`, not just non-`None`, to reject stale test mocks). If a recurring condition (e.g. nightly TV-off) keeps recreating a "new" issue despite Ignore, this is PR #473's fix area (issue #463) — verify the fix is actually present: resolved issues are now partitioned into truly-resolved (deleted) vs user-ignored (left alone) so deleting doesn't wipe the ignore flag |
| Is `InputType.SLEEP` showing up in health checks at all? | It shouldn't — `_EXCLUDED_TYPES = {InputType.SLEEP}` excludes it globally |

Repair issue ID format (useful for grepping `core.issue_registry`): sensor-scope = `sensor_health_{area_id}_{entity_id_with_dots_replaced}_{type}`; pipeline-scope = `pipeline_health_{area_id}_{type}`. Keyed on the stable HA `area_id`, not area name, so renames don't orphan issues.

Fix pattern: check the toggle first (fastest), then the purpose/threshold table, then the ignore-state mechanics — in that order of likelihood.

Verified: `custom_components/area_occupancy/data/health.py` lines 39-116 (thresholds/multipliers/exemption), 293-323 (`_is_ignored`), 824-901 (`_update_repair_issues` partitioning) — identical on `main` and this working tree; `const.py:93,138` (`CONF_HEALTH_ENABLED`/`DEFAULT_HEALTH_ENABLED`, verified against `main`); `data/config.py:189-198` (`IntegrationConfig.health_enabled`); commits `3471e7a` (#474), `b9df513` (#473), `67e53ac` (#472).

⚠️ **Docs trap**: `docs/docs/features/sensor-health.md` is stale as of 2026-07-06 — it still says the motion stuck-active threshold is 2 hours and doesn't mention the purpose multipliers, the `media_player.*` exemption, the `health_enabled` toggle, or the sticky-ignore fix. Trust the code (`data/health.py`) over that page. (Re-check before relying on this: `grep -n "2 hours" docs/docs/features/sensor-health.md`.)

### 5. Custom setting ignored / silently reset

| First checks | Discriminating experiment |
|---|---|
| Is this `decay_half_life`? Its `0` sentinel means **"use purpose default"** — a stored `0` is not literally a zero-second half-life | Diagnostics `config.decay` shows the *stored* value; if it's `0`, cross-reference `get_default_decay_half_life(purpose)` (or the Purpose table below) for the value actually in effect. See burned lesson #2 for the exact historical bug in *how* values get normalized to this sentinel |
| Is this `min_prior_override`? Its `0.0` sentinel means **"disabled"**, a *different* semantic from decay's "auto" — don't conflate the two 0-sentinels | `const.py:134` `DEFAULT_MIN_PRIOR_OVERRIDE = 0.0  # 0.0 = disabled by default` |
| Did the config-flow silently round-trip a custom value to the sentinel on save? | Check `Purpose.is_purpose_half_life(value, purpose)` — as of 2026-07-06 this is correctly scoped to compare **only** against the *selected* purpose's own default (fixed by #440; the legacy bug matched against *any* purpose's default, see burned lesson #2) |
| Is the value out of the accepted config-flow range? | Decay half-life config-flow validation accepts `0` (auto) or `10 ≤ value ≤ 3600`; anything else raises `errors[CONF_DECAY_HALF_LIFE] = "invalid_decay_half_life"` — note this range **cannot literally express** the SLEEPING purpose's own default of 1200s except via the `0` sentinel, which is expected, not a bug |

Fix pattern: for any "my custom X is being ignored" report, first determine whether X uses a magic-sentinel-plus-lookup pattern (decay half-life does; most other settings don't) before assuming a general config-flow bug — this specific pattern is the one with the incident history.

Verified: `custom_components/area_occupancy/data/purpose.py` (`is_purpose_half_life`, docstring cites #439); `config_flow.py` (`_apply_purpose_based_decay_default`, ~line 1539, validation ~lines 1966-1971); `const.py:134`.

### 6. Database growing / slow

| First checks | Discriminating experiment |
|---|---|
| Which retention constant applies? Two different numbers exist and the docs site states a third (wrong) one | `RETENTION_DAYS = 365` (`const.py:238`) is what actually hard-deletes raw interval rows (`db/operations.py::prune_old_intervals`). `RETENTION_RAW_INTERVALS_DAYS = 28` (`const.py:248`) only controls when raw intervals get rolled into daily aggregates (`db/aggregation.py`) — it does not delete anything. **`docs/docs/technical/database-schema.md` claims "Raw intervals: 60 days" — this is stale/wrong; trust the constants, not that doc** (re-check: `grep -n "60 days" docs/docs/technical/database-schema.md`) |
| Are diagnostic sensors writing to the recorder unnecessarily? | 7 sensor classes (`PriorsSensor`, `EvidenceSensor`, `DecaySensor`, `PresenceProbabilitySensor`, `EnvironmentalConfidenceSensor`, `ActivityConfidenceSensor`, `SensorHealthSensor`) are disabled-by-default **for newly registered areas only** (PR #488, merged 2026-07-06) — existing installs' enabled/disabled state is preserved across upgrades (HA's `async_get_or_create` on an existing registry entry can't touch `disabled_by`). A full delete+re-add of an area *does* count as fresh registration and comes back with diagnostics disabled |
| Is state precision inflating row count? | `CONF_SENSOR_PRECISION` (default `DEFAULT_SENSOR_PRECISION = ROUNDING_PRECISION = 2` decimal places) — lowering to 0 (whole-percent) measurably cuts recorder rows (PR #486, merged 2026-07-06, measured 55-79% fewer rows in the reporting install) |
| Is correlation analysis running on too little data / too often? | `MIN_CORRELATION_SAMPLES = 50` (`const.py:286`) gates whether a correlation is computed at all, and confidence is discounted below full strength until well above 50 samples |

Fix pattern: issue #467 ("throttle/gate DB writes") is the umbrella tracking issue and remains **open** as of 2026-07-06 pending real-world numbers from #486/#488 — don't assume it's fully closed just because those two PRs landed.

Verified: `const.py:238,248,286,231-233` (all against `main`); `db/operations.py` (`prune_old_intervals` uses `RETENTION_DAYS`); `db/aggregation.py` (uses `RETENTION_RAW_INTERVALS_DAYS`); `custom_components/area_occupancy/sensor.py` (7 `set_enabled_default(False)` call sites, identical on `main` and this working tree); `gh pr view 486`/`gh pr view 488` (both `MERGED`, `mergedAt: 2026-07-06`).

### 7. Config flow errors

| First checks | Discriminating experiment |
|---|---|
| Which `errors[key]` fired? | Grep `config_flow.py` for the literal string, e.g. `invalid_decay_half_life`, `invalid_threshold`, `invalid_weight`, `purpose_required`, `area_already_configured`, `area_not_found`, `person_already_configured`, `motion_required`, `prob_true_must_exceed_false`, `door_state_required`/`window_state_required`/`cover_state_required`/`appliance_states_required`/`media_states_required` |
| Does the error key have a translation in **both** `strings.json` and `translations/en.json`? | These two files can drift — as of 2026-07-06, `translations/en.json` has a `person_already_configured` string that **`strings.json` is missing entirely** (re-check: `grep -n person_already_configured custom_components/area_occupancy/strings.json custom_components/area_occupancy/translations/en.json`). If you add a new config-flow error key, add it to both files or users see the raw key instead of a message, and `hassfest` (in `validate.yml`) may flag the mismatch |
| Is the failing field a numeric range check? | Weight fields use `WEIGHT_MIN`/`WEIGHT_MAX`; decay half-life uses the `0`-or-`[10,3600]` rule from Symptom 5 above |

Fix pattern: reproduce the exact flow step, read the `errors["base"]` or `errors[<field>]` key from the failed `FlowResult`, then grep that literal string in `config_flow.py` to find the guard clause — the error keys are stable strings, not translated at the Python layer.

Verified: `grep -n 'errors\[.*\] = "' custom_components/area_occupancy/config_flow.py` (full list of ~20 distinct error keys); `translations/en.json` vs `strings.json` diff for `person_already_configured`.

### 8. Entities unavailable / occupancy wrong immediately after restart

| First checks | Discriminating experiment |
|---|---|
| Was `_reconcile_entity_state()` actually called during `coordinator.setup()`? | It should run once right after `db.load_data()` and correlation refresh, before the first analysis cycle — verify with `grep -n "_reconcile_entity_state" coordinator.py` |
| Are rooms showing near-100% occupied right after an upgrade/reload with no one home? | This is exactly issue #379's symptom, fixed by PR #386. Root cause: decay/evidence state is persisted to the DB on shutdown and restored **verbatim** on reload; if a sensor was active before shutdown, `previous_evidence=True` gets restored without comparing against the sensor's *current* (now possibly inactive) HA state, and combined with a high learned prior this drives probability toward 100% |
| Does the fix still hold for a regression you're investigating? | `_reconcile_entity_state()` does two things: (1) ticks every area's decay immediately (resolves anything that expired while unloaded) (2) calls `entity.has_new_evidence()` on every entity to reconcile stale `previous_evidence` against live HA state, which also correctly starts/stops decay based on reality |

Fix pattern: if you see this symptom recur, first confirm `_reconcile_entity_state()` still runs unconditionally early in `setup()` (not gated behind a flag that could skip it) — this was a fast-startup-mode-vs-correctness tradeoff once before.

Verified: `custom_components/area_occupancy/coordinator.py` lines 368-383 (`_reconcile_entity_state` docstring and body) and its call sites at lines ~433 and ~985; `gh issue view 386` (merged, closes #379).

---

## Burned lesson #1: timezone / DST bugs

**The rule (current, correct, and load-bearing):**

```
Runtime arithmetic/comparisons:  timezone-aware UTC
Database persistence (SQLite):   naive UTC (tzinfo=None, interpreted as UTC)
Wall-clock bucketing (time priors, daily/weekly grouping): Home Assistant local timezone
```

This is a doc comment at the top of `time_utils.py`, and it exists because getting this wrong cost real time twice:

1. **Naive-vs-aware `TypeError`** (issues #444/#445, April 2026): `dt_util.utcnow()` returns tz-aware; `entity.last_updated` restored from storage could be tz-naive. Subtracting them raised `TypeError: can't subtract offset-naive and offset-aware datetimes` inside `_check_stuck_sensor`/`_check_unavailable`, causing ~87 health-check failures over ~21 hours on affected installs. Fixed by PR #446 (`dt_util.as_utc(entity.last_updated)` before the subtraction).
2. **DST-unsafe time-of-day bucketing**: the original time-prior calculation risked either double-counting or skipping the repeated/skipped local hour around a DST transition if it bucketed by naive local time directly. Fixed across two PRs: #304 (Dec 2025, introduced `time_utils.py`'s `to_utc`/`to_local` split and bumped `CONF_VERSION` to 16) and #322 (Dec 2025, further UTC-storage refactor, explicitly called out DST handling improvements in its release notes).

**How the current code avoids repeat-hour ambiguity**: `PriorAnalyzer.calculate_time_priors()` walks the analysis period **hour-by-hour in UTC** (not local), converting each UTC instant to local only to derive the bucket key, and threads the local datetime's `fold` attribute through `.replace(...)` when constructing slot boundaries — this is what disambiguates the repeated local hour during a fall-back transition.

**Wrong path people take**: introducing `datetime.now()` or comparing a persisted (possibly naive) datetime directly against `dt_util.utcnow()` without going through `to_utc()`/`from_db_utc()` first. Symptom is either a crash (naive-vs-aware `TypeError`) or a silent one-hour bucketing error that only shows up twice a year.

**Discriminating experiment**: `grep -n "datetime.now()" custom_components/area_occupancy/**/*.py` (should be empty — everything should go through `dt_util.utcnow()`); for any new datetime handling, run it through `time_utils.assert_utc_aware()` in a test to catch a naive value immediately rather than downstream.

**Fix pattern**: never introduce a new datetime read/write path without deciding, explicitly, which of the three policy tiers it belongs to (arithmetic / persistence / bucketing) and routing it through the matching `time_utils.py` helper (`to_utc`, `to_db_utc`, `from_db_utc`, `to_local`).

Verified: `custom_components/area_occupancy/time_utils.py` lines 1-8 (policy docstring), 44-73 (`to_local`, `assert_utc_aware`); `data/analysis.py` lines ~709-721 (hour-by-hour UTC walk, `fold` handling, inline comment "Iterate in UTC to avoid ambiguity during DST fall-back"); commit `4dc22cf` (#446); PR #304 (merged 2025-12-12, bumped `CONF_VERSION` to 16); PR #322 (merged 2025-12-29, "Refactor timezone handling with UTC storage", CodeRabbit summary cites "Better daylight saving time (DST) and timezone handling").

## Burned lesson #2: decay half-life config bugs (three incidents, same mechanism)

The recurring failure shape: **"is this half-life value the purpose default, or did the user customize it?"** gets re-derived from the raw number in more than one place, and every place that does it independently is a fresh chance to get it wrong.

1. **Issue #439 (2026-04-17)**: `Purpose.is_purpose_half_life()` returned `True` whenever the entered value matched **any** purpose's built-in default (12 round values: 45, 60, 90, 180, 240, 360, 450, 480, 520, 600, 620, 1200 seconds) — used by the config flow to decide whether to normalize a value to the `0` ("auto") sentinel on save. So a Living Room user who typed `600` (Office's default, not Living Room's) got silently normalized to `0`, and their custom value reverted to the Living Room default (520s) on next load. **Fixed same day** by PR #440: the comparison is now scoped to compare only against the *currently selected* purpose's own default.
2. **Issue #481 (2026-05-20, SETTLED — fixed by PR #493, merged 2026-07-06)**: the identical bug class recurred in a *different* code path. `Decay.half_life` implements a sleep/awake split for SLEEPING-purpose areas: use the configured half-life during the sleep window, switch to the purpose's `awake_half_life` (620s) outside it. The switch applied **unconditionally** — so a user's custom 10-second half-life got silently replaced by 620s (a ~15 minute clear-out) during every waking hour, with no way to keep a short custom half-life active outside the sleep window. Reported symptom: bedroom that should clear in ~10s instead took ~15 minutes.
3. **PR #493 (merged 2026-07-06)**: fix mirrors #440's pattern exactly — the awake/sleep switch is restricted to only engage when the area's half-life still **equals the purpose's own default** (i.e., the user left it on auto). A custom value now wins at all times of day. Implemented as `Decay._resolve_purpose_half_life()`'s `if self._base_half_life != self._purpose.half_life: return self._base_half_life` guard, with the adjacent-areas `modifier_factor` (from #454, also merged 2026-07-06) multiplying on top of the resolved value in the `half_life` property.

**Wrong path people take**: fixing the config-flow-side normalization (#440) and assuming the bug class is closed. It reappeared in the *runtime* decay-resolution code (`data/decay.py`), a completely separate file, over a month later — because both places independently ask "is this a default or a custom value?" by comparing against known defaults, rather than carrying an explicit "this was customized" flag through the pipeline.

**Discriminating experiment**: for a SLEEPING-purpose area, set a custom half-life that is *not* 1200s (the purpose default) or 620s (`awake_half_life`), then check the diagnostics `config.decay` value both inside and outside the configured sleep window — it should read the same custom value in both cases (confirm this now holds post-#493). Also check any *other* purpose combined with a numerically coincidental custom value (e.g. Living Room + 600s, Office's default) to catch a #439-style regression.

**Fix pattern**: any code that decides "is this the default or a custom override" for a purpose-scoped setting must scope the comparison to the *specific selected* purpose only — never match against the full list of all purposes' defaults. If you add a new purpose-scoped setting with a similar auto/custom split, write a test for exactly this coincidental-value case before shipping.

Verified: `custom_components/area_occupancy/data/purpose.py` (`is_purpose_half_life` docstring explicitly cites #439 and describes the legacy bug); `custom_components/area_occupancy/data/decay.py` lines ~81-122 (`_resolve_purpose_half_life`, the `!= self._purpose.half_life` guard now gates the awake/sleep switch — confirmed present and fixed on `main` as of 2026-07-06, HEAD `17b71d2`); `config_flow.py` ~line 1539 (`Purpose.is_purpose_half_life(user_set_decay, purpose)`, comment citing #439); `gh pr view 440`, `gh pr view 493` (both state MERGED, #493 mergedAt 2026-07-06, body: "mirrors the custom-vs-default semantics established for #440").

## Burned lesson #3: prior pinned at 0.99 (#483 quiet-tail denominator) — SETTLED, fixed by PR #491, merged 2026-07-06

**Root cause**: `PriorAnalyzer.calculate_and_update_prior()` computes `global_prior = occupied_duration / actual_period_duration`. The code used to set `actual_period_end = last_interval_end` (instead of `now`) whenever the area had been quiet for **more than 3600 seconds (1 hour)** before the analysis run. This truncated the denominator's "known unoccupied" tail while the numerator kept counting all historically occupied seconds — every hourly analysis cycle that ran during an overnight or weekend quiet stretch pushed `global_prior` a little higher, and repeated cycles compounded the effect until it saturated at the `MAX_PRIOR = 0.99` clamp.

**Manifestation** (from the reporter, @mscharwere): a kitchen mmWave sensor with a true occupancy rate of ~28-35% had its learned `global_prior` pinned at 0.99 — nowhere near the actual rate.

**Wrong path people take**: treating a 0.99-pinned prior as a sensor/weight/threshold misconfiguration and tuning those instead. The actual defect was in the *learning* code's period-window math, not in anything sensor-facing — nothing in the entity config was wrong.

**Discriminating experiment**: check whether the area has had any extended quiet stretch (no occupied intervals for >1h) before recent analysis cycles, and whether `global_prior` has been monotonically climbing cycle over cycle rather than settling. Diagnostics' `prior.global_prior` field plus manual arithmetic (`occupied_duration_seconds / total_period_seconds` over the *full*, non-truncated window) will disagree with the stored value if this bug is still active anywhere — but on current `main` it should no longer reproduce at all.

**Status as of 2026-07-06**: fixed on `main` (HEAD `17b71d2`). `data/analysis.py` now sets `actual_period_end = now` unconditionally — the old `if (now - last_interval_end).total_seconds() > 3600: actual_period_end = last_interval_end` truncation branch is gone. PR #491 merged 2026-07-06; the original branch's stated purpose (guarding a degenerate just-after-startup period) is covered instead by a separate `actual_period_duration <= 0` guard.

**Lineage**: this is not the first time prior calculation has been reworked — PRs #246 ("Refactor time-prior calculation to use Python instead of SQL"), #251 ("Simplify prior calculation to motion/presence sensors only"), #266 ("Restructure prior calculation flow with immediate database persistence"), and #356 ("Improve prior calculation accuracy") all touched this same area before #483/#491. Treat `data/analysis.py`'s prior-calculation code as chronically fragile — favor small, well-tested changes with an explicit before/after numeric example in the PR description over broad refactors, even now that this particular incident is closed.

**Fix pattern**: any time you touch the observation-period bounds in prior calculation, write a test with a concrete numeric scenario (known occupied seconds, known total seconds, expected prior) — the bug here was caught by exactly the kind of test (`test_valid_calculation_sets_correct_prior`) that had previously encoded the *buggy* expected value (asserting 0.99 where 0.25 was correct) and had to be corrected alongside the fix.

Verified: `custom_components/area_occupancy/data/analysis.py` lines ~516-518 (`actual_period_end = now`, unconditional, truncation branch removed); `gh issue view 483` (closed); `gh pr view 491` (state MERGED, mergedAt 2026-07-06, body confirms root cause and fix); `gh pr view 246`, `gh pr view 251`, `gh pr view 266`, `gh pr view 356` (all MERGED, all prior-calculation titles).

---

## Provenance and maintenance

Date-stamped 2026-07-06 (post-merge sweep), integration version still 2026.5.17 — none of the 2026-07-06 merge wave (#454, #486, #488, #491-496, etc.) has shipped in a tagged release yet (`git log -1 --oneline` on `main` → `17b71d2 feat: adjacent-areas — learned next-door room influence (#454)`). All line numbers and constant values in this file were read directly from `main` via `git show main:<path>`; the working tree is now checked out to `main` itself (the former `feat/adjacent-areas` branch merged and is gone), so working-tree and `main` facts are identical as of this sweep.

Re-verification commands by volatile fact category:

- **Clamp/threshold constants** (`MIN/MAX_PROBABILITY`, `MIN/MAX_PRIOR`, `TIME_PRIOR_*_BOUND`, `PRIOR_FLOOR_THRESHOLD_MARGIN`, `RETENTION_DAYS`, `RETENTION_RAW_INTERVALS_DAYS`, `MIN_CORRELATION_SAMPLES`, `DEFAULT_THRESHOLD`, `DEFAULT_HEALTH_ENABLED`, `DEFAULT_MIN_PRIOR_OVERRIDE`): `git show main:custom_components/area_occupancy/const.py | grep -n "MIN_PROBABILITY\|MAX_PROBABILITY\|MIN_PRIOR\|MAX_PRIOR\|RETENTION\|MIN_CORRELATION_SAMPLES\|DEFAULT_THRESHOLD\|HEALTH_ENABLED\|MIN_PRIOR_OVERRIDE"`
- **Purpose half-life table**: `git show main:custom_components/area_occupancy/data/purpose.py | grep -n "_half_life="`
- **Decay half-life sentinel resolution and sleep/awake switch**: `git show main:custom_components/area_occupancy/data/decay.py` and `git show main:custom_components/area_occupancy/data/entity.py | grep -n "half_life"`
- **Health-check thresholds/exemptions/toggle**: `git show main:custom_components/area_occupancy/data/health.py | sed -n '39,120p'`
- **Config-flow error keys and half-life validation range**: `git show main:custom_components/area_occupancy/config_flow.py | grep -n 'errors\[.*\] = "'` and `grep -n "10 or decay_window > 3600" custom_components/area_occupancy/config_flow.py`
- **strings.json / translations drift**: `diff <(grep -o '"[a-z_]*":' custom_components/area_occupancy/strings.json) <(grep -o '"[a-z_]*":' custom_components/area_occupancy/translations/en.json)`
- **Diagnostic-sensor default-disabled list**: `grep -n "set_enabled_default(False)" custom_components/area_occupancy/sensor.py`
- **DB retention docs staleness**: `grep -n "60 days" docs/docs/technical/database-schema.md`
- **Restart reconciliation**: `grep -n "_reconcile_entity_state" custom_components/area_occupancy/coordinator.py`
- **Open/merged status of every cited PR/issue** (#440, #481, #483, #486, #488, #491, #493, #454, #386/#379, #439, #444, #445, #446, #463, #465, #466, #467, #468, #472, #473, #474): `gh pr view <n> --json state,mergedAt,baseRefName` / `gh issue view <n> --json state` — as of this sweep (2026-07-06) #440, #481/#493, #483/#491, #454, #486, #488 are all MERGED/closed; #466 and #467 remain OPEN — re-check before describing anything as shipped in a release (the integration version itself hasn't bumped past 2026.5.17 yet).
- **Integration version / commit**: `git log -1 --oneline` (on `main`; the working tree now tracks `main` directly post-merge — check `git branch --show-current` first if this changes again).
