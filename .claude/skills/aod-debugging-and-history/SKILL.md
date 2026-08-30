---
name: aod-debugging-and-history
description: Use when triaging a live Area Occupancy Detection symptom report — occupancy stuck on/won't clear, occupancy won't turn on, probability pinned at 0.99 or 0.01, wrong or spammy repair issues, a config setting that seems to be "ignored" (especially decay half-life or min_prior_override), the database growing/slow, config-flow errors, or entities showing 100% occupied right after a restart. Also load this before touching timezone/DST datetime handling, decay half-life resolution, or prior/global_prior calculation code — these are the project's three most expensive historical bug classes and this skill has the exact traps. Also use before touching decay half-life, prior/global-prior calculation, sensor-health repairs, sleep-presence detection, recorder/DB write volume, adjacent-areas, timezone/DST-sensitive datetime code, or the config-flow advanced-options gating — to check whether the bug you're about to fix (or the fix you're about to write) already happened before (root-cause investigation, "why did X happen historically", stale branches). Load this when a symptom rhymes with "custom value silently reverted to a default", "repairs fire spuriously / return after Ignore", "prior pinned near 0.99 or 0.01", "TypeError offset-naive and offset-aware", or "recorder database growing too fast". Also covers the executable, decision-gated campaign for making learned priors and likelihoods (global prior, 168 time-priors, per-sensor P(E|H) from correlation analysis, decay half-life resolution) trustworthy on real homes — load this when a user reports "prior stuck at 0.99/0.01", "occupancy probability doesn't match reality", "false transitions", "decay clears too fast/slow", "correlation/likelihood looks wrong", or when asked to touch data/prior.py, data/analysis.py's PriorAnalyzer, db/correlation.py, or data/decay.py half-life resolution (prior/likelihood accuracy work). This is the flagship operation order for the project's hardest live problem — treat that part as a runbook, not background reading.
---

# AOD Debugging and History

## What this covers

A symptom-to-fix runbook for Area Occupancy Detection's real, recurring failure modes — the ones that have burned real debugging time across this project's history. Each symptom in the triage table below gives first-checks, a discriminating experiment (an exact query/log/diagnostics field to look at, not a guess), and the fix pattern. Three specific bug classes get their own full incident narrative (in the Failure Archaeology section below) because they recurred multiple times: timezone/DST handling, decay half-life config normalization, and global-prior inflation.

Beyond live triage, this skill is also a chronicle of every major bug investigation in this repo that is settled, partially settled, or still open, recorded as symptom → root cause → evidence → status → lesson. The goal is that a future zero-context session (human or AI) checks this file *before* re-diagnosing a bug from scratch, and before proposing a fix that was already tried and found insufficient. Every SAGA entry is verified directly against git history and the GitHub issue/PR tracker (see each entry's Evidence line); dates and PR numbers are exact as of 2026-07-06.

Finally, it contains the end-to-end loop for diagnosing and fixing bad learned values (priors/likelihoods) on a **real** Home Assistant install: pull ground truth, hand-verify the stored math, diagnose which stage is wrong (interval detection, denominator/period selection, bucketing, likelihood estimation, or decay half-life resolution), pick a fix from a theory-obligated menu, and validate before it ever reaches `main`. It exists because this project has shipped the same class of bug — a denominator or comparison subtly excluding data it shouldn't — at least five times (see the Learning-Accuracy Campaign's Phase 2 and Provenance sections).

## When NOT to use this

- You need the Bayesian formula itself (sigmoid/logit pipeline, exact constants, clamping) rather than "why is my number wrong" → `aod-math-reference`.
- You want the *current*, authoritative semantics of decay/prior/health calculations (not their history) → `aod-math-reference` (calculation) or read `data/decay.py`, `data/prior.py`, `data/health.py` directly.
- For the underlying formulas themselves (sigmoid/logit math, clamps, `combine_priors` derivation) → `aod-math-reference`.
- You're deciding whether a proposed code change is safe to ship / needs a migration / could break configs → `aod-change-and-validation`.
- You're deciding whether a fix is safe to ship (silent-math-change / config-break rules) → `aod-change-and-validation`.
- For the actual PR/merge process once a fix is validated → `aod-change-and-validation` (this skill never merges anything itself).
- You need the full config-surface reference (every `CONF_*` key, defaults, what's user-facing vs internal) rather than "this one setting looks ignored" → `aod-architecture-and-config`.
- You need diagnostics/tooling mechanics (how the simulator works, `visualize_distributions.py` internals, diagnostics JSON schema in full) beyond what's quoted here → `aod-diagnostics-and-tooling`.
- For pulling a diagnostics export or DB copy off a real install → `aod-diagnostics-and-tooling` (the Learning-Accuracy Campaign section below assumes you already have both, see its Phase 0).
- For the adjacent-areas / transition-learning subsystem specifically (merged 2026-07-06, PR #454) — it consumes learned priors but doesn't produce them, so it's out of scope for the Learning-Accuracy Campaign section below.

---

## Symptom → Triage Table

### Debug logging recipe

Add to `configuration.yaml`, then restart Home Assistant:

```yaml
logger:
  logs:
    custom_components.area_occupancy: debug
```

Logs land in **Settings → System → Logs** and, in the devcontainer, `config/home-assistant.log`. This is step 3 of the project's own debugging order — do this *last*, not first (see next section).

Verified: `docs/docs/technical/debug.md` lines 11-20; `config/configuration.yaml` already ships this block for the dev sensor rig.

### Order of operations: diagnostics → repairs → logs

This is the project's own documented convention, not something inferred — follow it in this order:

1. **Download diagnostics first.** Integration card (Settings → Devices & Services → Area Occupancy Detection) → **⋮** menu → **Download diagnostics**. No config change needed; captures every prior/weight/evidence/decay/correlation/health value in one JSON file.
2. **Check Settings → System → Repairs** for `sensor_health_*` / `pipeline_health_*` issues. A stuck/unavailable sensor or stale cache is a common root cause and is surfaced automatically.
3. **Only then** enable debug logging and reproduce live.

Verified: `docs/docs/technical/debug.md` lines 1-9.

#### Reading the diagnostics export

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

### 1. Occupancy stuck on / won't clear

| First checks | Discriminating experiment |
|---|---|
| Is `decay.enabled` true for this area? | Diagnostics `areas[].config.decay.enabled` |
| What half-life is actually in effect? Is `decay_half_life` the `0` sentinel ("use purpose default")? | Diagnostics `entities[].decay` vs the area's `purpose` default (see Purpose table below) — if stored value is `0`, the *effective* half-life is `get_default_decay_half_life(purpose)`, not literally instant |
| Is this a Wasp-in-Box or Sleep-presence entity? Both bypass purpose/sleep semantics and have their own fixed half-life | Wasp entities: `half_life = 0.1s` (should clear in well under a second — if one looks stuck, the wasp door/motion wiring is wrong, not decay). Sleep-presence virtual entities: `SLEEP_PRESENCE_HALF_LIFE = 7200s` (2h) is *intentional* persistent presence, not a bug |
| Bedroom (SLEEPING purpose) outside the configured sleep window: is a **custom** half-life being silently replaced by the purpose's `awake_half_life` (620s)? | See § SAGA 3 below (issue #481, fixed by PR #493, merged 2026-07-06) — compare the diagnostics `config.decay` stored half-life against `Purpose.awake_half_life`; if they still differ and the room still takes ~620s (10 min) to clear, the fix has regressed |
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
| Is `0.99` from **global-prior denominator inflation** during a long quiet stretch? | See § SAGA 4 below (issue #483, fixed by PR #491, merged 2026-07-06) — `actual_period_end` is now always `now`, so this should no longer reproduce; if `global_prior` still keeps climbing every hourly recalculation during quiet periods, the fix has regressed |

Fix pattern: don't chase weight/threshold config first — pinned values almost always trace to the prior-calculation denominator or a fallback path, both in `data/analysis.py`, not to sensor configuration.

Verified: `const.py` lines 170-176 (`MIN/MAX_PROBABILITY`, `MIN/MAX_PRIOR`); `data/analysis.py` (`calculate_and_update_prior`, fallback `set_global_prior(0.01)` call sites).

### 4. Repair issues that are wrong (false positives / spam / won't stay dismissed)

| First checks | Discriminating experiment |
|---|---|
| Is the integration-level kill switch off? | `CONF_HEALTH_ENABLED` (`const.py:93`), default `True` (`const.py:138`) — if `False`, `area.health_monitor.clear_all_issues()` runs instead of checks (added PR #472 to close issue #463's "40+ issues every morning") |
| Is the stuck-active threshold purpose-appropriate? | Check the effective threshold = base × purpose multiplier (e.g. motion in a bedroom: 8h × 6 = 48h). The canonical threshold/multiplier/exemption table lives in `aod-diagnostics-and-tooling` §2. Historical context: motion's base was 2h before PR #474, which caused the original false-positive wave (issues #465, #468); see § SAGA 2 below for the full campaign |
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
| Is this `decay_half_life`? Its `0` sentinel means **"use purpose default"** — a stored `0` is not literally a zero-second half-life | Diagnostics `config.decay` shows the *stored* value; if it's `0`, cross-reference `get_default_decay_half_life(purpose)` (or the Purpose table below) for the value actually in effect. See § SAGA 3 below for the exact historical bug in *how* values get normalized to this sentinel |
| Is this `min_prior_override`? Its `0.0` sentinel means **"disabled"**, a *different* semantic from decay's "auto" — don't conflate the two 0-sentinels | `const.py:134` `DEFAULT_MIN_PRIOR_OVERRIDE = 0.0  # 0.0 = disabled by default` |
| Did the config-flow silently round-trip a custom value to the sentinel on save? | Check `Purpose.is_purpose_half_life(value, purpose)` — as of 2026-07-06 this is correctly scoped to compare **only** against the *selected* purpose's own default (fixed by #440; the legacy bug matched against *any* purpose's default, see § SAGA 3 below) |
| Is the value out of the accepted config-flow range? | Decay half-life config-flow validation accepts `0` (auto) or `10 ≤ value ≤ 3600`; anything else raises `errors[CONF_DECAY_HALF_LIFE] = "invalid_decay_half_life"` — note this range **cannot literally express** the SLEEPING purpose's own default of 1200s except via the `0` sentinel, which is expected, not a bug |

Fix pattern: for any "my custom X is being ignored" report, first determine whether X uses a magic-sentinel-plus-lookup pattern (decay half-life does; most other settings don't) before assuming a general config-flow bug — this specific pattern is the one with the incident history.

Verified: `custom_components/area_occupancy/data/purpose.py` (`is_purpose_half_life`, docstring cites #439); `config_flow.py` (`_apply_purpose_based_decay_default`, ~line 1539, validation ~lines 1966-1971); `const.py:134`.

### 6. Database growing / slow

| First checks | Discriminating experiment |
|---|---|
| Which retention constant applies? Two different numbers exist and the docs site states a third (wrong) one | `RETENTION_DAYS = 365` (`const.py:238`) is what actually hard-deletes raw interval rows (`db/operations.py::prune_old_intervals`). `RETENTION_RAW_INTERVALS_DAYS = 28` (`const.py:248`) only controls when raw intervals get rolled into daily aggregates (`db/aggregation.py`) — it does not delete anything. **`docs/docs/technical/database-schema.md` claims "Raw intervals: 60 days" — this is stale/wrong; trust the constants, not that doc** (re-check: `grep -n "60 days" docs/docs/technical/database-schema.md`) |
| Are diagnostic sensors writing to the recorder unnecessarily? | 7 sensor classes (`PriorsSensor`, `EvidenceSensor`, `DecaySensor`, `PresenceProbabilitySensor`, `EnvironmentalConfidenceSensor`, `ActivityConfidenceSensor`, `SensorHealthSensor`) are disabled-by-default **for newly registered areas only** (PR #488, merged 2026-07-06) — existing installs' enabled/disabled state is preserved across upgrades (HA's `async_get_or_create` on an existing registry entry can't touch `disabled_by`). A full delete+re-add of an area *does* count as fresh registration and comes back with diagnostics disabled |
| Is state precision inflating row count? | `CONF_SENSOR_PRECISION` (default `DEFAULT_SENSOR_PRECISION = ROUNDING_PRECISION = 2` decimal places) — lowering to 0 (whole-percent) measurably cuts recorder rows (PR #486, merged 2026-07-06, measured 55-79% fewer rows in the reporting install; see § SAGA 5 below for the measured table) |
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

Three of the above symptoms (timezone/DST, decay half-life config, and the 0.99-pinned prior) trace to bug classes that recurred multiple times across the project's history — their full incident narratives (root cause, evidence, blow-by-blow, lineage) live in § SAGA 1, § SAGA 3, and § SAGA 4 of the Failure Archaeology section below, not repeated here.

### Triage table provenance and maintenance

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

---

## Failure Archaeology: SAGA Narratives

### Status legend

- **SETTLED** — merged to `main`, no known recurrence, root cause fully understood.
- **PARTIALLY-SETTLED** — merged fix helped, but the underlying bug class or a related issue remains open.
- **OPEN** — PR exists but is not yet merged to `main` as of 2026-07-06 (verify current state with `gh pr view <n>`), or no fix has been proposed yet.

Do not describe anything marked OPEN below as shipped. Re-check merge state before relying on it: `gh pr view <n> --json state,mergedAt,baseRefName`.

---

### SAGA 1 — Timezone / DST datetime bugs (maintainer-flagged costliest failure class)

**Symptom (round 1, Dec 2025):** Issue #301 "Invalid period duration errors in 2025.12.2" — prior calculation logged `Invalid period duration (-10800.00 seconds) for area Hallway ... Using safe fallback prior of 0.01`, repeatedly. Exactly −3h (10800s) is the giveaway: a timezone-offset-sized negative duration, not random clock skew.

**Root cause:** Datetime handling before PR #304 mixed naive and aware datetimes, and DST-window arithmetic used wall-clock comparisons that broke across the fall-back/spring-forward transition, producing an "end before start" period.

**Fix:** PR #304 "Implement timezone normalization and local bucketing utilities" (merged 2025-12-12) introduced `time_utils.py` and bumped `CONF_VERSION` to 16. PR #322 "Refactor timezone handling with UTC storage" (merged 2025-12-29) followed up with DST-aware time calculations and UTC-based storage.

**Current mechanism (verified, `custom_components/area_occupancy/time_utils.py:1-8`):** an explicit three-tier policy stated in the module docstring — runtime arithmetic uses timezone-aware UTC; SQLite persistence uses naive UTC (`tzinfo=None`, interpreted as UTC); wall-clock bucketing (time priors, daily/weekly grouping) uses HA's local timezone via `dt_util.as_local()`. Helpers: `to_utc()`, `to_db_utc()`, `from_db_utc()`, `to_local()`, `assert_utc_aware()` (debug-only invariant check).

**Symptom (round 2, April 2026 — the bug class recurred in a *different* module):** Issues #444/#445, "TypeError: can't subtract offset-naive and offset-aware datetimes in `_check_stuck_sensor`" — `data/health.py` line 257, ~87–116 failures over 21 hours on real installs. `now = dt_util.utcnow()` (aware) minus `entity.last_updated` (naive when restored from storage) raised `TypeError`, failing `sensor_health_check` on every analysis cycle.

**Fix:** PR #446 "Fix TypeError on naive vs aware datetime in health checks" (merged 2026-04-27) normalizes via `dt_util.as_utc(entity.last_updated)` in both `_check_stuck_sensor` and `_check_unavailable`, plus normalizes on entity restore from DB.

**Status: PARTIALLY-SETTLED.** The `time_utils.py` policy (SAGA 1 round 1) is the settled, load-bearing fix for the core prior/analysis pipeline. But round 2 shows the naive/aware mismatch class recurred in a module (`health.py`) that predates or bypassed the `time_utils.py` convention — meaning any *new* code that touches `entity.last_updated`, `entity.last_changed`, or any datetime pulled from HA state/registry objects must not assume it is timezone-aware.

**The lesson:** Never subtract or compare two datetimes without first normalizing both through `time_utils.to_utc()` (or `dt_util.as_utc()` for HA registry/state objects specifically) — HA's own state machine and the storage layer both silently hand you naive datetimes in some code paths. Grep for raw `datetime.now()`, `entity.last_updated`, or bare subtraction of two datetime variables in any new PR touching health checks, decay, or analysis; this bug class has bitten this repo twice in two different modules seven months apart. DST transitions specifically break *wall-clock window* comparisons (e.g. "is current local time within sleep_start–sleep_end") — see SAGA 3, which is a different but adjacent bug in the same neighborhood of code.

Evidence: issue #301 (closed 2025-12-29); PR #304 (merged 2025-12-12, `time_utils.py` introduced, CONF_VERSION→16); PR #322 (merged 2025-12-29); issues #444, #445 (closed 2026-04-27); PR #446 (merged 2026-04-27); `custom_components/area_occupancy/time_utils.py:1-72` (current, read directly).

---

### SAGA 2 — Sensor-health false-positive campaign (#429 → #455 → #459 → #466 → #472 → #473 → #474 → #485)

**Origin:** PR #429 "Add sensor health monitoring with HA repairs integration" (merged 2026-03-31) introduced `HealthMonitor`: per-sensor-type stuck-active/unavailable/never-triggered detection surfaced through HA's Repairs UI. Motivated by real production findings (a motion sensor stuck "on" 25h, appliance sensors that never triggered, sensors permanently unavailable). Initial thresholds: motion 2h active before flagging, door 48h, appliance 28 days inactive.

**Symptom wave 1 (issue #455, 2026-05-03):** "Repeatedly generates Repair notifications... disappear when I reload or click Ignore, return again after a few minutes." Log showed `New sensor health issues in area 'X': None (correlation_failures)` for every area simultaneously.

**Root cause (3 bugs in one issue):**
1. `translations/en.json` was missing the entire `issues` block, so every repair rendered as a title-only card with no guidance.
2. `pipeline_health_correlation_failures` fired during the warm-up grace period — soft "not enough data yet" states were treated as real failures.
3. `sensor_health_unavailable` measured duration from the persisted `entity.last_updated` (can be days old) instead of a live "since observed unavailable" clock — so a slow-loading source integration (Z2M, ESPHome) at HA startup instantly tripped every sensor past the 1h threshold.

**Fix:** PR #459 "fix(health): silence false-positive repairs from #455" (merged 2026-05-04, same day as opened) — mirrored the translations block, gated correlation-failure checks on `PRIORS_TRAINING_GRACE_PERIOD`, and added an in-memory `_unavailable_since` map keyed by entity_id instead of trusting `last_updated`.

**Abandoned parallel branch — `origin/hotfix-repairs-455`:** a stale branch, 8 commits ahead / 6 behind `main` (verified: `git rev-list --left-right --count origin/main...origin/hotfix-repairs-455`). Its first 3 commits (`180a93a`, `6dbe9ab`, `9c4972d`) closely parallel what shipped in #459. Its last 3 commits (`a0d4d86`, `d5cb762`, `2b0da16`, "anchor `db_aggregation` tests to local midnight instead of UTC midnight") were **dropped entirely** — #459 as merged (squashed as `368c03a`) does not contain that test-anchoring work. If you find this branch, do not assume it's a superset of `main`'s fix; it's a divergent draft, partially superseded.

**Symptom wave 2 (issues #463, #465, #466, #468, all opened 2026-05-05 to 2026-05-10):** users "overwhelmed by repair noise" — one reported "I wake up every morning to 40+ issues." Two concrete complaints: (a) a TV `media_player` going `unavailable` every power-off triggers a repair every morning; (b) bedroom mmWave motion sensors legitimately stay "on" for hours during sleep, tripping the 2h stuck-active threshold nightly. Also: dismissing ("Ignore") a repair didn't stick — it returned on the next condition recurrence.

**Fix (three independent, deliberately-decoupled PRs, all merged 2026-05-16/17):**
- PR #472 "add integration-level toggle to disable repair monitoring" — `CONF_HEALTH_ENABLED` global boolean (default **on**); when off, both sensor- and pipeline-scope checks short-circuit and `HealthMonitor.clear_all_issues()` empties the Repairs UI (distinct from `cleanup()`, which additionally wipes the `_unavailable_since` clock — `clear_all_issues()` deliberately preserves it so re-enabling doesn't cause an instant re-trip).
- PR #473 "preserve user-ignored repairs across condition flaps" — root cause was that `_update_repair_issues` called `ir.async_delete_issue()` on every condition-clear, which also wiped HA's `is_ignored` flag; the fix partitions resolved issues into truly-resolved (delete) vs. user-ignored (skip deletion, keep tracked so a recurrence isn't logged as "new").
- PR #474 "purpose-aware stuck-active thresholds and saner defaults" — motion default raised 2h → **8h** (base); per-purpose multiplier table `SLEEPING×6` (=48h), `RELAXING×4` (=32h), `WORKING×3` (=24h); `media_player.*` entity-id prefix exempted from the unavailable check entirely.

**Current thresholds:** see the canonical table in `aod-diagnostics-and-tooling` §2 (base thresholds, purpose multipliers, `media_player.*` exemption, `InputType.SLEEP` exclusion) — kept in one home so a future threshold change edits one file.

**Status: PARTIALLY-SETTLED.** All of #463/#465/#466/#468's reported symptoms are addressed. Two follow-on asks remain **OPEN**:
- **#466** — per-sensor suppression ("mark this specific entity as expected, never alert on it") — deliberately deferred; the PR #472/#474 authors state explicitly this is out of scope for the toggle-and-defaults approach.
- **#485** — vacation-aware alert suppression (a boolean indicating "long absence is expected here," so stuck/no-trigger checks don't fire) — open, no PR yet, reporter notes the irony that during vacation, a sensor *triggering* is more suspicious than one staying silent.

**The lesson:** Health-check false positives are a *defaults and UX* problem more than a *detection-logic* problem — three of the four shipped fixes (#459's unavailable clock, #472's toggle, #473's sticky-ignore, #474's thresholds) all worked around the same underlying tension: a single global timeout can't be right for every sensor type, purpose, and household schedule simultaneously. When you get a new "false positive repair" report, first check whether it fits an existing per-purpose/per-type carve-out before adding a new global threshold — and check #466/#485 before designing a new per-sensor mechanism, since both are explicitly requested and unclaimed.

Evidence: PR #429 (merged 2026-03-31); issue #455 (closed 2026-05-04); PR #459 (merged 2026-05-04, commit `368c03a`); `origin/hotfix-repairs-455` branch (`git log origin/main..origin/hotfix-repairs-455`, 6 commits, last 3 dropped); issues #463, #465, #466 (open), #468; PRs #472, #473, #474 (all merged 2026-05-16/17); `custom_components/area_occupancy/data/health.py:40-80` (current, read directly); issue #485 (open).

---

### SAGA 3 — Decay half-life "custom value silently overridden by default" (two incidents, same bug class, different code path)

**Incident 1 — issue #439 (2026-04-17), reported via discussion #433:** custom decay half-life appeared to save in the options flow but reverted to the previous value on reopen.

**Root cause:** `Purpose.is_purpose_half_life(value)` returned `True` whenever the entered value matched *any* purpose's built-in default (12 round values: 45, 60, 90, 180, 240, 360, 450, 480, 520, 600, 620, 1200 seconds) — not just the *currently selected* purpose's default. So e.g. a Living Room user (purpose Social, default 520s) entering `600` (which happens to equal the Office purpose's default) got silently normalized to `0` ("use purpose default"), discarding their intended custom value.

**Fix:** PR #440 (merged same day, 2026-04-17) scopes the comparison to only the selected purpose's own default: `is_purpose_half_life(value, purpose=None)`.

**Status: SETTLED** for incident 1's exact code path.

**Incident 2 — issue #481 (2026-05-20), reported ~1 month later:** "Custom Decay Half-Life is ignored when purpose is Bedroom" — a user set purpose=Bedroom, custom half-life=10s; the room took ~15 minutes (900s+) to clear instead of ~10s, but only *outside* the configured sleep window. Changing purpose away from Bedroom fixed it immediately.

**Root cause:** `Decay._resolve_purpose_half_life()` (in `data/decay.py`) implements a sleep/awake split unique to `AreaPurpose.SLEEPING`: inside the sleep window it correctly uses the area's configured half-life; outside it, the code **unconditionally** returns `self._purpose.awake_half_life` (620s), discarding whatever custom half-life the user set — the same "custom-vs-default" semantics bug as #439, recurring in a sibling code path that #440's fix never touched.

**Fix — PR #493 "respect custom half-life for Bedroom purpose" (merged 2026-07-06, commit `55a0aae`):** adds the guard `if self._base_half_life != self._purpose.half_life: return self._base_half_life` — any half-life differing from the purpose default is treated as a deliberate user override and the awake/asleep switch is skipped entirely; the switch only engages when the half-life equals the purpose default (i.e., the user left it on auto), explicitly modeled on #440's fix pattern. Verified directly on `main` (`custom_components/area_occupancy/data/decay.py:90-91`): the guard is a single added `if` before the sleep-window check, exactly as the PR diff showed pre-merge.

**Status: SETTLED** (merged 2026-07-06; guard confirmed present on `main`).

**The lesson:** "Custom value vs. purpose default" is a recurring semantic distinction in this codebase that has now broken twice in two different functions seven months apart, both times because a piece of code treated "value equals a known default" as proof of "value was never customized," instead of tracking customization as its own explicit signal. Any new purpose-aware or type-aware default-switching logic (decay, thresholds, weights) should be checked against this exact failure mode before shipping: does the switch key off "does this look like a default" (fragile, re-breaks) or off "did the user actually configure something else" (correct)? #440 and #493 both had to retrofit the latter after finding the former shipped first.

Evidence: issue #439 (closed 2026-04-17); PR #440 (merged 2026-04-17); issue #481 (closed 2026-07-06); PR #493 (merged 2026-07-06, commit `55a0aae`, body: "mirrors the custom-vs-default semantics established for #440"); `custom_components/area_occupancy/data/decay.py:81-91` on `main` (guard now present, read directly).

---

### SAGA 4 — Global prior accuracy: the quiet-tail denominator bug, and the test that encoded it (#483 → #491)

**Symptom (issue #483, 2026-05-29):** A kitchen area with a single mmWave sensor had `global_prior` pinned at the hard cap **0.99**, despite a true occupancy rate of ~28–35% measured over 7 days of recorder history.

**Root cause (found by the reporter, @mscharwere, and confirmed by direct code read):** `PriorAnalyzer.calculate_and_update_prior()` in `data/analysis.py` contained:
```python
if (now - last_interval_end).total_seconds() > 3600:
    actual_period_end = last_interval_end   # drops the quiet tail from the denominator
else:
    actual_period_end = now
```
The comment's intent ("use it if very recent") was meant to guard against a near-zero denominator right after startup, but the condition actually fires **every time the area has been quiet for more than an hour** — overnight, every weekend, any extended absence. Effect: `global_prior = occupied_duration / (period − quiet_tail)` instead of `occupied_duration / period`, and because the prior recalculates hourly, every quiet stretch re-inflates it, ratcheting the prior toward the 0.99 cap with no correcting force.

**The test that encoded the bug:** `tests/test_data_analysis.py::TestPriorAnalyzerCalculateAndUpdatePrior::test_valid_calculation_sets_correct_prior` asserted, for a scenario of "2h occupied, ending 6h ago" (an 8h total window): `assert area.prior.global_prior == 0.99`. The correct value for that scenario is `2h / 8h = 0.25`. The test wasn't testing correctness — it was pinned to the bug's own output, so the bug shipped, and any refactor that preserved the buggy behavior would keep passing CI. This is a load-bearing lesson for `aod-change-and-validation` and for anyone reviewing prior-calculation PRs: **a green test suite does not mean the math is right if a test's expected value was itself derived from running the buggy code.**

**Fix — PR #491 "keep quiet tail in global prior denominator" (merged 2026-07-06, commit `3f0895a`):** `actual_period_end` is now unconditionally `now`; the pre-existing `actual_period_duration <= 0` guard already covers the degenerate-denominator case the old conditional was trying to protect. The corrected test now asserts `0.25`, and a new regression test `test_quiet_tail_included_in_denominator` was added for the overnight-quiet scenario specifically. Verified directly on `main` (`custom_components/area_occupancy/data/analysis.py:515-518`): `actual_period_end = now`, unconditional, exactly as the pre-merge PR diff showed.

**Status: SETTLED** (merged 2026-07-06; confirmed live on `main`). Note this bug directly matched the maintainer's stated "costliest past failure": prior pinned at 0.99.

**Context — priors have been reworked repeatedly:** CodeRabbit's auto-linked related-PR history on #491 surfaces PRs #219, #246, #251, #266, #356 as prior related prior-calculation rework — this is at least the 6th significant touch to prior calculation across the project's life. Treat prior/likelihood code as one of the highest-scrutiny surfaces in the repo (see this skill's Learning-Accuracy Campaign section below for the current state of this effort and `aod-change-and-validation`'s "no silent math changes" rule).

**The lesson:** When fixing prior/likelihood math, (1) recompute the expected test value by hand from the stated scenario before trusting an existing test's assertion, especially for edge-case tests around cap/floor values (0.01, 0.99) — a test asserting a boundary value is exactly where a "the bug is the spec" trap hides; (2) any conditional that special-cases "denominator would be small/degenerate" needs its trigger condition audited against *all* the real-world situations that could set it off, not just the one motivating case in the original commit message.

Evidence: issue #483 (closed 2026-07-06, root-cause credit to @mscharwere); PR #491 (merged 2026-07-06, commit `3f0895a`, body cites #483, "Credit to @mscharwere for the precise root-cause analysis"); `custom_components/area_occupancy/data/analysis.py:515-538` on `main` (read directly, confirms the conditional is gone and the corrected test assertion 0.99→0.25 landed).

---

### SAGA 5 — Recorder/DB write-load campaign (#467 → #486 + #488)

**Symptom (issue #467, 2026-05-10):** A user on a 7-day recorder rollover with ~770 recorded entities found AOD responsible for over 30% of total recorder rows (1.5M of ~5M states) — diagnostic sensors updating on the 10-second decay timer with 2-decimal-place states meant nearly every tick wrote a new recorder row per sensor.

**Measured numbers (PR #486, live 6-area install, v2026.5.17, 57 AOD entities — these are the exact figures to cite, do not round differently):**

| Window (3h) | Precision | Recorder rows | Δ vs baseline |
|---|---|---|---|
| Afternoon (active), baseline | 2 decimals | 15,952 | — |
| Evening 21:30–00:30 (going to bed) | 0 decimals | 7,058 | **−55%** |
| Morning (low activity) | 0 decimals | 3,323 | **−79%** |

**Fix 1 — PR #486 "configurable sensor state precision to reduce recorder load" (merged 2026-07-06):** adds a global "Sensor state precision" setting (0–2 decimals, default **2 = unchanged behavior**, so existing installs see no change unless they opt in). At 0 decimals the recorder only writes on whole-percent changes. Deliberately does **not** use HA's `suggested_display_precision` — that only rounds the UI display, not the recorded state, so it wouldn't reduce recorder rows. All decision logic (`area.occupied()`, wasp-in-box, activity scoring, thresholds) continues to operate on internal unrounded floats — this PR touches only publication-layer formatting (`format_float`), not calculation code.

**Fix 2 — PR #488 "disable diagnostic sensors by default for newly added areas" (merged 2026-07-06):** registers 7 diagnostic sensor classes (`PriorsSensor`, `EvidenceSensor`, `DecaySensor`, `PresenceProbabilitySensor`, `EnvironmentalConfidenceSensor`, `ActivityConfidenceSensor`, `SensorHealthSensor`) with `entity_registry_enabled_default = False`. The two primary entities (`ProbabilitySensor`, `DetectedActivitySensor`) and the `Occupancy Status` binary sensor stay enabled. This only affects **newly registered** entities.

**Entity-registry restore subtlety (verified via PR #488's own review comment, self-reviewed with Claude Code assistance):** `entity_registry_enabled_default` only applies at first registration. HA's `async_get_or_create()`, when called against an *already-existing* registry entry, routes to `_async_update_entity()` — which has no `disabled_by` parameter and structurally cannot touch it. This is *why* existing installs are unaffected by this change: not a special-cased migration guard, but a structural property of the HA entity registry API. **Documented edge case: this protection does not extend to delete-and-re-add.** Deleting an area and re-adding it counts as a fresh registration (a new registry entry), so it comes back with diagnostics disabled — the "restore previous state" behavior only applies across reload/upgrade of a *still-registered* entry, never across a full delete-then-recreate cycle. If you are ever debugging "why did my diagnostic sensor's enabled state reset," check whether the area was deleted and re-added (resets) versus just reloaded/upgraded (preserves) before assuming a regression.

**Status: PARTIALLY-SETTLED.** Both PRs landed together (2026-07-06) and are complementary (#486 shrinks rows for *enabled* diagnostics; #488 removes them entirely for new setups) but issue #467 itself remains **OPEN** — the reporter's request was a single global "significant figures" config knob plus attribute pruning/sorting, and neither PR fully closes that ask; they're presented as "primary drivers" addressed, not a complete fix.

**The lesson:** When reasoning about whether a "disabled by default for new X" change is backward compatible, the correct verification is reading the actual HA core function your code calls (`async_get_or_create` → `_async_update_entity` when already registered) rather than asserting compatibility from intent alone — and the one honest gap (delete+re-add doesn't preserve state) should be stated explicitly in the PR, not glossed over, exactly as PR #488 did.

Evidence: issue #467 (open, 2026-05-10); PR #486 (merged 2026-07-06, body contains the measured-numbers table verbatim); PR #488 (merged 2026-07-06, body: "documented & tested" edge case section, quote "async_get_or_create on an existing entity routes to _async_update_entity, which has no disabled_by parameter and structurally cannot touch it").

---

### SAGA 6 — Sleep-presence detection: multi-sensor support, then unknown-presence gating (#375 → #464 → #492)

**Foundation — PR #375 "Add multi-sensor sleep detection support" (merged 2026-02-22):** support for multiple sleep sensors per person, across both `sensor` and `binary_sensor` domains, OR-combined ("any active sensor triggers sleep detection"). Config-version bump 16→17 with backward-compatible migration.

**Symptom (issue #464, 2026-05-07):** A binary sleep sensor (e.g. an `input_boolean`-backed template) reported `state: 'on', active: true` in the sensor's own attribute breakdown, but the top-level `sleeping` field was `false` and occupancy was not detected as a result.

**Root cause (self-diagnosed by reporter @laszlojakab, pinpointing `binary_sensor.py:859-865`):** `_evaluate_sleep_state()` gated on the person entity being home (`home_state.state != STATE_HOME` → treated as away, sleep never checked). A person entity with **no device tracker assigned** reports state `unknown` — and the equality check above treated `unknown` identically to a definitive "away," silently disabling sleep detection for anyone without a device tracker, even though their sleep sensor was correctly reporting active.

**Caution — a plausible-looking but wrong diagnosis was floated for this same issue:** an earlier read of the code (checking commit `368c03a`, unchanged since PR #375) found that a device-tracker/person-entity fallback *already existed* in that exact function, contradicting a claim that the fallback was "missing." The real bug is specifically the *unknown-state handling*, not an absent fallback path — if you're investigating #464-shaped reports, verify against the current code's handling of the `unknown` state specifically, not just "is there a fallback at all."

**Fix — PR #492 "detect sleep when person presence is unknown" (merged 2026-07-06, commit `2099025`):** adds a `_person_home_state()` helper returning `True`/`False`/`None` (tri-state: `None` = indeterminate — `unknown`/`unavailable`/missing entity). `_evaluate_sleep_state()` now skips a person only when **definitively away**; when presence is unknown, sleep sensors are trusted directly. 13 new tests added (the sensor had zero prior test coverage).

**Status: SETTLED** (merged 2026-07-06).

**The lesson:** Home Assistant person/device-tracker entities have (at least) three meaningfully different states — home, away, and unknown — and code written as a two-way boolean check (`== STATE_HOME` or `!= STATE_HOME`) will always silently collapse "unknown" into whichever branch it wasn't explicitly testing for. Any gating logic keyed on a person/tracker entity's state should be audited for this same tri-state collapse before shipping.

Evidence: PR #375 (merged 2026-02-22, CONF_VERSION 16→17); issue #464 (closed 2026-07-06, reporter's own attribute dump showing `active: true` / `sleeping: false`); PR #492 (merged 2026-07-06, commit `2099025`, body: "Fixes #464 ... exactly as @laszlojakab traced in the issue"); `git show 368c03a9:custom_components/area_occupancy/binary_sensor.py` (fallback-already-present check, read directly at the commit cited by a since-superseded diagnosis).

---

### SAGA 7 — `show_advanced_options` deprecation, and a false-blocker verification episode (#487 → #489)

**Symptom (issue #487, 2026-06-09):** HA system log warning on every startup: `The deprecated function show_advanced_options was called from area_occupancy. It will be removed in HA Core 2027.6.`

**Fix — PR #489 (contributor Ecronika; approved and merged by the maintainer 2026-07-06, commit `704c89e`):** removed the `show_advanced: bool = False` parameter from five config-flow schema builders and un-gated the four `if show_advanced:` blocks entirely — because HA's deprecated property already unconditionally returns `True` during its deprecation window, un-gating is a behavior-preserving deletion, not a UX change. Diff limited to `config_flow.py` (−25 lines). Verified live on a 6-area install: warning stopped appearing, advanced fields still shown identically.

**The false-blocker episode (verify current claims with `gh pr view 489 --json reviews`):** PR #489's body and the maintainer's approving review cite a specific source for the deprecation: *"Deprecation of advanced mode in data entry flow,"* developers.home-assistant.io, dated **2026-05-26**. At the time of this review, this repo's `pyproject.toml` pinned `homeassistant==2026.2.2` for its own test suite — a version that predates the cited blog post by three months (that pin has since moved to `homeassistant==2026.7.1` via the same-day dependency refresh, PR #496 — reverify with `grep -n '"homeassistant==' pyproject.toml`). A reviewer or verification pass that reasons "the pinned test dependency doesn't know about this deprecation yet, therefore the citation must be fabricated" **would be wrong**: pinned test dependencies in an actively-developed integration are routinely older than the *production* HA version users are running the integration against, and upstream deprecation announcements apply to the ecosystem going forward, not retroactively to whatever version a repo's CI happens to test on. The correct verification is checking the cited blog post's existence/date and the actual runtime behavior (which PR #489 did — "verified live... deprecation warning no longer appears"), not comparing it against an unrelated pinned test-dependency version.

**Status: SETTLED** (merged 2026-07-06).

**The lesson:** when verifying a claim that cites an external (non-repo) source — a vendor blog post, an upstream deprecation notice, a changelog entry — check the source directly (fetch it, confirm the date and content) rather than using an unrelated repo-internal version pin as a proxy for "could this be true yet." A stale test-dependency pin is extremely common in real projects and is not evidence that a forward-looking claim about the pinned dependency's *future* is false.

Evidence: issue #487 (closed 2026-07-06); PR #489 (merged 2026-07-06, commit `704c89e`; maintainer review body quotes the exact blog post title/date/URL); `pyproject.toml` pinned `homeassistant==2026.2.2` at review time (since bumped to `2026.7.1` at line 25 by PR #496, same-day merge — the pin cited during this episode is no longer what's on `main`).

---

### SAGA 8 — Adjacent-areas: a feature dormant since 2025, then built in five phases, merged 2026-07-06

**Dormant schema (verified: `git show a99ad49:custom_components/area_occupancy/db/schema.py`, commit dated 2025-11-17):** the `Areas.adjacent_areas` JSON column, `AreaRelationships` table, and `CrossAreaStats` table have existed in the schema since November 2025 — over five months before anything used them. No config-flow UI, no producer, no consumer existed until 2026.

**Origin of the ask — discussion #431 (opened 2026-04-11 by jeroen-zzx, still unanswered/unclosed, 0 comments):** requested a "next door room" option (e.g., bedroom → hall) with two hand-tunable confidence parameters ("no motion next door" raises confidence; "motion next door" lowers it, since it could be someone else).

**Implementation — PR #454 "feat: adjacent-areas (next-door room influence)" (created 2026-05-03; merged 2026-07-06, commit `17b71d2`, now `main` HEAD), a 5-phase build:**
1. Plumbing — `CONF_ADJACENT_AREAS` constant.
2. Config flow + persistence — per-area multi-select UI; symmetric write at the persistence layer (if A lists B as adjacent, B is automatically updated to list A); deliberately NO `CONF_VERSION` bump — the new `adjacent_areas` column and `AreaTransitions` table are purely additive, created via `Base.metadata.create_all(checkfirst=True)` on startup (see `aod-architecture-and-config` and `aod-change-and-validation` for this precedent).
3. Transition learning — new `AreaTransitions` table recording 1-hop and 2-hop room-to-room transitions, bucketed by hour-of-week; a 6-level smoothing fallback (`lookup_transition_probability`) walks from most-specific (2-hop, hour-of-week, min. 5 observations) down to a static default (`DEFAULT_INFLUENCE_WEIGHTS["adjacent"] = 0.3`) as data sparsity increases.
4. Bayesian wiring (**folded in from a separate stacked PR, #456**, merged 2026-07-06 into the `feat/adjacent-areas` branch, which itself merged to `main` the same day via #454) — a logit-additive boost in `Area.probability()` (`apply_logit_boost`, confirmed wired into `area/area.py` on `main`) plus a decay-half-life stretch modifier (capped at 1.75×, `Decay.set_modifier_factor`/`compute_decay_modifier` in `coordinator.py`) for areas whose adjacent exits have been quiet.
5. Tests + docs top-up.

**Design decision explicitly locked in (per the PR body), directly answering discussion #431's ask differently than requested:** influence is **learned** from observed transitions (Phase 3), not a hand-tuned static confidence parameter as #431's author suggested. This is a deliberate design choice worth knowing if anyone later asks "why isn't there a simple slider for this" — there is a per-pair influence weight, but the directional strength itself comes from learned transition data, not user-set numbers.

**CodeRabbit review nitpicks — still present on `main` post-merge (verified via `gh api .../pulls/454/reviews` and a direct read of the current file):** (1) `db/relationships.py`'s module docstring still describes the Bayesian/decay consumers as "still uncalled — they're the Phase 4 work" — false as of this merge, since Phase 4 (`apply_logit_boost`, `compute_decay_modifier`) is now wired in and called from `area/area.py` and `coordinator.py`; the docstring was never updated when Phase 4 landed, so it's now a stale doc-comment trap for anyone reading that file in isolation; (2) no regression test existed for `CONF_ADJACENT_AREAS` being a malformed non-list value at the time of review — addressed by `_normalize_adjacent_areas` defensive coercion (`config_flow.py:1614`, confirmed present on `main`).

**Tunables are explicitly unvalidated (verified, `const.py:189-221` on `main`, comment: "First-pass values; tune from real data once Phase 3 is collecting transitions"):** `ADJACENCY_TRANSITION_WINDOW_S=60`, `ADJACENCY_RECENCY_HALF_LIFE_DAYS=30`, `ADJACENCY_TRAJECTORY_WINDOW_S=300`, `ADJACENCY_BOOST_GAIN=0.5`, `ADJACENCY_DECAY_MODIFIER_GAIN=0.75`, `ADJACENCY_DECAY_MODIFIER_MAX=1.75`, and four minimum-observation smoothing thresholds (5/20/50/20). No commit or test exercises these against real recorder data — only synthetic/mocked entities in tests. This remains true after the merge: the feature landing on `main` did not include a real-home validation pass, so treat these constants as an unvalidated candidate default, not a tuned one.

**Status: PARTIALLY-SETTLED.** PR #454 merged to `main` 2026-07-06 (commit `17b71d2`, now `main` HEAD) — the feature (config flow, persistence, transition learning, Bayesian wiring, tests/docs) is complete and live on `main`. It has **not** yet reached a tagged release: the integration's released version is still `2026.5.17`, so this is "shipped to `main`," not "shipped to users," until the next release cut. It also remains **unvalidated on real homes** — the tunables above are first-pass values exercised only against synthetic/mocked test data — so treat adjacent-areas as a functional-but-unproven feature, not a fully settled one, until real-world tuning data comes in.

**The lesson:** this was the single largest long-dormant-then-merged feature in the repo's history (unmerged for five months of active build-out before landing 2026-07-06) and a case study in "features can sit dormant in the schema for months before anyone builds the rest" — if you're investigating what looks like dead/unused schema (columns or tables with no readers), check open PRs and discussions before assuming it's abandoned; it may be a future feature's foundation laid early. Also: when a feature request suggests a specific mechanism (#431's hand-tuned confidence sliders), the shipped implementation is free to choose a different mechanism (learned influence) that satisfies the underlying need — document that divergence explicitly in the PR, as #454 did, so the original requester's mental model doesn't silently mismatch what shipped.

Evidence: `git show a99ad49:custom_components/area_occupancy/db/schema.py` (schema fields present 2025-11-17, read directly); discussion #431 (open, 0 comments, verified via `gh api repos/.../discussions/431`); PR #454 (merged 2026-07-06T16:50:40Z into `main`, commit `17b71d2`, full 5-phase body read directly); PR #456 (state MERGED, `mergedAt: 2026-07-06T10:01:29Z`, base branch `feat/adjacent-areas` — merged into the feature branch, which then merged to `main` same day via #454); `custom_components/area_occupancy/const.py:189-221` on `main` (tunables, read directly); `custom_components/area_occupancy/db/schema.py` on `main` (`AreaTransitions` table present, 15 tables total, read directly); PR #454 review comments (`gh api repos/Hankanman/Area-Occupancy-Detection/pulls/454/reviews`, CodeRabbit nitpicks on `db/relationships.py` docstring and `_normalize_adjacent_areas` docstring accuracy).

---

### Honorable mention — recurring DB-cleanup-on-deletion bug class (three rounds, ~1 month apart)

Not one of the required sagas above, but a clear "don't re-fight this" pattern: **#390 → #405 (merged 2026-03-04)** fixed orphaned records left in multiple tables by `delete_area_data`. **#421 → #423 (merged 2026-03-21)** then found a *different* deletion pathway uncovered by #405: areas removed from HA's own area registry first (rather than through AOD's own UI) left orphaned AOD-side data. **#436 → #438 (merged 2026-04-17)**, paired with **#451** (the "Reset Learning" button, merged 2026-05-03), found a third gap: entry/area removal wasn't purging learned history (priors, correlations, intervals, aggregates). Each round found a genuinely different deletion pathway the previous fix hadn't covered, as new DB tables were added elsewhere in the codebase over time. **The lesson:** there is no single "cascade delete" test that covers this class — any new DB table needs its own explicit cleanup-on-area-deletion test, covering deletion through AOD's own config flow, deletion via HA's area registry, and deletion via config-entry removal, because historically these three pathways have each broken independently.

Evidence: issue #390 (closed 2026-03-04), PR #405 (merged); issue #421 (closed 2026-03-21), PR #423 (merged); issue #436 (closed 2026-04-17), PR #438 (merged); PR #451 (merged 2026-05-03) — all titles/dates verified via `gh pr view`/`gh issue view`.

### No reverted commits found

`git log --all --grep="^Revert"` and a case-insensitive `revert` grep across all branches/refs return zero true `git revert` commits in this repository's history (checked 2026-07-06). Every saga above was fixed forward with new corrective commits/PRs, never rolled back. If you're looking for "what did we try and undo," the answer is: nothing, by that mechanism — look for superseding PRs instead (e.g. SAGA 3's #440 superseded by nothing, but extended by #493; SAGA 4's #491 is a straight fix, not a revert).

Evidence: `git log --all --oneline --grep="^Revert"` (empty); `git log --all --oneline | grep -i revert` (only one false-positive match, a fixture-cleanup commit `4694529` unrelated to code revert).

---

### SAGA narratives provenance and maintenance

Re-verified 2026-07-06 (post-merge-wave sweep) against integration version 2026.5.17, `main` branch, HEAD `17b71d2`. The merge wave (PRs #486, #488, #489, #491, #492, #493, #454/#456, #494, #495, #496) landed on `main` the same day this section was originally compiled; `feat/adjacent-areas` is now fully merged and the working tree is on `main`. Every claim above was re-verified directly against the current `main` working tree (`git show origin/main:<path>`, `gh pr view --json baseRefName,mergedAt,state`, `gh issue view --json state,closedAt`), not carried over from the pre-merge draft.

Re-verification commands, one per volatile fact category in this section:

- **PR/issue merge state (any SAGA marked OPEN):** `gh pr view <number> --json state,mergedAt,baseRefName`
- **Current decay half-life logic:** `git show origin/main:custom_components/area_occupancy/data/decay.py | sed -n '55,92p'`
- **Current prior calculation logic:** `git show origin/main:custom_components/area_occupancy/data/analysis.py | grep -n "actual_period_end" -A3 -B3`
- **Current health-check thresholds:** `git show origin/main:custom_components/area_occupancy/data/health.py | grep -n "STUCK_ACTIVE_THRESHOLDS\|_PURPOSE_STUCK_ACTIVE_MULTIPLIER\|_UNAVAILABLE_EXEMPT_PREFIXES" -A8`
- **Timezone policy module:** `git show origin/main:custom_components/area_occupancy/time_utils.py | head -10`
- **Adjacent-areas merge status and tunables:** `gh pr view 454 --json state,mergedAt` (now returns `MERGED`, 2026-07-06); `git show origin/main:custom_components/area_occupancy/const.py | grep -n ADJACENCY_`
- **Recorder-load measured numbers:** `gh pr view 486 --json body -q .body` (table is in the PR body verbatim)
- **No-revert-commits claim:** `git log --all --oneline --grep="^Revert"`
- **Config version (affects any migration-adjacent saga):** `git show origin/main:custom_components/area_occupancy/const.py | grep -n "CONF_VERSION"`

---

## Learning-Accuracy Campaign

The end-to-end loop for diagnosing and fixing bad learned values on a **real** Home Assistant install: pull ground truth, hand-verify the stored math, diagnose which stage is wrong (interval detection, denominator/period selection, bucketing, likelihood estimation, or decay half-life resolution), pick a fix from a theory-obligated menu, and validate before it ever reaches `main`. It exists because this project has shipped the same class of bug — a denominator or comparison subtly excluding data it shouldn't — at least five times (see Phase 2 and Provenance below).

### Phase 0 — Ground truth harness

**Goal**: prove you can hand-recompute a stored learned value from raw data before you trust anything else in this campaign. If you skip this, every later phase is built on sand.

1. **Get a real-home diagnostics export and DB copy.** Use `aod-diagnostics-and-tooling` for the exact pull mechanics (diagnostics JSON via Settings → Devices & Services → Area Occupancy Detection → kebab menu → Download diagnostics; DB file at `<config_dir>/.storage/area_occupancy.db` on the source install). You need both: diagnostics gives you the *claimed* current values (`prior.diagnostic_snapshot()` per area — see `custom_components/area_occupancy/diagnostics.py:215-313`), the DB gives you the raw intervals to recompute from.

2. **Hand-recompute the global prior from the DB directly** (bypasses the integration's own code — this is the point):

   ```bash
   sqlite3 area_occupancy.db <<'SQL'
   .headers on
   .mode column
   SELECT area_name, prior_value, total_occupied_seconds, total_period_seconds,
          data_period_start, data_period_end,
          ROUND(total_occupied_seconds / total_period_seconds, 6) AS recomputed_ratio
   FROM global_priors;
   SQL
   ```

   `recomputed_ratio` (clamped to `[0.01, 0.99]`) must equal `prior_value` to within `1e-6`. This only catches *arithmetic* bugs in the stored row — it will NOT catch the #483 class of bug, where the period bounds themselves were wrong before the ratio was ever computed. For that, independently rebuild the period from `occupied_intervals_cache`:

   ```bash
   sqlite3 area_occupancy.db <<'SQL'
   SELECT area_name, MIN(start_time) AS earliest_cached, MAX(end_time) AS latest_cached,
          SUM(duration_seconds) AS total_occupied_seconds_recomputed
   FROM occupied_intervals_cache
   WHERE area_name = '<AREA_NAME>'
   GROUP BY area_name;
   SQL
   ```

   Then compare `latest_cached` to `global_priors.data_period_end` for that area. **They should be very close to "now" at analysis time** — if `data_period_end` is stuck hours or days behind `latest_cached`/the current wall clock while the area has been quiet, you have re-found the #483 pattern; see § SAGA 4 above for the full root-cause narrative and fix. The fix (period always ends at `now`, via `actual_period_end = now`) landed on `main` via PR #491, merged 2026-07-06; re-verify with `git show main:custom_components/area_occupancy/data/analysis.py | grep -n "actual_period_end"` if you suspect a regression.

3. **Hand-recompute one time-prior bucket.** Time priors are bucketed by *local* wall-clock day-of-week/hour, 168 buckets per area, in `custom_components/area_occupancy/db/schema.py`'s `priors` table (`day_of_week`, `time_slot`, `prior_value`, `data_points`). You need the HA instance's configured timezone (Settings → System → General → Time Zone on the source install — it is **not** stored per-area in the DB). Recompute bucket `(dow, hour)` by hand:

   ```python
   import sqlite3
   from datetime import timedelta
   from zoneinfo import ZoneInfo

   TZ = ZoneInfo("<HA_INSTANCE_TIMEZONE>")   # from Settings > System > General
   AREA = "<AREA_NAME>"
   DOW, HOUR = 0, 8   # Monday 08:00-09:00 local, e.g.

   con = sqlite3.connect("area_occupancy.db")
   rows = con.execute(
       "SELECT start_time, end_time FROM occupied_intervals_cache WHERE area_name = ?",
       (AREA,),
   ).fetchall()

   occupied = 0.0
   total = 0.0
   # Walk every local hour in [sample_period_start, sample_period_end) for this area/bucket
   # from the `priors` table row, summing overlap with `rows` for occupied seconds
   # and the full hour length for total seconds. (Mirrors calculate_time_priors in
   # custom_components/area_occupancy/data/analysis.py — walk in UTC, bucket by local time,
   # to stay DST-safe; see that function's docstring before you copy this loop verbatim.)
   ```

   Compare your hand value (clamped to `[0.03, 0.9]` — `TIME_PRIOR_MIN_BOUND`/`TIME_PRIOR_MAX_BOUND`, `const.py:186-187`) to the stored `prior_value` for that `(day_of_week, time_slot)` row. **Expected: within 1e-6.**

4. **If the stored value differs from your hand value**: you found a bug in the period-selection, bucketing, or denominator logic, not a rounding artifact — branch to this skill's Symptom → Triage Table section above, treating it as a "#483-class" bug (systematic denominator/period truncation, see § SAGA 4 above), and come back here once fixed to re-verify before moving to Phase 1.

---

### Phase 1 — Baseline measurement

Define these metrics precisely before touching any code, and compute them for the specific real-home area under investigation.

| Metric | Definition | How to compute | Known-good range |
|---|---|---|---|
| Prior calibration error | `\|stored global_prior − true occupancy rate over the same window\|` | True rate: from an independent reference signal (e.g. manual log, a second known-reliable presence sensor) covering the *same* `data_period_start`→`data_period_end` window from `global_priors`. Stored: that row's `prior_value`. | No repo-wide target exists. The one documented real case (#483, kitchen mmWave) had true rate **28–35%** over 7 days while the buggy prior pinned at **0.99**; after PR #491's fix the same synthetic scenario in `tests/test_data_analysis.py::test_valid_calculation_sets_correct_prior` recomputes to **0.25** for 2h occupied / 8h period. Use "does the recomputed ratio match the true rate to within a few points" as the bar, not a repo-blessed number. |
| False-transition count | Number of times the area's `binary_sensor.<area>_occupancy` (via `Area.occupied()`, threshold-gated at `config.threshold`, default 0.50) flips state without a corroborating flip in your chosen reference signal, over N days | Pull `occupied`/`probability` history for the binary sensor from the recorder (or replay `intervals` + `global_priors`/`priors` through the simulator, see Phase 3 fenced-path warning) and diff transition timestamps against the reference signal's transitions, with a small tolerance window (a few seconds to a couple minutes, matching `motion_timeout` — `DEFAULT_MOTION_TIMEOUT = 300s`, `const.py:132`) | No repo-wide target exists; this is a per-area empirical baseline you establish, not a constant to look up. |
| Sample sufficiency | Number of samples correlation analysis had for this entity | `SELECT sample_count FROM correlations WHERE area_name=? AND entity_id=?;` | Must be ≥ `MIN_CORRELATION_SAMPLES = 50` (`const.py:323`) before you trust `prob_given_true`/`prob_given_false` for that entity at all — below 50, `db/correlation.py`'s gate returns early / discounts confidence (see Phase 2). |
| Time-prior data density | Weeks of data feeding a given `(day_of_week, time_slot)` bucket | `SELECT data_points FROM priors WHERE area_name=? AND day_of_week=? AND time_slot=?;` | `data_points` is a **diagnostic-only** field — `calculate_time_priors` (`data/analysis.py:675-803`) does NOT gate on it before writing a slot's prior. A bucket with `data_points = 1` is exactly as "trusted" by the code as one with `data_points = 12`. This is a real gap — see Phase 3's min-sample-gate proposal. |

Also record, for the area under investigation: `config.threshold`, `purpose`, whether it has a custom half-life override, and whether it's SLEEPING-purpose (the only purpose with an awake/sleep half-life split — see Phase 2).

---

### Phase 2 — Diagnosis tree

Work top-down; each branch has a discriminating query so you don't have to guess.

#### If priors look wrong (calibration error large)

```
Is occupied_intervals_cache itself wrong (interval detection)?
├─ Query: SELECT * FROM occupied_intervals_cache WHERE area_name=? ORDER BY start_time DESC LIMIT 50;
│  Compare each interval's start/end against the raw `intervals` table for motion/sleep/media
│  entities in that window (get_occupied_intervals in db/queries.py merges motion+sleep+media,
│  applies motion timeout ONLY to motion intervals via apply_motion_timeout — a media-only
│  "occupied" interval never gets extended past its actual end).
│  → If cache intervals don't match raw sensor activity: bug is in
│    build_motion_query/build_presence_query/process_query_results (db/queries.py) or in
│    apply_motion_timeout/merge_overlapping_intervals (utils.py). Branch to this skill's
│    Symptom → Triage Table section above.
│
Is the DENOMINATOR/period wrong (the #483 bug class)?
├─ Query: compare global_priors.data_period_end to "now" at the time of the last analysis run
│  (coordinator.py's hourly analysis timer, ANALYSIS_INTERVAL=3600s, const.py:343).
│  → If data_period_end lags "now" by more than one analysis cycle while the area has valid
│    recent cache entries: this IS the #483 pattern re-emerging on a branch that predates or
│    regressed the fix (see § SAGA 4 above for the full incident). The fix itself (period always
│    ends at `now`) is on `main` via PR #491, merged 2026-07-06 — if you're seeing this on
│    current `main`, treat it as a regression, not the original bug, and re-verify with
│    `gh pr view 491`.
│
Is the BUCKETING wrong (time priors only)?
├─ Query: pick one bucket, redo Phase 0 step 3 by hand. If the stored value only diverges for
│  buckets spanning a DST transition, check calculate_time_priors's UTC-walk logic
│  (data/analysis.py:675-803) — it's deliberately designed to walk hour-by-hour in UTC and
│  derive the local bucket key per iteration specifically to avoid DST fall-back ambiguity
│  (repeated local hours). A regression here would most likely reappear as a bucket with an
│  implausible data_points count (too high, from double-counting a repeated local hour) around
│  a DST boundary date.
```

#### If likelihoods look wrong (a sensor's contribution to probability feels off)

1. Confirm sample sufficiency first: `sample_count >= MIN_CORRELATION_SAMPLES (50)` in the `correlations` table for that entity — below 50, don't trust the numbers at all; `db/correlation.py:99` gates the raw computation, `:1056` discounts confidence as `abs_correlation * (1 - 50/sample_count)`, and `:1541` re-invalidates on reload if it ever drops back under 50.
2. For **binary** sensors (motion/door/media/appliance/etc.), likelihoods come from `analyze_binary_likelihoods` (`db/correlation.py:324-620`) — a **duration-overlap** method, not a point-sample method: it sums how many seconds the sensor was in an active state during occupied vs. unoccupied cache intervals, then divides. This means **the same `occupied_intervals_cache` ground truth that feeds priors also feeds likelihoods** — if you found an interval-detection bug in the priors branch above, expect corresponding likelihood distortion for every binary sensor in that area, not just the prior.
3. Check for the "black hole" clamp: `prob_given_true`/`prob_given_false` are clamped to `[0.05, 0.95]` (`db/correlation.py:584-588`) — a sensor at exactly 0.05 or 0.95 might be a real strong signal, or might be masking a near-0/near-1 raw ratio; check `analysis_error` on the `correlations`/`entities` row (`no_occupied_intervals`, `no_active_intervals`, `no_active_during_occupied`) before trusting the clamped number — any of those error strings means the type-default likelihood was used instead of a learned one, not the clamped value.
4. For **numeric/environmental** sensors, use `scripts/visualize_distributions.py "<Area Name>" <entity_id> --db-path config/.storage/area_occupancy.db --days 30` to plot the raw histogram against the fitted Gaussian and visually confirm the learned `mean`/`std` actually fit the data, rather than trusting `correlation_strength` blind (this is exactly what that script is for — see its docstring).

#### If decay looks wrong (clears too fast/slow, or ignores a custom half-life)

1. Trace `Decay.half_life` → `Decay._resolve_purpose_half_life()` (`custom_components/area_occupancy/data/decay.py`). Only `AreaPurpose.SLEEPING` has an `awake_half_life` (620s, `data/purpose.py`) — every other purpose returns `self._base_half_life` unconditionally, so this branch only matters for Bedroom/Sleeping-purpose areas with configured `sleep_start`/`sleep_end`.
2. **Settled bug (#481, fixed via PR #493, merged 2026-07-06)** — see § SAGA 3 above for the full root-cause narrative (same custom-vs-default semantics bug as #439/#440, recurring in `_resolve_purpose_half_life()`). This is live on `main`; re-verify with `gh pr view 493 --json state,mergedAt` if you suspect a regression.
3. If you hit this bug class again elsewhere: the general pattern (see § SAGA 3 above) is "is-this-value-a-user-override-or-a-coincidentally-matching-default" — `Purpose.is_purpose_half_life()` in `data/purpose.py` is the load-bearing comparison; it was already burned once (#439/#440) by comparing against *any* purpose's default instead of only the *selected* purpose's default. Any new decay/half-life logic must re-check which comparison it's making.
4. Also check `Decay.modifier_factor` (adjacency Phase 4 decay-stretch, only relevant if the area has adjacent-area config — merged to `main` via PR #454, 2026-07-06; still out of scope for a *learning-accuracy* investigation per this skill's "When NOT to use this" note, since it consumes decay rather than producing learned values, but it is no longer branch-gated).

---

### Phase 3 — Solution menu (ranked, each with a derivation obligation)

**Rule for every item below: predict the numeric effect on your Phase 1 baseline BEFORE writing code.** If you can't state a predicted number, you don't understand the mechanism well enough to touch it yet.

| # | Fix | Mechanism | Derivation obligation (predict BEFORE coding) | Risk | Test plan |
|---|---|---|---|---|---|
| 1 | Interval-merge / motion-timeout tuning | `apply_motion_timeout`/`merge_overlapping_intervals` (`utils.py`) control how raw motion edges become "occupied" intervals; `DEFAULT_MOTION_TIMEOUT=300s` (`const.py:132`) | For a specific area's raw motion log, hand-compute the merged interval set at the current timeout vs. a candidate timeout; state the predicted change in total occupied seconds and interval count | Changes ground truth for BOTH priors and likelihoods simultaneously (shared cache) — a "fix" here silently re-derives every downstream learned value for that area | Add a fixture with known raw motion edges + expected merged intervals at both timeouts; run `tests/test_db_correlation.py`/interval-merge tests before/after |
| 2 | Min-sample gate before trusting learned priors/likelihoods (fallback to default below threshold) | Mirror `MIN_CORRELATION_SAMPLES` (used for likelihoods) — time priors currently have NO such gate (`data_points` is diagnostic-only, Phase 1) | State the exact threshold (e.g. weeks-of-data or `data_points` count) and predict which currently-populated buckets in your real-home DB would flip to "insufficient data → use global/default prior" at that threshold | Sacred config surface risk is low (no user-facing config change needed) but changes silent behavior for every area with sparse history — could visibly change existing installs' probability on upgrade | Unit test: a bucket with `data_points` below threshold must fall back to global_prior/purpose default, not its own (possibly noisy) `prior_value`; regression test with hand-picked before/after `data_points` values at the boundary |
| 3 | Time-prior smoothing across adjacent buckets | Currently each of the 168 buckets is independent — no borrowing from neighboring hours/days | Predict the smoothed value for one sparse bucket given its neighbors (e.g. simple weighted average with 2 adjacent hours) and confirm it moves calibration error in the right direction for a real sparse bucket, not just in theory | Changes math for literally every area — this is the highest blast-radius item on this list; touches the "no silent math changes" unwritten law hardest | Full regression suite on `tests/test_data_prior.py` + `tests/test_data_analysis.py`; hand-computed expected values for at least 3 real buckets (dense, sparse, DST-boundary) before merging |
| 4 | Likelihood learning improvements from correlation data (e.g. confidence-weighted blending with type defaults instead of hard `analysis_error` fallback) | `analyze_binary_likelihoods`/`analyze_correlation` (`db/correlation.py`) currently either use a fully-learned clamped value or fully fall back to type defaults — no blend | Predict, for one real entity with borderline sample count (just above/below 50), how a confidence-weighted blend changes its effective `prob_given_true` vs. today's binary choice | Same shared-ground-truth risk as #1; also touches the `strength_multiplier`/`effective_weight` pipeline in `utils.py` which is otherwise stable | Unit tests with sample_count fixed at 49/50/51/100 boundary values; confirm blend continuity (no discontinuous jump at the threshold) |
| 5 | Purpose-floor recalibration (per-purpose `min_prior` values) | `Purpose.min_prior` table (`data/purpose.py`) currently: PASSAGEWAY=0.1, DRIVEWAY=0.05, all others=0.0 — capped below `config.threshold - PRIOR_FLOOR_THRESHOLD_MARGIN(0.01)` per issue #435 | Predict which real areas' effective prior floor would change and by how much; this is the lowest-leverage item since it only ever raises the floor, never fixes a systematically wrong learned value | Lowest risk of the five (bounded, capped-below-threshold by construction) but easy to convince yourself it "fixed" a bug that was actually elsewhere — a floor change can mask a still-broken denominator | `tests/test_data_prior.py::test_min_prior_override_scenarios`-style parametrized cases; explicitly verify a genuinely-wrong learned prior still surfaces as wrong (floor doesn't hide it) |

Rank order above is intentional: fix interval/denominator truth (mechanical, shared-ground-truth) before adding smoothing or blending logic on top of it — smoothing a wrong signal just gives you a smoother wrong signal.

---

### Phase 4 — Validation and promotion

No fix from Phase 3 goes anywhere near `main` without all of the following, in order:

1. **Regression tests encoding the correct math**, not just "test doesn't crash." Follow the PR #491 pattern exactly: the old test (`tests/test_data_analysis.py::test_valid_calculation_sets_correct_prior`) had the *buggy* expected value (`0.99`) baked into its assertion; the fix corrected the assertion to the hand-derived value (`0.25`) and added a dedicated regression test (`test_quiet_tail_included_in_denominator`) that fails if the bug ever comes back. Any fix in this campaign needs the equivalent: a test whose expected value you hand-derived in Phase 0/2, not one copied from the current (possibly wrong) output.
2. **Simulator scenario** — paste a `run_analysis` service-response snapshot (from `custom_components.area_occupancy.service.py`'s `run_analysis` action) into `simulator/app.py` (run via `python main.py` from repo root) and interactively toggle sensors to sanity-check the fix's effect end-to-end. **Do not treat the simulator's own internal state machine as ground truth** — it has zero test coverage of its own (`simulator/` has no `test_*.py` files; confirmed by direct listing, only `app.py`/`__init__.py`), it's a visualization/debugging aid built on real integration code (`EntityType`/`Entity` classes imported directly), not a validated oracle.
3. **Pre-merge DB-copy check** — run the fix against the actual DB copy from Phase 0 (or a fresh sync from the source install) for at least one full analysis cycle (hourly, `ANALYSIS_INTERVAL=3600s`), and record before/after deltas for every Phase 1 metric on that specific area. "It looks better" is not a result; the number must move in the predicted direction from your Phase 3 derivation, by roughly the predicted amount. Note this is the *pre-merge* validation stage against copied data — it is distinct from, and does not replace, the *post-merge* live-install soak in `aod-change-and-validation`'s idea lifecycle (where the merged fix rides along in the next release on a real install before being declared done). A campaign fix goes through both, in that order.
4. **Route through `aod-change-and-validation`** — this campaign's output is a diagnosed bug + a validated fix + before/after numbers. It is never merged on eyeball, and it is never merged by this skill directly.

---

### Fenced wrong paths (do not do these)

- **Widening the `[0.01, 0.99]` clamps** (`MIN_PROBABILITY`/`MAX_PROBABILITY`, `MIN_PRIOR`/`MAX_PRIOR`, `const.py:173-176`) to "fix" a pinned prior. The clamp isn't the bug; something upstream produced a ratio that saturated it. Widening the clamp just moves where saturation becomes visible.
- **Truncating the observation period** to make a metric look better (this is literally the #483 bug — see Phase 2 above). The period must always extend to "now"; anything else drops known-unoccupied time from the denominator.
- **Hand-tuning constants per-home.** Every threshold in this file (`MIN_CORRELATION_SAMPLES`, half-life defaults, `TIME_PRIOR_MIN_BOUND`/`MAX_BOUND`, etc.) is a repo-wide default. A fix that only works because you hardcoded a number for one specific install's DB is not a fix — the "config surface is sacred" unwritten law means new *user-facing* knobs need config-flow + migration work (see `aod-architecture-and-config`), not silent per-home constant edits.
- **Trusting the simulator backend as ground truth** for anything you haven't independently hand-verified per Phase 0/4 — it is untested code, not an oracle, however convenient it is to click sensors and watch numbers move.
- **Changing math and tests in the same breath** without a hand-computed expected value first. If your only evidence a change is correct is "the test I also just wrote passes," you've encoded your assumption, not verified it. Do Phase 0/1's hand computation, THEN write the code, THEN write the test against the hand-derived number.

---

### Campaign provenance and maintenance

Date-stamped: 2026-07-06 (post-merge sweep, `main` HEAD `17b71d2`), integration version still 2026.5.17 (`custom_components/area_occupancy/manifest.json`, `pyproject.toml`, `const.py::DEVICE_SW_VERSION`) — none of the fixes below have shipped in a tagged release yet.

Merged since the prior sweep, all confirmed on `main` as of 2026-07-06:
- PR #491 (fix: quiet-tail global-prior denominator, fixes #483) — merged 2026-07-06; re-verify: `gh pr view 491 --json state,mergedAt`
- PR #493 (fix: bedroom custom half-life outside sleep window, fixes #481) — merged 2026-07-06; re-verify: `gh pr view 493 --json state,mergedAt`
- PR #454 (feat: adjacent-areas) — merged 2026-07-06; out of scope for this campaign but shares the coordinator tick; re-verify: `gh pr view 454 --json state,mergedAt`

Re-verification commands for every volatile fact category in this section:
- Clamp/threshold constants (`MIN_PROBABILITY`, `MIN_PRIOR`, `TIME_PRIOR_MIN_BOUND`, `MIN_CORRELATION_SAMPLES`, etc.): `grep -n "MIN_PROBABILITY\|MAX_PROBABILITY\|MIN_PRIOR\|MAX_PRIOR\|TIME_PRIOR_MIN_BOUND\|TIME_PRIOR_MAX_BOUND\|MIN_CORRELATION_SAMPLES\|PRIOR_FLOOR_THRESHOLD_MARGIN\|DEFAULT_MOTION_TIMEOUT" custom_components/area_occupancy/const.py`
- Global prior period-selection logic: `grep -n "actual_period_end\|actual_period_duration" custom_components/area_occupancy/data/analysis.py`
- Time-prior bucketing/DST handling: read `custom_components/area_occupancy/data/analysis.py`'s `calculate_time_priors` function in full (docstring + body)
- Likelihood computation and clamps: `grep -n "def analyze_binary_likelihoods\|clamp_probability(prob_given" custom_components/area_occupancy/db/correlation.py`
- Decay half-life resolution: read `custom_components/area_occupancy/data/decay.py`'s `_resolve_purpose_half_life` in full; purpose defaults in `custom_components/area_occupancy/data/purpose.py`
- PR/issue merge state for anything cited above: `gh pr view <n> --json state,mergedAt,statusCheckRollup` / `gh issue view <n> --json state`
- Coverage gate: `grep -n "fail_under" pyproject.toml` (currently 85%, comment claims 90% — believe the number, not the comment)
- Simulator test coverage: `find simulator -maxdepth 1 -iname "test*"` (expect no matches as of this writing)
