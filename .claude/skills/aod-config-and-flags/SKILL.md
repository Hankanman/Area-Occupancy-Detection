---
name: aod-config-and-flags
description: Use when adding, renaming, removing, or debugging any CONF_* / DEFAULT_* configuration key, config_flow.py schema, strings.json/translations entry, migrations.py version bump, or sensor default in Area Occupancy Detection; when a user reports a setting "not sticking", reverting, or silently overwritten (e.g. decay half-life, threshold, sensor precision); or when deciding whether a change needs a CONF_VERSION bump. Covers the full CONF_* catalog with defaults/ranges/stability, the decay half-life 0-sentinel, the #440 normalisation rule, and the add-a-new-option checklist.
---

# AOD Config and Flags

## What this covers

The complete configuration surface of Area Occupancy Detection: every `CONF_*`/`DEFAULT_*` key in `const.py`, the `IntegrationConfig` vs `AreaConfig` split, config-flow/options-flow schema wiring, the decay-half-life sentinel and its #440 anti-clobber rule, `migrations.py`'s version ladder, and the checklist (with re-verification commands) for adding a new option safely. This is the "what does this setting do, what's its default/range, and how do I add one without breaking existing users" skill.

## When NOT to use this

- Changing the actual Bayesian math (weights' *effect*, sigmoid formula, decay curve shape) → `bayesian-occupancy-reference`.
- New sensor *type* end-to-end wiring beyond the config keys (InputType enum, EntityFactory, likelihood defaults) → still start here for the config-key steps, but cross-check `aod-architecture-contract` for the entity/type pipeline.
- Whether a change is safe to ship / needs a version bump philosophy and review process → `aod-change-control`.
- Debugging a *running* instance's config (inspecting a live DB/entry) → `aod-diagnostics-and-tooling` or `aod-debugging-playbook`.
- Adjacent-areas' Bayesian boost/decay-modifier math itself (not the config keys) → `aod-research-frontier` (it is unmerged, PR #454).

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

`MIN_PRIOR`/`MAX_PRIOR`/`MIN_PROBABILITY`/`MAX_PROBABILITY` = 0.01/0.99 (const.py:173-176) are the hard safety clamp everywhere in the Bayesian pipeline — see `bayesian-occupancy-reference` for how `clamp_probability()` uses them. `TIME_PRIOR_MIN_BOUND`/`MAX_BOUND` = 0.03/0.9 (const.py:186-187) are a separate, tighter bound applied only to time-of-day priors.

Prior floor mechanism (issue #435, fixed): `Prior.value` (`data/prior.py:78-140`) applies `area.purpose.min_prior` and `config.min_prior_override` as floors, but both are capped at `floor_cap = max(MIN_PRIOR, config.threshold - PRIOR_FLOOR_THRESHOLD_MARGIN)`. A configured/purpose floor alone can never push a stale, no-evidence area's prior up to or above its own occupancy threshold — only a genuinely learned prior can cross the threshold.

### Decay

| CONF_* key | Default | Range | Stability |
|---|---|---|---|
| `CONF_DECAY_ENABLED` | `DEFAULT_DECAY_ENABLED`=True | bool | production |
| `CONF_DECAY_HALF_LIFE` | `DEFAULT_DECAY_HALF_LIFE`=0 (sentinel, see below) | 0, or 10-3600s (`_validate_config`, `invalid_decay_half_life`) | production, but see open bug PR #493 below |

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

Purpose drives the decay half-life default (45s Passageway → 1200s Sleeping) and, for `SLEEPING` only, an `awake_half_life`=620s used outside the configured sleep window — see the sentinel section, and PR #493 below for the open bug in that switch.

### Health

| CONF_* key | Default | Scope | Stability |
|---|---|---|---|
| `CONF_HEALTH_ENABLED` | `DEFAULT_HEALTH_ENABLED`=True | `IntegrationConfig` (global, not per-area) | production (landed PR #472) |

When False, `data/analysis.py` short-circuits both sensor-health and pipeline-health checks and calls `area.health_monitor.clear_all_issues()`, but leaves the in-memory `_unavailable_since` clock intact so re-enabling doesn't instantly trip every currently-unavailable sensor.

### Adjacency — experimental, unmerged as of 2026-07-06

| Key | Default | Stability |
|---|---|---|
| `CONF_ADJACENT_AREAS` ("adjacent_areas") | `[]` | **experimental** — only exists on branch `feat/adjacent-areas` (PR #454). Confirmed absent from `main`: `git show main:custom_components/area_occupancy/const.py \| grep -c ADJACENT_AREAS` → 0 as of 2026-07-06. Verify current status with `gh pr view 454` before describing as shipped. |
| `ADJACENCY_TRANSITION_WINDOW_S` | 60 | experimental, hardcoded (not a `CONF_*`, no UI) |
| `ADJACENCY_RECENCY_HALF_LIFE_DAYS` | 30 | experimental, hardcoded |
| `ADJACENCY_TRAJECTORY_WINDOW_S` | 300 | experimental, hardcoded |
| `ADJACENCY_BOOST_GAIN` | 0.5 | experimental, hardcoded |
| `ADJACENCY_DECAY_MODIFIER_GAIN` | 0.75 | experimental, hardcoded |
| `ADJACENCY_DECAY_MODIFIER_MAX` | 1.75 | experimental, hardcoded |
| `ADJACENCY_N_SPECIFIC` / `ADJACENCY_N_HOUR` / `ADJACENCY_N_CHAIN` / `ADJACENCY_N_PAIR` | 5 / 20 / 50 / 20 | experimental, hardcoded (min-observation thresholds for a 6-level smoothing fallback) |

All ten `ADJACENCY_*` constants carry the in-code comment "First-pass values; tune from real data once Phase 3 is collecting transitions" (`const.py:189-190` on the feature branch) — there is no real-recorder validation yet. None are user-configurable; only `CONF_ADJACENT_AREAS` (the neighbour list itself) has a config-flow UI. Treat any specific numeric claim about adjacency behavior as provisional and re-check `gh pr view 454` for merge status before citing it as current.

**adjacent_areas symmetric-write**: the config flow enforces mutual adjacency as a pure-function transform over the flat `CONF_AREAS` list (`config_flow.py:1614-1780`, unmerged branch only). `_normalize_adjacent_areas()` coerces any stored shape (None/str/list/tuple/set/other) to `list[str]`, defensively handling hand-edited storage JSON. `_apply_symmetric_adjacency(areas, updated_area)`: when area A saves neighbours `[B, C]`, it rewrites A's own row (self-reference stripped, sorted), adds A to B's and C's lists if missing, and removes A from any other area that used to list A but no longer should. `_strip_adjacency_references()` removes a deleted area's id from every surviving area when that area is removed. This mirroring is pure Python at config-flow save time — `AreaConfig._load_config` (`data/config.py:485-493`) just reads and string-filters the list; it does **not** itself enforce symmetry, so any other write path (e.g. a future service call) that touches `adjacent_areas` directly must replicate the mirror logic or symmetry will silently drift.

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

This exact bug class recurred in a different code path: issue #481 (a Bedroom/SLEEPING area's custom 10s half-life was overridden by the purpose's `awake_half_life`=620s outside the sleep window, because the sleep/awake switch in `Decay._resolve_purpose_half_life()` — `data/decay.py:81-116` — applied unconditionally instead of only when the half-life still equalled the purpose default). Fix is PR #493, **CI-green, awaiting merge as of 2026-07-06** — verify current state with `gh pr view 493` before telling a user it's fixed. Its body explicitly says it "mirrors the custom-vs-default semantics established for #440."

## How missing keys default, and why no CONF_VERSION bump is needed for additive keys

Every read of a `CONF_*` key in `AreaConfig._load_config` and `IntegrationConfig` properties goes through `data.get(CONF_X, DEFAULT_X)` (or `config_entry.options.get(...)`). A config entry saved before a key existed simply doesn't have it in its dict — `.get()` returns the default, no migration required, no `CONF_VERSION` bump required, **as long as the default produces correct/safe behavior for pre-existing configs.**

Two real precedents, both confirmed in the repo:

1. **PR #486** (`CONF_SENSOR_PRECISION`, merged 2026-07-06, commit `7e3a856`) added a brand-new global option with zero `CONF_VERSION` involvement — it lives in `config_entry.options`, read live via `.get()` with a clamped default. No migration, no version touch. `git show 7e3a856 -- custom_components/area_occupancy/const.py | grep CONF_VERSION` → no output.
2. **v17→v18** (`CONF_EXCLUDE_FROM_ALL_AREAS`, `migrations.py:571-581`) *did* bump `CONF_VERSION`, but the migration itself does no data mutation — the comment says so explicitly: "No data changes needed — missing key handled by `AreaConfig._load_config()`." The bump here was belt-and-suspenders (a version marker for tooling/tests), not a technical requirement of the additive change itself.

Rule of thumb: **a purely additive, `.get()`-defaulted key never needs a `CONF_VERSION` bump.** Only bump when you need to (a) mutate/rename/restructure existing stored data, or (b) force a one-time side effect (e.g. a DB reset) on upgrade. Bumping `CONF_VERSION` is not free: any mismatch versus the DB's stored schema version triggers `db/maintenance.py`'s `_ensure_schema_up_to_date`, which **deletes and recreates the entire SQLite database** (wipes all learned priors/history) — see `aod-architecture-contract`/`aod-debugging-playbook` for that mechanism. This is exactly why the (unmerged) adjacent-areas feature's own `AreaTransitions` table and `adjacent_areas` column were deliberately kept out of the version bump (`migrations.py:583-591` on the `feat/adjacent-areas` branch, not on `main`): it's additive (new column defaults on load, new table created via `Base.metadata.create_all(checkfirst=True)`), so bumping would have wiped every user's learned history for no reason.

## migrations.py rules

`async_migrate_entry` (`migrations.py:515` on) runs under a module-level `asyncio.Lock` to prevent concurrent migrations. Current ladder (`CONF_VERSION=18`, `CONF_VERSION_MINOR=0`, `const.py:33-34`):

| From → To | What happens | Idempotent? |
|---|---|---|
| `< 14` | `async_reset_database_if_needed()` deletes `.storage/area_occupancy.db` (+`-wal`/`-shm`) — breaking schema change from v13 | yes, guarded by file-exists checks |
| `13 <= v < 15` | `_migrate_energy_to_power()` strips legacy `energy_sensors`/`weight_energy` keys; unconditionally bumps to 16 even if nothing was found | yes (gated by version range) |
| `== 16` | `_migrate_sleep_sensor_to_list()` converts `CONF_PERSON_SLEEP_SENSOR` (str) → `CONF_PERSON_SLEEP_SENSORS` (list) in both `data` and `options`; bumps to 17 | yes |
| `== 17` | Pure version bump to 18 for `CONF_EXCLUDE_FROM_ALL_AREAS` — no data mutation | yes |
| `< 13` (true legacy single-area entries) | `_combine_config_entries()` merges every such entry into one target entry's `CONF_AREAS` list (deterministic target = lowest `entry_id`); invalid areas dropped; old entries marked deleted, registries cleaned | yes (only entries still `< 13` are touched) |

All numeric-version branches are gated with `if config_entry.version == N` (or a range check), so re-running the whole function on an already-migrated entry is a no-op — this is the idempotency guarantee CLAUDE.md requires. When you add a new migration step: gate it the same way, mutate both `data` and `options` dicts if the key could live in either, and log what you did.

## Checklist: adding a new config option

Verified end-to-end against how PR #486 added `CONF_SENSOR_PRECISION` (a global setting) — for a per-area/per-sensor-type option, steps 2-4 target `AreaConfig`/`Sensors`/`Weights` instead of `IntegrationConfig`.

1. **`const.py`**: add `CONF_<NAME>: Final = "<snake_case_key>"` and `DEFAULT_<NAME>: Final = <value>`. Group it near its section (weights, decay, global settings, etc.) — don't scatter.
2. **Schema section in `config_flow.py`**: add a `vol.Required`/`vol.Optional` entry with an appropriate selector (`NumberSelector`, `BooleanSelector`, `TimeSelector`, `DurationSelector`, …) and range/step matching the intended domain. For a global setting this goes in `_create_global_settings_schema` (`config_flow.py:1919-1950`); for a per-area setting, into the relevant `_create_*_section_schema` **and** wired into `_nest_config_for_sections()` — skip that second step and "suggested values" won't repopulate on edit.
3. **`strings.json` AND `translations/en.json` — BOTH files, every time.** PR #486 added the `sensor_precision` label+description to both in the same commit (`git show 7e3a856 -- custom_components/area_occupancy/strings.json custom_components/area_occupancy/translations/en.json`). The project has an existing, unfixed drift where `strings.json` is missing 10 keys that `en.json` has (the whole `services.*` block plus `person_already_configured` under both `config.error` and `options.error`) — confirmed via a JSON key-diff (`python3` flatten-and-diff, 0 keys unique to `strings.json`, 10 unique to `en.json`). Do not repeat that mistake; `strings.json` is HA's canonical source that hassfest (`validate.yml`) checks and other locales derive from.
4. **Parsing with clamp**: add the property/field read via `.get(CONF_X, DEFAULT_X)`. If the value has a valid range, clamp defensively at the read site (see `IntegrationConfig.sensor_precision`'s `max(0, min(2, precision))` inside a `try/except (ValueError, TypeError, OverflowError)`) — never trust that the selector's client-side bounds were actually respected (hand-edited YAML/JSON, old snapshots, API calls all bypass the selector).
5. **Server-side validation** in `_validate_config` if the config-flow UI doesn't already fully constrain it (compare: `CONF_SENSOR_PRECISION` relies solely on the `NumberSelector` + read-site clamp and has no `_validate_config` entry — acceptable because the read-site clamp is the true backstop; `CONF_THRESHOLD` and `CONF_DECAY_HALF_LIFE` do have explicit `_validate_config` checks because they gate deeper pipeline behavior).
6. **Tests**: `tests/test_config_flow.py` (schema/flow) and `tests/test_data_config.py` (parsing/clamping) — PR #486 touched exactly these two files for its non-UI logic (`git show 7e3a856 --stat`).
7. **Docs**: `docs/docs/getting-started/configuration.md` (or the relevant `features/*.md` page) — PR #486 added 3 lines there. See `aod-docs-and-writing` for house style.
8. **No `CONF_VERSION` bump** unless the new key requires migrating *existing* stored data (see previous section) — a purely additive `.get()`-defaulted key does not need one.

## Entity-registry enabled-default for diagnostic sensors (#488)

PR #488 (merged, commit `2c28849`) added `set_enabled_default(False)` (`sensor.py:90-92`, sets `self._attr_entity_registry_enabled_default`) to 7 diagnostic sensor classes: `PriorsSensor`, `EvidenceSensor`, `DecaySensor`, `PresenceProbabilitySensor`, `EnvironmentalConfidenceSensor`, `ActivityConfidenceSensor`, `SensorHealthSensor` (verify: `grep -n set_enabled_default custom_components/area_occupancy/sensor.py`). `ProbabilitySensor` and `DetectedActivitySensor` remain enabled by default. Rationale: these sensors update on the 10s decay timer and were measured writing ~16k recorder rows in 3 hours on a 6-area install (issue #467).

**Restore caveat, load-bearing for anyone touching entity registration**: this only applies at **first registration**. HA's `entity_registry.async_get_or_create()` on an *already-existing* registry entry routes to `_async_update_entity`, which has no `disabled_by` parameter and structurally cannot touch it — so existing installs upgrading through this change keep whatever enabled/disabled state they already had (verified by a regression test that seeds an "existing install" registry entry as enabled and asserts it stays enabled). **Deleting and re-adding an area counts as a fresh registration** — diagnostics come back disabled in that case, they are not "restored" the way an in-place reload/upgrade preserves them. If a user reports "my diagnostics sensors disappeared after I removed and re-added the area," this is why, not a bug.

## Re-verification one-liners

Run these to regenerate this catalog's facts against current `main` (note: this working tree may be checked out on a feature branch — check `git branch --show-current` first; `feat/adjacent-areas` is expected to still show `ADJACENCY_*`/`CONF_ADJACENT_AREAS` until PR #454 merges):

```bash
# Full CONF_*/DEFAULT_* symbol list with line numbers
grep -n '^CONF_\|^DEFAULT_' custom_components/area_occupancy/const.py

# Confirm adjacency is still unmerged (expect 0 on main)
git show main:custom_components/area_occupancy/const.py | grep -c ADJACENCY_
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

# Open-bug status for the #440-recurrence (PR #493) — verify before citing as fixed
gh pr view 493 --json state,mergeStateStatus
```

## Provenance and maintenance

Date-stamped 2026-07-06, integration version 2026.5.17 (main branch HEAD `704c89e`; HEAD drifts — re-derive with `git log -1 --oneline origin/main`). Facts in this skill were verified directly against the repo unless noted otherwise:

- Directly verified by reading source: full `const.py` (all `CONF_*`/`DEFAULT_*`/`ADJACENCY_*` lines and values), `data/config.py` (`IntegrationConfig`, `AreaConfig`, `Sensors`/`Weights`/`DecayConfig`/`WaspInBox` dataclasses, `_load_config`, `update_config`), `data/purpose.py` (`is_purpose_half_life`, `PURPOSE_DEFINITIONS`, `get_default_decay_half_life`), `migrations.py` (`async_migrate_entry` full ladder), `config_flow.py` (`_create_global_settings_schema`, `_apply_purpose_based_decay_default`, `_validate_config`, `_normalize_adjacent_areas`/`_apply_symmetric_adjacency`/`_strip_adjacency_references`), `sensor.py` (`set_enabled_default` call sites), `data/entity.py:731` (`MIN_WEIGHT`/`MAX_WEIGHT` usage), `number.py` (`Threshold` entity bounds).
- Directly verified by command: `git branch --show-current` (confirmed this working tree is on `feat/adjacent-areas`, not `main`); `git show main:.../const.py | grep -c ADJACENCY_` → 0; `gh pr view 454/486/488/493` for merge status; `git show 7e3a856 --stat` (PR #486's file list) and `git show 7e3a856 -- const.py | grep CONF_VERSION` → empty; the Python key-diff script for `strings.json` vs `en.json` (10 keys only in en.json, 0 only in strings.json); dead-const grep for `CONF_WEIGHT_SLEEP`, `CONF_WEIGHT_WASP`, `DEFAULT_SLEEP_WEIGHT`, and the per-type `const.py` probability constants.
- Taken from the discovery dossier and spot-checked (not independently re-derived line-by-line): the exact historical narrative timing of issues #439/#481/#467/#435 and PR #440's commit hash (`68d576b`) — the causal chain and current code state were verified directly, but the dossier's issue-comment quotes were not re-fetched via `gh issue view` in this session.
- PRs #491, #492, #493, #494, #454 are **CI-green and unmerged** as of 2026-07-06 — re-check with `gh pr view <n>` before treating any of their behavior as shipped on `main`.
