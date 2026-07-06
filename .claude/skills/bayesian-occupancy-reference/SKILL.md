---
name: bayesian-occupancy-reference
description: Load when you need the actual math behind Area Occupancy Detection's probability engine — logit-space evidence combination, prior composition (global/time/purpose-floor), exponential decay half-lives, the sleep awake/asleep half-life switch, wasp-in-box state logic, adjacent-areas Bayesian boost/decay-modifier, or the transition-lookup smoothing fallback. Use before touching utils.py, data/prior.py, data/decay.py, data/purpose.py, data/adjacency.py, data/trajectory.py, db/transitions.py, or db/correlation.py, or before answering "why does this area's probability say X" / "why is decay too fast/slow" / "why is the prior pinned near 0.99" questions.
---

# Bayesian Occupancy Reference

What this covers: the exact formulas this repo uses to turn sensor evidence into an occupancy
probability — logit-space evidence combination, the three-part prior (global + time + purpose
floor), exponential decay with purpose half-lives and the sleep awake/asleep switch, wasp-in-box
state logic, the adjacent-areas Bayesian boost and decay modifier, and the six-level transition
smoothing fallback. Every formula below is quoted from the file/line it lives in, with one
hand-computed worked example each. Read this before changing any of these files — the maintainer's
unwritten law #1 is **no silent math changes**.

When NOT to use this: for *why* a specific fix was made historically (root causes of past bugs),
use `aod-failure-archaeology`. For which PRs are open/pending and how to evaluate learning-accuracy
regressions on real data, use `aod-learning-accuracy-campaign`. For config keys/flags that drive
these formulas, use `aod-config-and-flags`. For the module dependency graph and where these files
sit in the pipeline, use `aod-architecture-contract`.

## 0. Two different "Bayesian" functions exist — know which one is live

`utils.py` defines **two** probability-combination functions. Only one is wired into production.

| Function | Space | Status (2026-07-06) | Called from |
|---|---|---|---|
| `sigmoid_probability()` (utils.py:136) | **logit space**, additive weighted terms | **LIVE** (2026-07-06) — this is the actual engine | `presence_probability()`, `environmental_confidence()` → `area/area.py::Area._base_probability()` → `Area.probability()` → `sensor.py` state |
| `bayesian_probability()` (utils.py:451) | log-probability space, per-entity log-likelihood accumulation (mathematically a weighted naive-Bayes update) | **DEAD CODE** — defined, unit-tested, never called from any production path | only `tests/test_utils.py` |

Verified: `grep -rn "bayesian_probability" custom_components/ tests/` returns only `utils.py`
(definition) and `tests/test_utils.py`.

**Why this matters**: if asked to "fix the Bayesian calculation" and you patch
`bayesian_probability()`, you change nothing in the running integration. CLAUDE.md's "Modifying
Bayesian Calculation" section points at `bayesian_probability()`, which is now stale — the live
formula is `sigmoid_probability()`. Flag this if you update CLAUDE.md.

## 1. Logit space: what it is and why the engine uses it

**Logit** (log-odds) of a probability `p` is `logit(p) = ln(p / (1-p))` (utils.py:123,
`logit()`). Its inverse is the **sigmoid**, `sigmoid(z) = 1/(1+e^-z)` (utils.py:107). Both clamp
`p` to `[0.01, 0.99]` first (`clamp_probability()`, utils.py:45, driven by `MIN_PROBABILITY`/
`MAX_PROBABILITY` in const.py:173-174) so `ln` and division never blow up.

Why logit space, not raw probability averaging:
- **Additive evidence**: each sensor's contribution is a term added to a running sum `z` — adding
  a tenth sensor is just "add one more term," no renormalization of everything else.
- **Order independence**: summation is commutative, so `dict` iteration order over entities can't
  change the result (a real risk otherwise, since dict order isn't a stable contract).
- **Symmetric high/low evidence**: `logit(0.99) ≈ +4.6`, `logit(0.01) ≈ -4.6` — confident
  "occupied" and confident "empty" sensors push equally hard in opposite directions, whereas
  averaging raw probabilities compresses low values disproportionately.

### The live formula — `sigmoid_probability()` (utils.py:136-199)

```
z = logit(prior) + Σ_i [ effective_weight_i × evidence_i × correlation_i × (prob_given_true_i × strength_multiplier_i) ]
P = sigmoid(z)
```

Term definitions (define once):
- `prior` — the area's current `Prior.value` (see §2) or 0.5 default.
- `evidence_i` — `1.0` if the sensor is currently active, `entity.decay_factor` (∈[0,1]) if
  decaying-but-inactive, `0.0` if inactive and not decaying (utils.py:177-182).
- `correlation_i` — learned per-entity correlation multiplier from `db/correlation.py`
  (`get_entity_correlations`), defaults to `1.0` if not yet learned.
- `prob_given_true_i` — "signal strength": how strongly this sensor type indicates occupancy when
  active (per-`InputType` default, see table in §4).
- `strength_multiplier_i` — a per-type logit-space amplifier: `3.0` for `MOTION` and `SLEEP`
  (ground-truth-like sensors), `2.0` for everything else (`data/entity_type.py` `DEFAULT_TYPES`,
  field `strength_multiplier`).
- `effective_weight_i = weight_i × information_gain_i` (`data/entity.py:330-332`).
  `information_gain_i = min(1, |prob_given_true_i − prob_given_false_i| / max(prob_given_true_i, prob_given_false_i, 0.01))`
  (`data/entity.py:309-327`) — a sensor whose active/inactive likelihoods are nearly identical
  (uninformative) contributes almost nothing regardless of its configured weight.

Two composition layers sit on top of `sigmoid_probability()`, both also in logit space:
- `presence_probability()` (utils.py:202) filters to `PRESENCE_INPUT_TYPES` (motion, media,
  appliance, door, window, cover, power, sleep — `data/entity_type.py:192-201`) and calls
  `sigmoid_probability()`.
- `environmental_confidence()` (utils.py:236) filters to `ENVIRONMENTAL_INPUT_TYPES` (temperature,
  humidity, illuminance, co2, co, sound_pressure, pressure, air_quality, voc, pm25, pm10,
  environmental) with `prior=0.5` so the result is pure environmental signal.
- `combined_probability(presence, environmental)` (utils.py:267) blends them with an **80/20
  logit-space weighted average**: `z = 0.8×logit(presence) + 0.2×logit(environmental)`. Called
  from `Area._base_probability()` (`area/area.py:216`) — **skipped** (returns `presence` directly)
  when there are zero environmental entities configured, so the 80/20 blend never needlessly
  compresses a presence-only area toward 0.5 (`area/area.py:240-241`).

After `_base_probability()`, `Area.probability()` (`area/area.py:245-274`) applies, in order:
1. **Activity boost** (`apply_activity_boost`, utils.py:293) — if `detect_activity()` finds a
   strong activity (TV, shower, cooking), `z = logit(base) + activity_boost × activity_confidence`,
   then sigmoid back. Boost magnitudes are `ACTIVITY_BOOST_HIGH=1.5` (showering/bathing/sleeping),
   `_STRONG=1.2` (TV), `_MODERATE=1.0` (cooking/working), `_MILD=0.8` (music/eating) — const.py:167-170.
2. **Adjacency boost** (`apply_logit_boost`, see §6) — merged 2026-07-06 (PR #454), always applied.

### Worked example — motion sensor lifts a low prior

Area prior = 0.30 (quiet room). One motion sensor, never analyzed yet (uses `InputType.MOTION`
defaults: `prob_given_true=0.95`, `prob_given_false=0.005`, `weight=1.0`, `strength_multiplier=3.0`
— `data/entity_type.py:226-233`), currently **active**, no learned correlation (`correlation=1.0`).

```
information_gain = min(1, |0.95 − 0.005| / max(0.95, 0.005, 0.01)) = 0.9947
effective_weight = 1.0 × 0.9947 = 0.9947
contribution = 0.9947 × 1.0 (evidence) × 1.0 (correlation) × (0.95 × 3.0) = 2.8350
z = logit(0.30) + 2.8350 = −0.8473 + 2.8350 = 1.9877
P = sigmoid(1.9877) = 1 / (1 + e^−1.9877) ≈ 0.8795
```

Result: `presence_probability()` returns **≈ 0.880** (hand-executed against the actual code path).

## 2. Prior composition

Three layers, combined in `data/prior.py::Prior._compute_value_and_floor()` (lines 97-140):
`learned = combine(global_prior, time_prior)`, then raised (never lowered) by whichever floor
(`purpose.min_prior` or `config.min_prior_override`) is higher, capped below the area's threshold.

### 2a. Global prior — `occupied_seconds / period_seconds`

Computed once per analysis cycle in `data/analysis.py` (`PriorAnalyzer`, ~line 400-620):

```
global_prior = clamp(occupied_duration / actual_period_duration, 0.01, 0.99)
```

- `occupied_duration` = sum of merged occupied-interval durations over the period.
- `actual_period_duration` = `actual_period_end − first_interval_start`.
- **`actual_period_end` — SETTLED, PR #491 merged 2026-07-06 (main HEAD `17b71d2`).**
  `actual_period_end` is now **always** `now` (bar the existing clock-skew/invalid-bounds
  fallbacks) — `data/analysis.py:518`, comment there cites the historical bug as `#483`. Before the
  fix, `actual_period_end` fell back to `last_interval_end` whenever the area had been quiet >1h,
  which dropped known-quiet time from the denominator so every overnight recalculation re-inflated
  `global_prior` until it pinned at `MAX_PRIOR = 0.99` — one of the maintainer's three costliest
  historical bugs (issue #483, now closed). See `aod-failure-archaeology` for the full saga; this
  section only tracks the live formula.
- Fallback: invalid bounds, clock skew, or non-positive duration → `global_prior = 0.01` (hardcoded
  literal, same value as but independent of `MIN_PRIOR` — `data/analysis.py:472,494,541,565`).

**Worked example**: an area has 5 hours occupied over the last 2 days (172,800 s), and the last
interval ended <1h ago so both old and new code agree `actual_period_end = now`:
`global_prior = 5×3600 / 172800 = 18000/172800 ≈ 0.1042`.

### 2b. Time prior — 168 day-hour buckets, LOCAL time

`Prior.time_prior` (`data/prior.py:162-174`) looks up `(day_of_week, time_slot)` in a cache of all
168 buckets loaded via `db.get_all_time_priors()`. `day_of_week` is `to_local(dt_util.utcnow()).weekday()`
(0=Monday..6=Sunday); `time_slot = (hour*60+minute) // 60` i.e. plain hour-of-day 0-23
(`DEFAULT_SLOT_MINUTES = 60`, `data/prior.py:38,181-185`). **Local time, not UTC** — deliberate:
bucketing by UTC hour would silently shift every learned bucket by 1 hour across a DST transition,
corrupting months of learned data twice a year. `to_local()`/`to_utc()` live in `time_utils.py`;
timezone/DST bugs are the maintainer's #1 named historical failure class (see
`aod-failure-archaeology`). Values are bounded to
`[TIME_PRIOR_MIN_BOUND=0.03, TIME_PRIOR_MAX_BOUND=0.9]` (const.py:186-187) after cache load
(`data/prior.py:227-231`). Missing bucket → `DEFAULT_TIME_PRIOR = 0.5` (const.py:226).

### 2c. Combining global + time prior — logit-space weighted average

`combine_priors(area_prior, time_prior, time_weight=0.4)` (utils.py:558-618), called as
`combine_priors(self.global_prior, self.time_prior)` — **default `time_weight=0.4` is what's
actually used** (`data/prior.py:113`):

```
combined_logit = (1 − time_weight) × logit(area_prior) + time_weight × logit(time_prior)
combined_prior = sigmoid(combined_logit)
```

Edge cases handled explicitly before the logit math: `time_weight ∈ {0,1}` returns the
corresponding input untouched; `area_prior`/`time_prior` of exactly `0.0`/`1.0` are nudged to
`MIN_PROBABILITY`/`MAX_PROBABILITY` first (a literal 0 or 1 would blow up `logit()`); identical
priors within `1e-10` short-circuit to avoid needless float churn.

**Worked example**: `global_prior = 0.1042` (from §2a), `time_prior = 0.6` (a historically busy
hour), default `time_weight=0.4`:
```
logit(0.1042) = ln(0.1042/0.8958) ≈ −2.1512
logit(0.6)    = ln(0.6/0.4)       ≈  0.4055
combined_logit = 0.6×(−2.1512) + 0.4×0.4055 = −1.2907 + 0.1622 = −1.1285
combined_prior = sigmoid(−1.1285) ≈ 0.2445
```
Result: the area's learned prior before any floor is **≈ 0.245** — pulled up from the 0.104 global
figure because this hour is historically busier.

### 2d. Purpose floor and `min_prior_override` — why transit spaces need a floor

`_compute_value_and_floor()` (`data/prior.py:97-140`) raises `learned` (never lowers it) to the
higher of `purpose.min_prior` and `config.min_prior_override`, each **capped** at
`config.threshold − PRIOR_FLOOR_THRESHOLD_MARGIN` (`0.01`, const.py:183) so a floor alone can never
hold an area "occupied" above threshold with zero active evidence (issue #435 — see
`aod-failure-archaeology`).

Only two purposes carry a non-zero `min_prior` (`data/purpose.py:161-174`):
`PASSAGEWAY: min_prior=0.1`, `DRIVEWAY: min_prior=0.05`. Every other purpose is `0.0` (no floor).

**Why transit spaces specifically**: a passageway/hallway's *duration-based* occupied fraction is
naturally tiny — people walk through in seconds, so `occupied_seconds/period_seconds` trends toward
zero even on a busy household. Without a floor every quick walk-through would need near-certain
sensor evidence just to register; the floor guarantees a baseline reflecting "used briefly but
often," not "basically never used."

**Worked example**: passageway, `threshold=0.5` (`DEFAULT_THRESHOLD=50.0` stored as `/100` —
`data/config.py:478`), `min_prior=0.1`: `floor_cap = max(0.01, 0.5−0.01) = 0.49`;
`capped_purpose_floor = min(0.1, 0.49) = 0.1`. If the learned prior from §2c comes out below 0.1
(plausible for a passageway), the effective `Prior.value` is raised to exactly `0.1`.

## 3. Decay: exponential half-life with a practical-zero cutoff

`Decay.decay_factor` (`data/decay.py:118-152`):

```
factor = 0.5 ^ (age / half_life)     # age = seconds since decay_start
if factor < 0.05: return 0.0          # practical-zero cutoff
```

`age` is `dt_util.utcnow() − decay_start`, always UTC (no local-time conversion needed here — pure
elapsed duration). `tick()` (`data/decay.py:154-170`) is what actually stops `is_decaying` once the
factor drops below the 5% floor, called from the coordinator's 10-second decay timer (per
CLAUDE.md's Timers section).

### Purpose-based half-lives (`data/purpose.py:160-236`) — full table

| Purpose | Name | `half_life` (s) | `min_prior` | `awake_half_life` (s) |
|---|---|---:|---:|---:|
| `passageway` | Passageway | 45 | 0.1 | — |
| `driveway` | Driveway | 60 | 0.05 | — |
| `utility` | Utility | 90 | 0 | — |
| `garage` | Garage | 180 | 0 | — |
| `food_prep` | Kitchen | 240 | 0 | — |
| `garden` | Garden | 360 | 0 | — |
| `bathroom` | Bathroom | 450 | 0 | — |
| `eating` | Dining Room | 480 | 0 | — |
| `social` | Living Room | 520 | 0 | — |
| `working` | Office | 600 | 0 | — |
| `relaxing` | Media Room | 620 | 0 | — |
| `sleeping` | Bedroom | 1200 | 0 | 620 |

Only `sleeping` (Bedroom) has an `awake_half_life`. Default purpose used when none configured is
`DEFAULT_PURPOSE` (check `const.py` for current value; social/Living Room is a reasonable prior
guess but verify before relying on it).

### Worked example — half-life decay curve

Living Room (`half_life = 520`s), motion stops at `t=0`:
- `age=260` (half a half-life): `factor = 0.5^(260/520) = 0.5^0.5 ≈ 0.7071`
- `age=1560` (3 half-lives): `factor = 0.5^3 = 0.125` — still counted as evidence
- `age≈2247` (`520 × log2(20) ≈ 2247`s ≈ 37.5 min): `factor ≈ 0.050` — right at the cutoff
- `age=2250`: `factor ≈ 0.0498 < 0.05` → **returns 0.0**, `tick()` sets `is_decaying=False`

General rule: practical-zero is reached at `half_life × log2(20) ≈ half_life × 4.32`, i.e. about
4.3 half-lives after the last evidence, regardless of which purpose's half-life is in play.

### The SLEEPING awake/asleep switch and the custom-override rule (PR #493)

`Decay._resolve_purpose_half_life()` (`data/decay.py:81-116`): if the purpose has an
`awake_half_life` (only `sleeping` does) **and** `sleep_start`/`sleep_end` are configured, the
half-life alternates: purpose base `half_life` (1200s) *inside* the sleep window (local `HH:MM:SS`,
overnight windows like `23:00→07:00` handled via `start_time > end_time`), `awake_half_life` (620s)
*outside* it — a bedroom holds "occupied" through sleep but clears within ~45min (4.3×620s) once up.

**Custom-override rule — SETTLED, PR #493 merged 2026-07-06 (main HEAD `17b71d2`).**
`Decay._resolve_purpose_half_life()` (`data/decay.py:81-119`) now has the guard:
`if self._base_half_life != self._purpose.half_life: return self._base_half_life` (line 90) — any
half-life differing from the purpose default is treated as a deliberate override, so a **custom**
half-life set for a Bedroom (anything other than the built-in 1200s) skips the sleep/awake switch
entirely and is respected as-is. Before the fix (issue #481, now closed), a custom Bedroom
half-life was still silently switched to `awake_half_life=620` outside the sleep window. Matches
the maintainer's #2 named historical failure class ("decay half-life config bugs") — see
`aod-failure-archaeology` for the full saga.

## 4. Likelihoods per sensor type — `P(evidence|occupied)` / `P(evidence|not-occupied)`

`prob_given_true` = `P(sensor active | area occupied)`. `prob_given_false` = `P(sensor active |
area NOT occupied)`. Defaults live in `data/entity_type.py::DEFAULT_TYPES` (not exhaustive below —
read the file for all ~20 types, including the environmental sub-types' `active_range` tuples):

| InputType | weight | prob_given_true | prob_given_false | strength_multiplier |
|---|---:|---:|---:|---:|
| MOTION | 1.0 | 0.95 | 0.005 | 3.0 |
| SLEEP | 0.9 | 0.95 | 0.02 | 3.0 |
| MEDIA | 0.85 | 0.65 | 0.02 | 2.0 |
| COVER | 0.5 | 0.35 | 0.02 | 2.0 |
| APPLIANCE | 0.4 | 0.2 | 0.02 | 2.0 |
| POWER | 0.3 | 0.2 | 0.02 | 2.0 |
| DOOR | 0.3 | 0.2 | 0.02 | 2.0 |
| WINDOW | 0.2 | 0.2 | 0.02 | 2.0 |
| environmental sub-types (temperature, humidity, illuminance, co2, co, ...) | 0.1 | 0.09 | 0.01 | 2.0 |
| UNKNOWN | 0.85 | 0.15 | 0.03 | 2.0 |

**Learning** (replacing these defaults with data): `db/correlation.py::analyze_binary_likelihoods()`
(lines 324-620) computes duration-weighted likelihoods directly from occupied-interval overlap:

```
prob_given_true  = seconds_active_while_occupied / total_seconds_occupied
prob_given_false = seconds_active_while_unoccupied / total_seconds_unoccupied
```

then clamps both to `[0.05, 0.95]` (`db/correlation.py:583-588` — deliberately avoids "black hole"
values of exactly 0 or 1 that would dominate the logit sum with `±∞`-adjacent contributions). If
the sensor was never active during any occupied interval, the function returns `analysis_error`
instead of a clamped near-zero value, so the entity falls back to the `EntityType` default rather
than learning a spuriously tiny likelihood from limited data (`db/correlation.py:547-560`).
Continuous/numeric sensors instead use Pearson correlation (`calculate_pearson_correlation()`,
`db/correlation.py:82`) gated by `MIN_CORRELATION_SAMPLES = 50` (const.py:323) — below 50 samples,
analysis is skipped and type defaults are used.

## 5. Wasp-in-box state logic

`WaspInBoxSensor` (`binary_sensor.py:169-612`) is a virtual binary sensor built from door + motion
state, for rooms (typically bathrooms) where a single motion sensor can't see the whole space:

- **Turns ON** when all configured doors are closed AND (motion is currently active OR motion was
  active within `motion_timeout` seconds — default `300`s, `data/config.py` /
  `DEFAULT_WASP_MOTION_TIMEOUT`) — `_process_door_state()`/`_process_motion_state()`,
  `binary_sensor.py:498-588`.
- **Turns OFF immediately** the instant *any* door opens while the room was occupied — regardless
  of motion state (`binary_sensor.py:521-525`). "Wasp trapped in a box": once the door (the box's
  only opening) closes with someone inside, they're assumed present until it opens again.
- Registered into the area as a **`MOTION`-type input entity** — `AreaConfig.get_motion_sensors()`
  appends the wasp entity's ID to the motion sensor list when `wasp_in_box.enabled`
  (`data/config.py:303-344`), so it uses `InputType.MOTION`'s weight/likelihood defaults, not a
  dedicated "wasp" likelihood.
- **Decay is forced to near-zero** (`half_life = 0.1`s) when `area.wasp_entity_id == entity_id`
  (`data/entity.py:769-774, 865-870`) — the sensor's own state already flips OFF the instant the
  door opens, so there's no purpose-based half-life to apply; the ~0.5s decay just smooths the
  edge. Sleep/purpose semantics are bypassed entirely for wasp entities (`purpose_for_decay = None`).
- Config knobs: `motion_timeout` (300s default), `max_duration` (3600s default — safety timeout
  that forces the sensor back off), `verification_delay` (0 = disabled by default), `weight` (0.8
  default) — all under `AreaConfig.wasp_in_box` (`data/config.py`, `const.py:363-366`).

## 6. Adjacent-areas math — Bayesian boost + decay modifier

**MERGED to `main` 2026-07-06 (PR #454, main HEAD `17b71d2`).** `feat/adjacent-areas` is complete
and live: `data/adjacency.py`, `data/trajectory.py`, `db/transitions.py` all exist on `main`, and
`Area.probability()` (`area/area.py`) calls `apply_logit_boost()` unconditionally as step 2 after
the activity boost — no feature flag or unmerged-branch caveat remains. Adjacency itself is still
labeled a **candidate**, not validated against real households, pending real-data tuning of the
gain/threshold constants below. Design rationale: discussion #431 (PR #456 was closed as merged
into #454).

### Boost — `compute_adjacency_boost()` (`data/adjacency.py:122-186`)

```
boost = gain × logit(P(target_area | trajectory, hour_of_week))     # gain = ADJACENCY_BOOST_GAIN = 0.5
```
Applied post-Bayesian, pre-decay, via `apply_logit_boost()` (`data/adjacency.py:189-204`):
`new_probability = sigmoid(logit(base_probability) + boost)`. `trajectory` is the household's
2-hop recent-area-exit history (`data/trajectory.py::TrajectoryTracker`); `P(target|trajectory,hour)`
comes from the six-level lookup in §7. No boost fires if there's no recent trajectory
(`trajectory.prev_area is None`).

**Worked example**: `gain=0.5`, learned `P(target|trajectory,hour) = 0.7` (specific chain, well
observed), base sensor-only probability `= 0.50`:
```
contribution = 0.5 × logit(0.7) = 0.5 × ln(0.7/0.3) = 0.5 × 0.8473 = 0.4237
new_logit = logit(0.50) + 0.4237 = 0 + 0.4237
new_probability = sigmoid(0.4237) ≈ 0.6042
```
The area's probability is nudged from 0.50 to **≈ 0.604** purely from "the household usually comes
here next."

### Decay modifier — `compute_decay_modifier()` (`data/adjacency.py:210-311`)

```
silence_score = Σ_{X ∈ adjacent(target)} (1 − P_X_lagged) × P(target → X | trajectory, hour)
decay_modifier = min(1 + gain × silence_score, cap)     # gain = 0.75, cap = 1.75
effective_half_life = base_half_life × decay_modifier
```
`P_X_lagged` is neighbour X's probability from the *previous* tick (`coordinator.py:526-537`,
`lagged_probabilities`). `silence_score` clamps to `[0,1]` after summing (`data/adjacency.py:301-303`).
Intuition: an area whose only learned exit has gone quiet gets decay stretched toward the `1.75×`
cap (they probably didn't leave, just went still); many divergent exits → smaller stretch.

**Worked example**: target area has two adjacent neighbours: hallway (`P_lagged=0.1`,
`P(target→hallway)=0.6`) and kitchen (`P_lagged=0.05`, `P(target→kitchen)=0.2`):
```
silence_score = (1−0.1)×0.6 + (1−0.05)×0.2 = 0.54 + 0.19 = 0.73
decay_modifier = min(1 + 0.75×0.73, 1.75) = min(1.5475, 1.75) = 1.5475
effective_half_life = 520s (Living Room base) × 1.5475 ≈ 805s
```
The room's decay half-life is stretched from 520s to **≈ 805s** because both learned exits have
been quiet since the room's last evidence.

Constants (`const.py:189-221`, all first-pass values pending real-data tuning per the file's own
comment): `ADJACENCY_TRANSITION_WINDOW_S=60`, `ADJACENCY_RECENCY_HALF_LIFE_DAYS=30`,
`ADJACENCY_TRAJECTORY_WINDOW_S=300`, `ADJACENCY_BOOST_GAIN=0.5`,
`ADJACENCY_DECAY_MODIFIER_GAIN=0.75`, `ADJACENCY_DECAY_MODIFIER_MAX=1.75`.

## 7. Six-level smoothing fallback for transition lookups

`lookup_transition_probability()` (`db/transitions.py:573-651`) answers "`P(to_area | from_area,
mid_area, hour_of_week)`" by walking progressively wider (less-specific, more-populated) scopes
until one has enough observations to trust. The threshold is on **total observations at that
level**, not the specific `to_area` count — once trusted, an unobserved destination is a real
learned zero, not "no data yet."

| # | Level constant | Scope | Threshold constant | Value |
|---|---|---|---|---:|
| 1 | `LEVEL_2HOP_HOUR_OF_WEEK` | specific `W→X→Y` chain at exact (weekday,hour) | `ADJACENCY_N_SPECIFIC` | 5 |
| 2 | `LEVEL_2HOP_HOUR_OF_DAY` | same chain, weekdays collapsed to hour-of-day | `ADJACENCY_N_HOUR` | 20 |
| 3 | `LEVEL_2HOP_UNBUCKETED` | same chain, all time collapsed | `ADJACENCY_N_CHAIN` | 50 |
| 4 | `LEVEL_1HOP_HOUR_OF_WEEK` | fallback to 1-hop `X→Y` at exact (weekday,hour) | `ADJACENCY_N_SPECIFIC` | 5 |
| 5 | `LEVEL_1HOP_UNBUCKETED` | 1-hop `X→Y`, all time collapsed | `ADJACENCY_N_PAIR` | 20 |
| 6 | `LEVEL_STATIC_DEFAULT` | no threshold — always available | — | `DEFAULT_INFLUENCE_WEIGHTS["adjacent"] = 0.3` (`db/relationships.py:35`) |

If `mid_area=""` is passed (no 2-hop trajectory known yet), levels 1-3 are skipped entirely and the
lookup starts at level 4 (`db/transitions.py:606-608`).

**Worked example**: querying a specific 2-hop chain at the exact hour-of-week finds only 3 total
observations (below the level-1 threshold of 5) → falls through to level 2 (hour-of-day collapsed
across weekdays), which finds `observed=10` out of `total=25` (≥ 20, trusted):
`probability = 10/25 = 0.4`, `level = "2hop_hour_of_day"`, `observed_count=10`, `total_count=25`.

Storage convention (`db/transitions.py` docstring, lines 1-26): chain `W → X → Y` stores
`from_area=W` (oldest hop), `mid_area=X`, `to_area=Y` (target) — `mid_area=""` marks a 1-hop row.
Counts decay exponentially each cycle by `0.5 ^ (hours_since_last_run / (24 × ADJACENCY_RECENCY_HALF_LIFE_DAYS))`
(`_apply_recency_decay_in_session`, `db/transitions.py:240-267` — same half-life mechanic as §3
but on transition *counts*), so the model adapts as household patterns change instead of
accumulating forever.

## 8. Correlation analysis — basics

`db/correlation.py::analyze_correlation()` (line 633) computes Pearson correlation
(`calculate_pearson_correlation()`, line 82) between a numeric sensor's value stream and occupancy,
requiring `MIN_CORRELATION_SAMPLES = 50` (const.py:323) or the analysis is skipped
(`too_few_samples`). Binary sensors instead use the duration-overlap method in §4
(`analyze_binary_likelihoods()`) — a direct likelihood estimate, not a correlation coefficient.
Results bucket into `CorrelationType` (`STRONG_POSITIVE`, `POSITIVE`, `STRONG_NEGATIVE`,
`NEGATIVE`, `NONE`, `BINARY_LIKELIHOOD` — `data/entity_type.py:20-28`). For statistical
methodology and how this feeds accuracy work, see `aod-research-methodology` and
`aod-learning-accuracy-campaign` — this skill only anchors the formulas actually in code.

## Provenance and maintenance

Date-stamped: **2026-07-06** (post-merge sweep), integration version **2026.5.17**
(`pyproject.toml:7`, `manifest.json:20` — this is a released-version number only; none of the PRs
below are in a tagged release yet). Checked out branch: `main`, HEAD `17b71d2`. PRs #454
(adjacent-areas), #491 (global-prior quiet-tail fix), #492 (sleep unknown-presence), #493
(bedroom half-life override guard), #494 (README purpose link) are all merged into `main` as of
this sweep — verified with `git log --oneline -1` and per-fact `grep`s below rather than
`gh pr view`/`git merge-base` against unmerged branches.

Re-verification commands, one per volatile fact category:

```bash
# Which probability function is actually live (§0)
grep -rn "bayesian_probability\|sigmoid_probability" custom_components/area_occupancy/area/area.py custom_components/area_occupancy/utils.py

# Global prior period-end behavior — confirm #491's fix still holds (§2a)
grep -n "actual_period_end = " custom_components/area_occupancy/data/analysis.py

# Purpose half-life / min_prior table (§3)
sed -n '/PURPOSE_DEFINITIONS/,/^}/p' custom_components/area_occupancy/data/purpose.py

# Bedroom custom half-life override guard — confirm #493's fix still holds (§3)
grep -n "_base_half_life != self._purpose.half_life" custom_components/area_occupancy/data/decay.py

# Sensor-type likelihood defaults (§4)
sed -n '/^DEFAULT_TYPES/,/^}/p' custom_components/area_occupancy/data/entity_type.py

# Adjacent-areas constants and wiring (§6, §7)
grep -n "^ADJACENCY_" custom_components/area_occupancy/const.py
grep -n "apply_logit_boost" custom_components/area_occupancy/area/area.py

# Correlation sample-size floor (§8)
grep -n "MIN_CORRELATION_SAMPLES" custom_components/area_occupancy/const.py

# Confirm current branch / HEAD before trusting any date-stamped fact above
git branch --show-current && git log --oneline -1
```
