---
name: aod-math-reference
description: Load when you need the actual math behind Area Occupancy Detection's probability engine — logit-space evidence combination, prior composition (global/time/purpose-floor), exponential decay half-lives, the sleep awake/asleep half-life switch, wasp-in-box state logic, adjacent-areas Bayesian boost/decay-modifier, or the transition-lookup smoothing fallback. Use before touching utils.py, data/prior.py, data/decay.py, data/purpose.py, data/adjacency.py, data/trajectory.py, db/transitions.py, or db/correlation.py, or before answering "why does this area's probability say X" / "why is decay too fast/slow" / "why is the prior pinned near 0.99" questions. Also use when you are about to touch, review, or debug any probability/statistics math in Area Occupancy Detection (priors, likelihoods, decay half-lives, logit-space boosts, correlation confidence, timezone/interval bucketing) and need to hand-verify the numbers BEFORE trusting a PR, a bug report, or your own patch. Trigger on tasks like "does this prior look right", "is this decay curve correct", "why did the prior pin at 0.99", "check this ratio/denominator", "audit this new coupling between areas for feedback loops", or any request to prove a calculation from first principles rather than just running the test suite.
---

# AOD Math Reference

## What this covers

The exact formulas this repo uses to turn sensor evidence into an occupancy probability —
logit-space evidence combination, the three-part prior (global + time + purpose floor),
exponential decay with purpose half-lives and the sleep awake/asleep switch, wasp-in-box state
logic, the adjacent-areas Bayesian boost and decay modifier, and the six-level transition
smoothing fallback. Every formula is quoted from the file/line it lives in, with one
hand-computed worked example each. Read this before changing any of these files — the
maintainer's unwritten law #1 is **no silent math changes**.

Plus seven first-principles verification recipes for this repo's probability math:
hand-computing a prior from raw intervals, checking a logit-space boost's algebra, auditing a
decay curve, detecting UTC/local timezone mixing, judging whether learned data is statistically
sufficient to trust, auditing any ratio's numerator/denominator for window consistency, and
analyzing a new area-to-area coupling for self-reinforcing feedback. Each recipe is a runnable
check, not just a formula — copy the Python one-liners, run them, compare to the worked example.
Use these whenever you need to **prove** a number is right, not just observe that a test passed
(tests can encode the same bug they're supposed to catch — see Recipe 1).

## When NOT to use this

For *why* a specific fix was made historically (root causes of past bugs), use
`aod-debugging-and-history`. For which PRs are open/pending and how to evaluate learning-accuracy
regressions on real data, use `aod-debugging-and-history`. For config keys/flags that drive
these formulas, use `aod-architecture-and-config`. For the module dependency graph and where these files
sit in the pipeline, use `aod-architecture-and-config`. For the day-to-day campaign of collecting
real-home data and tuning constants, see `aod-debugging-and-history`. For a narrative of past
bugs and their fixes (more sagas, less how-to), see `aod-debugging-and-history`. For open research
questions beyond the fixes below, see `aod-research-frontier`.

---

## Formulas: the probability engine

### 0. There used to be two "Bayesian" functions — only one survived

`utils.py` used to define **two** probability-combination functions, only one of which was ever
wired into production.

| Function | Space | Status | Called from |
|---|---|---|---|
| `sigmoid_probability()` (utils.py:136) | **logit space**, additive weighted terms | **LIVE** — this is the actual engine, and has been since PR #353 (2026.2.1) | `presence_probability()`, `environmental_confidence()` → `area/area.py::Area._base_probability()` → `Area.probability()` → `sensor.py` state |
| `bayesian_probability()` | log-probability space, per-entity log-likelihood accumulation (mathematically a weighted naive-Bayes update) | **DELETED** (PR #529, 2026.8.1) — had zero production call sites for its entire lifetime; kept dormant as a revert safety net through the 2026.7.1 GA release, then removed along with its three private helpers once that window elapsed | historical only |

Verified (2026.8.1): `grep -rn "bayesian_probability" custom_components/ tests/` returns nothing —
the function and every reference to it are gone. If you see it mentioned elsewhere (an old PR, a
stale comment, another skill's cached text), it's describing history, not current code.

**Why this mattered**: before the removal, patching `bayesian_probability()` in response to "fix
the Bayesian calculation" would have changed nothing in the running integration — AGENTS.md's
"Modifying Bayesian Calculation" section used to point at it. AGENTS.md's equivalent section is now
named "Modifying Probability Calculation" and points at `sigmoid_probability()` and its callers
directly.

### 1. Logit space: what it is and why the engine uses it

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

#### The live formula — `sigmoid_probability()` (utils.py:136-199)

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
- `combined_probability(presence, environmental)` (utils.py:267) applies environmental as an
  **additive logit-space update damped to 20%**: `z = logit(presence) + 0.2×logit(environmental)`.
  Called from `Area._base_probability()` (`area/area.py:216`) — short-circuited (returns
  `presence` directly) when there are zero environmental entities configured, which is exactly
  what the formula yields anyway since `logit(0.5) == 0` (`area/area.py:240-241`).
  **This was an 80/20 weighted average until 2026-07 (PR "additive environmental evidence").**
  Averaging pulled the result toward the less-confident channel, and the environmental channel is
  structurally low-confidence (one env sensor contributes ~0.018 logits at defaults: `weight 0.1
  × prob_given_true 0.09 × strength_multiplier 2.0`), so supporting environmental data *lowered*
  the probability — one motion sensor gave 94.5%, adding a CO2 sensor at a normal 450 ppm dropped
  it to 90.8%. Environmental contributions are non-negative by construction (evidence ∈ [0,1],
  weights ≥ 0, learned correlations clamped to [0,1] at `db/correlation.py:1537,1548`), so
  `environmental_confidence()` ≥ 0.5 always and the additive form can only raise or hold.

After `_base_probability()`, `Area.probability()` (`area/area.py:245-274`) applies, in order:
1. **Activity boost** (`apply_activity_boost`, utils.py:293) — if `detect_activity()` finds a
   strong activity (TV, shower, cooking), `z = logit(base) + activity_boost × activity_confidence`,
   then sigmoid back. Boost magnitudes are `ACTIVITY_BOOST_HIGH=1.5` (showering/bathing/sleeping),
   `_STRONG=1.2` (TV), `_MODERATE=1.0` (cooking/working), `_MILD=0.8` (music/eating) — const.py:167-170.
2. **Adjacency boost** (`apply_logit_boost`, see §6) — merged 2026-07-06 (PR #454), always applied.

#### Worked example — motion sensor lifts a low prior

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

### 2. Prior composition

Three layers, combined in `data/prior.py::Prior._compute_value_and_floor()` (lines 97-140):
`learned = combine(global_prior, time_prior)`, then raised (never lowered) by whichever floor
(`purpose.min_prior` or `config.min_prior_override`) is higher, capped below the area's threshold.

#### 2a. Global prior — `occupied_seconds / period_seconds`

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
  historical bugs (issue #483, now closed). See `aod-debugging-and-history` for the full saga; this
  section only tracks the live formula.
- Fallback: invalid bounds, clock skew, or non-positive duration → `global_prior = 0.01` (hardcoded
  literal, same value as but independent of `MIN_PRIOR` — `data/analysis.py:472,494,541,565`).

**Worked example**: an area has 5 hours occupied over the last 2 days (172,800 s), and the last
interval ended <1h ago so both old and new code agree `actual_period_end = now`:
`global_prior = 5×3600 / 172800 = 18000/172800 ≈ 0.1042`.

#### 2b. Time prior — 168 day-hour buckets, LOCAL time

`Prior.time_prior` (`data/prior.py:162-174`) looks up `(day_of_week, time_slot)` in a cache of all
168 buckets loaded via `db.get_all_time_priors()`. `day_of_week` is `to_local(dt_util.utcnow()).weekday()`
(0=Monday..6=Sunday); `time_slot = (hour*60+minute) // 60` i.e. plain hour-of-day 0-23
(`DEFAULT_SLOT_MINUTES = 60`, `data/prior.py:38,181-185`). **Local time, not UTC** — deliberate:
bucketing by UTC hour would silently shift every learned bucket by 1 hour across a DST transition,
corrupting months of learned data twice a year. `to_local()`/`to_utc()` live in `time_utils.py`;
timezone/DST bugs are the maintainer's #1 named historical failure class (see
`aod-debugging-and-history`). Values are bounded to
`[TIME_PRIOR_MIN_BOUND=0.03, TIME_PRIOR_MAX_BOUND=0.9]` (const.py:186-187) after cache load
(`data/prior.py:227-231`). Missing bucket → `DEFAULT_TIME_PRIOR = 0.5` (const.py:226).

#### 2c. Combining global + time prior — logit-space weighted average

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

#### 2d. Purpose floor and `min_prior_override` — why transit spaces need a floor

`_compute_value_and_floor()` (`data/prior.py:97-140`) raises `learned` (never lowers it) to the
higher of `purpose.min_prior` and `config.min_prior_override`, each **capped** at
`config.threshold − PRIOR_FLOOR_THRESHOLD_MARGIN` (`0.01`, const.py:183) so a floor alone can never
hold an area "occupied" above threshold with zero active evidence (issue #435 — see
`aod-debugging-and-history`).

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

### 3. Decay: exponential half-life with a practical-zero cutoff

`Decay.decay_factor` (`data/decay.py:118-152`):

```
factor = 0.5 ^ (age / half_life)     # age = seconds since decay_start
if factor < 0.05: return 0.0          # practical-zero cutoff
```

`age` is `dt_util.utcnow() − decay_start`, always UTC (no local-time conversion needed here — pure
elapsed duration). `tick()` (`data/decay.py:154-170`) is what actually stops `is_decaying` once the
factor drops below the 5% floor, called from the coordinator's 10-second decay timer (per
AGENTS.md's Timers section).

#### Purpose-based half-lives (`data/purpose.py:160-236`) — full table

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

#### Worked example — half-life decay curve

Living Room (`half_life = 520`s), motion stops at `t=0`:
- `age=260` (half a half-life): `factor = 0.5^(260/520) = 0.5^0.5 ≈ 0.7071`
- `age=1560` (3 half-lives): `factor = 0.5^3 = 0.125` — still counted as evidence
- `age≈2247` (`520 × log2(20) ≈ 2247`s ≈ 37.5 min): `factor ≈ 0.050` — right at the cutoff
- `age=2250`: `factor ≈ 0.0498 < 0.05` → **returns 0.0**, `tick()` sets `is_decaying=False`

General rule: practical-zero is reached at `half_life × log2(20) ≈ half_life × 4.32`, i.e. about
4.3 half-lives after the last evidence, regardless of which purpose's half-life is in play.

#### The SLEEPING awake/asleep switch and the custom-override rule (PR #493)

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
`aod-debugging-and-history` for the full saga.

### 4. Likelihoods per sensor type — `P(evidence|occupied)` / `P(evidence|not-occupied)`

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

### 5. Wasp-in-box state logic

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

### 6. Adjacent-areas math — Bayesian boost + decay modifier

**MERGED to `main` 2026-07-06 (PR #454, main HEAD `17b71d2`).** `feat/adjacent-areas` is complete
and live: `data/adjacency.py`, `data/trajectory.py`, `db/transitions.py` all exist on `main`, and
`Area.probability()` (`area/area.py`) calls `apply_logit_boost()` unconditionally as step 2 after
the activity boost — no feature flag or unmerged-branch caveat remains. Adjacency itself is still
labeled a **candidate**, not validated against real households, pending real-data tuning of the
gain/threshold constants below. Design rationale: discussion #431 (PR #456 was closed as merged
into #454).

#### Boost — `compute_adjacency_boost()` (`data/adjacency.py:122-186`)

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

#### Decay modifier — `compute_decay_modifier()` (`data/adjacency.py:210-311`)

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

### 7. Six-level smoothing fallback for transition lookups

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

### 8. Correlation analysis — basics

`db/correlation.py::analyze_correlation()` (line 633) computes Pearson correlation
(`calculate_pearson_correlation()`, line 82) between a numeric sensor's value stream and occupancy,
requiring `MIN_CORRELATION_SAMPLES = 50` (const.py:323) or the analysis is skipped
(`too_few_samples`). Binary sensors instead use the duration-overlap method in §4
(`analyze_binary_likelihoods()`) — a direct likelihood estimate, not a correlation coefficient.
Results bucket into `CorrelationType` (`STRONG_POSITIVE`, `POSITIVE`, `STRONG_NEGATIVE`,
`NEGATIVE`, `NONE`, `BINARY_LIKELIHOOD` — `data/entity_type.py:20-28`). For statistical
methodology and how this feeds accuracy work, see `aod-change-and-validation` and
`aod-debugging-and-history` — this skill only anchors the formulas actually in code.

### Provenance and maintenance (formulas)

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

---

## Verification recipes: proving a number is right

---

### Recipe 1 — Hand-compute a prior from raw intervals

**What it proves**: that `global_prior = occupied_duration / observation_period` uses a
period that actually reflects reality, not a truncated window.

**Steps**:
1. Find every occupied interval `(start, end)` for the area in the lookback window.
2. Compute `occupied_duration = Σ (end - start)` in seconds.
3. Compute the observation period as `(first_interval_start, actual_period_end)`. As of PR #491
   (merged 2026-07-06), `actual_period_end` on `main` is always `now` (see worked example below
   for the bug this replaced).
4. Compute the prior using the additive global-prior formula — see § 2a above (Formulas section).
5. Sanity-check against the reporter's real-world estimate. If the computed prior is pinned at
   0.99 (or 0.01) while a human says "this room is occupied ~30% of the time," the period
   window is wrong — go straight to Recipe 6.

**Worked example — issue #483, fixed by PR #491 (merged 2026-07-06)**
(`custom_components/area_occupancy/data/analysis.py:513-520`,
test `tests/test_data_analysis.py::test_valid_calculation_sets_correct_prior`):

Fixture: one occupied interval `(now - 8h, now - 6h)` — 2 hours occupied, and the area has been
quiet for 6 hours since.

| Quantity | Buggy (pre-fix, before PR #491) | Correct (current `main`, PR #491's fix) |
|---|---|---|
| `first_interval_start` | now − 8h | now − 8h |
| `actual_period_end` | `last_interval_end` = now − 6h (because `(now - last_interval_end) > 3600s` triggers a truncation branch) | `now` |
| `actual_period_duration` | 2h | 8h |
| `occupied_duration` | 2h | 2h |
| `prior` | 2h / 2h = 1.0 → clamped to **0.99** | 2h / 8h = **0.25** |

The buggy code drops the "quiet tail" (the 6 hours of known non-occupancy since the interval
ended) from the denominator every time an area has been quiet more than 1 hour — which is
every night, every workday, every weekend. Each hourly pipeline run during a quiet stretch
recomputes the prior over a shrinking window, so it walks toward 1.0/0.99 monotonically. This
is why a kitchen with a true 28–35% occupancy rate was observed pinned at 0.99 (the original
bug report).

**Run it yourself**:
```python
from datetime import timedelta
now = ...  # dt_util.utcnow() equivalent
occupied_start = now - timedelta(hours=8)
occupied_end = now - timedelta(hours=6)
occupied_duration = (occupied_end - occupied_start).total_seconds()  # 7200

# buggy: truncates because (now - occupied_end) = 6h > 3600s
buggy_period = (occupied_end - occupied_start).total_seconds()        # 7200 -> prior 1.0/0.99
# correct: always now
correct_period = (now - occupied_start).total_seconds()               # 28800 -> prior 0.25
print(occupied_duration / buggy_period, occupied_duration / correct_period)
```

**When to run this**: before trusting any prior-related bug report ("prior stuck at 0.99/0.01",
"prior doesn't match how often I'm actually in the room"); before merging any change to
`PriorAnalyzer.calculate_and_update_prior()`; whenever a test asserts a specific prior value —
recompute it by hand first, because a test can encode the bug it's meant to catch (this is
exactly what happened here: `test_valid_calculation_sets_correct_prior` asserted `0.99` for
years before anyone hand-checked it).

---

### Recipe 2 — Logit-space algebra check for a boost

**What it proves**: that a "boost" (any additive adjustment applied in log-odds space) produces
the probability shift you think it does, given its stated gain.

**Background you need**: `logit`/`sigmoid`, their `[0.01, 0.99]` clamp, and why boosts are applied
in log-odds space — see § 1 above (Formulas section). Any boost of the form
`new_logit = logit(base) + contribution` then `new_prob = sigmoid(new_logit)` is "logit-space"
because it adds in log-odds, not probability — a `+1.0` contribution is a much bigger swing
near `p=0.5` than near `p=0.9`.

**Steps**:
1. Identify the contribution formula and its gain constant.
2. Compute `logit(reference_probability)` by hand (or Python).
3. Multiply by the gain to get `contribution`.
4. Add to `logit(base_probability)` and run through `sigmoid` to get the shifted probability.
5. Compare against the code's own diagnostic field (e.g. `logit_contribution` in the
   `BoostContribution` dataclass) if you have a live diagnostics dump.

**Worked example — adjacent-areas Bayesian boost** (`custom_components/area_occupancy/data/adjacency.py:122-204`,
constant `ADJACENCY_BOOST_GAIN = 0.5` at `const.py:206`). **This feature (PR #454, merged to `main`
2026-07-06) is now on `main` — the adjacency feature remains unvalidated on real homes, so treat
its constants as candidates, not settled tuning.**

The formula is `logit_contribution = gain × (logit(P) − logit(0.5))`. Since `logit(0.5) = 0`,
this reduces to `gain × logit(P)` (the centring term is documented as a deliberate no-op, kept
for clarity — see the comment at `adjacency.py:115-118`).

Say the learned transition probability `P` (that the household moves into this area from its
neighbour) is `0.9`, and the area's own base probability before the boost is `0.5`:

```python
import math
def logit(p): return math.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + math.exp(-z))

gain = 0.5
P = 0.9
contribution = gain * logit(P)        # 0.5 * ln(9) = 0.5 * 2.19722 = 1.09861

base = 0.5
new_prob = sigmoid(logit(base) + contribution)   # sigmoid(0 + 1.09861)
print(contribution, new_prob)   # 1.0986122886681098  0.75
```

Result: `logit(0.9) ≈ 2.1972`, gain-scaled contribution `≈ 1.0986`, and a `0.5` base probability
is pushed to exactly `0.75` — a fixed ~1.0986 logit-space nudge is a big swing at `p=0.5` (25
points) but shrinks fast near the clamped edges (`p=0.9 → ~0.947`, only +4.7 points) because
sigmoid saturates. **This nonlinearity is the whole point of doing it in logit space** — it means
the same boost can never push a confident "occupied" reading past ~0.99 or a confident "empty"
reading below ~0.01, but it can meaningfully swing an uncertain 0.5.

**When to run this**: before merging any change to a gain/weight constant that's applied in
logit space (`ADJACENCY_BOOST_GAIN`, `strength_multiplier`, activity `occupancy_boost`, the
`0.8/0.2` combined-probability weighting in `combined_probability()`); whenever a bug report says
"the boost seems too strong/weak" — hand-compute the actual percentage-point shift at the
probabilities in question, don't reason about the gain number in isolation; before trusting any
new "influence weight" or "confidence multiplier" config surface.

---

### Recipe 3 — Decay curve audit

**What it proves**: that `decay_factor` at a given elapsed time matches `0.5^(age/half_life)`,
and that the 5% floor cutoff fires at the right elapsed time.

**Formula**: `factor = 0.5 ** (age_seconds / half_life)`, floored to `0.0` once it drops below
`0.05` — see § 3 above (Formulas section, `data/decay.py:148-152`).

**Steps**:
1. Get `half_life` (seconds) and `age` (seconds since the last evidence / decay start).
2. Compute `0.5 ** (age / half_life)` by hand.
3. Compare to the code's live value (diagnostics `entities[].decay` field, or unit test table
   below).
4. If auditing "when does this practically stop mattering", solve for the cutoff:
   `age = half_life × ln(0.05)/ln(0.5) ≈ half_life × 4.3219`.

**Worked example** — from `tests/test_data_decay.py::test_decay_factor`, `half_life=60.0`:

| age (s) | age / half_life | expected `decay_factor` |
|---|---|---|
| 0 | 0.00 | 1.0 |
| 15 | 0.25 | 0.8409 |
| 30 | 0.50 | **0.7071** |
| 45 | 0.75 | 0.5946 |
| 60 | 1.00 | 0.5 |
| 90 | 1.50 | 0.3536 |
| 120 | 2.00 | 0.25 |
| 258 | 4.30 | 0.0501 (just above the floor) |
| 260 | 4.33 | 0.0 (below 0.05, floored) |

```python
print(0.5 ** (30/60))     # 0.7071067811865476
print(0.5 ** (258/60))    # 0.05068...
print(0.5 ** (260/60))    # 0.04943... -> below 0.05, code returns 0.0
import math
print(math.log(0.05)/math.log(0.5))   # 4.321928094887363 half-lives to cross the floor
```

At a 60s half-life, the floor trips at ≈259.3s (`60 × 4.3219`), matching the table's boundary
between 258s (still non-zero) and 260s (floored). **This 4.32-half-life constant is universal**
— it's independent of the actual half-life value, so you can sanity-check any half-life's
"effectively done decaying" time by multiplying by 4.32 (e.g. Bedroom's sleeping half-life of
1200s effectively floors at ~5186s ≈ 86 minutes).

**Special-cased half-lives to know when auditing** (`data/entity.py`, `data/purpose.py:160-236`,
`const.py:144-146`): Wasp-in-Box entities use `half_life = 0.1s` (effectively no decay — clears
in under half a second); sleep-presence virtual entities use `SLEEP_PRESENCE_HALF_LIFE = 7200s`
(2 hours, "persistent presence"); both bypass purpose/sleep-window switching entirely
(`purpose=None` is passed to `Decay`). The `SLEEPING` purpose additionally multiplies its
resolved half-life by an adjacency `_modifier_factor` (clamped ≥ 1.0, cap `ADJACENCY_DECAY_MODIFIER_MAX
= 1.75` — see Recipe 7) — when auditing a *reported* half-life against the *configured* one,
check `Decay.half_life` (the multiplied, effective value) vs `Decay._base_half_life` (what the
user actually set), since PR #493 (merged 2026-07-06, fixed issue #481) exists precisely
because this distinction was collapsed for Bedroom areas outside the sleep window — `main` now
carries the `_resolve_purpose_half_life()` guard (`!= purpose default → return base`) plus the
adjacency `modifier_factor` multiplying on top.

**When to run this**: whenever a user reports a room "clearing" (occupancy flipping to
unoccupied) faster or slower than expected; before changing any purpose's default half-life in
`data/purpose.py`; before merging any change to `Decay.decay_factor` or `Decay.half_life`; when
auditing whether a custom half-life is actually being honored (compare `_base_half_life` to the
purpose default — see issue #439/#481 pattern in `aod-debugging-and-history`).

---

### Recipe 4 — Interval/timezone audit (UTC vs local mixing)

**What it proves**: whether a reported anomaly (negative durations, off-by-N-hours bucketing,
priors that look shifted) is caused by mixing timezone-aware and naive/local datetimes.

**Policy this repo commits to** (`custom_components/area_occupancy/time_utils.py:1-8`):
- Runtime arithmetic/comparisons: timezone-aware **UTC**.
- Database persistence (SQLite): naive **UTC** (`tzinfo=None`, interpreted as UTC).
- Wall-clock bucketing (time priors, daily/weekly/monthly grouping): **HA local timezone**.

Any code path that accidentally does arithmetic between a naive-assumed-local value and an
aware-UTC value produces an error equal to the local UTC offset — and that's the discriminating
signature.

**The discriminating check**: a timezone-mixing bug shows up as an error that is a **suspiciously
round number of hours** — the reporter's UTC offset (e.g. exactly ±3h, ±5h, ±5.5h for
half-hour-offset zones), not a random float, and not exactly zero. Compare against the
reporter's timezone in Home Assistant (`hass.config.time_zone`) or the browser locale they
mention.

**Worked example — issue #301** ("Invalid period duration errors in 2025.12.2", closed
2025-12-29, fixed by 2025.12.4 alongside commit `3dcb6f1` "Implement timezone normalization and
local bucketing utilities" which introduced `time_utils.py`): logs showed
`Invalid period duration (-10800.00 seconds) for area Hallway` and
`(-10797.30 seconds)`. `-10800s = -3.0h` exactly; the second occurrence is ~3h plus a few seconds
of clock skew. The reporter confirmed their HA timezone was US Eastern. **The lesson to copy**:
a clean or near-clean multiple-of-3600-seconds error is the signature to search for — grep your
own reported "invalid duration"/"negative period" logs for values divisible by 3600 (or 1800 for
half-hour zones) before looking anywhere else. This exact bug class was closed by introducing
`time_utils.py` (commit `3dcb6f1`, 2025-12-12) as the single source of truth for UTC/local
conversion — if you find a *new* raw `datetime` subtraction outside that module, treat it as a
prime suspect.

**How to reproduce/detect this class of bug yourself**:
```python
from homeassistant.util import dt as dt_util
from datetime import timedelta

# Simulate what a naive-vs-aware subtraction looks like:
aware_now = dt_util.utcnow()
naive_stored = aware_now.replace(tzinfo=None) - timedelta(hours=3)  # e.g. read from SQLite as naive-local
# If code does `aware_now - naive_stored` it raises TypeError (good — caught fast).
# If code does `aware_now - naive_stored.replace(tzinfo=dt_util.UTC)` when naive_stored
# was actually local (not UTC), you get a silent 3h-shaped error instead of a crash — this is the
# dangerous case, because it doesn't raise, it just quietly biases every duration/bucket by
# the local UTC offset.
```

**Checklist for any new datetime-touching code**:
- [ ] Every persisted timestamp read from the DB passes through `from_db_utc()` before use.
- [ ] Every timestamp written to the DB passes through `to_db_utc()`.
- [ ] Every runtime comparison/subtraction operates on values that went through `to_utc()`.
- [ ] Bucketing (day-of-week, hour-of-day, daily/weekly rollups) explicitly calls `to_local()`
      — never buckets on raw UTC hour, or DST transitions shift every bucket by an hour twice a
      year.
- [ ] If you must iterate hour-by-hour across a DST boundary, iterate in UTC (fixed 3600s steps,
      no ambiguity) and derive the local bucket key only at the end — this is exactly what
      `PriorAnalyzer.calculate_time_priors()` does (`data/analysis.py:675-803`, comment at
      :702-704) specifically to avoid the repeated-local-hour ambiguity during fall-back DST.
- [ ] Any new duration/period calculation: sanity-check the result isn't a suspiciously round
      number of hours away from what you'd expect — that's the smoking gun.

**When to run this**: any bug report mentioning wrong times, negative/huge durations,
"prior looks shifted by X hours", "aggregation happens at the wrong hour", or anything that
reproduces differently for users in different timezones; before merging any new code that reads
raw datetimes from the DB or does datetime arithmetic outside `time_utils.py`'s existing helpers.

---

### Recipe 5 — Statistical sufficiency: is this learned number trustworthy?

**What it proves**: whether a learned probability/correlation/transition estimate has enough
observations behind it to act on, or whether it's noise dressed up as a number.

**The core idea, in one line**: as sample count `n` grows, an estimate's *precision* improves
roughly as `1/√n`, so somewhere below a project-specific floor the noise swamps the signal and
you should fall back to a wider/coarser default rather than trust the specific number. This
integration hard-codes that floor in **two structurally identical places**:

**A. Correlation analysis — `MIN_CORRELATION_SAMPLES = 50`** (`const.py:323`,
`db/correlation.py:99,1056,1541`):
- Below 50 samples: correlation isn't even computed (`return (0.0, 1.0)` — zero strength, p=1).
- At/above 50: `confidence = min(1.0, abs_correlation × (1 − 50/sample_count))`. At exactly 50
  samples, confidence is forced to 0 regardless of the raw correlation coefficient; it only
  approaches the raw coefficient asymptotically as `sample_count → ∞`.
- Reloading a previously-saved correlation re-checks `sample_count < 50` and discounts/ignores
  it if the count has since dropped (e.g. after a data purge) below the floor.

  ```python
  def confidence(abs_corr, n, floor=50):
      return min(1.0, abs_corr * (1 - floor / n)) if n >= floor else 0.0
  print(confidence(0.8, 50))    # 0.0   -- exactly at the floor, zero trust
  print(confidence(0.8, 100))   # 0.4   -- half-discounted
  print(confidence(0.8, 1000))  # 0.76  -- close to the raw 0.8
  ```

**B. Adjacency transition smoothing — the same idea, four thresholds gating six fallback levels**
(`custom_components/area_occupancy/db/transitions.py:487-651`, constants at `const.py:216-221`
— **PR #454, merged to `main` 2026-07-06**): `lookup_transition_probability()` walks from most-specific to least-specific,
using the first level whose **total observation count** clears its threshold:

| Level | Scope | Threshold constant | Value |
|---|---|---|---|
| 1 | 2-hop chain, exact hour-of-week | `ADJACENCY_N_SPECIFIC` | 5 |
| 2 | 2-hop chain, hour-of-day (weekdays collapsed) | `ADJACENCY_N_HOUR` | 20 |
| 3 | 2-hop chain, un-bucketed | `ADJACENCY_N_CHAIN` | 50 |
| 4 | 1-hop chain, exact hour-of-week | `ADJACENCY_N_SPECIFIC` | 5 |
| 5 | 1-hop chain, un-bucketed | `ADJACENCY_N_PAIR` | 20 |
| 6 | static default | — | no threshold (0 observations signalled) |

This is the *same* sufficiency principle as correlation's single 50-sample floor, just applied
per-bucket-width instead of globally: a narrower bucket (exact hour-of-week) needs fewer total
observations to be "specific enough to matter" (5) because a false read there is cheap and
gets diluted quickly by the wider fallback; a wider un-bucketed pool needs more (50) before
you'll trust it over falling all the way back to a static default, because there's no narrower
level left to catch an error.

**How to judge "is this learned number trustworthy" in general** — apply this order of checks:
1. **Count check (hard floor)**: is `sample_count`/`observed_count`/`total_count` at or above the
   relevant constant? If below, don't trust the number at all — it should already be gated out
   in code; if you see a raw learned value being used *despite* a sub-floor count, that's a bug.
2. **Confidence/discount check (soft floor)**: even above the hard floor, is the discount factor
   (e.g. correlation's `1 - 50/n`) still heavily attenuating the value? A value with n=52 and a
   raw correlation of 0.9 still only carries confidence `0.9 * (1 - 50/52) ≈ 0.035` — technically
   "computed" but practically noise.
3. **Variance/spread check**: for numeric (Gaussian) correlations, look at
   `std_occupied`/`std_unoccupied` relative to the gap between `mean_occupied` and
   `mean_unoccupied` — if the two distributions overlap heavily (means within ~1 std of each
   other), no sample count fixes that; use `scripts/visualize_distributions.py` to plot it
   directly rather than trusting `correlation_strength` blind.
4. **Time-coverage check** (priors specifically): `calculate_time_priors()` tracks
   `slot_weeks_total` (distinct ISO weeks contributing to each of the 168 day-of-week × hour-of-day
   buckets) as a diagnostic, but as of 2026-07-06 this is **not** used to gate whether a slot's
   prior is trusted — no minimum-weeks threshold is enforced before a slot prior is written. If
   you're investigating a noisy time-prior, check `data_points_per_slot` yourself; the code
   won't stop you from trusting a slot backed by a single week's data.

**When to run this**: whenever a bug report or PR claims "the learned prior/correlation/adjacency
influence is wrong" — first ask how many samples/observations back it, using the table above,
before assuming the math is broken; before lowering any `MIN_*`/`ADJACENCY_N_*` threshold (lowering
it trades stability for responsiveness — quantify the confidence at the new floor using the
formula above); when reviewing a PR that adds a new learned-from-history feature — it needs an
explicit sufficiency floor of its own, not an implicit "well it'll average out."

---

### Recipe 6 — Denominator/period reasoning (auditing any ratio)

**What it proves**: that a ratio's numerator and denominator cover the *same* time window —
the general bug class behind issue #483 (Recipe 1), but applicable to any `X / Y` in the
codebase, not just the global prior.

**The failure pattern, generalized**: someone narrows (or widens) one side of a ratio for a
legitimate-sounding reason ("guard against a degenerate startup period", "exclude the tail we
don't have data for yet") without symmetrically adjusting the other side. The ratio then silently
answers a different question than its name claims to answer.

**Checklist — apply to any ratio you're adding, reviewing, or debugging**:
- [ ] **Name the window explicitly.** Write down, in plain language, "numerator = X measured
      over window W; denominator = Y measured over window W" — the *same* `W`. If you can't
      state one shared `W`, the ratio is already suspect.
- [ ] **Trace both sides to their window-defining variables independently.** In #483, the
      numerator (`occupied_duration`) summed intervals over `[first_interval_start, now]` while
      the denominator (`actual_period_duration`) used `[first_interval_start, last_interval_end]`
      — different right edges. Find the equivalent left/right-edge variables for whatever ratio
      you're auditing and diff them.
- [ ] **Ask what happens during a real "boring" period** (quiet overnight, weekend, vacation).
      Boring periods are exactly when truncation logic tends to kick in ("no new data since
      X, so shrink the window") — and they're exactly when a numerator/denominator mismatch
      does the most damage, because they run every single hourly cycle, compounding.
- [ ] **Check whether "now" is used consistently.** Ratios computed against `dt_util.utcnow()`
      should use one captured `now` value throughout, not one `now()` call for the numerator and
      a later one for the denominator (races are small here, but the *conceptual* mismatch of
      "which `now`" is the same bug shape).
- [ ] **Look for an asymmetric guard clause.** Grep for `if` branches near the ratio that adjust
      *one* side's bound "defensively" — e.g. `if (now - last_interval_end).total_seconds() >
      3600: actual_period_end = last_interval_end` (the exact #483 line,
      `data/analysis.py:517-520`) truncates the denominator's end but never touches the
      numerator. A defensive clamp on one side without an equal clamp on the other is the
      signature to hunt for.
- [ ] **Re-run Recipe 1's style of hand computation** with concrete numbers before and after your
      change — if you can't produce a two-column "buggy vs correct" table like the one above,
      you haven't verified the fix.

**Other ratios in this codebase worth this treatment if you touch them**: correlation's
`abs_correlation × (1 − MIN_CORRELATION_SAMPLES/sample_count)` (Recipe 5 — numerator's
`abs_correlation` and denominator's `sample_count` must be computed over the same filtered sample
set); `information_gain = |pgt − pgf| / max(pgt, pgf, 0.01)` (`data/entity.py:309-327` — both
`pgt`/`pgf` must come from the same correlation run, not one stale and one fresh); the adjacency
`silence_score` sum (Recipe 7 — every neighbour's lagged probability and transition probability
must be from the *same* tick).

**When to run this**: before merging any change that computes a ratio/rate/percentage from two
independently-sourced quantities; whenever a learned value looks systematically biased in one
direction (inflated, deflated, always saturating) rather than just noisy — systematic bias is the
tell of a window mismatch, not randomness; as a mandatory step when reviewing any PR touching
`data/analysis.py`'s period/window calculations or `db/correlation.py`'s confidence math.

---

### Recipe 7 — Feedback-loop analysis for area-to-area couplings

**What it proves**: that a new coupling between areas (or between an area and its own history)
cannot amplify itself into a runaway loop within a single computation tick.

**Why this matters here specifically**: the adjacency feature (PR #454, merged to `main`
2026-07-06) is the first place this integration lets one area's probability influence another's —
it remains unvalidated on real homes, so treat it as a candidate feature under active scrutiny,
not settled behavior. Any coupling of this shape is a feedback-loop risk by
construction: if area A's boost depends on area B's *current* probability, and area B's boost
depends on area A's *current* probability, a single tick could see both areas inflate each other
before either settles — worse, over multiple ticks this could compound instead of converge.

**How this codebase avoids it — the lagged-snapshot pattern** (`coordinator.py:526-574`
`update()`, `:603-652` `_compute_adjacency_state`): every coordinator tick first snapshots the
**previous** tick's per-area probability/occupied state into `self._lagged_probabilities` /
`was_occupied` *before* computing anything new for the current tick. `compute_decay_modifier()`'s
`silence_score` and `TrajectoryTracker.observe()`'s end-edge detection both read exclusively from
this lagged snapshot — never from an in-flight, still-being-recomputed value. All areas'
adjacency boosts and decay modifiers are precomputed together in one pass
(`_compute_adjacency_state`) before any area's `probability()`/`half_life` is recomputed for the
tick, so no area's own recompute can feed back into its neighbours' inputs within that same tick.

**Checklist for auditing any new coupling (between areas, or between an entity and its own
derived state) for self-reinforcement**:
- [ ] **Identify the read**: what upstream value does the new logic consume (another area's
      probability, its own decayed state, a correlation computed earlier in the same pipeline
      run)?
- [ ] **Identify the write**: what does the new logic produce, and does that output feed back —
      directly or via a later pipeline step in the *same* run — into the value it just read?
- [ ] **Is the read lagged or live?** If it's live (this tick's freshly-computed value), you have
      a same-tick feedback risk. Change it to read the previous tick/previous pipeline-step's
      snapshot instead, mirroring the `_lagged_probabilities` pattern.
- [ ] **Is there a cap on the output regardless of input?** Even with lagging, an unbounded gain
      could let cross-tick oscillation grow slowly. Check for an explicit ceiling — the decay
      modifier's `cap = ADJACENCY_DECAY_MODIFIER_MAX = 1.75` (`const.py:214`) and the boost path's
      implicit ceiling (logit-space additions saturate through `sigmoid`/`clamp_probability` at
      `[0.01, 0.99]`, so no additive boost can push a probability outside that band — see Recipe
      2) are the two examples here. A new coupling without an analogous cap is under-specified.
- [ ] **Simulate two ticks by hand**: pick two coupled areas, assign starting probabilities, run
      one tick of your new formula using "tick 0" values for both, then run "tick 1" using the
      lagged "tick 0" outputs. Confirm the values move toward a fixed point, not away from one. If
      you're not sure how to compute a fixed point analytically, at least confirm empirically
      (in a quick Python script or the `simulator/` Flask app) that 10+ synthetic ticks converge
      rather than diverge or oscillate with growing amplitude.
- [ ] **Check the recency-decay term for transition counts** if the coupling learns from history:
      `ADJACENCY_RECENCY_HALF_LIFE_DAYS = 30` (`const.py:198`) exponentially decays old transition
      counts each pipeline run before adding new ones — this is what keeps the learned influence
      adapting to current patterns rather than accumulating unboundedly forever. Any new
      learned-history coupling needs an equivalent recency mechanism or it will ossify around
      whatever pattern existed when it first accumulated enough samples.

**When to run this**: before merging any PR that lets one area read another area's state
(current or historical); before merging any coupling between a value and its own past output
(e.g. a rolling average, an exponential smoother); whenever a bug report describes probability
"oscillating," "climbing without bound," or "two rooms bouncing off each other"; as a mandatory
design-review step for any future extension of the adjacency feature (multi-hop chains beyond
2-hop, weighted multi-neighbour boosts, etc.) or any other cross-area feature the roadmap adds
(e.g. "Occupancy Zone Hierarchies" from the README's Planned Features).

---

### Provenance and maintenance (recipes)

Date-stamped: 2026-07-06, integration version 2026.5.17 (`custom_components/area_occupancy/manifest.json:20`,
`pyproject.toml:7`, `const.py:32` as `DEVICE_SW_VERSION` — no tagged release yet carries today's
merge wave; this is still main's HEAD version). All facts in this skill were verified directly
against the repository at `main` HEAD (`17b71d2`, post-merge-wave of 2026-07-06) unless marked
"unverified."

Adjacency-feature facts (Recipes 2, 5-table-B, 7) describe code that merged to `main` via PR #454
on 2026-07-06 (#456 closed as merged into it). The feature is complete on `main` but remains
**unvalidated on real homes** — still a candidate for future tuning, not settled behavior.
Re-verify current state before relying on exact file paths:
```bash
git show main:custom_components/area_occupancy/const.py | grep -c ADJACENCY_   # >0 = merged, present
```

Prior quiet-tail fix (Recipe 1, 6) is PR #491, merged 2026-07-06 (closed issue #483):
```bash
gh pr view 491 --json state,mergeable,title
```

Re-verification commands for this skill's volatile facts:

| Fact category | Command |
|---|---|
| Version numbers | `grep -n version custom_components/area_occupancy/manifest.json pyproject.toml` |
| `MIN_CORRELATION_SAMPLES` value | `grep -n "MIN_CORRELATION_SAMPLES" custom_components/area_occupancy/const.py` |
| `ADJACENCY_*` constants | `grep -n "ADJACENCY_" custom_components/area_occupancy/const.py` |
| Decay floor / formula | `sed -n '140,155p' custom_components/area_occupancy/data/decay.py` |
| Prior period-truncation bug | `sed -n '510,525p' custom_components/area_occupancy/data/analysis.py` |
| Decay-curve test table | `grep -n "60.0" tests/test_data_decay.py` |
| PR #454 (adjacency) merge confirmation | `gh pr view 454 --json state,mergeable` |
| PR #491 (prior fix) merge confirmation | `gh pr view 491 --json state,mergeable` |
| PR #493 (bedroom half-life) merge confirmation | `gh pr view 493 --json state,mergeable` |
| Timezone policy doc | `sed -n '1,10p' custom_components/area_occupancy/time_utils.py` |
| Six-level transition fallback | `sed -n '573,652p' custom_components/area_occupancy/db/transitions.py` |
| Issue #301 (timezone precedent) | `gh issue view 301 --json state,comments` |
</content>
