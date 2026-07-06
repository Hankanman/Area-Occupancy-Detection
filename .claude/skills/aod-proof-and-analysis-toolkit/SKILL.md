---
name: aod-proof-and-analysis-toolkit
description: Use when you are about to touch, review, or debug any probability/statistics math in Area Occupancy Detection (priors, likelihoods, decay half-lives, logit-space boosts, correlation confidence, timezone/interval bucketing) and need to hand-verify the numbers BEFORE trusting a PR, a bug report, or your own patch. Trigger on tasks like "does this prior look right", "is this decay curve correct", "why did the prior pin at 0.99", "check this ratio/denominator", "audit this new coupling between areas for feedback loops", or any request to prove a calculation from first principles rather than just running the test suite.
---

# AOD Proof-and-Analysis Toolkit

## What this covers

Seven first-principles verification recipes for this repo's probability math: hand-computing
a prior from raw intervals, checking a logit-space boost's algebra, auditing a decay curve,
detecting UTC/local timezone mixing, judging whether learned data is statistically sufficient
to trust, auditing any ratio's numerator/denominator for window consistency, and analyzing a
new area-to-area coupling for self-reinforcing feedback. Each recipe is a runnable check, not
just a formula — copy the Python one-liners, run them, compare to the worked example. Use this
skill whenever you need to **prove** a number is right, not just observe that a test passed
(tests can encode the same bug they're supposed to catch — see Recipe 1).

**When NOT to use this**: for the underlying formulas' definitions and *why* they're shaped
this way, see `bayesian-occupancy-reference`. For the day-to-day campaign of collecting real-home
data and tuning constants, see `aod-learning-accuracy-campaign`. For a narrative of past bugs and
their fixes (more sagas, less how-to), see `aod-failure-archaeology`. For open research questions
beyond the fixes below, see `aod-research-frontier`.

---

## Recipe 1 — Hand-compute a prior from raw intervals

**What it proves**: that `global_prior = occupied_duration / observation_period` uses a
period that actually reflects reality, not a truncated window.

**Steps**:
1. Find every occupied interval `(start, end)` for the area in the lookback window.
2. Compute `occupied_duration = Σ (end - start)` in seconds.
3. Compute the observation period as `(first_interval_start, actual_period_end)`. As of
   2026-07-06, `actual_period_end` on `main` is picked with a bug (see worked example below);
   the correct definition is always `actual_period_end = now`.
4. `prior = clamp(occupied_duration / actual_period_duration, 0.01, 0.99)`.
5. Sanity-check against the reporter's real-world estimate. If the computed prior is pinned at
   0.99 (or 0.01) while a human says "this room is occupied ~30% of the time," the period
   window is wrong — go straight to Recipe 6.

**Worked example — issue #483 / PR #491** (`custom_components/area_occupancy/data/analysis.py:513-520`,
test `tests/test_data_analysis.py::test_valid_calculation_sets_correct_prior`):

Fixture: one occupied interval `(now - 8h, now - 6h)` — 2 hours occupied, and the area has been
quiet for 6 hours since.

| Quantity | Buggy (current `main`, PR #491 still open) | Correct (PR #491's fix) |
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

## Recipe 2 — Logit-space algebra check for a boost

**What it proves**: that a "boost" (any additive adjustment applied in log-odds space) produces
the probability shift you think it does, given its stated gain.

**Background you need**: `logit(p) = ln(p/(1-p))`, clamped to `p ∈ [0.01, 0.99]` before taking
the log so it never diverges (`custom_components/area_occupancy/utils.py:123-133`,
`clamp_probability` at `utils.py:45-75`). `sigmoid(z) = 1/(1+e^-z)` inverts it. Any boost of the
form `new_logit = logit(base) + contribution` then `new_prob = sigmoid(new_logit)` is "logit-space"
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
constant `ADJACENCY_BOOST_GAIN = 0.5` at `const.py:206`). **This feature is on the `feat/adjacent-areas`
branch / PR #454, merging as of 2026-07-06 — verify current status with `gh pr view 454` before
relying on file paths; it is not on `main` yet.**

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

## Recipe 3 — Decay curve audit

**What it proves**: that `decay_factor` at a given elapsed time matches `0.5^(age/half_life)`,
and that the 5% floor cutoff fires at the right elapsed time.

**Formula** (`custom_components/area_occupancy/data/decay.py:148-152`):
```
factor = 0.5 ** (age_seconds / half_life)
if factor < 0.05: return 0.0   # "practical zero" floor
```

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
user actually set), since PR #493 (open as of 2026-07-06, fixing issue #481) exists precisely
because this distinction was collapsed for Bedroom areas outside the sleep window.

**When to run this**: whenever a user reports a room "clearing" (occupancy flipping to
unoccupied) faster or slower than expected; before changing any purpose's default half-life in
`data/purpose.py`; before merging any change to `Decay.decay_factor` or `Decay.half_life`; when
auditing whether a custom half-life is actually being honored (compare `_base_half_life` to the
purpose default — see issue #439/#481 pattern in `aod-failure-archaeology`).

---

## Recipe 4 — Interval/timezone audit (UTC vs local mixing)

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

## Recipe 5 — Statistical sufficiency: is this learned number trustworthy?

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
— **on `feat/adjacent-areas` / PR #454, merging as of 2026-07-06, not yet on `main`; verify with
`gh pr view 454`**): `lookup_transition_probability()` walks from most-specific to least-specific,
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

## Recipe 6 — Denominator/period reasoning (auditing any ratio)

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

## Recipe 7 — Feedback-loop analysis for area-to-area couplings

**What it proves**: that a new coupling between areas (or between an area and its own history)
cannot amplify itself into a runaway loop within a single computation tick.

**Why this matters here specifically**: the adjacency feature (PR #454, merging as of 2026-07-06,
not yet on `main` — verify with `gh pr view 454`) is the first place this integration lets one
area's probability influence another's. Any coupling of this shape is a feedback-loop risk by
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

## Provenance and maintenance

Date-stamped: 2026-07-06, integration version 2026.5.17 (`custom_components/area_occupancy/manifest.json:20`,
`pyproject.toml:7`, `const.py:32` as `DEVICE_SW_VERSION`). All facts in this skill were verified
directly against the repository at this commit (`704c89e`, branch `feat/adjacent-areas` — a
28-commit-ahead-of-`main` superset per PR #454) unless marked "unverified."

Adjacency-feature facts (Recipes 2, 5-table-B, 7) describe code that lives on `feat/adjacent-areas`
/ PR #454 and is **not yet on `main`** as of 2026-07-06. Re-verify merge status before relying on
exact file paths:
```bash
gh pr view 454 --json state,mergeable,statusCheckRollup
git show main:custom_components/area_occupancy/const.py | grep -c ADJACENCY_   # 0 = still unmerged
```

Prior quiet-tail fix (Recipe 1, 6) is PR #491, also open as of 2026-07-06:
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
| PR #454 (adjacency) status | `gh pr view 454 --json state,mergeable` |
| PR #491 (prior fix) status | `gh pr view 491 --json state,mergeable` |
| PR #493 (bedroom half-life) status | `gh pr view 493 --json state,mergeable` |
| Timezone policy doc | `sed -n '1,10p' custom_components/area_occupancy/time_utils.py` |
| Six-level transition fallback | `sed -n '573,652p' custom_components/area_occupancy/db/transitions.py` |
| Issue #301 (timezone precedent) | `gh issue view 301 --json state,comments` |
