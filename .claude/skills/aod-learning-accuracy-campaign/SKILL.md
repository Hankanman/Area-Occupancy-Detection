---
name: aod-learning-accuracy-campaign
description: Executable, decision-gated campaign for making learned priors and likelihoods (global prior, 168 time-priors, per-sensor P(E|H) from correlation analysis, decay half-life resolution) trustworthy on real homes. Load this when a user reports "prior stuck at 0.99/0.01", "occupancy probability doesn't match reality", "false transitions", "decay clears too fast/slow", "correlation/likelihood looks wrong", or when asked to touch data/prior.py, data/analysis.py's PriorAnalyzer, db/correlation.py, or data/decay.py half-life resolution. This is the flagship operation order for the project's hardest live problem — treat it as a runbook, not background reading.
---

## What this covers

The end-to-end loop for diagnosing and fixing bad learned values on a **real** Home Assistant install: pull ground truth, hand-verify the stored math, diagnose which stage is wrong (interval detection, denominator/period selection, bucketing, likelihood estimation, or decay half-life resolution), pick a fix from a theory-obligated menu, and validate before it ever reaches `main`. It exists because this project has shipped the same class of bug — a denominator or comparison subtly excluding data it shouldn't — at least five times (see Phase 2 and Provenance).

**When NOT to use this**: for the underlying formulas themselves (sigmoid/logit math, clamps, combine_priors derivation) use `bayesian-occupancy-reference`. For pulling a diagnostics export or DB copy off a real install, use `aod-diagnostics-and-tooling` (this skill assumes you already have both, see Phase 0). For the actual PR/merge process once a fix is validated, use `aod-change-control` — this skill never merges anything itself. For known-closed war stories (not open investigations), use `aod-failure-archaeology`. For the adjacent-areas / transition-learning subsystem specifically (merged 2026-07-06, PR #454), that is out of scope here — it consumes learned priors but doesn't produce them.

---

## Phase 0 — Ground truth harness

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

   Then compare `latest_cached` to `global_priors.data_period_end` for that area. **They should be very close to "now" at analysis time** — if `data_period_end` is stuck hours or days behind `latest_cached`/the current wall clock while the area has been quiet, you have re-found the #483 pattern (period truncated at `last_interval_end` instead of running to `now`). See `custom_components/area_occupancy/data/analysis.py` around the `calculate_and_update_prior` period-selection block — the correct behavior (period always ends at `now`, via `actual_period_end = now`) landed on `main` via PR #491, merged 2026-07-06 (fixes #483); re-verify with `git show main:custom_components/area_occupancy/data/analysis.py | grep -n "actual_period_end"` if you suspect a regression.

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

4. **If the stored value differs from your hand value**: you found a bug in the period-selection, bucketing, or denominator logic, not a rounding artifact — branch to `aod-debugging-playbook`, treating it as a "#483-class" bug (systematic denominator/period truncation), and come back here once fixed to re-verify before moving to Phase 1.

---

## Phase 1 — Baseline measurement

Define these metrics precisely before touching any code, and compute them for the specific real-home area under investigation.

| Metric | Definition | How to compute | Known-good range |
|---|---|---|---|
| Prior calibration error | `\|stored global_prior − true occupancy rate over the same window\|` | True rate: from an independent reference signal (e.g. manual log, a second known-reliable presence sensor) covering the *same* `data_period_start`→`data_period_end` window from `global_priors`. Stored: that row's `prior_value`. | No repo-wide target exists. The one documented real case (#483, kitchen mmWave) had true rate **28–35%** over 7 days while the buggy prior pinned at **0.99**; after PR #491's fix the same synthetic scenario in `tests/test_data_analysis.py::test_valid_calculation_sets_correct_prior` recomputes to **0.25** for 2h occupied / 8h period. Use "does the recomputed ratio match the true rate to within a few points" as the bar, not a repo-blessed number. |
| False-transition count | Number of times the area's `binary_sensor.<area>_occupancy` (via `Area.occupied()`, threshold-gated at `config.threshold`, default 0.50) flips state without a corroborating flip in your chosen reference signal, over N days | Pull `occupied`/`probability` history for the binary sensor from the recorder (or replay `intervals` + `global_priors`/`priors` through the simulator, see Phase 3 fenced-path warning) and diff transition timestamps against the reference signal's transitions, with a small tolerance window (a few seconds to a couple minutes, matching `motion_timeout` — `DEFAULT_MOTION_TIMEOUT = 300s`, `const.py:132`) | No repo-wide target exists; this is a per-area empirical baseline you establish, not a constant to look up. |
| Sample sufficiency | Number of samples correlation analysis had for this entity | `SELECT sample_count FROM correlations WHERE area_name=? AND entity_id=?;` | Must be ≥ `MIN_CORRELATION_SAMPLES = 50` (`const.py:323`) before you trust `prob_given_true`/`prob_given_false` for that entity at all — below 50, `db/correlation.py`'s gate returns early / discounts confidence (see Phase 2). |
| Time-prior data density | Weeks of data feeding a given `(day_of_week, time_slot)` bucket | `SELECT data_points FROM priors WHERE area_name=? AND day_of_week=? AND time_slot=?;` | `data_points` is a **diagnostic-only** field — `calculate_time_priors` (`data/analysis.py:675-803`) does NOT gate on it before writing a slot's prior. A bucket with `data_points = 1` is exactly as "trusted" by the code as one with `data_points = 12`. This is a real gap — see Phase 3's min-sample-gate proposal. |

Also record, for the area under investigation: `config.threshold`, `purpose`, whether it has a custom half-life override, and whether it's SLEEPING-purpose (the only purpose with an awake/sleep half-life split — see Phase 2).

---

## Phase 2 — Diagnosis tree

Work top-down; each branch has a discriminating query so you don't have to guess.

### If priors look wrong (calibration error large)

```
Is occupied_intervals_cache itself wrong (interval detection)?
├─ Query: SELECT * FROM occupied_intervals_cache WHERE area_name=? ORDER BY start_time DESC LIMIT 50;
│  Compare each interval's start/end against the raw `intervals` table for motion/sleep/media
│  entities in that window (get_occupied_intervals in db/queries.py merges motion+sleep+media,
│  applies motion timeout ONLY to motion intervals via apply_motion_timeout — a media-only
│  "occupied" interval never gets extended past its actual end).
│  → If cache intervals don't match raw sensor activity: bug is in
│    build_motion_query/build_presence_query/process_query_results (db/queries.py) or in
│    apply_motion_timeout/merge_overlapping_intervals (utils.py). Branch to
│    aod-debugging-playbook.
│
Is the DENOMINATOR/period wrong (the #483 bug class)?
├─ Query: compare global_priors.data_period_end to "now" at the time of the last analysis run
│  (coordinator.py's hourly analysis timer, ANALYSIS_INTERVAL=3600s, const.py:343).
│  → If data_period_end lags "now" by more than one analysis cycle while the area has valid
│    recent cache entries: this IS the #483 pattern re-emerging on a branch that predates or
│    regressed the fix. The fix itself (period always ends at `now`) is on `main` via PR #491,
│    merged 2026-07-06 — if you're seeing this on current `main`, treat it as a regression, not
│    the original bug, and re-verify with `gh pr view 491`.
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

### If likelihoods look wrong (a sensor's contribution to probability feels off)

1. Confirm sample sufficiency first: `sample_count >= MIN_CORRELATION_SAMPLES (50)` in the `correlations` table for that entity — below 50, don't trust the numbers at all; `db/correlation.py:99` gates the raw computation, `:1056` discounts confidence as `abs_correlation * (1 - 50/sample_count)`, and `:1541` re-invalidates on reload if it ever drops back under 50.
2. For **binary** sensors (motion/door/media/appliance/etc.), likelihoods come from `analyze_binary_likelihoods` (`db/correlation.py:324-620`) — a **duration-overlap** method, not a point-sample method: it sums how many seconds the sensor was in an active state during occupied vs. unoccupied cache intervals, then divides. This means **the same `occupied_intervals_cache` ground truth that feeds priors also feeds likelihoods** — if you found an interval-detection bug in the priors branch above, expect corresponding likelihood distortion for every binary sensor in that area, not just the prior.
3. Check for the "black hole" clamp: `prob_given_true`/`prob_given_false` are clamped to `[0.05, 0.95]` (`db/correlation.py:584-588`) — a sensor at exactly 0.05 or 0.95 might be a real strong signal, or might be masking a near-0/near-1 raw ratio; check `analysis_error` on the `correlations`/`entities` row (`no_occupied_intervals`, `no_active_intervals`, `no_active_during_occupied`) before trusting the clamped number — any of those error strings means the type-default likelihood was used instead of a learned one, not the clamped value.
4. For **numeric/environmental** sensors, use `scripts/visualize_distributions.py "<Area Name>" <entity_id> --db-path config/.storage/area_occupancy.db --days 30` to plot the raw histogram against the fitted Gaussian and visually confirm the learned `mean`/`std` actually fit the data, rather than trusting `correlation_strength` blind (this is exactly what that script is for — see its docstring).

### If decay looks wrong (clears too fast/slow, or ignores a custom half-life)

1. Trace `Decay.half_life` → `Decay._resolve_purpose_half_life()` (`custom_components/area_occupancy/data/decay.py`). Only `AreaPurpose.SLEEPING` has an `awake_half_life` (620s, `data/purpose.py`) — every other purpose returns `self._base_half_life` unconditionally, so this branch only matters for Bedroom/Sleeping-purpose areas with configured `sleep_start`/`sleep_end`.
2. **Settled bug (#481, fixed via PR #493, merged 2026-07-06)**: outside the configured sleep window, `_resolve_purpose_half_life()` used to unconditionally return `self._purpose.awake_half_life` (620s) even when the user had configured a custom half-life for that area — silently discarding it. The fix ("mirrors the custom-vs-default semantics established for #440") added a guard *before* the sleep-window check: `if self._base_half_life != self._purpose.half_life: return self._base_half_life` — i.e. if the configured half-life isn't the purpose's own default, the sleep/awake split never overrides it. This is live on `main`; re-verify with `gh pr view 493 --json state,mergedAt` if you suspect a regression.
3. If you hit this bug class again elsewhere: the general pattern (see Saga 2 in `aod-failure-archaeology`) is "is-this-value-a-user-override-or-a-coincidentally-matching-default" — `Purpose.is_purpose_half_life()` in `data/purpose.py` is the load-bearing comparison; it was already burned once (#439/#440) by comparing against *any* purpose's default instead of only the *selected* purpose's default. Any new decay/half-life logic must re-check which comparison it's making.
4. Also check `Decay.modifier_factor` (adjacency Phase 4 decay-stretch, only relevant if the area has adjacent-area config — merged to `main` via PR #454, 2026-07-06; still out of scope for a *learning-accuracy* investigation per this skill's "When NOT to use this" note, since it consumes decay rather than producing learned values, but it is no longer branch-gated).

---

## Phase 3 — Solution menu (ranked, each with a derivation obligation)

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

## Phase 4 — Validation and promotion

No fix from Phase 3 goes anywhere near `main` without all of the following, in order:

1. **Regression tests encoding the correct math**, not just "test doesn't crash." Follow the PR #491 pattern exactly: the old test (`tests/test_data_analysis.py::test_valid_calculation_sets_correct_prior`) had the *buggy* expected value (`0.99`) baked into its assertion; the fix corrected the assertion to the hand-derived value (`0.25`) and added a dedicated regression test (`test_quiet_tail_included_in_denominator`) that fails if the bug ever comes back. Any fix in this campaign needs the equivalent: a test whose expected value you hand-derived in Phase 0/2, not one copied from the current (possibly wrong) output.
2. **Simulator scenario** — paste a `run_analysis` service-response snapshot (from `custom_components.area_occupancy.service.py`'s `run_analysis` action) into `simulator/app.py` (run via `python main.py` from repo root) and interactively toggle sensors to sanity-check the fix's effect end-to-end. **Do not treat the simulator's own internal state machine as ground truth** — it has zero test coverage of its own (`simulator/` has no `test_*.py` files; confirmed by direct listing, only `app.py`/`__init__.py`), it's a visualization/debugging aid built on real integration code (`EntityType`/`Entity` classes imported directly), not a validated oracle.
3. **Pre-merge DB-copy check** — run the fix against the actual DB copy from Phase 0 (or a fresh sync from the source install) for at least one full analysis cycle (hourly, `ANALYSIS_INTERVAL=3600s`), and record before/after deltas for every Phase 1 metric on that specific area. "It looks better" is not a result; the number must move in the predicted direction from your Phase 3 derivation, by roughly the predicted amount. Note this is the *pre-merge* validation stage against copied data — it is distinct from, and does not replace, the *post-merge* live-install soak in `aod-research-methodology`'s idea lifecycle (where the merged fix rides along in the next release on a real install before being declared done). A campaign fix goes through both, in that order.
4. **Route through `aod-change-control`** — this campaign's output is a diagnosed bug + a validated fix + before/after numbers. It is never merged on eyeball, and it is never merged by this skill directly.

---

## Fenced wrong paths (do not do these)

- **Widening the `[0.01, 0.99]` clamps** (`MIN_PROBABILITY`/`MAX_PROBABILITY`, `MIN_PRIOR`/`MAX_PRIOR`, `const.py:173-176`) to "fix" a pinned prior. The clamp isn't the bug; something upstream produced a ratio that saturated it. Widening the clamp just moves where saturation becomes visible.
- **Truncating the observation period** to make a metric look better (this is literally the #483 bug — see Phase 2). The period must always extend to "now"; anything else drops known-unoccupied time from the denominator.
- **Hand-tuning constants per-home.** Every threshold in this file (`MIN_CORRELATION_SAMPLES`, half-life defaults, `TIME_PRIOR_MIN_BOUND`/`MAX_BOUND`, etc.) is a repo-wide default. A fix that only works because you hardcoded a number for one specific install's DB is not a fix — the "config surface is sacred" unwritten law means new *user-facing* knobs need config-flow + migration work (see `aod-config-and-flags`), not silent per-home constant edits.
- **Trusting the simulator backend as ground truth** for anything you haven't independently hand-verified per Phase 0/4 — it is untested code, not an oracle, however convenient it is to click sensors and watch numbers move.
- **Changing math and tests in the same breath** without a hand-computed expected value first. If your only evidence a change is correct is "the test I also just wrote passes," you've encoded your assumption, not verified it. Do Phase 0/1's hand computation, THEN write the code, THEN write the test against the hand-derived number.

---

## Provenance and maintenance

Date-stamped: 2026-07-06 (post-merge sweep, `main` HEAD `17b71d2`), integration version still 2026.5.17 (`custom_components/area_occupancy/manifest.json`, `pyproject.toml`, `const.py::DEVICE_SW_VERSION`) — none of the fixes below have shipped in a tagged release yet.

Merged since the prior sweep, all confirmed on `main` as of 2026-07-06:
- PR #491 (fix: quiet-tail global-prior denominator, fixes #483) — merged 2026-07-06; re-verify: `gh pr view 491 --json state,mergedAt`
- PR #493 (fix: bedroom custom half-life outside sleep window, fixes #481) — merged 2026-07-06; re-verify: `gh pr view 493 --json state,mergedAt`
- PR #454 (feat: adjacent-areas) — merged 2026-07-06; out of scope for this skill but shares the coordinator tick; re-verify: `gh pr view 454 --json state,mergedAt`

Re-verification commands for every volatile fact category in this skill:
- Clamp/threshold constants (`MIN_PROBABILITY`, `MIN_PRIOR`, `TIME_PRIOR_MIN_BOUND`, `MIN_CORRELATION_SAMPLES`, etc.): `grep -n "MIN_PROBABILITY\|MAX_PROBABILITY\|MIN_PRIOR\|MAX_PRIOR\|TIME_PRIOR_MIN_BOUND\|TIME_PRIOR_MAX_BOUND\|MIN_CORRELATION_SAMPLES\|PRIOR_FLOOR_THRESHOLD_MARGIN\|DEFAULT_MOTION_TIMEOUT" custom_components/area_occupancy/const.py`
- Global prior period-selection logic: `grep -n "actual_period_end\|actual_period_duration" custom_components/area_occupancy/data/analysis.py`
- Time-prior bucketing/DST handling: read `custom_components/area_occupancy/data/analysis.py`'s `calculate_time_priors` function in full (docstring + body)
- Likelihood computation and clamps: `grep -n "def analyze_binary_likelihoods\|clamp_probability(prob_given" custom_components/area_occupancy/db/correlation.py`
- Decay half-life resolution: read `custom_components/area_occupancy/data/decay.py`'s `_resolve_purpose_half_life` in full; purpose defaults in `custom_components/area_occupancy/data/purpose.py`
- PR/issue merge state for anything cited above: `gh pr view <n> --json state,mergedAt,statusCheckRollup` / `gh issue view <n> --json state`
- Coverage gate: `grep -n "fail_under" pyproject.toml` (currently 85%, comment claims 90% — believe the number, not the comment)
- Simulator test coverage: `find simulator -maxdepth 1 -iname "test*"` (expect no matches as of this writing)
