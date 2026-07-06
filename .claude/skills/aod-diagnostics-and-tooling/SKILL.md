---
name: aod-diagnostics-and-tooling
description: Use when you need to MEASURE what Area Occupancy Detection is actually doing instead of guessing — downloading and reading the HA diagnostics export, interpreting a health/repair issue, reading hourly-analysis debug logs, or using the simulator to compare a predicted probability against the live sensor. Load this before answering "why is this area stuck occupied", "is the prior learned correctly", "what does this repair mean", or "how do I reproduce this probability by hand".
---

# AOD Diagnostics and Tooling

## What this covers

The four instruments this project has for observing real runtime state
instead of eyeballing code: the HA **diagnostics export** (a JSON snapshot,
no debug logging required), the **health/repairs system** (structured
anomaly detection, not just log noise), **debug log** interpretation (which
lines mean what during the hourly analysis pipeline), and the **simulator**
(feed a scenario, compare a predicted probability against reality). It also
ships three tested, read-only scripts against the real SQLite schema for
answering "what did the DB actually learn" without waiting for a diagnostics
export.

## When NOT to use this

For the underlying Bayesian math those numbers come from (sigmoid pipeline,
logit combination, decay formula), use `bayesian-occupancy-reference`. For
*why* a number is wrong and how to fix the code, use
`aod-debugging-playbook`. For interpreting DB schema/columns in general
(not diagnostics-specific), see `aod-architecture-contract`. For the
open-ended campaign to actually improve prior/likelihood accuracy using
these tools, see `aod-learning-accuracy-campaign`.

---

## 1. The diagnostics export

**How to download**: Home Assistant UI → **Settings → Devices & Services**
→ click the **Area Occupancy Detection** integration card → **⋮** menu →
**Download diagnostics**. Produces a JSON file. Implemented in
`custom_components/area_occupancy/diagnostics.py`
(`async_get_config_entry_diagnostics`); user-facing walkthrough already
exists at `docs/docs/technical/diagnostics.md` — this section is the
ground-truth field reference, cross-checked against the source.

Each area subsection is captured in its own `try/except`
(`diagnostics.py:_area_snapshot`) — if one subsection fails, you get a
sibling `<section>_error` key instead of losing the whole dump. That design
choice matters when triaging: a partial diagnostic is still useful, read
past the error key for the rest.

### Top-level shape

```
{
  "integration": { ... },
  "areas": [ { ... one per configured area ... } ],
  "database": { ... }
}
```

### `integration`

| Field | Meaning |
|---|---|
| `version` | `DEVICE_SW_VERSION` (matches `manifest.json`) |
| `config_version` / `config_version_minor` | Config-entry schema version (`CONF_VERSION`/`CONF_VERSION_MINOR`) — a *different* versioning axis from the release version, used by `migrations.py` |
| `entry_id` / `entry_title` | HA config entry identifier and label |
| `setup_complete` | `true` once the coordinator finished its first analysis cycle |
| `area_count` | Number of configured areas |
| `sleep_start` / `sleep_end` | Configured sleep window (integration-level) |
| `people_count` | Number of configured people for sleep-presence tracking |

### `areas[]`

Each entry has `area_name`, `area_id`, `purpose`, `threshold`, then these
subsections:

#### `current`

The live snapshot at the moment the diagnostic ran.

| Field | Meaning |
|---|---|
| `probability` | `area.probability()` — computed occupancy probability, 0.0-1.0 |
| `occupied` | `probability >= threshold` |
| `decay_factor` | Average decay multiplier across all entities |
| `active_entity_count` / `decaying_entity_count` / `entity_count` | Sensor counts |
| `adjacency` | Present only if the adjacent-areas feature has fired this tick — see below |

**`adjacency` sub-block** (shipped in PR #454, merged 2026-07-06 — now on
`main`. Omitted entirely, not a null field, when there's no boost/modifier
yet — e.g. before the first `update()` tick, or the area has no configured
neighbours):

| Field | Meaning |
|---|---|
| `boost.fired` | Whether a Bayesian logit-space boost was applied this tick |
| `boost.trajectory_prev` / `trajectory_prev_prev` | The 2-hop trajectory (last two recently-occupied neighbour areas) used to look up the transition probability |
| `boost.hour_of_week` | 0-167 bucket (day-of-week × 24 + hour) used for the lookup |
| `boost.raw_probability` | The learned/fallback transition probability before gain is applied |
| `boost.fallback_level` | Which of the 6 smoothing levels answered the lookup — `2hop_hour_of_week`, `2hop_hour_of_day`, `2hop_unbucketed`, `1hop_hour_of_week`, `1hop_unbucketed`, or `static_default` (falls back to the hand-configured `influence_weight`, observed/total forced to 0 to signal "no learned data") |
| `boost.observed_count` / `total_count` | How many observations backed the chosen level |
| `boost.logit_contribution` | `ADJACENCY_BOOST_GAIN (0.5) * (logit(raw_probability) - logit(0.5))` — the actual amount added to the area's logit-space probability this tick |
| `decay_modifier.fired` | Whether a decay half-life stretch was applied |
| `decay_modifier.silence_score` | Σ over silent neighbours of `(1 - P_neighbour_lagged) * P(target→neighbour \| trajectory)`, clamped to [0,1] |
| `decay_modifier.decay_modifier` | `min(1 + 0.75*silence_score, 1.75)` — the multiplier stretching this area's decay half-life (max +75%) |
| `decay_modifier.silent_neighbours[]` | Per-neighbour `(neighbour, lagged_probability, transition_probability)` breakdown — read this to see *which* neighbour is holding this area's decay open |

A `boost.fallback_level` of `static_default` with `observed_count`/`total_count`
both 0 means the adjacency feature has no learned data for that pair yet —
it's using the flat config-time `influence_weight`, not learned behavior.

#### `prior`

From `Prior.diagnostic_snapshot()` (`data/prior.py`) — surfaces which term
is driving occupancy, most useful when an area looks "stuck" with no active
evidence:

| Field | Meaning |
|---|---|
| `prior_value` | Effective prior used in the current calculation (post-combination, post-floor) |
| `global_prior` | Long-run learned probability (`null` if not learned yet) |
| `time_prior` | Day-of-week + time-slot specific learned prior |
| `min_prior_floor_applied` | `none`, `purpose`, or `override` — which floor (if any) raised the value. A non-`none` floor here plus a probability sitting suspiciously close to `threshold` is a strong signal the floor (not learned evidence) is what's keeping the area "occupied" |
| `threshold` | Configured occupancy threshold for the area |

A floor can never push the prior to or above `threshold` by itself (capped
at `threshold - 0.01`, see `data/prior.py`, fix for issue #435) — so if the
area actually reads *occupied* purely from prior with no evidence, that came
from a genuinely learned value, not the floor.

#### `config`

Sensor counts (not entity IDs) per type, decay settings, wasp-in-box
settings, per-type weights, `min_prior_override`, `motion_timeout`. Shape
only — cross-reference `aod-config-and-flags` for what each value means.

#### `entities[]`

One row per configured sensor:

| Field | Meaning |
|---|---|
| `entity_id` | The HA entity |
| `input_type` | `motion`, `media`, `door`, `temperature`, etc. |
| `weight` | Configured per-type weight (not `effective_weight` — the diagnostic exposes raw `weight`; `effective_weight = weight * information_gain` is computed at calculation time, not persisted here) |
| `prob_given_true` / `prob_given_false` | Likelihoods used in the Bayesian update — learned via correlation analysis if available, else the `EntityType` default |
| `evidence` | Current state contribution: `true` / `false` / `null` (unavailable/unknown) |
| `previous_evidence` | What it was before the last transition — used by `has_new_evidence()` to decide whether to start/stop decay |
| `last_updated` | ISO-8601 UTC timestamp of the last evidence transition |
| `analysis_error` | Why correlation analysis was skipped or failed. Verified exact values (`data/entity_type.py::AnalysisStatus`): `not_analyzed` (hasn't run yet), `motion_sensor_excluded` (by design — motion always uses configured priors, never learned correlation), `analyzed` (ran; check `correlation_type`/`correlation_strength` for the result). Correlation-*failure* strings (`too_few_samples`, `no_occupied_intervals`, etc.) live in `db/correlation.py::CORRELATION_FAILURE_ERRORS`, a different, larger set than the 3-value `AnalysisStatus` enum — don't conflate the two when reading this field |
| `correlation_strength` | Cached learned Pearson correlation or binary-likelihood strength (`null` if not learned) |
| `correlation_type` | `strong_positive`, `positive`, `strong_negative`, `negative`, `none`, or `binary_likelihood` |
| `learned_active_range` | Learned numeric active-range override for environmental sensors (`null` if not learned) |
| `learned_gaussian_params` | `mean_occupied`/`std_occupied`/`mean_unoccupied`/`std_unoccupied` for continuous-likelihood sensors (`null` if not learned) |
| `decay.is_decaying` / `half_life` / `decay_start` / `decay_factor` | Current decay state for this entity |

#### `health`

The **cached** output of the last health check (no re-check triggered) —
see §2 for the full issue-type/threshold reference. Same array shape as the
HA Repairs UI shows; `issue_type` distinguishes sensor-scope
(`stuck_active`, etc., with `entity_id`/`input_type` populated) from
pipeline-scope (`insufficient_priors`, etc., both `null`).

### `database`

| Field | Meaning |
|---|---|
| `interval_count` | Total occupancy-state interval rows |
| `prior_count` | Time-prior rows across all areas (compare against `area_count * 168` to see time-prior learning completeness) |
| `correlation_count` | Stored correlation rows |
| `entity_count` / `area_count` | Persisted entity / area row totals |
| `occupied_intervals_cache` | Per-area `{ "valid": true/false }` — `false` means the next hourly analysis cycle rebuilds it |

### Privacy

Contains: HA entity IDs, current sensor evidence, learned priors/weights,
config shape, area names. Does **not** contain: HA access tokens, user
account info, raw recorder history. Entity IDs/area names are still
user-identifying strings — review before attaching to a public issue.

---

## 2. Health / repairs as a diagnostic

Implemented in `custom_components/area_occupancy/data/health.py`. Two
independent check families, both surfaced as HA repair issues
(Settings → Repairs) and both cached into the diagnostics export's
`health` section without re-running:

- **Sensor-scope** (per-entity): `stuck_active`, `stuck_inactive`,
  `unavailable`, `never_triggered`
- **Pipeline-scope** (per-area, `entity_id`/`input_type` both `null`):
  `insufficient_priors`, `stale_intervals_cache`, `slow_analysis`,
  `correlation_failures`

`InputType.SLEEP` is excluded from **all** health checks
(`_EXCLUDED_TYPES`). `media_player.*` entities are exempt from the
`unavailable` check specifically (TVs powering off is normal, not a fault —
this was the #466 complaint).

### Sensor-scope thresholds

| Check | Base threshold | Purpose multiplier | Notes |
|---|---|---|---|
| `stuck_active` — motion | 8h | ×6 sleeping / ×4 relaxing / ×3 working | Was 2h; raised because mmWave/presence sensors legitimately stay on for hours in bedrooms/offices (#465, #468) |
| `stuck_active` — media | 12h | same multiplier table | |
| `stuck_active` — appliance | 24h | same | |
| `stuck_active` — door | 48h | same | |
| `stuck_active` — window | 72h | same | |
| `stuck_active` — cover | 24h | same | |
| `stuck_inactive` — motion | 7 days | none | |
| `stuck_inactive` — media | 14 days | none | |
| `stuck_inactive` — appliance | 28 days | none | |
| `stuck_inactive` — door/window/cover/power | 14 days each | none | |
| `unavailable` | 1h | none | Clock starts from first-seen-unavailable *this HA session*, not persisted `last_updated` — avoids false trips when a source integration is just slow to load at startup |
| `never_triggered` | 7 days | none | Uses persisted `last_updated`; only for `_STUCK_CHECK_TYPES` (binary + power + motion + cover) |

Purpose multiplier table (`_PURPOSE_STUCK_ACTIVE_MULTIPLIER`, applies only
to `stuck_active`): `SLEEPING` ×6 (8h→48h), `RELAXING` ×4 (8h→32h),
`WORKING` ×3 (8h→24h). Purposes not listed use the base threshold
(multiplier 1.0).

### Pipeline-scope thresholds

| Check | Threshold | Grace period |
|---|---|---|
| `insufficient_priors` | No `global_prior` after grace | 7 days (`PRIORS_TRAINING_GRACE_PERIOD`) |
| `stale_intervals_cache` | Cache older than 25h, or never populated past grace | Same 7-day grace for the "never populated" case |
| `slow_analysis` | Last full cycle > 180,000ms (3 min) | n/a — deliberately conservative; large installs can legitimately exceed 30s on a first warm cycle |
| `correlation_failures` | ≥50% of correlatable entities have a real (not soft/expected) `analysis_error` | Same 7-day grace |

### Exemptions / suppression the user controls

- **Ignored repairs survive condition flaps.** If a user clicks "Ignore" in
  HA's Repairs UI, the monitor won't delete-then-recreate the issue when the
  underlying condition briefly clears and recurs (e.g. TV unavailable again
  the next night) — `HealthMonitor._is_ignored()` checks
  `IssueEntry.dismissed_version is not None` and preserves the ignore across
  flap cycles (fix for the pattern behind #472/#473).
- **Integration-level off-switch.** `health_enabled` (coordinator-level
  toggle, PR #472) — when `False`, `clear_all_issues()` deletes every active
  repair without resetting the in-memory unavailable-since clock, so
  re-enabling later doesn't cause every currently-unavailable sensor to trip
  instantly.
- There is **no per-sensor suppression** yet (requested in open issue #466)
  and **no vacation-aware suppression** yet (requested in open issue #485) —
  don't describe either as shipped; re-check with `gh issue view 466` /
  `gh issue view 485`.

---

## 3. Debug log interpretation

Enable per CLAUDE.md's standard snippet:

```yaml
logger:
  default: info
  logs:
    custom_components.area_occupancy: debug
    custom_components.area_occupancy.db: debug
```

### Step timings (every hourly analysis cycle)

Each pipeline step logs its own timing
(`data/analysis.py::_run_step`):

```
Step 7: recalculate_priors completed in 842.13 ms
```

On failure: `Step %d: %s FAILED in %.2f ms` (via `_LOGGER.exception`, so a
full traceback follows). At the end of a full cycle:

```
Full analysis completed: 12/12 steps succeeded in 4213.87 ms
```

or, with failures: `Analysis completed: 11/12 steps succeeded (FAILED: correlation_analysis) in ... ms`. A cycle cancelled by
`EVENT_HOMEASSISTANT_STOP` mid-run logs
`Analysis cancelled mid-run after ... ms — shutdown in progress` instead —
deliberately not counted toward the slow-analysis health threshold.

On `main` (post PR #454, merged 2026-07-06) the pipeline is **13 steps**, in
order: `sync_states`, `health_check_and_prune` (pruning + integrity),
`sensor_health_check`, `populate_occupied_intervals_cache`,
`interval_aggregation`, `numeric_aggregation`, `recalculate_priors`,
`correlation_analysis`, `transition_learning`, `pipeline_health_check`,
`save_data_before_refresh`, `refresh_coordinator`, `save_data_after_refresh`.
Totals log as `13/13`. Re-verify:
`grep -n "total_steps" custom_components/area_occupancy/data/analysis.py`.

### Prior calculation — the "Period calculation" line

The single most useful debug line for prior-accuracy debugging
(`data/analysis.py`, inside `calculate_and_update_prior`):

```
Period calculation for area %s: first_interval_start=%s (tz=%s), last_interval_end=%s (tz=%s), now=%s (tz=%s)
```

This is exactly the line that would have exposed **issue #483** (global
prior inflating during quiet periods) before it was diagnosed from code —
if `last_interval_end` is far in the past relative to `now` and the stored
prior still looks too high, check which formula produced it (see script #2
below). **SETTLED**: PR #491, merged 2026-07-06, changed the period-end
logic so `actual_period_end` is now always `now` — it no longer silently
truncates to `last_interval_end` and the quiet-tail inflation bug (#483) is
closed. Verified on `main`: `actual_period_end = now` unconditionally
(`data/analysis.py`); the old `actual_period_end = last_interval_end`
behavior is gone. Kept here as dated history because the "Period
calculation" line is still the right first place to look if a similar
quiet-period discrepancy ever resurfaces.

Companion debug lines in the same function: `Prior calculation for area %s:
%d merged intervals, %.1f hours total duration` (logged early, before period
math) and `Prior analysis completed for area %s: global_prior=%.3f (occupied:
%.1f hours over %.1f days, %d intervals)` (logged at the end, the final
answer). Reading these three lines together tells you the full story of one
area's prior calculation without touching the DB.

### Interval query performance

`db/queries.py::get_occupied_intervals` logs
`Interval query executed in %.3fs for %s (total=%d, motion=%d)` and
`Unified occupancy calculation for %s: %d raw -> %d merged intervals
(processing: %.3fs)` — useful for spotting a slow correlation/prior
recalculation caused by index gaps or an oversized `intervals` table before
assuming it's a math bug.

---

## 4. The simulator as a measurement instrument

Located at `simulator/` (Flask backend, `app.py`) + a browser UI served from
the MkDocs site (`docs/docs/simulator/simulator.html`). Workflow:

1. In HA, call the `area_occupancy.run_analysis` service (Developer Tools →
   Actions, or automations) — it forces an immediate analysis cycle and
   returns YAML/dict output per area: `current_probability`, `current_occupied`,
   `current_threshold`, `current_prior`, `global_prior`, `time_prior`,
   `prior_entity_ids`, `entity_states`, `likelihoods` (see
   `custom_components/area_occupancy/service.py::_build_analysis_data`).
2. Run the simulator locally: `python main.py` from repo root (listens on
   `0.0.0.0:5000` by default; `PORT` env var overrides), then either open the
   simulator UI directly or `cd docs && mkdocs serve` and browse to
   `/simulator/`, pointing its **API Connection** field at
   `http://127.0.0.1:5000`.
3. Paste the `run_analysis` output, click **Load Simulation** — this
   reconstructs the area's `Entity` objects and priors inside the simulator
   process.
4. Toggle binary sensors / adjust numeric values and read the recomputed
   probability plus the per-entity contribution breakdown
   (`calculate_probability_breakdown` in `simulator/app.py`) to see which
   sensor is driving the number.

### Important caveat: the simulator does not run production's math

The simulator's sensor-fusion step calls the **legacy dead-code
`bayesian_probability()` path**, not the live sigmoid/logit pipeline — see
`aod-run-and-operate` §6 for the full verified explanation (PR #353 history,
what is and isn't shared). Measurement consequence here: treat simulator
probabilities as a *relative sensitivity tool* (which sensor moves the
number, which direction, roughly how much), never as an exact stand-in for
the live probability sensor. For exact production numbers, read the live
sensor or the diagnostics export's `current.probability`.

---

## 5. Scripts (in `scripts/`)

Three read-only scripts, tested against a scratch SQLite DB built from the
real schema (`db/schema.py`) during authoring — no test databases are
committed. Each takes a path to a real `area_occupancy.db` (default location:
`config/.storage/area_occupancy.db` inside the HA config dir) and uses only
the Python stdlib `sqlite3` module — no Home Assistant import required, so
they run in any Python 3.10+ environment.

| Script | Answers | Requires |
|---|---|---|
| `dump_priors.py <db> <area>` | What did the area learn — global prior + all 168 time-of-week buckets? | `global_priors`, `priors` tables (present on `main` today) |
| `prior_quiet_tail_diff.py <db> <area> [--now ISO]` | Is this area's stored prior exhibiting the #483 quiet-tail-inflation pattern? Recomputes both the pre-#491 and post-#491 formulas by hand and diffs against what's stored | `intervals`, `entities`, `global_priors` tables (present on `main` today) |
| `transitions_summary.py <db> [--top N]` | How much has adjacency learning actually observed — which chains, how recent, do they clear the smoothing-fallback thresholds? | `area_transitions` table — shipped with PR #454, merged 2026-07-06, present on `main` today; the script still detects a missing table and says so rather than crashing (useful against older DBs created pre-migration) |

Each script's own docstring has a full usage line, example output, and an
interpretation guide — read the script header before running it. Run
directly, e.g.:

```bash
python .claude/skills/aod-diagnostics-and-tooling/scripts/dump_priors.py \
  config/.storage/area_occupancy.db "Living Room"
```

Do not point these at a live HA instance's DB file while HA is running
without copying it first — SQLite allows concurrent readers, but copy the
file if you want a stable snapshot to diff against a second run.

---

## Provenance and maintenance

Re-verified 2026-07-06 against `main` at `17b71d2` (post merge-wave; integration
release version is still 2026.5.17 in `pyproject.toml` line 7 /
`manifest.json` line 20 / `const.py` `DEVICE_SW_VERSION` — none of this
wave's merges are in a tagged release yet). `feat/adjacent-areas` (PR #454)
merged into `main` 2026-07-06 — the `AreaTransitions` table, `adjacency`
diagnostics sub-block, and the six-level transition-lookup smoothing
described in §1/§5 are now on `main` and generally available; the feature
remains unvalidated on real homes (still worth treating as a candidate
signal, not ground truth, until it's watched running for a while).

Re-verification commands, by volatile fact category:

- **Diagnostics export shape**: `sed -n '1,410p' custom_components/area_occupancy/diagnostics.py` (re-read `_area_snapshot`, `_entity_snapshot`, `_adjacency_snapshot`)
- **Health thresholds**: `grep -n "THRESHOLD\|MULTIPLIER" custom_components/area_occupancy/data/health.py`
- **Adjacent-areas / PR #454 merge state**: `gh pr view 454 --json state,mergeStateStatus,mergeable` (confirmed MERGED 2026-07-06)
- **Prior quiet-tail fix / PR #491 merge state**: `gh pr view 491 --json state,mergeable` (confirmed MERGED 2026-07-06; `actual_period_end = now` unconditionally on `main`)
- **`AnalysisStatus` exact values**: `grep -n "NOT_ANALYZED\|MOTION_EXCLUDED\|ANALYZED =" custom_components/area_occupancy/data/entity_type.py`
- **Correlation failure-error set**: `grep -n "CORRELATION_FAILURE_ERRORS" -A 20 custom_components/area_occupancy/db/correlation.py`
- **Adjacency constants**: `grep -n "ADJACENCY_" custom_components/area_occupancy/const.py`
- **Simulator legacy-math caveat still true**: `grep -n "bayesian_probability" simulator/app.py` (confirms it's still imported/called) and `git log --oneline -- simulator/app.py` (confirms no post-#353 migration commit)
- **Open per-sensor-suppression / vacation-suppression issues**: `gh issue view 466`, `gh issue view 485`
- **Scripts still run clean against current schema**: re-run each script's usage example against a fresh scratch DB built from `custom_components/area_occupancy/db/schema.py::Base.metadata.create_all()`
