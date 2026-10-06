# Diagnostics

The integration ships with Home Assistant's standard **diagnostics** export so you can capture a complete snapshot of its runtime state without enabling DEBUG logging. This is the fastest way to figure out why occupancy "looks wrong" — and the right thing to attach when opening a GitHub issue.

## How to download

1. Go to **Settings → Devices & Services**.
2. Click **Area Occupancy Detection**.
3. Open the **⋮** menu next to the integration card and choose **Download diagnostics**.

You'll get a JSON file you can open in any text editor.

## What's in it

The export is grouped into three sections.

### `integration`

Metadata about the integration itself — useful for confirming versions when reporting bugs.

| Field | Meaning |
|---|---|
| `version` | Integration software version (matches `manifest.json`) |
| `config_version` / `config_version_minor` | Schema version of your stored config |
| `entry_id` / `entry_title` | The HA config entry identifier and label |
| `setup_complete` | `true` once the coordinator finished its first analysis cycle |
| `area_count` | Number of configured areas |
| `sleep_start` / `sleep_end` | Configured sleep window |
| `people_count` | Number of configured people for sleep tracking |

### `areas`

One entry per configured area. Each entry has these subsections — and if any subsection fails to capture, you'll see a sibling `<section>_error` field so the rest of the dump still comes through.

#### `current`

The live calculation snapshot at the moment the diagnostic ran.

| Field | Meaning |
|---|---|
| `probability` | Computed occupancy probability (0.0–1.0) |
| `occupied` | Whether `probability >= threshold` |
| `decay_factor` | Average decay multiplier across all entities |
| `active_entity_count` | How many sensors are currently in their "active" state |
| `decaying_entity_count` | How many sensors are currently mid-decay |
| `entity_count` | Total sensors configured for the area |

`current` also carries three **shadow-mode** blocks. Each is marked `shadow_mode: true`: computed and recorded, but never used by the live probability. They exist to gather evidence before any of them is allowed to change behaviour.

- **`accuracy`** (#499): how well the area's decisions matched motion-confirmed ground truth over the last 24 hours. It shows agreement, calibration error, false-on and false-off rates, and a suggested threshold. A rate (and the suggestion) stays blank until the window has at least an hour of that truth class. The window lives in memory and restarts with Home Assistant, so `history` keeps one summary per day (the last 14 shown, 90 kept) that survives restarts.
- **`online_prior`** (#500): a continuously updated estimate of the area's prior, compared with the hourly database calculation (`diff`, `diff_history`, `days_within_tolerance`). It restarts automatically when the area's motion, media or sleep sensors (or their active states) change, or when its history predates the database by more than a day (after a database reset), because statistics from another sensor set never line up with the database again.
- **`fusion`** (#501): learned per-sensor weights, reported once the area has `min_samples` ticks. For each sensor, `learned_weight` is what the learner has arrived at, `effective_weight` is the live pipeline's weight (configured weight × information gain, so `0` when the sensor's learned likelihoods carry no information), and `samples` is how many ticks that sensor contributed. A sensor removed from the area, or whose active states change, is forgotten.

#### `prior`

The learned prior breakdown — surfaces *which term* is driving the occupancy probability, especially useful when an area appears stuck without active evidence.

| Field | Meaning |
|---|---|
| `prior_value` | Effective prior used in the current calculation |
| `global_prior` | Long-term learned probability for the area (`null` if not learned yet) |
| `time_prior` | Day-of-week + time-slot specific prior |
| `min_prior_floor_applied` | One of `none`, `purpose`, `override` — which floor (if any) raised the learned prior |
| `threshold` | Configured occupancy threshold for the area |

#### `config`

The shape of how the area is configured — sensor counts (not entity IDs), weights, decay, and wasp-in-box settings.

#### `entities`

One row per configured sensor. The most useful fields when debugging:

| Field | Meaning |
|---|---|
| `entity_id` | The HA entity |
| `input_type` | `motion`, `media`, `door`, `temperature`, etc. |
| `weight` | Effective weight applied during calculation |
| `prob_given_true` / `prob_given_false` | Likelihoods used in the Bayesian update |
| `evidence` | Current state contribution (`true` / `false` / `null` if unavailable) |
| `previous_evidence` | What it was before the last transition |
| `last_updated` | When evidence last changed (ISO-8601 UTC) |
| `correlation_strength` | Cached learned correlation with occupancy (`null` if not learned) |
| `correlation_type` | `correlation` or `binary_likelihood` |
| `analysis_error` | Why correlation analysis was skipped (e.g. `not_analyzed`, `motion_excluded`, `too_few_samples`) |
| `decay` | Current decay state — `is_decaying`, `half_life`, `decay_factor`, `decay_start` |

#### `health`

The cached output of the [Sensor Health Monitoring](../features/sensor-health.md) checks — no re-check is triggered to capture them, so you see exactly what the user sees in **Repairs**. Both sensor-scope issues (with `entity_id` and `input_type` populated) and pipeline-scope issues (with both `null`) appear in the same `issues` array; the `issue_type` field tells them apart — e.g. `stuck_active` vs. `insufficient_priors`, `stale_intervals_cache`, `slow_analysis`, `correlation_failures`.

### `database`

| Field | Meaning |
|---|---|
| `interval_count` | Total occupancy-state intervals stored |
| `prior_count` | Time-prior rows (1 per day-of-week × time-slot × area) |
| `correlation_count` | Stored sensor-correlation rows |
| `entity_count` / `area_count` | Persisted entity / area row totals |
| `occupied_intervals_cache` | Per-area cache freshness — `{ "area": { "valid": true } }`. A stale cache means the next analysis cycle will rebuild it. |

## What questions diagnostics can answer

- **"Why is the area stuck occupied?"** → check `current.probability`, `current.decay_factor`, and the `entities` list for any sensor with `evidence: true` you didn't expect, or a high `correlation_strength` paired with a frozen `last_updated`. Also look at `prior.min_prior_floor_applied` — a floor (`purpose` or `override`) might be holding the value above the threshold.
- **"Has the integration finished learning?"** → `prior.global_prior` is `null` until enough history is collected; `prior.time_prior` populates per slot. Compare `database.prior_count` against your area count × 168 (24h × 7 days × 1 slot/hour).
- **"Is correlation analysis working for sensor X?"** → find the entity in `entities` and read `analysis_error`. `not_analyzed` means it hasn't run yet; `too_few_samples` means there isn't enough data; `motion_excluded` is by design (motion sensors always use configured priors).
- **"Is my health repair stale?"** → `health.last_check` shows when the last check ran (hourly).
- **"Is the cache fresh?"** → `database.occupied_intervals_cache.<area>.valid` — if `false`, the next analysis cycle will rebuild it.

## Privacy

The export contains:

- :material-check: HA entity IDs, current sensor evidence states, learned priors and weights, configuration shape (sensor counts, thresholds, decay), area names.
- :material-close: No HA access tokens, no user account info, no raw recorder history.

Entity IDs and area names are descriptive labels for *your* installation. Review the JSON before attaching it to a public issue if you're sensitive about either.

## When to share it

When opening an issue about wrong probabilities, stuck areas, or unexpected calculation results, attach the diagnostic file. It saves a round-trip of "can you turn on debug logging and reproduce" — the snapshot already contains every field a triager would otherwise have to ask for.
