# Occupancy Forecast

How Area Occupancy turns learned history into a **forward-looking** occupancy
probability: the module that computes it, the maths behind every number it
returns, and the limits you must know before acting on it.

## Why this exists

The `occupancy_probability` sensor answers *"is this area occupied right now?"*.
Its lead time is zero: it reacts to evidence as evidence arrives. That is the
right answer for lighting, and the wrong one for anything with inertia — a
radiator, an underfloor loop, a heat pump — where you must act **before**
occupancy in order to be comfortable **at** occupancy.

The learned time priors already contain what such a consumer needs: a weekly
profile of how often each area is occupied at each hour. `data/forecast.py`
exposes that profile for arbitrary — including future — slots.

!!! info "Scope: the integration is a pure oracle"
    The forecast answers exactly one question: *P(area occupied at slot T)*.
    It deliberately does **not** choose a horizon, apply a trust gate, or decide
    when to actuate. Those belong to the consuming control system, which already
    owns its own control loop; duplicating them here would create two sets of
    tuning knobs fighting each other.

## Module map

| File | Role |
|---|---|
| `data/forecast.py` | Forecast maths and service-response shaping. Holds no state |
| `data/prior.py` | `Prior` — cache accessors `all_time_priors()`, `all_time_prior_points()`, `prior_for()` |
| `data/analysis.py` | `calculate_time_priors()` — builds the weekly matrix from occupied intervals |
| `db/queries.py` | `get_stored_time_priors()` (learned slots only), `get_all_time_priors()` (grid, defaults filled) |
| `service.py` | `area_occupancy.get_time_priors` — the read-only service entry point |
| `lovelace/area-occupancy-time-priors-card.js` | Heatmap card rendering the weekly matrix |

`forecast.py` is kept separate from `prior.py` on purpose: exposing the learned
priors to other integrations should touch the core probability model as little
as possible. `prior.py` keeps thin cache accessors; the projection maths and the
response shape live here.

### Public functions

#### `forecast_prior(global_prior, slot_time_prior, *, prior_factor) -> float`

The forecast for a single slot. Combines the area-wide learned prior with the
slot's learned time prior and clamps to `[MIN_PRIOR, MAX_PRIOR]` — mirroring the
learned term of `Prior.value`, minus the configuration floors, which are
threshold-relative safety nets for the live estimate and are not meaningful to
project onto a future slot. When `global_prior` is `None` (never learned) the
slot's own bounds-clamped time prior is returned as a best-effort fallback.

#### `build_area_time_priors(area, slot_minutes) -> dict`

One real area's weekly forecast, as three parallel per-slot maps keyed
`"day,slot"` (`day` 0=Monday…6=Sunday, `slot` = `hour * 60 // slot_minutes`).

#### `build_aggregate_time_priors(members, slot_minutes, area_id, name) -> dict | None`

The same shape for an aggregate zone (the *All Areas* device, per-floor devices),
whose occupancy is derived from member areas rather than stored. Values are the
clamped mean across members, mirroring `AllAreas.area_prior()` but per slot.
`data_points` takes the **minimum** across members: a zone is only as
well-learned as its least-observed room, so a consumer never over-trusts a mixed
aggregate. Returns `None` when there are no members.

## The maths, end to end

### Step 1 — Occupied intervals

Ground truth is motion sensors only, merged and extended by the motion timeout
(`db/queries.py:get_occupied_intervals`). Using a single sensor class keeps the
denominator consistent; see [Time Prior Flow](time-prior-flow.md).

### Step 2 — Per-slot ratio

For each weekly slot, `calculate_time_priors()` accumulates occupied seconds and
total available seconds over the analysis period, bucketing by **Home Assistant
local wall-clock time** while doing the overlap arithmetic in UTC (so DST
transitions neither duplicate nor lose an hour):

```text
time_prior[d,s] = clamp(occupied_seconds[d,s] / total_seconds[d,s],
                        TIME_PRIOR_MIN_BOUND, TIME_PRIOR_MAX_BOUND)
                = clamp(…, 0.03, 0.9)
```

`data_points[d,s]` records the number of distinct ISO weeks contributing to that
slot — the honest measure of how much evidence is behind the value.

The loop iterates the **denominators**, not the occupied buckets. A slot the
period covered but that saw zero occupancy is a real observation and is stored at
the lower bound. Only slots the period never covered stay unwritten.

### Step 3 — Unlearned slots

An unwritten slot still has to return something. It returns
`Prior.unlearned_slot_prior` — the area's own `global_prior`.

That choice is not cosmetic. `combine_priors(g, g) = g`, so a slot filled this
way contributes **no tilt in either direction**. The previous flat 0.5 was
neutral only in isolation: blended against a small global prior it came out
*higher* than every genuinely low-occupancy slot, so "never observed" outranked
"observed to be empty", and the inflation reached the live sensor as well as the
forecast. Before any global prior exists there is nothing neutral to fall back
to, so `DEFAULT_TIME_PRIOR` (0.5) is still used in that one case.

### Step 4 — Combination

```text
combined = sigmoid((1 - w) · logit(global_prior) + w · logit(time_prior))
w = 0.4      # utils.py:combine_priors, time_weight
forecast = clamp(combined · PRIOR_FACTOR, MIN_PRIOR, MAX_PRIOR)
```

### Dynamic range

The blend keeps 60% of the weight on the global prior, so the combined value
cannot move far from it. Concretely, sweeping the time prior across its whole
legal range:

| `time_prior` | g = 0.05 | g = 0.10 | g = 0.15 | g = 0.25 | g = 0.40 |
|---|---|---|---|---|---|
| 0.03 (min) | 0.041 | 0.062 | 0.081 | 0.114 | 0.163 |
| 0.50 | 0.146 | 0.211 | 0.261 | 0.341 | 0.439 |
| 0.90 (max) | 0.292 | 0.392 | 0.460 | 0.555 | 0.654 |

!!! warning "A low global prior caps the forecast below any useful threshold"
    For *any* slot to reach 50%, the area needs `global_prior > 0.188`. Below
    that, no hour of the week can ever cross a 50% cutoff no matter how
    consistently the area is occupied — the ceiling is the transform, not the
    learning. Rooms occupied a few hours a day routinely sit at
    `global_prior ≈ 0.02–0.13`.

**This is why the service returns `slots_raw`.** The raw time prior carries the
weekly *shape* without the compression, so it is the right input for ranking
hours against each other. Use `slots` when you need a number comparable to the
area's occupancy threshold; use `slots_raw` when you need to know *when* the area
is habitually busy.

## Service response

`area_occupancy.get_time_priors` (`SupportsResponse.ONLY`) returns:

```yaml
slot_minutes: 60
areas:
  Studio:
    area_id: studio
    global_prior: 0.1294
    slot_minutes: 60
    slots:       { "0,0": 0.0735, "0,9": 0.3512, ... }   # combined, threshold-comparable
    slots_raw:   { "0,0": 0.0300, "0,9": 0.6100, ... }   # learned time prior, full range
    data_points: { "0,0": 4,      "0,9": 4,      ... }   # weeks of evidence; 0 = unlearned
  Piano Terra:
    area_id: piano_terra
    aggregate: true
    members: [salotto, cucina, sala_da_pranzo]
    ...
```

Always check `data_points`. A slot with `0` is a filled placeholder, not a
measurement, and should be skipped or treated as unknown rather than acted upon.

## Known limits

| Limit | Value | Consequence |
|---|---|---|
| Effective history window | ~28 days (`RETENTION_RAW_INTERVALS_DAYS`); ~10 days on a fresh install | At most ~4 weekly repetitions per slot |
| Recency weighting | None — flat mean | A change of habits takes ~4 weeks to be absorbed |
| Upper clamp | `TIME_PRIOR_MAX_BOUND = 0.9` | A permanently occupied hour saturates; long stuck-`on` sensors are indistinguishable from real occupancy |
| Period start | First *occupied* interval, not window start | An area idle at the start of the window has those hours excluded from the denominator, biasing its prior upward |
| Slot resolution | 60 min (`DEFAULT_SLOT_MINUTES`) | No sub-hour structure |

The nominal lookback is `DEFAULT_LOOKBACK_DAYS = 60`, but raw intervals older
than 28 days are rolled into daily aggregates and deleted, and the forecast reads
only raw intervals — so 28 days is the real ceiling.

## Lovelace card

`lovelace/area-occupancy-time-priors-card.js` renders the weekly matrix as a 7×24
heatmap. It defaults to `metric: "raw"` and `scale: "area"` — plotting the
uncompressed time prior, with the colour ramp stretched over the area's own
min–max — because absolute-scaled combined values make every low-prior room look
uniformly cold. Slots with `data_points = 0` are hatched as *no data* and are
excluded from both the ramp and the comfort-hours total.

See `lovelace/README.md` for installation and the full option list.

## Related documentation

- [Time Prior Flow](time-prior-flow.md) — how the weekly matrix is learned and stored
- [Global Prior Flow](global-prior-flow.md) — the area-wide term
- [Services](../features/services.md) — service reference
- [Prior Learning](../features/prior-learning.md) — user-facing overview
