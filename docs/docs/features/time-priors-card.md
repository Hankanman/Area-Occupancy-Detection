# Time Priors Card

A small, dependency-free custom Lovelace card that visualises the learned weekly
occupancy forecast (7×24 = 168 slots per area) returned by the
`area_occupancy.get_time_priors` service.

It renders one heatmap per area (day-of-week × hour) coloured by learned
occupancy probability, with a **comfort-threshold** slider that highlights the
slots each area is habitually occupied — a quick way to see when a predictive
automation (e.g. climate pre-heating) would act.

## Requirements

Area Occupancy 2026.9.1 or later, which exposes the
[`get_time_priors`](services.md) service. The priors need recorder history to be meaningful — a
fresh install shows every slot hatched as *no data* until the hourly analysis has
learned something.

Builds that also return `slots_raw` and `data_points` enable the raw metric and
the no-data hatching. Against an older build the card falls back to the combined
`slots` map and renders every slot as learned.

## Install

Nothing to do: the card ships with the integration. On start-up Area Occupancy
serves it and loads it on every dashboard, so it appears in the card picker as
**Area Occupancy — Time Priors Heatmap** (or add it in YAML, below) without
copying a file or registering a resource.

It is served from
`/area_occupancy/frontend/area-occupancy-time-priors-card.js?v=<version>`. The
version in the query string changes with every release, so each update is a new
cache key and the browser always loads the card that shipped with the installed
integration — no hard refresh, and no stale card in the companion app.

### Upgrading from the manual install

If you installed the card by hand from an earlier release:

1. **Settings → Dashboards → ⋮ → Resources** — delete the
   `/local/area-occupancy-time-priors-card.js` resource.
2. Delete `/config/www/area-occupancy-time-priors-card.js`.
3. Reload the dashboard.

The old copy is never updated, so leaving it registered means two versions load.
If the old copy loads first it wins, and it's the one that never updates, so
remove it rather than rely on load order.

## Card options

```yaml
type: custom:area-occupancy-time-priors-card
title: Occupancy forecast   # optional
threshold: 50               # optional, comfort cutoff % (default 50)
refresh_minutes: 3          # optional, re-poll interval (default 3)
metric: live                # optional, "live" | "baseline" | "raw" (default "live")
scale: area                 # optional, "area" | "absolute" (default "area")
columns: auto               # optional, "auto" | 1 | 2 (default "auto")
# area_id: living_room      # optional, limit to one area
```

| Option | Meaning |
|---|---|
| `metric: live` | Evidence-conditioned forecast. The current slot (outlined as *now*) and the next one light up while the area is actually occupied, then relax back to habit |
| `metric: baseline` | The same forecast without live evidence: the stable weekly schedule |
| `metric: raw` | The learned time prior alone. Carries the weekly shape at full dynamic range — best for reading habits |
| `scale: area` | Colour ramp stretched over the area's own habitual min-max, and the threshold becomes a position within that range |
| `scale: absolute` | Colour ramp and threshold pinned to 0-100% |
| `columns` | Areas per row; `auto` uses two columns only when the card itself is wider than 1100px |

`metric: live` is the default because it answers the question you actually have
looking at the card: *is this room lit because someone is in it, or out of habit?*
The tooltip shows both — the live value and the habit for that slot. `scale: area`
is the default because on an absolute scale every low-prior room looks uniformly
cold: the blend keeps 60% of its weight on the global prior, so an area below
~0.19 can never reach 50% at any hour. See
[Occupancy Forecast](../technical/occupancy-forecast.md) for the maths.

The range and the comfort cutoff are always measured on the **stable** series
(`slots_baseline`), never on the series being drawn. Measuring them on the live
one made a single occupied slot set the area's maximum: with a room at 43% and a
habit spanning 5..10%, the other 167 cells normalised to the coldest colour and
the comfort total fell from 50 hours to 1, with the learned data unchanged. Live
values above the habitual maximum are clamped to the hot end of the ramp, so the
current slot still reads as occupied without flattening the rest of the week.

For the same reason the **h/week comfort** figure is counted on the habit: it
answers "how long would the heating run", which does not change because somebody
just walked into the room.

Since the live metric moves with the evidence, the card re-polls every 3 minutes
by default; raise `refresh_minutes` if you only care about the baseline. A failed
poll no longer blanks the card: the last good forecast stays on screen behind a
*stale* badge, and the card retries after 5s, 15s and 45s before falling back to
the normal interval.

Slots with `data_points: 0` were never observed; they are hatched as *no data*
and excluded from both the colour ramp and the comfort-hours total.

## Layout

The card sizes itself to the width Lovelace gives it (CSS container queries), so
it fits a narrow masonry column and a full-width panel alike. Note that
**masonry view caps column width** — for a genuinely full-width heatmap put the
card in a *Sections* view with `column_span`, or in a *Panel* view.

The card calls `get_time_priors` on load and every `refresh_minutes`; the
threshold slider re-renders instantly without re-fetching. Weekday labels follow
the browser locale.
