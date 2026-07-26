# Lovelace card — Time Priors Heatmap

A small, dependency-free custom Lovelace card that visualises the learned weekly
occupancy forecast (7×24 = 168 slots per area) returned by the
`area_occupancy.get_time_priors` service.

It renders one heatmap per area (day-of-week × hour) coloured by learned
occupancy probability, with a **comfort-threshold** slider that highlights the
slots each area is habitually occupied — a quick way to see when a predictive
automation (e.g. climate pre-heating) would act.

![weekly heatmap](../docs/docs/images/) <!-- add a screenshot if desired -->

## Requirements

The Area Occupancy build that exposes the `get_time_priors` service
(`SupportsResponse.ONLY`). The priors need recorder history to be meaningful — a
fresh install shows the default (~0.5) everywhere until it has learned.

## Install

1. Copy `area-occupancy-time-priors-card.js` to `/config/www/`.
2. **Settings → Dashboards → ⋮ → Resources → Add Resource**
   - URL: `/local/area-occupancy-time-priors-card.js`
   - Type: **JavaScript Module**
3. Hard-refresh the browser (Ctrl/Cmd + Shift + R).
4. Add the card to a dashboard.

## Card options

```yaml
type: custom:area-occupancy-time-priors-card
title: Occupancy forecast   # optional
threshold: 50               # optional, comfort cutoff % (default 50)
refresh_minutes: 10         # optional, re-poll interval (default 10)
# area_id: living_room      # optional, limit to one area
```

The card calls `get_time_priors` on load and every `refresh_minutes`; the
threshold slider re-renders instantly without re-fetching. Weekday labels follow
the browser locale.
