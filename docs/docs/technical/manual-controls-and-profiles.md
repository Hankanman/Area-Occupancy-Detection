# Manual Controls and Combined Profiles

## Manual state

Manual state is stored with Home Assistant's `Store`, keyed by the config
entry and area name. Each record contains the requested boolean state, the
set time, and an absolute expiry time. On restore, invalid and expired
records are ignored.

The coordinator resolves visible occupancy in this order:

1. An active manual override, if present.
2. The normal `Area.occupied()` result.

The normal probability path is never given the manual value. Consequently,
manual corrections cannot train priors or sensor likelihoods. A scheduled
expiry refreshes entities and makes the transition back to automatic state
observable without waiting for a sensor event.

## Combined usage profiles

`purpose_profiles` is an additive configuration field. Older configurations
that only contain `purpose` continue to behave as before. Profiles are
normalised in input order and duplicate or unknown values are discarded. The
first profile is retained as the legacy primary purpose for display and
purpose-specific health rules.

When the half-life is set to the purpose default (`0`), the effective default
is the **maximum** half-life across the selected profiles. This conservative
rule prevents a room used for a slower activity from releasing early. An
explicit half-life still wins and remains unchanged. No learned probability
or historical data is rewritten when profiles change.
