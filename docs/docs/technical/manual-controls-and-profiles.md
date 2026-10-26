# Manual occupancy references and combined purpose profiles

## Manual reference data

Reference data is stored with Home Assistant's `Store`, keyed by config entry.
Each record contains a generated ID, area ID, UTC recording time, confirmed
state, and the automatic probability captured at that instant when available.

The coordinator never reads a reference into `Area.occupied()` or the
probability calculation. Binary sensor state, probability, recorder input,
priors, likelihoods, and online learners therefore remain automatic. Entity
attributes expose the latest reference separately for quality evaluation.

The store retains 365 days and at most 1,000 records. Unknown-area and
malformed records are ignored during restore and save. Same-state submissions
for one area within five seconds are rejected. Version 1 state-changing
overrides are intentionally discarded during the Store migration rather than
being reinterpreted as historical reference events.

## Combined usage profiles

`purpose_profiles` is an additive configuration field. Older configurations
that only contain `purpose` continue to behave as before. Profiles are
normalised in input order; duplicate and unknown values are discarded. The
first profile remains the legacy primary purpose for display and health rules.

When decay half-life is automatic (`0`), the effective default is the maximum
half-life across selected profiles. This conservative rule prevents a slower
room use from releasing early. An explicit half-life wins unchanged. Profile
changes do not rewrite learned probability or historical data.
