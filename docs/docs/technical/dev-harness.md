# Test Harness

`scripts/harness` builds throwaway Home Assistant instances for testing the
integration against a real Home Assistant, rather than against the mocks a
unit test provides.

Each instance is one directory under `instances/` (gitignored) containing a
generated `configuration.yaml`, a seeded `.storage`, a seeded
`area_occupancy.db`, the logs, and a `harness.json` recording how it was
built. Destroying an instance is a single recursive delete, so there is no
state to clean up by hand and nothing shared with the repo's own `config/`
development instance.

## Why it exists

Unit tests cover the calculation and the configuration handling, and
`scripts/develop` gives one hand-configured instance to click through. Between
those two sits a class of failure neither reaches:

- a selector schema Home Assistant core rejects at render time, so the form
  returns HTTP 400 and the user sees nothing
- a menu step id with no matching `async_step_*` handler behind it
- an entity that ends up filed under the wrong config subentry, or none
- a migration that runs but does not persist
- a learned value that comes out pinned at a probability bound

All of those need a real instance, real storage and the real frontend API.

## Quick start

```bash
# Build a five-area instance with two weeks of history and start it
scripts/harness new

# Same, but seeded as a pre-subentry (v18) entry so startup runs the
# real migration
scripts/harness new --name upgrade --entry-version 18

# Build, run every check, report, and delete it again
scripts/harness verify

# What can be built
scripts/harness profiles
```

`new` and `start` print the instance's URL, its login and a bearer token.
The instance listens on a free port, so several can run at once.

## Commands

| Command | What it does |
| --- | --- |
| `new` | Create an instance and start it. `--no-start` to only build it, `--detach` to leave it running in the background. |
| `start` | Start an existing instance. |
| `stop` | Stop it, waiting for a clean shutdown. |
| `info` | Print what an instance is and whether it is running. |
| `destroy` | Stop and delete it. |
| `profiles` | List the available profiles. |
| `verify` | Build an instance, run every check, print a report, delete it. `--keep` to keep it, `--reuse` to check an existing one. |
| `shots` | Capture the documentation screenshots from a running instance into `docs/docs/images/`. |
| `upgrade-base` | Build a lived-in instance on a released version (`--from <tag>`) and snapshot it. See [Upgrade testing](#upgrade-testing). |
| `upgrade` | Restore that snapshot, swap in another ref (`--to next`, `--to working`), and report what the upgrade changed. |

Build options (`new` and `verify`): `--profile`, `--entry-version`, `--days`,
`--seed`, `--time-zone`, `--port`, `--no-frontend`, `--no-priors`, `--force`.

## Profiles

A profile declares which areas exist, which mock sensors each has, and how
those sensors behave relative to occupancy. It is the single source of truth
for an instance: the generated YAML, the seeded config entry and the
synthesised history are all derived from it, so adding a channel to a profile
adds it in all three places.

| Profile | Contents |
| --- | --- |
| `minimal` | One social area, motion only. Fastest to boot. |
| `single` | One area with every channel. For sensor-group and preview work. |
| `house` | Five adjacent areas covering every purpose. The default. |

Profiles live in `harness/profiles.py`. Adding one is a `Profile` entry in
`PROFILES`; adding a sensor class is a `ChannelSpec` in `CHANNELS`.

## What gets seeded

### Mock entities

Every configured sensor is a template entity (or a universal media player)
over an `input_boolean`, `input_select` or `input_number` helper. Moving the
helper moves what the integration sees, which is how an instance is driven by
hand or from a script:

```yaml
input_boolean.mock_living_room_motion_1   ->   binary_sensor.living_room_motion_1
input_select.mock_living_room_media_1     ->   media_player.living_room_media_1
input_number.mock_living_room_co2_1       ->   sensor.living_room_co2_1
```

Only the components the harness needs are configured -- no `default_config`
-- so an instance boots in seconds and its log is not buried in
missing-dependency errors for integrations nobody is testing.

### The config entry

Written straight into `.storage/core.config_entries` before Home Assistant
starts, along with one Home Assistant area per profile area. This is the only
way to get a **historical** entry version: the config flow can only ever
create an entry at the current `CONF_VERSION`, so testing the v18 to v19
migration means fabricating a v18-shaped entry whose areas are still in the
legacy `areas` list.

```bash
scripts/harness new --entry-version 18   # areas in the legacy list
scripts/harness new --entry-version 19   # one config subentry per area
```

Only the keys a profile actually decides are written. Everything else is left
absent so the integration's own defaults apply -- which is both closer to a
hand-configured entry and a standing test that those defaults hold.

### Learned history

An instance with no history is only half an instance: priors sit at the
default and nothing that depends on learning can be looked at. `--days N`
synthesises N days of it.

Occupancy is generated per area as a two-state Markov chain: the rate out of
"occupied" comes from the profile's `mean_visit_minutes`, and the rate into it
is solved so the stationary occupancy matches the hourly routine. Visits
therefore have realistic dwell times while the day still adds up to the
routine asked for. Every sensor is then a second chain conditioned on that
occupancy, with its own dwell time, so a motion sensor stays on for a couple
of minutes rather than flickering every slot.

Because a sensor with a dwell time cannot track occupancy exactly, the
realised correlation always drifts from the profile's declared
`p_active_occupied` / `p_active_empty`. The generator measures what it
actually produced and seeds *that* as each entity's likelihood, so the
database is internally consistent with its own intervals.

Rows go into the integration's own SQLite database -- the same `intervals`
table the recorder sync fills -- so everything downstream runs on them
unchanged: the occupied-interval cache, priors, correlation analysis and
transition learning. The database is stamped with `DB_SCHEMA_VERSION`, so the
integration adopts it rather than deciding it is stale and recreating it.

`--no-priors` seeds the intervals but not the priors they imply, which is the
state a real install is in before its first analysis run.

## Verification

`scripts/harness verify` builds an instance, runs every check and prints a
report. Nothing raises on a failure, so one run reports every problem rather
than only the first, and the exit code is non-zero if any check failed.

| Check | What it holds the integration to |
| --- | --- |
| `entry_loaded` | The seeded entry set up without error. |
| `migration` | A legacy-seeded entry reached the current version, dropped the legacy key, and has one subentry per area. |
| `mock_sensors` | Every entity the config points at exists. |
| `entities` | Every area exposes its entities and none are unavailable. |
| `sensor_response` | Turning a motion sensor on raises that area's probability. |
| `options_flow` | Every reachable options-flow step renders. |
| `subentry_flow` | Every reachable step of an area's reconfigure flow renders. |
| `analysis` | `run_analysis` completes and derives a prior for every area that is not pinned at a bound. |
| `subentry_linkage` | Each area's entities are filed under that area's subentry, and the aggregates are not. |
| `transitions` | Adjacency rows name real areas (not Home Assistant area ids), and the analysis learned transitions from the seeded history. |

The flow checks walk the menu graph breadth-first, starting a fresh flow for
each path -- a flow is a state machine, so stepping into one spoke rules out
its siblings. Only menu navigation is ever submitted, never form data, so the
walk cannot change the instance's configuration.

## Upgrade testing

`--entry-version 18` tests one thing: that the current code migrates an entry
the *harness* fabricated in the old shape. It cannot catch the upgrade bugs
that matter most, the ones in state a previous release actually wrote: the
config entry its config flow produced, its persisted stores, its database,
and the entity registry after a user has renamed and enabled things. Before
every release, run the real thing:

```bash
# Once per base release: install 2026.8.1's code, let it live a little
scripts/harness upgrade-base --from 2026.8.1

# Then as often as needed: restore that baseline, upgrade, diff
scripts/harness upgrade --to next
scripts/harness upgrade --to working       # the working tree, symlinked
```

`upgrade-base` builds `instances/upgrade` like this:

1. Installs the `--from` ref's integration as a real copy of
   `git archive <ref>`, not the usual symlink to the working tree.
2. Boots once to onboard (`dev` / `devpassword`) and create the recorder
   schema, then backfills the profile's synthetic history into the
   **recorder** rather than the integration's database, so the old code
   imports and learns it itself. Two floors are added so floor sensors exist.
3. Adds the integration through that version's **real config flow**. Each
   form's schema comes back from Home Assistant, and every field the
   profile's area config has a value for is filled in (inside collapsible
   sections too). A few hand-set values are layered on top so the migration
   has to carry them: a custom decay half-life and threshold, a custom motion
   timeout, an area excluded from "All Areas", an inverted door, a minimum
   prior override and wasp-in-box. Config keys the old flow has no field for
   are listed and skipped.
4. Adds a person with sleep detection through the options flow.
5. Makes the changes a user would: enables two areas' diagnostic entities,
   renames and re-ids the "All Areas" occupancy sensor, and moves a threshold slider.
6. Walks the house live (motion on, area by area) so the trajectory tracker,
   online prior and shadow learners have real ticks, then runs the analysis.
7. Captures everything below as `captures/baseline.json` and snapshots the
   stopped instance to `instances/upgrade@baseline`.

`upgrade` restores `instances/upgrade@baseline` (skip with `--no-restore`),
swaps in the `--to` ref, boots it, lets it run for `--settle` seconds, runs
the analysis, captures again, and writes
`captures/baseline--<ref>.md`. It compares:

| Area | Flagged when |
| --- | --- |
| Config entry | An area's stored value changed or disappeared (including areas moving into subentries), or the people config changed. |
| Entity registry | An entity lost its registry entry, or its entity id, name, or enabled/hidden state changed. Keyed by unique id; a unique-id change is reported as a re-key. |
| Entity states | An entity that had a state became `unavailable`/`unknown`. |
| Services | A service disappeared. |
| Persisted stores | A store vanished or lost a non-empty field (online prior, fusion). |
| Integration database | Never flagged. Row counts, metadata and priors are reported so a deliberate `DB_SCHEMA_VERSION` reset is visible. |
| Diagnostics | The download fails. |
| Log | Any `ERROR`, or any `WARNING` from the integration, since the upgraded boot. |

Additions are reported but never flagged: new entities, stores and config
keys are what an upgrade is for. When the target writes the working tree's
`CONF_VERSION`, the standard [checks](#verification) run afterwards too
(`--no-checks` skips them), so the `migration` and `subentry_linkage` checks
hold a real old entry to the same bar as a fabricated one. They encode the
working tree's expectations, so they are skipped for any other target.

`scripts/harness upgrade --to <the base's own tag>` is the null test: it
should report no red flags, which is what shows the diff does not cry wolf. The exit code is non-zero on any red flag or failed check,
and the upgraded instance is left running to poke at.

The config-flow driver speaks the 2026.7-2026.8 single-entry wizard (an
area wizard, then an `add_area` / `finish_setup` menu). Once a release using
subentries becomes the base, `run_config_flow` in `harness/upgrade.py` needs
a subentry-flow branch.

## Three things worth knowing

**A pinned port has to be the *confirmed* HTTP config.** Since 2026.9 the
HTTP configuration is a user-managed store holding a confirmed `stable`
config and an unconfirmed `pending` one. A port arriving any other way -- in
YAML, or as a changed built-in default -- is staged as a pending *trial*, and
if nothing promotes it within five minutes Home Assistant reverts to stable
and restarts itself to do it. An instance whose port came from YAML therefore
died five minutes in, and the next start looked for a port nothing was
listening on. The harness seeds the port into `stable` with no pending
config, and marks the YAML migration done so a stray `http:` block cannot
restage it.

**Home Assistant defers writes while it is starting.** Stores and registries
are not flushed until it reaches `RUNNING`, so reading `.storage` before then
shows pre-startup state -- a migration that ran looks like one that never
happened. The harness waits for `RUNNING` before any check reads from disk,
and `stop` waits for a clean shutdown rather than killing the process.

**Entity ids are not stable across Home Assistant versions.** Since 2026.9
core builds them from area name, device name and entity name, so an area
device named after its area produces
`sensor.living_room_living_room_occupancy_probability`. The checks match on
friendly name instead, which stayed stable.

## Documentation screenshots

Screenshots rot: the config flow changes, the pictures do not, and a reader
following a walkthrough goes looking for a button that moved two releases ago.
`scripts/harness shots` re-takes them from a seeded instance, so refreshing the
whole set after a UI change is one command:

```bash
scripts/harness new --name shots --profile house --days 14 --detach
scripts/harness shots --name shots
scripts/harness destroy --name shots
```

It needs Playwright, which is an optional extra (`uv sync --extra shots`);
everything else in the harness works without it. The shot list lives in
`harness/screenshots.py` — each entry is the clicks to reach a screen and the
element to crop to. `--only <name>` re-takes a single one.

Because the shots come from a seeded instance, they show a populated
installation rather than an empty one: five areas with their devices and
entities, learned values in the summaries, and a live preview with a real
number in it.

## Adding a check

Checks live in `harness/verify.py`. A check takes the instance and a client,
returns a `Result`, and is added to `LIVE_CHECKS` (runs against the live
instance) or `STOPPED_CHECKS` (runs after shutdown, for anything that reads
`.storage`).

The client in `harness/client.py` is a small stdlib-only REST wrapper with
helpers for states, services, config entries and the three flow managers, so
a check usually needs no HTTP of its own.
