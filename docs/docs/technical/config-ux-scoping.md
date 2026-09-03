# Configuration UX: native flow vs. a custom panel

Scoping document, 2026-09-02. Not in the mkdocs nav; it is a working document
for maintainers, like `ha-config-flow-ui-reference.md`.

## Status (2026-09-03)

Delivered on `feat/config-ux`:

- Prerequisite: `DB_SCHEMA_VERSION` decoupled from `CONF_VERSION`, so a
  config-entry migration no longer wipes learned history.
- HA pinned to 2026.9.0, with the device-registry API changes that release
  requires (closes #530).
- Phase 0: `config_helpers.py` holds every hass-free validator and transform;
  the threshold number entity validates through it.
- Phase 1: hub-and-spoke editing with native menu descriptions, and a
  per-group sub-menu for additional sensors.
- Phase 2: live preview on the motion, sensors and behaviour pages of the
  options flow.

Open: Phase 3 scope (custom sensors vs. per-entity overrides), the optional
config-entities idea, and the Phase 4 subentry migration, which now has its
prerequisite in place. `hacs.json` still advertises a 2024.8.0 minimum; the
true minimum has not been established.

## Summary and recommendation

The per-area configuration surface is now 52 fields across a 4-step wizard,
and three open requests (#159 per-sensor active states, #458 per-sensor
weights, #531 arbitrary "custom" sensors) all want per-entity rows, which the
current wizard has no way to express. The question is whether to keep pushing
the native config/options flow or build a dedicated configuration panel.

**Recommendation: do the native work first, in phases, and gate the panel on a
concrete test.** Verified against HA core 2026.7.1 and the frontend `dev`
branch, the native toolkit has four capabilities this integration does not use
yet, and together they remove most of the pain:

| Native capability | What it fixes | Status |
|---|---|---|
| Menu steps with per-option descriptions and placeholders | Edit one section directly instead of walking all four wizard pages | Verified, `menu_option_descriptions` in `strings.json` |
| `ObjectSelector(fields=..., multiple=True)` | Per-entity rows (entity + active state + weight) rendered as an editable list | Verified in `helpers/selector.py` and `ha-selector-object.ts` |
| Form `preview=` with the generic preview component | Live "with these settings the room reads X%" while editing | Verified, domain-agnostic (`<domain>/start_preview`) |
| Config subentries (one per area) | Native area list on the integration page, per-area reconfigure/delete, device ownership | Verified; needs a migration and a DB-version decoupling first |

A custom panel is a second codebase (TypeScript, build tooling, CI, frontend
compatibility tax on every HA release) that a single maintainer has to carry
forever. It is the right call only if, after the native phases, the remaining
pain is "I want to see and tune every area on one screen" rather than "the
forms cannot express the config". The decision gate is in the last section.

The HA dependency pin can move to the latest release (2026.8.3 on PyPI at the
time of writing; it requires Python 3.14.2 or newer, which the project already
requires). Nothing in the plan needs anything newer than 2026.7.1, but the bump
is cheap and `hacs.json` currently advertises a 2024.8.0 minimum that is
already wrong for the selectors in use. That bump is its own PR because it
also has to move `pytest-homeassistant-custom-component` and regenerate
`uv.lock`.

## Where we are

### The surface

Per area (from `strings.json`, options flow):

| Step | Top-level fields | Fields inside collapsed sections | Sections |
|---|---|---|---|
| `area_basics` | 3 | 0 | |
| `area_motion` | 5 | 0 | |
| `area_sensors` | 0 | 34 | doors/windows/locks/covers, media, appliances, environmental (11 entity lists), power, wifi clients |
| `area_behavior` | 5 | 5 | wasp in box |

Plus 4 global fields and 5 fields per person. Every new sensor channel adds
two or three fields to `area_sensors` (locks in #526 and wifi clients in #525
did exactly that).

### The click path to change one number

To change a decay half-life on an existing area today:

1. Integrations page, Configure
2. Options menu: Manage areas
3. Radio list: pick the area
4. Area menu: Edit
5. Wizard step 1 (basics), Next
6. Wizard step 2 (motion), Next
7. Wizard step 3 (sensors), Next
8. Wizard step 4 (behaviour), change value, Submit

Eight dialogs, three of which carry no relevant content. The wizard is the
right shape for *adding* an area and the wrong shape for *editing* one.

### What is coming

- #159, #458, #531 all reduce to "a row per entity with its own active state
  and weight, and no domain restriction". The current schema stores flat
  `list[str]` entity lists per type with one shared active-state selector and
  one shared weight per type. There is no repeating-group widget in the current
  flow, so the maintainer's comment on #159 ("limited by the options available
  for config in the UI, it would get very clunky") is accurate for the widgets
  used so far, but not for the toolkit as a whole (see ObjectSelector below).
- #501 (learned fusion) plans to make most per-entity tuning unnecessary.
  Whatever UI is chosen should not bake in a large per-entity knob surface
  that #501 then has to hide again.

### What already exists that either path reuses

- `BaseOccupancyFlow._validate_config()` is the only server-side validator.
  It is a method on the flow class, so a panel or a websocket API cannot call
  it without instantiating a flow. Moving it (with the pure helpers
  `_apply_symmetric_adjacency`, `_strip_adjacency_references`,
  `_apply_purpose_based_decay_default`, `_update_area_in_list`,
  `_remove_area_from_list`) into a `config_helpers.py` module is a prerequisite
  for every option below and fixes the existing three-write-path threshold
  inconsistency noted in the architecture skill.
- `_async_entry_updated` in `__init__.py` already distinguishes structural
  changes (reload) from settings changes (in-place update). Any new write path
  should go through `hass.config_entries.async_update_entry` so this listener
  keeps doing that job.
- `simulator/app.py` already recomputes probability in-process from
  `Entity`/`EntityType`/`Decay`. That is the seed of a preview endpoint.
- `number.py::Threshold` is the precedent for a per-area config entity that
  writes back through `AreaConfig.update_config()`.

## Option A: native flow, phased

Effort figures are for one maintainer, working days, and include tests and
docs. Phases are independent except where stated, so each can ship alone.

### Phase 0: shared config module and HA pin bump (1 to 2 days)

- Extract validation and the list-transform helpers from `config_flow.py`
  into `config_helpers.py`. Pure functions, no `hass`. Existing tests move with
  them; `tests/test_config_flow.py` (81 tests) keeps passing unchanged apart
  from imports.
- Route `number.py::Threshold.async_set_native_value` through the shared
  validator so all threshold writers agree on bounds.
- Bump `homeassistant` in `pyproject.toml` to 2026.8.3, move
  `pytest-homeassistant-custom-component` in lockstep, regenerate `uv.lock`,
  and set `hacs.json` `homeassistant` to the real minimum.

### Phase 1: hub-and-spoke editing (3 to 4 days)

Keep the linear wizard for **add**. For **edit**, replace `area_action` with a
hub menu whose options are the four sections plus reset learning and remove:

```
Edit area: Living Room
  Basics                Purpose: Social · Adjacent: Kitchen, Hall
  Motion sensors        2 sensors · weight 1.0 · timeout 5m
  Additional sensors    Doors 1 · Media 1 · Appliances 0 · Environmental 3 · Power 0 · Wi-Fi 0
  Detection behaviour   Threshold 50% · decay on (purpose default) · wasp off
  Reset learning
  Remove area
```

The secondary line per option is native: `menu_option_descriptions` in
`strings.json`, and the frontend substitutes `description_placeholders` into
them (verified in `show-dialog-config-flow.ts`). Each spoke is one form that
saves on submit and closes the dialog. The click path from the section above
becomes five dialogs, none of them empty.

"Additional sensors" becomes a second hub (one option per sensor group) so no
form is longer than a dozen fields and nothing is collapsed by default. This
is also where a new sensor channel gets added in future: one small step, not
another collapsed section in a 34-field form.

Files: `config_flow.py` (new menu steps, spokes reuse the existing
`_create_*_step_schema` builders), `strings.json` and `translations/en.json`
(menu options and descriptions), `tests/test_config_flow.py`,
`docs/docs/getting-started/configuration.md`.

Risk: low. No storage change, no migration.

### Phase 2: live preview on the behaviour and motion spokes (2 to 3 days)

HA's generic flow preview is domain-agnostic: a flow returns
`async_show_form(..., preview="area_occupancy")`, defines
`async_setup_preview()` to register a websocket command named
`area_occupancy/start_preview`, and the frontend renders whatever
`state`/`attributes` the command streams (`flow-preview-generic.ts`,
`data/preview.ts`). It works for `config_flow`, `options_flow` and
`config_subentries_flow`.

The command would compute, from the live evidence the coordinator already
holds for that area plus the candidate weights/threshold/decay in
`user_input`, the probability and occupied/not-occupied result, and re-send
on every form change. The maths is the same call chain `Area.probability()`
uses; the simulator already shows how to run it outside the coordinator.

This is the one feature a custom panel would have offered that the native
form otherwise lacks, and it is the cheapest part of the whole plan.

Risk: low. Read-only; nothing is persisted from a preview.

### Phase 3: per-entity rows where they are wanted (3 to 5 days, decision needed)

`ObjectSelector(ObjectSelectorConfig(fields={...}, multiple=True,
label_field="entity", description_field=...))` renders an editable list of
rows, each row a small form, with add/remove/reorder. Verified in core
`helpers/selector.py` (2026.7.1) and `ha-selector-object.ts`.

Two ways to use it, which are not exclusive:

1. **Custom sensors (#531)**: a new `custom_sensors` list of
   `{entity, active_states, weight}` rows, any domain. Additive key, no
   migration, closes #531 outright and gives #159 users an escape hatch
   without touching the typed lists.
2. **Per-entity overrides (#159/#458)**: an additive `entity_overrides` map
   keyed by entity id, edited as rows of `{entity, weight, active_states,
   timeout}`. This is the knob surface #501 wants to make unnecessary, so
   ship it only if #501 stalls, or ship it as "advanced" behind the same
   learned-value shadowing #501 proposes.

Data model stays additive either way, so no `CONF_VERSION` bump, which
matters because of the next phase's finding.

Risk: medium. The row widget is newer than the rest of the toolkit; confirm
the minimum HA version that renders it before advertising it in `hacs.json`.

### Phase 4: config subentries, one per area (8 to 12 days, biggest structural win)

With `ConfigSubentryFlow`, each area becomes a subentry of the single entry.
The integration page then lists every area natively with its own
"Reconfigure" and "Delete", the "Add area" button is native, and each area's
device is linked to its subentry (`config_subentry_id` in the device
registry). This deletes the hand-rolled `manage_areas`, `area_action`,
`remove_area`, `confirm_remove_area` and related steps in both flows
(roughly fifteen step handlers), and the hub menu from Phase 1 becomes the
subentry's `reconfigure` step.

Verified behaviour that makes this fit the current architecture:

- Adding, updating or removing a subentry goes through
  `ConfigEntries._async_update_entry(subentries=...)`, which fires the
  entry's update listeners, so `_async_entry_updated` keeps deciding between
  reload and in-place update exactly as now.
- Subentry flows allow arbitrary internal steps and `async_show_menu`; only
  the entry points are restricted to `user` and `reconfigure`.
- `sections`, `menu_option_descriptions` and `preview` all work inside
  subentry steps (hassfest schema and the frontend `flow_type` union both
  include `config_subentries_flow`).

What it costs:

- **A `CONF_VERSION` bump with real data mutation** (18 to 19): move every
  item of `CONF_AREAS` into a subentry and link the existing device to it.
- **The DB reset trap.** `db/maintenance.py::_ensure_schema_up_to_date`
  compares the SQLite version stamp against `CONF_VERSION` and deletes the
  entire database on mismatch. Bumping `CONF_VERSION` for subentries would
  therefore wipe every user's learned priors. The migration must first
  introduce a separate `DB_SCHEMA_VERSION` constant (initialised to 18 so
  existing databases are kept) and make the maintenance check compare
  against that. This is a prerequisite commit, and it is worth doing
  regardless of subentries.
- `AreaConfig` reads from `subentry.data` instead of scanning a list, and
  `update_config` calls `async_update_subentry`.
- The four entity platforms iterate `entry.subentries` and pass
  `config_subentry_id` to `async_add_entities` instead of iterating
  `coordinator.areas`.
- `_apply_symmetric_adjacency` operates across subentries rather than a
  single list; still a pure transform, now over a dict.
- Tests: `test_config_flow.py`, `test_migrations.py`, `test_init.py`, the
  platform tests, plus a migration test seeded with a v18 entry.

Risk: medium-high because of the migration blast radius, but it is the only
option, native or custom, that gives a proper list-of-areas overview inside
HA's own settings UI.

### Optional: tunables as config entities on the device page

`Threshold` already exists as a `number` entity with
`EntityCategory.CONFIG`. The same pattern would put decay enabled (switch),
decay half-life (number, 0 keeps the purpose default), wasp enabled (switch),
purpose (select), minimum prior (number) and the per-type weights (number)
on the area's device page under "Configuration", where HA puts tunables for
most hardware integrations. Then the options flow only handles *structure*
(which entities belong to the area), and tuning never opens a form at all.

Cost: 10 to 15 extra registry entities per area, more translations, and the
"config surface is sacred" review bar applies because it is a visible
surface expansion even though every knob already exists. Value: tunables
become automatable and visible in one place per room. Worth a maintainer
decision after Phase 1 rather than a default.

## Option B: a dedicated configuration panel

Scoped so it can be compared honestly, and so that if the gate below says
"build it", the work is already laid out.

### Architecture

```
custom_components/area_occupancy/
  panel.py            register sidebar panel + static path (Alarmo pattern)
  websocket.py        admin-only websocket commands
  config_helpers.py   shared validation/transforms (Phase 0, needed here too)
  frontend/dist/      built panel bundle, shipped in the HACS zip
frontend/             TypeScript source, package.json, vite/rollup, not shipped
```

Registration (verified against 2026.7.1 signatures):

- `hass.http.async_register_static_paths([StaticPathConfig("/area_occupancy_static", <dist>, cache_headers=False)])`
- `panel_custom.async_register_panel(hass, frontend_url_path="area_occupancy", webcomponent_name="area-occupancy-panel", module_url=".../panel.js?v=<version>", sidebar_title=..., sidebar_icon=..., require_admin=True, config_panel_domain=DOMAIN)`
- `frontend.async_remove_panel` on unload.

The config flow must stay for first-run (`config_flow: true`), and the
options flow shrinks to a single step whose description links to the panel.
That is how Alarmo and the scheduler component do it.

### Websocket API (all `require_admin`)

| Command | Purpose |
|---|---|
| `area_occupancy/config/get` | Entry-wide snapshot: areas, global settings, people, plus HA area names and candidate entities per channel (reusing `_get_include_entities`) |
| `area_occupancy/config/area/save` | Validate via `config_helpers`, apply symmetric adjacency, write through `async_update_entry`; the existing update listener handles reload vs. in-place |
| `area_occupancy/config/area/remove` | Same path as `_remove_area_from_list` plus `_strip_adjacency_references` |
| `area_occupancy/config/global/save`, `.../people/save` | As above for `IntegrationConfig` keys |
| `area_occupancy/area/subscribe` | Push probability, evidence, decay and activity per area for the live view |
| `area_occupancy/area/preview` | Same computation as Phase 2, callable outside a flow |
| `area_occupancy/area/reset_learning` | Wraps the existing purge service |

### Frontend

Lit web component. The tempting shortcut is to reuse HA's own
`ha-entity-picker`, `ha-area-picker`, `ha-form` and `ha-selector` elements at
runtime. They are lazy-loaded internals, so a panel has to force-load them
(the `loadHaForm` trick Alarmo uses) and breaks whenever the frontend renames
one, which has happened to Alarmo and the scheduler card repeatedly. The
safer choice is to bundle the pickers the panel needs, which is more code to
own.

Screens: area list with status; area editor (tabs mirroring the four wizard
sections, per-entity rows for #159/#458/#531); global settings; people; a
live "why is this room occupied" view. The last one is the only screen that
has no native equivalent.

### Build, release, test

- Node toolchain in CI; the HACS release workflow must build before zipping
  (`hacs.json` has `zip_release: true`).
- Decide whether `frontend/dist` is committed (simple for HACS installs from
  source, noisy diffs) or built only in the release workflow.
- Backend tests with `hass_ws_client` from
  `pytest-homeassistant-custom-component`; panel tests with Playwright
  against the devcontainer HA instance. Neither exists today.

### Effort and ongoing cost

| Item | Days |
|---|---|
| Phase 0 shared helpers (shared with Option A) | 1 to 2 |
| Panel registration, static serving, build tooling, CI | 2 to 3 |
| Websocket API with tests | 3 to 4 |
| Area list + area editor (52 fields, pickers, validation errors) | 8 to 12 |
| Global settings, people | 1 to 2 |
| Live view + preview | 3 to 4 |
| Docs, screenshots, migration of the configuration guide | 2 to 3 |
| **Total** | **20 to 30** |

Ongoing: every HA frontend release is a potential breakage if HA internals
are reused; every new config key is now implemented twice (flow for first
run, panel for editing) unless the first-run flow is cut down to "pick an
area and one motion sensor" and everything else lives in the panel.

### What the panel does not fix

- It does not change the data model. Per-entity rows still need the additive
  keys from Phase 3.
- It does not give the integration page a native area list; that is
  subentries.
- It adds a second place where configuration can be edited, so the
  `_async_entry_updated` contract becomes load-bearing for two writers.

## Comparison

| | Native phases 0 to 3 | Native phase 4 (subentries) | Custom panel |
|---|---|---|---|
| Effort (days) | 9 to 14 | 8 to 12 | 20 to 30 |
| New toolchain | none | none | Node, TS, Playwright |
| Migration / DB risk | none | `CONF_VERSION` bump; needs DB-version decoupling first | none |
| Fixes 8-dialog edit path | yes (5 dialogs) | yes (3 dialogs) | yes (0 dialogs, sidebar) |
| Per-entity rows (#159/#458/#531) | yes (ObjectSelector) | yes | yes |
| Live preview while editing | yes (generic preview) | yes | yes |
| Overview of all areas at once | no (one list, one at a time) | partial (integration page list with titles) | yes |
| Live "why occupied" view | no | no | yes |
| Ongoing compatibility tax | HA form widgets, maintained by HA | same | frontend internals, maintained by us |
| Reversible | yes | migration is one-way | yes, but users lose the panel |

## Decision gate for the panel

Build the panel only if, after Phases 0 to 2 have shipped and been used for
one release cycle, at least one of these is true:

1. Users still ask for a single screen showing and tuning all areas, and the
   subentry list (Phase 4) is judged not enough.
2. The live "why is this room occupied" view becomes a product priority,
   because that is the only screen with no native equivalent.
3. Per-entity configuration grows past what one ObjectSelector row can hold
   (for example per-entity likelihood curves once #501 lands and needs
   inspection rather than editing).

If none holds, the native path is complete at Phase 4 and the panel is not
built.

## Immediate next steps

1. Phase 0 PR: `config_helpers.py` extraction, threshold write-path
   consolidation, HA pin to 2026.8.3 with lockfile regeneration.
2. Phase 1 PR: hub-and-spoke edit menus with descriptive secondary lines.
3. Phase 2 PR: preview command and `preview="area_occupancy"` on the
   behaviour and motion spokes.
4. Decide on Phase 3 scope (#531 custom sensors first; per-entity overrides
   only if #501 stalls) and on the optional config-entities idea.
5. Prerequisite PR for Phase 4: introduce `DB_SCHEMA_VERSION` and stop tying
   the SQLite reset to `CONF_VERSION`. Then the subentry migration.

## Verification notes

Checked on 2026-09-02 against HA core tag `2026.7.1` (the current pin) and
the frontend `dev` branch, by fetching the source files directly because the
sandbox could not install Python 3.14.2:

- `homeassistant/config_entries.py`: `ConfigSubentryFlow`,
  `async_get_supported_subentry_types`, `async_update_and_abort`,
  `_get_reconfigure_subentry`; `async_add_subentry`/`async_update_subentry`
  route through `_async_update_entry`, which calls update listeners.
- `homeassistant/helpers/selector.py`: `ObjectSelectorConfig` has `fields`,
  `multiple`, `label_field`, `description_field`, `translation_key`.
- `homeassistant/data_entry_flow.py`: `async_show_form(preview=...)`,
  `_async_setup_preview`, `async_show_menu(description_placeholders=...)`.
- `homeassistant/components/threshold/config_flow.py`: reference
  implementation of `async_setup_preview` + `<domain>/start_preview`.
- `homeassistant/components/frontend/__init__.py`,
  `panel_custom/__init__.py`, `http/__init__.py`: panel and static path
  signatures.
- `homeassistant/helpers/device_registry.py`: `config_subentry_id`.
- `script/hassfest/translations.py`: `menu_option_descriptions` and
  `sections` accepted in step translations.
- Frontend `src/data/preview.ts`, `previews/flow-preview-generic.ts`,
  `step-flow-form.ts`, `step-flow-menu.ts`,
  `show-dialog-config-flow.ts`, `ha-selector-object.ts`.
- This repo: `db/maintenance.py::_ensure_schema_up_to_date` compares the DB
  stamp to `CONF_VERSION` and deletes the DB on mismatch.
- PyPI: latest `homeassistant` is 2026.8.3, `requires_python >= 3.14.2`.
- Alarmo's `panel.py` for the panel registration pattern.
