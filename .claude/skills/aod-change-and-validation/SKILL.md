---
name: aod-change-and-validation
description: Use before opening, reviewing, or merging any PR against Area Occupancy Detection (AOD) — especially anything touching Bayesian math (utils.py sigmoid pipeline), priors, decay/half-life, likelihoods, config schema (const.py CONF_*/DEFAULT_*), the config-entry migration ladder (migrations.py), or the DB schema (db/schema.py). Also load when deciding how to reply to CodeRabbit, when bumping the release version, when someone proposes a new per-sensor config knob, or when you're unsure whether a change needs a version bump, a migration, or extra validation before merge. Also use when writing, reviewing, or deciding what evidence is required for ANY change to Area Occupancy Detection — before opening a PR, when asked "is this tested enough?", when adding a test for math/behavior/config/DB code, when coverage or CI is failing, or when a reviewer/CodeRabbit flags a test as rigged, redundant, or encoding a bug. Covers the evidence bar per change class, conftest.py fixture anatomy, the golden test-file map, and lint/coverage gates. Also use when turning a hunch, user report, or measurement into an accepted change to AOD's Bayesian/learning math — deciding whether a root-cause theory is proven, designing the discriminating experiment (DB query, simulator run, or predicted-numbers test), shipping an experimental capability behind a config flag, or adversarially refuting a claimed bug/fix. Also load this before trusting any "verified" claim from a single home's data or a single refutation pass.
---

# AOD Change and Validation

## What this covers

How changes get classified, validated, and merged in this project: the three
non-negotiable rules the retiring maintainer stated explicitly (with the
incident behind each), how CI gates actually work today (not what AGENTS.md
says), the version-bump and release mechanics, and the CodeRabbit-reply
convention — "am I allowed to ship this, and how" — process, not
implementation detail. It also covers the evidence bar this project actually
enforces before a change merges: what counts as proof for docs vs. behavior
vs. math changes, how the test suite in `tests/` is organized (fixtures,
markers, coverage), how to add a test for each architectural layer, and the
map of which test file guards which invariant — including two real, cited
anti-patterns from this repo's own PR history that look like passing tests
but prove nothing. Finally, it covers the discipline that turns a hunch about
occupancy-detection accuracy into an accepted, merged change: the evidence
bar a hypothesis must clear, how ideas here are validated with predicted
numbers *before* code is written, the idea lifecycle from issue to release
notes, and the anti-patterns that have cost real time (fixing symptoms, tests
that encode the bug, trusting one home's data). It is written for a
zero-context AI session picking up this project cold.

## When NOT to use this

- For *how* the Bayesian pipeline actually computes (sigmoid/logit math; the
  now-deleted `bayesian_probability()` legacy engine and its removal in PR
  #529) or the actual Bayesian formulas to hand-compute against, use
  `aod-math-reference`.
- For the historical blow-by-blow of specific incidents (full saga
  narratives, root causes, stale branches) or understanding *why* a past bug
  happened and what invariant it taught us, use `aod-debugging-and-history`.
- For how to run tests/lint/CI locally, use `aod-build-run-and-release`.
- For what "beyond SOTA" claims are and aren't earned yet, use
  `aod-docs-and-positioning`.
- Root-causing a live/reported bug step by step, or a bug with a known,
  obvious mechanism and no live debate about root cause, use
  `aod-debugging-and-history`.
- Building or running the cross-check simulator / correlation tooling itself,
  use `aod-math-reference`.
- You want the current list of open research questions / frontier ideas
  themselves, not the method for validating them, use `aod-research-frontier`.
- You're running an accuracy campaign across many homes/areas, use
  `aod-debugging-and-history`.

This skill owns: classification rules, gates, the three unwritten laws,
versioning/release mechanics, the CodeRabbit convention, the evidence bar per
change class, conftest.py fixture anatomy, the golden test-file map,
lint/coverage gates, and the hypothesis-validation research methodology.

---

## 0. Orientation trap — read this first (SETTLED 2026-07-06)

The environment banner and AGENTS.md may both claim you're on `main` with a
clean tree. **Verify, don't trust**: `git status` and `git branch
--show-current`. This was a live trap through 2026-07-06: the repo was
checked out on `feat/adjacent-areas` (PR #454) for an extended period, and an
entire feature's worth of code (adjacency boost/decay-modifier,
`db/transitions.py`, `data/adjacency.py`, ~10 `ADJACENCY_*` constants in
`const.py`) existed only on that branch and was invisible if you assumed the
banner was correct. **PR #454 merged to `main` on 2026-07-06** (main HEAD is
now `17b71d2`, "feat: adjacent-areas — learned next-door room influence
(#454)"); the working tree is on `main` and the adjacency code is now part
of it. The general lesson still applies for the *next* long-lived feature
branch — before citing any file content as "what's in main," run `git show
main:<path>` or `git diff main...HEAD -- <path>` to check rather than trust
the banner.

---

## 1. The three unwritten laws

These come directly from the maintainer. They are not aspirational — each has
a real incident behind it, and violating one is treated as an unreviewable
mistake, not a style nit.

### Law 1 — NO SILENT MATH CHANGES

**Rule**: any change to Bayesian/prior/decay/likelihood output must be
validated with predicted numbers *before* running — either against the
simulator (`simulator/app.py`, paste a `run_analysis` service response and
watch the recomputed probability) or against real recorder data
(`scripts/visualize_distributions.py`). State what you expect the numbers to
do, then run it, then compare. (This is also the core discipline behind the
research-methodology "hypothesis-predicts-numbers-before-running" house
style in §9 below — Law 1 is its canonical statement for math-affecting code
changes specifically.)

**Incidents**:

- **Prior inflation (issue #483 → PR #491, merged 2026-07-06, SETTLED)**:
  `PriorAnalyzer.calculate_and_update_prior()` (`data/analysis.py`) truncated
  the observation window at `last_interval_end` whenever an area had been
  quiet more than an hour — dropping the known-unoccupied tail from the
  denominator while the numerator kept all occupied time. Every hourly
  recalculation during a quiet stretch (overnight, weekends) pushed
  `global_prior` further toward the 0.99 cap. A real kitchen with true
  occupancy ~28–35% pinned at 0.99. The existing test
  (`test_valid_calculation_sets_correct_prior`) had **encoded the bug** —
  asserting 0.99 was correct — and had to be corrected to assert the true
  0.25 as part of the fix. Fix on main: `actual_period_end` is now
  unconditionally `now` (the old conditional truncation is gone; the
  degenerate-period-after-startup case it was guarding is already covered by
  the `actual_period_duration <= 0` check), and a regression test for the
  overnight quiet-tail case was added. Verify: `gh pr view 491 --json
  state,mergedAt`.
- **Half-life override silently discarded (issue #481 → PR #493, merged
  2026-07-06, SETTLED)**: `Decay._resolve_purpose_half_life()`
  (`data/decay.py`) has a sleep/awake split for the Bedroom/SLEEPING purpose
  (uses the area's half-life during the sleep window, switches to the
  purpose's fixed `awake_half_life`=620s outside it). The switch applied
  **unconditionally**, so a user's custom 10s half-life silently became a
  620s (~10 minute) clear-out during waking hours. This is the *same bug
  class* recurring in a different code path: issue #439/PR #440 already
  fixed an identical "custom value gets silently normalized to a purpose
  default" bug for the general half-life field months earlier; #481 was the
  sleep/awake switch not respecting that established custom-vs-default
  semantics. Fix on main: `_resolve_purpose_half_life()` now carries an
  explicit "if the resolved value != the purpose default, return the base
  (custom) value" guard before the sleep/awake switch applies, with the
  adjacency modifier_factor still multiplying on top of whichever half-life
  (custom or default) that guard resolves to. Verify: `gh pr view 493 --json
  state,mergedAt`.

**Takeaway**: a test that encodes buggy output (asserts the bug's number) is
not evidence a math change is safe — it's evidence nobody predicted the
correct number independently before the test was written. A green test suite
proves nothing if the assertions themselves are wrong. When you touch
`utils.py` (sigmoid/logit pipeline), `data/prior.py`, `data/decay.py`, or
correlation likelihood code in `db/correlation.py`, compute the expected
output by hand or via the simulator *before* trusting the test suite's green
checkmark.

### Law 2 — NEVER BREAK USER CONFIGS

**Rule**: migrations must be idempotent (safe to run twice), missing keys
must be defaulted via `.get(KEY, DEFAULT)` rather than assumed present, and DB
schema changes must not be destructive unless a `DB_SCHEMA_VERSION` bump is
deliberately accepted (which wipes the whole DB — see below).

**Why a DB version bump is dangerous**: `db/maintenance.py`'s
`_ensure_schema_up_to_date` treats any mismatch between the stored DB version
and `DB_SCHEMA_VERSION` (`const.py`) as fatal — it **deletes and recreates the
entire database** (all learned priors, correlations, intervals, aggregates,
for every area) rather than migrating it. There is no DB-level migration
script, unlike the config-entry migrations in `migrations.py`. Until the
`feat/config-ux` decoupling (2026-09) this check was keyed on `CONF_VERSION`,
so a config-entry migration also wiped the DB; that is no longer true, and a
`CONF_VERSION` bump on its own is now safe for learned data. Bumping
`DB_SCHEMA_VERSION`
for a schema change you don't actually need to gate is a real way to wipe
every user's learned history on upgrade.

**Precedent for additive-only schema changes**: the `AreaTransitions` table
and `Areas.adjacent_areas` column (feat/adjacent-areas, PR #454 — **merged to
main 2026-07-06**, verify with `gh pr view 454 --json state,mergedAt`)
deliberately does **not** bump `CONF_VERSION`, because the change is purely
additive: a new JSON column with a default-on-missing value in the loader,
plus a new table created via `Base.metadata.create_all(checkfirst=True)`.
The `migrations.py` comment states the rationale explicitly: bump the
version and every user's DB gets wiped for no reason, since
`checkfirst=True` on `create_all` only adds what's missing without touching
existing tables. `db/schema.py` on main now has 15 tables including
`AreaTransitions`, and `CONF_VERSION` was **not** bumped for this change
(still 18 — see §4). **Use `create_all(checkfirst=True)` for additive schema
changes; reserve a `CONF_VERSION` bump for changes that genuinely require the
nuclear wipe.**

**Config-entry migration pattern** (`migrations.py::async_migrate_entry`,
current on `main`): runs under a module-level `asyncio.Lock` (prevents
concurrent migrations), checks `config_entry.version` against `CONF_VERSION`
(currently 18 — `const.py`) in an explicit ladder of `if
config_entry.version == N` blocks, and every step calls
`hass.config_entries.async_update_entry(config_entry, version=N+1, ...)` —
re-running an already-migrated entry is a no-op because the version check
gates it. New optional config keys (e.g. v17→v18's
`exclude_from_all_areas`) are handled with **no data migration at all** —
the loader (`AreaConfig._load_config`) defaults the missing key via `.get()`,
and the migration step only bumps the version number. This is the preferred
pattern: prefer "loader defaults it" over "migration writes it," because the
former is idempotent by construction and works even for entries that skip
straight from an old version to the latest.

**Entity-registry state preservation subtlety (PR #488)**: PR #488 made 7
diagnostic sensor classes (`PriorsSensor`, `EvidenceSensor`, `DecaySensor`,
`PresenceProbabilitySensor`, `EnvironmentalConfidenceSensor`,
`ActivityConfidenceSensor`, `SensorHealthSensor`) register with
`entity_registry_enabled_default=False` for newly-registered areas, to cut
recorder write volume (issue #467). The PR body claimed "deleting and
re-adding an area counts as a new registration, so its diagnostics come up
disabled" — **this claim is wrong, and the PR's own shipped test disproves
it**: `tests/test_sensor.py::test_re_add_area_restores_previous_enabled_state`
demonstrates that HA's entity registry keeps removed entities in a
`deleted_entities` store and **restores the previous `disabled_by` state**
when the same `unique_id` is re-created — `async_get_or_create` passing
`disabled_by=INTEGRATION` (per `enabled_default=False`) is overridden by the
registry's restore of the prior state. **The lesson**: `deleted_entities`
restore beats `enabled_default` on re-registration. Don't assume
delete+re-add is equivalent to a fresh install for entity-registry purposes —
verify against the registry's actual restore behavior (or this test) before
describing re-add semantics in a PR body.

### Law 3 — CONFIG SURFACE IS SACRED

**Rule**: prefer purpose-based smart defaults over adding a new user-facing
config knob. Every new `CONF_*` option is a permanent commitment (it must be
defaulted forever via `.get()`, documented, and migrated correctly). The bar
for adding one is high.

**Precedent — declining a per-sensor config UI (issue #159, "Allow per
Sensor active states")**: maintainer's response (verify: `gh issue view 159
--json comments`):

> "this is something I would like as well, at the moment I am limited by the
> options available for config in the UI, it would get very clunky, very
> quickly, some integrations have built their own config UI, but this is an
> endeavour on its own, so not likely to happen too soon."

Per-entity granularity was rejected not on principle but on config-flow UI
cost — building a custom config UI to support it is treated as a separate,
larger undertaking than the feature itself.

**Precedent — partial-build decision (issue #466, "Configurable repair issue
thresholds + ability to suppress repairs per sensor")**: three asks came in;
the maintainer shipped the purpose-aware defaults (#474: bedrooms get 48h,
media rooms 32h, offices 24h vs 8h base; `media_player.*` exempted from the
unavailable check entirely) and the integration-level on/off toggle (#472)
plus sticky-ignore-across-flaps (#473) as smart defaults that avoid adding
new knobs, then explicitly declined to build per-sensor threshold
configurability and per-sensor suppression *yet*, verify with `gh issue view
466 --json comments`:

> "Per-sensor configurable thresholds: not implemented. Open question whether
> this is worth the config-surface cost given (3) covers the main offenders."
>
> "Leaving this open to track per-sensor suppression specifically. If the
> saner defaults + sticky-ignore combo turns out to cover the real-world
> cases, we can close this without building (1)."

**Takeaway**: when a feature request asks for new per-sensor/per-area knobs,
first ask whether a purpose-aware default, a smoothing/fallback mechanism, or
a single integration-level toggle can cover the real cases. Only add a new
`CONF_*` option after that's demonstrably insufficient — and say so
explicitly in the PR/issue, the way #466's response does.

---

## 2. Change classification

Classify every change before starting work — it determines what validation
and review depth is required.

| Class | Examples | Requires |
|---|---|---|
| **Docs-only** | `docs/docs/**`, `README.md`, docstring wording | Standard PR + CI; no math/behavior review needed. Note: `docs/docs/features/sensor-health.md` and `docs/docs/technical/database-schema.md` are *already* known-stale vs code (thresholds and retention numbers) — don't propagate their numbers into a new doc without re-verifying against the source file |
| **Test-only** | New/changed test cases, fixtures, no production code touched | Standard PR + CI; if the test encodes a previously-buggy expected value, treat it as a math-affecting change in disguise (see Law 1's #483 lesson) |
| **Behavior-affecting, non-math** | Health-check thresholds, repair-issue UX, service schemas, entity registry defaults | Standard PR + CI + manual verification of the specific behavior changed (e.g. `scripts/develop` + the synthetic sensor rig in `config/configuration.yaml`) |
| **Math-affecting** | Anything touching `utils.py` (sigmoid/logit pipeline), `data/prior.py`, `data/decay.py`, correlation/likelihood code in `db/correlation.py`, adjacency boost/decay-modifier math | Law 1 applies: predicted numbers before running, simulator or `scripts/visualize_distributions.py` validation, no reliance on tests alone |
| **Config-schema-affecting** | New `CONF_*`/`DEFAULT_*` in `const.py`, new options-flow field, new per-area setting | Law 3 applies: justify why a smart default can't cover it first; if shipped, `.get()`-based defaulting is mandatory, plus a migration-ladder entry if the key needs a version-gated introduction |
| **DB-schema-affecting** | New table/column in `db/schema.py` | Law 2 applies: additive-only changes use `create_all(checkfirst=True)` and skip the `CONF_VERSION` bump; anything requiring an actual data transformation needs a real migration plan, not just a version bump (a bump alone triggers the destructive full-DB recreate) |

---

## 3. Gates as they actually exist

Five GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | What it checks |
|---|---|---|
| `test.yml` ("CI: Test") | push to `main`, PR to `main` (rc/dev triggers removed 2026-07-06 — see §5) | `uv run pytest --cov=custom_components/area_occupancy --cov-report=xml --cov-report=term-missing`. Coverage gate: `pyproject.toml` `[tool.coverage.report] fail_under = 85` (the inline comment says "Enforce 90% coverage minimum" — that comment is stale/wrong; the enforced number is 85, treated as the floor with 90 as an aspiration) |
| `lint.yml` ("Ruff") | push to `main`, PR to `main` | `uv run ruff check .` (no `--fix`) then `uv run ruff format . --check` (no mutation) — CI never auto-fixes; run `scripts/lint` locally first |
| `validate.yml` | push to `main`, PR to `main`, daily cron, manual dispatch | Hassfest validation (`home-assistant/actions/hassfest@master`) and HACS validation (`hacs/action@main`) — both pinned to floating `@master`/`@main` refs, so occasional failures are upstream flakiness (e.g. observed Cloudflare 525 from `brands.home-assistant.io` crashing HACS's brands validator), not necessarily your PR's fault — check the failure log before assuming your change broke it |
| `release.yml` | on `release: published` | Hard-fails if `manifest.json`'s version doesn't equal the GitHub release's tag name; then zips `custom_components/area_occupancy` and uploads it as the HACS-installable asset via `gh release upload` |
| `docs.yml` | push to `main` only | Builds and deploys `docs/` via `mkdocs gh-deploy --force` to the `gh-pages` branch |

**Branch protection**: *classic* branch protection is absent
(`gh api repos/Hankanman/Area-Occupancy-Detection/branches/main/protection`
returns `404 Branch not protected`), but a **repository ruleset named "Main"
does exist** (`gh api repos/Hankanman/Area-Occupancy-Detection/rulesets`,
id 6210511, enforcement: active): it forbids deletion and non-fast-forward
pushes and **requires changes to `main` to come through a pull request** —
however, repository admins and one GitHub App integration bypass it
("bypass_mode": "always"), so the maintainer's own direct pushes go through
with a "Bypassed rule violations" warning. No status checks are required by
the ruleset: CI is still advisory. Practical rule: work through PRs; treat
the admin bypass as an escape hatch for meta-content (e.g. this skills
library), not for code. Discipline (running `scripts/lint`/`scripts/test`
locally, waiting for green CI, waiting for CodeRabbit review) remains the
real gate — GitHub will not block a merge on red CI.

**CodeRabbit**: reviews every PR automatically (no `.coderabbit.yaml` present
in the repo, so it runs on CodeRabbit's default configuration/GitHub App
install). It posts inline suggestions and can auto-link related/duplicate
issues (e.g. it flagged #465 as a possible duplicate of #466, and linked
#429/#459 as related PRs).

**Merge convention**: the repo has all three merge strategies enabled
(`allow_squash_merge`, `allow_merge_commit`, `allow_rebase_merge` all true
per the GitHub API), but **squash-merge is what's actually used in
practice** — verified directly: PR #489's merge commit (`704c89e`) has a
single parent (`git cat-file -p 704c89e` shows one `parent` line), and its
title is the PR title with `(#489)` appended, GitHub's standard squash-title
format. Recent merged-PR titles consistently follow Conventional Commits
(`feat:`, `fix(health):`, `chore:`, `perf(correlation):`) with a trailing
`(#NNN)`.

---

## 4. Versioning and release

**Three files carry the version string and must be bumped together** (verified
against the actual bump commit `704c89e` "chore: bump version to 2026.5.17
(#475)" — diff touched exactly these three files, 2 lines each):

1. `pyproject.toml` — `version = "2026.5.17"`
2. `custom_components/area_occupancy/manifest.json` — `"version": "2026.5.17"`
3. `custom_components/area_occupancy/const.py` — `DEVICE_SW_VERSION: Final =
   "2026.5.17"` (note: not literally named `VERSION`)

**Do not confuse this with `CONF_VERSION`** (currently 18) and
`CONF_VERSION_MINOR` (currently 0) in the same `const.py` file — those gate
the config-entry migration ladder (`migrations.py`) and are a completely
separate axis from the release version. Bumping one does not bump the other.

**Version scheme is CalVer (`YYYY.M.N`), not SemVer** — AGENTS.md's line "Use
semantic versioning (MAJOR.MINOR.PATCH)" is stale. Actual releases (`gh
release list`) run `2026.5.17`, `2026.5.2`, `2026.5.1`, `2026.4.1`,
`2026.3.4`... — year.month.increment, not major.minor.patch.

**Release mechanics**: create a GitHub Release (tag = the exact version
string in the three files above); `release.yml` hard-fails the build if
`manifest.json`'s version doesn't match the release tag, so bump the version
files *before* cutting the release, not after. Release bodies are
hand-edited on top of GitHub's auto-generated "What's Changed" PR list —
narrative summary + linked originating issues, then the auto-generated
per-PR bullet list.

**CHANGELOG.md is STALE — do not trust or update-in-place expecting
continuity**: the last real entry is `[2026.3.3] - 2026-03-09`, and the file
now carries an explicit deprecation banner at the top saying so and pointing
to GitHub Releases as the changelog of record. Every release since
(`2026.3.4` through `2026.5.17`, i.e. the entire second quarter of this
project's 2026 history) has no CHANGELOG.md entry — release notes live only
in GitHub Releases (`gh release view <tag>`), not in this file. If asked to
"update the changelog," clarify whether that means CHANGELOG.md (dead
convention, now self-documenting its own deprecation) or a new GitHub
Release body (the actual live convention).

**Note on today's merge wave (2026-07-06)**: PRs #489, #488, #486, #491,
#492, #493, #494, #495, #496, and #454 all merged to `main` on 2026-07-06,
but the integration release version is **still `2026.5.17`** — none of
these changes are in a tagged release yet. Don't describe them as "shipped
in release 2026.5.x" until a new version is actually cut; they're merged to
`main` but pre-release.

---

## 5. Branch reality (AGENTS.md is stale here)

**AGENTS.md says**: "Development happens on `dev` branch. PR from `dev` to
`preview` for prereleases. PR from `preview` to `main` for full releases."

**The repo shows this is no longer true.** Verified:

- `git ls-remote --heads origin` — no `dev`, `preview`, or `rc` branches
  exist on the remote today.
- Every one of the last 15 merged PRs (#446 through #489, spanning
  2026-04-27 through 2026-07-06) has `baseRefName` = `main`.
- The last PR merged into a non-`main` base was `#343` into `dev`, dated
  2026-01-30 — the dev→preview/rc→main pipeline was real practice through
  roughly January 2026, then abandoned in favor of direct feature-branch →
  `main` PRs from February 2026 onward.
- **CI trigger cleanup (PR #495, "CI hygiene," merged 2026-07-06,
  SETTLED)**: `.github/workflows/lint.yml`, `test.yml`, and `validate.yml`
  used to still list `pull_request: branches: ["main", "rc", "dev"]` —
  triggering on PRs targeting branches that no longer existed. That dead
  config has been removed; all three workflows now trigger only on `push`/
  `pull_request` to `main` (see §3). This was a stale-config smell, not
  evidence the old flow was still live, and it's now cleaned up.

**Practical instruction**: open PRs directly against `main` from a
feature/fix branch (e.g. `feat/<name>`, `fix/<name>`, `chore/<name>`). Ignore
AGENTS.md's dev/preview/rc branching section — it describes a workflow this
project stopped using about five months before this skill was written.
`CONTRIBUTING.md` is consistent with this: it explicitly says to "Fork the
repo and create your branch from `main`."

---

## 6. When to reply to CodeRabbit

Convention observed directly in this repo's PR history: when CodeRabbit
flags something and you disagree or intentionally don't act on it, **post
the rationale as a reply on the PR thread** rather than silently ignoring
it — both so the reviewer (human or AI) sees you considered it, and so the
project accumulates a record of deliberate design decisions.

Two concrete shipped examples:

- **Accepting with commentary** (PR #491): `@coderabbitai Good catch —
  updated docs/docs/technical/global-prior-flow.md in the follow-up commit:
  the period-end rule, the code sample, and the now-obsolete known-issue
  entry about the 1-hour threshold all reflect the always-`now` behaviour.`
- **Declining with rationale** (PR #438) — the canonical pattern for a
  partial/declined finding, structured as Accepted / Partially accepted /
  declined-with-reasoning:

  > "I did **not** make the shell re-persist a fatal failure. The rationale:
  > By the time `save_area_data` runs, `delete_area_data` has already
  > succeeded — the user-visible purge is complete. [...] Raising here would
  > tell the caller 'the purge failed' when in fact the purge succeeded and
  > only a best-effort re-persist failed, which is misleading. If you'd
  > prefer hard failure, happy to flip it — but the current behavior matches
  > what the user actually experiences."

Follow this shape: **state which findings you accepted (and what changed),
which you partially accepted, and which you declined — with the actual
reasoning, not just "won't fix."** This applies to any CodeRabbit finding on
math (Law 1), config (Law 3), or migration safety (Law 2) — those categories
should almost never be silently dismissed without a written reason on the PR.

---

## 7. The evidence bar, by change class

The evidence bar this project actually enforces before a change merges — what
counts as proof for docs vs. behavior vs. math changes.

| Change class | Minimum evidence before merge | Notes |
|---|---|---|
| Docs only (`docs/`, README, docstrings) | CI green (`lint.yml` + `test.yml` unaffected/pass) | No new tests required. Verify links/build via `aod-docs-and-positioning`. |
| Behavior change (config flow, sensors, coordinator wiring, health monitor, DB CRUD) | A unit test that exercises the **real code path** — the actual entry point a user/HA would hit — not a shortcut that manipulates internal state to force the assertion true | See anti-pattern #1 below. Must pass locally (`uv run pytest <file> -v`) and coverage gate must still pass. |
| Math change (anything touching `utils.py`'s sigmoid/logit pipeline, `data/prior.py`, `data/decay.py`, `data/analysis.py` prior/period logic, `data/adjacency.py`, or `simulator/app.py`'s `calculate_probability_breakdown`/`_base_probability` — since PR #529 (2026.8.1) that function mirrors the live pipeline rather than a dead formula, so it's real math now, not just a UI helper) | (1) Hand-computed expected number(s) written down **before** running the code, (2) cross-check via the simulator or a DB-backed test (real interval data, not synthetic zeros), (3) a regression test that encodes the **correct** behavior with the hand-computed number in the assertion | See anti-pattern #2 below. This is the highest-stakes class in the project — the maintainer's stated hardest problem is prior/likelihood learning accuracy on real homes. |

Never make a silent math change — any modification to a probability/decay/prior formula must be called out explicitly in the PR description with before/after numbers, per the project's unwritten law "no silent math changes" (see §1, Law 1 above).

### Anti-pattern #1: the rigged re-add test (PR #488)

PR #488 (disable diagnostic sensors by default for new areas, merged) shipped a test asserting that re-adding a deleted area produces disabled-by-default diagnostics. The test passed — but only because it popped the entity's record out of `registry.deleted_entities` before re-running setup, which is exactly the internal mechanism (`EntityRegistry.async_remove` preserves `disabled_by` in `deleted_entities`, and `async_get_or_create` restores it on re-registration) that makes the real HA behavior the *opposite* of what the test asserted. A same-PR follow-up review caught this and rewrote the test to go through the real registry restore path; the corrected test asserts a re-added area **keeps its previous enabled/disabled state**, not the new default.

Lesson: if your test setup pokes at a private/internal collection (`_something`, a mock's internal registry, a manually-constructed object graph) specifically to make an assertion pass, ask "does production code ever reach this state through that path?" If not, the test is asserting a fiction. Prefer driving the real entry point (`entity_registry.async_get_or_create`, `hass.config_entries.async_setup`, an actual state write via `hass.states.async_set`) and asserting on its output.

### Anti-pattern #2: the test that encoded the bug (issue #483 / PR #491)

See § Law 1 above for the full #483/#491 incident narrative (root cause,
the truncated observation window, the fix). The specific point this
anti-pattern makes that Law 1 doesn't already spell out: **a green test
suite proves nothing if the assertions themselves are wrong.** The existing
test `test_valid_calculation_sets_correct_prior` passed on every CI run
right up until the fix — nothing failed, because the test's *expectation*
(0.99) was itself the bug, not because the code was untested. For any math
change, compute the expected number by hand (or via the simulator / a
known-good DB query) **before** you run the code, then make the test assert
that number — never "whatever the code currently outputs."

## 8. Test suite anatomy

Location: `tests/` — **on `main` as of 2026-07-06 (HEAD `17b71d2`): 37 files, 1779 tests.** The former branch-vs-main split is gone: PR #454 (adjacent-areas) merged into `main` in the 2026-07-06 wave, so the 4 adjacency-only files (`test_data_adjacency.py`, `test_coordinator_adjacency.py`, `test_data_trajectory.py`, `test_db_transitions.py`) are now part of the single `main` count, not a branch-only add-on. Re-verify with `ls tests/test_*.py | wc -l` and `uv run pytest --collect-only -q`. Run:

```bash
scripts/test                      # full suite + coverage report (xml + term-missing)
uv run pytest tests/test_area_area.py            # one file
uv run pytest tests/test_area_area.py::test_area_initialization -v   # one test
```

### Framework and config (`pyproject.toml [tool.pytest.ini_options]`)

- Uses `pytest-homeassistant-custom-component==0.13.345` (pinned in `pyproject.toml` under `[project.optional-dependencies].test`, bumped alongside `homeassistant==2026.7.1` in the #496 dependency refresh), which supplies the `hass` fixture and HA-flavored test infrastructure.
- `asyncio_mode = "auto"` — async test functions run without needing `@pytest.mark.asyncio`.
- `filterwarnings = ["error::sqlalchemy.exc.SAWarning", ...]` — any SQLAlchemy warning (e.g. an unclosed session, a implicit-cartesian-product query) is promoted to a hard test failure. If you add a DB test and see a cryptic `SAWarning` failure, that is the gate working, not a flake — fix the session handling, don't suppress it.
- Only one custom marker is registered: `expected_lingering_timers` (from `pytest-homeassistant-custom-component`) — mark a test with `@pytest.mark.parametrize("expected_lingering_timers", [True])` (used in `test_config_flow.py`, `test_binary_sensor.py`, `test_coordinator.py`) when it deliberately leaves an HA timer running past teardown (e.g. testing the 10-second decay timer or hourly analysis timer) so the plugin doesn't fail the test for a timer leak.
- `norecursedirs = [".git", "testing_config"]`, `testpaths = ["tests"]`.

### Key fixtures in `tests/conftest.py` (1926 lines)

| Fixture | Autouse? | What it gives you |
|---|---|---|
| `coordinator` | Yes (all tests) | A **real** `AreaOccupancyCoordinator` wired to a real `hass`, real areas loaded from `mock_realistic_config_entry`, and a real in-memory SQLite engine (`db_engine`). Not a mock — this is the primary integration-style fixture; most tests get it for free. |
| `coordinator_with_sensors` | No | Builds on `coordinator`: writes real states via `hass.states.async_set(...)` for a motion pair, a media player, and an appliance, appends them to the area's sensor config, and rebuilds `EntityManager` so entity evidence logic runs for real. Use this whenever a test needs live sensor state, not a `Mock` entity. |
| `mock_config_entry` | No | A bare `MockConfigEntry` — use for config-entry-shape tests that don't need a live coordinator. |
| `db_engine` | No | In-memory SQLite (`sqlite:///:memory:?cache=shared`, `StaticPool`, FK pragma on) with all tables created via `Base.metadata.create_all`. Use for any DB-layer test; never point tests at a real file-backed DB. |
| `db_test_session` / `db_session` / `transactional_db_session` | No | Session variants over `db_engine` — the transactional one rolls back after each test for isolation. |
| `config_flow_flow` / `config_flow_options_flow` | No | Bare `AreaOccupancyConfigFlow` / `AreaOccupancyOptionsFlow` instances with `hass` attached directly — **see the FlowManager caveat below before using these.** |
| `mock_realistic_config_entry`, `mock_area_occupancy_db_data`, `sample_*_data` | No | Fixture data builders for areas/entities/intervals/priors — read `conftest.py` lines ~1013-1657 for exact shapes before hand-rolling your own test data. |
| `auto_cancel_timers`, `mock_track_point_in_time_globally`, `mock_frame_helper`, `mock_data_update_coordinator_debouncer` | Yes | Housekeeping autouse fixtures that prevent HA's internal timer/debounce/frame-reporting machinery from causing unrelated test flakiness. You don't call these directly; know they exist so you don't reinvent them. |

### Config flow tests: prefer FlowManager, know the redundant-revalidation trade-off (PR #486 nit)

This repo's entire `tests/test_config_flow.py` (2031 lines) drives the flow by calling `await flow.async_step_x(user_input)` **directly** on a bare flow instance (via the `config_flow_flow` / `config_flow_options_flow` fixtures), rather than going through Home Assistant's real `FlowManager` entry points (`hass.config_entries.flow.async_init(...)` / `hass.config_entries.flow.async_configure(flow_id, user_input)`, or the options equivalent). No test in the suite currently uses the FlowManager entry point (verified: `grep -rn "config_entries.flow.async_init\|config_entries.options.async_init" tests/` returns nothing).

This is a known, reviewed trade-off, not an oversight: on PR #486 (configurable sensor precision) the reviewer noted *"the schema re-validation in `async_step_global_settings` is redundant with FlowManager's own coercion (harmless, only needed because the test calls the step directly)."* Calling a step function directly skips the schema coercion/validation that `FlowManager` normally performs on `user_input` before handing it to your step — so if your step code relies on that coercion, a direct-call test can pass while the redundant validation code silently masks the gap, or conversely a bug only reachable through real coercion goes untested.

Guidance for new config-flow tests:
- If you're testing wizard navigation, step sequencing, or draft-state bookkeeping, the existing direct-call pattern (matching `test_config_flow.py`) is consistent with the suite and fine to follow.
- If you're testing anything that depends on voluptuous schema coercion/defaults/validation (numeric ranges, selector normalization, `vol.Required`/`vol.Optional` defaults), either add a schema-level test (validate the `vol.Schema` object itself, see `TestBaseOccupancyFlow` in `test_config_flow.py`) or explicitly note in the test docstring that it bypasses FlowManager coercion, so the next reader doesn't assume full end-to-end coverage.

### DB layer tests: in-memory engine, not mocks

DB tests (`test_db_*.py`) should use the `db_engine` fixture (real SQLAlchemy engine, in-memory SQLite, real schema from `db/schema.py`) and exercise the real `db/operations.py` / `db/queries.py` functions — not a mocked `Session`. This is what lets `filterwarnings = ["error::sqlalchemy.exc.SAWarning"]` actually catch real ORM misuse (implicit I/O, uncommitted state, N+1 patterns) instead of nothing.

### Entity platform tests (sensor/binary_sensor/number)

Use `coordinator_with_sensors` (or build your own real states via `hass.states.async_set`) plus the real entity classes from `sensor.py` / `binary_sensor.py` / `number.py`. `tests/test_sensor.py` (82 tests) is the pattern reference — e.g. its diagnostic-sensor-default tests iterate the actual class hierarchy (`entity_category == EntityCategory.DIAGNOSTIC`) rather than a hardcoded tuple of class names, specifically so a newly added diagnostic sensor is automatically covered (a nit raised and fixed within PR #488 itself — a hardcoded `isinstance(entity, (PriorsSensor, EvidenceSensor, ...))` tuple was flagged as going stale the moment a new diagnostic sensor is added).

### Diagnostics tests

`tests/test_diagnostics.py` (`TestDiagnosticsExport`, 317 lines) exercises `diagnostics.py`'s config-entry diagnostics export — assert on the real dict `async_get_config_entry_diagnostics` returns, including redaction of sensitive fields, not on a hand-built stand-in dict.

### Pure math tests

Live in `tests/test_utils.py` (1234 lines) — **not** `test_calculate_prob.py`. AGENTS.md's "Common Workflows → Modifying Bayesian Calculation" section says to update `tests/test_calculate_prob.py or similar`; the repo shows the actual file is `tests/test_utils.py`, class `TestBayesianProbability` (plus `TestCombinePriors`, `TestSigmoidFunctions`, `TestApplyActivityBoost`, `TestCombinedProbability`, `TestSigmoidVsBayesian`, `TestPresenceEnvironmentalSplit`, `TestMapBinaryStateToSemantic`). Treat AGENTS.md's filename here as stale and use this table instead. (There is also a stale `.github/instructions/testing_requirements.instructions.md` that lists files like `test_calculate_prior.py`, `test_calculate_prob.py`, `test_storage.py`, `test_probabilities.py`, `test_types.py`, `test_ml_models.py` — **none of these exist in `tests/` today.** Do not trust that file's file list; it predates the current architecture. Use `ls tests/` as ground truth.)

## The golden test-file map

Each row is the file(s) that guard a specific invariant — if you touch the named production code, run (at minimum) the paired test file, and add to it rather than creating a new file for the same concern. Coverage percentages below were measured on `main` as of 2026-07-06 (HEAD `17b71d2`), post-merge of PR #454 (adjacent-areas) — the former branch-vs-main split no longer applies since the adjacency test files are now part of `main`'s own denominator.

| Production area | Guarding test file(s) | Invariant(s) enforced |
|---|---|---|
| `utils.py` sigmoid/combine pipeline (`sigmoid_probability`, `presence_probability`, `environmental_confidence`, `combined_probability`) | `test_utils.py` | Core probability math: combining priors, sigmoid transforms, activity boost application, presence/environmental split. The now-dead `bayesian_probability()` and its `TestBayesianProbability` test class were removed in PR #529 (2026.8.1) — don't expect to find them. 93% file coverage as of 2026-07-06, pre-removal; re-measure if this matters to you. |
| `data/decay.py` | `test_data_decay.py` (`TestDecay`, `TestDecayHalfLife`, `TestDecayModifierFactor`) | Decay curve correctness, invalid/very-large half-life handling, timezone-naive datetime handling (`test_timezone_naive_datetime_handling`), purpose-half-life compounding with adjacency decay-modifier factor via `_resolve_purpose_half_life()` (PR #493, merged 2026-07-06 — see `aod-debugging-and-history` for the #481 guard story). 100% file coverage. This file is the direct descendant of the costly historical "decay half-life config bug" — see `aod-debugging-and-history`. |
| `data/prior.py` | `test_data_prior.py` | Prior class computation/update semantics. 99% file coverage. |
| `data/analysis.py` (full hourly pipeline: sync → prune → cache → aggregate → prior → transition_learning → correlate, 13 steps on `main`) | `test_data_analysis.py` (`TestPriorAnalyzerCalculateAndUpdatePrior`, `TestTimePriorsDST`, `TestPriorAnalyzerCalculateTimePriors`, `TestOrchestrationFunctions`, `TestRunFullAnalysisCancellation`, `TestIsTimestampOccupied`, interval-merging classes) | The prior/period-window arithmetic (site of the #483 quiet-tail bug, fixed by PR #491, merged 2026-07-06), DST-safe time-of-day bucketing (`TestTimePriorsDST` — this project's costliest historical bug class), motion-interval segmentation/timeout logic, pipeline cancellation safety. 91% file coverage. |
| `data/adjacency.py` + coordinator wiring | `test_data_adjacency.py` (`TestComputeAdjacencyBoost`, `TestApplyLogitBoost`, `TestComputeDecayModifier`), `test_coordinator_adjacency.py` (`TestLaggedProbabilities`, `TestAdjacencyBoostWiring`, `TestDecayModifierWiring`, `TestTrajectoryBookkeeping`) | Adjacent-areas Bayesian boost and decay-modifier math (Phase 4) and its coordinator-level wiring. PR #454 merged 2026-07-06 (#456 closed as merged into it) — on `main` now, but the feature remains unvalidated on real homes (still a candidate feature). 99% file coverage on `data/adjacency.py`. |
| `data/trajectory.py` | `test_data_trajectory.py` | Household trajectory tracker (Phase 4b) used by adjacency boost. |
| `data/health.py` | `test_data_health.py` (21 classes incl. `TestStuckActive`, `TestStuckInactive`, `TestPurposeAwareStuckActive`, `TestSanerDefaults`, `TestStickyIgnore`, `TestNaiveLastUpdatedRegression`, `TestPipelineHealth`) | Sensor health/repair-issue detection, purpose-aware stuck-active thresholds (#474), sticky-ignore preservation across condition flaps (#473), naive-datetime regression guard. |
| `data/purpose.py` | `test_data_purpose.py` | Room-purpose default decay/behavior settings. |
| `data/config.py` | `test_data_config.py` (71 tests) | Config validation/normalization for both integration- and area-level settings. |
| `data/entity.py` | `test_data_entity.py` | Entity evidence detection (`has_new_evidence`), state tracking. 96% file coverage. |
| `data/entity_type.py` | `test_data_entity_type.py` | `InputType` classification behavior. |
| `time_utils.py` | `test_time_utils.py` (`TestTimeUtils`, incl. `test_to_local_uses_default_timezone`) | UTC storage / local-time bucketing conversions — guards the project's historical timezone/DST bug class jointly with `TestTimePriorsDST` above. |
| `coordinator.py` | `test_coordinator.py` | Lifecycle, timers (decay/analysis/save), multi-area orchestration. 86% file coverage. Note: the analysis-timer re-arm in `run_analysis` moved out of the `finally` block (Python 3.14 `SyntaxWarning` fix, part of the #496 toolchain refresh). |
| `area/area.py` | `test_area_area.py`, `test_area_all_areas.py` | Per-area config/entity/prior/calculation encapsulation. |
| `config_flow.py` | `test_config_flow.py` | Wizard steps, options flow, schema construction/validation, sensor-keyword classification (door/window/weather detection). 74% file coverage — one of the lowest-covered files (see `db/relationships.py` below for the single lowest); a good place to add tests if you're looking for high-value coverage work. Note: PR #489 (merged 2026-07-06) removed `show_advanced_options` from this file. |
| `migrations.py` | `test_migrations.py` | Config-entry migration idempotency and data preservation across `CONF_VERSION` bumps. 80% file coverage. |
| `db/schema.py` | `test_db_schema.py` | Table definitions, constraints. 100% file coverage. |
| `db/operations.py` | `test_db_operations.py` | CRUD for entities/intervals. |
| `db/queries.py` | `test_db_queries.py` | Occupied-interval queries, cache validation. |
| `db/aggregation.py` | `test_db_aggregation.py` | Hourly/daily/weekly/monthly rollups. |
| `db/correlation.py` | `test_db_correlation.py` | Sensor-occupancy statistical correlation (minimum-50-sample rule). 89% file coverage. |
| `db/sync.py` | `test_db_sync.py` | Recorder import correctness. 99% file coverage. |
| `db/maintenance.py` | `test_db_maintenance.py` | Health checks, pruning, backups. 83% file coverage — second-lowest; validate carefully before trusting untested branches here. |
| `db/transitions.py` | `test_db_transitions.py` | State-transition interval bookkeeping. |
| `db/relationships.py` | `test_db_relationships.py` | ORM relationship integrity. **66% file coverage — the single lowest-covered production file in the repo as of 2026-07-06.** Treat changes here as needing new tests even for small edits. |
| `db/core.py`, `db/utils.py` | `test_db_core.py`, `test_db_utils.py` | DB init/session management; shared DB helpers. |
| `sensor.py`, `binary_sensor.py`, `number.py` | `test_sensor.py` (82 tests), `test_binary_sensor.py`, `test_number.py` | Entity platform state/attribute correctness, diagnostic-default registration (#488, merged 2026-07-06) and sensor precision (#486, merged 2026-07-06). `binary_sensor.py` at 85% coverage. |
| `diagnostics.py` | `test_diagnostics.py` | Config-entry diagnostics export shape/redaction. 77% coverage. |
| `service.py` | `test_service.py` | Service call handlers. |
| `data/activity.py` | `test_activity.py` (61 tests) | Detected-activity scoring. 94% file coverage. |
| `const.py` | `test_const.py` | Constant-set sanity (small file, 4 tests, 96% coverage). |
| Whole-integration setup | `test_init.py` | `async_setup_entry`/unload smoke test. |

Regenerate this table's coverage numbers with `scripts/test` (or `uv run pytest --cov=custom_components/area_occupancy --cov-report=term-missing`) — the percentages above are a snapshot, not a contract.

## Coverage: enforced vs. aspirational

- **Enforced (CI-blocking):** `pyproject.toml [tool.coverage.report] fail_under = 85` — the whole-project global gate. `scripts/test` and `.github/workflows/test.yml` both run `pytest --cov=custom_components/area_occupancy --cov-report=xml --cov-report=term-missing`; pytest-cov fails the run if total coverage drops below 85%. As of 2026-07-06 (`main` HEAD `17b71d2`), actual total coverage is **88.23% (1779 tests passing)** — comfortable headroom, but a large low-coverage change can still eat through it.
- **The former repo inconsistency is resolved:** the `fail_under = 85` line in `pyproject.toml` now carries the trailing comment `# Enforced global minimum; aim for 90%+ on core calculation modules (AGENTS.md)`, which agrees with the enforced value (85) instead of contradicting it. Historically the comment read `# Enforce 90% coverage minimum` and disagreed with the number — that mismatch has been fixed as part of the 2026-07-06 merge wave; trust the number (85) either way.
- **Aspirational (not separately enforced by any tool):** AGENTS.md states "85%+ coverage requirement (90% for core calculations)." There is no per-module coverage gate, `.coveragerc`, or codecov config anywhere in the repo enforcing a separate 90% threshold on `utils.py`, `data/prior.py`, `data/decay.py`, or `data/analysis.py` — verified by `grep -rn "90" pyproject.toml` (no separate coverage-config hit beyond the comment above) and no `codecov.yml`/`.coveragerc` file present. In practice the core-calculation files already run 90-100% (`data/decay.py` 100%, `data/prior.py` 99%, `utils.py` 93%, `data/analysis.py` 91%) — treat 90%+ on these specific files as a review expectation you self-enforce (check `--cov-report=term-missing` output for the file you touched), not something CI will catch for you if it slips.

## Lint gates

CI job `lint.yml` ("Ruff") runs on every PR to `main` — as of the 2026-07-06 merge wave (#495, CI hygiene) the lint/test/validate PR triggers were narrowed to `main`-only; the old `rc`/`dev` trigger targets are gone:
```bash
uv run ruff check .          # must exit 0 — no lint errors
uv run ruff format . --check # must exit 0 — no formatting diffs
```
Local equivalent (auto-fixes instead of just checking): `scripts/lint`, which runs `uv run ruff format .` then `uv run ruff check . --fix`. If pre-commit's ruff hook fails on commit, review its changes, `git add -u`, and commit again (per AGENTS.md) — do not bypass with `--no-verify`.

`pyproject.toml` now pins `ruff==0.15.2` in dev dependencies, and `.pre-commit-config.yaml` pins the same `rev: v0.15.2` for the `ruff-pre-commit` hook — the triple-skew trap (dev deps vs. pre-commit vs. CI all drifting to different ruff versions) that used to require manual reconciliation is resolved as of the #496 dependency refresh (2026-07-06); all three now agree. `[tool.ruff.lint]` enables a large rule set (`ASYNC`, `B`, `BLE`, `C`, `D`, `PL`, `PT`, `PTH`, `RET`, `SIM`, `TRY`, `UP`, etc. — see `[tool.ruff.lint].select`) with a documented `ignore` list (e.g. `E501` line-too-long, `PLR2004` magic values, `PT011` broad `pytest.raises`). PEP 758's unparenthesized multi-except syntax (`except A, B:` without parens) is now house style and ruff-enforced as part of the same refresh. If ruff flags something you believe is a false positive for this codebase, don't silence it inline without checking whether it's already in the `ignore` list for a reason — read `pyproject.toml` first.

## What a PR must show before merge

Per this project's change-control discipline (full process detail in §§1-6 above), from a QA-evidence standpoint a PR should show, in its description:
1. Which change class it is (docs/behavior/math) and the evidence for that class from the table above.
2. For math changes: the hand-computed expected number(s), stated in the PR body, *before* describing what the code now returns — so a reviewer can check your arithmetic independently of your test.
3. `scripts/test` output (or CI green) demonstrating the full suite still passes and coverage still clears 85%.
4. `scripts/lint` clean (or CI green on `lint.yml`).
5. For config-surface changes: an explicit backward-compatibility statement (existing config entries unaffected) — config surface is sacred per the maintainer's unwritten laws.
6. If a CodeRabbit/reviewer nitpick is deliberately skipped rather than fixed, a reply on the PR thread stating why (project convention — see the `feedback_coderabbit_skip_replies` memory note).

---

## 9. Research methodology: turning a hunch into an accepted change

The discipline that turns a hunch about occupancy-detection accuracy into an
accepted, merged change in this repo: the evidence bar a hypothesis must
clear, how ideas are validated with predicted numbers before code is
written, the idea lifecycle from issue to release notes, and the
anti-patterns that have cost real time.

### The evidence bar for a hypothesis

A hypothesis about *why* the math is wrong (or *why* a fix will work) is not accepted here on
plausibility alone. Two things must both hold:

1. **One mechanism must explain ALL observations, including the negatives.** Not just the
   symptom the reporter complained about — also why it *doesn't* happen in the cases it doesn't.
   Example (issue #483 / PR #491): the claim was "global prior inflates when an area is quiet
   for over an hour." The accepted mechanism — `PriorAnalyzer.calculate_and_update_prior()`
   truncating `actual_period_end` at `last_interval_end` once quiet-time exceeded 3600s — explains
   both why a kitchen mmWave sensor pinned at 0.99 despite true occupancy of 28-35%, *and* why
   the bug is worse overnight/weekends (longer quiet stretches → more truncation each hourly
   recalculation) *and* why areas that are rarely quiet for over an hour don't show it. A theory
   that only explains the headline symptom and is silent on the negatives is not yet a mechanism.
2. **It must survive assigned adversarial refutation** — someone (or some session) deliberately
   trying to prove it's wrong, using the *right* source-of-truth evidence base for the claim.

### The #489 false-blocker lesson: verify against the right version, not the convenient one

PR #489 (fix for issue #487, "deprecation of `show_advanced_options`") is the canonical example
of adversarial refutation going wrong because it checked the wrong ground truth. The reporter's
installation ran HA core-2026.6.1 (see issue #487's System Health block). `FlowHandler.show_advanced_options`
only unconditionally returns `True` — triggering the deprecation warning on every flow run — from
HA **2026.6 onward** (confirmed in the maintainer's PR #489 review). But this repo's *pinned test
dependency* was, at the time, `homeassistant==2026.2.2` — four months older than the
version that exhibits the behavior (the pin has since moved to `2026.7.1` via PR #496;
re-verify with `grep -n '"homeassistant==' pyproject.toml`). A refutation attempt that ran against the pinned test
environment (the "convenient", already-installed source of truth) would not reproduce the
warning at all, and could wrongly conclude the reported deprecation was fabricated or already
fixed. It wasn't: the deprecation was real, confirmed against HA's own 2026-05-26 blog post
(`developers.home-assistant.io`) and the reporter's actual running version.

**Status: SETTLED.** PR #489 merged 2026-07-06, and the follow-up dependency refresh (PR #496,
merged 2026-07-06) moved the pinned test dependency itself to `homeassistant==2026.7.1` — past the
version that exhibits the behavior — so the skew this lesson warns about no longer exists between
CI and the reporter's install. Keep the story as a dated case study in *why* you verify against
the right version, not as a live trap: the general lesson (identify which version is authoritative
for a given claim — usually the reporter's live version or latest stable, not whatever's pinned in
`pyproject.toml`) still applies whenever pins and reality can diverge again.

**Lesson, generalized:** when refuting a claim about behavior that depends on an external
version (HA core, a library, a user's own install), identify which version is authoritative for
*that specific claim* — usually the reporter's live version, or the latest stable release, not
whatever happens to be pinned in `pyproject.toml` for CI stability. The pinned CI version
(`homeassistant==2026.7.1` as of 2026-07-06 — reverify with `grep homeassistant pyproject.toml`)
now tracks current HA core closely post-#496; treat it as ground truth for *this repo's test
suite*, and re-check for drift the next time HA ships a release this repo hasn't picked up yet.

### Hypothesis-predicts-numbers-before-running

The house style — state the number your hypothesis predicts, *then* run the experiment (query,
simulator, or test) that measures it, *then* compare — is the same discipline as §1's Law 1
("no silent math changes"); Law 1 is the canonical statement of it for math-affecting code
changes. The two worked examples below are additional evidence of the discipline in practice,
one of which (PR #486) isn't covered anywhere else in this skill.

**Worked example 1 — PR #486 (sensor-state recorder-write load, addressing issue #467).**
Hypothesis: rounding diagnostic-sensor state to 0 decimals (vs the 2-decimal default) will cut
recorder rows because most 10-second decay-timer ticks don't cross a whole-percent boundary.
Measured on a live 6-area install (v2026.5.17, 57 AOD entities), 3-hour windows:

| Window | Precision | Rows | Δ vs 2-decimal baseline |
|---|---|---|---|
| Afternoon (active) | 2 (baseline) | 15,952 | — |
| Evening 21:30-00:30 | 0 | 7,058 | **-55%** |
| Morning (low activity) | 0 | 3,323 | **-79%** |

The predicted direction (fewer rows, more reduction the quieter the home) matched the measured
result before the PR was written up. (Verified directly: `gh pr view 486`.)

**Worked example 2 — PR #491 (global prior denominator, issue #483).** Same discipline applied to
the incident already narrated in full in §1, Law 1: the hypothesis committed to a predicted number
(~0.25, the kitchen's true occupancy rate) *before* the fix was validated, rather than accepting
whatever the post-fix code happened to output. See § Law 1 above for the incident detail.

**Takeaway for your own hypotheses:** before running a DB query, a `scripts/visualize_distributions.py`
plot, or a simulator session, write down the number(s) you expect and why. If you can't derive a
number from the mechanism, you don't have a mechanism yet — you have a guess.

### The idea lifecycle here

```
issue / observation (usually from a real home)
   → hypothesis with predicted numbers
   → discriminating experiment (DB query against config/.storage/area_occupancy.db,
     scripts/visualize_distributions.py, or a simulator/ run reproducing the reported snapshot)
   → PR with regression tests that encode the CORRECT behavior (not the old behavior)
   → CodeRabbit review + CI (Ruff, CI:Test, Hassfest, HACS validation)
   → maintainer merge
   → real-home soak (the fix rides along in the next release on the reporter's or maintainer's
     live install before being declared "done"; for learning-accuracy work this post-merge soak
     comes ON TOP OF the pre-merge DB-copy check in aod-debugging-and-history Phase 4 —
     two stages, not one)
   → release notes with the measured numbers (not just "fixed a bug")
   → docs update (features/*.md or technical/*.md, mkdocs site)
```

Notes on specific stages:

- **Discriminating experiment.** For prior/likelihood questions this is almost always a SQL
  query against `config/.storage/area_occupancy.db` (see `aod-math-reference` for
  query recipes) or a `scripts/visualize_distributions.py` run to check whether a numeric
  sensor's learned Gaussian actually fits its real occupied/unoccupied distributions. For
  end-to-end Bayesian-math questions, the `simulator/` Flask app (`main.py`, imports the real
  `EntityType`/`Entity` classes rather than reimplementing the math) lets you paste a captured
  `area_occupancy.run_analysis` service response and interactively toggle sensors.
- **Regression tests encode correct behavior, not current behavior.** PR #491 is the template:
  when a test's assertion *is* the bug (asserting 0.99 where reality is 0.25), fix the assertion,
  don't just patch around it. A test suite where every assertion matches "whatever the code
  currently does" cannot catch regressions in the thing you're trying to fix.
- **CodeRabbit + CI gate, but rate limits are real.** CodeRabbit has hit its per-developer review
  rate limit mid-PR at least twice on prior-calculation work (PR #491's review history). Don't
  read "no CodeRabbit review yet" as "no review needed" — it may just be queued.

### Experiment flags: how an experimental capability ships and graduates

The adjacent-areas feature (PR #454, `feat/adjacent-areas` branch — **merged to main 2026-07-06**;
verify with `gh pr view 454 --json state,mergedAt`) is the reference pattern for shipping something
whose *tuning* is genuinely unknown at merge time:

1. **Off by default, not behind a separate feature toggle.** There's no `enable_adjacency: bool`.
   Instead `CONF_ADJACENT_AREAS` defaults to `[]` (`data/config.py`, `raw_adjacent = data.get(CONF_ADJACENT_AREAS, [])`)
   — an area with no configured neighbours never enters the adjacency code path at all. Zero
   configured adjacency == zero behavior change. This is the pattern to copy for any new
   learned-influence feature: make the empty/default case a true no-op, not a flag.
2. **Constants are explicitly marked unvalidated.** `const.py` (around lines 189-221) carries the
   comment *"Adjacent-areas / transition learning tunables (Phase 3 of feat/adjacent-areas).
   First-pass values; tune from real data once Phase 3 is collecting transitions."* followed by
   10 named constants (`ADJACENCY_TRANSITION_WINDOW_S=60`, `ADJACENCY_RECENCY_HALF_LIFE_DAYS=30`,
   `ADJACENCY_BOOST_GAIN=0.5`, `ADJACENCY_DECAY_MODIFIER_GAIN=0.75`,
   `ADJACENCY_DECAY_MODIFIER_MAX=1.75`, four `ADJACENCY_N_*` smoothing thresholds) with no
   empirical backing yet — verified present, no test exercises them against real recorder data.
   **Copy this pattern**: when you ship a first-pass tunable with no real-data validation, say so
   in the constant's own comment, not just in the PR description — PR descriptions get lost,
   `const.py` comments travel with the code.
3. **Graduation path**: a flagged/first-pass capability graduates when real-home data
   (via `aod-debugging-and-history`) validates or retunes its constants and the "tune from
   real data later" comment is replaced with a cited number. Until then it stays labeled
   first-pass/candidate — do not remove the caveat comment just because the feature merged.
4. **Retirement**: if a first-pass feature turns out wrong (not just untuned), it is documented as
   a failure in `aod-debugging-and-history`, not silently deleted — the record of *why* it didn't
   work is itself the deliverable.

### Where good ideas historically come from: the community IS the sensor network

This project has exactly one maintainer and no dedicated QA team. Nearly every real accuracy fix
in its history originated from a user's own report, often with a full root-cause diagnosis
already attached:

- **#467** — a user measured recorder-row growth on their own live 6-area install (15,952 rows /
  3h) and quantified the storage-growth problem before any fix existed; that measurement directly
  shaped PR #486's before/after table.
- **#483** — user `@mscharwere` did the root-cause analysis themselves ("denominator excludes the
  quiet tail"), which PR #491's description explicitly credits ("Credit to @mscharwere for the
  precise root-cause analysis").
- **#464** — user `@laszlojakab` located the exact offending lines
  (`binary_sensor.py` lines 859-865, cited against a specific commit SHA) and proposed the
  tri-state fix that PR #492 implements almost verbatim.
- **Discussion #431** — a user (`jeroen-zzx`) proposed a "next door room" feature with a
  hand-tuned-confidence design; PR #454 answered it with a *learned* (not user-configured)
  version instead — a case where the community supplied the requirement, not the implementation
  design.

**Practical implication:** when triaging a new issue, look for whether the reporter already did
part of the diagnostic work (a line number, a measured number, a specific commit SHA) before
re-deriving it yourself — and credit it in the PR, matching this project's own convention.

### Anti-patterns (each has cost real time here)

- **Fixing symptoms without a mechanism.** The health/repairs subsystem's early history (PR #429
  through PR #474) is the cautionary tale: a naive/aware-datetime crash was patched
  (`dt_util.as_utc`), then a threshold was patched, then a second threshold, then an ignore-flag
  bug, then per-purpose multipliers — six rounds of forward-fixes across issues
  #444/#445/#455/#463/#465/#466/#468 before the system reached its current purpose-aware,
  media-player-exempt, sticky-ignore design. Each round fixed a real bug, but the *pattern* of
  shipping the next patch as soon as the current symptom stopped reproducing — without asking
  "what's the general shape of the thing that keeps recurring" — is why it took six rounds
  instead of one redesign. See `aod-debugging-and-history` for the full chronicle.
- **Tests that encode the bug.** `test_valid_calculation_sets_correct_prior` asserting 0.99 (the
  buggy output) rather than 0.25 (the correct output) is the concrete example — see § Law 1 and
  Anti-pattern #2 above. Before trusting a green test suite as evidence a calculation is right, check whether the
  assertions were derived from a hand-computed expected value or copy-pasted from a prior run's
  output.
- **Trusting one home's data.** A single install's measured numbers (e.g. PR #486's 6-area/57-entity
  sample) are good enough to *justify shipping* a low-risk, reversible change (a display-precision
  setting, default-off), but are not sufficient to retune a `const.py` probability constant that
  affects every area's math — that requires the multi-home validation described in
  `aod-debugging-and-history`. Do not generalize "it worked on my kitchen" into a global
  default change without that step.

---

## Provenance and maintenance

Date-stamped 2026-07-06 (post-merge-wave), integration release version still
2026.5.17 (`pyproject.toml` line 7 / `manifest.json` line 20 / `const.py`
`DEVICE_SW_VERSION` — see §4's note on the merge wave not being released
yet). Repo is checked out on `main`, HEAD `17b71d2` ("feat: adjacent-areas —
learned next-door room influence (#454)") — see §0 for the now-settled
history of that branch trap. PRs #489, #488, #486, #491, #492, #493, #494,
#495, #496, and #454 all **merged to `main` on 2026-07-06** — their content
above is now described as shipped-on-main, not pending.

Re-verification commands, by volatile fact category:

- **Which branch you're actually on**: `git status && git branch --show-current` (expect `main`, HEAD at or descended from `17b71d2`)
- **Branch protection state on main**: `gh api repos/Hankanman/Area-Occupancy-Detection/branches/main/protection`
- **Merge-strategy settings**: `gh api repos/Hankanman/Area-Occupancy-Detection --jq '{squash:.allow_squash_merge, merge_commit:.allow_merge_commit, rebase:.allow_rebase_merge}'`
- **PR #454/#491/#492/#493/#494/#495/#496 merge state**: `gh pr view <number> --json state,mergedAt,statusCheckRollup` (one number per call) — all should show `state: MERGED` as of 2026-07-06
- **Coverage gate**: `grep -n "fail_under" pyproject.toml`; confirm live number via `scripts/test`
- **CI workflow trigger branches (rc/dev cleanup from PR #495)**: `grep -n "branches" .github/workflows/*.yml` — should show `main` only, no `rc`/`dev`
- **Version triple in sync**: `grep -n version pyproject.toml custom_components/area_occupancy/manifest.json; grep -n DEVICE_SW_VERSION custom_components/area_occupancy/const.py`
- **CONF_VERSION / migration ladder**: `grep -n "CONF_VERSION\b" custom_components/area_occupancy/const.py`; read `custom_components/area_occupancy/migrations.py::async_migrate_entry`
- **CHANGELOG.md staleness/deprecation banner**: `head -15 CHANGELOG.md` vs `gh release list --limit 5`
- **Remote branch existence (dev/preview/rc)**: `git ls-remote --heads origin`
- **Issue #159 / #466 maintainer responses**: `gh issue view 159 --json comments`; `gh issue view 466 --json comments`
- **PR #438 CodeRabbit-reply example**: `gh api repos/Hankanman/Area-Occupancy-Detection/issues/438/comments --jq '.[] | select(.user.login=="Hankanman")'`
- **PR #488 entity-registry restore test**: `grep -n "test_re_add_area_restores_previous_enabled_state" tests/test_sensor.py`
- **AreaTransitions create_all precedent (now on `main` post-#454)**: `git show main:custom_components/area_occupancy/migrations.py | grep -n "create_all\|CONF_VERSION"`
- Test count / pass/fail / coverage %: `scripts/test` (or `uv run pytest --cov=custom_components/area_occupancy --cov-report=term-missing`)
- Test file inventory: `ls tests/*.py`
- Pytest config (markers, filterwarnings, asyncio mode): `grep -n -A3 "\[tool.pytest.ini_options\]" pyproject.toml`
- Fixture list/behavior: `grep -n "@pytest.fixture" tests/conftest.py` then read the specific fixture
- Lint rule set / ignores: `grep -n -A5 "\[tool.ruff.lint\]" pyproject.toml`
- Ruff version pinning across dev deps / pre-commit: `grep -n "ruff==" pyproject.toml` and `grep -n -A2 "ruff-pre-commit" .pre-commit-config.yaml`
- CI job definitions and PR-trigger branches: `cat .github/workflows/test.yml .github/workflows/lint.yml`
- PR #488 rigged-test anti-pattern detail: `gh api repos/Hankanman/Area-Occupancy-Detection/pulls/488/reviews -q '.[].body'`
- PR #491 / issue #483 encoded-bug detail and merge state: `gh issue view 483` and `gh pr view 491 --json state,mergeCommit`
- PR #486 redundant-revalidation nit and merge state: `gh api repos/Hankanman/Area-Occupancy-Detection/pulls/486/reviews -q '.[].body'`
- PR #454 (adjacency) merge state: `gh pr view 454 --json state,mergeCommit`
- Whether any config-flow test uses FlowManager entry points: `grep -rn "config_entries.flow.async_init\|config_entries.options.async_init" tests/`
- `pyproject.toml:25` — `homeassistant==2026.7.1` pinned test dependency (bumped from 2026.2.2 by
  PR #496, merged 2026-07-06). Reverify: `grep -n '"homeassistant==' pyproject.toml`
- `custom_components/area_occupancy/const.py` lines ~189-221 — adjacency tunables and their
  "first-pass, tune from real data later" comment, exact constant names and values. Reverify:
  `sed -n '185,225p' custom_components/area_occupancy/const.py`
- `custom_components/area_occupancy/const.py:323` — `MIN_CORRELATION_SAMPLES = 50`. Reverify:
  `grep -n MIN_CORRELATION_SAMPLES custom_components/area_occupancy/const.py`
- `data/config.py` — `CONF_ADJACENT_AREAS` defaults to `[]`. Reverify:
  `grep -n CONF_ADJACENT_AREAS custom_components/area_occupancy/data/config.py`
- PR #489 body + maintainer review text (deprecation confirmed real, HA 2026.6+ behavior) and
  issue #487's System Health block (reporter on core-2026.6.1). Reverify:
  `gh pr view 489 --json body,reviews` and `gh issue view 487 --json body`
- PR #486 body (measured recorder-row table: 15,952 / 7,058 / 3,323). Reverify: `gh pr view 486 --json body`
- PR #491 body (0.99→0.25 correction, credit to @mscharwere). Reverify: `gh pr view 491 --json body`
- Issue #464 comment by `@laszlojakab` (line-level root cause). Reverify: `gh issue view 464 --json comments`
- Discussion #431 body (user's "next door room" request). Reverify:
  `gh api repos/Hankanman/Area-Occupancy-Detection/discussions/431`
- PR #454 state (adjacent-areas feature) — **MERGED 2026-07-06**. Reverify:
  `gh pr view 454 --json state,mergedAt`
- PRs #491/#492/#493/#494 — all **MERGED 2026-07-06**. Reverify:
  `gh pr view <n> --json state,mergedAt` for each.
- Health-saga PR/issue chain (#429, #444, #445, #446, #455, #459, #463, #465, #466, #472, #473,
  #474) — taken from the discovery dossier's git/PR archaeology lens; spot-verify any single
  claim with `gh pr view <n> --json body,mergedAt` before citing a specific number from it in new
  work.
</content>
