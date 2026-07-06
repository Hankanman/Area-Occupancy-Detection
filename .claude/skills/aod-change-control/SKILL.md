---
name: aod-change-control
description: Use before opening, reviewing, or merging any PR against Area Occupancy Detection (AOD) — especially anything touching Bayesian math (utils.py sigmoid pipeline), priors, decay/half-life, likelihoods, config schema (const.py CONF_*/DEFAULT_*), the config-entry migration ladder (migrations.py), or the DB schema (db/schema.py). Also load when deciding how to reply to CodeRabbit, when bumping the release version, when someone proposes a new per-sensor config knob, or when you're unsure whether a change needs a version bump, a migration, or extra validation before merge.
---

# AOD Change Control

## What this covers

How changes get classified, validated, and merged in this project: the three
non-negotiable rules the retiring maintainer stated explicitly (with the
incident behind each), how CI gates actually work today (not what CLAUDE.md
says), the version-bump and release mechanics, and the CodeRabbit-reply
convention. This is the skill for "am I allowed to ship this, and how" —
process, not implementation detail.

## When NOT to use this

For *how* the Bayesian pipeline actually computes (sigmoid/logit math, what
`bayesian_probability()` in utils.py is and why it's dead code) use
`bayesian-occupancy-reference`. For the historical blow-by-blow of specific
incidents (full saga narratives, root causes, stale branches) use
`aod-failure-archaeology`. For how to run tests/lint/CI locally use
`aod-build-and-env`. For what "beyond SOTA" claims are and aren't earned yet,
use `aod-external-positioning`. This skill only owns: classification rules,
gates, the three unwritten laws, versioning/release mechanics, and the
CodeRabbit convention.

---

## 0. Orientation trap — read this first

The environment banner and CLAUDE.md may both claim you're on `main` with a
clean tree. **Verify, don't trust**: `git status` and `git branch
--show-current`. At time of writing this repo was checked out on
`feat/adjacent-areas` (PR #454), not `main` — an entire feature's worth of
code (adjacency boost/decay-modifier, `db/transitions.py`,
`data/adjacency.py`, ~10 `ADJACENCY_*` constants in `const.py`) exists only on
that branch and is invisible if you assume the banner is correct. Before
citing any file content as "what's in main," run `git show
main:<path>` or `git diff main...HEAD -- <path>` to check.

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
do, then run it, then compare.

**Incidents**:

- **Prior inflation (issue #483 → PR #491, open as of 2026-07-06)**:
  `PriorAnalyzer.calculate_and_update_prior()` (`data/analysis.py`) truncated
  the observation window at `last_interval_end` whenever an area had been
  quiet more than an hour — dropping the known-unoccupied tail from the
  denominator while the numerator kept all occupied time. Every hourly
  recalculation during a quiet stretch (overnight, weekends) pushed
  `global_prior` further toward the 0.99 cap. A real kitchen with true
  occupancy ~28–35% pinned at 0.99. The existing test
  (`test_valid_calculation_sets_correct_prior`) had **encoded the bug** —
  asserting 0.99 was correct — and had to be corrected to assert the true
  0.25 as part of the fix. Verify current status: `gh pr view 491 --json
  state,body`.
- **Half-life override silently discarded (issue #481 → PR #493, open)**:
  `Decay._resolve_purpose_half_life()` (`data/decay.py`) has a sleep/awake
  split for the Bedroom/SLEEPING purpose (uses the area's half-life during
  the sleep window, switches to the purpose's fixed `awake_half_life`=620s
  outside it). The switch applied **unconditionally**, so a user's custom
  10s half-life silently became a 620s (~10 minute) clear-out during waking
  hours. This is the *same bug class* recurring in a different code path:
  issue #439/PR #440 already fixed an identical "custom value gets silently
  normalized to a purpose default" bug for the general half-life field
  months earlier; #481 was the sleep/awake switch not respecting that
  established custom-vs-default semantics. Verify: `gh pr view 493 --json
  state,body`.

**Takeaway**: a test that encodes buggy output (asserts the bug's number) is
not evidence a math change is safe — it's evidence nobody predicted the
correct number independently before the test was written. When you touch
`utils.py` (sigmoid/logit pipeline), `data/prior.py`, `data/decay.py`, or
correlation likelihood code in `db/correlation.py`, compute the expected
output by hand or via the simulator *before* trusting the test suite's green
checkmark.

### Law 2 — NEVER BREAK USER CONFIGS

**Rule**: migrations must be idempotent (safe to run twice), missing keys
must be defaulted via `.get(KEY, DEFAULT)` rather than assumed present, and DB
schema changes must not be destructive unless a `CONF_VERSION` bump is
deliberately accepted (which wipes the whole DB — see below).

**Why a version bump is dangerous**: `db/maintenance.py`'s
`_ensure_schema_up_to_date` treats any mismatch between the stored DB version
and `CONF_VERSION` as fatal — it **deletes and recreates the entire
database** (all learned priors, correlations, intervals, aggregates, for
every area) rather than migrating it. There is no DB-level migration script,
unlike the config-entry migrations in `migrations.py`. Bumping `CONF_VERSION`
for a schema change you don't actually need to gate is a real way to wipe
every user's learned history on upgrade.

**Precedent for additive-only schema changes**: the `AreaTransitions` table
and `Areas.adjacent_areas` column (feat/adjacent-areas, PR #454 — **still
open/unmerged as of 2026-07-06**, verify with `gh pr view 454`) deliberately
does **not** bump `CONF_VERSION`, because the change is purely additive: a
new JSON column with a default-on-missing value in the loader, plus a new
table created via `Base.metadata.create_all(checkfirst=True)`. The
`migrations.py` comment on this branch states the rationale explicitly: bump
the version and every user's DB gets wiped for no reason, since
`checkfirst=True` on `create_all` only adds what's missing without touching
existing tables. **Use `create_all(checkfirst=True)` for additive schema
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
| `test.yml` ("CI: Test") | push to `main`, PR to `main`/`rc`/`dev` | `uv run pytest --cov=custom_components/area_occupancy --cov-report=xml --cov-report=term-missing`. Coverage gate: `pyproject.toml` `[tool.coverage.report] fail_under = 85` (the inline comment says "Enforce 90% coverage minimum" — that comment is stale/wrong; the enforced number is 85) |
| `lint.yml` ("Ruff") | same triggers | `uv run ruff check .` (no `--fix`) then `uv run ruff format . --check` (no mutation) — CI never auto-fixes; run `scripts/lint` locally first |
| `validate.yml` | push to `main`, PR to `main`/`rc`/`dev`, daily cron, manual dispatch | Hassfest validation (`home-assistant/actions/hassfest@master`) and HACS validation (`hacs/action@main`) — both pinned to floating `@master`/`@main` refs, so occasional failures are upstream flakiness (e.g. observed Cloudflare 525 from `brands.home-assistant.io` crashing HACS's brands validator), not necessarily your PR's fault — check the failure log before assuming your change broke it |
| `release.yml` | on `release: published` | Hard-fails if `manifest.json`'s version doesn't equal the GitHub release's tag name; then zips `custom_components/area_occupancy` and uploads it as the HACS-installable asset via `gh release upload` |
| `docs.yml` | push to `main` only | Builds and deploys `docs/` via `mkdocs gh-deploy --force` to the `gh-pages` branch |

**Branch protection**: `gh api repos/Hankanman/Area-Occupancy-Detection/branches/main/protection`
returns `404 Branch not protected` — **there is no GitHub branch protection
on `main`**. CI is advisory only; nothing stops a direct push or a merge with
failing/pending checks. Discipline (running `scripts/lint`/`scripts/test`
locally, waiting for green CI, waiting for CodeRabbit review) is the only
gate. Do not treat a red CI check as something GitHub will block for you —
it won't.

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

**Version scheme is CalVer (`YYYY.M.N`), not SemVer** — CLAUDE.md's line "Use
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
continuity**: the last real entry is `[2026.3.3] - 2026-03-09`. Every release
since (`2026.3.4` through `2026.5.17`, i.e. the entire second quarter of
this project's 2026 history) has no CHANGELOG.md entry — release notes live
only in GitHub Releases (`gh release view <tag>`), not in this file. If
asked to "update the changelog," clarify whether that means CHANGELOG.md
(dead convention) or a new GitHub Release body (the actual live convention).

---

## 5. Branch reality (CLAUDE.md is stale here)

**CLAUDE.md says**: "Development happens on `dev` branch. PR from `dev` to
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
- **Stale CI artifact corroborating this**: `.github/workflows/lint.yml`
  and `test.yml` and `validate.yml` still list `pull_request: branches:
  ["main", "rc", "dev"]` — triggering on PRs targeting branches that no
  longer exist. This is dead config, not evidence the old flow is still
  live.

**Practical instruction**: open PRs directly against `main` from a
feature/fix branch (e.g. `feat/<name>`, `fix/<name>`, `chore/<name>`). Ignore
CLAUDE.md's dev/preview/rc branching section — it describes a workflow this
project stopped using about five months before this skill was written.
CONTRIBUTING.md, if present, should be treated with the same suspicion until
separately verified.

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

## Provenance and maintenance

Date-stamped 2026-07-06, integration version 2026.5.17 (`pyproject.toml`
line 7 / `manifest.json` line 20 / `const.py` `DEVICE_SW_VERSION`). Repo was
checked out on branch `feat/adjacent-areas` (PR #454) at time of writing, not
`main` — see §0. PRs #491, #492, #493, #494, #454 were open/CI-green/awaiting
merge as of this date — do not describe their content as shipped on `main`
without re-checking.

Re-verification commands, by volatile fact category:

- **Which branch you're actually on**: `git status && git branch --show-current`
- **Branch protection state on main**: `gh api repos/Hankanman/Area-Occupancy-Detection/branches/main/protection`
- **Merge-strategy settings**: `gh api repos/Hankanman/Area-Occupancy-Detection --jq '{squash:.allow_squash_merge, merge_commit:.allow_merge_commit, rebase:.allow_rebase_merge}'`
- **PR #454/#491/#492/#493/#494 merge state**: `gh pr view <number> --json state,mergeable,statusCheckRollup` (one number per call)
- **Coverage gate**: `grep -n "fail_under" pyproject.toml`; confirm live number via `scripts/test`
- **CI workflow trigger branches (stale dev/rc reference)**: `grep -n "branches" .github/workflows/*.yml`
- **Version triple in sync**: `grep -n version pyproject.toml custom_components/area_occupancy/manifest.json; grep -n DEVICE_SW_VERSION custom_components/area_occupancy/const.py`
- **CONF_VERSION / migration ladder**: `grep -n "CONF_VERSION\b" custom_components/area_occupancy/const.py`; read `custom_components/area_occupancy/migrations.py::async_migrate_entry`
- **CHANGELOG.md staleness**: `head -15 CHANGELOG.md` vs `gh release list --limit 5`
- **Remote branch existence (dev/preview/rc)**: `git ls-remote --heads origin`
- **Issue #159 / #466 maintainer responses**: `gh issue view 159 --json comments`; `gh issue view 466 --json comments`
- **PR #438 CodeRabbit-reply example**: `gh api repos/Hankanman/Area-Occupancy-Detection/issues/438/comments --jq '.[] | select(.user.login=="Hankanman")'`
- **PR #488 entity-registry restore test**: `grep -n "test_re_add_area_restores_previous_enabled_state" tests/test_sensor.py`
- **AreaTransitions create_all precedent (only on feat/adjacent-areas branch)**: `git show feat/adjacent-areas:custom_components/area_occupancy/migrations.py | grep -n "create_all\|CONF_VERSION"` (or `main` once #454 merges)
