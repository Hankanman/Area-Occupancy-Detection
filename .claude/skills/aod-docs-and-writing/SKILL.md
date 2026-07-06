---
name: aod-docs-and-writing
description: Use when writing or editing anything under docs/docs/ (mkdocs site), README.md, CONTRIBUTING.md, CHANGELOG.md, docstrings, commit messages, PR titles, or GitHub Release notes for Area Occupancy Detection — or when a code change needs an accompanying doc update and you're deciding what/where to write. Trigger words: "update the docs", "add a features page", "write a docstring", "release notes", "mkdocs nav", "stale doc", "CHANGELOG".
---

# AOD docs and writing

## What this covers

The docs-of-record for Area Occupancy Detection: the mkdocs site under `docs/docs/`, `README.md`, `CONTRIBUTING.md`, docstring style, commit/PR title conventions, and how release notes actually get published. It tells you where a given fact belongs, what house style looks like (with quoted exemplars), which existing docs are known-stale landmines, and how to avoid re-creating the #491 stale-doc problem (code changed, doc didn't).

## When NOT to use this

- Deciding *what* the Bayesian math should say (formulas, constants, thresholds) — that's `bayesian-occupancy-reference`.
- Deciding whether a code change is safe / needs a migration / breaks configs — that's `aod-change-control`.
- Writing content for `aod-external-positioning` (README competitive framing, HACS listing copy) beyond the structural conventions below — check that skill for messaging strategy.
- Investigating *why* a historical bug happened — that's `aod-failure-archaeology`.

## Docs tree map

```
docs/
├── mkdocs.yml                      # nav, theme, plugins — site_url is
│                                    # https://hankanman.github.io/Area-Occupancy-Detection/
└── docs/
    ├── index.md                    # site home
    ├── getting-started/
    │   ├── installation.md
    │   ├── configuration.md
    │   ├── why.md                  # sales pitch / motivation, mirrors README tone
    │   └── basic-usage.md
    ├── features/                   # USER-FACING: what it does, why you'd want it
    │   ├── purpose.md, sensors.md, calculation.md, prior-learning.md,
    │   │   likelihood.md, sensor-correlation.md, adjacent-areas.md,
    │   │   wasp-in-box.md, sleep-presence.md, activity-detection.md,
    │   │   decay.md, entities.md, sensor-health.md, services.md
    ├── technical/                  # MECHANISM: how it's implemented, for debugging/extending
    │   ├── deep-dive.md, analysis-chain.md, bayesian-calculation.md,
    │   │   calculation-flow.md, data-flow.md, global-prior-flow.md,
    │   │   time-prior-flow.md, transition-learning.md, entity-evidence.md,
    │   │   likelihood-calculation.md, database-schema.md, prerelease.md,
    │   │   diagnostics.md, debug.md
    ├── simulator/index.md          # interactive probability simulator page
    └── images/, assets/, javascript/
```

Verify the live nav any time you add/remove/rename a page — `mkdocs.yml`'s `nav:` block is a hand-maintained list, not auto-generated. A page that exists on disk but is missing from `nav:` will not appear in the built site (confirmed by reading `docs/mkdocs.yml` 2026-07-06: nav lists 14 features pages and 14 technical pages, one entry per file, no glob).

Doc-site build/deploy is `docs.yml` (GitHub Actions workflow, referenced by the docs badge in README) — it's the CI job that runs `mkdocs build`/`gh-pages` deploy; a stray `gh-pages` branch in the repo is its output, not something to hand-edit.

## The features/ vs technical/ split

**Rule of thumb: features/ answers "what do I get and how do I configure it"; technical/ answers "how does this actually work under the hood."** Every features page should be readable by a user who has never opened the Python source. Every technical page assumes the reader is about to read or modify code.

The newest and best exemplar pair for this split is **`features/adjacent-areas.md`** (user outcome) paired with **`technical/transition-learning.md`** (mechanism). Use this pair as the template when writing a new feature's docs:

- `features/adjacent-areas.md` opens with the user-visible effect ("Adjacent Areas lets rooms that are physically connected... influence each other's occupancy calculation"), covers configuration steps, an "Example" walkthrough in plain language, an "Observing it" section pointing at diagnostics, and an FAQ. It explicitly punts implementation detail: *"Both effects come from **learned transition history**, not from a fixed 'influence' setting you configure. See [Transition Learning](../technical/transition-learning.md) for how that history is built and the underlying maths."*
- `technical/transition-learning.md` opens with the storage schema (`AreaTransitions` table columns), the detection algorithm (pipeline step name, deque-based windowing, exact constants like `ADJACENCY_TRANSITION_WINDOW_S (60s)`), and a 6-level smoothing-fallback table with threshold constants — content a features page would never include.

When you write a new feature, produce both files as a pair, cross-linked in both directions, rather than one page that tries to do both jobs.

## House style

Plain prose over bureaucratic tone; short paragraphs; admonitions for callouts; tables for anything enumerable (fields, defaults, thresholds); mermaid for flow/sequence, not for anything a table would do better.

**Admonition syntax is mkdocs Material's `!!!` block form, not GitHub's `> [!NOTE]`** — the two are not interchangeable and only one renders correctly depending on which renderer processes the file (GitHub's own Markdown preview understands `> [!NOTE]`/`> [!WARNING]`/`> [!CAUTION]`; mkdocs-material understands `!!! note`). `docs/docs/**` files use the mkdocs form exclusively (verified: `grep -rn '^!!! ' docs/docs/features/*.md` matches 12 admonitions across 5 files, zero `> [!` GitHub-style blocks in `docs/docs/`). `CHANGELOG.md` and GitHub Release bodies use the GitHub form instead (verified: `CHANGELOG.md:163` uses `> [!WARNING]`, shifted from `:159` after PR #495 added the deprecation banner), because those render on github.com, not through mkdocs.

Exemplars, quoted directly:

1. **Admonition with a title** (`docs/docs/features/services.md:110-111`):
   ```
   !!! warning "This is destructive"
       All learned priors, correlations, intervals, and aggregates for the selected area are permanently deleted. The integration will start re-learning from scratch on the next analysis cycle (hourly by default). Other areas are unaffected.
   ```
2. **Table for a service's return payload** (`docs/docs/features/services.md`, `purge_area_history` return table):
   ```
   | Key | Description |
   |-----|-------------|
   | `area_id` | The area_id that was purged |
   | `entities_deleted` | Number of entity rows removed from the database for this area |
   ```
3. **Mermaid sequence diagram** (`docs/docs/technical/data-flow.md:9-15`):
   ```
   ```mermaid
   sequenceDiagram
       participant HA as Home Assistant
       participant Coord as Coordinator
       ...
       HA->>Coord: async_config_entry_first_refresh()
   ```
   ```
   Six files under `docs/docs/` use mermaid as of 2026-07-06 (`grep -rl '```mermaid' docs/docs/`): `time-prior-flow.md`, `wasp-in-box.md`, `global-prior-flow.md`, `data-flow.md`, `analysis-chain.md`, `transition-learning.md`. Use `sequenceDiagram` for request/response or component interaction; `flowchart TD` for pipeline/decision steps (see `docs/docs/technical/data-flow.md:40` for a flowchart example).

Avoid hedge-words like "likely" or "may vary — check your instance" in feature docs (an older page, `docs/docs/features/decay.md`, has this smell in its Output section: *"likely as a percentage... Note: The exact representation might vary; check the sensor's state in your HA instance."*). If you don't know what a sensor outputs, read the code or run the integration — don't ship a hedge. Newer pages (`adjacent-areas.md`, `sleep-presence.md`, `services.md`) state behavior definitively; hold new/edited pages to that bar and consider it a low-priority cleanup opportunity (not urgent) if you touch `decay.md` for another reason.

## README structure and feature-list conventions

`README.md` (175 lines, verified 2026-07-06) is structured: badges → hook paragraph → `## The Quick Answer` (HA-vs-AOD comparison table) → `## Creating Automations with AOD` (workflow + what-it-provides) → `## Documentation` (link out) → `## Features` → `## Planned Features` → `## Installation` (HACS + manual) → `## Entities Created` → `## Debugging` → `## Support & Feedback`.

Feature-list convention (`## Features`, line 75 on): each bullet is `**Bold Name**: one-sentence description of user-visible behavior`, no implementation detail, no config key names. Sub-bullets are used only for one entry (`**Multiple Sensor Support**`) that itself enumerates sensor categories. `## Planned Features` uses the identical bullet format but for **unshipped** ideas — before adding something there, check it isn't already shipped (cross-check against `## Features` and the actual `InputType` enum), and before removing something, check it hasn't quietly shipped without the README being updated (the reverse of the #491 problem).

When a feature ships: move its bullet from `## Planned Features` to `## Features` in the same PR that ships the code, or as an immediate same-day follow-up — don't leave a shipped feature listed as "planned" (this exact gap is why PR #494, "fix broken purpose link in README", exists as a class of fix — the README is a doc-of-record that drifts from code just like everything else).

## Docstring convention

Google-style docstrings, full type annotations (Python 3.13+), as directed by `CLAUDE.md`. Verified against the actual codebase (`custom_components/area_occupancy/data/prior.py`, `utils.py`):

```python
def clamp_probability(
    value: float, min_val: float | None = None, max_val: float | None = None
) -> float:
    """Clamp probability value to valid range.

    Args:
        value: Probability value to clamp
        min_val: Minimum value (default: MIN_PROBABILITY from const)
        max_val: Maximum value (default: MAX_PROBABILITY from const)
    """
```

Module-level docstrings are one-liners or a short paragraph (`prior.py`'s module docstring: `"""Area baseline prior (P(room occupied) *before* current evidence).` followed by a two-sentence elaboration). Not every function needs an `Args`/`Returns` block — trivial one-liners (`format_percentage`, `format_value`) get a single-line docstring only. Add the full `Args:`/`Returns:` block once a function has more than one parameter or a non-obvious return, and always for anything touching the Bayesian calculation (per `CLAUDE.md`'s "100% coverage for calculation changes" instruction — docs and tests should match that rigor).

## Commit / PR conventions

Conventional-commit prefixes are used and enforced by convention (not a checked-in linter rule as of 2026-07-06 — verified no commitlint config in repo root): `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`, optionally scoped, e.g. `feat(health):`, `fix(prior):`, `docs(adjacent-areas):`. Verified from `git log --oneline -30`:

```
3471e7a feat(health): purpose-aware stuck-active thresholds and saner defaults (#474)
b9df513 fix(health): preserve user-ignored repairs across condition flaps (#473)
fd61713 docs(adjacent-areas): Phase 5 documentation
8840a50 refactor(analysis): hoist step helpers to module level for C901
```

Squash-merge titles append the PR number in parentheses, e.g. `chore: bump version to 2026.5.17 (#475)`. Same convention held through the 2026-07-06 merge wave, e.g. `feat: adjacent-areas — learned next-door room influence (#454)` (verified: current `main` HEAD as of this sweep, `17b71d2`). When writing a PR title, don't add the `(#N)` yourself — GitHub's squash-merge UI appends it automatically from the PR number.

## Release notes and CHANGELOG.md — resolved 2026-07-06, pointer banner in place

**Release notes are written by hand in GitHub Releases**, not generated from `CHANGELOG.md`. Verified via `gh release view 2026.5.17`: a hand-written release body with a narrative intro ("Saner repair defaults + the off-switch you asked for"), a numbered "What's fixed" section explaining each change's motivation, and a "What's Changed" list of `feat(scope): ... by @Hankanman in <PR URL>` lines matching the actual merged PRs (#472, #473, #474, #475). This is materially richer than a changelog entry — it explains *why*, tells users what to check (e.g. "Make sure your bedroom area's purpose is set to `Sleeping`"), and is the artifact users actually read on the releases page and via HACS.

**`CHANGELOG.md` was abandoned (last dated entry `## [2026.3.3] - 2026-03-09`, `CHANGELOG.md:11`, while the actual latest release was `2026.5.17` with several releases in between not represented) — this is now SETTLED.** PR #495 (merged 2026-07-06) added a banner immediately under the `# Changelog` heading: *"this file is no longer maintained (last entry 2026.3.3). The changelog of record is [GitHub Releases](https://github.com/Hankanman/Area-Occupancy-Detection/releases)..."* This is the lighter-weight version of the two options this doc previously proposed (pointer banner, not a full delete) — don't re-propose deletion or backfilling as if the ambiguity were still open. Do not add new dated entries to the file's body; if a task asks you to log a release there, point to the banner and use GitHub Releases instead.

## STALE-DOC landmines — do not trust without re-verifying

These are documents that describe (or described) a process the repo no longer follows. Two of the three rows below were fixed in the 2026-07-06 merge wave (PR #495) — kept here as settled history rather than deleted, since the failure mode (a doc silently diverging from repo reality) recurs and the fix references are useful precedent. Re-check before relying on any row, since things may have drifted again since this sweep.

| Doc | What it said | Status as of 2026-07-06 | Re-verify with |
|---|---|---|---|
| `CONTRIBUTING.md:16` | "Fork the repo and create your branch from `dev`." | **SETTLED** (PR #495, merged 2026-07-06): now reads "Fork the repo and create your branch from `main`." No `dev` branch exists on the remote; all recent feature/fix branches (`fix/global-prior-quiet-tail`, `feat/adjacent-areas`, `fix/bedroom-half-life-override`, etc.) targeted `main` directly, and the doc now matches. | `git branch -r` (look for absence of `dev`/`preview`); `sed -n '16p' CONTRIBUTING.md` |
| `CLAUDE.md` "Branch and Release Strategy" section | "Development happens on `dev` branch. PR from `dev` to `preview` for prereleases. PR from `preview` to `main` for full releases." | **STILL A LIVE LANDMINE** — CLAUDE.md is unchanged and still says dev→preview→main. The repo has no open PRs as of this sweep (the 2026-07-06 wave, #454/#491/#492/#493/#494/#495/#496, all merged direct to `main`), and `.github/workflows/release.yml` triggers on `release: published` (a manually-created GitHub Release), not a branch-merge event. There is no `preview` branch either. | `git branch -r`; `cat .github/workflows/release.yml`; `gh pr list --json baseRefName` |
| `pyproject.toml` (line 114, now line 113) | Inline comment `fail_under = 85 # Enforce 90% coverage minimum` — the enforced number and the comment's stated number disagreed with each other in the same line. | **SETTLED** (PR #495, merged 2026-07-06): now reads `fail_under = 85 # Enforced global minimum; aim for 90%+ on core calculation modules (CLAUDE.md)` — internally consistent and cross-references `CLAUDE.md`'s "85%+ coverage requirement (90% for core calculations)" line instead of contradicting it. | `grep -n fail_under pyproject.toml` |

Do not silently "fix" `CLAUDE.md`'s branch-strategy section as a drive-by edit — it's explicit project instruction content; flag it to the maintainer or fold the correction into a PR whose primary purpose is a docs/process cleanup, per `aod-change-control`'s rules on touching CLAUDE.md. (The other two rows show this is a real, fixable pattern once someone owns it — CLAUDE.md's branch-strategy section is the one that hasn't been picked up yet.)

## Keeping docs in sync when code changes — the #491 cautionary example

**Cautionary example (PR #491, `fix(prior): keep quiet tail in global prior denominator`, merged 2026-07-06 — verify with `gh pr view 491`):** the code change altered how `PriorAnalyzer.calculate_and_update_prior()` picks the end of its observation period (previously truncated to `last_interval_end` when the area had been quiet >1h; now always `now` — this is the current, permanent behavior on `main`, i.e. `actual_period_end` is always `now`, not conditionally truncated). `docs/docs/technical/global-prior-flow.md` documented the *old* (buggy) behavior in prose — "If last interval is more than 1 hour old: Use `last_interval_end` / Otherwise: Use current time" — and had to be edited in the same PR (7 additions, 16 deletions) to stop describing the bug as intended behavior. This was caught as a review nitpick, not by the original author remembering to update the doc.

**Rule extracted:** when a change touches `data/analysis.py`, `data/prior.py`, `data/decay.py`, or `utils.py::bayesian_probability()` (the files `CLAUDE.md` names under "Modifying Bayesian Calculation"), grep `docs/docs/technical/` and `docs/docs/features/` for any prose description of the specific mechanism you changed before opening the PR:

```bash
grep -rn "last_interval_end\|actual_period_end\|<your changed concept>" docs/docs/
```

If a technical page describes the old behavior, update it in the same PR — don't wait for a reviewer to catch it. This applies most to `technical/global-prior-flow.md`, `technical/time-prior-flow.md`, `technical/bayesian-calculation.md`, `technical/calculation-flow.md`, `technical/decay` content inside `deep-dive.md`, and `technical/transition-learning.md`.

## When a change needs docs at all

Use this checklist before deciding a PR doesn't need a doc touch:

- [ ] Does it change a formula, threshold, default, or constant a technical page states as fact? → update the relevant `technical/*.md`.
- [ ] Does it change what a user configures, sees, or can do (new sensor type, new service, new config field, new sensor entity)? → update the relevant `features/*.md` and, if user-visible enough, the README `## Features` bullet list.
- [ ] Does it ship something previously listed under README `## Planned Features`? → move the bullet.
- [ ] Does it change a config-flow step, migration behavior, or anything `CONTRIBUTING.md`/`CLAUDE.md` describes procedurally? → flag for a separate docs/process PR rather than silently drifting further (see landmines above).
- [ ] Is it a pure internal refactor with no behavior change (e.g. `refactor(analysis): hoist step helpers to module level for C901`)? → no doc update needed; say so explicitly in the PR description so a reviewer doesn't go looking for one.
- [ ] Did you add a new feature? → write the features/ + technical/ pair together (see adjacent-areas.md / transition-learning.md above), and add both to `mkdocs.yml`'s `nav:`.

## Provenance and maintenance

Date-stamped 2026-07-06 (post-merge sweep), `main` HEAD `17b71d2` (`feat: adjacent-areas — learned next-door room influence (#454)`). Integration **release** version is still `2026.5.17` — the 2026-07-06 merge wave (#454, #486, #488, #489, #491–#496) is on `main` but not yet in a tagged release; don't describe any of that wave's changes as "shipped in release" until a new tag/release exists.

Re-verification commands by volatile fact category:

- **Docs tree / nav completeness**: `find docs/docs -name '*.md' | sort` vs `grep -oE '[a-z-]+/[a-z-]+\.md' docs/mkdocs.yml | sort -u`
- **site_url / repo_url**: `grep -E 'site_url|repo_url' docs/mkdocs.yml`
- **Admonition syntax in use**: `grep -rn '^!!! ' docs/docs/**/*.md | wc -l` (mkdocs form) vs `grep -rln '^> \[!' docs/docs/**/*.md` (should be empty)
- **Mermaid usage sites**: `grep -rl '```mermaid' docs/docs/`
- **README structure/line count**: `grep -n '^## ' README.md`; `wc -l README.md`
- **Docstring style sample**: `grep -n '"""' -A 8 custom_components/area_occupancy/utils.py`
- **Commit prefix convention**: `git log --oneline -30`
- **Release notes source of truth**: `gh release view <latest-tag>` and compare against `CHANGELOG.md`'s last `## [` entry
- **CHANGELOG.md staleness**: `grep -n '^## \[' CHANGELOG.md | head -3` vs `gh release list --limit 3`
- **Branch strategy reality**: `git branch -r`; `gh pr list --json number,baseRefName,headRefName` (no open PRs as of this sweep)
- **Coverage threshold**: `grep -n fail_under pyproject.toml`
- **PR #491 status** (merged 2026-07-06 — re-check it's still merged, not reverted): `gh pr view 491 --json state,mergeable,baseRefName`
- **Feature/planned-feature drift**: `grep -n '^## Features' -A 30 README.md` and `grep -n '^## Planned Features' -A 10 README.md`, cross-check against `custom_components/area_occupancy/data/entity_type.py`'s `InputType` enum
