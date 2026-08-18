---
name: aod-docs-and-positioning
description: Use when writing or editing anything under docs/docs/ (mkdocs site), README.md, CONTRIBUTING.md, CHANGELOG.md, docstrings, commit messages, PR titles, or GitHub Release notes for Area Occupancy Detection — or when a code change needs an accompanying doc update and you're deciding what/where to write. Trigger words — "update the docs", "add a features page", "write a docstring", "release notes", "mkdocs nav", "stale doc", "CHANGELOG". Also use when writing or reviewing anything that positions Area Occupancy Detection (AOD) against the rest of the Home Assistant ecosystem — README claims, release notes, "how is this different from X" answers, comparisons to HA core's bayesian platform or Bermuda/ESPresense, or any statement that a feature is "novel", "state of the art", or "proven". Also load before approving a PR description or doc page that cites accuracy/precision numbers, or before answering "is adjacency validated on real homes yet".
---

# AOD Docs and Positioning

## What this covers

The docs-of-record for Area Occupancy Detection: the mkdocs site under `docs/docs/`, `README.md`, `CONTRIBUTING.md`, docstring style, commit/PR title conventions, and how release notes actually get published. It tells you where a given fact belongs, what house style looks like (with quoted exemplars), which existing docs are known-stale landmines, and how to avoid re-creating the #491 stale-doc problem (code changed, doc didn't).

It also covers where AOD actually sits in the HA ecosystem today (2026-07-06), which of its
mechanisms are genuinely novel vs. commodity, the house standard for what
evidence a claim needs before it goes in README/release notes, and the four
maintainer-stated "beyond SOTA" ambitions with what each would require to
claim externally. This is the skill for **outward-facing claims** — what you
may say about AOD to the world, and what you may not say yet.

## When NOT to use this

- Deciding *what* the Bayesian math should say (formulas, constants, thresholds) — that's `aod-math-reference`.
- Deciding whether a code change is safe / needs a migration / breaks configs — that's `aod-change-and-validation`.
- Investigating *why* a historical bug happened — that's `aod-debugging-and-history`.
- For the actual open research questions (what to build/measure next on prior
  and likelihood learning), use `aod-research-frontier` and
  `aod-debugging-and-history` instead.
- For internal architecture facts (how the sigmoid pipeline actually computes), use `aod-architecture-and-config`
  or `aod-math-reference`. This skill is about *positioning claims*,
  not implementation.

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

Google-style docstrings, full type annotations (Python 3.13+), as directed by `AGENTS.md`. Verified against the actual codebase (`custom_components/area_occupancy/data/prior.py`, `utils.py`):

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

Module-level docstrings are one-liners or a short paragraph (`prior.py`'s module docstring: `"""Area baseline prior (P(room occupied) *before* current evidence).` followed by a two-sentence elaboration). Not every function needs an `Args`/`Returns` block — trivial one-liners (`format_percentage`, `format_value`) get a single-line docstring only. Add the full `Args:`/`Returns:` block once a function has more than one parameter or a non-obvious return, and always for anything touching the Bayesian calculation (per `AGENTS.md`'s "100% coverage for calculation changes" instruction — docs and tests should match that rigor).

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
| `AGENTS.md` "Branch and Release Strategy" section | "Development happens on `dev` branch. PR from `dev` to `preview` for prereleases. PR from `preview` to `main` for full releases." | **STILL A LIVE LANDMINE** — AGENTS.md is unchanged and still says dev→preview→main. The repo has no open PRs as of this sweep (the 2026-07-06 wave, #454/#491/#492/#493/#494/#495/#496, all merged direct to `main`), and `.github/workflows/release.yml` triggers on `release: published` (a manually-created GitHub Release), not a branch-merge event. There is no `preview` branch either. | `git branch -r`; `cat .github/workflows/release.yml`; `gh pr list --json baseRefName` |
| `pyproject.toml` (line 114, now line 113) | Inline comment `fail_under = 85 # Enforce 90% coverage minimum` — the enforced number and the comment's stated number disagreed with each other in the same line. | **SETTLED** (PR #495, merged 2026-07-06): now reads `fail_under = 85 # Enforced global minimum; aim for 90%+ on core calculation modules (AGENTS.md)` — internally consistent and cross-references `AGENTS.md`'s "85%+ coverage requirement (90% for core calculations)" line instead of contradicting it. | `grep -n fail_under pyproject.toml` |

Do not silently "fix" `AGENTS.md`'s branch-strategy section as a drive-by edit — it's explicit project instruction content; flag it to the maintainer or fold the correction into a PR whose primary purpose is a docs/process cleanup, per `aod-change-and-validation`'s rules on touching AGENTS.md. (The other two rows show this is a real, fixable pattern once someone owns it — AGENTS.md's branch-strategy section is the one that hasn't been picked up yet.)

## Keeping docs in sync when code changes — the #491 cautionary example

**Cautionary example (PR #491, `fix(prior): keep quiet tail in global prior denominator`, merged 2026-07-06 — verify with `gh pr view 491`):** the code change altered how `PriorAnalyzer.calculate_and_update_prior()` picks the end of its observation period (previously truncated to `last_interval_end` when the area had been quiet >1h; now always `now` — this is the current, permanent behavior on `main`, i.e. `actual_period_end` is always `now`, not conditionally truncated). `docs/docs/technical/global-prior-flow.md` documented the *old* (buggy) behavior in prose — "If last interval is more than 1 hour old: Use `last_interval_end` / Otherwise: Use current time" — and had to be edited in the same PR (7 additions, 16 deletions) to stop describing the bug as intended behavior. This was caught as a review nitpick, not by the original author remembering to update the doc.

**Rule extracted:** when a change touches `data/analysis.py`, `data/prior.py`, `data/decay.py`, or `utils.py`'s sigmoid/logit pipeline (the files `AGENTS.md` names under "Modifying Probability Calculation"), grep `docs/docs/technical/` and `docs/docs/features/` for any prose description of the specific mechanism you changed before opening the PR:

```bash
grep -rn "last_interval_end\|actual_period_end\|<your changed concept>" docs/docs/
```

If a technical page describes the old behavior, update it in the same PR — don't wait for a reviewer to catch it. This applies most to `technical/global-prior-flow.md`, `technical/time-prior-flow.md`, `technical/bayesian-calculation.md`, `technical/calculation-flow.md`, `technical/decay` content inside `deep-dive.md`, and `technical/transition-learning.md`.

## When a change needs docs at all

Use this checklist before deciding a PR doesn't need a doc touch:

- [ ] Does it change a formula, threshold, default, or constant a technical page states as fact? → update the relevant `technical/*.md`.
- [ ] Does it change what a user configures, sees, or can do (new sensor type, new service, new config field, new sensor entity)? → update the relevant `features/*.md` and, if user-visible enough, the README `## Features` bullet list.
- [ ] Does it ship something previously listed under README `## Planned Features`? → move the bullet.
- [ ] Does it change a config-flow step, migration behavior, or anything `CONTRIBUTING.md`/`AGENTS.md` describes procedurally? → flag for a separate docs/process PR rather than silently drifting further (see landmines above).
- [ ] Is it a pure internal refactor with no behavior change (e.g. `refactor(analysis): hoist step helpers to module level for C901`)? → no doc update needed; say so explicitly in the PR description so a reviewer doesn't go looking for one.
- [ ] Did you add a new feature? → write the features/ + technical/ pair together (see adjacent-areas.md / transition-learning.md above), and add both to `mkdocs.yml`'s `nav:`.

## 1. What AOD actually is (verified facts)

| Fact | Value | Verified |
|---|---|---|
| Distribution | HACS custom integration (not HA core) | `hacs.json`: `render_readme`, `zip_release: true`, `filename: area_occupancy.zip` |
| Stars | 308 | `gh repo view Hankanman/Area-Occupancy-Detection --json stargazerCount` |
| Min HA version declared | 2024.8.0 | `hacs.json`: `"homeassistant": "2024.8.0"` |
| HA version actually tested against | 2026.7.1 | `pyproject.toml` pins `homeassistant==2026.7.1` (dependency refresh, PR #496, merged 2026-07-06) |
| Docs site | `hankanman.github.io/Area-Occupancy-Detection/` | `gh repo view` `homepageUrl` |
| Release scheme | CalVer `YYYY.M.N` (not the SemVer AGENTS.md's release section describes) | `gh release list`, e.g. `2026.5.17` |

Do not describe AOD as "part of Home Assistant" or "an HA core feature" — it
is a third-party custom component installed via HACS, competing/complementing
core's own primitives (see §2).

## 2. Ecosystem position: competitors and complements

| Project | Relationship | Key difference |
|---|---|---|
| Raw motion sensor + automation timeout | Commodity baseline AOD replaces | Binary on/off, no context, no learning; README's own comparison table (README.md lines 26-32) is the canonical framing to reuse |
| HA core `binary_sensor.bayesian` platform | Same math family, different product | See §3 below — do not claim "we invented Bayesian occupancy in HA," core already ships it |
| Template sensors / Jinja groups | Commodity AOD replaces | README's "Replace Dozens of Templates and Groups" framing (README.md line ~19) |
| Bermuda BLE trilateration (`agittins/bermuda`) | Complementary, active collaboration thread | Issue #25 (opened 2025-03-01, oldest open roadmap issue, labels `on roadmap`+`enhancement`). Bermuda's maintainer (`agittins`) posted directly on #25 (2025-10-24) that Bermuda plans to add per-area `OCCUPANCY`-device-class boolean entities so users can pick "use for occupancy" per tracked BLE device — once that ships, those entities are a natural additional binary input to AOD, not a replacement for it. As of 2026-07-06 this integration point does **not exist yet** on either side — do not describe BLE/Bermuda ingestion as shipped. AOD's own README "Planned Features" lists "Location Aware: Leveraging BLE, WiFi, GPS" (README.md line 104) as future work, unimplemented. |
| ESPresense-style room-presence (WiFi/BLE RSSI room inference) | Complementary, same category as Bermuda | Not yet integrated; same "future input signal" status as Bermuda |

**House rule:** when asked to compare AOD to Bermuda/ESPresense, say they are
complementary location/trilateration signal sources AOD does not yet consume,
not competitors to be dismissed — the maintainer has an open, friendly
upstream thread (#25), not a rivalry.

## 3. Novel vs. commodity, as of 2026-07-06

**Commodity — already exists in HA core or elsewhere, don't oversell:**

- **Naive-Bayes sensor fusion itself.** HA core's `bayesian` binary_sensor
  platform (`home-assistant.io/integrations/bayesian/`) already combines a
  user-specified `prior` with per-observation `prob_given_true`/
  `prob_given_false` via Bayes' rule to produce a posterior against a
  threshold. AOD's core probability pipeline is a **superset in mechanism,
  not in kind**: it uses a sigmoid/logit-space combination (`sigmoid_probability`
  in `utils.py`), not the log-odds accumulator naive-Bayes core uses — but the
  underlying idea ("combine sensor states probabilistically instead of
  boolean AND/OR") is the same idea core already ships. The real differentiators
  are what feeds the model (below), not the existence of Bayesian fusion.
  Do not write "AOD brings Bayesian probability to Home Assistant" — core
  already has that. Write "AOD's fusion model is learned and time-aware where
  core's bayesian platform requires the user to hand-specify every prior and
  likelihood statically in YAML with no history, no decay, and no per-sensor
  auto-tuning."

**Genuinely novel (as a *combination*; verify each claim's current merge state before citing) — the differentiators to lead with:**

1. **Learned time-of-week priors from recorder history.** 168 (day-of-week ×
   hour-of-day) buckets per area, computed from actual `recorder` history via
   the hourly analysis pipeline, DST-safe (walks in UTC, buckets by local
   wall-clock). Nothing hand-configured. This is on `main` today.
2. **Purpose-based decay semantics.** Per-area "purpose" (bedroom, office,
   kitchen, passageway, etc.) drives default half-life and floor/threshold
   behavior automatically, including a sleep/wake half-life split for
   bedrooms — a room-type-aware decay model, not a single global timeout.
   On `main` today (though see `aod-debugging-and-history` for the recurring
   custom-vs-default-value bug class in this exact mechanism — #439/#440,
   #481/#493).
3. **Wasp-in-Box.** Virtual sensor purpose-built for single-entry/exit rooms
   (bathrooms) where motion sensors can't see the whole space: door-close
   with no motion still holds high occupancy. On `main` today.
4. **Adjacency / transition learning (learned neighbor influence).** Learns
   room-to-room transition probabilities from observed history (not a
   hand-tuned "if kitchen occupied, boost dining room by X%" static rule) and
   applies both a Bayesian logit-space boost and a decay-half-life stretch to
   neighboring areas. This is PR #454 — **merged 2026-07-06, now on `main`**
   (squash merge, main HEAD `17b71d2`). It directly answers community
   discussion #431 (a user request for exactly this feature, unanswered by
   the requester's own suggested static-config approach — AOD's design
   instead learns influence from data).

**No-oversell rule for #4 specifically:** every constant driving the
adjacency boost/decay-modifier math (`ADJACENCY_BOOST_GAIN=0.5`,
`ADJACENCY_DECAY_MODIFIER_GAIN=0.75`, `ADJACENCY_DECAY_MODIFIER_MAX=1.75`,
the four `ADJACENCY_N_*` smoothing-fallback minimum-observation thresholds)
carries the exact in-repo comment (`const.py` lines 190-191): *"First-pass
values; tune from real data once Phase 3 is collecting transitions."* There
is no real-household validation of these numbers anywhere in the repo — no
recorder-derived accuracy measurement exists for the adjacency feature at
all (the test suite uses synthetic/mocked entities only). **Label adjacency
"candidate" / "unvalidated on real homes," not "proven" or "state of the
art," in any external-facing writing until the learning-accuracy campaign
(see `aod-debugging-and-history`) produces a real measurement.**

## 4. The house standard for publishing a claim

Before README, release notes, or any external doc states a quantitative
claim, it must meet the bar set by PR #486 (the diagnostic-sensor-precision
fix) — this is the canonical "how we prove a change worked" example in this
project's own history:

> Measured on a live 6-area installation (v2026.5.17, 57 AOD entities):
> Afternoon (active) baseline 15,952 recorder rows/3h → Evening 7,058 rows/3h
> (−55%) → Morning 3,323 rows/3h (−79%).

What made that claim publishable (verify against `gh pr view 486` — do not
paraphrase from memory):

- **Real installation, not synthetic fixture data** — "live 6-area
  installation," named HA version and entity count.
- **Explicit method** — what was measured (recorder row count), over what
  window (3h), under what condition (time-of-day activity level), against
  what baseline (precision=2, the pre-change default).
- **Before/after with a stated delta**, not just an after-number.
- **A stated mechanism for why the effect is real**, not coincidental (the
  PR explains *why* precision-0 reduces rows: sub-decimal noise is quantized
  away while genuine decay-driven whole-percent transitions still record).

**Checklist for any new external claim:**
- [ ] Measured against a real (or explicitly-labeled synthetic) dataset — name which
- [ ] States the exact metric, window, and baseline compared against
- [ ] Gives a before/after delta, not a single absolute number
- [ ] Explains the causal mechanism, not just correlation
- [ ] Cites the PR/issue number it came from, so a future session can `gh pr view` it
- [ ] Does not claim something is "solved"/"validated" when the repo shows it as unmerged or unvalidated — check merge state first

If a claim doesn't clear this bar, it stays internal (research-frontier /
learning-accuracy-campaign territory) — do not promote it to README or
release notes yet.

## 5. Reproducibility bar for any published comparison

Any comparison of AOD's accuracy/behavior against a prior version, a
competitor, or a baseline (raw motion sensor, HA core bayesian) must specify,
at minimum:

1. **AOD version** (the `manifest.json`/`pyproject.toml`/`const.py`
   `DEVICE_SW_VERSION` triplet — see `aod-change-and-validation` for why these three
   must move together).
2. **Data source**: real recorder history from a named install (household
   size/area count, like PR #486's "6-area installation, 57 entities") vs.
   synthetic fixtures (`config/configuration.yaml`'s 5-room mock rig) —
   never blur the two. A synthetic-fixture result is a sanity check, not an
   accuracy claim.
3. **Sample size**: this project's own internal bar for trusting a learned
   correlation is `MIN_CORRELATION_SAMPLES = 50` (`const.py:323` on main as of 2026-07-06; line drifts by branch — `grep -n MIN_CORRELATION_SAMPLES custom_components/area_occupancy/const.py`) — any
   external accuracy comparison with fewer than 50 underlying observations
   per area/sensor should be labeled preliminary, not a result.
4. **The exact metric** (precision/recall on occupied-interval detection,
   recorder-row count, false-repair rate, etc.) with its formula or query
   spelled out well enough that a future session could recompute it from the
   same DB (`config/.storage/area_occupancy.db`) — see
   `aod-math-reference` for the actual measurement tools
   (`scripts/visualize_distributions.py`, the `run_analysis` service +
   simulator).
5. **What's held constant**: same purpose/threshold/decay config across the
   before/after, unless the config change itself is the thing being measured.

## 6. The four SOTA ambitions and what each needs externally

The maintainer named these four as what "beyond SOTA" means for this
project. None are claimable externally yet on today's evidence — each needs
a specific kind of proof first:

| Ambition | What it means | What must be true before you claim it publicly |
|---|---|---|
| **Reliability king (primary)** | AOD's occupancy calls are more trustworthy than raw motion/naive automations across real, messy households — this is the maintainer's stated #1 priority | A multi-household measured comparison (precision/recall or false-trigger rate) of AOD vs. a documented naive-motion-sensor baseline, using the PR #486-standard method (§4). No such measurement exists in the repo today — this is the single most important gap; see `aod-debugging-and-history` |
| **Learned-model research** | Priors/likelihoods/adjacency learned from data, not hand-tuned, is a research contribution worth publishing/citing | Needs a written methodology (sampling, validation split, real-household data) that survives scrutiny — not just "we learn from history" as a slogan. The current adjacency tunables are explicitly first-pass/unvalidated (§3) — this ambition is blocked on the same accuracy-measurement gap as reliability king |
| **HA-core quality** | Code quality, test coverage, config-migration discipline on par with what HA core requires of built-in integrations | Concretely measurable today: 85% coverage gate (`pyproject.toml`, enforced by `scripts/test`), full type annotations, Google-style docstrings, ruff-enforced style (see `aod-build-run-and-release`) — this ambition is the closest to already being true and is the easiest to defend externally with a number (current coverage %, run `scripts/test` to reverify) |
| **Predictive occupancy** | Not just detecting current occupancy but predicting near-future occupancy/transitions (the adjacency trajectory work is a first step toward this) | Needs an explicit prediction-vs-outcome evaluation (e.g., "predicted transition to area X within N minutes, did it happen") — no such evaluation exists yet; adjacency's trajectory tracking computes a *boost signal*, not a scored prediction, so today's code is an ingredient, not a demonstrated predictive-occupancy result |

None of these four should appear in README/release notes as accomplished
facts. They are legitimate *direction* statements (roadmap language is fine:
"working toward reliability across real households") but any "we are the
most reliable/most predictive" framing needs the measurement described in
§4-5 first.

## Provenance and maintenance

Date-stamped 2026-07-06 (post-merge sweep), `main` HEAD `17b71d2` (`feat: adjacent-areas — learned next-door room influence (#454)`). Integration **release** version is still `2026.5.17` — the 2026-07-06 merge wave (#454, #486, #488, #489, #491–#496) is on `main` but not yet in a tagged release; don't describe any of that wave's changes as "shipped in release" until a new tag/release exists.

PRs #454 (adjacent-areas), #491, #492, #493, #494 were **merged 2026-07-06**
(squash merges) — main HEAD is now `17b71d2`. Re-verify merge state before
citing any of their content as shipped if working from an older checkout.

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
- **Star count / repo metadata**: `gh repo view Hankanman/Area-Occupancy-Detection --json stargazerCount,description,homepageUrl`
- **Adjacent-areas / PR #454 merge state** (merged 2026-07-06; reconfirm on an older checkout): `gh pr view 454 --json state,mergedAt`
- **Same-day bugfix PRs #491-494 merge state** (all merged 2026-07-06): `gh pr view 491 492 493 494 --json state,mergedAt` (run each individually; `gh pr view` takes one number at a time)
- **Bermuda thread #25 latest status**: `gh issue view 25 --json state,comments,labels`
- **Adjacency tunables / "first-pass" disclaimer still present**: `grep -n "First-pass values" custom_components/area_occupancy/const.py`
- **PR #486 measured numbers (15,952/7,058/3,323 rows, −55%/−79%)**: `gh pr view 486 --json body`
- **HA core bayesian platform current behavior**: fetch `https://www.home-assistant.io/integrations/bayesian/` (external, may change independent of this repo)
- **Current release version / CalVer scheme**: `gh release list --limit 5`
- **Coverage gate percentage**: `grep -n "fail_under\|fail-under" pyproject.toml`; confirm live number via `scripts/test`
- **MIN_CORRELATION_SAMPLES threshold**: `grep -n "MIN_CORRELATION_SAMPLES" custom_components/area_occupancy/const.py`
