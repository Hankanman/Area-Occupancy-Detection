---
name: aod-research-methodology
description: Use when turning a hunch, user report, or measurement into an accepted change to AOD's Bayesian/learning math — deciding whether a root-cause theory is proven, designing the discriminating experiment (DB query, simulator run, or predicted-numbers test), shipping an experimental capability behind a config flag, or adversarially refuting a claimed bug/fix. Also load this before trusting any "verified" claim from a single home's data or a single refutation pass. NOT for routine bugfixes with an obvious, already-known mechanism (use aod-debugging-playbook) or for the state of specific frontier research questions (use aod-research-frontier).
---

# AOD Research Methodology

## What this covers

The discipline that turns a hunch about occupancy-detection accuracy into an accepted, merged
change in this repo. It defines the evidence bar a hypothesis must clear, shows how ideas here
are validated with predicted numbers *before* code is written, walks the idea lifecycle from
issue to release notes, and names the anti-patterns that have cost real time (fixing symptoms,
tests that encode the bug, trusting one home's data). It is written for a zero-context AI session
picking up this project cold.

## When NOT to use this

- A bug with a known, obvious mechanism and no live debate about root cause → use
  `aod-debugging-playbook`.
- You want the current list of open research questions / frontier ideas themselves, not the
  method for validating them → use `aod-research-frontier`.
- You're running an accuracy campaign across many homes/areas → use
  `aod-learning-accuracy-campaign`.
- You need the actual Bayesian formulas → use `bayesian-occupancy-reference`.
- You need PR/CI/CodeRabbit mechanics → use `aod-change-control`.

---

## The evidence bar

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
dependency* is `homeassistant==2026.2.2` (`pyproject.toml:26`) — four months older than the
version that exhibits the behavior. A refutation attempt that ran against the pinned test
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

## Hypothesis-predicts-numbers-before-running

The house style here is: state the number your hypothesis predicts, *then* run the experiment
(query, simulator, or test) that measures it, *then* compare. A hypothesis that can't commit to a
number before the data comes back is not falsifiable and doesn't clear the bar.

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

**Worked example 2 — PR #491 (global prior denominator, issue #483).** Hypothesis: the reporter's
kitchen, with true occupancy ~28-35%, should compute to a prior of roughly **0.25** once the
quiet-tail truncation bug is fixed, not 0.99. The existing unit test
`test_valid_calculation_sets_correct_prior` had literally *encoded the bug* — asserting 0.99 for
a scenario where 0.25 is correct — and PR #491 corrected the assertion to 0.25 and added a new
regression test for the overnight-quiet-tail case. (Verified directly: `gh pr view 491`.)

**Takeaway for your own hypotheses:** before running a DB query, a `scripts/visualize_distributions.py`
plot, or a simulator session, write down the number(s) you expect and why. If you can't derive a
number from the mechanism, you don't have a mechanism yet — you have a guess.

## The idea lifecycle here

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
     comes ON TOP OF the pre-merge DB-copy check in aod-learning-accuracy-campaign Phase 4 —
     two stages, not one)
   → release notes with the measured numbers (not just "fixed a bug")
   → docs update (features/*.md or technical/*.md, mkdocs site)
```

Notes on specific stages:

- **Discriminating experiment.** For prior/likelihood questions this is almost always a SQL
  query against `config/.storage/area_occupancy.db` (see `aod-proof-and-analysis-toolkit` for
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

## Experiment flags: how an experimental capability ships and graduates

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
   (via `aod-learning-accuracy-campaign`) validates or retunes its constants and the "tune from
   real data later" comment is replaced with a cited number. Until then it stays labeled
   first-pass/candidate — do not remove the caveat comment just because the feature merged.
4. **Retirement**: if a first-pass feature turns out wrong (not just untuned), it is documented as
   a failure in `aod-failure-archaeology`, not silently deleted — the record of *why* it didn't
   work is itself the deliverable.

## Where good ideas historically come from: the community IS the sensor network

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

## Anti-patterns (each has cost real time here)

- **Fixing symptoms without a mechanism.** The health/repairs subsystem's early history (PR #429
  through PR #474) is the cautionary tale: a naive/aware-datetime crash was patched
  (`dt_util.as_utc`), then a threshold was patched, then a second threshold, then an ignore-flag
  bug, then per-purpose multipliers — six rounds of forward-fixes across issues
  #444/#445/#455/#463/#465/#466/#468 before the system reached its current purpose-aware,
  media-player-exempt, sticky-ignore design. Each round fixed a real bug, but the *pattern* of
  shipping the next patch as soon as the current symptom stopped reproducing — without asking
  "what's the general shape of the thing that keeps recurring" — is why it took six rounds
  instead of one redesign. See `aod-failure-archaeology` for the full chronicle.
- **Tests that encode the bug.** `test_valid_calculation_sets_correct_prior` asserting 0.99 (the
  buggy output) rather than 0.25 (the correct output) is the concrete example — see PR #491
  above. Before trusting a green test suite as evidence a calculation is right, check whether the
  assertions were derived from a hand-computed expected value or copy-pasted from a prior run's
  output.
- **Trusting one home's data.** A single install's measured numbers (e.g. PR #486's 6-area/57-entity
  sample) are good enough to *justify shipping* a low-risk, reversible change (a display-precision
  setting, default-off), but are not sufficient to retune a `const.py` probability constant that
  affects every area's math — that requires the multi-home validation described in
  `aod-learning-accuracy-campaign`. Do not generalize "it worked on my kitchen" into a global
  default change without that step.

## Provenance and maintenance

Date-stamped 2026-07-06 (post-merge, main HEAD `17b71d2`), integration version still 2026.5.17
(per `custom_components/area_occupancy/manifest.json` and `pyproject.toml` at that date — none of
the 2026-07-06 merge wave has shipped in a tagged release yet). Facts verified directly against
the repo/GitHub during authoring (not taken solely from a secondary dossier):

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
