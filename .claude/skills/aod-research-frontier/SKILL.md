---
name: aod-research-frontier
description: Use when scoping new research, a thesis/blog post, a grant-style pitch, or a "what should we work on next" planning session for Area Occupancy Detection — maps open, unproven "beyond state-of-the-art" problems (reliability vs raw sensors, per-home transition learning, predictive pre-heat/pre-light, Home-Assistant-core-grade quality) to concrete first steps in this repo. Do not use for day-to-day bugfixing, the active prior/likelihood accuracy campaign, or exact formula lookup — see "When NOT to use this" below.
---

# AOD Research Frontier

## What this covers

This is a map of **open, unproven** research directions where Area Occupancy
Detection (AOD) could measurably exceed what a naive Home Assistant motion
sensor or template does today. Every problem below is labeled
**candidate/open** — nothing here has been validated on real data yet, and
none of it is a merged feature unless stated otherwise. It exists so a
zero-context future session can pick up a frontier problem and know exactly
where to start reading code, what to build first, and what result would
prove (or falsify) the idea — without re-deriving any of this from scratch.

## When NOT to use this

- Actively fixing a specific prior/likelihood accuracy bug or running the
  ongoing accuracy-improvement campaign → **aod-learning-accuracy-campaign**.
- You need the exact Bayesian formula, a constant's value, or the calculation
  pipeline's call graph → **bayesian-occupancy-reference** (and
  **aod-architecture-contract** for the module map).
- You need the story of a past incident (prior pinned at 0.99, decay
  half-life bugs, timezone/DST bugs) → **aod-failure-archaeology**.
- You're about to change calculation code and need the process/guardrails →
  **aod-change-control**.
- You want existing scripts/tools to run an analysis, not to invent a new
  one → **aod-proof-and-analysis-toolkit**.
- You want marketing/competitor-comparison framing for README or docs →
  **aod-external-positioning**.

---

## Ground-truth snapshot (verify dates before trusting)

| Fact | Value | Verify with |
|---|---|---|
| Integration version | 2026.7.1-pre1 (pre-release, published 2026-07-07; last GA 2026.5.17) | `cat custom_components/area_occupancy/manifest.json \| grep version` |
| Quality scale (HA manifest) | `silver` | `grep quality_scale custom_components/area_occupancy/manifest.json` |
| Runtime pip requirements | `[]` (none) | `grep -A2 '"requirements"' custom_components/area_occupancy/manifest.json` |
| Test coverage gate | `fail_under = 85` in pyproject, comment now reads "Enforced global minimum; aim for 90%+ on core calculation modules" — the old stale-comment mismatch was fixed by #495 (CI hygiene) | `grep -n fail_under pyproject.toml` |
| Live probability engine | sigmoid/logit pipeline in `utils.py` (`sigmoid_probability`, `presence_probability`, `environmental_confidence`, `combined_probability`), driven from `area/area.py::Area._base_probability/probability` | `grep -n "def sigmoid_probability\|def presence_probability\|def combined_probability" custom_components/area_occupancy/utils.py` |
| Dead legacy engine | `utils.py::bayesian_probability()` (classic log-odds accumulator) has **zero production call sites** — only unit tests in `tests/test_utils.py` still exercise it. CLAUDE.md's "Modifying Bayesian Calculation" workflow points at this function first — that pointer is stale; start from `sigmoid_probability`/`presence_probability` instead. | `grep -rn "bayesian_probability(" custom_components/area_occupancy/` (only the `def` line should show) |
| Adjacent-areas feature (frontier #2 asset) | **Merged 2026-07-06** — PR #454 merged into `main` (squash), superseding PR #456 which was closed as merged into it. `data/adjacency.py`, `data/trajectory.py`, `db/transitions.py`, the `AreaTransitions` table, and all `ADJACENCY_*` constants are on `main` now. Still **unvalidated on real homes** — candidate/open labeling for the research question stays even though the code shipped. | `gh pr view 454 --json state,mergeStateStatus` |
| Prior/decay/sleep accuracy fixes | PRs #491 (prior quiet-tail), #492 (sleep unknown-presence), #493 (bedroom half-life), #494 (README link) — **all merged 2026-07-06, on `main`** | `gh pr view 491` / `492` / `493` / `494` |

`main` HEAD is now `17b71d2`. The `data/adjacency.py`, `data/trajectory.py`,
`db/transitions.py` code paths described under Frontier #2 exist on `main`
today — confirm with
`git show main:custom_components/area_occupancy/const.py | grep -c ADJACENCY_`
(returns `10` on main as of this writing, not `0`).

---

## Roadmap epics (filed 2026-07-07 — the frontiers made concrete)

The maintainer converted this skill's frontiers into five tracked epics with
agreed sequencing. Each issue carries first-steps and a falsifiable
"you have a result when" gate — read the issue before starting work on a
frontier; it is the live version of the corresponding section below.

| # | Epic | Maps to | Sequencing |
|---|---|---|---|
| [#499](https://github.com/Hankanman/Area-Occupancy-Detection/issues/499) | Per-area trust score: calibration error, false-transition rate, earned auto-threshold | Frontier 1 (Reliability King) | **First** — it is the measurement harness every other epic validates against |
| [#500](https://github.com/Hankanman/Area-Occupancy-Detection/issues/500) | Retire the sidecar SQLite DB: online learning via sufficient statistics | Frontier 4 enabler (removes the core-quality blocker) | Second, parallel with #501; shadow-mode only until diffed for 30 days |
| [#501](https://github.com/Hankanman/Area-Occupancy-Detection/issues/501) | Learned sensor fusion: per-home trained weights/likelihoods (subsumes #93/#159/#458) | Frontier 2 (learned-model research) | Second, parallel with #500 |
| [#502](https://github.com/Hankanman/Area-Occupancy-Detection/issues/502) | Predictive occupancy: P(occupied within N min) + likely-next-area | Frontier 3 | Third — needs adjacency data maturity + #499 scoring |
| [#503](https://github.com/Hankanman/Area-Occupancy-Detection/issues/503) | Person-level occupancy: Bermuda identity fusion (continues #25) | Near frontier → promoted | Whenever Bermuda ships their side; steps 1–2 unblocked today |

Development model: roadmap epics build on the `next` branch; `main` stays the
stable/hotfix line for the current release family.

---

## Frontier 1 — Reliability King: fewer false occupancy transitions [PRIMARY]

**The ambition:** AOD should provably produce fewer spurious
occupied↔unoccupied flips than raw motion sensors or a naive HA template,
on real homes, with a number you can defend.

**Why current "SOTA" (a raw motion-sensor automation or HA template) fails:**
a bare motion sensor has no memory — it drops to "clear" the instant its own
built-in clear delay expires, regardless of whether anyone actually left.
There is no learned baseline, no fusion across sensor types, and no decay
curve — every clear event is a potential false transition, and users patch
this today with ad-hoc automation timeout hacks (this is literally the
comparison table AOD's own README already draws — see
**aod-external-positioning** for that framing; this skill is about proving
the claim with a number, not stating it).

**This project's specific asset:** a full pipeline already deployed on real
homes that fuses evidence across `InputType`s (motion, media, appliance,
door, environmental — see `data/entity_type.py`), learns a per-area,
per-168-hour-bucket time prior (`data/prior.py`, `db/schema.py` `Priors`
table), and applies purpose-aware decay (`data/decay.py`,
`data/purpose.py`) — plus a recorder-backed history
(`db/queries.py::get_occupied_intervals`) that already reconstructs
"was this area actually occupied" ground truth from motion-only intervals.
Nothing today, however, measures *false transition rate* — this is unbuilt.

**First three concrete steps in this repo:**
1. Define the metric precisely against `db/queries.py::get_occupied_intervals`
   (the existing motion-ground-truth reconstruction) — e.g. count of
   occupied→unoccupied→occupied flips within a short window (say 2× the
   area's configured decay half-life) as a proxy for "spurious," and diff
   that count between the raw motion-only ground truth and AOD's
   `probability() >= threshold` occupied series over the same historical
   window. Write this definition down in a design note before touching code
   — it is the single most falsifiable choice in this whole skill.
2. Build a comparator harness as an offline script (see
   **aod-proof-and-analysis-toolkit** for where existing analysis scripts
   like `scripts/visualize_distributions.py` and the `simulator/` web app
   already replay analysis output — extend one of those rather than
   starting cold) that replays recorder history through both "raw motion
   sensor" and "AOD `probability()`" and counts flips for each.
3. Run it against at least one real, already-configured area's history
   (via `db.sync_states` + `get_occupied_intervals`, no live HA needed) and
   publish the method (not just the number) as a doc page under
   `docs/docs/technical/` so the comparison is reproducible by a skeptic.

**You have a result when:** you can show, on **≥2 real homes**, over an
**N-day window you pre-register before looking at the data** (so you can't
p-hack the window), a measured reduction in false-transition count for
AOD vs. raw motion, with the method published such that someone else could
rerun it and get the same number. Anything short of that (a single home, a
cherry-picked week, an eyeballed chart) is not a result — it's a demo.

---

## Frontier 2 — Learned-model research: per-home transition models

**The ambition:** learn each home's actual room-to-room movement patterns
(not hand-configured rules) and use them to inform occupancy.

**Why current SOTA fails:** static "if area A occupied, boost area B" rules
(the kind users ask for in Discussion #431 — "no motion next door raises
confidence, motion next door lowers it") don't adapt per home, per time of
day, or to the fact that some adjacent-area pairs are much more predictive
than others (a hallway→bedroom transition at 11pm is very different signal
than the same pair at 2pm).

**This project's specific asset:** the `AreaTransitions` table
(`db/schema.py`, class defined around line 692 — **merged to `main` 2026-07-06
via PR #454**) is a genuinely novel local
dataset: it records observed `(from_area, mid_area, to_area, hour_of_week)`
transition counts with exponential recency decay
(`ADJACENCY_RECENCY_HALF_LIFE_DAYS = 30`), supporting both 1-hop
(`mid_area = ""` sentinel) and 2-hop chains, bucketed into the same 168
day-of-week × hour-of-day grid as the existing time-priors. The lookup
(`db/transitions.py::lookup_transition_probability`) already implements a
6-level specificity-with-minimum-sample-size smoothing fallback
(`ADJACENCY_N_SPECIFIC=5` for a specific 2-hop/hour-of-week chain, down to
`ADJACENCY_N_PAIR=20` for an un-bucketed 1-hop pair, down to a static
default). The boost is applied in logit space
(`data/adjacency.py::compute_adjacency_boost`, gain
`ADJACENCY_BOOST_GAIN=0.5`) and a matching decay-modifier
(`compute_decay_modifier`, gain `ADJACENCY_DECAY_MODIFIER_GAIN=0.75`, capped
at `ADJACENCY_DECAY_MODIFIER_MAX=1.75`) slows decay when an adjacent area's
trajectory suggests the person is still nearby.

**The built-in A/B toggle:** adjacency influence is **zero unless the user
has configured adjacent-area pairs for that area** (`CONF_ADJACENT_AREAS`
empty → no transitions recorded → `lookup_transition_probability` falls
through to the static default with `observed/total` forced to 0, which is
the documented "no data" signal). That means every home with *some* areas
configured with neighbors and some without already has a natural,
per-area, config-driven control group — no separate feature flag needed.

**First three concrete steps in this repo (#454 merged 2026-07-06, code is
live on `main`):**
1. Pull the full test suite for the feature:
   `uv run pytest tests/test_data_adjacency.py tests/test_data_trajectory.py
   tests/test_coordinator_adjacency.py tests/test_db_relationships.py -v`
   (all four files exist on `main` and pass as of this writing).
2. Instrument a calibration comparison: for areas with adjacency configured,
   compare predicted-probability calibration (e.g. reliability diagram —
   see **bayesian-occupancy-reference** for how probability outputs are
   already clamped/computed) against otherwise-similar areas with no
   adjacency configured, using the same recorder-history replay approach as
   Frontier 1's harness.
3. Because `ADJACENCY_*` constants in `const.py` (lines 194–221) are
   explicitly commented as "first-pass values; tune from real data once
   Phase 3 is collecting transitions" — treat them as hypotheses, not
   defaults to trust. Any tuning must go through **aod-change-control**
   (no silent math changes) since they affect every configured-adjacent
   area's live probability.

**You have a result when:** on real per-home data, calibration (not just
raw accuracy) for adjacency-enabled areas is demonstrably better than for
otherwise-comparable adjacency-disabled areas — the config toggle *is* the
A/B split, so state which areas were in which arm and why they're
comparable (similar purpose, similar sensor count) before reporting a
delta.

---

## Frontier 3 — Predictive occupancy: anticipate the next area before entry

**The ambition:** use learned time-of-day patterns plus transition chains to
predict "person is about to enter area X" *before* they cross the threshold,
enabling pre-heat/pre-light automations — a capability no reactive
motion-sensor-based system can offer by construction.

**Why current SOTA fails:** every occupancy signal (PIR, mmWave, door
contact) is inherently reactive — it fires only once someone is already
in the space. There is no anticipatory signal on the market built from a
home's own learned movement history.

**This project's specific asset:** the same two learned structures as
Frontier 1 and 2 — the 168-bucket time-priors (`data/prior.py`,
`Priors` table in `db/schema.py`) already encode "how likely is this area
occupied at this day/hour," and (merged 2026-07-06 via PR #454) the
1-hop/2-hop `AreaTransitions` chains encode "given the last one or two areas
occupied, where does the person go next." Combined, these are exactly the
inputs a next-area predictor needs, and they are **already being learned
today** — no new data collection is required to start the offline
evaluation.

**First three concrete steps in this repo:**
1. Do this **entirely offline against existing DB history first** — do not
   create a new entity or sensor yet. Write an evaluation script that, for
   each historical occupied-interval transition in `AreaTransitions`
   (now populated directly on `main`; cross-check against
   `get_occupied_intervals` across all areas in an entry), checks whether
   the time-prior + transition-chain
   combination would have predicted the *next* area correctly within N
   minutes, using a held-out time split (train on weeks 1–3, evaluate on
   week 4 — never randomly shuffled, since the whole premise is temporal).
2. Compute precision/recall (not just "top-1 accuracy") for a few candidate
   lookahead windows (e.g. N = 2, 5, 10 minutes) and a few confidence
   thresholds, and pick the pair that a real pre-heat/pre-light automation
   could tolerate (false positives here mean wasted energy, not just a
   wrong log line — that is a stricter bar than the passive probability
   sensor's day-to-day tolerance).
3. Only after the offline numbers clear a pre-agreed bar, propose the
   sensor entity design and document the intended automation pattern in
   `docs/docs/features/` — shipping the entity before proving the
   accuracy number inverts the maintainer's own stated law ("no silent
   math changes" extends to "no shipped-but-unvalidated predictive
   claims").

**You have a result when:** you have a precision/recall number *on
historical data, from a temporally-held-out split*, that clears a bar you
set before you shipped anything — stated as "at confidence threshold T and
lookahead N minutes, precision = X%, recall = Y%" — not before any user
sees a new entity.

---

## Frontier 4 — HA-core quality: gaps vs. Home Assistant core acceptance bar

**The ambition:** this integration should be engineerable to the standard
of a Home Assistant **core** (bundled) integration, even though it will
likely remain a HACS custom component. `manifest.json` already declares
`"quality_scale": "silver"` — treat "gold"/"platinum"-equivalent rigor
(HA's own scale; see HA developer docs for the current rubric, which
changes over time and should be re-checked, not assumed) as the frontier
target.

**What's already aligned (verified, don't re-litigate):**
- `manifest.json` `"requirements": []` — zero extra pip dependencies beyond
  Home Assistant itself and its declared `"dependencies": ["recorder",
  "sensor", "binary_sensor", "number"]` HA-domain dependencies. This matches
  core's strong preference against adding new third-party packages.
- Google-style docstrings enforced by ruff's `pydocstyle` convention
  (`pyproject.toml`, `[tool.ruff.lint.pydocstyle] convention = "google"`).
- Full type annotations required per CLAUDE.md, Python 3.14+ (`pyproject.toml`
  `requires-python = ">=3.14.2"`, bumped via #496 2026-07-06 alongside the
  HA 2026.7.1 dependency refresh).

**Concrete gaps to verify and close (check current state before acting —
these move over time):**
| Gap | How to check | Owning skill for the fix itself |
|---|---|---|
| No `quality_scale.yaml` rule-by-rule justification file (core integrations at gold/platinum carry one) | `find custom_components/area_occupancy -iname '*quality_scale*'` (empty as of this writing) | aod-config-and-flags / aod-architecture-contract |
| Coverage gate stale-comment mismatch — **fixed by #495 (merged 2026-07-06)**; comment now reads "Enforced global minimum; aim for 90%+ on core calculation modules", matching the enforced `fail_under = 85` | `grep -n fail_under pyproject.toml` | aod-validation-and-qa |
| No `mypy`/`pyright` static-type gate found in `pyproject.toml` or `scripts/` (only ruff) — core increasingly expects strict typing enforcement, not just annotations-present. Still an open gap as of 2026-07-06. | `grep -n 'mypy\|pyright' pyproject.toml scripts/*` | aod-build-and-env |
| Branch strategy documented in CLAUDE.md (`dev`→`preview`/`rc`→`main`) is stale — CI now triggers lint/test/validate on `main` only (rc/dev removed, per #495), and `dev`/`preview`/`rc` branches no longer exist on the remote; CLAUDE.md itself has not been corrected yet | `git ls-remote --heads origin \| grep -E 'dev$\|preview$\|rc$'` (expect empty) | aod-change-control |

**First three concrete steps in this repo:**
1. Pull HA's current Integration Quality Scale rubric (it is versioned and
   changes — do not rely on a cached copy) and diff it rule-by-rule against
   this integration; write the `quality_scale.yaml` even if the target
   stays "silver" for now, since the act of writing it surfaces gaps.
2. Add a static type-checking gate (`mypy` or `pyright`) to `scripts/lint`
   or CI, scoped narrowly at first (e.g. `utils.py`, `data/prior.py`, the
   calculation hot path) since full-repo strict typing is a bigger lift.
3. File the CLAUDE.md branch-strategy correction as its own small PR (docs
   only, low risk) — this is a "the repo shows CLAUDE.md is stale" case
   explicitly called out for correction, not a silent change.

**You have a result when:** a rule-by-rule `quality_scale.yaml` exists and
is checked against HA's current published rubric, and CI enforces at least
one static-typing gate beyond ruff's lint rules.

---

## Near frontier: named open issues (smaller, more tractable)

These are not "advance SOTA" ambitions — they're concrete, scoped user asks
already sitting in the issue tracker that a future session could pick up
directly. All verified open as of 2026-07-06 via `gh issue view <n>`.

| # | Title | Labels | Relevance |
|---|---|---|---|
| #93 | Add support for ignoring animals | `enhancement`, `on roadmap` | Negative-evidence case: user wants a motion-like signal that *lowers* occupancy confidence (e.g. a camera's animal-detected event) — no `InputType` today supports negative-weight evidence; see `data/entity_type.py` for the type registry this would extend. |
| #159 | Allow per-sensor active states | `enhancement`, `on roadmap` | Today `active_states` is per `InputType` default, not per configured entity instance — user has sensors where "on" means occupied for one and "off" means occupied for another and can't mix them in one area. |
| #458 | Weight per motion sensor | *(no labels)* | Same shape as #159 but for `weight`/timeout instead of active state — currently one weight applies to every motion sensor of a given type in an area; concrete case cited: a T-shaped room with 3 motion sensors that should weight differently. |
| #25 | Bermuda Integration / Collaboration | `enhancement`, `on roadmap` | Oldest open roadmap issue (~16 months as of 2026-07-06). Bermuda is BLE-trilateration room-presence — a genuinely different signal source (radio proximity, not binary motion) AOD doesn't ingest today. This is the concrete instance of the README's own "Planned Features: BLE/WiFi/GPS location-awareness" bullet. |

Re-check labels/state before acting — `gh issue view <n> --json state,labels,title`.

---

## Provenance and maintenance

Compiled 2026-07-06, integration version 2026.5.17 (per
`custom_components/area_occupancy/manifest.json` — no tagged release has
shipped today's merge wave yet, so "shipped in release" language should
still say "on `main`, not yet released"). Swept post-merge-wave 2026-07-06:
`feat/adjacent-areas` (PR #454) has merged into `main`; `main` HEAD is now
`17b71d2`. Frontier 2's code paths (`data/adjacency.py`, `data/trajectory.py`,
`db/transitions.py`, the `AreaTransitions` table, all `ADJACENCY_*`
constants) are live on `main` today — the *research question* (is adjacency
actually helping) remains open/unvalidated even though the code shipped.

Re-verification commands, by volatile fact category:

```bash
# Version / quality scale / requirements
cat custom_components/area_occupancy/manifest.json

# Is the adjacency feature merged yet?
gh pr view 454 --json state,mergeStateStatus,mergeable,baseRefName

# Are the accuracy-fix PRs merged yet?
gh pr view 491 --json state,mergeStateStatus
gh pr view 492 --json state,mergeStateStatus
gh pr view 493 --json state,mergeStateStatus

# Is bayesian_probability() still dead code?
grep -rn "bayesian_probability(" custom_components/area_occupancy/

# Do the near-frontier issues still carry the same labels/state?
gh issue view 93 --json state,labels
gh issue view 159 --json state,labels
gh issue view 458 --json state,labels
gh issue view 25 --json state,labels

# Coverage gate current value
grep -n fail_under pyproject.toml

# Branch strategy reality check (should be empty if dev/preview/rc are gone)
git ls-remote --heads origin | grep -E 'dev$|preview$|rc$'

# Does a quality_scale.yaml exist yet?
find custom_components/area_occupancy -iname '*quality_scale*'
```
