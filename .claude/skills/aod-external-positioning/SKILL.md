---
name: aod-external-positioning
description: Use when writing or reviewing anything that positions Area Occupancy Detection (AOD) against the rest of the Home Assistant ecosystem — README claims, release notes, "how is this different from X" answers, comparisons to HA core's bayesian platform or Bermuda/ESPresense, or any statement that a feature is "novel", "state of the art", or "proven". Also load before approving a PR description or doc page that cites accuracy/precision numbers, or before answering "is adjacency validated on real homes yet".
---

# AOD External Positioning

## What this covers

Where AOD actually sits in the HA ecosystem today (2026-07-06), which of its
mechanisms are genuinely novel vs. commodity, the house standard for what
evidence a claim needs before it goes in README/release notes, and the four
maintainer-stated "beyond SOTA" ambitions with what each would require to
claim externally. This is the skill for **outward-facing claims** — what you
may say about AOD to the world, and what you may not say yet.

## When NOT to use this

For the actual open research questions (what to build/measure next on prior
and likelihood learning), use `aod-research-frontier` and
`aod-learning-accuracy-campaign` instead. For internal architecture facts
(how the sigmoid pipeline actually computes), use `aod-architecture-contract`
or `bayesian-occupancy-reference`. This skill is about *positioning claims*,
not implementation.

## 1. What AOD actually is (verified facts)

| Fact | Value | Verified |
|---|---|---|
| Distribution | HACS custom integration (not HA core) | `hacs.json`: `render_readme`, `zip_release: true`, `filename: area_occupancy.zip` |
| Stars | 308 | `gh repo view Hankanman/Area-Occupancy-Detection --json stargazerCount` |
| Min HA version declared | 2024.8.0 | `hacs.json`: `"homeassistant": "2024.8.0"` |
| HA version actually tested against | 2026.2.2 | `pyproject.toml` pins `homeassistant==2026.2.2` |
| Docs site | `hankanman.github.io/Area-Occupancy-Detection/` | `gh repo view` `homepageUrl` |
| Release scheme | CalVer `YYYY.M.N` (not the SemVer CLAUDE.md's release section describes) | `gh release list`, e.g. `2026.5.17` |

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
   On `main` today (though see `aod-failure-archaeology` for the recurring
   custom-vs-default-value bug class in this exact mechanism — #439/#440,
   #481/#493).
3. **Wasp-in-Box.** Virtual sensor purpose-built for single-entry/exit rooms
   (bathrooms) where motion sensors can't see the whole space: door-close
   with no motion still holds high occupancy. On `main` today.
4. **Adjacency / transition learning (learned neighbor influence).** Learns
   room-to-room transition probabilities from observed history (not a
   hand-tuned "if kitchen occupied, boost dining room by X%" static rule) and
   applies both a Bayesian logit-space boost and a decay-half-life stretch to
   neighboring areas. This is PR #454 — **merging as of 2026-07-06, NOT yet on
   `main`. Verify current state with `gh pr view 454`** before describing it as
   shipped. It directly answers community discussion #431 (a user request for
   exactly this feature, unanswered by the requester's own suggested static-
   config approach — AOD's design instead learns influence from data).

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
(see `aod-learning-accuracy-campaign`) produces a real measurement.**

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
   `DEVICE_SW_VERSION` triplet — see `aod-change-control` for why these three
   must move together).
2. **Data source**: real recorder history from a named install (household
   size/area count, like PR #486's "6-area installation, 57 entities") vs.
   synthetic fixtures (`config/configuration.yaml`'s 5-room mock rig) —
   never blur the two. A synthetic-fixture result is a sanity check, not an
   accuracy claim.
3. **Sample size**: this project's own internal bar for trusting a learned
   correlation is `MIN_CORRELATION_SAMPLES = 50` (`const.py:286` on main; line drifts by branch — `grep -n MIN_CORRELATION_SAMPLES custom_components/area_occupancy/const.py`) — any
   external accuracy comparison with fewer than 50 underlying observations
   per area/sensor should be labeled preliminary, not a result.
4. **The exact metric** (precision/recall on occupied-interval detection,
   recorder-row count, false-repair rate, etc.) with its formula or query
   spelled out well enough that a future session could recompute it from the
   same DB (`config/.storage/area_occupancy.db`) — see
   `aod-proof-and-analysis-toolkit` for the actual measurement tools
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
| **Reliability king (primary)** | AOD's occupancy calls are more trustworthy than raw motion/naive automations across real, messy households — this is the maintainer's stated #1 priority | A multi-household measured comparison (precision/recall or false-trigger rate) of AOD vs. a documented naive-motion-sensor baseline, using the PR #486-standard method (§4). No such measurement exists in the repo today — this is the single most important gap; see `aod-learning-accuracy-campaign` |
| **Learned-model research** | Priors/likelihoods/adjacency learned from data, not hand-tuned, is a research contribution worth publishing/citing | Needs a written methodology (sampling, validation split, real-household data) that survives scrutiny — not just "we learn from history" as a slogan. The current adjacency tunables are explicitly first-pass/unvalidated (§3) — this ambition is blocked on the same accuracy-measurement gap as reliability king |
| **HA-core quality** | Code quality, test coverage, config-migration discipline on par with what HA core requires of built-in integrations | Concretely measurable today: 85% coverage gate (`pyproject.toml`, enforced by `scripts/test`), full type annotations, Google-style docstrings, ruff-enforced style (see `aod-build-and-env`) — this ambition is the closest to already being true and is the easiest to defend externally with a number (current coverage %, run `scripts/test` to reverify) |
| **Predictive occupancy** | Not just detecting current occupancy but predicting near-future occupancy/transitions (the adjacency trajectory work is a first step toward this) | Needs an explicit prediction-vs-outcome evaluation (e.g., "predicted transition to area X within N minutes, did it happen") — no such evaluation exists yet; adjacency's trajectory tracking computes a *boost signal*, not a scored prediction, so today's code is an ingredient, not a demonstrated predictive-occupancy result |

None of these four should appear in README/release notes as accomplished
facts. They are legitimate *direction* statements (roadmap language is fine:
"working toward reliability across real households") but any "we are the
most reliable/most predictive" framing needs the measurement described in
§4-5 first.

## Provenance and maintenance

Date-stamped 2026-07-06, integration version 2026.5.17 (`pyproject.toml`
line 7 / `manifest.json` line 20 / `const.py` `DEVICE_SW_VERSION`).
PRs #454 (adjacent-areas), #491, #492, #493, #494 were **open, CI-green,
unmerged** at time of writing — re-verify before citing any of their content
as shipped.

Re-verification commands, by volatile fact category:

- **Star count / repo metadata**: `gh repo view Hankanman/Area-Occupancy-Detection --json stargazerCount,description,homepageUrl`
- **Adjacent-areas / PR #454 merge state**: `gh pr view 454 --json state,mergeable,statusCheckRollup`
- **Same-day bugfix PRs #491-494 merge state**: `gh pr view 491 492 493 494 --json state,mergeable` (run each individually; `gh pr view` takes one number at a time)
- **Bermuda thread #25 latest status**: `gh issue view 25 --json state,comments,labels`
- **Adjacency tunables / "first-pass" disclaimer still present**: `grep -n "First-pass values" custom_components/area_occupancy/const.py`
- **PR #486 measured numbers (15,952/7,058/3,323 rows, −55%/−79%)**: `gh pr view 486 --json body`
- **HA core bayesian platform current behavior**: fetch `https://www.home-assistant.io/integrations/bayesian/` (external, may change independent of this repo)
- **Current release version / CalVer scheme**: `gh release list --limit 5`
- **Coverage gate percentage**: `grep -n "fail_under\|fail-under" pyproject.toml`; confirm live number via `scripts/test`
- **MIN_CORRELATION_SAMPLES threshold**: `grep -n "MIN_CORRELATION_SAMPLES" custom_components/area_occupancy/const.py`
