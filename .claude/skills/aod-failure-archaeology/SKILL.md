---
name: aod-failure-archaeology
description: Use before touching decay half-life, prior/global-prior calculation, sensor-health repairs, sleep-presence detection, recorder/DB write volume, adjacent-areas, timezone/DST-sensitive datetime code, or the config-flow advanced-options gating — to check whether the bug you're about to fix (or the fix you're about to write) already happened before. Load this when a symptom rhymes with "custom value silently reverted to a default", "repairs fire spuriously / return after Ignore", "prior pinned near 0.99 or 0.01", "TypeError offset-naive and offset-aware", or "recorder database growing too fast".
---

# AOD Failure Archaeology

## What this covers

A chronicle of every major bug investigation in this repo that is settled, partially settled, or still open, recorded as symptom → root cause → evidence → status → lesson. The goal is that a future zero-context session (human or AI) checks this file *before* re-diagnosing a bug from scratch, and before proposing a fix that was already tried and found insufficient. Every entry below is verified directly against git history and the GitHub issue/PR tracker (see each entry's Evidence line); dates and PR numbers are exact as of 2026-07-06.

## When NOT to use this

- You want the *current*, authoritative semantics of decay/prior/health calculations (not their history) → `bayesian-occupancy-reference` (calculation) or read `data/decay.py`, `data/prior.py`, `data/health.py` directly.
- You're actively debugging a *new* symptom and need a systematic triage method, not a lookup of past sagas → `aod-debugging-playbook`.
- You want the campaign-level plan for the "priors/likelihoods accuracy on real homes" effort (the maintainer's stated hardest problem) rather than a single past bug → `aod-learning-accuracy-campaign`.
- You're deciding whether a fix is safe to ship (silent-math-change / config-break rules) → `aod-change-control`.

## Status legend

- **SETTLED** — merged to `main`, no known recurrence, root cause fully understood.
- **PARTIALLY-SETTLED** — merged fix helped, but the underlying bug class or a related issue remains open.
- **OPEN** — PR exists but is not yet merged to `main` as of 2026-07-06 (verify current state with `gh pr view <n>`), or no fix has been proposed yet.

Do not describe anything marked OPEN below as shipped. Re-check merge state before relying on it: `gh pr view <n> --json state,mergedAt,baseRefName`.

---

## SAGA 1 — Timezone / DST datetime bugs (maintainer-flagged costliest failure class)

**Symptom (round 1, Dec 2025):** Issue #301 "Invalid period duration errors in 2025.12.2" — prior calculation logged `Invalid period duration (-10800.00 seconds) for area Hallway ... Using safe fallback prior of 0.01`, repeatedly. Exactly −3h (10800s) is the giveaway: a timezone-offset-sized negative duration, not random clock skew.

**Root cause:** Datetime handling before PR #304 mixed naive and aware datetimes, and DST-window arithmetic used wall-clock comparisons that broke across the fall-back/spring-forward transition, producing an "end before start" period.

**Fix:** PR #304 "Implement timezone normalization and local bucketing utilities" (merged 2025-12-12) introduced `time_utils.py` and bumped `CONF_VERSION` to 16. PR #322 "Refactor timezone handling with UTC storage" (merged 2025-12-29) followed up with DST-aware time calculations and UTC-based storage.

**Current mechanism (verified, `custom_components/area_occupancy/time_utils.py:1-8`):** an explicit three-tier policy stated in the module docstring — runtime arithmetic uses timezone-aware UTC; SQLite persistence uses naive UTC (`tzinfo=None`, interpreted as UTC); wall-clock bucketing (time priors, daily/weekly grouping) uses HA's local timezone via `dt_util.as_local()`. Helpers: `to_utc()`, `to_db_utc()`, `from_db_utc()`, `to_local()`, `assert_utc_aware()` (debug-only invariant check).

**Symptom (round 2, April 2026 — the bug class recurred in a *different* module):** Issues #444/#445, "TypeError: can't subtract offset-naive and offset-aware datetimes in `_check_stuck_sensor`" — `data/health.py` line 257, ~87–116 failures over 21 hours on real installs. `now = dt_util.utcnow()` (aware) minus `entity.last_updated` (naive when restored from storage) raised `TypeError`, failing `sensor_health_check` on every analysis cycle.

**Fix:** PR #446 "Fix TypeError on naive vs aware datetime in health checks" (merged 2026-04-27) normalizes via `dt_util.as_utc(entity.last_updated)` in both `_check_stuck_sensor` and `_check_unavailable`, plus normalizes on entity restore from DB.

**Status: PARTIALLY-SETTLED.** The `time_utils.py` policy (SAGA 1 round 1) is the settled, load-bearing fix for the core prior/analysis pipeline. But round 2 shows the naive/aware mismatch class recurred in a module (`health.py`) that predates or bypassed the `time_utils.py` convention — meaning any *new* code that touches `entity.last_updated`, `entity.last_changed`, or any datetime pulled from HA state/registry objects must not assume it is timezone-aware.

**The lesson:** Never subtract or compare two datetimes without first normalizing both through `time_utils.to_utc()` (or `dt_util.as_utc()` for HA registry/state objects specifically) — HA's own state machine and the storage layer both silently hand you naive datetimes in some code paths. Grep for raw `datetime.now()`, `entity.last_updated`, or bare subtraction of two datetime variables in any new PR touching health checks, decay, or analysis; this bug class has bitten this repo twice in two different modules seven months apart. DST transitions specifically break *wall-clock window* comparisons (e.g. "is current local time within sleep_start–sleep_end") — see SAGA 3, which is a different but adjacent bug in the same neighborhood of code.

Evidence: issue #301 (closed 2025-12-29); PR #304 (merged 2025-12-12, `time_utils.py` introduced, CONF_VERSION→16); PR #322 (merged 2025-12-29); issues #444, #445 (closed 2026-04-27); PR #446 (merged 2026-04-27); `custom_components/area_occupancy/time_utils.py:1-72` (current, read directly).

---

## SAGA 2 — Sensor-health false-positive campaign (#429 → #455 → #459 → #466 → #472 → #473 → #474 → #485)

**Origin:** PR #429 "Add sensor health monitoring with HA repairs integration" (merged 2026-03-31) introduced `HealthMonitor`: per-sensor-type stuck-active/unavailable/never-triggered detection surfaced through HA's Repairs UI. Motivated by real production findings (a motion sensor stuck "on" 25h, appliance sensors that never triggered, sensors permanently unavailable). Initial thresholds: motion 2h active before flagging, door 48h, appliance 28 days inactive.

**Symptom wave 1 (issue #455, 2026-05-03):** "Repeatedly generates Repair notifications... disappear when I reload or click Ignore, return again after a few minutes." Log showed `New sensor health issues in area 'X': None (correlation_failures)` for every area simultaneously.

**Root cause (3 bugs in one issue):**
1. `translations/en.json` was missing the entire `issues` block, so every repair rendered as a title-only card with no guidance.
2. `pipeline_health_correlation_failures` fired during the warm-up grace period — soft "not enough data yet" states were treated as real failures.
3. `sensor_health_unavailable` measured duration from the persisted `entity.last_updated` (can be days old) instead of a live "since observed unavailable" clock — so a slow-loading source integration (Z2M, ESPHome) at HA startup instantly tripped every sensor past the 1h threshold.

**Fix:** PR #459 "fix(health): silence false-positive repairs from #455" (merged 2026-05-04, same day as opened) — mirrored the translations block, gated correlation-failure checks on `PRIORS_TRAINING_GRACE_PERIOD`, and added an in-memory `_unavailable_since` map keyed by entity_id instead of trusting `last_updated`.

**Abandoned parallel branch — `origin/hotfix-repairs-455`:** a stale branch, 8 commits ahead / 6 behind `main` (verified: `git rev-list --left-right --count origin/main...origin/hotfix-repairs-455`). Its first 3 commits (`180a93a`, `6dbe9ab`, `9c4972d`) closely parallel what shipped in #459. Its last 3 commits (`a0d4d86`, `d5cb762`, `2b0da16`, "anchor `db_aggregation` tests to local midnight instead of UTC midnight") were **dropped entirely** — #459 as merged (squashed as `368c03a`) does not contain that test-anchoring work. If you find this branch, do not assume it's a superset of `main`'s fix; it's a divergent draft, partially superseded.

**Symptom wave 2 (issues #463, #465, #466, #468, all opened 2026-05-05 to 2026-05-10):** users "overwhelmed by repair noise" — one reported "I wake up every morning to 40+ issues." Two concrete complaints: (a) a TV `media_player` going `unavailable` every power-off triggers a repair every morning; (b) bedroom mmWave motion sensors legitimately stay "on" for hours during sleep, tripping the 2h stuck-active threshold nightly. Also: dismissing ("Ignore") a repair didn't stick — it returned on the next condition recurrence.

**Fix (three independent, deliberately-decoupled PRs, all merged 2026-05-16/17):**
- PR #472 "add integration-level toggle to disable repair monitoring" — `CONF_HEALTH_ENABLED` global boolean (default **on**); when off, both sensor- and pipeline-scope checks short-circuit and `HealthMonitor.clear_all_issues()` empties the Repairs UI (distinct from `cleanup()`, which additionally wipes the `_unavailable_since` clock — `clear_all_issues()` deliberately preserves it so re-enabling doesn't cause an instant re-trip).
- PR #473 "preserve user-ignored repairs across condition flaps" — root cause was that `_update_repair_issues` called `ir.async_delete_issue()` on every condition-clear, which also wiped HA's `is_ignored` flag; the fix partitions resolved issues into truly-resolved (delete) vs. user-ignored (skip deletion, keep tracked so a recurrence isn't logged as "new").
- PR #474 "purpose-aware stuck-active thresholds and saner defaults" — motion default raised 2h → **8h** (base); per-purpose multiplier table `SLEEPING×6` (=48h), `RELAXING×4` (=32h), `WORKING×3` (=24h); `media_player.*` entity-id prefix exempted from the unavailable check entirely.

**Current thresholds:** see the canonical table in `aod-diagnostics-and-tooling` §2 (base thresholds, purpose multipliers, `media_player.*` exemption, `InputType.SLEEP` exclusion) — kept in one home so a future threshold change edits one file.

**Status: PARTIALLY-SETTLED.** All of #463/#465/#466/#468's reported symptoms are addressed. Two follow-on asks remain **OPEN**:
- **#466** — per-sensor suppression ("mark this specific entity as expected, never alert on it") — deliberately deferred; the PR #472/#474 authors state explicitly this is out of scope for the toggle-and-defaults approach.
- **#485** — vacation-aware alert suppression (a boolean indicating "long absence is expected here," so stuck/no-trigger checks don't fire) — open, no PR yet, reporter notes the irony that during vacation, a sensor *triggering* is more suspicious than one staying silent.

**The lesson:** Health-check false positives are a *defaults and UX* problem more than a *detection-logic* problem — three of the four shipped fixes (#459's unavailable clock, #472's toggle, #473's sticky-ignore, #474's thresholds) all worked around the same underlying tension: a single global timeout can't be right for every sensor type, purpose, and household schedule simultaneously. When you get a new "false positive repair" report, first check whether it fits an existing per-purpose/per-type carve-out before adding a new global threshold — and check #466/#485 before designing a new per-sensor mechanism, since both are explicitly requested and unclaimed.

Evidence: PR #429 (merged 2026-03-31); issue #455 (closed 2026-05-04); PR #459 (merged 2026-05-04, commit `368c03a`); `origin/hotfix-repairs-455` branch (`git log origin/main..origin/hotfix-repairs-455`, 6 commits, last 3 dropped); issues #463, #465, #466 (open), #468; PRs #472, #473, #474 (all merged 2026-05-16/17); `custom_components/area_occupancy/data/health.py:40-80` (current, read directly); issue #485 (open).

---

## SAGA 3 — Decay half-life "custom value silently overridden by default" (two incidents, same bug class, different code path)

**Incident 1 — issue #439 (2026-04-17), reported via discussion #433:** custom decay half-life appeared to save in the options flow but reverted to the previous value on reopen.

**Root cause:** `Purpose.is_purpose_half_life(value)` returned `True` whenever the entered value matched *any* purpose's built-in default (12 round values: 45, 60, 90, 180, 240, 360, 450, 480, 520, 600, 620, 1200 seconds) — not just the *currently selected* purpose's default. So e.g. a Living Room user (purpose Social, default 520s) entering `600` (which happens to equal the Office purpose's default) got silently normalized to `0` ("use purpose default"), discarding their intended custom value.

**Fix:** PR #440 (merged same day, 2026-04-17) scopes the comparison to only the selected purpose's own default: `is_purpose_half_life(value, purpose=None)`.

**Status: SETTLED** for incident 1's exact code path.

**Incident 2 — issue #481 (2026-05-20), reported ~1 month later:** "Custom Decay Half-Life is ignored when purpose is Bedroom" — a user set purpose=Bedroom, custom half-life=10s; the room took ~15 minutes (900s+) to clear instead of ~10s, but only *outside* the configured sleep window. Changing purpose away from Bedroom fixed it immediately.

**Root cause:** `Decay._resolve_purpose_half_life()` (in `data/decay.py`) implements a sleep/awake split unique to `AreaPurpose.SLEEPING`: inside the sleep window it correctly uses the area's configured half-life; outside it, the code **unconditionally** returns `self._purpose.awake_half_life` (620s), discarding whatever custom half-life the user set — the same "custom-vs-default" semantics bug as #439, recurring in a sibling code path that #440's fix never touched.

**Fix — PR #493 "respect custom half-life for Bedroom purpose" (merged 2026-07-06, commit `55a0aae`):** adds the guard `if self._base_half_life != self._purpose.half_life: return self._base_half_life` — any half-life differing from the purpose default is treated as a deliberate user override and the awake/asleep switch is skipped entirely; the switch only engages when the half-life equals the purpose default (i.e., the user left it on auto), explicitly modeled on #440's fix pattern. Verified directly on `main` (`custom_components/area_occupancy/data/decay.py:90-91`): the guard is a single added `if` before the sleep-window check, exactly as the PR diff showed pre-merge.

**Status: SETTLED** (merged 2026-07-06; guard confirmed present on `main`).

**The lesson:** "Custom value vs. purpose default" is a recurring semantic distinction in this codebase that has now broken twice in two different functions seven months apart, both times because a piece of code treated "value equals a known default" as proof of "value was never customized," instead of tracking customization as its own explicit signal. Any new purpose-aware or type-aware default-switching logic (decay, thresholds, weights) should be checked against this exact failure mode before shipping: does the switch key off "does this look like a default" (fragile, re-breaks) or off "did the user actually configure something else" (correct)? #440 and #493 both had to retrofit the latter after finding the former shipped first.

Evidence: issue #439 (closed 2026-04-17); PR #440 (merged 2026-04-17); issue #481 (closed 2026-07-06); PR #493 (merged 2026-07-06, commit `55a0aae`, body: "mirrors the custom-vs-default semantics established for #440"); `custom_components/area_occupancy/data/decay.py:81-91` on `main` (guard now present, read directly).

---

## SAGA 4 — Global prior accuracy: the quiet-tail denominator bug, and the test that encoded it (#483 → #491)

**Symptom (issue #483, 2026-05-29):** A kitchen area with a single mmWave sensor had `global_prior` pinned at the hard cap **0.99**, despite a true occupancy rate of ~28–35% measured over 7 days of recorder history.

**Root cause (found by the reporter, @mscharwere, and confirmed by direct code read):** `PriorAnalyzer.calculate_and_update_prior()` in `data/analysis.py` contained:
```python
if (now - last_interval_end).total_seconds() > 3600:
    actual_period_end = last_interval_end   # drops the quiet tail from the denominator
else:
    actual_period_end = now
```
The comment's intent ("use it if very recent") was meant to guard against a near-zero denominator right after startup, but the condition actually fires **every time the area has been quiet for more than an hour** — overnight, every weekend, any extended absence. Effect: `global_prior = occupied_duration / (period − quiet_tail)` instead of `occupied_duration / period`, and because the prior recalculates hourly, every quiet stretch re-inflates it, ratcheting the prior toward the 0.99 cap with no correcting force.

**The test that encoded the bug:** `tests/test_data_analysis.py::TestPriorAnalyzerCalculateAndUpdatePrior::test_valid_calculation_sets_correct_prior` asserted, for a scenario of "2h occupied, ending 6h ago" (an 8h total window): `assert area.prior.global_prior == 0.99`. The correct value for that scenario is `2h / 8h = 0.25`. The test wasn't testing correctness — it was pinned to the bug's own output, so the bug shipped, and any refactor that preserved the buggy behavior would keep passing CI. This is a load-bearing lesson for `aod-validation-and-qa` and for anyone reviewing prior-calculation PRs: **a green test suite does not mean the math is right if a test's expected value was itself derived from running the buggy code.**

**Fix — PR #491 "keep quiet tail in global prior denominator" (merged 2026-07-06, commit `3f0895a`):** `actual_period_end` is now unconditionally `now`; the pre-existing `actual_period_duration <= 0` guard already covers the degenerate-denominator case the old conditional was trying to protect. The corrected test now asserts `0.25`, and a new regression test `test_quiet_tail_included_in_denominator` was added for the overnight-quiet scenario specifically. Verified directly on `main` (`custom_components/area_occupancy/data/analysis.py:515-518`): `actual_period_end = now`, unconditional, exactly as the pre-merge PR diff showed.

**Status: SETTLED** (merged 2026-07-06; confirmed live on `main`). Note this bug directly matched the maintainer's stated "costliest past failure": prior pinned at 0.99.

**Context — priors have been reworked repeatedly:** CodeRabbit's auto-linked related-PR history on #491 surfaces PRs #219, #246, #251, #266, #356 as prior related prior-calculation rework — this is at least the 6th significant touch to prior calculation across the project's life. Treat prior/likelihood code as one of the highest-scrutiny surfaces in the repo (see `aod-learning-accuracy-campaign` for the current state of this effort and `aod-change-control`'s "no silent math changes" rule).

**The lesson:** When fixing prior/likelihood math, (1) recompute the expected test value by hand from the stated scenario before trusting an existing test's assertion, especially for edge-case tests around cap/floor values (0.01, 0.99) — a test asserting a boundary value is exactly where a "the bug is the spec" trap hides; (2) any conditional that special-cases "denominator would be small/degenerate" needs its trigger condition audited against *all* the real-world situations that could set it off, not just the one motivating case in the original commit message.

Evidence: issue #483 (closed 2026-07-06, root-cause credit to @mscharwere); PR #491 (merged 2026-07-06, commit `3f0895a`, body cites #483, "Credit to @mscharwere for the precise root-cause analysis"); `custom_components/area_occupancy/data/analysis.py:515-538` on `main` (read directly, confirms the conditional is gone and the corrected test assertion 0.99→0.25 landed).

---

## SAGA 5 — Recorder/DB write-load campaign (#467 → #486 + #488)

**Symptom (issue #467, 2026-05-10):** A user on a 7-day recorder rollover with ~770 recorded entities found AOD responsible for over 30% of total recorder rows (1.5M of ~5M states) — diagnostic sensors updating on the 10-second decay timer with 2-decimal-place states meant nearly every tick wrote a new recorder row per sensor.

**Measured numbers (PR #486, live 6-area install, v2026.5.17, 57 AOD entities — these are the exact figures to cite, do not round differently):**

| Window (3h) | Precision | Recorder rows | Δ vs baseline |
|---|---|---|---|
| Afternoon (active), baseline | 2 decimals | 15,952 | — |
| Evening 21:30–00:30 (going to bed) | 0 decimals | 7,058 | **−55%** |
| Morning (low activity) | 0 decimals | 3,323 | **−79%** |

**Fix 1 — PR #486 "configurable sensor state precision to reduce recorder load" (merged 2026-07-06):** adds a global "Sensor state precision" setting (0–2 decimals, default **2 = unchanged behavior**, so existing installs see no change unless they opt in). At 0 decimals the recorder only writes on whole-percent changes. Deliberately does **not** use HA's `suggested_display_precision` — that only rounds the UI display, not the recorded state, so it wouldn't reduce recorder rows. All decision logic (`area.occupied()`, wasp-in-box, activity scoring, thresholds) continues to operate on internal unrounded floats — this PR touches only publication-layer formatting (`format_float`), not calculation code.

**Fix 2 — PR #488 "disable diagnostic sensors by default for newly added areas" (merged 2026-07-06):** registers 7 diagnostic sensor classes (`PriorsSensor`, `EvidenceSensor`, `DecaySensor`, `PresenceProbabilitySensor`, `EnvironmentalConfidenceSensor`, `ActivityConfidenceSensor`, `SensorHealthSensor`) with `entity_registry_enabled_default = False`. The two primary entities (`ProbabilitySensor`, `DetectedActivitySensor`) and the `Occupancy Status` binary sensor stay enabled. This only affects **newly registered** entities.

**Entity-registry restore subtlety (verified via PR #488's own review comment, self-reviewed with Claude Code assistance):** `entity_registry_enabled_default` only applies at first registration. HA's `async_get_or_create()`, when called against an *already-existing* registry entry, routes to `_async_update_entity()` — which has no `disabled_by` parameter and structurally cannot touch it. This is *why* existing installs are unaffected by this change: not a special-cased migration guard, but a structural property of the HA entity registry API. **Documented edge case: this protection does not extend to delete-and-re-add.** Deleting an area and re-adding it counts as a fresh registration (a new registry entry), so it comes back with diagnostics disabled — the "restore previous state" behavior only applies across reload/upgrade of a *still-registered* entry, never across a full delete-then-recreate cycle. If you are ever debugging "why did my diagnostic sensor's enabled state reset," check whether the area was deleted and re-added (resets) versus just reloaded/upgraded (preserves) before assuming a regression.

**Status: PARTIALLY-SETTLED.** Both PRs landed together (2026-07-06) and are complementary (#486 shrinks rows for *enabled* diagnostics; #488 removes them entirely for new setups) but issue #467 itself remains **OPEN** — the reporter's request was a single global "significant figures" config knob plus attribute pruning/sorting, and neither PR fully closes that ask; they're presented as "primary drivers" addressed, not a complete fix.

**The lesson:** When reasoning about whether a "disabled by default for new X" change is backward compatible, the correct verification is reading the actual HA core function your code calls (`async_get_or_create` → `_async_update_entity` when already registered) rather than asserting compatibility from intent alone — and the one honest gap (delete+re-add doesn't preserve state) should be stated explicitly in the PR, not glossed over, exactly as PR #488 did.

Evidence: issue #467 (open, 2026-05-10); PR #486 (merged 2026-07-06, body contains the measured-numbers table verbatim); PR #488 (merged 2026-07-06, body: "documented & tested" edge case section, quote "async_get_or_create on an existing entity routes to _async_update_entity, which has no disabled_by parameter and structurally cannot touch it").

---

## SAGA 6 — Sleep-presence detection: multi-sensor support, then unknown-presence gating (#375 → #464 → #492)

**Foundation — PR #375 "Add multi-sensor sleep detection support" (merged 2026-02-22):** support for multiple sleep sensors per person, across both `sensor` and `binary_sensor` domains, OR-combined ("any active sensor triggers sleep detection"). Config-version bump 16→17 with backward-compatible migration.

**Symptom (issue #464, 2026-05-07):** A binary sleep sensor (e.g. an `input_boolean`-backed template) reported `state: 'on', active: true` in the sensor's own attribute breakdown, but the top-level `sleeping` field was `false` and occupancy was not detected as a result.

**Root cause (self-diagnosed by reporter @laszlojakab, pinpointing `binary_sensor.py:859-865`):** `_evaluate_sleep_state()` gated on the person entity being home (`home_state.state != STATE_HOME` → treated as away, sleep never checked). A person entity with **no device tracker assigned** reports state `unknown` — and the equality check above treated `unknown` identically to a definitive "away," silently disabling sleep detection for anyone without a device tracker, even though their sleep sensor was correctly reporting active.

**Caution — a plausible-looking but wrong diagnosis was floated for this same issue:** an earlier read of the code (checking commit `368c03a`, unchanged since PR #375) found that a device-tracker/person-entity fallback *already existed* in that exact function, contradicting a claim that the fallback was "missing." The real bug is specifically the *unknown-state handling*, not an absent fallback path — if you're investigating #464-shaped reports, verify against the current code's handling of the `unknown` state specifically, not just "is there a fallback at all."

**Fix — PR #492 "detect sleep when person presence is unknown" (merged 2026-07-06, commit `2099025`):** adds a `_person_home_state()` helper returning `True`/`False`/`None` (tri-state: `None` = indeterminate — `unknown`/`unavailable`/missing entity). `_evaluate_sleep_state()` now skips a person only when **definitively away**; when presence is unknown, sleep sensors are trusted directly. 13 new tests added (the sensor had zero prior test coverage).

**Status: SETTLED** (merged 2026-07-06).

**The lesson:** Home Assistant person/device-tracker entities have (at least) three meaningfully different states — home, away, and unknown — and code written as a two-way boolean check (`== STATE_HOME` or `!= STATE_HOME`) will always silently collapse "unknown" into whichever branch it wasn't explicitly testing for. Any gating logic keyed on a person/tracker entity's state should be audited for this same tri-state collapse before shipping.

Evidence: PR #375 (merged 2026-02-22, CONF_VERSION 16→17); issue #464 (closed 2026-07-06, reporter's own attribute dump showing `active: true` / `sleeping: false`); PR #492 (merged 2026-07-06, commit `2099025`, body: "Fixes #464 ... exactly as @laszlojakab traced in the issue"); `git show 368c03a9:custom_components/area_occupancy/binary_sensor.py` (fallback-already-present check, read directly at the commit cited by a since-superseded diagnosis).

---

## SAGA 7 — `show_advanced_options` deprecation, and a false-blocker verification episode (#487 → #489)

**Symptom (issue #487, 2026-06-09):** HA system log warning on every startup: `The deprecated function show_advanced_options was called from area_occupancy. It will be removed in HA Core 2027.6.`

**Fix — PR #489 (contributor Ecronika; approved and merged by the maintainer 2026-07-06, commit `704c89e`):** removed the `show_advanced: bool = False` parameter from five config-flow schema builders and un-gated the four `if show_advanced:` blocks entirely — because HA's deprecated property already unconditionally returns `True` during its deprecation window, un-gating is a behavior-preserving deletion, not a UX change. Diff limited to `config_flow.py` (−25 lines). Verified live on a 6-area install: warning stopped appearing, advanced fields still shown identically.

**The false-blocker episode (verify current claims with `gh pr view 489 --json reviews`):** PR #489's body and the maintainer's approving review cite a specific source for the deprecation: *"Deprecation of advanced mode in data entry flow,"* developers.home-assistant.io, dated **2026-05-26**. At the time of this review, this repo's `pyproject.toml` pinned `homeassistant==2026.2.2` for its own test suite — a version that predates the cited blog post by three months (that pin has since moved to `homeassistant==2026.7.1` via the same-day dependency refresh, PR #496 — reverify with `grep -n '"homeassistant==' pyproject.toml`). A reviewer or verification pass that reasons "the pinned test dependency doesn't know about this deprecation yet, therefore the citation must be fabricated" **would be wrong**: pinned test dependencies in an actively-developed integration are routinely older than the *production* HA version users are running the integration against, and upstream deprecation announcements apply to the ecosystem going forward, not retroactively to whatever version a repo's CI happens to test on. The correct verification is checking the cited blog post's existence/date and the actual runtime behavior (which PR #489 did — "verified live... deprecation warning no longer appears"), not comparing it against an unrelated pinned test-dependency version.

**Status: SETTLED** (merged 2026-07-06).

**The lesson:** when verifying a claim that cites an external (non-repo) source — a vendor blog post, an upstream deprecation notice, a changelog entry — check the source directly (fetch it, confirm the date and content) rather than using an unrelated repo-internal version pin as a proxy for "could this be true yet." A stale test-dependency pin is extremely common in real projects and is not evidence that a forward-looking claim about the pinned dependency's *future* is false.

Evidence: issue #487 (closed 2026-07-06); PR #489 (merged 2026-07-06, commit `704c89e`; maintainer review body quotes the exact blog post title/date/URL); `pyproject.toml` pinned `homeassistant==2026.2.2` at review time (since bumped to `2026.7.1` at line 25 by PR #496, same-day merge — the pin cited during this episode is no longer what's on `main`).

---

## SAGA 8 — Adjacent-areas: a feature dormant since 2025, then built in five phases, merged 2026-07-06

**Dormant schema (verified: `git show a99ad49:custom_components/area_occupancy/db/schema.py`, commit dated 2025-11-17):** the `Areas.adjacent_areas` JSON column, `AreaRelationships` table, and `CrossAreaStats` table have existed in the schema since November 2025 — over five months before anything used them. No config-flow UI, no producer, no consumer existed until 2026.

**Origin of the ask — discussion #431 (opened 2026-04-11 by jeroen-zzx, still unanswered/unclosed, 0 comments):** requested a "next door room" option (e.g., bedroom → hall) with two hand-tunable confidence parameters ("no motion next door" raises confidence; "motion next door" lowers it, since it could be someone else).

**Implementation — PR #454 "feat: adjacent-areas (next-door room influence)" (created 2026-05-03; merged 2026-07-06, commit `17b71d2`, now `main` HEAD), a 5-phase build:**
1. Plumbing — `CONF_ADJACENT_AREAS` constant.
2. Config flow + persistence — per-area multi-select UI; symmetric write at the persistence layer (if A lists B as adjacent, B is automatically updated to list A); deliberately NO `CONF_VERSION` bump — the new `adjacent_areas` column and `AreaTransitions` table are purely additive, created via `Base.metadata.create_all(checkfirst=True)` on startup (see `aod-config-and-flags` and `aod-change-control` for this precedent).
3. Transition learning — new `AreaTransitions` table recording 1-hop and 2-hop room-to-room transitions, bucketed by hour-of-week; a 6-level smoothing fallback (`lookup_transition_probability`) walks from most-specific (2-hop, hour-of-week, min. 5 observations) down to a static default (`DEFAULT_INFLUENCE_WEIGHTS["adjacent"] = 0.3`) as data sparsity increases.
4. Bayesian wiring (**folded in from a separate stacked PR, #456**, merged 2026-07-06 into the `feat/adjacent-areas` branch, which itself merged to `main` the same day via #454) — a logit-additive boost in `Area.probability()` (`apply_logit_boost`, confirmed wired into `area/area.py` on `main`) plus a decay-half-life stretch modifier (capped at 1.75×, `Decay.set_modifier_factor`/`compute_decay_modifier` in `coordinator.py`) for areas whose adjacent exits have been quiet.
5. Tests + docs top-up.

**Design decision explicitly locked in (per the PR body), directly answering discussion #431's ask differently than requested:** influence is **learned** from observed transitions (Phase 3), not a hand-tuned static confidence parameter as #431's author suggested. This is a deliberate design choice worth knowing if anyone later asks "why isn't there a simple slider for this" — there is a per-pair influence weight, but the directional strength itself comes from learned transition data, not user-set numbers.

**CodeRabbit review nitpicks — still present on `main` post-merge (verified via `gh api .../pulls/454/reviews` and a direct read of the current file):** (1) `db/relationships.py`'s module docstring still describes the Bayesian/decay consumers as "still uncalled — they're the Phase 4 work" — false as of this merge, since Phase 4 (`apply_logit_boost`, `compute_decay_modifier`) is now wired in and called from `area/area.py` and `coordinator.py`; the docstring was never updated when Phase 4 landed, so it's now a stale doc-comment trap for anyone reading that file in isolation; (2) no regression test existed for `CONF_ADJACENT_AREAS` being a malformed non-list value at the time of review — addressed by `_normalize_adjacent_areas` defensive coercion (`config_flow.py:1614`, confirmed present on `main`).

**Tunables are explicitly unvalidated (verified, `const.py:189-221` on `main`, comment: "First-pass values; tune from real data once Phase 3 is collecting transitions"):** `ADJACENCY_TRANSITION_WINDOW_S=60`, `ADJACENCY_RECENCY_HALF_LIFE_DAYS=30`, `ADJACENCY_TRAJECTORY_WINDOW_S=300`, `ADJACENCY_BOOST_GAIN=0.5`, `ADJACENCY_DECAY_MODIFIER_GAIN=0.75`, `ADJACENCY_DECAY_MODIFIER_MAX=1.75`, and four minimum-observation smoothing thresholds (5/20/50/20). No commit or test exercises these against real recorder data — only synthetic/mocked entities in tests. This remains true after the merge: the feature landing on `main` did not include a real-home validation pass, so treat these constants as an unvalidated candidate default, not a tuned one.

**Status: PARTIALLY-SETTLED.** PR #454 merged to `main` 2026-07-06 (commit `17b71d2`, now `main` HEAD) — the feature (config flow, persistence, transition learning, Bayesian wiring, tests/docs) is complete and live on `main`. It has **not** yet reached a tagged release: the integration's released version is still `2026.5.17`, so this is "shipped to `main`," not "shipped to users," until the next release cut. It also remains **unvalidated on real homes** — the tunables above are first-pass values exercised only against synthetic/mocked test data — so treat adjacent-areas as a functional-but-unproven feature, not a fully settled one, until real-world tuning data comes in.

**The lesson:** this was the single largest long-dormant-then-merged feature in the repo's history (unmerged for five months of active build-out before landing 2026-07-06) and a case study in "features can sit dormant in the schema for months before anyone builds the rest" — if you're investigating what looks like dead/unused schema (columns or tables with no readers), check open PRs and discussions before assuming it's abandoned; it may be a future feature's foundation laid early. Also: when a feature request suggests a specific mechanism (#431's hand-tuned confidence sliders), the shipped implementation is free to choose a different mechanism (learned influence) that satisfies the underlying need — document that divergence explicitly in the PR, as #454 did, so the original requester's mental model doesn't silently mismatch what shipped.

Evidence: `git show a99ad49:custom_components/area_occupancy/db/schema.py` (schema fields present 2025-11-17, read directly); discussion #431 (open, 0 comments, verified via `gh api repos/.../discussions/431`); PR #454 (merged 2026-07-06T16:50:40Z into `main`, commit `17b71d2`, full 5-phase body read directly); PR #456 (state MERGED, `mergedAt: 2026-07-06T10:01:29Z`, base branch `feat/adjacent-areas` — merged into the feature branch, which then merged to `main` same day via #454); `custom_components/area_occupancy/const.py:189-221` on `main` (tunables, read directly); `custom_components/area_occupancy/db/schema.py` on `main` (`AreaTransitions` table present, 15 tables total, read directly); PR #454 review comments (`gh api repos/Hankanman/Area-Occupancy-Detection/pulls/454/reviews`, CodeRabbit nitpicks on `db/relationships.py` docstring and `_normalize_adjacent_areas` docstring accuracy).

---

## Honorable mention — recurring DB-cleanup-on-deletion bug class (three rounds, ~1 month apart)

Not one of the required sagas above, but a clear "don't re-fight this" pattern: **#390 → #405 (merged 2026-03-04)** fixed orphaned records left in multiple tables by `delete_area_data`. **#421 → #423 (merged 2026-03-21)** then found a *different* deletion pathway uncovered by #405: areas removed from HA's own area registry first (rather than through AOD's own UI) left orphaned AOD-side data. **#436 → #438 (merged 2026-04-17)**, paired with **#451** (the "Reset Learning" button, merged 2026-05-03), found a third gap: entry/area removal wasn't purging learned history (priors, correlations, intervals, aggregates). Each round found a genuinely different deletion pathway the previous fix hadn't covered, as new DB tables were added elsewhere in the codebase over time. **The lesson:** there is no single "cascade delete" test that covers this class — any new DB table needs its own explicit cleanup-on-area-deletion test, covering deletion through AOD's own config flow, deletion via HA's area registry, and deletion via config-entry removal, because historically these three pathways have each broken independently.

Evidence: issue #390 (closed 2026-03-04), PR #405 (merged); issue #421 (closed 2026-03-21), PR #423 (merged); issue #436 (closed 2026-04-17), PR #438 (merged); PR #451 (merged 2026-05-03) — all titles/dates verified via `gh pr view`/`gh issue view`.

## No reverted commits found

`git log --all --grep="^Revert"` and a case-insensitive `revert` grep across all branches/refs return zero true `git revert` commits in this repository's history (checked 2026-07-06). Every saga above was fixed forward with new corrective commits/PRs, never rolled back. If you're looking for "what did we try and undo," the answer is: nothing, by that mechanism — look for superseding PRs instead (e.g. SAGA 3's #440 superseded by nothing, but extended by #493; SAGA 4's #491 is a straight fix, not a revert).

Evidence: `git log --all --oneline --grep="^Revert"` (empty); `git log --all --oneline | grep -i revert` (only one false-positive match, a fixture-cleanup commit `4694529` unrelated to code revert).

---

## Provenance and maintenance

Re-verified 2026-07-06 (post-merge-wave sweep) against integration version 2026.5.17, `main` branch, HEAD `17b71d2`. The merge wave (PRs #486, #488, #489, #491, #492, #493, #454/#456, #494, #495, #496) landed on `main` the same day this skill was originally compiled; `feat/adjacent-areas` is now fully merged and the working tree is on `main`. Every claim above was re-verified directly against the current `main` working tree (`git show origin/main:<path>`, `gh pr view --json baseRefName,mergedAt,state`, `gh issue view --json state,closedAt`), not carried over from the pre-merge draft.

Re-verification commands, one per volatile fact category in this skill:

- **PR/issue merge state (any SAGA marked OPEN):** `gh pr view <number> --json state,mergedAt,baseRefName`
- **Current decay half-life logic:** `git show origin/main:custom_components/area_occupancy/data/decay.py | sed -n '55,92p'`
- **Current prior calculation logic:** `git show origin/main:custom_components/area_occupancy/data/analysis.py | grep -n "actual_period_end" -A3 -B3`
- **Current health-check thresholds:** `git show origin/main:custom_components/area_occupancy/data/health.py | grep -n "STUCK_ACTIVE_THRESHOLDS\|_PURPOSE_STUCK_ACTIVE_MULTIPLIER\|_UNAVAILABLE_EXEMPT_PREFIXES" -A8`
- **Timezone policy module:** `git show origin/main:custom_components/area_occupancy/time_utils.py | head -10`
- **Adjacent-areas merge status and tunables:** `gh pr view 454 --json state,mergedAt` (now returns `MERGED`, 2026-07-06); `git show origin/main:custom_components/area_occupancy/const.py | grep -n ADJACENCY_`
- **Recorder-load measured numbers:** `gh pr view 486 --json body -q .body` (table is in the PR body verbatim)
- **No-revert-commits claim:** `git log --all --oneline --grep="^Revert"`
- **Config version (affects any migration-adjacent saga):** `git show origin/main:custom_components/area_occupancy/const.py | grep -n "CONF_VERSION"`
