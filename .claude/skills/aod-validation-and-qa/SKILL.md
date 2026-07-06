---
name: aod-validation-and-qa
description: Use when writing, reviewing, or deciding what evidence is required for ANY change to Area Occupancy Detection — before opening a PR, when asked "is this tested enough?", when adding a test for math/behavior/config/DB code, when coverage or CI is failing, or when a reviewer/CodeRabbit flags a test as rigged, redundant, or encoding a bug. Covers the evidence bar per change class, conftest.py fixture anatomy, the golden test-file map, and lint/coverage gates.
---

# AOD Validation and QA

## What this covers

The evidence bar this project actually enforces before a change merges: what counts as proof for docs vs. behavior vs. math changes, how the test suite in `tests/` is organized (fixtures, markers, coverage), how to add a test for each architectural layer, and the map of which test file guards which invariant. It also names two real, cited anti-patterns from this repo's own PR history that look like passing tests but prove nothing — use them as red-flag comparisons when reviewing your own or Claude's test additions.

## When NOT to use this

- Deciding *whether* a change needs review/what process gate applies → `aod-change-control`.
- Understanding *why* a past bug happened and what invariant it taught us → `aod-failure-archaeology`.
- Root-causing a live/reported bug step by step → `aod-debugging-playbook`.
- Building or running the cross-check simulator / correlation tooling itself → `aod-proof-and-analysis-toolkit`.
- The Bayesian math formulas themselves (what to hand-compute against) → `bayesian-occupancy-reference`.

## The evidence bar, by change class

| Change class | Minimum evidence before merge | Notes |
|---|---|---|
| Docs only (`docs/`, README, docstrings) | CI green (`lint.yml` + `test.yml` unaffected/pass) | No new tests required. Verify links/build via `aod-docs-and-writing`. |
| Behavior change (config flow, sensors, coordinator wiring, health monitor, DB CRUD) | A unit test that exercises the **real code path** — the actual entry point a user/HA would hit — not a shortcut that manipulates internal state to force the assertion true | See anti-pattern #1 below. Must pass locally (`uv run pytest <file> -v`) and coverage gate must still pass. |
| Math change (anything touching `utils.py::bayesian_probability`, `data/prior.py`, `data/decay.py`, `data/analysis.py` prior/period logic, `data/adjacency.py`) | (1) Hand-computed expected number(s) written down **before** running the code, (2) cross-check via the simulator or a DB-backed test (real interval data, not synthetic zeros), (3) a regression test that encodes the **correct** behavior with the hand-computed number in the assertion | See anti-pattern #2 below. This is the highest-stakes class in the project — the maintainer's stated hardest problem is prior/likelihood learning accuracy on real homes. |

Never make a silent math change — any modification to a probability/decay/prior formula must be called out explicitly in the PR description with before/after numbers, per the project's unwritten law "no silent math changes" (see `aod-change-control`).

### Anti-pattern #1: the rigged re-add test (PR #488)

PR #488 (disable diagnostic sensors by default for new areas, merged) shipped a test asserting that re-adding a deleted area produces disabled-by-default diagnostics. The test passed — but only because it popped the entity's record out of `registry.deleted_entities` before re-running setup, which is exactly the internal mechanism (`EntityRegistry.async_remove` preserves `disabled_by` in `deleted_entities`, and `async_get_or_create` restores it on re-registration) that makes the real HA behavior the *opposite* of what the test asserted. A same-PR follow-up review caught this and rewrote the test to go through the real registry restore path; the corrected test asserts a re-added area **keeps its previous enabled/disabled state**, not the new default.

Lesson: if your test setup pokes at a private/internal collection (`_something`, a mock's internal registry, a manually-constructed object graph) specifically to make an assertion pass, ask "does production code ever reach this state through that path?" If not, the test is asserting a fiction. Prefer driving the real entry point (`entity_registry.async_get_or_create`, `hass.config_entries.async_setup`, an actual state write via `hass.states.async_set`) and asserting on its output.

### Anti-pattern #2: the test that encoded the bug (issue #483 / PR #491, merged 2026-07-06 — SETTLED)

Issue #483 reported `global_prior` pinned at the 0.99 hard cap on a real installation with a true ~28-35% occupancy rate. Root cause: `PriorAnalyzer.calculate_and_update_prior()` in `data/analysis.py` truncated the observation window to `last_interval_end` (dropping the quiet tail from the denominator) whenever the area had been quiet for over an hour — which fires on every ordinary overnight/weekend lull, silently inflating the prior on every hourly recalculation.

The existing test `test_valid_calculation_sets_correct_prior` in `tests/test_data_analysis.py` had **encoded the bug**: with 2h occupied time ending 6h ago inside an 8h window, it asserted `area.prior.global_prior == 0.99` — the buggy, capped output — as if that were correct. Nothing failed because the test's expectation was wrong, not the code. PR #491 (fixing #483) fixed both: `actual_period_end` is now always `now` on `main`, and the same test now asserts the hand-computable correct value `0.25` (2h / 8h), plus a new dedicated overnight-quiet-tail regression test. Issue #483 is closed; the fix has been on `main` since the 2026-07-06 merge wave, though it is not yet in a tagged release (still version 2026.5.17).

Lesson: a green test suite proves nothing if the assertions themselves are wrong. For any math change, compute the expected number by hand (or via the simulator / a known-good DB query) **before** you run the code, then make the test assert that number — never "whatever the code currently outputs."

## Test suite anatomy

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

Live in `tests/test_utils.py` (1234 lines) — **not** `test_calculate_prob.py`. CLAUDE.md's "Common Workflows → Modifying Bayesian Calculation" section says to update `tests/test_calculate_prob.py or similar`; the repo shows the actual file is `tests/test_utils.py`, class `TestBayesianProbability` (plus `TestCombinePriors`, `TestSigmoidFunctions`, `TestApplyActivityBoost`, `TestCombinedProbability`, `TestSigmoidVsBayesian`, `TestPresenceEnvironmentalSplit`, `TestMapBinaryStateToSemantic`). Treat CLAUDE.md's filename here as stale and use this table instead. (There is also a stale `.github/instructions/testing_requirements.instructions.md` that lists files like `test_calculate_prior.py`, `test_calculate_prob.py`, `test_storage.py`, `test_probabilities.py`, `test_types.py`, `test_ml_models.py` — **none of these exist in `tests/` today.** Do not trust that file's file list; it predates the current architecture. Use `ls tests/` as ground truth.)

## The golden test-file map

Each row is the file(s) that guard a specific invariant — if you touch the named production code, run (at minimum) the paired test file, and add to it rather than creating a new file for the same concern. Coverage percentages below were measured on `main` as of 2026-07-06 (HEAD `17b71d2`), post-merge of PR #454 (adjacent-areas) — the former branch-vs-main split no longer applies since the adjacency test files are now part of `main`'s own denominator.

| Production area | Guarding test file(s) | Invariant(s) enforced |
|---|---|---|
| `utils.py::bayesian_probability`, sigmoid/combine helpers | `test_utils.py` | Core Bayesian math: combining priors, sigmoid transforms, activity boost application, presence/environmental split. 93% file coverage as of 2026-07-06. |
| `data/decay.py` | `test_data_decay.py` (`TestDecay`, `TestDecayHalfLife`, `TestDecayModifierFactor`) | Decay curve correctness, invalid/very-large half-life handling, timezone-naive datetime handling (`test_timezone_naive_datetime_handling`), purpose-half-life compounding with adjacency decay-modifier factor via `_resolve_purpose_half_life()` (PR #493, merged 2026-07-06 — see `aod-failure-archaeology` for the #481 guard story). 100% file coverage. This file is the direct descendant of the costly historical "decay half-life config bug" — see `aod-failure-archaeology`. |
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
- **The former repo inconsistency is resolved:** the `fail_under = 85` line in `pyproject.toml` now carries the trailing comment `# Enforced global minimum; aim for 90%+ on core calculation modules (CLAUDE.md)`, which agrees with the enforced value (85) instead of contradicting it. Historically the comment read `# Enforce 90% coverage minimum` and disagreed with the number — that mismatch has been fixed as part of the 2026-07-06 merge wave; trust the number (85) either way.
- **Aspirational (not separately enforced by any tool):** CLAUDE.md states "85%+ coverage requirement (90% for core calculations)." There is no per-module coverage gate, `.coveragerc`, or codecov config anywhere in the repo enforcing a separate 90% threshold on `utils.py`, `data/prior.py`, `data/decay.py`, or `data/analysis.py` — verified by `grep -rn "90" pyproject.toml` (no separate coverage-config hit beyond the comment above) and no `codecov.yml`/`.coveragerc` file present. In practice the core-calculation files already run 90-100% (`data/decay.py` 100%, `data/prior.py` 99%, `utils.py` 93%, `data/analysis.py` 91%) — treat 90%+ on these specific files as a review expectation you self-enforce (check `--cov-report=term-missing` output for the file you touched), not something CI will catch for you if it slips.

## Lint gates

CI job `lint.yml` ("Ruff") runs on every PR to `main` — as of the 2026-07-06 merge wave (#495, CI hygiene) the lint/test/validate PR triggers were narrowed to `main`-only; the old `rc`/`dev` trigger targets are gone:
```bash
uv run ruff check .          # must exit 0 — no lint errors
uv run ruff format . --check # must exit 0 — no formatting diffs
```
Local equivalent (auto-fixes instead of just checking): `scripts/lint`, which runs `uv run ruff format .` then `uv run ruff check . --fix`. If pre-commit's ruff hook fails on commit, review its changes, `git add -u`, and commit again (per CLAUDE.md) — do not bypass with `--no-verify`.

`pyproject.toml` now pins `ruff==0.15.2` in dev dependencies, and `.pre-commit-config.yaml` pins the same `rev: v0.15.2` for the `ruff-pre-commit` hook — the triple-skew trap (dev deps vs. pre-commit vs. CI all drifting to different ruff versions) that used to require manual reconciliation is resolved as of the #496 dependency refresh (2026-07-06); all three now agree. `[tool.ruff.lint]` enables a large rule set (`ASYNC`, `B`, `BLE`, `C`, `D`, `PL`, `PT`, `PTH`, `RET`, `SIM`, `TRY`, `UP`, etc. — see `[tool.ruff.lint].select`) with a documented `ignore` list (e.g. `E501` line-too-long, `PLR2004` magic values, `PT011` broad `pytest.raises`). PEP 758's unparenthesized multi-except syntax (`except A, B:` without parens) is now house style and ruff-enforced as part of the same refresh. If ruff flags something you believe is a false positive for this codebase, don't silence it inline without checking whether it's already in the `ignore` list for a reason — read `pyproject.toml` first.

## What a PR must show before merge

Per this project's change-control discipline (full process detail owned by `aod-change-control`), from a QA-evidence standpoint a PR should show, in its description:
1. Which change class it is (docs/behavior/math) and the evidence for that class from the table above.
2. For math changes: the hand-computed expected number(s), stated in the PR body, *before* describing what the code now returns — so a reviewer can check your arithmetic independently of your test.
3. `scripts/test` output (or CI green) demonstrating the full suite still passes and coverage still clears 85%.
4. `scripts/lint` clean (or CI green on `lint.yml`).
5. For config-surface changes: an explicit backward-compatibility statement (existing config entries unaffected) — config surface is sacred per the maintainer's unwritten laws.
6. If a CodeRabbit/reviewer nitpick is deliberately skipped rather than fixed, a reply on the PR thread stating why (project convention — see the `feedback_coderabbit_skip_replies` memory note).

## Provenance and maintenance

Compiled 2026-07-06, re-verified 2026-07-06 post-merge-wave against integration version 2026.5.17 (`pyproject.toml` version field — still the last tagged release; none of the 2026-07-06 merges are in a tagged release yet) on `main` HEAD `17b71d2`.

Re-verification commands, by fact category:
- Test count / pass/fail / coverage %: `scripts/test` (or `uv run pytest --cov=custom_components/area_occupancy --cov-report=term-missing`)
- Coverage gate value: `grep -n "fail_under" pyproject.toml`
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
