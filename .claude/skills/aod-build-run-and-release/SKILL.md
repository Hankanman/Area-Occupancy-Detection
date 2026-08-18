---
name: aod-build-run-and-release
description: Use when setting up, resetting, or debugging the local development environment for Area Occupancy Detection — bootstrap fails, `uv sync` errors, wrong Python version, ruff or pytest behaving differently locally vs CI, uv.lock showing unexpected diffs, devcontainer questions, or anything mentioning scripts/bootstrap, scripts/setup, scripts/lint, scripts/test, libturbojpeg, or "three uv projects" — and when you need to actually run the integration or its satellite tools — launch the devcontainer's Home Assistant instance, add an area through the config-flow UI, enable debug logging, find the SQLite DB, read or interpret the hourly analysis pipeline's log lines, run the Flask simulator locally, or understand the release/HACS/docs-deploy machinery. Triggers on "how do I run this", "start HA", "add an area", "where's the database", "Step N FAILED", "sync_states", "run the simulator", "cut a release", "HACS", or "docs site".
---

# AOD Build, Run, and Release

## What this covers

How to recreate this repo's dev environment from a clean clone (three independent `uv` projects, devcontainer, bootstrap script sequence), the differences between local `scripts/lint`/`scripts/test` and their CI counterparts, and the specific environment traps that have wasted time here before: Python-version skew between CI and local, ruff triple-version skew, `pytest-homeassistant-custom-component` quirks, the test-only DB-init env var, and uv.lock churn. It also covers command anatomy and artifact conventions for actually operating this project: starting the devcontainer's Home Assistant instance and adding an area through the UI, the debug-logging recipe, where the SQLite database lives (real install vs. tests), the hourly analysis pipeline as an operational process (steps, log lines, failure/backoff), recorder-sync watermark mechanics, the Flask simulator, and the release / HACS / docs-deploy machinery.

## When NOT to use this

- Writing or fixing tests / coverage strategy beyond "why does pytest behave oddly" → `aod-change-and-validation`
- Runtime debugging of the running integration (HA logs, coordinator behavior) → `aod-debugging-and-history`
- Debugging *why a probability number is wrong* (math, priors, likelihoods, decay) → `aod-debugging-and-history` and `aod-math-reference`. This skill gets you to a running system and tells you what the logs mean structurally; it does not interpret Bayesian output.
- Diagnostic tooling/scripts for analyzing data — including diagnostics.json field-by-field, `db/maintenance.py` integrity checks, `scripts/visualize_distributions.py` → `aod-diagnostics-and-tooling`
- Release/branching/version-bump *policy* (what may change, gates, the three-file bump rule, CalVer scheme) → `aod-change-and-validation`. This skill only covers the mechanical commands to cut a release.

---

## Environment Setup

### Bootstrap sequence (copy-pasteable)

```bash
# First-time clone setup — just calls bootstrap, then prints readiness message
scripts/setup

# Equivalent to running scripts/bootstrap directly. Idempotent — safe to re-run.
scripts/bootstrap
```

`scripts/bootstrap` does, in this exact order (verified against the script source, `scripts/bootstrap`):

1. **Install `uv`** if not already on PATH (via `curl -LsSf https://astral.sh/uv/install.sh | sh`), falling back to checking `$HOME/.local/bin` and `/usr/local/bin`.
2. **Install `libturbojpeg`**, a system (non-Python) dependency required by Home Assistant's camera component (`PyTurboJPEG`, listed in `pyproject.toml` dependencies). Debian/Ubuntu: `apt-get install -y libturbojpeg0`. Fedora/RHEL: `dnf install -y turbojpeg`. Anything else: warns and continues (camera snapshot support degraded, not fatal).
3. **Create the venv pinned to Python 3.14**: `uv venv --python 3.14` (only if `.venv` doesn't already exist).
4. **Sync root project deps**: `uv sync --extra dev --extra test --extra viz`.
5. **Sync simulator deps**, in its own directory, explicitly unsetting `VIRTUAL_ENV` first and pointing at the root venv's interpreter: `(cd simulator && unset VIRTUAL_ENV && uv sync --python ../.venv/bin/python)`.
6. **Sync docs deps**, same pattern: `(cd docs && unset VIRTUAL_ENV && uv sync --python ../.venv/bin/python)`.
7. **Install pre-commit hooks**: `uv run pre-commit install`.

Why `unset VIRTUAL_ENV` before the simulator/docs syncs: after step 4, the shell (or a prior `source .venv/bin/activate`) may have `VIRTUAL_ENV` pointing at the root venv, which would make `uv sync` inside `simulator/`/`docs/` try to reuse the wrong project's environment. Unsetting it forces `uv` to build each subproject's own `.venv` (confirmed present on disk: `simulator/.venv/pyvenv.cfg` and `docs/.venv/pyvenv.cfg` each independently report `version_info = 3.14.3`, sourced from the same Python 3.14 interpreter resolved for the root venv, but as physically separate venvs).

### Three separate uv projects — not a workspace

This is not a `uv` workspace (`git grep -n workspace pyproject.toml docs/pyproject.toml simulator/pyproject.toml` → zero hits; no `[tool.uv.workspace]` anywhere). It is **three fully independent `uv` projects**, each with its own `pyproject.toml`, its own `.venv/`, and its own `uv.lock`:

| Project | Path | `pyproject.toml` | `requires-python` |
|---|---|---|---|
| Integration (root) | `.` | `pyproject.toml` | `>=3.14.2` |
| Docs (mkdocs) | `docs/` | `docs/pyproject.toml` | `>=3.13` (unchanged by the 2026-07-06 toolchain bump — docs project was not moved to 3.14) |
| Simulator (Flask) | `simulator/` | `simulator/pyproject.toml` | `>=3.14.2` |

Consequence for maintenance: a dependency bump in one project's `pyproject.toml`/`uv.lock` never touches the other two. If you add a Python dependency, check which of the three projects actually needs it before running `uv add` — running it from the repo root only ever touches the root project's lock.

### Devcontainer

`.devcontainer.json` (single file at repo root, not a `.devcontainer/` directory) defines:

- Base image: `mcr.microsoft.com/devcontainers/python:3.14` — this is where the "3.14" pin for local development actually originates (bumped from `python:3.13` in the 2026-07-06 toolchain refresh, #496).
- `postCreateCommand: scripts/setup` — runs the full bootstrap automatically on container creation.
- `postStartCommand: scripts/motd` — prints a welcome banner (repo name, branch, `python3 --version`) plus `scripts/help` on every container start; not load-bearing, just orientation.
- Forwards port 8123 (the devcontainer's own Home Assistant instance, config at `config/configuration.yaml`).
- Installs `ffmpeg`, `libturbojpeg0`, `libpcap-dev` via the `apt-packages` devcontainer feature (note: this covers the same `libturbojpeg` need as `scripts/bootstrap`'s manual apt/dnf branch, so inside the devcontainer that step is a no-op).
- VS Code customizations: ruff extension set as the default Python formatter, format-on-save enabled, Pylance in `basic` type-checking mode, default interpreter pinned to `${containerEnv:PWD}/.venv/bin/python` (the root project's venv — docs/simulator venvs are not wired into the editor by default).

Use the devcontainer when you want a known-good, pre-provisioned environment (including a runnable HA instance) without touching your host machine. Everything below still applies inside it. The CI-vs-local Python skew described below is now settled history (both sides run 3.14 as of 2026-07-06) — the devcontainer locks to 3.14, same as the manual bootstrap.

### scripts/lint vs CI lint — order reversed, converges in practice

| | Step 1 | Step 2 | Mutates files? |
|---|---|---|---|
| `scripts/lint` (local) | `uv run ruff format .` | `uv run ruff check . --fix` | Yes — rewrites files in place |
| `.github/workflows/lint.yml` (CI) | `uv run ruff check .` (no `--fix`) | `uv run ruff format . --check` (no rewrite) | No — fails the job instead |

The two pipelines run format/check in **opposite order**, and CI never mutates while local always does. In practice this converges: once a file is clean under both a formatter pass and a fixed lint pass, order doesn't matter — but if you only ever run one half locally (e.g. just `ruff check --fix` without `ruff format`) you can still get a CI-only failure. Always run the full `scripts/lint` before pushing, not a partial ruff invocation.

CI's lint job installs deps with `uv sync --extra dev` (no `--extra test`), so don't rely on test-only packages being present when reasoning about what the lint job's environment contains.

### scripts/test

```bash
scripts/test
# equivalent to:
uv run pytest --cov=custom_components/area_occupancy --cov-report=xml --cov-report=term-missing
```

Coverage gate: `[tool.coverage.report] fail_under = 85` in `pyproject.toml` (line 113). **SETTLED (2026-07-06):** the adjacent comment used to read `# Enforce 90% coverage minimum`, which was stale and contradicted the actual `fail_under = 85` — that mismatch was fixed as part of the 2026-07-06 merge wave and now reads `# Enforced global minimum; aim for 90%+ on core calculation modules (AGENTS.md)`, correctly distinguishing the enforced 85% floor from the 90% aspiration. AGENTS.md's "85%+ coverage requirement (90% for core calculations)" phrasing is consistent with this: 85 is the enforced CI gate, 90 remains an unenforced aspiration for calculation-critical files, not a tool-config gate.

CI's `test.yml` runs the identical pytest invocation but additionally sets `AREA_OCCUPANCY_AUTO_INIT_DB: "1"` as a job-level env var (see trap below — `tests/conftest.py` already sets this for you locally, so you don't normally need to set it by hand).

### Traps (verified, with fixes)

#### 1. [SETTLED 2026-07-06] CI ran Python 3.14 while local venvs ran 3.13, with no `.python-version` file to warn you

**Status: fixed, as of the 2026-07-06 toolchain-refresh merge wave (#496). Keeping the full story below since the failure mode is instructive if this ever regresses.**

Historically, no `.python-version` file existed anywhere in the repo (root, `docs/`, `simulator/` all checked — none found). Root `pyproject.toml` declared `requires-python = ">=3.13.2"` with no upper bound, and local tooling (bootstrap, devcontainer) pinned 3.13 while CI's `uv sync` (no explicit version pin in the workflow) picked up whatever `uv` resolved by default — which had drifted to 3.14 on GitHub's runners. That produced a real local-3.13-vs-CI-3.14 mismatch that could hide bugs that only reproduced on one side.

**Current state, verified 2026-07-06 against `main` HEAD `17b71d2`:**
- A `.python-version` file now exists at repo root and pins `3.14`.
- Root and simulator `pyproject.toml` both declare `requires-python = ">=3.14.2"` (docs project is unchanged at `>=3.13`, see the three-projects table above — it doesn't run CI-critical code so this asymmetry is not itself a trap).
- `scripts/bootstrap` now runs `uv venv --python 3.14`, and the devcontainer's base image is `mcr.microsoft.com/devcontainers/python:3.14`. Confirmed: `.venv/bin/python --version` → `Python 3.14.3`.
- CI (`test.yml`, `lint.yml`) still calls `astral-sh/setup-uv@v7` then `uv sync` with no explicit python-version pin in the workflow YAML itself, but now picks up the committed `.python-version` file. Verified directly from a real CI run's logs (2026-07-06, run `28808228593`, `headSha 17b71d2...`): uv resolved `Using CPython 3.14.6` and pytest reported `platform linux -- Python 3.14.6`.
- `pyproject.toml` `classifiers` now list only `Programming Language :: Python :: 3.14` — the prior dual `3.13`/`3.14` classifier listing (which had signaled deliberate dual support) is gone, consistent with 3.13 being dropped rather than merely one of two supported versions.

**How to check both, if you suspect this has regressed:**
```bash
# Local
.venv/bin/python --version
cat .python-version

# CI (latest run on the branch you care about)
gh run list --workflow=test.yml --limit 1 --json databaseId -q '.[0].databaseId'
gh run view <databaseId> --log | grep -i "Using CPython"
```
If you hit a bug that only reproduces in CI (or only locally), check for a version mismatch first before assuming it's a logic bug — this exact failure mode has happened before in this repo.

#### 2. [SETTLED 2026-07-06] Ruff triple-version skew

**Status: fixed, as of the 2026-07-06 toolchain-refresh merge wave (#496). Keeping the full story below since the failure mode is instructive if this ever regresses.**

Historically, three different places pinned three different ruff versions: the `pyproject.toml` floor (`required-version = ">=0.13.0"`, a minimum not a pin), the pre-commit hook rev (`v0.14.2`), and whatever `uv.lock` actually resolved (`0.15.2`). Because pre-commit hooks run in their own isolated hook environment (not the project's `uv`-managed venv), `pre-commit run --all-files` linted with 0.14.2 while `scripts/lint`/CI linted with whatever `uv.lock` resolved. If ruff added/removed/renamed a rule between those versions, a file could pass one and fail the other.

**Current state, verified 2026-07-06 against `main` HEAD `17b71d2`:**

| Source | Version | Where |
|---|---|---|
| `pyproject.toml` floor | `>=0.13.0` (unchanged — still just a floor, not a pin) | `[tool.ruff] required-version = ">=0.13.0"` |
| `pyproject.toml` dev-dep pin (new) | `==0.15.2` | `[project.optional-dependencies] dev = ["ruff==0.15.2", ...]` |
| pre-commit hook pin | `v0.15.2` | `.pre-commit-config.yaml` `rev: v0.15.2` |
| Actually resolved/installed by `uv` | `0.15.2` | confirmed via `uv run ruff --version` → `ruff 0.15.2` |

The dev-dep pin and the pre-commit rev now agree, and both match what `uv` actually resolves — the three-way skew is closed. `pre-commit run --all-files` and `scripts/lint`/CI now lint with the same ruff version.

**How to check, if you suspect this has regressed:**
```bash
grep required-version pyproject.toml
grep 'ruff==' pyproject.toml
grep "rev:" .pre-commit-config.yaml
uv run ruff --version
```
If you see a lint disagreement between a pre-commit run and `scripts/lint`/CI, re-run these four checks before assuming your ruff config is broken.

#### 3. `pytest-homeassistant-custom-component` quirks

Pinned version: `pytest-homeassistant-custom-component==0.13.345` (`pyproject.toml` test extra; bumped from `0.13.315` in the 2026-07-06 dependency refresh, #496, alongside `homeassistant==2026.7.1`).

- **`expected_lingering_timers` marker**: registered in `pyproject.toml` (`markers = ["expected_lingering_timers: mark test as expected to have lingering timers (Home Assistant test plugin)"]`). Apply it to a test when HA's test harness would otherwise fail the test for leaving a timer running past teardown — but only when the lingering timer is actually expected/benign for that test, not as a blanket suppressor.
- **`asyncio_mode = "auto"`**: set in `[tool.pytest.ini_options]` — async test functions run without needing `@pytest.mark.asyncio` on each one. `asyncio_default_fixture_loop_scope = "function"` is also set alongside it.
- **`SAWarning` promoted to a hard error**: `filterwarnings = ["error::sqlalchemy.exc.SAWarning", ...]`. Any SQLAlchemy warning (e.g. from a malformed query or an implicit cartesian product) fails the test outright instead of just printing. If a test fails with an `SAWarning`-turned-exception, that's a real SQLAlchemy usage issue to fix, not a warning to silence — don't add a blanket `ignore` for it.
- Other deliberate `filterwarnings` ignores worth knowing about before you "fix" a warning that isn't actually a problem: unclosed-sqlite-connection `ResourceWarning`s (Python 3.13 is stricter about unclosed resources than SQLAlchemy's pooling teardown timing allows for), and a couple of asyncio-loop `DeprecationWarning`s specific to CI environments. See `pyproject.toml` `[tool.pytest.ini_options] filterwarnings` for the exact list and inline comments explaining each.

#### 4. `AREA_OCCUPANCY_AUTO_INIT_DB=1` — test-only executor bypass, never use in production code

`custom_components/area_occupancy/db/core.py` (`AreaOccupancyDB.__init__`) does:
```python
if os.getenv("AREA_OCCUPANCY_AUTO_INIT_DB") == "1":
    self.initialize_database()
```
`initialize_database()` performs **blocking I/O** (`maintenance.ensure_db_exists`). Its own docstring says explicitly: *"In production environments, it should be called via `hass.async_add_executor_job()`... In test environments (when `AREA_OCCUPANCY_AUTO_INIT_DB=1` is set), this method may be called directly."*

- `tests/conftest.py` sets it unconditionally at import time: `os.environ["AREA_OCCUPANCY_AUTO_INIT_DB"] = "1"` — this is why you don't need to export it yourself for `scripts/test` to work locally.
- CI's `test.yml` also sets it explicitly as a job-level `env:` (belt-and-suspenders with conftest).
- **Never** gate this env var into any code path that isn't test setup. If you're tempted to reach for it to "quickly init the DB" in a script or a migration, use `hass.async_add_executor_job(...)` instead, per AGENTS.md's "Database Operations" rule — this var exists purely to let synchronous test fixtures avoid needing an event loop for DB setup, not as a general sync/async escape hatch.

#### 5. uv.lock churn from `uv run`/`uv sync`

Several dependencies in the root `pyproject.toml` (e.g. `pre-commit`, unpinned or loosely pinned) have no upper-bound version pin. Running `uv sync` or `uv run <anything>` can silently re-resolve and rewrite `uv.lock` to a newer transitively-compatible version even when you didn't touch `pyproject.toml` — producing a lock-file diff unrelated to your actual change. Confirmed historical example: commit `b297d54` ("Bump pre-commit version from 4.5.0 to 4.5.1 in uv.lock") touched only `uv.lock`, with no `pyproject.toml` change, purely from a routine `uv` re-resolution picking up a newer `pre-commit` release.

**Practical guidance:**
- Before committing, check whether `uv.lock` changed for a reason connected to your work: `git diff uv.lock` and read the package names in the diff hunks.
- If you see unrelated packages bumped in your diff, that's expected churn, not something you broke — but call it out separately in the PR description rather than silently bundling it, especially since **no silent math/behavior changes** is one of this project's unwritten laws (a dependency bump that changes runtime behavior is exactly the kind of silent change to avoid bundling invisibly).
- `uv sync --locked` (or `--frozen`) will refuse to modify the lock file if you want to confirm your environment matches the committed lock exactly, without triggering a re-resolution.

### Environment Setup — Provenance and maintenance

Verified 2026-07-06 against integration version 2026.5.17 (`pyproject.toml`, `custom_components/area_occupancy/manifest.json`, `custom_components/area_occupancy/const.py` `DEVICE_SW_VERSION`), on `main` at HEAD `17b71d2` post-merge-wave (facts in this skill are branch-independent build/env facts, not tied to any in-flight PR). Note the version number itself hasn't moved — the 2026-07-06 merge wave (toolchain refresh, adjacent-areas, etc.) landed on `main` but is not yet in a tagged release.

Re-verification commands, by volatile fact:

| Fact | Re-check with |
|---|---|
| Bootstrap sequence / commands | `cat scripts/bootstrap` |
| Coverage gate threshold | `grep fail_under pyproject.toml` |
| Local vs CI lint commands | `cat scripts/lint`; `cat .github/workflows/lint.yml` |
| Devcontainer base image / features | `cat .devcontainer.json` |
| Three-uv-project layout | `find . -maxdepth 2 -name uv.lock -not -path './.venv/*'`; `grep -rn workspace pyproject.toml docs/pyproject.toml simulator/pyproject.toml` |
| CI Python version actually used | `gh run list --workflow=test.yml --limit 1 --json databaseId -q '.[0].databaseId'` then `gh run view <id> --log \| grep -i "Using CPython"` |
| Local Python version | `.venv/bin/python --version` |
| Ruff version skew (3 sources) | `grep required-version pyproject.toml`; `grep "rev:" .pre-commit-config.yaml`; `uv run ruff --version` |
| pytest-homeassistant-custom-component pin | `grep pytest-homeassistant-custom-component pyproject.toml` |
| pytest markers/filterwarnings | `grep -A15 '\[tool.pytest.ini_options\]' pyproject.toml` |
| `AREA_OCCUPANCY_AUTO_INIT_DB` usage sites | `grep -rn AREA_OCCUPANCY_AUTO_INIT_DB --include='*.py' .` |
| uv.lock churn precedent | `git log --oneline -- uv.lock \| head -20` |

---

## Running, Operating, and Releasing

### 1. Devcontainer Home Assistant instance

The devcontainer config is `.devcontainer.json` at repo root (not a `.devcontainer/` folder — easy to look in the wrong place). It builds `mcr.microsoft.com/devcontainers/python:3.14` and forwards port 8123 — see the Devcontainer subsection under Environment Setup above for the full base-image/version-bump detail — and runs `scripts/setup` (→ `scripts/bootstrap`) on create.

**Start Home Assistant:**

```bash
scripts/develop
```

This does three things: (1) creates `./config` via `uv run hass --config ./config --script ensure_config` if it doesn't exist yet, (2) sets `PYTHONPATH` to include `$PWD/custom_components` so `custom_components.area_occupancy` is importable without symlinking into HA's config dir, (3) runs `uv run hass --config ./config --debug`, piped through a colorizer (red=ERROR/CRITICAL, yellow=WARNING, green=INFO, dim=DEBUG). HA comes up at **http://localhost:8123**.

`config/configuration.yaml` is a hand-built minimal config (not `default_config:`, to avoid pulling every HA integration as a dependency). It already wires in `frontend`, `recorder`, `history`, `logbook`, debug logging for this integration (see §2), and a full synthetic sensor rig — one `input_boolean`/`input_select`/`input_number` plus a `template:` binary_sensor/sensor/cover per `InputType`, spread across 5 rooms (Living Room, Kitchen, Bathroom, Bedroom, Hallway), each exercising a different feature (full sensor suite, wasp-in-box, sleep/long-decay, minimal/short-decay passageway). You drive it by flipping those input helpers in **Developer Tools → Actions** or the dashboard — there is no real hardware in the loop.

**Install the integration into this instance:** it's already there — `custom_components/area_occupancy` is on `PYTHONPATH` via `scripts/develop`, so as soon as HA boots the integration is installable through the normal UI flow (no HACS needed inside the devcontainer).

**Add an area via the UI:**

1. Settings → Devices & Services → Add Integration → search "Area Occupancy Detection" (first time only; this is the `user` config-flow step).
2. You land in the "Add New Area" flow, which walks four steps in order: `area_basics` (name, purpose) → `area_motion` (motion sensors) → `area_sensors` (media/appliance/door/window/environmental sensors) → `area_behavior` (weights, decay, threshold) → finishes via `finish_setup`.
3. To add a *second* area to an already-configured integration entry: gear icon on the integration card → Configure → "Manage Areas" → "Add New Area" (same four-step flow, driven by the options flow's `async_step_add_area`).

Verified: `.devcontainer.json` (full contents), `scripts/develop`, `config/configuration.yaml` lines 1–40, `custom_components/area_occupancy/config_flow.py` (`async_step_area_basics`/`area_motion`/`area_sensors`/`area_behavior`/`add_area`/`manage_areas` step definitions), `custom_components/area_occupancy/strings.json` lines 12–18, 204–333.

For anything that goes wrong *setting up* this environment (bootstrap failures, uv sync errors, version skew) — see the Environment Setup section above.

---

### 2. Debug logging recipe

The recipe is already live in the devcontainer's `config/configuration.yaml`:

```yaml
logger:
  default: error
  logs:
    custom_components.area_occupancy: debug
```

For a real HA install, add this block to `configuration.yaml` and restart HA (no need to set `default: error` there — that's just to keep the devcontainer's own logs quiet). Logs land in **Settings → System → Logs** in the UI and, in the devcontainer, at `config/home-assistant.log` on disk.

The project's own `docs/docs/technical/debug.md` states an explicit operator order, which is worth following exactly because it's cheapest-first: **(1) download diagnostics** from the integration card's ⋮ menu — no config change needed, captures every prior/weight/evidence/decay/correlation/health value in one JSON file; **(2) check Settings → System → Repairs** for `sensor_health_*`/`pipeline_health_*` issues — a stuck or unavailable sensor is a common root cause and is surfaced automatically; **(3) only then** enable debug logging and reproduce live.

Verified: `config/configuration.yaml` lines 1–21; `docs/docs/technical/debug.md` (full file, "Diagnostics Export" / "Sensor Health Repairs" / "Debug Logging" sections).

---

### 3. Where the SQLite database lives

`DB_NAME = "area_occupancy.db"` (`const.py:265`). The path is always `<hass.config.config_dir>/.storage/area_occupancy.db`, computed in `AreaOccupancyDB._setup_paths()` (`db/core.py`). In the devcontainer that resolves to `config/.storage/area_occupancy.db` from repo root — confirmed present on disk with a `.db.backup` sibling (the periodic-backup mechanism in `db/maintenance.py` copies the file, after a WAL checkpoint, on a configurable interval). Engine: SQLAlchemy `sqlite:///{db_path}`, `NullPool`, `check_same_thread=False`, `timeout=10s`.

It's a normal SQLite3 file — inspect it directly with `sqlite3 config/.storage/area_occupancy.db` from repo root, or with `scripts/visualize_distributions.py --db-path <path>` for a matplotlib entry point. Reads are safe anytime; stop HA (or accept WAL-mode concurrent-read semantics) before writing directly.

**In tests**, there is no real file by default: the `db_engine` fixture in `tests/conftest.py` creates an **in-memory** SQLite engine (`sqlite:///:memory:?cache=shared`, `StaticPool`, `check_same_thread=False`) so state is visible across executor-thread connections within one test process. A separate helper, `setup_test_db_engine(db, db_path)`, exists for the minority of tests that need a real file-backed DB (e.g. testing backup/restore), pointed at a `tmp_path`-style path.

Verified: `custom_components/area_occupancy/const.py:265`; `custom_components/area_occupancy/db/core.py` `_setup_paths`/`_setup_engine`; `ls -la config/.storage/` (area_occupancy.db 966,656 bytes + area_occupancy.db.backup 516,096 bytes present); `db/maintenance.py` `_backup_database`/`periodic_health_check`; `tests/conftest.py` `db_engine` fixture and `setup_test_db_engine` helper.

---

### 4. The analysis pipeline as an operational process

`run_full_analysis()` in `data/analysis.py` runs on an hourly timer (see below) and orchestrates the whole learning loop. **On `main` at HEAD (2026.5.17, commit `17b71d2`) it is 13 steps** — PR #454 ("adjacent-areas", merged 2026-07-06) inserted a `transition_learning` step between correlation analysis and pipeline health check. Each step is wrapped by an inner `_run_step()` that times it, logs on success/failure, and — critically — **swallows the exception and continues to the next step** rather than aborting the whole run:

| # | Step name (as logged) | What it does |
|---|---|---|
| 1 | `sync_states` | Pull recorder history since the last watermark (§5) |
| 2 | `health_check_and_prune` | DB integrity check + backup + prune intervals older than `RETENTION_DAYS` (365) |
| 3 | `sensor_health_check` | Per-entity anomaly detection → HA repair issues (skipped entirely if the integration-level `health_enabled` toggle is off) |
| 4 | `populate_occupied_intervals_cache` | Rebuild the motion-ground-truth cache, only if stale/invalid |
| 5 | `interval_aggregation` | Raw intervals → daily/weekly/monthly rollups |
| 6 | `numeric_aggregation` | Raw numeric samples → hourly/weekly rollups (feeds Gaussian correlation) |
| 7 | `recalculate_priors` | Per-area `PriorAnalyzer`: global prior + 168 (day-of-week × hour) time-priors |
| 8 | `correlation_analysis` | `db/correlation.py`: statistical sensor↔occupancy correlation, needs ≥`MIN_CORRELATION_SAMPLES` (50) |
| 9 | `transition_learning` | Adjacent-areas Phase 3: count room-to-room transition observations into `AreaTransitions`, feeding the Bayesian boost and decay modifier (§ adjacency remains unvalidated on real homes — see below) |
| 10 | `pipeline_health_check` | Area-scope anomalies (no global prior after grace period, stale cache, slow analysis, high correlation-failure rate) → repair issues |
| 11 | `save_data_before_refresh` | Persist DB (preserves decay state ahead of the refresh) |
| 12 | `refresh_coordinator` | Recompute `probability()` for every area |
| 13 | `save_data_after_refresh` | Persist DB again |

Each step logs one of:

```
Step N: <step_name> completed in X.XX ms      # INFO, on success
Step N: <step_name> FAILED in X.XX ms          # ERROR (via _LOGGER.exception, includes traceback), on any exception
```

**What "Step N FAILED" means operationally:** that one step raised (any `Exception`), the pipeline logged it and moved on to step N+1 — a failure in `correlation_analysis` does not stop `pipeline_health_check` or the refresh from running. At the end of the run you get one summary line:

```
Analysis completed: S/13 steps succeeded (FAILED: step_a, step_b) in X.XX ms   # WARNING, if any step failed
Full analysis completed: 13/13 steps succeeded in X.XX ms                      # INFO, if all succeeded
```

If *any* step failed, `run_full_analysis` raises `HomeAssistantError` after the finally-block summary, which the coordinator's timer handler (`coordinator.run_analysis`) catches. **Backoff on failure: retry in 15 minutes** instead of the normal hourly cadence (`coordinator.py`, `run_analysis`: `next_update = _now + timedelta(minutes=15)` when `_failed`). On a clean run it reschedules for `analysis_interval` seconds later (`ANALYSIS_INTERVAL = 3600`, not currently exposed as a config-flow option — it's a fixed constant, not per-area). Note: the analysis-timer re-arm logic was moved out of the `finally` block in this same merge wave (Python 3.14 `SyntaxWarning` fix) — functionally identical, but if you're diffing `coordinator.run_analysis` against an older read of this code, that's why the shape changed.

Two shutdown-safety details worth knowing when reading logs: if HA starts shutting down mid-pipeline, remaining steps log `Step N: <name> skipped — shutdown in progress` (DEBUG) rather than FAILED, and the run's duration is deliberately **not** persisted (a fast, aborted run must not mask a previously-slow one in the health check's slow-analysis threshold). The **first** analysis run after HA (re)starts is deferred via `async_at_started` plus an additional fixed 5-minute delay, specifically so analysis never blocks HA's own bootstrap.

**Adjacent-areas feature status:** PR #454 is merged and `transition_learning` runs on every pipeline cycle now, but the feature itself remains an **unvalidated candidate** — it hasn't been proven out against real-home data yet, only tested (the 4 adjacency test files landed with #454). Treat boost/decay-modifier behavior from adjacency as something to watch, not a settled result.

Verified directly: `custom_components/area_occupancy/data/analysis.py` (`git show main:...`) lines 35–235 (`run_full_analysis`, `_run_step`, docstring listing all 13 steps, the `total_steps = 13` literal, and every `_LOGGER` call); `custom_components/area_occupancy/coordinator.py` (`git show main:...`) `_start_analysis_timer`/`run_analysis` (5-minute post-boot defer, 15-minute failure backoff re-armed outside `finally`, `ANALYSIS_INTERVAL`); `custom_components/area_occupancy/const.py:323,343` (`MIN_CORRELATION_SAMPLES=50`, `ANALYSIS_INTERVAL=3600`); `gh pr view 454 --json state,mergedAt` (state MERGED, `mergedAt` 2026-07-06T16:50:40Z).

Two other always-on timers worth knowing about while reading logs: a **decay timer** (`DECAY_INTERVAL=10s`, `const.py:342`) ticks every area's decay and triggers a coordinator refresh if any area has decay enabled, and a **save timer** (`SAVE_INTERVAL=600s`, `const.py:344`) persists the DB every 10 minutes independent of the analysis pipeline.

---

### 5. Recorder sync mechanics (step 1)

`sync_states(db)` (`db/sync.py`) is the pipeline's step 1. It computes a time window and pulls HA recorder history for the union of every configured entity across all areas:

- `start_time = queries.get_latest_interval(db)` — a **single global watermark**, not per-entity/per-area: `SELECT max(end_time) FROM intervals`, minus a fixed **1-hour overlap** to re-catch any interval whose end time was still open when last synced.
- **First run** (empty/missing `intervals` table, or any `SQLAlchemyError`/`ValueError`/etc. reading it): the watermark defaults to `utcnow() - 10 days` — a 10-day backfill window.
- `end_time = dt_util.utcnow()`.
- States are fetched via HA's `get_significant_states(hass, start_time, end_time, entity_ids, minimal_response=False)`, converted to `Intervals` and `NumericSamples` rows, and committed in dedup-checked batches of 250.

If the recorder call raises (`SQLAlchemyError`, `HomeAssistantError`, `TimeoutError`, `OSError`, `RuntimeError` — e.g. a concurrent recorder purge), `sync_states` logs `"Failed to sync states: %s"` and re-raises as `HomeAssistantError`, which is exactly what produces `Step 1: sync_states FAILED` in the pipeline log. Because the watermark is global (not per-area), a sync failure blocks fresh interval data for **every** area that cycle, not just one.

Verified: `custom_components/area_occupancy/db/sync.py` (`git show main:...`) `sync_states` (lines ~301–366) and `custom_components/area_occupancy/db/queries.py` `get_latest_interval` (lines 42–66, including the 10-day and 1-hour constants).

---

### 6. The simulator

`simulator/app.py` is a **Flask** web app that lets you paste the YAML/dict output of the `area_occupancy.run_analysis` service and interactively toggle sensors to see probability recalculate live. It imports and calls the **real** `EntityType`/`Entity`/`Decay` classes from `custom_components.area_occupancy.data.*` — not a reimplementation of those data classes.

**Resolved nuance for anyone using it to sanity-check math (fixed in PR #529, 2026.8.1):** the simulator's probability calculation used to call `bayesian_probability()` from `utils.py` — the classic naive-Bayes log-odds accumulator — which had **zero call sites in the live production coordinator/area path** ever since PR #353 (merged 2026-02-15) moved production onto a sigmoid/logistic pipeline (`sigmoid_probability`/`presence_probability`/`environmental_confidence`/`combined_probability`). So for over a release's worth of history, the simulator reproduced real `EntityType`/`Entity` state handling faithfully, but its probability *math* was the legacy formula, not what a running HA instance actually computed. PR #529 rewired `calculate_probability_breakdown()` onto the live pipeline (mirroring `Area._base_probability()`) and deleted `bayesian_probability()` entirely, so the simulator's numbers should now match production. Still worth a spot-check against the diagnostics export's `current.probability` if you're relying on exact figures, but it's no longer a *known* divergence.

**Run it locally:**

```bash
# one-time: install simulator deps into the shared .venv (same command scripts/bootstrap
# runs for you automatically — see Bootstrap sequence under Environment Setup above)
(cd simulator && uv sync --python ../.venv/bin/python)

# from repo root
python main.py
```

`main.py` at repo root imports `simulator.app:app` and runs it with `PORT` (default `5000`), `FLASK_DEBUG` (default `1`), `FLASK_HOST` (default `0.0.0.0`) env vars. `simulator/app.py` inserts the repo root onto `sys.path` itself, so `custom_components` imports work without any extra `PYTHONPATH` — unlike `scripts/develop`, no manual path wiring needed. Routes: `POST /api/analyze`, `POST /api/load`, `GET /api/get-purposes`.

Optionally pair it with a local docs preview (`cd docs && mkdocs serve`, open `http://localhost:8000/Area-Occupancy-Detection/simulator/`, point its "API Connection" field at `http://127.0.0.1:5000`).

**`simulator/README.md` is stale**: it instructs `pip install -r simulator/requirements.txt`, but no such file exists anywhere in the repo — `simulator/` is a `uv`/`pyproject.toml`-managed project (own `uv.lock`), so that line will fail as written. Use the `uv sync` command above instead.

**Docker / IBM Cloud deployment:** `simulator/Dockerfile` + `simulator/docker-compose.yml` build and run the same Flask app for container deployment (default port `10000` in that path, vs. `5000` via `main.py` locally — two different defaults, don't be surprised). `simulator/README.md` documents a manual `ibmcloud cr` push flow to IBM Cloud Container Registry.

**The docs site's interactive simulator is a thin client, not a local server.** `docs/docs/assets/simulator/app.js` hardcodes a production backend URL:

```js
const DEFAULT_API_BASE_URL = "https://area-occupancy-simulator.23ffgm1eszu1.eu-gb.codeengine.appdomain.cloud";
```

That's an IBM Cloud Code Engine instance the maintainer runs and updates manually. **The deploy process for that Code Engine instance is undocumented** — there is no CI workflow, script, or doc anywhere in this repo that automates or even describes pushing a new image to it (verified: no `.github/workflows/*.yml` references `simulator`; no repo doc mentions "Code Engine"). Treat this as a known gap, not something to reverse-engineer or invent a process for — if you need to update the live docs-site simulator backend, that requires the maintainer's out-of-band IBM Cloud access.

**Zero automated tests.** `pyproject.toml`'s `[tool.coverage.run]` `source` is scoped to `custom_components.area_occupancy` only, and no file under `tests/` references `simulator/`. `simulator/app.py` (~1000 lines) has no test coverage at all — changes to it are unverified by CI beyond `ruff` linting.

Verified (2026-07-06, pre-#529): `simulator/app.py` lines 1–45 (imports, `bayesian_probability` import, `sys.path` insertion), lines 448–1000+ (route/function definitions); `simulator/README.md` (full file — local-dev steps, stale `requirements.txt` reference, Docker/IBM Cloud section, "How It Works" section naming `bayesian_probability()`); `main.py` (repo root, full file); `simulator/Dockerfile`, `simulator/docker-compose.yml` (PORT=10000 default); `docs/docs/assets/simulator/app.js` lines 1–10 (hardcoded URL); `grep -rln "simulator" .github/workflows/` → no matches; `grep -rln "Code Engine\|codeengine"` across `*.md`/`*.yml` → no matches; `pyproject.toml` `[tool.coverage.run]` lines ~94–98; git blame on `def sigmoid_probability` → commit `a90f77b` "Add sigmoid-based occupancy detection framework (#353)". **Post-#529 (2026.8.1)**: `simulator/app.py` now imports `combine_priors`, `combined_probability`, `environmental_confidence`, `presence_probability` instead of `bayesian_probability`; `simulator/README.md`'s "How It Works" section names the live functions. The "zero automated tests" gap above is unchanged by that PR — it was smoke-tested manually, not added to CI.

---

### 7. Releases

**`gh release list` / `gh release view <tag>` is the changelog of record** — read it before assuming a fix isn't shipped. Releases use CalVer `YYYY.M.N` (e.g. `2026.5.17`), **not** the `MAJOR.MINOR.PATCH` semver AGENTS.md's "Branch and Release Strategy" section claims — that section is stale on this point (verified: `gh release list --limit 8` shows `2026.5.17, 2026.5.2, 2026.5.1, 2026.4.1, 2026.3.4, ...`, clearly calendar-versioned, not semver). Release bodies are hand-edited on top of GitHub's auto-generated "What's Changed" PR list — expect prose explaining the *why*, tables for structured changes (e.g. purpose→threshold mappings), and links back to originating issues.

**AGENTS.md's `dev → preview → main` release-branch flow is also stale.** `git ls-remote --heads origin` currently shows only `main`, `gh-pages`, and feature/fix/chore branches — no `dev`, `preview`, or `rc` branch exists on the remote. Spot-checking recently merged PRs shows all but one (#456, merged into the then-active `feat/adjacent-areas` branch, itself merged into `main` as part of #454) targeted `main` directly. The historical `dev`/`preview`/`rc` pipeline was real practice through roughly January 2026 and was abandoned in favor of direct feature-branch → `main` PRs some time after; `CONTRIBUTING.md` now says to branch from `main`. As of the 2026-07-06 CI-hygiene pass (#495), the stale artifact this section used to flag is **gone**: `.github/workflows/lint.yml` and `.github/workflows/validate.yml` both now trigger only on `main` (push and PR) — the old `rc`/`dev` entries in `pull_request.branches` were removed, not just dead weight.

**Version bump:** three files must change together (`manifest.json`, `pyproject.toml`, `const.py::DEVICE_SW_VERSION`). The versioning *policy* — the full three-file table, the CalVer scheme, and the `DEVICE_SW_VERSION`-vs-`CONF_VERSION` distinction — is owned by `aod-change-and-validation` §4; consult it before any bump. Operationally: edit all three to the identical `YYYY.M.N` string, merge, then create the GitHub release with the tag exactly equal to `manifest.json`'s version (release.yml hard-fails otherwise, next paragraph).

**HACS distribution:** `.github/workflows/release.yml` triggers on `release: types: [published]`. It hard-fails (`::error::` + exit 1) if `manifest.json`'s `version` doesn't exactly equal the release tag, then zips `custom_components/area_occupancy/` (excluding `__pycache__`/`*.pyc`) into `area_occupancy.zip` and uploads it via `gh release upload "$TAG" area_occupancy.zip --clobber`. `hacs.json` sets `"zip_release": true` and `"filename": "area_occupancy.zip"` to match — HACS installs users on the uploaded zip, not a raw source checkout. Note `hacs.json` declares minimum HA version `2024.8.0`. The dependency refresh in #496 (merged 2026-07-06) bumped the actual pin to `homeassistant==2026.7.1` (from 2026.2.2) plus `pytest-homeassistant-custom-component==0.13.345` — so the gap between "minimum HA HACS will allow" and "HA version actually tested against locally/in CI" is now nearly **2 years**, wider than before this merge wave, not narrower. HACS will let a user on a much older HA install this integration.

**`.github/workflows/validate.yml`** ("Validate") runs two independent jobs — `hassfest` (`home-assistant/actions/hassfest@master`) and `hacs` (`hacs/action@main`, `category: integration`) — on `workflow_dispatch`, a **daily cron (`0 0 * * *`)**, push to `main`, and PRs targeting `main` (the `rc`/`dev` PR triggers were removed in the 2026-07-06 CI-hygiene pass, #495 — see above). Both actions are pinned to floating refs (`@master`/`@main`), so a break in either upstream action can fail this workflow without any change in this repo. The `hacs` job specifically has a known transient-failure mode: it validates the presence of a local brand icon by calling out to `https://brands.home-assistant.io/domains.json`, and an upstream Cloudflare 525 there crashes the job with an uncaught `aiohttp.client_exceptions.ContentTypeError` — a repo-external flake, not a real validation failure, if you see that exact traceback.

Verified: `gh release list --limit 8`; `gh release view 2026.5.17 --json body`; `git ls-remote --heads origin`; `gh pr list --state merged --limit 15 --json number,baseRefName,mergedAt`; `.github/workflows/lint.yml` (full file, `branches: ["main"]` on both triggers); `.github/workflows/release.yml` (full file); `.github/workflows/validate.yml` (full file, `branches: ["main"]` on both triggers); `hacs.json` (full file); `pyproject.toml` lines 25 (`homeassistant==2026.7.1`), 42 (`pytest-homeassistant-custom-component==0.13.345`); `custom_components/area_occupancy/manifest.json:20`; `custom_components/area_occupancy/const.py:32`.

---

### 8. Docs deploy workflow

`.github/workflows/docs.yml` ("Build and Deploy Docs") triggers **only on push to `main`** (not PRs, not a cron). It checks out, configures git as `github-actions[bot]`, installs `uv` (`astral-sh/setup-uv@v7`), restores a `mkdocs-material-<ISO week>` cache, runs `uv sync --package area-occupancy-docs` (working directory `./docs`), then `uv run mkdocs gh-deploy --force` (also `./docs`) with `NO_MKDOCS_2_WARNING=1` set — deploying to the `gh-pages` branch. `permissions: contents: write`.

The docs stack is deliberately pinned: `mkdocs>=1.6.0,<2.0.0` and `mkdocs-material>=9.5.0,<10.0.0`, because MkDocs 2.0 is incompatible with Material for MkDocs and Material itself entered maintenance mode (per `docs/MIGRATION.md`, which sets an explicit 2026-08 revisit date for a possible move to the "Zensical" successor).

Verified: `.github/workflows/docs.yml` (full file); `docs/MIGRATION.md`.

---

### Running/Operating/Releasing — Provenance and maintenance

Date-stamped 2026-07-06 (post-merge sweep), integration version 2026.5.17 (main branch, commit `17b71d2`) — note the release version itself has **not** bumped since the merge wave; none of the day's PRs are in a tagged release yet, so "merged" here means "on `main`," not "shipped in a release." Everything above marked "Verified" was checked directly against the repo or `gh` at that commit — nothing here was taken from a dossier/summary without a direct read.

Re-verification commands for volatile facts:

| Fact category | Re-check with |
|---|---|
| Current version / step count / whether PR #454 has merged | `git log -1 --format=%H main`; `grep '"version"' custom_components/area_occupancy/manifest.json`; `grep -n 'total_steps' custom_components/area_occupancy/data/analysis.py`; `gh pr view 454 --json state,mergedAt` |
| Analysis pipeline step list/order | `grep -n '_run_step(' custom_components/area_occupancy/data/analysis.py` |
| Analysis timer intervals / backoff | `grep -n 'ANALYSIS_INTERVAL\|DECAY_INTERVAL\|SAVE_INTERVAL' custom_components/area_occupancy/const.py`; `grep -n 'timedelta(minutes=15)\|timedelta(minutes=5)' custom_components/area_occupancy/coordinator.py` |
| Recorder sync watermark constants | `grep -n 'timedelta(days=10)\|timedelta(hours=1)' custom_components/area_occupancy/db/queries.py` |
| DB location | `grep -n 'DB_NAME' custom_components/area_occupancy/const.py`; `ls -la config/.storage/` |
| Simulator's probability function (dead vs. live) | `grep -n 'bayesian_probability\|sigmoid_probability' simulator/app.py custom_components/area_occupancy/area/area.py` |
| Docs-site simulator backend URL | `grep -n 'DEFAULT_API_BASE_URL' docs/docs/assets/simulator/app.js` |
| Release list / changelog | `gh release list --limit 10` |
| Branch strategy reality (dev/preview/rc dead?) | `git ls-remote --heads origin`; `gh pr list --state merged --limit 15 --json number,baseRefName` |
| HACS / validate.yml cron and floating refs | `cat .github/workflows/validate.yml` |
| HA version pin vs. HACS minimum | `grep -n 'homeassistant==' pyproject.toml`; `grep -n 'homeassistant' hacs.json` |
| PRs referenced here (#489–496, #454) — all merged 2026-07-06, so this row is about catching the *next* wave, not re-litigating this one | `gh pr view <n> --json state,mergedAt` |
