---
name: aod-build-and-env
description: Use when setting up, resetting, or debugging the local development environment for Area Occupancy Detection — bootstrap fails, `uv sync` errors, wrong Python version, ruff or pytest behaving differently locally vs CI, uv.lock showing unexpected diffs, devcontainer questions, or anything mentioning scripts/bootstrap, scripts/setup, scripts/lint, scripts/test, libturbojpeg, or "three uv projects".
---

# AOD Build and Environment

## What this covers

How to recreate this repo's dev environment from a clean clone (three independent `uv` projects, devcontainer, bootstrap script sequence), the differences between local `scripts/lint`/`scripts/test` and their CI counterparts, and the specific environment traps that have wasted time here before: Python-version skew between CI and local, ruff triple-version skew, `pytest-homeassistant-custom-component` quirks, the test-only DB-init env var, and uv.lock churn.

## When NOT to use this

- Writing or fixing tests / coverage strategy beyond "why does pytest behave oddly" → `aod-validation-and-qa`
- Runtime debugging of the running integration (HA logs, coordinator behavior) → `aod-debugging-playbook`
- Diagnostic tooling/scripts for analyzing data → `aod-diagnostics-and-tooling`
- Release/branching/version-bump process → `aod-change-control`

## Bootstrap sequence (copy-pasteable)

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

## Three separate uv projects — not a workspace

This is not a `uv` workspace (`git grep -n workspace pyproject.toml docs/pyproject.toml simulator/pyproject.toml` → zero hits; no `[tool.uv.workspace]` anywhere). It is **three fully independent `uv` projects**, each with its own `pyproject.toml`, its own `.venv/`, and its own `uv.lock`:

| Project | Path | `pyproject.toml` | `requires-python` |
|---|---|---|---|
| Integration (root) | `.` | `pyproject.toml` | `>=3.14.2` |
| Docs (mkdocs) | `docs/` | `docs/pyproject.toml` | `>=3.13` (unchanged by the 2026-07-06 toolchain bump — docs project was not moved to 3.14) |
| Simulator (Flask) | `simulator/` | `simulator/pyproject.toml` | `>=3.14.2` |

Consequence for maintenance: a dependency bump in one project's `pyproject.toml`/`uv.lock` never touches the other two. If you add a Python dependency, check which of the three projects actually needs it before running `uv add` — running it from the repo root only ever touches the root project's lock.

## Devcontainer

`.devcontainer.json` (single file at repo root, not a `.devcontainer/` directory) defines:

- Base image: `mcr.microsoft.com/devcontainers/python:3.14` — this is where the "3.14" pin for local development actually originates (bumped from `python:3.13` in the 2026-07-06 toolchain refresh, #496).
- `postCreateCommand: scripts/setup` — runs the full bootstrap automatically on container creation.
- `postStartCommand: scripts/motd` — prints a welcome banner (repo name, branch, `python3 --version`) plus `scripts/help` on every container start; not load-bearing, just orientation.
- Forwards port 8123 (the devcontainer's own Home Assistant instance, config at `config/configuration.yaml`).
- Installs `ffmpeg`, `libturbojpeg0`, `libpcap-dev` via the `apt-packages` devcontainer feature (note: this covers the same `libturbojpeg` need as `scripts/bootstrap`'s manual apt/dnf branch, so inside the devcontainer that step is a no-op).
- VS Code customizations: ruff extension set as the default Python formatter, format-on-save enabled, Pylance in `basic` type-checking mode, default interpreter pinned to `${containerEnv:PWD}/.venv/bin/python` (the root project's venv — docs/simulator venvs are not wired into the editor by default).

Use the devcontainer when you want a known-good, pre-provisioned environment (including a runnable HA instance) without touching your host machine. Everything below still applies inside it. The CI-vs-local Python skew described below is now settled history (both sides run 3.14 as of 2026-07-06) — the devcontainer locks to 3.14, same as the manual bootstrap.

## scripts/lint vs CI lint — order reversed, converges in practice

| | Step 1 | Step 2 | Mutates files? |
|---|---|---|---|
| `scripts/lint` (local) | `uv run ruff format .` | `uv run ruff check . --fix` | Yes — rewrites files in place |
| `.github/workflows/lint.yml` (CI) | `uv run ruff check .` (no `--fix`) | `uv run ruff format . --check` (no rewrite) | No — fails the job instead |

The two pipelines run format/check in **opposite order**, and CI never mutates while local always does. In practice this converges: once a file is clean under both a formatter pass and a fixed lint pass, order doesn't matter — but if you only ever run one half locally (e.g. just `ruff check --fix` without `ruff format`) you can still get a CI-only failure. Always run the full `scripts/lint` before pushing, not a partial ruff invocation.

CI's lint job installs deps with `uv sync --extra dev` (no `--extra test`), so don't rely on test-only packages being present when reasoning about what the lint job's environment contains.

## scripts/test

```bash
scripts/test
# equivalent to:
uv run pytest --cov=custom_components/area_occupancy --cov-report=xml --cov-report=term-missing
```

Coverage gate: `[tool.coverage.report] fail_under = 85` in `pyproject.toml` (line 113). **SETTLED (2026-07-06):** the adjacent comment used to read `# Enforce 90% coverage minimum`, which was stale and contradicted the actual `fail_under = 85` — that mismatch was fixed as part of the 2026-07-06 merge wave and now reads `# Enforced global minimum; aim for 90%+ on core calculation modules (CLAUDE.md)`, correctly distinguishing the enforced 85% floor from the 90% aspiration. CLAUDE.md's "85%+ coverage requirement (90% for core calculations)" phrasing is consistent with this: 85 is the enforced CI gate, 90 remains an unenforced aspiration for calculation-critical files, not a tool-config gate.

CI's `test.yml` runs the identical pytest invocation but additionally sets `AREA_OCCUPANCY_AUTO_INIT_DB: "1"` as a job-level env var (see trap below — `tests/conftest.py` already sets this for you locally, so you don't normally need to set it by hand).

## Traps (verified, with fixes)

### 1. [SETTLED 2026-07-06] CI ran Python 3.14 while local venvs ran 3.13, with no `.python-version` file to warn you

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

### 2. [SETTLED 2026-07-06] Ruff triple-version skew

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

### 3. `pytest-homeassistant-custom-component` quirks

Pinned version: `pytest-homeassistant-custom-component==0.13.345` (`pyproject.toml` test extra; bumped from `0.13.315` in the 2026-07-06 dependency refresh, #496, alongside `homeassistant==2026.7.1`).

- **`expected_lingering_timers` marker**: registered in `pyproject.toml` (`markers = ["expected_lingering_timers: mark test as expected to have lingering timers (Home Assistant test plugin)"]`). Apply it to a test when HA's test harness would otherwise fail the test for leaving a timer running past teardown — but only when the lingering timer is actually expected/benign for that test, not as a blanket suppressor.
- **`asyncio_mode = "auto"`**: set in `[tool.pytest.ini_options]` — async test functions run without needing `@pytest.mark.asyncio` on each one. `asyncio_default_fixture_loop_scope = "function"` is also set alongside it.
- **`SAWarning` promoted to a hard error**: `filterwarnings = ["error::sqlalchemy.exc.SAWarning", ...]`. Any SQLAlchemy warning (e.g. from a malformed query or an implicit cartesian product) fails the test outright instead of just printing. If a test fails with an `SAWarning`-turned-exception, that's a real SQLAlchemy usage issue to fix, not a warning to silence — don't add a blanket `ignore` for it.
- Other deliberate `filterwarnings` ignores worth knowing about before you "fix" a warning that isn't actually a problem: unclosed-sqlite-connection `ResourceWarning`s (Python 3.13 is stricter about unclosed resources than SQLAlchemy's pooling teardown timing allows for), and a couple of asyncio-loop `DeprecationWarning`s specific to CI environments. See `pyproject.toml` `[tool.pytest.ini_options] filterwarnings` for the exact list and inline comments explaining each.

### 4. `AREA_OCCUPANCY_AUTO_INIT_DB=1` — test-only executor bypass, never use in production code

`custom_components/area_occupancy/db/core.py` (`AreaOccupancyDB.__init__`) does:
```python
if os.getenv("AREA_OCCUPANCY_AUTO_INIT_DB") == "1":
    self.initialize_database()
```
`initialize_database()` performs **blocking I/O** (`maintenance.ensure_db_exists`). Its own docstring says explicitly: *"In production environments, it should be called via `hass.async_add_executor_job()`... In test environments (when `AREA_OCCUPANCY_AUTO_INIT_DB=1` is set), this method may be called directly."*

- `tests/conftest.py` sets it unconditionally at import time: `os.environ["AREA_OCCUPANCY_AUTO_INIT_DB"] = "1"` — this is why you don't need to export it yourself for `scripts/test` to work locally.
- CI's `test.yml` also sets it explicitly as a job-level `env:` (belt-and-suspenders with conftest).
- **Never** gate this env var into any code path that isn't test setup. If you're tempted to reach for it to "quickly init the DB" in a script or a migration, use `hass.async_add_executor_job(...)` instead, per CLAUDE.md's "Database Operations" rule — this var exists purely to let synchronous test fixtures avoid needing an event loop for DB setup, not as a general sync/async escape hatch.

### 5. uv.lock churn from `uv run`/`uv sync`

Several dependencies in the root `pyproject.toml` (e.g. `pre-commit`, unpinned or loosely pinned) have no upper-bound version pin. Running `uv sync` or `uv run <anything>` can silently re-resolve and rewrite `uv.lock` to a newer transitively-compatible version even when you didn't touch `pyproject.toml` — producing a lock-file diff unrelated to your actual change. Confirmed historical example: commit `b297d54` ("Bump pre-commit version from 4.5.0 to 4.5.1 in uv.lock") touched only `uv.lock`, with no `pyproject.toml` change, purely from a routine `uv` re-resolution picking up a newer `pre-commit` release.

**Practical guidance:**
- Before committing, check whether `uv.lock` changed for a reason connected to your work: `git diff uv.lock` and read the package names in the diff hunks.
- If you see unrelated packages bumped in your diff, that's expected churn, not something you broke — but call it out separately in the PR description rather than silently bundling it, especially since **no silent math/behavior changes** is one of this project's unwritten laws (a dependency bump that changes runtime behavior is exactly the kind of silent change to avoid bundling invisibly).
- `uv sync --locked` (or `--frozen`) will refuse to modify the lock file if you want to confirm your environment matches the committed lock exactly, without triggering a re-resolution.

## Provenance and maintenance

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
