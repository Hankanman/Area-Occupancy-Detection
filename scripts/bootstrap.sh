#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Python dev env with uv + pre-commit.
# Optional: export PYTHON_VERSION=3.12 to force a version.

export PYTHON_VERSION=3.13

cd "$(dirname "$0")/.."  # ensure repo root (script in scripts/)

# 1) Ensure uv
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] Installing uv..."
  curl -fsSL https://astral.sh/uv/install.sh | sh -s -- -y
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) Create venv at .venv
if [[ -n "${PYTHON_VERSION:-}" ]]; then
  echo "[setup] Creating venv with Python ${PYTHON_VERSION}..."
  uv venv --python "${PYTHON_VERSION}" .venv
else
  echo "[setup] Creating venv with system Python..."
  uv venv .venv
fi

# 3) Activate venv
# (uv can run without activation, but activation keeps tooling consistent)
# shellcheck disable=SC1091
source .venv/bin/activate

# 4) Install requirements
if [[ -f requirements.txt ]]; then
  echo "[setup] Installing requirements.txt..."
  uv pip install -r requirements.txt
fi
if [[ -f requirements_test.txt ]]; then
  echo "[setup] Installing requirements_test.txt..."
  uv pip install -r requirements_test.txt
fi

# 5) Install and set up pre-commit hooks via uvx
echo "[setup] Installing pre-commit hooks..."
uvx pre-commit install --install-hooks

# Optional: update hook versions to latest on first run
uvx pre-commit autoupdate || true

# Optional: run all hooks once to warm caches
uvx pre-commit run -a || true

echo "[done] Environment ready. Activate with: source .venv/bin/activate"
