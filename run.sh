#!/usr/bin/env bash
# Entry point: runs the advisor against a persona.
# Usage:
#   ./run.sh                   # default persona: david
#   ./run.sh --persona priya
#   ./run.sh --all
set -euo pipefail

cd "$(dirname "$0")"

# Silence chromadb's posthog telemetry warnings.
export ANONYMIZED_TELEMETRY=False
export CHROMA_TELEMETRY_DISABLED=true
export POSTHOG_DISABLED=true

# Load .env if present (so OPENROUTER_API_KEY etc. are exported).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Make sure the venv exists.
if [[ ! -x .venv/bin/python ]]; then
  echo "No .venv found. Creating one and installing dependencies…" >&2
  python3.11 -m venv .venv
  .venv/bin/pip install --upgrade pip --quiet
  .venv/bin/pip install -r requirements.txt --quiet
fi

exec .venv/bin/python -m src.main "$@"
