#!/usr/bin/env bash
# One-command reproduction from a clean checkout (blueprint §11.2).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-full}"
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[test]"
pytest
python -m vista.run_all --config "configs/experiments/${CONFIG}.json"
python scripts/build_dashboard.py "${CONFIG}"
echo "done: results/${CONFIG}/, paper_artifacts/figures/${CONFIG}/, paper_artifacts/dashboard_${CONFIG}.html"
