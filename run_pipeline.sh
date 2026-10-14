#!/usr/bin/env bash
# Full offline pipeline. Pass --quick for a fast smoke run.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
export PYTHONUNBUFFERED=1
$PY scripts/01_train_classifier.py "$@"
$PY scripts/02_profile.py "$@"
$PY scripts/03_build_masks.py "$@"
$PY scripts/04_benchmark.py "$@"
