#!/bin/sh
set -e
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e "$ROOT"
echo "Weight-tune venv ready. Activate with:"
echo "  source $PWD/.venv/bin/activate"
