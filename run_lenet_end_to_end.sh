#!/usr/bin/env bash
# Fresh committee recovery, automatic C5/F6 recovery, head fit, and validation.
# Usage: ./run_lenet_end_to_end.sh --device cuda:0 --output runs/lenet_fresh
set -euo pipefail
cd "$(dirname "$0")"
exec python -u lenet_hybrid.py "$@"
