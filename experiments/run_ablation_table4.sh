#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m experiments.run_section table4 "$@"
