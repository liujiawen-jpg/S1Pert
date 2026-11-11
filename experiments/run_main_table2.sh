#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m experiments.main_table2 "$@"
