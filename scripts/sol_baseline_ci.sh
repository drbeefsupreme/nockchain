#!/usr/bin/env bash
set -euo pipefail

##############################################################################
# sol_baseline_ci.sh — Thin CI wrapper for sol_baseline_run.sh
#
# Prepends --allow-dirty (CI checkout may modify working tree) and forwards
# all arguments to the core runner script. CI and local use the same core
# script for parity.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/sol_baseline_run.sh" --allow-dirty "$@"
