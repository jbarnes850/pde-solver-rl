#!/usr/bin/env bash
# Launch one evaluation trial under a named Modal app, then export its arm to probe-results.
# Usage: run_eval_arm.sh APP RUN_ID VERIFIER_DIR
set -uo pipefail
APP="$1"; RUN="$2"; V="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="${RLVP_CAMPAIGN_DIR:-$(cd "$HERE/.." && pwd)}"
mkdir -p "$S/logs"
"$S/offline/launch_named.sh" "$APP" "$S/trials/$RUN.json" "$S/$V/verifier-connection.json" || { echo "EVAL FAILED $RUN"; exit 1; }
"$S/offline/export_probe_arm.sh" "$RUN" >> "$S/logs/probe-exports.jsonl"
echo "DONE $RUN"
