#!/usr/bin/env bash
# Launch one bounded GPU trial with the live entrypoint; record UTC start/end for the spend ledger.
# Usage: launch_training.sh TRIAL_JSON VERIFIER_CONNECTION_JSON
# Environment: RLVP_ROOT (repository root), RLVP_CAMPAIGN_DIR (campaign directory holding offline/,
# trials/ and logs/; defaults to the pipeline directory), MODAL_BIN, MODAL_PROFILE (passed through when set),
# RLVP_MODAL_APP_NAME, RLVP_INPUTS_VOLUME, RLVP_RUNS_VOLUME, RLVP_BRIDGE_DIR (read by scripts/modal/gpu_app.py).
set -uo pipefail
TRIAL="$1"; CONN="$2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RLVP_ROOT:-$(cd "$HERE/../.." && pwd)}"
S="${RLVP_CAMPAIGN_DIR:-$(cd "$HERE/.." && pwd)}"
M="${MODAL_BIN:-$ROOT/.venv-modal/bin/modal}"
NAME=$(basename "$TRIAL" .json)
mkdir -p "$S/logs"
echo "{\"run\":\"$NAME\",\"start_utc\":\"$(date -u +%FT%TZ)\"}" >> "$S/logs/launch-ledger.jsonl"
cd "$ROOT" && timeout 37000 "$M" run scripts/modal/gpu_app.py --config "$TRIAL" --verifier-connection "$CONN" > "$S/logs/$NAME.log" 2>&1
CODE=$?
echo "{\"run\":\"$NAME\",\"end_utc\":\"$(date -u +%FT%TZ)\",\"exit\":$CODE}" >> "$S/logs/launch-ledger.jsonl"
exit $CODE
