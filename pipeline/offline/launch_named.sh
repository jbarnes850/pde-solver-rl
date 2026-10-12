#!/usr/bin/env bash
# Launch a trial under an explicitly named Modal app so parallel runs never share an app name.
# Usage: launch_named.sh APP_NAME TRIAL_JSON VERIFIER_CONNECTION_JSON
APP="$1"; TRIAL="$2"; CONN="$3"
export RLVP_MODAL_APP_NAME="$APP"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_training.sh" "$TRIAL" "$CONN"
