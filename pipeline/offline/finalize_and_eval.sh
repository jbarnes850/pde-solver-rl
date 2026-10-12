#!/usr/bin/env bash
# After a training run completes: hash its final checkpoint (two independent downloads), write a
# v2 freeze before any outcome exists, upload it, write the sealed fresh-test trial, launch the
# evaluation under its own app name, and export the arm.
# Usage: finalize_and_eval.sh TRAIN_RUN STEP EVAL_RUN APP CONTRACT_YAML [MAX_MODEL_LEN]
# Optional env: VERIFIER_DIR (default verifier-sealed2), EVIDENCE (default night-plan-20261011.json), LABEL,
# FREEZE_TEMPLATE (default final-eval/selection-freeze.json; transfer-eval/selection-freeze.json for held-out),
# TRIAL_TEMPLATE (default trials/calib-fresh-trained32.json; trials/calib-heldout-ext160.json for held-out),
# VERIFIER_SOURCE_TREE (only after a deliberate, re-qualified verifier source change such as the step-224 merge).
# Paths and Modal names: RLVP_ROOT, RLVP_CAMPAIGN_DIR, RLVP_PYTHON, MODAL_BIN, MODAL_PROFILE (passed through
# when set), RLVP_INPUTS_VOLUME, RLVP_RUNS_VOLUME.
set -uo pipefail
TRAIN="$1"; STEP="$2"; EVAL="$3"; APP="$4"; CONTRACT="$5"; MML="${6:-9728}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RLVP_ROOT:-$(cd "$HERE/../.." && pwd)}"
S="${RLVP_CAMPAIGN_DIR:-$(cd "$HERE/.." && pwd)}"
M="${MODAL_BIN:-$ROOT/.venv-modal/bin/modal}"
PY="${RLVP_PYTHON:-$ROOT/.venv/bin/python}"
INPUTS_VOLUME="${RLVP_INPUTS_VOLUME:-rlvp-modal-20261009-inputs}"
RUNS_VOLUME="${RLVP_RUNS_VOLUME:-rlvp-modal-20261009-runs}"
export RLVP_CAMPAIGN_DIR="$S"
V=${VERIFIER_DIR:-verifier-sealed2}; EVIDENCE=${EVIDENCE:-night-plan-20261011.json}
FREEZE_TEMPLATE=${FREEZE_TEMPLATE:-final-eval/selection-freeze.json}; TRIAL_TEMPLATE=${TRIAL_TEMPLATE:-trials/calib-fresh-trained32.json}
export V EVIDENCE LABEL FREEZE_TEMPLATE TRIAL_TEMPLATE VERIFIER_SOURCE_TREE
until grep -q "\"$TRAIN\",\"end_utc\"" $S/logs/launch-ledger.jsonl; do sleep 30; done
grep "\"$TRAIN\",\"end_utc\"" $S/logs/launch-ledger.jsonl | grep -q '"exit":0' || { echo "TRAIN FAILED $TRAIN"; exit 1; }
CK=/$TRAIN/checkpoints/grpo/step_$STEP
OUT=$S/eval-$EVAL; mkdir -p $OUT/a $OUT/b
for d in a b; do "$M" volume get --force "$RUNS_VOLUME" $CK/policy/weights $OUT/$d/ >/dev/null 2>&1; done
H=$(cd "$ROOT" && "$PY" -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from rlvp.checkpoint_bootstrap import tree_sha256
a=tree_sha256(Path('$OUT/a/weights')); b=tree_sha256(Path('$OUT/b/weights'))
assert a==b, (a,b); print(a)") || { echo "HASH MISMATCH"; exit 1; }
python3 - "$TRAIN" "$STEP" "$EVAL" "$CONTRACT" "$H" "$MML" <<'PY'
import hashlib, json, os, sys, time, datetime
train, step, ev, contract, h, mml = sys.argv[1:]
S = os.environ["RLVP_CAMPAIGN_DIR"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
fz = json.load(open(f"{S}/" + os.environ["FREEZE_TEMPLATE"]))
fz.update({"frozen_unix": time.time(), "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "selected_checkpoint_tree_sha256": h, "evaluation_contract_sha256": sha(contract),
           "selection_evidence_sha256": sha(f"{S}/" + os.environ["EVIDENCE"]),
           "selected_checkpoint": f"/runs/{train}/checkpoints/grpo/step_{step}",
           "label": os.environ.get("LABEL") or "pre-specified overnight checkpoint measurement on the already-used fresh set"})
if os.environ.get("VERIFIER_SOURCE_TREE"):
    fz["verifier_source_tree_sha256"] = os.environ["VERIFIER_SOURCE_TREE"]
fp = f"{S}/eval-{ev}/selection-freeze.json"
open(fp, "w").write(json.dumps(fz, indent=1, sort_keys=True) + "\n")
t = json.load(open(f"{S}/" + os.environ["TRIAL_TEMPLATE"]))
t.update({"run_id": ev, "resume_checkpoint": f"/runs/{train}/checkpoints/grpo/step_{step}", "weight_version": int(step),
          "sealed_test_policy": "selected", "selection_freeze_path": f"/inputs/calib-20261010/overnight/{ev}/selection-freeze.json",
          "selection_freeze_sha256": sha(fp), "evaluation_config": f"/inputs/calib-20261010/overnight/{ev}/contract.yaml",
          "solver_image_digest": open(f"{S}/" + os.environ["V"] + "/solver-digest.txt").read().strip(), "max_seconds": 6000,
          "max_estimated_dollars": 29.3})
if mml != "9728":
    t["max_model_len"] = int(mml)
json.dump(t, open(f"{S}/trials/{ev}.json", "w"), indent=2, sort_keys=True)
print("freeze", sha(fp), "selected", h)
PY
"$M" volume put --force "$INPUTS_VOLUME" $OUT/selection-freeze.json /calib-20261010/overnight/$EVAL/selection-freeze.json >/dev/null
"$M" volume put --force "$INPUTS_VOLUME" $CONTRACT /calib-20261010/overnight/$EVAL/contract.yaml >/dev/null
$S/offline/launch_named.sh $APP $S/trials/$EVAL.json $S/$V/verifier-connection.json || { echo "EVAL FAILED $EVAL"; exit 1; }
$S/offline/export_probe_arm.sh $EVAL >> $S/logs/probe-exports.jsonl
echo "DONE $EVAL"
