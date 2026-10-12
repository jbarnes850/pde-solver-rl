"""Freezes and trials for the final evaluation on the sealed fresh-v4 and held-out-v2 sets (step 160 vs 256).

The final checkpoint (step 256) was declared before any step-224 or step-256 outcome existed.
Each set gets one v2 freeze: baseline = step 160 (d9bcf4af...), selected = step 256. Freezes are written
before either set is opened; the sealed bundles are then prepared through sealed_eval_gate.py.
Usage: python prepare_final256.py freezes STEP256_TREE_SHA256
       python prepare_final256.py trials V4_SOLVER_DIGEST HELDOUT_V2_SOLVER_DIGEST
"""
import copy
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))
PRIVATE = Path(os.environ.get("RLVP_PRIVATE_DIR", ROOT / ".artifacts/private")) / "rlvp-calibration-20261010"
S160_TREE = "d9bcf4af588f9bfed5b804a07ba4c4eccd4beba63bd3c581557ba7844b09c039"
CK = {"s160": "/runs/calib-train-base-p2-anchored-ext160/checkpoints/grpo/step_160",
      "s256": "/runs/calib-train-base-p2-anchored-ext256/checkpoints/grpo/step_256"}
SETS = {
    "v4": {"dir": "final-eval-v4", "freeze": "selection-freeze-v4.json", "role": "rl_fresh_test_20261012v4", "count": 288,
           "contract": "final-eval-v4/fresh-test-v4-p2.yaml", "manifest": "sealed-fresh-v4-public/rl-fresh-test-20261012v4-p2.jsonl",
           "store": "rl_fresh_test_20261012v4.cases.json", "prereg": "final-eval-v4/v4-preregistration.json",
           "compare": "final-eval-v4/compare_v4.py", "inputs": "sealed-v4", "template": "trials/calib-freshv3-s160.json",
           "run": "calib-freshv4"},
    "hv2": {"dir": "heldout-v2-eval", "freeze": "selection-freeze-heldout-v2.json", "role": "held_out_evaluation_20261012v2", "count": 137,
            "contract": "heldout-v2-eval/heldout-v2-p1.yaml", "manifest": "sealed-heldout-v2-public/held-out-evaluation-20261012v2.jsonl",
            "store": "held_out_evaluation_20261012v2.cases.json", "prereg": "heldout-v2-eval/heldout-v2-preregistration.json",
            "compare": "heldout-v2-eval/compare_heldout_v2.py", "inputs": "sealed-heldout-v2", "template": "trials/calib-heldout-ext160.json",
            "run": "calib-heldoutv2"},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freezes(h256):
    base = json.loads((HERE / "final-eval/selection-freeze.json").read_text())
    out = {}
    for key, s in SETS.items():
        pre = json.loads((HERE / s["prereg"]).read_text())
        fz = {"schema": "rlvp.selected-policy-freeze.v2", "test_unopened_at_freeze": True,
              "sealed_role": s["role"], "sealed_case_count": s["count"],
              "baseline_checkpoint_tree_sha256": S160_TREE, "selected_checkpoint_tree_sha256": h256,
              "evaluation_contract_sha256": sha(HERE / s["contract"]), "public_manifest_sha256": sha(HERE / s["manifest"]),
              "private_case_store_sha256": sha(PRIVATE / s["store"]),
              "verifier_source_tree_sha256": base["verifier_source_tree_sha256"],
              "selection_evidence_sha256": sha(HERE / s["prereg"]), "comparison_script_sha256": sha(HERE / s["compare"]),
              "decision_target": pre["primary"], "secondary": pre["secondary"], "statistics_method": base["statistics_method"],
              "label": "final evaluation: step 160 (baseline) vs step 256 (selected, pre-declared final)",
              "frozen_unix": time.time(), "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        assert fz["private_case_store_sha256"] == pre["private_case_store_sha256"]
        assert fz["evaluation_contract_sha256"] == pre["evaluation_contract_sha256"]
        assert fz["comparison_script_sha256"] == pre["comparison_script_sha256"]
        path = HERE / s["dir"] / s["freeze"]
        path.write_text(json.dumps(fz, indent=1, sort_keys=True) + "\n")
        out[key] = sha(path)
    print(json.dumps(out))


def trials(d_v4, d_hv2):
    digests = {"v4": d_v4, "hv2": d_hv2}
    for key, s in SETS.items():
        t0 = json.loads((HERE / s["template"]).read_text())
        fsha = sha(HERE / s["dir"] / s["freeze"])
        for arm, policy, step in (("s160", "baseline", 160), ("s256", "selected", 256)):
            t = copy.deepcopy(t0)
            t.update({"run_id": f"{s['run']}-{arm}", "resume_checkpoint": CK[arm], "weight_version": step,
                      "sealed_test_policy": policy, "solver_image_digest": digests[key],
                      "case_store_sha256": sha(PRIVATE / s["store"]),
                      "evaluation_config": f"/inputs/calib-20261010/{s['inputs']}/{Path(s['contract']).name}",
                      "evaluation_manifest": f"/inputs/calib-20261010/{s['inputs']}/{Path(s['manifest']).name}",
                      "selection_freeze_path": f"/inputs/calib-20261010/{s['inputs']}/{s['freeze']}",
                      "selection_freeze_sha256": fsha, "max_seconds": 6000, "max_estimated_dollars": 29.3})
            (HERE / f"trials/{t['run_id']}.json").write_text(json.dumps(t, indent=2, sort_keys=True))
            print(t["run_id"], policy, t["evaluation_manifest"])


if __name__ == "__main__":
    {"freezes": lambda: freezes(sys.argv[2]), "trials": lambda: trials(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
