"""Parity of the sealed verifier on fresh cases using input-only programs (no policy outputs).

Programs return zeros or the persisted initial state with the requested shape. Remote verifier
fields are compared with local scoring by the same library code on the same private cases.
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))  # repository root
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts/modal")]
from verifier_runtime import request  # noqa: E402
from rlvp.contracts import ExecutionResult  # noqa: E402
from rlvp.data import load_case_store  # noqa: E402
from rlvp.pdes.registry import environment_for_case  # noqa: E402
from rlvp.rewards import best_trivial_error, evaluate_case, trivial_predictions  # noqa: E402

conn = json.loads(Path(sys.argv[1]).read_text())
cases = load_case_store(Path(sys.argv[2]))
fams = {}
for cid in sorted(cases):
    fams.setdefault(cases[cid].family_id, cid)
chosen = list(fams.values())[:3]
PROGRAMS = {
    "zeros": "import numpy as np\ndef solver(*args):\n    a = np.asarray(args[0])\n    t = next((x for x in args if np.ndim(x) == 1 and len(x) > 1), None)\n"
             "    return np.zeros((a.shape[0], len(t), *a.shape[1:]) if t is not None else a.shape)\n",
}
rows, ok = [], True
for cid in chosen:
    case = cases[cid]
    env = environment_for_case(case)
    inv = env.invocation(case)
    best = best_trivial_error(env, case)
    local_out = trivial_predictions(inv)["zeros"]
    local = evaluate_case(ExecutionResult(executed=True, shape_valid=True, finite=True, output=local_out), env, case, best_trivial=best)
    remote = request(conn, f"/verify/full/{cid}", {"request_id": f"sealed-parity-{cid}",
                     "response": {"output": [{"type": "message", "content": [{"type": "output_text", "text": PROGRAMS["zeros"]}]}]}})
    same = (remote["validity"] == local.validity and
            math.isclose(remote["trajectory_loss_reference_range"], local.trajectory_loss_reference_range, rel_tol=1e-9) and
            math.isclose(remote["full_reward"], local.full_reward, rel_tol=1e-9, abs_tol=1e-300) and
            math.isclose(remote["best_trivial_error"], best, rel_tol=1e-9) and remote["anchored_reward"] == local.anchored_reward)
    ok &= same
    rows.append({"case": cid, "program": "zeros", "match": same, "remote_error": remote["execution_error"],
                 "e_remote": remote["trajectory_loss_reference_range"], "e_local": local.trajectory_loss_reference_range,
                 "anchored_remote": remote["anchored_reward"], "anchored_local": local.anchored_reward})
print(json.dumps({"passed": ok, "checks": rows}, indent=1))
