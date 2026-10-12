"""Compact per-attempt records for one training phase from the run's own rollout results.

Used for phase C (updates 225-256), whose training verifier expired before its terminal export
(scale288-amendments.json). Reads training-rollouts-ext256/rollout_results/*/ses_*.json and writes
training-rollouts-ext256/records.jsonl with case_id, dataset row, weight version, anchored reward
and strict (anchored reward == 1.0, exact under the anchored reward definition in rlvp.rewards).
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import C  # noqa: E402

SRC = C / "training-rollouts-ext256"


def main():
    out = []
    for path in sorted(glob.glob(str(SRC / "rollout_results" / "*" / "ses_*.json"))):
        d = json.loads(Path(path).read_text())
        meta = d["trajectory"]["metadata"]
        tm = meta["task_metadata"]
        reward = meta["evaluation"]["verifier_result"]["reward"]
        out.append({"request_id": "rlvp-rollout-" + d["session_id"], "reward_mode": "anchored", "source": "rollout_results",
                    "case_id": tm["task_name"], "row": tm["attempt_row_idx"], "group_id": tm["group_id"],
                    "weight_version": tm["weight_version"], "status": d["status"],
                    "anchored_reward": reward, "strict": reward is not None and reward >= 1.0})
    (SRC / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    print(len(out), "records;", sum(r["strict"] for r in out), "strict;", sum(r["status"] != "COMPLETED" for r in out), "not completed")


if __name__ == "__main__":
    main()
