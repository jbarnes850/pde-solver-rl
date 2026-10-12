"""Write the training trial for one phase from the previous phase's trial.

Only the phase-specific fields change: run id, start step and horizon, parent checkpoint and its frozen
weights hash, manifest and schedule paths and hashes, verifier bundle hash and solver digest. The recipe
fields (reward, MaxRL eps, KL reference, LR schedule, G16, concurrency) are copied unchanged; the
pre-approved contingencies are applied only when passed explicitly.
Usage: python make_phase_trial.py PREV_TRIAL PHASE START_STEP PARENT_WEIGHTS_SHA256 BUNDLE_CASES_SHA256 SOLVER_DIGEST [--skip-warmup]
C2 (training temperature) is not supported here: gpu_runtime.py fixes temperature 0.7 for rollouts and the
behavior-logprob path, so it needs a reviewed runtime change if its trigger ever fires.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main():
    prev, phase, start, parent_sha, cases_sha, digest = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    flags = sys.argv[7:]
    t = json.loads(Path(prev).read_text())
    assert t["steps"] == start, ("previous trial must end at this phase's start", t["steps"], start)
    cur = HERE / f"train-inputs/curriculum{phase}"
    receipt = json.loads((cur / "curriculum-receipt.json").read_text())
    manifest_sha = hashlib.sha256((cur / f"train-ordered-p2-new{phase}.jsonl").read_bytes()).hexdigest()
    schedule_sha = hashlib.sha256((cur / "schedule/schedule.jsonl").read_bytes()).hexdigest()
    assert manifest_sha == receipt["manifest_sha256"] and schedule_sha == receipt["schedule_sha256"]
    t.update({"run_id": f"calib-train-base-p2-anchored-ext{phase}",
              "maxrl_start_step": start, "steps": start + (32 if start + 32 <= 288 else 288 - start),
              "resume_checkpoint": f"/runs/{t['run_id']}/checkpoints/grpo/step_{start}",
              "expected_parent_weights_sha256": parent_sha,
              "expected_train_manifest_sha256": manifest_sha, "expected_schedule_sha256": schedule_sha,
              "manifest_path": f"/inputs/calib-20261010/train/train-ordered-p2-new{phase}.jsonl",
              "schedule_path": f"/inputs/calib-20261010/train/schedule-p2-new{phase}.jsonl",
              "case_store_sha256": cases_sha, "solver_image_digest": digest})
    if "--skip-warmup" in flags:  # C1: keep the 1e-6 qualification update, no warmup to the target LR
        t["maxrl_warmup_updates"] = 0
    unknown = [f for f in flags if f != "--skip-warmup"]
    assert not unknown, f"unsupported flags {unknown}"
    out = HERE / f"trials/{t['run_id']}.json"
    assert not out.exists(), f"{out} exists; run ids are single-use"
    out.write_text(json.dumps(t, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in t.items() if k not in ("system_prompt", "algorithm_overrides")}, indent=1))


if __name__ == "__main__":
    main()
