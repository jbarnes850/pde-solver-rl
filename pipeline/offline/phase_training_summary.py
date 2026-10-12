"""Per-family training signal for one training phase, from the training verifier log.

Maps each training group (16 responses to one window case) to its update through the phase manifest
and reports per family: strict rate, valid rate and crash share for the first and last 16 updates, the
all-zero groups, and for Darcy the strict count by (grid, field kind), by prompt form, and over the last
16 updates, which sets the Darcy share for the next phase.
Usage: python phase_training_summary.py PHASE START_STEP VERIFIER_JSONL OUT.json
"""
import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))
PRIVATE = Path(os.environ.get("RLVP_PRIVATE_DIR", ROOT / ".artifacts/private")) / "rlvp-calibration-20261010"


def strict(r):
    e = r.get("trajectory_loss_reference_range")
    return r.get("validity") == 1.0 and e is not None and e <= 0.01


def main():
    phase, start, log, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    rows = [json.loads(l) for l in (HERE / f"train-inputs/curriculum{phase}/train-ordered-p2-new{phase}.jsonl").read_text().splitlines() if l.strip()]
    offset = (start * 8) % 512
    window = rows[offset:offset + 256]
    upd = {r["physical_case_id"]: start + 1 + k // 8 for k, r in enumerate(window)}
    meta = {r["physical_case_id"]: r for r in window}
    store = {}
    for path in sorted(PRIVATE.glob("rl_train_2026101*.cases.json")):
        store.update(json.loads(path.read_text())["cases"])
    groups = collections.defaultdict(list)
    for line in log.read_text().splitlines():
        r = json.loads(line)
        if r.get("request_id", "").startswith("parity") or r.get("reward_mode") != "anchored" or r["case_id"] not in upd:
            continue
        groups[r["case_id"]].append(r)
    mid = start + 16
    fam = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    darcy_strata, darcy_forms, darcy_last16 = collections.Counter(), collections.Counter(), [0, 0]
    allzero = collections.Counter()
    for c, rs in groups.items():
        f = meta[c]["family_id"]
        half = "first16" if upd[c] <= mid else "last16"
        k = sum(strict(r) for r in rs)
        fam[f][half]["responses"] += len(rs)
        fam[f][half]["strict"] += k
        fam[f][half]["valid"] += sum(r.get("validity") == 1.0 for r in rs)
        fam[f][half]["crash"] += sum(r.get("execution_error") == "solver_exit_1" for r in rs)
        fam[f][half]["timeout"] += sum(r.get("execution_error") == "solver_timeout" for r in rs)
        if max(float(r.get("anchored_reward") or 0) for r in rs) == 0:
            allzero[f] += 1
        if f == "darcy2d":
            case = store.get(c, {})
            darcy_strata[f"{(case.get('grid_shape') or ['?'])[0]}-{(case.get('parameters') or {}).get('field_kind')}"] += k
            darcy_forms[meta[c].get("prompt_form")] += k
            if half == "last16":
                darcy_last16[0] += k
                darcy_last16[1] += len(rs)
    summary = {f: {h: {"strict_rate": round(v["strict"] / v["responses"], 4), "valid_rate": round(v["valid"] / v["responses"], 4),
                       "crash_rate": round(v["crash"] / v["responses"], 4), "timeout_rate": round(v["timeout"] / v["responses"], 4),
                       "responses": v["responses"]} for h, v in halves.items()} for f, halves in fam.items()}
    g5 = darcy_last16[0] / darcy_last16[1] if darcy_last16[1] else None
    res = {"phase": phase, "start_step": start, "groups": len(groups), "families": summary, "all_zero_groups": dict(allzero),
           "darcy": {"strict_last16_rate": g5, "strict_last16": darcy_last16, "strict_by_grid_fieldkind": dict(darcy_strata),
                     "strict_by_prompt_form": dict(darcy_forms),
                     "gate_G5_rows_next_phase": None if g5 is None else (64 if g5 >= 0.03 else 48 if g5 >= 0.015 else 32)},
           "log": str(log)}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
