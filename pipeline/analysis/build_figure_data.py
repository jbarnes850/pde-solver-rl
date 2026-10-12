"""Assemble every number the write-up figures use into one JSON file (RLVP_FIGURE_DATA).

Sources (all produced by the campaign, read-only here):
  final-report/leaderboard-final.json   held-out leaderboard, trajectory, by task type, residual best-of-8
  final-report/passk-large.json         64-sample pass@k curves for the untrained and final models
  probe-results/calib-heldoutk64-*      per-response verdicts, for residual-picked accuracy versus k
  final-eval/compare-*.json             fresh-set first-try accuracy per checkpoint
  offline/story/story-data.json         training reward per update, phases, fixed-mix reward, method shares
Figures must read from the output file only. Missing values are written as null.
"""
import os
import json
import math
import random
from pathlib import Path

import numpy as np

C = Path(os.environ.get("RLVP_CAMPAIGN_DIR", Path(__file__).resolve().parents[1]))
OUT = Path(os.environ.get("RLVP_FIGURE_DATA", "figure-data.json"))
TRIV = json.loads((C / "offline/heldout-trivial-errors.json").read_text())


def load_json(path):
    p = C / path
    return json.loads(p.read_text()) if p.exists() else None


def strict(v):
    e = v.get("trajectory_loss_reference_range")
    return v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def residual_curve(run, ks=(1, 2, 4, 8, 16, 32, 64), reps=40):
    p = C / "probe-results" / run / "groups.jsonl"
    if not p.exists():
        return None
    groups = {}
    for line in p.read_text().splitlines():
        g = json.loads(line)
        if TRIV.get(g["physical_case_id"], 1) > 0.01:
            groups[g["physical_case_id"]] = (g["family_id"], [r["verifier"] for r in g["responses"]])
    fams = sorted({f for f, _ in groups.values()})
    rng = random.Random(20261012)

    def pick(vs):
        valid = [v for v in vs if v.get("validity") == 1.0 and v.get("candidate_residual") is not None
                 and math.isfinite(v["candidate_residual"])]
        return float(strict(min(valid, key=lambda v: v["candidate_residual"]))) if valid else 0.0

    out = {}
    for k in ks:
        per_f = []
        for f in fams:
            vals = []
            for fam, vs in groups.values():
                if fam != f or len(vs) < k:
                    continue
                vals.append(pick(vs) if k == len(vs) else np.mean([pick(rng.sample(vs, k)) for _ in range(reps)]))
            per_f.append(np.mean(vals))
        out[str(k)] = float(np.mean(per_f))
    return out


def median_tokens_ours(run):
    p = C / "probe-results" / run / "groups.jsonl"
    if not p.exists():
        return None
    lens = []
    for line in p.read_text().splitlines():
        g = json.loads(line)
        if TRIV.get(g["physical_case_id"], 1) > 0.01:
            lens += [len(r.get("token_ids") or []) for r in g["responses"]]
    return float(np.median(lens)) if lens else None


def main():
    lb = load_json("final-report/leaderboard-final.json")
    budget = load_json("external-eval/budget-check.json") or {}
    pk = load_json("final-report/passk-large.json")
    story = json.loads((C / "offline/story/story-data.json").read_text())
    fresh = {}
    for step, run in ((0, "calib-fresh-base"), (32, "calib-fresh-trained32"), (64, "calib-fresh-cont64"), (96, "calib-fresh-ext96"),
                      (128, "calib-fresh-ext128"), (160, "calib-fresh-ext160"), (192, "calib-fresh-ext192"),
                      (224, "calib-fresh-ext224"), (256, "calib-fresh-ext256")):
        cmp = None
        for path in sorted((C / "final-eval").glob(f"compare-{run}-vs-calib-fresh-base.json")):
            cmp = json.loads(path.read_text())
        fresh[str(step)] = None if cmp is None else cmp["primary"]["trained_pass1"]
        if cmp is not None:
            fresh["0"] = cmp["primary"]["base_pass1"]  # the untrained model, from the same comparison files
    data = {
        "_about": "All blog figure data. Accuracy values are fractions. Equal family weights. Held-out = 135 problems from ten equation families never used in training.",
        "heldout_trajectory": None if lb is None else {name: row["overall"] for name, row in lb["rows"].items()},
        "heldout_by_type": None if lb is None else {name: row["by_type"] for name, row in lb["rows"].items()},
        "heldout_without_darcy_reaction": None if lb is None else {name: row["overall_without_darcy_reaction2d"] for name, row in lb["rows"].items()},
        "heldout_residual_best_of_8": None if lb is None else {name: row["best_of_8_by_residual"] for name, row in lb["rows"].items()},
        "heldout_pass_at_k_ours": None if lb is None else {name: row.get("pass_at_k") for name, row in lb["rows"].items() if row.get("pass_at_k")},
        "paired_tests_step256": None if lb is None else lb["step256_minus_other"],
        "fresh_trajectory": fresh,
        "passk_64": None if pk is None else pk["passk_equal_family"],
        "solved_in_64": None if pk is None else pk["solved_in_64"],
        "residual_pick_vs_k": {"untrained": residual_curve("calib-heldoutk64-base"), "final": residual_curve("calib-heldoutk64-s256d")},
        "training_reward_per_update": story["training"]["per_update"],
        "training_phases": story["training"]["phases"],
        "fixed_mix_reward": story["difficulty_controlled"]["fixed_mix_training_reward"],
        "solvable_states": story["solvable_sets"],
        "tokens_per_answer_median": {"Step 256 (final)": median_tokens_ours("calib-heldout-ext256"),
                                     "Base (Qwen3.5-9B)": median_tokens_ours("calib-heldout-base"),
                                     **{k: v.get("median_completion_tokens") for k, v in budget.items()}},
        "model_size_note": "Our model has 9B parameters. The frontier models are far larger open-weight models.",
        "frontier_budget_note": "Frontier models had up to 32,000 output tokens with reasoning. Ours answers once in about 9,000 tokens without a thinking mode.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1))
    print("wrote", OUT, {k: (v is not None) for k, v in data.items() if not k.startswith("_")})


if __name__ == "__main__":
    main()
