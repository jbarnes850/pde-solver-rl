"""Counterfactual: where would the old reward have put actor mass on the rollouts we trained on?

Reads verifier.jsonl exports (every training request logs both the original full reward and the
anchored reward). Groups are the 16 responses to one training case. For each reward we compute
MaxRL coefficients A = (R - mean) / (mean + eps) and report the share of positive actor mass on
valid outputs that do not beat the best trivial predictor (no-skill), as in the R13 analysis
(offline/transform_analysis.py), plus usable-group counts.

Usage: python reward_counterfactual.py OUT.json VERIFIER_JSONL [VERIFIER_JSONL ...]
Set RLVP_TRAINED_REWARD_MODE=full to read a run trained on the original reward (default anchored).
"""

import collections
import json
import os
import sys

STRICT = 0.01
ARMS = {"full_eps1e-6": ("full_reward", 1e-6), "anchored_eps1e-2": ("anchored_reward", 1e-2)}


def load(paths):
    groups = collections.defaultdict(list)
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r.get("request_id", "").startswith("parity") or r.get("reward_mode") != os.environ.get("RLVP_TRAINED_REWARD_MODE", "anchored"):
                continue
            groups[(path, r["case_id"])].append(r)
    return groups


def label(r):
    e, t = r.get("trajectory_loss_reference_range"), r.get("best_trivial_error")
    if not r.get("validity"):
        return "invalid"
    if e is not None and e <= STRICT:
        return "strict"
    if e is None or t is None or e >= t:
        return "noskill"
    return "skill"


def arm_stats(groups, key, eps):
    out = collections.Counter()
    mass = collections.Counter()
    for rows in groups.values():
        vals = [float(r[key]) for r in rows]
        m = sum(vals) / len(vals)
        if max(vals) == min(vals):
            continue
        out["nonconstant_groups"] += 1
        best = label(max(rows, key=lambda r: float(r[key])))
        out[f"groups_best_{best}"] += 1
        for r, v in zip(rows, vals):
            a = (v - m) / (m + eps)
            if a > 0:
                mass[label(r)] += a
    total = sum(mass.values())
    out["positive_mass_share"] = {k: round(v / total, 4) for k, v in sorted(mass.items())} if total else {}
    return out


def main():
    out_path, paths = sys.argv[1], sys.argv[2:]
    groups = load(paths)
    sizes = collections.Counter(len(v) for v in groups.values())
    labels = collections.Counter(label(r) for rows in groups.values() for r in rows)
    positive_full = collections.Counter(label(r) for rows in groups.values() for r in rows if float(r["full_reward"]) > 0)
    result = {
        "sources": paths,
        "groups": len(groups),
        "group_sizes": dict(sizes),
        "response_labels": dict(labels),
        "positive_full_reward_labels": dict(positive_full),
        "arms": {name: arm_stats(groups, key, eps) for name, (key, eps) in ARMS.items()},
    }
    json.dump(result, open(out_path, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
