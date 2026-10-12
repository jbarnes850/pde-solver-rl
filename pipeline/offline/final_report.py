"""Final held-out leaderboard and paired tests for the step-256 policy against the frozen frontier.

Reads the frozen frontier responses (external-eval/, never modified) and our held-out arms from probe-results,
on the same 135 cases (the two trivially solved Gray-Scott cases excluded) with equal family weights, as in
offline/leaderboard.py (unchanged, part of the frontier freeze). Adds: our trajectory (base, 128, 160, 192, 224,
256), pass@k for our arms, case-paired sign-flip tests of step 256 against each frontier model and against
step 160 and base, each with and without darcy_reaction2d, and a deployable best-of-8: among each model's valid
answers, keep the one with the smallest PDE residual (computed from the answer and the equation alone, no
reference) and score that single pick. Applied identically to every model.
Writes final-report/leaderboard-final.json and prints markdown tables.
"""
import os
import json
import math
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(os.environ.get("RLVP_CAMPAIGN_DIR", Path(__file__).resolve().parents[1]))
TRIV = json.loads((HERE / "offline/heldout-trivial-errors.json").read_text())
TYPES = {
    "1D transport and diffusion": ["heat1d", "advection_diffusion1d", "advection_reaction_diffusion1d"],
    "1D higher-order": ["kdv1d", "cahn_hilliard1d"],
    "2D transport": ["advection2d", "advection_diffusion2d"],
    "2D pattern formation": ["allen_cahn2d", "gray_scott2d"],
    "2D elliptic": ["darcy_reaction2d"],
}
OURS = [("Base (Qwen3.5-9B)", "calib-heldout-base"), ("Step 32", "calib-heldout-trained32"), ("Step 64", "calib-heldout-cont64"),
        ("Step 96", "calib-heldout-ext96"), ("Step 128", "calib-heldout-ext128"), ("Step 160", "calib-heldout-ext160"),
        ("Step 192", "calib-heldout-ext192"), ("Step 224", "calib-heldout-ext224"), ("Step 256 (final)", "calib-heldout-ext256")]
FRONTIER = [("Qwen3.8-Max", "heldout-qwen-qwen3-8-max-0902"), ("GLM-5.3", "heldout-z-ai-glm-5-3"),
            ("MiMo-V2.6-Pro", "heldout-xiaomi-mimo-v2-6-pro")]


def strict(v):
    e = v.get("trajectory_loss_reference_range")
    return v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def load_ours(run):
    p = HERE / "probe-results" / run / "groups.jsonl"
    if not p.exists():
        return None
    out = {}
    for line in p.read_text().splitlines():
        g = json.loads(line)
        out[g["physical_case_id"]] = {"family": g["family_id"], "s": [strict(r["verifier"]) for r in g["responses"]],
                                      "v": [r["verifier"].get("validity") == 1.0 for r in g["responses"]],
                                      "pick": residual_pick([r["verifier"] for r in g["responses"]])}
    return out


def load_frontier(run):
    out = {}
    for line in (HERE / "external-eval" / run / "responses.jsonl").read_text().splitlines():
        r = json.loads(line)
        c = out.setdefault(r["physical_case_id"], {"family": r["family_id"], "s": [], "v": []})
        c["s"].append(strict(r["verifier_strict"]))
        c["v"].append(r["verifier_strict"].get("validity") == 1.0)
        c.setdefault("_all", []).append(r["verifier_strict"])
    for c in out.values():
        c["pick"] = residual_pick(c.pop("_all"))
    return out


def residual_pick(verdicts):
    """1.0 if the valid answer with the smallest reference-free PDE residual is strict, else 0.0."""
    valid = [v for v in verdicts if v.get("validity") == 1.0 and v.get("candidate_residual") is not None
             and math.isfinite(v["candidate_residual"])]
    return [float(strict(min(valid, key=lambda v: v["candidate_residual"])))] if valid else [0.0]


def eqf(arm, fams, cases, key="s", k=None):
    per = []
    for f in fams:
        vals = []
        for c in (c for c in cases if arm[c]["family"] == f):
            xs = arm[c][key]
            if k is None:
                vals.append(sum(xs) / len(xs))
            else:  # unbiased pass@k from n samples with s successes
                n, s = len(xs), sum(xs)
                vals.append(1.0 - (comb(n - s, k) / comb(n, k) if n - s >= k else 0.0))
        per.append(np.mean(vals))
    return float(np.mean(per))


def paired(a, b, fams, cases):
    """b minus a, equal-family weights, two-sided case-paired sign-flip p (2,000,000 flips, seed 20261012)."""
    cases = [c for c in cases if a[c]["family"] in fams]
    by_f = {f: [c for c in cases if a[c]["family"] == f] for f in fams}
    w = {c: 1.0 / (len(fams) * len(by_f[a[c]["family"]])) for c in cases}
    d = np.array([w[c] * (np.mean(b[c]["s"]) - np.mean(a[c]["s"])) for c in cases])
    obs, nz = d.sum(), d[d != 0]
    rng = np.random.default_rng(20261012)
    hits = sum(int(((rng.choice([-1.0, 1.0], size=(100000, len(nz))) * np.abs(nz)).sum(axis=1) >= abs(obs) - 1e-15).sum())
               for _ in range(20)) if len(nz) else 0
    return {"difference_pp": round(100 * float(obs), 2), "two_sided_p": min(1.0, 2 * hits / 2_000_000) if len(nz) else 1.0}


def main():
    arms = {name: load_ours(run) for name, run in OURS}
    arms = {k: v for k, v in arms.items() if v}
    arms.update({name: load_frontier(run) for name, run in FRONTIER})
    base = arms["Base (Qwen3.5-9B)"]
    cases = sorted(c for c in base if TRIV[c] > 0.01)
    fams = sorted({base[c]["family"] for c in cases})
    no_darcy = [f for f in fams if f != "darcy_reaction2d"]
    rows = {}
    for name, arm in arms.items():
        rows[name] = {"overall": eqf(arm, fams, cases), "overall_without_darcy_reaction2d": eqf(arm, no_darcy, cases),
                      "validity": eqf(arm, fams, cases, key="v"),
                      "best_of_8_by_residual": eqf(arm, fams, cases, key="pick"),
                      "best_of_8_by_residual_without_darcy_reaction2d": eqf(arm, no_darcy, cases, key="pick"),
                      "by_type": {t: eqf(arm, fs, cases) for t, fs in TYPES.items()},
                      "by_family": {f: eqf(arm, [f], cases) for f in fams}}
        if name in dict(OURS):
            rows[name]["pass_at_k"] = {k: eqf(arm, fams, cases, k=k) for k in (1, 2, 4, 8)}
    final = "Step 256 (final)"
    tests = {}
    if final in arms:
        for other in ["Base (Qwen3.5-9B)", "Step 160"] + [n for n, _ in FRONTIER]:
            tests[other] = {"all_10_families": paired(arms[other], arms[final], fams, cases),
                            "without_darcy_reaction2d": paired(arms[other], arms[final], no_darcy, cases)}
    res = {"cases": len(cases), "families": fams, "task_types": TYPES, "rows": rows, "step256_minus_other": tests,
           "frontier_source": "external-eval/ (frozen; external-eval/frontier-freeze.json)",
           "budget_note": "frontier models had up to 32k output tokens with reasoning; ours answers in 9.2k tokens without a thinking mode"}
    out = HERE / "final-report"
    out.mkdir(exist_ok=True)
    (out / "leaderboard-final.json").write_text(json.dumps(res, indent=1))
    order = sorted(rows, key=lambda k: -rows[k]["overall"])
    print("| Model | Overall | Without Darcy-reaction | Best of 8 by residual | " + " | ".join(TYPES) + " |")
    print("|---|" + "---:|" * (3 + len(TYPES)))
    for k in order:
        r = rows[k]
        print(f"| {k} | {100*r['overall']:.1f} | {100*r['overall_without_darcy_reaction2d']:.1f} | {100*r['best_of_8_by_residual']:.1f} | "
              + " | ".join(f"{100*r['by_type'][t]:.1f}" for t in TYPES) + " |")
    print()
    for k, v in tests.items():
        print(f"step 256 minus {k}: all {v['all_10_families']}  without Darcy-reaction {v['without_darcy_reaction2d']}")
    for name, _ in OURS:
        if name in rows:
            print(name, "pass@k", {k: round(100 * x, 1) for k, x in rows[name]["pass_at_k"].items()})


if __name__ == "__main__":
    main()
