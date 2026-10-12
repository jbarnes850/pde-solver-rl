"""Exploratory held-out-family transfer comparison per transfer-eval/transfer-plan.json."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
TRIV = json.loads((HERE / "offline/heldout-trivial-errors.json").read_text())


def fin(x):
    return x is not None and math.isfinite(x)


def strict(v):
    return v["validity"] == 1.0 and fin(v.get("trajectory_loss_reference_range")) and v["trajectory_loss_reference_range"] <= 0.01


def load(run):
    out = {}
    for line in (HERE / "probe-results" / run / "groups.jsonl").read_text().splitlines():
        g = json.loads(line)
        for r in g["responses"]:
            assert hashlib.sha256(r["content"].encode()).hexdigest() == r["content_sha256"]
        out[g["physical_case_id"]] = {"family": g["family_id"], "n": len(g["responses"]),
            "strict": sum(strict(r["verifier"]) for r in g["responses"]),
            "valid": sum(r["verifier"]["validity"] == 1.0 for r in g["responses"]),
            "trunc": sum(r["finish_reason"] == "length" for r in g["responses"]),
            "seeds": [r["seed"] for r in g["responses"]]}
    return out


import sys
B = sys.argv[1] if len(sys.argv) > 1 else "calib-heldout-base"
T = sys.argv[2] if len(sys.argv) > 2 else "calib-heldout-trained32"
b, t = load(B), load(T)
assert set(b) == set(t) and all(b[c]["seeds"] == t[c]["seeds"] for c in b)
cases = sorted(c for c in b if TRIV[c] > 0.01)
fams = sorted({b[c]["family"] for c in cases})
by_f = {f: [c for c in cases if b[c]["family"] == f] for f in fams}
w = {c: 1.0 / (len(fams) * len(by_f[b[c]["family"]])) for c in cases}  # equal family weights
d = np.array([w[c] * (t[c]["strict"] / t[c]["n"] - b[c]["strict"] / b[c]["n"]) for c in cases])
obs = d.sum()
rng = np.random.default_rng(20261010)
nz = d[d != 0]
hits = sum(int(((rng.choice([-1.0, 1.0], size=(100000, len(nz))) * np.abs(nz)).sum(axis=1) >= obs - 1e-15).sum())
           for _ in range(20)) if len(nz) else 0
p = hits / 2_000_000 if len(nz) else 1.0
boots = []
for _ in range(10000):
    tot = 0.0
    for f, cs in by_f.items():
        pick = rng.choice(cs, size=len(cs), replace=True)
        tot += np.mean([t[c]["strict"] / t[c]["n"] - b[c]["strict"] / b[c]["n"] for c in pick]) / len(fams)
    boots.append(tot)


def eqf(arm, key):
    return float(np.mean([np.mean([arm[c][key] / arm[c]["n"] for c in cs]) for cs in by_f.values()]))


res = {"label": "exploratory transfer to unseen PDE families", "cases": len(cases), "families": len(fams),
       "base_eq_pass1": eqf(b, "strict"), "trained_eq_pass1": eqf(t, "strict"), "difference_pp": 100 * float(obs),
       "one_sided_p": p, "nonzero_weighted_cases": int(len(nz)),
       "bootstrap95_pp": [100 * float(np.quantile(boots, .025)), 100 * float(np.quantile(boots, .975))],
       "validity_change_pp": 100 * (eqf(t, "valid") - eqf(b, "valid")),
       "per_family": {f: {"cases": len(cs), "base_strict": sum(b[c]["strict"] for c in cs), "trained_strict": sum(t[c]["strict"] for c in cs),
                          "base_valid": sum(b[c]["valid"] for c in cs), "trained_valid": sum(t[c]["valid"] for c in cs),
                          "responses": sum(b[c]["n"] for c in cs), "base_trunc": sum(b[c]["trunc"] for c in cs),
                          "trained_trunc": sum(t[c]["trunc"] for c in cs)} for f, cs in by_f.items()}}
print(json.dumps(res, indent=1))
(HERE / f"transfer-eval/transfer-comparison-{T}-vs-{B}.json").write_text(json.dumps(res, indent=1, sort_keys=True))
