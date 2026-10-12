"""Large-k test of expansion versus sharpening on the frozen 135-case held-out set.

Reads the 64-sample arms (probe-results/calib-heldoutk64-base and calib-heldoutk64-s256d) and reports, with
equal family weights and the two trivially solved Gray-Scott cases excluded:
  - unbiased pass@k for k = 1, 2, 4, 8, 16, 32, 64 for each arm (Chen et al. estimator);
  - problems solved at least once in 64 by base only, by the trained model only, by both, by neither;
  - the same split per family, and the base's estimated per-problem success rate on problems only the
    trained model solves (an upper bound when base solved 0 of 64).
Writes final-report/passk-large.json.
"""
import os
import json
import math
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(os.environ.get("RLVP_CAMPAIGN_DIR", Path(__file__).resolve().parents[1]))
TRIV = json.loads((HERE / "offline/heldout-trivial-errors.json").read_text())
ARMS = {"base": "calib-heldoutk64-base", "step256": "calib-heldoutk64-s256d"}
KS = (1, 2, 4, 8, 16, 32, 64)


def strict(v):
    e = v.get("trajectory_loss_reference_range")
    return v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def load(run):
    out = {}
    for line in (HERE / "probe-results" / run / "groups.jsonl").read_text().splitlines():
        g = json.loads(line)
        out[g["physical_case_id"]] = {"family": g["family_id"], "n": len(g["responses"]),
                                      "s": sum(strict(r["verifier"]) for r in g["responses"])}
    return out


def passk(n, s, k):
    return 1.0 - (comb(n - s, k) / comb(n, k) if n - s >= k else 0.0)


def main():
    arms = {k: load(v) for k, v in ARMS.items()}
    base = arms["base"]
    cases = sorted(c for c in base if TRIV[c] > 0.01 and all(c in a for a in arms.values()))
    fams = sorted({base[c]["family"] for c in cases})
    by_f = {f: [c for c in cases if base[c]["family"] == f] for f in fams}
    curves = {name: {k: float(np.mean([np.mean([passk(a[c]["n"], a[c]["s"], k) for c in cs]) for cs in by_f.values()]))
                     for k in KS} for name, a in arms.items()}
    t = arms["step256"]
    split = {"both": [], "base_only": [], "trained_only": [], "neither": []}
    for c in cases:
        b, r = base[c]["s"] > 0, t[c]["s"] > 0
        split["both" if b and r else "base_only" if b else "trained_only" if r else "neither"].append(c)
    per_family = {f: {key: sum(1 for c in cs if c in split[key]) for key in split} for f, cs in by_f.items()}
    trained_only_rates = [t[c]["s"] / t[c]["n"] for c in split["trained_only"]]
    res = {"cases": len(cases), "samples_per_case": {k: int(np.median([a[c]["n"] for c in cases])) for k, a in arms.items()},
           "passk_equal_family": curves, "solved_in_64": {k: len(v) for k, v in split.items()},
           "per_family": per_family,
           "trained_only_cases": [{"case": c, "family": base[c]["family"], "trained_successes": t[c]["s"]} for c in split["trained_only"]],
           "base_only_cases": [{"case": c, "family": base[c]["family"], "base_successes": base[c]["s"]} for c in split["base_only"]],
           "trained_only_median_trained_rate": float(np.median(trained_only_rates)) if trained_only_rates else None,
           "note": "a problem the base solved 0 of 64 times has an estimated base success rate below about 4.6% (95% upper bound)"}
    out = HERE / "final-report"
    out.mkdir(exist_ok=True)
    (out / "passk-large.json").write_text(json.dumps(res, indent=1))
    print("k      " + "  ".join(f"{k:>6d}" for k in KS))
    for name, cur in curves.items():
        print(f"{name:7s}" + "  ".join(f"{100*cur[k]:6.1f}" for k in KS))
    print("solved at least once in 64:", res["solved_in_64"])
    for f, d in per_family.items():
        print(f"  {f:32s}", d)


if __name__ == "__main__":
    main()
