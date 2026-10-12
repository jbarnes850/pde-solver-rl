"""Is the near-inviscid Burgers regime solvable at 1% by correct independent schemes?

Training cases only (verifier-bundle), grouped by viscosity. Each of the solver-bank Burgers methods is
run on the case inputs and scored against reference_primary with the verifier's trajectory loss, as
in ns2d_ceiling.py.
Usage: python burgers_ceiling.py OUT.json [CASES_PER_NU]
"""
import os
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("RLVP_SRC", str(Path(__file__).resolve().parents[2] / "src")))
from rlvp.pdes.registry import get_environment  # noqa: E402
from rlvp.rewards import normalized_trajectory_loss  # noqa: E402
from rlvp.solver_bank import burgers1d as bank  # noqa: E402

HERE = Path(os.environ.get("RLVP_CAMPAIGN_DIR", Path(__file__).resolve().parents[1]))


def main():
    out_path = Path(sys.argv[1]); per_nu = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    env = get_environment("burgers1d")
    store = json.loads((HERE / "verifier-bundle/cases.json").read_text()); store = store.get("cases", store)
    cases = sorted((env.case_from_dict(v) for v in store.values() if v.get("family_id") == "burgers1d"), key=lambda c: c.case_id)
    by_nu = collections.defaultdict(list)
    for c in cases:
        by_nu[float(c.parameters["nu"])].append(c)
    solvers = {}
    for m in bank.METHOD_IDS:
        scope = {}
        try:
            exec(compile(bank.render(m), m, "exec"), scope)
            solvers[m] = scope["solver"]
        except Exception as exc:
            print("render failed", m, exc)
    rows = []
    for nu, cs in sorted(by_nu.items()):
        for case in cs[:per_nu]:
            ref = env.reference_primary(case)
            inv = env.invocation(case)
            args = [inv.arrays[k] if k in inv.arrays else inv.scalars[k] for k in inv.argument_order]
            row = {"case": case.case_id, "nu": nu}
            for m, fn in solvers.items():
                try:
                    out = np.asarray(fn(*args), dtype=np.float64)
                    row[m] = normalized_trajectory_loss(out, ref) if out.shape == ref.shape and np.all(np.isfinite(out)) else None
                except Exception:
                    row[m] = None
            rows.append(row)
            vals = [v for k, v in row.items() if k not in ("case", "nu") and v is not None]
            print(json.dumps({"case": case.case_id, "nu": nu, "pass": sum(v <= 0.01 for v in vals), "of": len(vals),
                              "median": float(np.median(vals)) if vals else None}), flush=True)
    summary = {}
    for nu in sorted(by_nu):
        vals = [v for r in rows if r["nu"] == nu for k, v in r.items() if k not in ("case", "nu") and v is not None]
        summary[str(nu)] = {"runs": len(vals), "pass_at_1pct": sum(v <= 0.01 for v in vals),
                            "median": float(np.median(vals)) if vals else None}
    out_path.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
