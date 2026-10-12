"""How much of each family is solvable at the 1% bar by a competent solver that is not the reference?

For each trained family, every solver-bank method (the teacher implementations) runs on training cases
and is scored against reference_primary with the verifier's trajectory loss. Reported per family: share
of cases where at least one method passes, share of all runs that pass, and median error. Darcy bank
methods share the reference's arithmetic face averaging, so Darcy's number is an upper bound.
Usage: python family_ceiling.py OUT.json [CASES_PER_FAMILY]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("RLVP_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
from rlvp.pdes.registry import get_environment  # noqa: E402
from rlvp.rewards import normalized_trajectory_loss  # noqa: E402
from rlvp.solver_bank.catalog import methods_for_family  # noqa: E402
from rlvp.solver_bank.render import render_teacher  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
FAMILIES = ["advection1d", "burgers1d", "reaction_diffusion1d", "diffusion_sorption1d", "reaction_diffusion2d", "darcy2d"]


def main():
    out_path = Path(sys.argv[1]); per = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    store = json.loads((HERE / "verifier-bundle/cases.json").read_text()); store = store.get("cases", store)
    summary, rows = {}, []
    for fam in FAMILIES:
        env = get_environment(fam)
        cases = sorted((env.case_from_dict(v) for v in store.values() if v.get("family_id") == fam), key=lambda c: c.case_id)
        step = max(1, len(cases) // per)
        cases = cases[::step][:per]
        solvers = {}
        for m in methods_for_family(fam):
            scope = {}
            try:
                exec(compile(render_teacher(m.method_id), m.method_id, "exec"), scope)
                solvers[m.method_id] = scope["solver"]
            except Exception as exc:
                print("render failed", m.method_id, type(exc).__name__)
        any_pass, all_runs = [], []
        for case in cases:
            ref = env.reference_primary(case)
            inv = env.invocation(case)
            args = [inv.arrays[k] if k in inv.arrays else inv.scalars[k] for k in inv.argument_order]
            errs = []
            for m, fn in solvers.items():
                try:
                    out = np.asarray(fn(*args), dtype=np.float64)
                    if out.shape == ref.shape and np.all(np.isfinite(out)):
                        errs.append(normalized_trajectory_loss(out, ref))
                except Exception:
                    pass
            any_pass.append(any(e <= 0.01 for e in errs))
            all_runs += errs
            rows.append({"family": fam, "case": case.case_id, "errors": errs})
        summary[fam] = {"cases": len(cases), "methods": len(solvers),
                        "cases_with_a_passing_method": sum(any_pass), "runs": len(all_runs),
                        "runs_passing": sum(e <= 0.01 for e in all_runs),
                        "median_error": float(np.median(all_runs)) if all_runs else None}
        print(fam, json.dumps(summary[fam]), flush=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
