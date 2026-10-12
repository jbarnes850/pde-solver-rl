"""Generic fresh-test arm comparison (same statistics as the frozen compare_fresh.py): B versus A.

Primary (selection-freeze.json): six-family equal-weight strict pass@1 difference over the
confirmatory families (48 cases each), one-sided case-paired sign-flip test, exact enumeration
if <= 24 nonzero case differences else 2,000,000 Monte Carlo flips (seed 20261010); descriptive
95% family-stratified case bootstrap (10,000 draws). Target: >= +2.0 pp, p <= 0.05, validity
drop <= 2 pp. Diffusion-sorption (3 cases) and secondary metrics are descriptive.
"""
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CONF = ["advection1d", "burgers1d", "reaction_diffusion1d", "reaction_diffusion2d", "darcy2d", "incompressible_ns2d"]
FREEZE = json.loads((HERE / "final-eval/selection-freeze.json").read_text())


def fin(x):
    return x is not None and math.isfinite(x)


def strict(v):
    return v["validity"] == 1.0 and fin(v.get("trajectory_loss_reference_range")) and v["trajectory_loss_reference_range"] <= 0.01


def load(run):
    out = HERE / "probe-results" / run
    cfg = json.loads((out / "probe_config.json").read_text())
    groups = [json.loads(l) for l in (out / "groups.jsonl").read_text().splitlines() if l.strip()]
    assert cfg["manifest_sha256"] == FREEZE["public_manifest_sha256"], "manifest differs from freeze"
    assert len(groups) == FREEZE["sealed_case_count"]
    cases = {}
    for g in groups:
        rs = g["responses"]
        for r in rs:
            assert hashlib.sha256(r["content"].encode()).hexdigest() == r["content_sha256"]
            assert r.get("model_retries", 0) == 0 and r.get("verifier_retries", 0) == 0
        errs = [r["verifier"]["trajectory_loss_reference_range"] for r in rs if r["verifier"]["validity"] == 1.0]
        cases[g["physical_case_id"]] = {
            "family": g["family_id"], "n": len(rs), "strict": sum(strict(r["verifier"]) for r in rs),
            "valid": sum(r["verifier"]["validity"] == 1.0 for r in rs),
            "truncated": sum(r["finish_reason"] == "length" for r in rs),
            "full": float(np.mean([r["verifier"]["full_reward"] for r in rs])),
            "anchored": float(np.mean([r["verifier"].get("anchored_reward") or 0 for r in rs])),
            "tokens": float(np.mean([len(r["token_ids"]) for r in rs])),
            "valid_log10_errors": [math.log10(e) for e in errs if fin(e) and e > 0],
            "seeds": [r["seed"] for r in rs]}
    return cfg, cases


def family_mean(cases, key, fams):
    return float(np.mean([np.mean([c[key] / c["n"] for c in cases.values() if c["family"] == f]) for f in fams]))


def main():
    import sys
    cfg_b, base = load(sys.argv[1])
    cfg_t, trained = load(sys.argv[2])
    assert set(base) == set(trained)
    for cid in base:
        assert base[cid]["seeds"] == trained[cid]["seeds"], "response seeds differ between arms"
    assert cfg_b["system_prompt_sha256"] == cfg_t["system_prompt_sha256"]
    print(f"comparing {sys.argv[2]} (B) against {sys.argv[1]} (A); same cases, same seeds", file=sys.stderr)
    conf_cases = sorted(c for c in base if base[c]["family"] in CONF)
    d = np.array([trained[c]["strict"] / trained[c]["n"] - base[c]["strict"] / base[c]["n"] for c in conf_cases])
    # Equal cases per family => equal-family mean equals the case mean.
    per_family_counts = {f: sum(base[c]["family"] == f for c in conf_cases) for f in CONF}
    assert len(set(per_family_counts.values())) == 1, per_family_counts
    observed = float(d.mean())
    nz = d[d != 0]
    rng = np.random.default_rng(20261010)
    if len(nz) <= 24:
        sums = [sum(s * abs(x) for s, x in zip(signs, nz)) for signs in itertools.product((-1, 1), repeat=len(nz))]
        p = float(np.mean(np.array(sums) >= nz.sum() - 1e-12))
        method = f"exact enumeration over {len(nz)} nonzero cases"
    else:
        hits, total = 0, 0
        for _ in range(20):
            flips = rng.choice([-1.0, 1.0], size=(100000, len(nz)))
            hits += int(((flips * np.abs(nz)).sum(axis=1) >= nz.sum() - 1e-12).sum())
            total += 100000
        p = hits / total
        method = f"Monte Carlo 2,000,000 sign flips over {len(nz)} nonzero cases"
    boots = []
    by_f = {f: [i for i, c in enumerate(conf_cases) if base[c]["family"] == f] for f in CONF}
    for _ in range(10000):
        idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in by_f.values()])
        boots.append(d[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    validity_change = family_mean(trained, "valid", CONF) - family_mean(base, "valid", CONF)
    result = {
        "primary": {"base_pass1": family_mean(base, "strict", CONF), "trained_pass1": family_mean(trained, "strict", CONF),
                    "difference_pp": 100 * observed, "one_sided_p": p, "test": method,
                    "bootstrap95_pp": [100 * lo, 100 * hi], "cases": len(conf_cases), "nonzero_cases": int(len(nz)),
                    "validity_change_pp": 100 * validity_change,
                    "target_met": bool(100 * observed >= 2.0 and p <= 0.05 and 100 * validity_change >= -2.0)},
        "per_family": {}, "secondary": {}}
    for f in CONF + ["diffusion_sorption1d"]:
        cs = [c for c in base if base[c]["family"] == f]
        result["per_family"][f] = {
            "cases": len(cs),
            "base_strict": sum(base[c]["strict"] for c in cs), "trained_strict": sum(trained[c]["strict"] for c in cs),
            "base_valid": sum(base[c]["valid"] for c in cs), "trained_valid": sum(trained[c]["valid"] for c in cs),
            "responses": sum(base[c]["n"] for c in cs),
            "base_truncated": sum(base[c]["truncated"] for c in cs), "trained_truncated": sum(trained[c]["truncated"] for c in cs),
            "base_cases_any_strict": sum(base[c]["strict"] > 0 for c in cs), "trained_cases_any_strict": sum(trained[c]["strict"] > 0 for c in cs)}
    for key in ("full", "anchored", "tokens"):
        result["secondary"][f"mean_{key}"] = {"base": float(np.mean([base[c][key] for c in conf_cases])),
                                              "trained": float(np.mean([trained[c][key] for c in conf_cases]))}
    for name, arm in (("base", base), ("trained", trained)):
        errs = [e for c in conf_cases for e in arm[c]["valid_log10_errors"]]
        result["secondary"][f"{name}_valid_log10_nrmse_quantiles"] = [float(np.quantile(errs, q)) for q in (0.1, 0.25, 0.5, 0.75)] if errs else None
    print(json.dumps(result, indent=1))
    (HERE / f"final-eval/compare-{sys.argv[2]}-vs-{sys.argv[1]}.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
