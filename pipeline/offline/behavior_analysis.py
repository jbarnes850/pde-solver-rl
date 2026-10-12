"""How did the model's behavior change across checkpoints? Outcome mix, error, length and method choice.

For each evaluated checkpoint on the fresh test set and on the never-trained-on (transfer) set, every
stored response is classified as strict (valid and within 1%), valid but inaccurate, invalid
(completed but crashed, wrong shape or non-finite) or truncated (hit the token limit). Code features
are simple, transparent regex markers of the numerical approach, matched on code with comments removed. Per-case transitions compare the
base model with the latest checkpoint.
Usage: python behavior_analysis.py OUT.json
"""
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FRESH = [("base", "calib-fresh-base"), ("step32", "calib-fresh-trained32"), ("step64", "calib-fresh-cont64"),
         ("step96", "calib-fresh-ext96"), ("step128", "calib-fresh-ext128"), ("step160", "calib-fresh-ext160")]
TRANSFER = [("base", "calib-heldout-base"), ("step32", "calib-heldout-trained32"), ("step64", "calib-heldout-cont64"),
            ("step96", "calib-heldout-ext96"), ("step128", "calib-heldout-ext128"), ("step160", "calib-heldout-ext160")]
FEATURES = {
    "solve_ivp_or_odeint": r"solve_ivp|odeint",
    "sparse_linear_solve": r"spsolve|sparse\.linalg|splu|\bcg\(|bicgstab|gmres|factorized",
    "fft": r"np\.fft|scipy\.fft|\brfft|\bfft\(|fft2|fftn",
    "np_roll_stencil": r"np\.roll",
    "np_gradient": r"np\.gradient",
    "runge_kutta": r"\bk4\b|\brk4\b|RK4|runge|\bk1\b.*\bk2\b",
    "stability_step": r"cfl|CFL|stability|dt_max|dt_stable|max_dt",
    "implicit_scheme": r"implicit|crank|Crank|backward euler|theta\s*=",
}


def strict(v):
    e = v.get("trajectory_loss_reference_range")
    return v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def outcome(r):
    v = r["verifier"]
    if r.get("finish_reason") == "length":
        return "truncated"
    if v.get("validity") != 1.0:
        return "invalid"
    return "strict" if strict(v) else "valid_inaccurate"


def load(run):
    p = HERE / "probe-results" / run / "groups.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else None


def summarize(groups):
    fam = collections.defaultdict(lambda: collections.Counter())
    errs = collections.defaultdict(list)
    tokens = collections.defaultdict(list)
    comment = collections.defaultdict(list)
    feats = collections.defaultdict(lambda: collections.Counter())
    feats_strict = collections.defaultdict(lambda: collections.Counter())
    for g in groups:
        f = g["family_id"]
        for r in g["responses"]:
            o = outcome(r)
            fam[f][o] += 1
            fam[f]["n"] += 1
            v = r["verifier"]
            e = v.get("trajectory_loss_reference_range")
            if v.get("validity") == 1.0 and e is not None and math.isfinite(e):
                errs[f].append(e)
            tokens[f].append(len(r.get("token_ids") or []))
            lines = [x for x in r["content"].splitlines() if x.strip()]
            comment[f].append(sum(x.strip().startswith("#") for x in lines) / max(1, len(lines)))
            code = "\n".join(x.split("#", 1)[0] for x in r["content"].splitlines())  # ignore comments
            for name, pat in FEATURES.items():
                if re.search(pat, code):
                    feats[f][name] += 1
                    if o == "strict":
                        feats_strict[f][name] += 1
    out = {}
    for f in sorted(fam):
        n = fam[f]["n"]
        e = sorted(errs[f])
        out[f] = {"n": n, **{k: fam[f][k] for k in ("strict", "valid_inaccurate", "invalid", "truncated")},
                  "valid_error_quantiles": [e[int(q * (len(e) - 1))] for q in (0.1, 0.25, 0.5, 0.75)] if e else None,
                  "median_tokens": statistics.median(tokens[f]), "mean_comment_fraction": statistics.fmean(comment[f]),
                  "feature_share": {k: feats[f][k] / n for k in FEATURES},
                  "feature_share_among_strict": {k: (feats_strict[f][k] / fam[f]["strict"] if fam[f]["strict"] else None) for k in FEATURES}}
    return out


def transitions(base, last):
    b = {g["physical_case_id"]: sum(strict(r["verifier"]) for r in g["responses"]) for g in base}
    t = {g["physical_case_id"]: sum(strict(r["verifier"]) for r in g["responses"]) for g in last}
    fam = {g["physical_case_id"]: g["family_id"] for g in base}
    out = collections.defaultdict(lambda: collections.Counter())
    for c in b:
        key = ("both" if b[c] and t[c] else "newly_solved" if t[c] and not b[c] else
               "lost" if b[c] and not t[c] else "neither")
        out[fam[c]][key] += 1
    return {f: dict(v) for f, v in out.items()}


def main():
    res = {"fresh": {}, "transfer": {}, "transitions": {}}
    for setname, arms in (("fresh", FRESH), ("transfer", TRANSFER)):
        loaded = {}
        for label, run in arms:
            g = load(run)
            if g is None:
                print("missing", run)
                continue
            loaded[label] = g
            res[setname][label] = summarize(g)
        last = [k for k, _ in arms if k in loaded][-1]
        res["transitions"][setname] = {"base_vs": last, "by_family": transitions(loaded["base"], loaded[last])}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
