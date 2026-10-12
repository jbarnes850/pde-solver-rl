"""Behavior over training on the fixed eval sets: method choice, numerical-judgment tags, length, deliberation,
crash types and truncation, per (set, checkpoint, family). Static regex labels on comment-stripped,
verifier-extracted code (extends offline/behavior_analysis.py and the trace-comparison classifier).
Writes work/behavior.json and work/labels-<set>-<ck>.jsonl (per-response label rows).
"""
import collections
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import W, available, load  # noqa: E402

LINEAR = {"advection1d", "heat1d", "advection_diffusion1d", "advection_diffusion2d", "advection2d"}
STIFF_NONLINEAR = {"cahn_hilliard1d", "kdv1d", "gray_scott2d", "allen_cahn2d"}
R = {
    "fft": r"\bfft\w*\(|np\.fft\.|scipy\.fft|\brfft\w*\(|\bfftn\(|\bfft2\(",
    "etdrk4": r"ETDRK4|etdrk4|\(-4\s*-\s*LR|LR\s*\*\*\s*3|\bLR\b",
    "phi": r"expm1|phi1|phi_1|\bphi\(|_phi\w*\(|ETD|etd|\(\s*np\.exp\([^)]*\)\s*-\s*1(\.0)?\s*\)\s*/|\(\s*E\w*\s*-\s*1(\.0)?\s*\)\s*/",
    "rk4": r"[Rr][Kk]4|[Rr]unge",
    "semi_imp": r"/\s*\(\s*1(\.0*)?\s*[+\-][^\n]*(dt|\bh\b|tau|step)|\bdenom\w*\s*=\s*\(?\s*1(\.0*)?\s*[+\-]|/\s*denom|/\s*\w*(lhs|LHS|implicit|imp)\w*",
    "solve_ivp": r"solve_ivp|odeint",
    "implicit_fd": r"spsolve|splu|factorized|solve_banded|sparse\.linalg|np\.linalg\.solve|scipy\.linalg\.solve|lu_factor|lu_solve|\bcg\(|gmres|bicgstab|[Cc]rank|theta\s*=",
    "semi_lagr": r"map_coordinates|np\.interp\(|ndimage\.shift|RegularGridInterpolator|interp1d",
    "adaptive_dt": r"cfl|CFL|stab\w*|dt_max|max_dt|dt_lim\w*|dt_safe|safety|dt_stable|dt_crit|dt_nl|dt_adv|dt_diff|dt_cfl|dt_expl",
    "rtol": r"rtol\s*=",
    "max_step": r"max_step\s*=",
    "dealias": r"dealias|2\.0?\s*/\s*3|N\w*\s*/\s*3|//\s*3\b",
    "batch_loop": r"for\s+\w+\s+in\s+range\(\s*B\s*\)|for\s+b\w*\s+in\s+range",
    "while_or_sub": r"while\s|n_sub|nsub|substep|n_steps|num_steps|nsteps|n_inner|steps\s*=",
    "np_roll": r"np\.roll",
    "fd_symbol": r"np\.sin\([^)\n]*(dx|/\s*N|h\b)",
    "upwind": r"upwind|[Ff]lux|minmod|limiter|[Gg]odunov|[Ll]ax.?[Ww]endroff|WENO|weno|MUSCL",
}
RC = {k: re.compile(v) for k, v in R.items()}
RTOL = re.compile(r"rtol\s*=\s*([0-9.eE\-+]+)")


def strip(code):
    return "\n".join(l.split("#", 1)[0].rstrip() for l in code.splitlines())


def classify(code, family):
    s = strip(code)
    t = {k: bool(c.search(s)) for k, c in RC.items()}
    has = lambda n: bool(re.search(r"\bk_?%d\w*\s*[,=]" % n, s))
    t["rk_stage"] = has(1) and has(2)
    t["rk4"] = t["rk4"] or (has(1) and has(3) and has(4))
    t["exp"] = bool(re.search(r"np\.exp\(|\bexp\(", s))
    m = RTOL.search(s)
    try:
        t["rtol_value"] = float(m.group(1)) if m else None
    except ValueError:
        t["rtol_value"] = None
    lin = family in LINEAR
    if family in ("darcy2d", "darcy_reaction2d"):
        if re.search(r"spsolve|splu|factorized", s):
            p = "sparse_direct"
        elif re.search(r"\bcg\(|gmres|bicgstab|minres", s):
            p = "krylov"
        elif re.search(r"np\.linalg\.solve|scipy\.linalg\.solve|lu_factor", s):
            p = "dense_direct"
        elif re.search(r"while|for\s+\w+\s+in\s+range\(\s*(max_iter|n_iter|iters|\d{3,})", s):
            p = "stationary_iteration"
        else:
            p = "other"
        if not re.search(r"def\s+solver", s):
            p = "no_code"
        return p, t
    if not re.search(r"def\s+solver", s):
        p = "no_code"
    elif t["solve_ivp"]:
        mm = re.search(r"method\s*=\s*['\"](\w+)['\"]", s)
        p = "solve_ivp:" + (mm.group(1) if mm else ("LSODA" if "odeint" in s else "RK45"))
    elif t["fft"] and t["etdrk4"]:
        p = "etd_or_if"
    elif t["fft"] and t["phi"]:
        p = "etd_or_if"
    elif t["fft"] and t["exp"] and lin and not t["rk_stage"] and not t["fd_symbol"]:
        p = "exact_spectral"
    elif t["fft"] and t["exp"] and lin and t["fd_symbol"]:
        p = "spectral_fd_symbol"
    elif t["fft"] and t["exp"] and not lin and not t["semi_imp"]:
        p = "etd_or_if"
    elif t["semi_imp"] and t["fft"]:
        p = "semi_implicit_spectral"
    elif t["implicit_fd"] or t["semi_imp"]:
        p = "implicit_fd"
    elif t["semi_lagr"]:
        p = "semi_lagrangian"
    elif t["rk4"]:
        p = "explicit_rk4"
    elif t["while_or_sub"] or re.search(r"\bdt\b", s):
        p = "explicit_low_order"
    else:
        p = "other"
    return p, t


def group(p):
    if p.startswith("solve_ivp:"):
        return "solve_ivp_stiff" if p.split(":")[1] in ("BDF", "Radau", "LSODA") else "solve_ivp_nonstiff"
    return p


def med(x):
    return statistics.median(x) if x else None


def main():
    out = {}
    for s in ("fresh", "heldout"):
        out[s] = {}
        for ck in available(s):
            rows = load(s, ck)
            fam = collections.defaultdict(list)
            with open(W / f"labels-{s}-{ck}.jsonl", "w") as fh:
                for r in rows:
                    p, t = classify(r["code"], r["family"])
                    r["prim"], r["tags"] = p, t
                    fam[r["family"]].append(r)
                    fh.write(json.dumps({k: r[k] for k in ("case", "family", "idx", "outcome", "err", "exec_error", "ntok", "finish", "cand_res")}
                                        | {"prim": p, "tags": {k: v for k, v in t.items() if v}}) + "\n")
            res = {}
            for f, rs in sorted(fam.items()) + [("ALL", rows)]:
                n = len(rs)
                oc = collections.Counter(r["outcome"] for r in rs)
                prim = collections.Counter(group(r["prim"]) for r in rs)
                prim_raw = collections.Counter(r["prim"] for r in rs)
                prim_strict = collections.Counter(group(r["prim"]) for r in rs if r["outcome"] == "strict")
                comp = [r for r in rs if r["finish"] != "length"]
                ivp = [r for r in rs if r["tags"].get("solve_ivp")]
                expl = [r for r in rs if group(r["prim"]) in ("explicit_rk4", "explicit_low_order")]
                inv = [r for r in rs if r["outcome"] == "invalid"]
                res[f] = {
                    "n": n, "outcomes": dict(oc),
                    "strict_rate": round(oc["strict"] / n, 4), "truncation_rate": round(oc["truncated"] / n, 4),
                    "median_tokens": med([r["ntok"] for r in rs]),
                    "median_tokens_strict": med([r["ntok"] for r in rs if r["outcome"] == "strict"]),
                    "median_tokens_fail": med([r["ntok"] for r in rs if r["outcome"] != "strict"]),
                    "median_tokens_completed": med([r["ntok"] for r in comp]),
                    "median_code_lines_strict": med([r["noncomment_lines"] for r in rs if r["outcome"] == "strict"]),
                    "median_code_lines": med([r["noncomment_lines"] for r in comp]),
                    "comment_share_completed": round(statistics.fmean(r["comment_lines"] / max(1, r["code_lines"]) for r in comp), 3) if comp else None,
                    "delib_comment_lines_per_response": round(statistics.fmean(r["delib_comment_lines"] for r in rs), 2),
                    "delib_comment_lines_per_truncated": round(statistics.fmean(r["delib_comment_lines"] for r in rs if r["finish"] == "length"), 2) if any(r["finish"] == "length" for r in rs) else None,
                    "outside_code_share_completed": round(statistics.fmean(r["outside_chars"] / max(1, r["chars"]) for r in comp), 3) if comp else None,
                    "multi_fence_share": round(sum(r["fences"] >= 4 for r in rs) / n, 3),
                    "no_fence_share": round(sum(r["fences"] == 0 for r in rs) / n, 3),
                    "method_share": {k: round(v / n, 4) for k, v in prim.most_common()},
                    "method_counts_raw": dict(prim_raw.most_common()),
                    "method_strict_rate": {k: [prim_strict[k], prim[k]] for k in prim},
                    "tags": {
                        "fft": round(sum(r["tags"]["fft"] for r in rs) / n, 3),
                        "solve_ivp": round(len(ivp) / n, 3),
                        "solve_ivp_rtol_set": round(sum(r["tags"]["rtol"] for r in ivp) / len(ivp), 3) if ivp else None,
                        "solve_ivp_rtol_le_1e-6": round(sum((r["tags"]["rtol_value"] or 1) <= 1e-6 for r in ivp) / len(ivp), 3) if ivp else None,
                        "solve_ivp_max_step": round(sum(r["tags"]["max_step"] for r in ivp) / len(ivp), 3) if ivp else None,
                        "explicit_with_stability_dt": round(sum(r["tags"]["adaptive_dt"] for r in expl) / len(expl), 3) if expl else None,
                        "explicit_share": round(len(expl) / n, 3),
                        "batch_loop": round(sum(r["tags"]["batch_loop"] for r in rs) / n, 3),
                        "dealias": round(sum(r["tags"]["dealias"] for r in rs) / n, 3),
                        "fd_symbol": round(sum(r["tags"]["fd_symbol"] for r in rs) / n, 3),
                    },
                    "invalid_exec_errors": dict(collections.Counter(r["exec_error"] for r in inv)),
                }
            out[s][ck] = res
            a = res["ALL"]
            print(s, ck, "trunc", a["truncation_rate"], "medtok", a["median_tokens"], a["median_tokens_strict"], a["median_tokens_fail"],
                  "comment", a["comment_share_completed"], "delib", a["delib_comment_lines_per_response"], "outside", a["outside_code_share_completed"])
    (W / "behavior.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
