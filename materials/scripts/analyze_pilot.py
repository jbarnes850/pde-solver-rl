"""Summarize a generation run: pass rates, repair gains, failure taxonomy, cost, selectors.

Usage: python analyze_pilot.py <candidates.jsonl> [sealed_scores.jsonl] > summary.json
Contribution labels: producer = qwen3.5-9b/{base,adapter}; intervention = none (plain),
claude_method_hint (hint), claude_scheme_formula (scheme). Repairs use only analytic-check feedback.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

cands = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines()]
sealed = {}
if len(sys.argv) > 2 and Path(sys.argv[2]).exists():
    for l in Path(sys.argv[2]).read_text().splitlines():
        r = json.loads(l)
        sealed[r["id"]] = r


def taxonomy(c):
    ch = c["check"]
    if c.get("gen_error"):
        return "generation_error"
    if not c.get("code", "").strip():
        return "no_code"
    errs = " ".join(ch.get("errors", {}).values())
    if ch.get("all_passed"):
        return "passed"
    if "timeout" in errs:
        return "too_slow"
    for pat, name in ((r"broadcast|shape mismatch|could not be broadcast|shape \(", "array_shape"),
                      (r"IndexError|index .* out of bounds", "indexing"),
                      (r"NameError|is not defined", "undefined_name"),
                      (r"SyntaxError|IndentationError", "malformed_code"),
                      (r"AttributeError|has no attribute|unexpected keyword|TypeError", "api_misuse"),
                      (r"nonfinite", "nonfinite"), (r"exit_", "runtime_error")):
        if re.search(pat, errs):
            return name
    failed = [k for k, v in ch.get("passed", {}).items() if not v]
    if failed == ["linear"] or (failed and set(failed) <= {"linear", "equilibrium_x", "equilibrium_y"}):
        return "wrong_physics_or_accuracy"
    if "runtime" in failed and len(failed) == 1:
        return "too_slow"
    return "other_check_failure:" + ",".join(sorted(failed))


lineages = collections.defaultdict(list)
for c in cands:
    lineages[c["lineage"]].append(c)
summary = {"n_candidates": len(cands), "n_lineages": len(lineages), "cells": {}}
for lin, recs in lineages.items():
    recs.sort(key=lambda r: r["round"])
cells = collections.defaultdict(list)
for lin, recs in lineages.items():
    cells[(recs[0]["arm"], recs[0]["variant"])].append(recs)
for (arm, var), lins in sorted(cells.items()):
    n = len(lins)
    first_pass = [l[0]["check"].get("all_passed", False) for l in lins]
    any_pass_by_round = []
    for R in range(3):
        any_pass_by_round.append(float(np.mean([any(r["check"].get("all_passed", False) for r in l if r["round"] <= R) for l in lins])))
    tokens = sum(r["completion_tokens"] for l in lins for r in l)
    gens = sum(len(l) for l in lins)
    verified = sum(any(r["check"].get("all_passed", False) for r in l) for l in lins)
    sealed_first = [sealed[l[0]["id"]]["passed"] / sealed[l[0]["id"]]["n"] for l in lins if l[0]["id"] in sealed]
    summary["cells"][f"{arm}/{var}"] = {
        "lineages": n, "first_attempt_check_pass": float(np.mean(first_pass)),
        "lineage_pass_within_rounds": any_pass_by_round, "generations": gens,
        "completion_tokens": tokens, "tokens_per_verified": tokens / verified if verified else None,
        "taxonomy_first_attempt": dict(collections.Counter(taxonomy(l[0]) for l in lins)),
        "taxonomy_all": dict(collections.Counter(taxonomy(r) for l in lins for r in l)),
        "sealed_pass_rate_first_attempt_scored": float(np.mean(sealed_first)) if sealed_first else None,
    }
# Selector analysis on candidates with sealed scores
rows = []
for c in cands:
    if c["id"] in sealed:
        s = sealed[c["id"]]
        rows.append({"id": c["id"], "all_passed": c["check"].get("all_passed", False),
                     "n_passed": c["check"].get("n_passed", 0), "residual": c["check"].get("residual"),
                     "sealed_frac": s["passed"] / s["n"]})
if rows:
    full = [r for r in rows if r["sealed_frac"] == 1.0]
    summary["selector"] = {
        "scored": len(rows),
        "sealed_full_pass": len(full),
        "P(sealed full | checks all pass)": float(np.mean([r["sealed_frac"] == 1.0 for r in rows if r["all_passed"]])) if any(r["all_passed"] for r in rows) else None,
        "P(checks all pass | sealed full)": float(np.mean([r["all_passed"] for r in full])) if full else None,
        "residual_by_sealed_full": {"full": [r["residual"] for r in full if r["residual"] is not None],
                                    "not_full": [r["residual"] for r in rows if r["sealed_frac"] < 1.0 and r["residual"] is not None]},
    }
print(json.dumps(summary, indent=1))
