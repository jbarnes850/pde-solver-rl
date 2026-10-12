"""Bucket solver_exit_1 failures from the local traceback replay into mechanistic classes."""
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def bucket(x):
    m = x["message"] or ""
    e = x["exception"]
    if x["timed_out"]:
        return "timeout"
    if x["returncode"] == 0:
        return "ran"
    if e in ("SyntaxError", "IndentationError"):
        if re.search(r"unterminated|was never closed|expected an indented block", m):
            return "syntax:incomplete_or_unclosed"
        return "syntax:malformed_code"
    if e == "TypeError" and "callable solver" in m:
        return "interface:no_solver_defined"
    if e == "TypeError" and re.search(r"positional argument|takes \d+", m):
        return "interface:signature_mismatch"
    if e == "ValueError" and "truth value" in m:
        return "array_truth_value"
    if e == "ValueError" and re.search(r"must have shape|must be|odd|requires|Expected|must contain|expected [a-z_]*\s*shape", m) \
            and not re.search(r"broadcast|method", m):
        return "self_imposed_input_check"
    if e in ("ValueError", "IndexError", "numpy.exceptions.AxisError") and re.search(
            r"broadcast|shape|index|indices|axis|unpack|dimension", m):
        return "array_shape_or_indexing"
    if e in ("AttributeError", "ImportError", "ModuleNotFoundError") or (e == "ValueError" and "method" in m) \
            or "unexpected keyword" in m:
        return "library_api_misuse"
    if e in ("NameError", "UnboundLocalError"):
        return "undefined_name"
    if e == "TypeError":
        return "type_error_other"
    if e in ("ZeroDivisionError", "numpy.linalg.LinAlgError", "OverflowError", "RuntimeError", "FloatingPointError"):
        return "numerical_runtime"
    return "other:" + str(e)


def main(corpus, replay, out):
    corp = {}
    for line in open(corpus):
        r = json.loads(line)
        key = f'{r["policy"]}|{r.get("row", r.get("response_index"))}|{r["case_id"]}|{r["candidate_sha256"][:16]}'
        corp[key] = r
    tab = collections.defaultdict(collections.Counter)
    fam = collections.defaultdict(collections.Counter)
    for line in open(replay):
        x = json.loads(line)
        r = corp[x["key"]]
        if r["verifier"]["execution_error"] != "solver_exit_1":
            continue
        b = bucket(x)
        groups = ["base"] if r["policy"] == "base" else (
            ["dev_all5"] + (["dev_cispo32"] if r["policy"] == "cispo32" else []) if r["source"] == "dev_validation"
            else ["train_r8_r13"] + (["train_r13"] if r.get("run") == "r13" else []))
        for g in groups:
            tab[g][b] += 1
        fam[("base" if r["policy"] == "base" else "lineage", r["family_id"])][b] += 1
    for g, c in tab.items():
        n = sum(c.values())
        print(f"== {g} solver_exit_1 n={n}")
        for k, v in c.most_common():
            print(f"   {k:34s} {v:5d} {v / n:6.1%}")
    for (src, f), c in sorted(fam.items()):
        n = sum(c.values())
        print(f"  {src:7s} {f:22s} n={n:4d} " + ", ".join(f"{k}:{v / n:.0%}" for k, v in c.most_common(4)))
    Path(out).write_text(json.dumps({g: dict(c) for g, c in tab.items()} |
                                    {"by_family": {f"{s}|{f}": dict(c) for (s, f), c in fam.items()}}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
