"""Compact per-response table from the fixed evaluation sets (read-only on campaign data).

For each (set, checkpoint) arm that exists, stream probe-results/<run>/groups.jsonl and write
work/rows-<set>-<ckpt>.jsonl with outcome, verifier fields, length/prose statistics and the
verifier-extracted code. Skips arms whose output already exists (re-run picks up new checkpoints).
Usage: python extract.py
"""
import os
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, os.environ.get("RLVP_SRC", str(Path(__file__).resolve().parents[2] / "src")))
from rlvp.prompts import extract_python_code  # noqa: E402

C = Path(os.environ.get("RLVP_CAMPAIGN_DIR", Path(__file__).resolve().parents[1]))
W = Path(__file__).resolve().parent
CKPTS = [("base", "base"), ("s32", "trained32"), ("s64", "cont64"), ("s96", "ext96"), ("s128", "ext128"),
         ("s160", "ext160"), ("s192", "ext192"), ("s224", "ext224"), ("s256", "ext256")]
SETS = {"fresh": "calib-fresh-", "heldout": "calib-heldout-"}
DELIB = re.compile(r"\b(wait|actually|hmm|hold on|let me|let's|re-?check|reconsider|but wait|no,|oops|alternatively|i think|maybe|correction)\b", re.I)


def strict(v):
    e = v.get("trajectory_loss_reference_range")
    return v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def outcome(v, finish):
    if strict(v):
        return "strict"
    if finish == "length":
        return "truncated"
    if v.get("validity") != 1.0:
        return "invalid"
    return "valid_inaccurate"


def text_stats(content, code):
    fences = content.count("```")
    first = content.find("```")
    pre = content[:first] if first >= 0 else ""
    lines = [l for l in code.splitlines() if l.strip()]
    com = [l for l in lines if l.strip().startswith("#")]
    inline = [l for l in lines if not l.strip().startswith("#") and "#" in l]
    delib = sum(1 for l in com if DELIB.search(l))
    return {"chars": len(content), "fences": fences, "pre_chars": len(pre), "code_chars": len(code),
            "outside_chars": max(0, len(content) - len(code)), "code_lines": len(lines),
            "noncomment_lines": len(lines) - len(com), "comment_lines": len(com), "inline_comments": len(inline),
            "delib_comment_lines": delib, "has_def_solver": bool(re.search(r"def\s+solver\s*\(", code))}


def run(arm):
    setname, ck, run = arm
    src = C / "probe-results" / run / "groups.jsonl"
    out = W / f"rows-{setname}-{ck}.jsonl"
    if out.exists() or not src.exists():
        return arm, "skip"
    tmp = out.with_suffix(".tmp")
    n = 0
    with tmp.open("w") as fh:
        for line in src.open():
            g = json.loads(line)
            for r in g["responses"]:
                v = r["verifier"]
                content = r.get("content") or ""
                code = extract_python_code(content)
                e = v.get("trajectory_loss_reference_range")
                row = {"set": setname, "ck": ck, "case": g["physical_case_id"], "family": g["family_id"],
                       "form": g.get("prompt_form"), "idx": r["response_index"], "seed": r.get("seed"),
                       "finish": r.get("finish_reason"), "ntok": len(r.get("token_ids") or []),
                       "validity": v.get("validity"), "err": e if (e is not None and math.isfinite(e)) else None,
                       "exec_error": v.get("execution_error"), "exec_s": v.get("execution_seconds"),
                       "cand_res": v.get("candidate_residual"), "anchored": v.get("anchored_reward"),
                       "triv": v.get("best_trivial_error"), "outcome": outcome(v, r.get("finish_reason")),
                       **text_stats(content, code), "code": code}
                fh.write(json.dumps(row) + "\n")
                n += 1
    tmp.rename(out)
    return arm, n


def main():
    arms = [(s, ck, pre + suf) for s, pre in SETS.items() for ck, suf in CKPTS]
    with ProcessPoolExecutor(4) as ex:
        for arm, n in ex.map(run, arms):
            print(arm, n, flush=True)


if __name__ == "__main__":
    main()
