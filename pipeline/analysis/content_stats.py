"""Whole-response deliberation and draft/final structure (read-only). Writes work/content-<set>-<ck>.jsonl.

delib_all: lines anywhere in the response (draft, comments, prose) containing self-correction markers.
draft_final_jaccard: overlap of non-empty stripped line sets between the pre-fence text and the fenced code.
"""
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import C, W  # noqa: E402
from extract import CKPTS, DELIB, SETS  # noqa: E402

MARK = re.compile(r"\b(wait|actually|hmm|but wait|oops|let me re|re-?check|reconsider|correction|no,)\b", re.I)


def run(arm):
    s, ck, runname = arm
    src = C / "probe-results" / runname / "groups.jsonl"
    out = W / f"content-{s}-{ck}.jsonl"
    if out.exists() or not src.exists():
        return arm, "skip"
    with out.open("w") as fh:
        for line in src.open():
            g = json.loads(line)
            for r in g["responses"]:
                c = r.get("content") or ""
                lines = [l for l in c.splitlines() if l.strip()]
                i = c.find("```")
                pre = c[:i] if i > 0 else ""
                m = re.search(r"```(?:python)?\s*(.*?)```", c, flags=re.DOTALL | re.I)
                fin = m.group(1) if m else ""
                a = {l.strip() for l in pre.splitlines() if l.strip() and not l.strip().startswith("#")}
                b = {l.strip() for l in fin.splitlines() if l.strip() and not l.strip().startswith("#")}
                jac = (len(a & b) / len(a | b)) if (a and b) else None
                fh.write(json.dumps({"case": g["physical_case_id"], "idx": r["response_index"], "lines_all": len(lines),
                                     "comment_lines_all": sum(l.strip().startswith("#") for l in lines),
                                     "delib_all": sum(1 for l in lines if MARK.search(l)),
                                     "delib_broad_all": sum(1 for l in lines if DELIB.search(l)),
                                     "pre_has_def_solver": bool(re.search(r"def\s+solver\s*\(", pre)),
                                     "draft_final_jaccard": jac}) + "\n")
    return arm, "ok"


if __name__ == "__main__":
    arms = [(s, ck, pre + suf) for s, pre in SETS.items() for ck, suf in CKPTS]
    with ProcessPoolExecutor(4) as ex:
        for a, st in ex.map(run, arms):
            print(a, st)
