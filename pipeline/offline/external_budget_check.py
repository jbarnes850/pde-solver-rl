"""Did the 32k output budget limit the external models? Diagnostic only, not a ranking input.

For each external model: share of responses that hit the output limit, share that returned no code at
all, reasoning-token usage (the 16k reasoning cap was requested but providers did not enforce it), and
strict pass@1 among responses that returned code. The ranked number stays the pre-registered pass@1
with every response counted. Writes external-eval/budget-check.json.
"""
import json
import math
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
TRIV = json.loads((HERE / "offline/heldout-trivial-errors.json").read_text())
RUNS = {"MiMo-V2.6-Pro": "heldout-xiaomi-mimo-v2-6-pro", "GLM-5.3": "heldout-z-ai-glm-5-3",
        "Qwen3.8-Max": "heldout-qwen-qwen3-8-max-0902"}


def strict(v):
    e = v.get("trajectory_loss_reference_range")
    return v.get("validity") == 1.0 and e is not None and math.isfinite(e) and e <= 0.01


def eqf(rows):
    fams = sorted({r["family_id"] for r in rows})
    per = []
    for f in fams:
        cases = sorted({r["physical_case_id"] for r in rows if r["family_id"] == f})
        rates = []
        for c in cases:
            rs = [strict(r["verifier_strict"]) for r in rows if r["physical_case_id"] == c]
            if rs:
                rates.append(sum(rs) / len(rs))
        if rates:
            per.append(sum(rates) / len(rates))
    return sum(per) / len(per)


def main():
    out = {}
    for name, run in RUNS.items():
        rows = [json.loads(l) for l in (HERE / "external-eval" / run / "responses.jsonl").read_text().splitlines()]
        rows = [r for r in rows if TRIV[r["physical_case_id"]] > 0.01]
        empty = [r for r in rows if not (r.get("content") or "").strip()]
        answered = [r for r in rows if (r.get("content") or "").strip()]
        reasoning = [r["usage"].get("completion_tokens_details", {}).get("reasoning_tokens") or 0 for r in rows if r.get("usage")]
        out[name] = {"responses": len(rows),
                     "hit_output_limit": sum(r.get("finish_reason") == "length" for r in rows) / len(rows),
                     "no_code_returned": len(empty) / len(rows),
                     "provider_error": sum(r.get("finish_reason") == "error" for r in rows) / len(rows),
                     "reasoning_over_16k": sum(t > 16384 for t in reasoning) / len(reasoning),
                     "median_completion_tokens": statistics.median(r["usage"]["completion_tokens"] for r in rows if r.get("usage")),
                     "no_code_by_family": dict(Counter(r["family_id"] for r in empty).most_common()),
                     "pass1_all_responses": eqf(rows),
                     "pass1_among_answered": eqf(answered)}
        o = out[name]
        print(f"{name:14s} limit {100*o['hit_output_limit']:4.1f}%  no code {100*o['no_code_returned']:4.1f}%  "
              f"reasoning>16k {100*o['reasoning_over_16k']:4.1f}%  median tokens {o['median_completion_tokens']:7.0f}  "
              f"pass@1 {100*o['pass1_all_responses']:4.1f}%  among answered {100*o['pass1_among_answered']:4.1f}%")
    (HERE / "external-eval/budget-check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
