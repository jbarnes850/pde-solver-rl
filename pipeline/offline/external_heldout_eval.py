"""Evaluate an external model (via OpenRouter) on the 135 unseen-family held-out cases.

Same inputs as our transfer arms: the held-out public manifest, the P1 system prompt from
transfer-eval/heldout-p1.yaml, temperature 0.7, top_p 1.0, 8 responses per case. Each response is
scored twice by the same CPU verifier service our model used:
  - strict: the raw response text, exactly as our model's outputs are scored;
  - extracted: code taken from the last fenced block (and any <think> block removed), so a model that
    wraps code in Markdown is not penalized for formatting it was never trained to avoid.
Reasoning is allowed but capped at half the output budget (OpenRouter reasoning.max_tokens) so every
model leaves room to write code. The OpenRouter key is read from the OPENROUTER_API_KEY environment
variable and never printed or written.
Usage: python external_heldout_eval.py MODEL_ID RUN_NAME VERIFIER_CONNECTION [MAX_TOKENS] [CONCURRENCY]
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parents[1]
KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
URL = "https://openrouter.ai/api/v1/chat/completions"


def post(url, payload, headers, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def generate(model, messages, seed, max_tokens):
    payload = {"model": model, "messages": messages, "temperature": 0.7, "top_p": 1.0, "max_tokens": max_tokens, "seed": seed,
               "reasoning": {"max_tokens": max_tokens // 2}}  # bounded reasoning leaves room for the answer
    for attempt in range(6):
        try:
            out = post(URL, payload, {"Authorization": f"Bearer {KEY}"}, timeout=600)
            choice = out["choices"][0]
            return {"content": choice["message"].get("content") or "", "finish_reason": choice.get("finish_reason"),
                    "reasoning_chars": len(choice["message"].get("reasoning") or ""),
                    "usage": out.get("usage"), "provider": out.get("provider"), "model": out.get("model"), "error": None}
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(5 * 2 ** attempt)
                continue
            return {"content": "", "finish_reason": "error", "error": f"HTTP {exc.code}"}
        except Exception as exc:  # network errors: retry, then record
            if attempt < 5:
                time.sleep(5 * 2 ** attempt)
                continue
            return {"content": "", "finish_reason": "error", "error": type(exc).__name__}


def extract(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    return (blocks[-1] if blocks else text).strip()


def verify(conn, case, messages, content, tag):
    payload = {"request_id": f"ext-{tag}", "responses_create_params": {"input": messages},
               "response": {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}]}}
    for attempt in range(4):
        try:
            return post(conn["url"].rstrip("/") + f"/verify/full/{case}", payload, {"Authorization": f"Bearer {conn['token']}"})
        except Exception:
            time.sleep(3 * 2 ** attempt)
    return {"validity": 0.0, "execution_error": "verifier_transport_error"}


def main():
    if not KEY:
        sys.exit("set OPENROUTER_API_KEY in the environment")
    model, run = sys.argv[1], sys.argv[2]
    conn = json.loads(Path(sys.argv[3]).read_text())
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 16384
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    system = yaml.safe_load((HERE / "transfer-eval/heldout-p1.yaml").read_text())["rollout"]["system_prompt"]
    rows = [json.loads(l) for l in (HERE / "transfer-eval/sealed-bundle/public/rl-test.jsonl").read_text().splitlines() if l.strip()]
    samples = 8
    if len(sys.argv) > 6:  # smoke test: first N cases, one sample each
        rows, samples = rows[:int(sys.argv[6])], 1
    out_dir = HERE / "external-eval" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(r, i) for r in rows for i in range(samples)]

    def one(job):
        r, i = job
        messages = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}]
        g = generate(model, messages, 2026 + i, max_tokens)
        strict = verify(conn, r["physical_case_id"], messages, g["content"], f"{run}-{r['physical_case_id']}-{i}-s")
        code = extract(g["content"])
        extracted = strict if code == g["content"] else verify(conn, r["physical_case_id"], messages, code, f"{run}-{r['physical_case_id']}-{i}-x")
        return {"physical_case_id": r["physical_case_id"], "family_id": r["family_id"], "response_index": i,
                "content": g["content"], "content_sha256": hashlib.sha256(g["content"].encode()).hexdigest(),
                "finish_reason": g["finish_reason"], "usage": g.get("usage"), "provider": g.get("provider"),
                "served_model": g.get("model"), "error": g.get("error"), "verifier_strict": strict, "verifier_extracted": extracted}

    path = out_dir / "responses.jsonl"
    done = set()
    if path.exists():  # resume: keep finished responses, redo the rest
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done.add((r["physical_case_id"], r["response_index"]))
    jobs = [j for j in jobs if (j[0]["physical_case_id"], j[1]) not in done]
    with cf.ThreadPoolExecutor(workers) as pool, path.open("a") as fh:
        for fut in cf.as_completed([pool.submit(one, j) for j in jobs]):
            fh.write(json.dumps(fut.result()) + "\n")
            fh.flush()
    results = [json.loads(line) for line in path.read_text().splitlines()]
    summary = {"model": model, "responses": len(results), "errors": sum(1 for r in results if r["error"]),
               "max_tokens": max_tokens, "usage_total": {k: sum((r.get("usage") or {}).get(k, 0) or 0 for r in results)
                                                         for k in ("prompt_tokens", "completion_tokens", "cost")}}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
