"""Run the unchanged solver task across paired model, guidance, and token-budget cells.

This adds scheduling, a sandbox concurrency guard and receipts;
the numerical checks, prompt variants and repair feedback come from the existing Runner.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "materials"))
from rlvpm.genrun import Runner
from rlvpm.sandbox import Sandbox


class GuardedSandbox(Sandbox):
    def run(self, code, inv):
        names = subprocess.check_output(["docker", "ps", "--format", "{{.Names}}"], text=True).splitlines()
        active = sum(n.startswith(("rlvpm-solver-", "rlvp-solver-")) for n in names)
        if active >= 4:
            raise RuntimeError(f"Sandbox guard: {active} solver sandboxes already running")
        return super().run(code, inv)


async def execute(args, out, jobs):
    semaphore = asyncio.Semaphore(args.gen_concurrency)
    pool = ThreadPoolExecutor(max_workers=args.check_workers)
    runners = {}
    for budget in (9216, 20000):
        folder = out / str(budget)
        folder.mkdir()
        r = Runner(folder / "candidates.jsonl", max_tokens=budget,
                   gen_concurrency=args.gen_concurrency, check_workers=args.check_workers)
        r.pool.shutdown()
        r.pool = pool
        r.gen_sem = semaphore
        r.sandbox = GuardedSandbox(timeout_seconds=60.0)
        runners[budget] = r
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*(runners[j["budget"]].lineage(
                client, **{k: v for k, v in j.items() if k != "budget"}) for j in jobs))
    finally:
        pool.shutdown(wait=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=20261022)
    ap.add_argument("--gen-concurrency", type=int, default=8)
    ap.add_argument("--check-workers", type=int, default=2)
    args = ap.parse_args()
    if not 1 <= args.gen_concurrency <= 8 or not 1 <= args.check_workers <= 2:
        ap.error("This run is capped at eight generations and two sandbox workers")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    jobs = []
    rng = random.Random(20261022)
    for i in range(args.n):
        group = [{"budget": budget, "arm": arm, "variant": variant,
                  "seed": args.seed0 + 1000 * i, "rounds": args.rounds,
                  "lineage": f"{arm}-{variant}-{i:03d}"}
                 for budget in (9216, 20000) for arm in ("base", "adapter")
                 for variant in ("plain", "scheme2")]
        rng.shuffle(group)
        jobs.extend(group)
    paths = [Path(__file__), *[ROOT / "materials/rlvpm" / (name + ".py")
                              for name in ("task", "genrun", "llm", "sandbox")]]
    plan = {"started_utc": datetime.now(timezone.utc).isoformat(),
            "args": {**vars(args), "out": str(out)}, "jobs": jobs,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths},
            "scope": "deterministic mathematical solver experiment; not a Cook decision comparison"}
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    started = time.monotonic()
    try:
        asyncio.run(execute(args, out, jobs))
    except BaseException as exc:
        (out / "interrupted.json").write_text(json.dumps({"error": repr(exc),
            "wall_s": time.monotonic() - started}) + "\n")
        raise
    (out / "done.json").write_text(json.dumps({"wall_s": time.monotonic() - started}) + "\n")
    print("Completed factorial", round(time.monotonic() - started), flush=True)


if __name__ == "__main__":
    main()
