"""Generation pilot: base vs adapter x guidance level, single attempt plus up to 2 repairs.

Frozen weights. Repair feedback comes only from the analytic check battery. Every candidate
(including failures) is logged to <out>/candidates.jsonl with provenance and cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "materials"))
from magnet.solver_runner import Runner  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="adapter,base")
ap.add_argument("--variants", default="plain,hint,scheme")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--seed0", type=int, default=20261021)
ap.add_argument("--gen-concurrency", type=int, default=48)
ap.add_argument("--check-workers", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=9216)
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
jobs = []
for arm in args.arms.split(","):
    for variant in args.variants.split(","):
        for i in range(args.n):
            jobs.append({"arm": arm, "variant": variant, "seed": args.seed0 + 1000 * i,
                         "rounds": args.rounds, "lineage": f"{arm}-{variant}-{i:03d}"})
(out / "plan.json").write_text(json.dumps({"args": vars(args), "jobs": jobs}, indent=1))
runner = Runner(out / "candidates.jsonl", gen_concurrency=args.gen_concurrency,
                check_workers=args.check_workers, max_tokens=args.max_tokens)
t0 = time.time()
asyncio.run(runner.run(jobs))
(out / "done.json").write_text(json.dumps({"wall_s": time.time() - t0}) + "\n")
print("done", time.time() - t0)
