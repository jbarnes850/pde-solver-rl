"""Generate completions for a prepared request file against a local vLLM server.

Requests come from run_magnetic_term.py (init or apply writes requests-turnN.json).
Completions are saved as text and never executed here; run_magnetic_term.py apply checks
them in the sandbox. Serve the base model and the adapter as described in materials/README.md.
"""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnet.llm import _BASE, generate  # noqa: E402


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='generations JSONL to write')
    ap.add_argument('--concurrency', type=int, default=16)
    args = ap.parse_args()
    plan = json.loads(args.requests.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        (await client.get(_BASE + '/health')).raise_for_status()

        async def one(job):
            async with sem:
                result = await generate(client, arm=job['arm'], messages=job['messages'], seed=job['seed'],
                                        temperature=.7, max_tokens=job['cap'], meta=job)
            with args.out.open('a') as f:
                f.write(json.dumps({**job, **asdict(result)}) + '\n')
            print(job['episode'], result.finish_reason, result.completion_tokens, result.error, flush=True)

        await asyncio.gather(*(one(j) for j in plan['jobs']))


if __name__ == '__main__':
    asyncio.run(main())
