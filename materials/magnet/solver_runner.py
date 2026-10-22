"""Generation, checking and repair of solver candidates with full provenance.

Each candidate record carries: arm (base/adapter), guidance variant, round (0 = first
attempt), parent candidate id (repairs), sampling seed, token counts, latency, the code hash,
and the analytic check report. Repair feedback is built only from the analytic checks and the
candidate's own outputs (solver_task.CheckReport.feedback); no reference trajectory, sealed case or
decision outcome is ever included.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import httpx

from .llm import generate
from .sandbox import Sandbox, extract
from .solver_task import messages, run_checks

REPAIR_TEMPLATE = (
    "Your solver was run on test problems with known analytic behavior. Some checks failed:\n"
    "{feedback}\n\n"
    "Fix the solver. Return the complete corrected module, following the same rules."
)


def code_sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


class Runner:
    def __init__(self, log_path: Path, *, gen_concurrency: int = 24, check_workers: int = 3,
                 max_tokens: int = 9216, temperature: float = 0.7):
        self.log_path = log_path
        self.gen_sem = asyncio.Semaphore(gen_concurrency)
        self.pool = ThreadPoolExecutor(max_workers=check_workers)
        self.sandbox = Sandbox(timeout_seconds=60.0)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache: dict[str, dict] = {}  # code sha -> check summary (identical code is reused)

    def _log(self, rec: dict) -> None:
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    async def _check(self, code: str) -> dict:
        sha = code_sha(code)
        if sha in self.cache:
            return {**self.cache[sha], "cached": True}
        loop = asyncio.get_running_loop()
        rep = await loop.run_in_executor(self.pool, run_checks, code, self.sandbox)
        summary = {"passed": rep.passed, "numbers": {k: v for k, v in rep.numbers.items()
                                                     if not k.endswith("_msg")},
                   "errors": {k: v[-700:] for k, v in rep.errors.items()},
                   "all_passed": rep.all_passed, "n_passed": rep.n_passed,
                   "check_seconds": rep.seconds, "residual": rep.residual,
                   "feedback": rep.feedback()}
        self.cache[sha] = summary
        return summary

    async def attempt(self, client, *, arm: str, variant: str, seed: int, history: list,
                      round_: int, parent: str | None, lineage: str) -> dict:
        msgs = messages(variant) + history
        async with self.gen_sem:
            g = await generate(client, arm=arm, messages=msgs, seed=seed,
                               temperature=self.temperature, max_tokens=self.max_tokens)
        # Preserve the raw response before any slow or interrupted sandbox check. This
        # also records the effective cap after the client fits a repair into its context.
        with self.log_path.with_name("generations.jsonl").open("a") as fh:
            fh.write(json.dumps({"id": f"{lineage}-r{round_}", "variant": variant,
                                 "round": round_, "requested_max_tokens": self.max_tokens,
                                 **asdict(g)}) + "\n")
        code = extract(g.content)
        rec = {"id": f"{lineage}-r{round_}", "lineage": lineage, "arm": arm, "variant": variant,
               "round": round_, "parent": parent, "seed": seed, "finish": g.finish_reason,
               "prompt_tokens": g.prompt_tokens, "completion_tokens": g.completion_tokens,
               "latency_s": g.latency_s, "gen_error": g.error, "code_sha": code_sha(code),
               "requested_max_tokens": self.max_tokens, "effective_max_tokens": g.max_tokens,
               "messages_sha256": g.messages_sha256,
               "code": code, "t_wall": time.time(), "producer": f"qwen3.5-9b/{arm}",
               "intervention": {"plain": "none", "hint": "method_hint",
                                "scheme": "scheme_formula",
                                "scheme2": "sbdf2_formula"}[variant]}
        if g.error or not code.strip():
            rec["check"] = {"all_passed": False, "n_passed": 0, "passed": {},
                            "feedback": "No code was returned.", "errors": {"gen": g.error}}
        else:
            rec["check"] = await self._check(code)
        self._log(rec)
        return rec

    async def lineage(self, client, *, arm: str, variant: str, seed: int, rounds: int,
                      lineage: str) -> list[dict]:
        """One first attempt plus up to `rounds` feedback-driven repairs."""
        recs = []
        history: list[dict] = []
        parent = None
        for r in range(rounds + 1):
            rec = await self.attempt(client, arm=arm, variant=variant, seed=seed + 7919 * r,
                                     history=history, round_=r, parent=parent, lineage=lineage)
            recs.append(rec)
            # A serving failure is not evidence about the solver and cannot supply
            # meaningful repair feedback. A retry belongs in a separate run receipt.
            if rec["gen_error"] or rec["check"].get("all_passed"):
                break
            # Repair context: the previous code and the analytic-check feedback only.
            history = [{"role": "assistant", "content": rec["code"] or "(no code)"},
                       {"role": "user", "content": REPAIR_TEMPLATE.format(
                           feedback=rec["check"]["feedback"])}]
            parent = rec["id"]
        return recs

    async def run(self, jobs: list[dict]) -> list[list[dict]]:
        async with httpx.AsyncClient() as client:
            return await asyncio.gather(*(self.lineage(client, **j) for j in jobs))
