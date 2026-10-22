"""Minimal async client for the local vLLM OpenAI-compatible endpoint.

Every call is logged with the served model name, sampling settings, seed, token counts,
finish reason and wall-clock latency, so generation cost can be accounted per candidate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

import os as _os

_BASE = _os.environ.get("RLVPM_VLLM", "http://127.0.0.1:18431")
ENDPOINT = _BASE + "/v1/chat/completions"
TOKENIZE = _BASE + "/tokenize"
MAX_MODEL_LEN = 32768
MODELS = {"base": "qwen35-9b-base", "adapter": "pde-solver"}


@dataclass
class Generation:
    arm: str
    model: str
    seed: int
    temperature: float
    max_tokens: int
    messages_sha256: str
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def messages_hash(messages: list[dict[str, str]]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


async def generate(
    client: httpx.AsyncClient,
    *,
    arm: str,
    messages: list[dict[str, str]],
    seed: int,
    temperature: float = 0.7,
    max_tokens: int = 9216,
    enable_thinking: bool = False,
    meta: dict[str, Any] | None = None,
) -> Generation:
    model = MODELS[arm]
    # Fit the completion budget to the context: long repair prompts shrink max_tokens.
    try:
        tok = await client.post(TOKENIZE, json={"model": model, "messages": messages,
                                                "chat_template_kwargs": {"enable_thinking": enable_thinking}},
                                timeout=60.0)
        n_prompt = int(tok.json()["count"])
        max_tokens = max(256, min(max_tokens, MAX_MODEL_LEN - n_prompt - 32))
    except Exception:  # noqa: BLE001 - fall back to the requested budget
        pass
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    started = time.monotonic()
    try:
        response = await client.post(ENDPOINT, json=body, timeout=1800.0)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return Generation(
            arm=arm,
            model=model,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            messages_sha256=messages_hash(messages),
            content=choice["message"]["content"] or "",
            finish_reason=str(choice.get("finish_reason")),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.monotonic() - started,
            meta={**dict(meta or {}), "enable_thinking": enable_thinking},
        )
    except Exception as exc:  # noqa: BLE001 - every failure is recorded, none is retried silently
        return Generation(
            arm=arm,
            model=model,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            messages_sha256=messages_hash(messages),
            content="",
            finish_reason="error",
            prompt_tokens=0,
            completion_tokens=0,
            latency_s=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
            meta={**dict(meta or {}), "enable_thinking": enable_thinking},
        )


async def generate_many(
    requests: list[dict[str, Any]], *, concurrency: int = 16, log_path: Path | None = None
) -> list[Generation]:
    """Run requests (kwargs for `generate`) with bounded concurrency, appending to a JSONL log."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[Generation | None] = [None] * len(requests)
    async with httpx.AsyncClient() as client:

        async def one(index: int, request: dict[str, Any]) -> None:
            async with semaphore:
                result = await generate(client, **request)
            results[index] = result
            if log_path is not None:
                with log_path.open("a") as handle:
                    handle.write(json.dumps(asdict(result)) + "\n")

        await asyncio.gather(*(one(i, r) for i, r in enumerate(requests)))
    return [r for r in results if r is not None]
