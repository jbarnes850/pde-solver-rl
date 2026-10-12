from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .prompts import SYSTEM_PROMPT

PROBE_SCHEMA = "rlvp.no-update-variance-probe.v2"
SEED_NAMESPACE = "rlvp-no-update-v1"
ARM_NAMES = ("validity", "trajectory", "full")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def response_seed(
    *,
    manifest_sha256: str,
    physical_case_id: str,
    attempt_index: int,
    response_index: int,
    seed_namespace: str = SEED_NAMESPACE,
) -> int:
    material = "|".join(
        (
            seed_namespace,
            manifest_sha256,
            physical_case_id,
            str(attempt_index),
            str(response_index),
        )
    )
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def build_seed_table(
    *,
    rows: Iterable[dict[str, Any]],
    manifest_sha256: str,
    attempt_index: int,
    generations: int,
    seed_namespace: str = SEED_NAMESPACE,
) -> list[dict[str, Any]]:
    return [
        {
            "group_index": group_index,
            "physical_case_id": row["physical_case_id"],
            "attempt_index": attempt_index,
            "response_index": response_index,
            "seed": response_seed(
                manifest_sha256=manifest_sha256,
                physical_case_id=str(row["physical_case_id"]),
                attempt_index=attempt_index,
                response_index=response_index,
                seed_namespace=seed_namespace,
            ),
        }
        for group_index, row in enumerate(rows)
        for response_index in range(generations)
    ]


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    )


def load_locked_rows(
    path: Path, *, expected_role: str = "rl_train"
) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("the locked manifest is empty")
    seen_cases: set[str] = set()
    for index, row in enumerate(rows):
        required = {
            "family_id",
            "physical_case_id",
            "prompt",
            "prompt_sha256",
            "role",
            "row_id",
        }
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"manifest row {index} is missing {missing}")
        if row["role"] != expected_role:
            raise ValueError(
                f"manifest row {index} has role {row['role']!r}; "
                f"expected {expected_role!r}"
            )
        prompt_sha256 = hashlib.sha256(str(row["prompt"]).encode()).hexdigest()
        if prompt_sha256 != row["prompt_sha256"]:
            raise ValueError(f"manifest row {index} has a prompt hash mismatch")
        physical_case_id = str(row["physical_case_id"])
        if physical_case_id in seen_cases:
            raise ValueError(f"duplicate physical case ID: {physical_case_id}")
        seen_cases.add(physical_case_id)
    return rows


# Optional contract fields are recorded only when set, so unchanged configs keep their digests.
OPTIONAL_CONTRACT_DEFAULTS = {"system_prompt": None, "seed_manifest_sha256": None, "case_concurrency": 1}


def load_experiment_contract(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise TypeError("the experiment config must contain a YAML mapping")
    algorithm = raw.get("algorithm") or {}
    rollout = raw.get("rollout") or {}
    model = raw.get("model") or {}
    evaluation = raw.get("evaluation") or {}
    execution = raw.get("execution") or {}
    reward = raw.get("reward") or {}
    runtime = raw.get("runtime") or {}
    required = {
        "experiment_id": raw.get("id"),
        "algorithm_name": algorithm.get("name"),
        "advantage_estimator": algorithm.get("advantage"),
        "generations_per_prompt": rollout.get("generations_per_prompt"),
        "temperature": rollout.get("temperature"),
        "top_p": rollout.get("top_p"),
        "max_new_tokens": rollout.get("max_new_tokens"),
        "enable_thinking": rollout.get("enable_thinking"),
        "model_name": model.get("name"),
        "attempt_indices": evaluation.get("attempt_indices"),
        "seed_namespace": evaluation.get("seed_namespace"),
        "max_transport_repaired_fraction": evaluation.get("max_transport_repaired_fraction"),
        "workers": execution.get("workers"),
        "model_timeout_seconds": execution.get("model_timeout_seconds"),
        "verifier_timeout_seconds": execution.get("verifier_timeout_seconds"),
        "transport_retries": execution.get("transport_retries"),
        "cross_host_output_equality": evaluation.get("cross_host_output_equality"),
        "minimum_aggregate_saved_output_tokens_per_second": evaluation.get(
            "minimum_aggregate_saved_output_tokens_per_second"
        ),
        "manifest_role": evaluation.get("manifest_role", "rl_train"),
    }
    runtime_required = {
        "image_ref": runtime.get("image_ref"),
        "image_digest": runtime.get("image_digest"),
        "vllm_version": runtime.get("vllm_version"),
        "batch_invariance_enabled": runtime.get("batch_invariance_enabled"),
        "tensor_parallel_size": runtime.get("tensor_parallel_size"),
        "quantization": runtime.get("quantization"),
        "gpu_memory_utilization": runtime.get("gpu_memory_utilization"),
        "max_model_len": runtime.get("max_model_len"),
        "max_num_batched_tokens": runtime.get("max_num_batched_tokens"),
        "max_num_seqs": runtime.get("max_num_seqs"),
        "prefix_caching_enabled": runtime.get("prefix_caching_enabled"),
        "chunked_prefill_enabled": runtime.get("chunked_prefill_enabled"),
        "mamba_cache_mode": runtime.get("mamba_cache_mode"),
        "seed": runtime.get("seed"),
    }
    reward_required = {
        "trajectory_normalization": reward.get("trajectory_normalization"),
        "trajectory_temperature": reward.get("trajectory_temperature"),
        "residual_temperature": reward.get("residual_temperature"),
        "epsilon": reward.get("epsilon"),
    }
    missing = sorted(key for key, value in required.items() if value is None)
    missing.extend(f"runtime.{key}" for key, value in runtime_required.items() if value is None)
    missing.extend(f"reward.{key}" for key, value in reward_required.items() if value is None)
    if missing:
        raise ValueError(f"the experiment config is missing {missing}")
    if required["cross_host_output_equality"] != "diagnostic":
        raise ValueError("cross-host output equality must be diagnostic")
    if float(required["minimum_aggregate_saved_output_tokens_per_second"]) < 0.0:
        raise ValueError("the aggregate throughput telemetry limit cannot be negative")
    if reward_required["trajectory_normalization"] not in {
        "reference_l2",
        "reference_range",
    }:
        raise ValueError("the trajectory normalization is invalid")
    for key in ("trajectory_temperature", "residual_temperature", "epsilon"):
        if float(reward_required[key]) <= 0.0:
            raise ValueError(f"reward.{key} must be positive")
    system_prompt = rollout.get("system_prompt")
    if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ValueError("rollout.system_prompt must be a nonempty string when present")
    seed_manifest_sha256 = evaluation.get("seed_manifest_sha256")
    if seed_manifest_sha256 is not None and (
        not isinstance(seed_manifest_sha256, str) or len(seed_manifest_sha256) != 64
    ):
        raise ValueError("evaluation.seed_manifest_sha256 must be a SHA-256 hex digest")
    case_concurrency = int(execution.get("case_concurrency", 1))
    if not 1 <= case_concurrency <= 16:
        raise ValueError("execution.case_concurrency must be between 1 and 16")
    return {
        **required,
        "system_prompt": system_prompt,
        "seed_manifest_sha256": seed_manifest_sha256,
        "case_concurrency": case_concurrency,
        "kl_penalty": algorithm.get("kl_penalty"),
        "normalize_rewards": algorithm.get("normalize_rewards"),
        "use_leave_one_out_baseline": algorithm.get("use_leave_one_out_baseline"),
        "dynamic_sampling": bool(algorithm.get("dynamic_sampling", False)),
        "reward_shaping": bool(algorithm.get("reward_shaping", False)),
        "top_k": rollout.get("top_k"),
        "resampling_policy": evaluation.get("resampling_policy"),
        "selection_policy": evaluation.get("selection_policy"),
        "reward_arms": evaluation.get("reward_arms"),
        "reward": reward_required,
        "runtime": runtime_required,
    }


def select_row_shard(
    rows: list[dict[str, Any]], *, shard_index: int = 0, shard_count: int = 1
) -> list[dict[str, Any]]:
    """Select a deterministic host shard without changing the canonical manifest hash."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return rows[shard_index::shard_count]


def select_physical_case(
    rows: list[dict[str, Any]], physical_case_id: str | None
) -> list[dict[str, Any]]:
    if physical_case_id is None:
        return rows
    matches = [row for row in rows if str(row["physical_case_id"]) == physical_case_id]
    if len(matches) != 1:
        raise ValueError(f"physical case filter matched {len(matches)} rows: {physical_case_id}")
    return matches


def sampling_from_experiment(contract: dict[str, Any]) -> dict[str, Any]:
    """Return the model-behavior sampling contract without CLI duplication."""

    sampling = {
        "generations": int(contract["generations_per_prompt"]),
        "temperature": float(contract["temperature"]),
        "top_p": float(contract["top_p"]),
        "top_k": contract["top_k"],
        "max_tokens": int(contract["max_new_tokens"]),
        "enable_thinking": contract["enable_thinking"],
    }
    if sampling["generations"] < 1 or sampling["max_tokens"] < 1:
        raise ValueError("experiment sampling counts must be positive")
    if not 0.0 < sampling["top_p"] <= 1.0:
        raise ValueError("experiment top_p must be in (0, 1]")
    if not isinstance(sampling["enable_thinking"], bool):
        raise TypeError("experiment enable_thinking must be a boolean")
    if sampling["top_k"] is not None:
        sampling["top_k"] = int(sampling["top_k"])
    return sampling


def arm_rewards(verifier_results: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    values = {name: [] for name in ARM_NAMES}
    for result in verifier_results:
        validity = float(result["validity"])
        trajectory = validity * float(result["trajectory_reward"])
        full = float(result["full_reward"])
        values["validity"].append(validity)
        values["trajectory"].append(trajectory)
        values["full"].append(full)
    return values


def _round_to_bfloat16(values: np.ndarray) -> np.ndarray:
    """Round finite float32 values to BF16 and return them as float32."""

    values32 = np.asarray(values, dtype=np.float32)
    bits = values32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def arm_statistics(rewards: Iterable[float]) -> dict[str, Any]:
    values64 = np.asarray(list(rewards), dtype=np.float64)
    if values64.ndim != 1 or values64.size == 0:
        raise ValueError("rewards must be a non-empty one-dimensional sequence")
    if not np.isfinite(values64).all():
        raise ValueError("rewards must be finite")
    values32 = values64.astype(np.float32)
    values_bfloat16 = _round_to_bfloat16(values32)
    std64 = float(values64.std(ddof=0))
    std32 = float(values32.std(ddof=0, dtype=np.float32))
    std_bfloat16 = float(values_bfloat16.std(ddof=0, dtype=np.float32))
    return {
        "rewards_float64": values64.tolist(),
        "rewards_float32": [float(value) for value in values32],
        "rewards_bfloat16": [float(value) for value in values_bfloat16],
        "mean_float64": float(values64.mean()),
        "min_float64": float(values64.min()),
        "max_float64": float(values64.max()),
        "population_std_float64": std64,
        "population_std_float32": std32,
        "population_std_bfloat16": std_bfloat16,
        "exact_zero_variance_float64": std64 == 0.0,
        "exact_zero_variance_float32": std32 == 0.0,
        "exact_zero_variance_bfloat16": std_bfloat16 == 0.0,
        "distinct_float64": int(np.unique(values64).size),
        "distinct_float32": int(np.unique(values32).size),
        "distinct_bfloat16": int(np.unique(values_bfloat16).size),
    }


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise TypeError(f"expected a JSON object from {url}")
    return result


def _get_json(url: str, *, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise TypeError(f"expected a JSON object from {url}")
    return result


def _with_transport_retries(
    operation: Callable[[], dict[str, Any]],
    *,
    retries: int,
) -> tuple[dict[str, Any], int]:
    failures: list[str] = []
    for retry_index in range(retries + 1):
        try:
            return operation(), retry_index
        except (TimeoutError, urllib.error.URLError, ConnectionError) as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            if retry_index >= retries:
                raise RuntimeError("transport retries exhausted: " + " | ".join(failures)) from exc
            time.sleep(min(2**retry_index, 4))
    raise AssertionError("unreachable")


def _extract_choice(
    response: dict[str, Any],
) -> tuple[str, list[int], list[float], list[int], str]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"expected one model choice, got {len(choices)}")
    choice = choices[0]
    content = str((choice.get("message") or {}).get("content") or "")
    token_ids = choice.get("token_ids") or response.get("token_ids") or []
    logprob_rows = (choice.get("logprobs") or {}).get("content") or []
    logprobs = [float(item["logprob"]) for item in logprob_rows]
    if not logprob_rows:
        raise ValueError("the model response has no generated log probabilities")
    rows_with_token_ids = ["token_id" in item for item in logprob_rows]
    if any(rows_with_token_ids) and not all(rows_with_token_ids):
        raise ValueError("generated log-probability rows have inconsistent token IDs")
    logprob_token_ids = (
        [int(item["token_id"]) for item in logprob_rows] if all(rows_with_token_ids) else []
    )
    if not token_ids and logprob_token_ids:
        token_ids = logprob_token_ids
    token_ids = [int(value) for value in token_ids]
    if not token_ids:
        raise ValueError("the model response has no generated token IDs")
    if len(token_ids) != len(logprobs):
        raise ValueError(
            f"token/log-probability mismatch: {len(token_ids)} token IDs and {len(logprobs)} values"
        )
    if logprob_token_ids and token_ids != logprob_token_ids:
        raise ValueError("generated token IDs do not match log-probability-row token IDs")
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("the model response has a nonfinite log probability")
    prompt_token_ids = [int(value) for value in response.get("prompt_token_ids") or []]
    if not prompt_token_ids:
        raise ValueError("the model response has no prompt token IDs")
    return (
        content,
        token_ids,
        logprobs,
        prompt_token_ids,
        str(choice.get("finish_reason") or ""),
    )


def _stable_model_identity(model_health: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = model_health.get("data") or []
    matches = [row for row in models if str(row.get("id")) == model_name]
    if len(matches) != 1:
        raise RuntimeError(f"expected one stable model identity for {model_name!r}")
    row = matches[0]
    return {
        "id": row.get("id"),
        "root": row.get("root"),
        "owned_by": row.get("owned_by"),
        "max_model_len": row.get("max_model_len"),
        "identity_source": "openai_models_endpoint",
    }


def _resolve_model_identity(
    *,
    model_base_url: str,
    model_name: str,
    policy_artifact: dict[str, Any] | None,
    max_model_len: int,
) -> dict[str, Any]:
    """Resolve a live model identity without requiring a non-standard route.

    NeMo RL's native vLLM HTTP adapter exposes chat completion and tokenization,
    but not ``/v1/models``. A policy-artifact receipt supplies the immutable
    model and checkpoint identity for that server. The OpenAPI check still
    proves that the intended live HTTP surface is present.
    """

    try:
        model_health = _get_json(model_base_url.rstrip("/") + "/v1/models", timeout=30)
    except urllib.error.HTTPError as error:
        if error.code != 404 or policy_artifact is None:
            raise
        if policy_artifact.get("served_model_name") != model_name:
            raise ValueError("the served model name does not match the policy receipt")
        openapi = _get_json(model_base_url.rstrip("/") + "/openapi.json", timeout=30)
        paths = openapi.get("paths") or {}
        required_paths = {"/v1/chat/completions", "/tokenize"}
        if not required_paths.issubset(paths):
            raise RuntimeError("the NeMo model server is missing required HTTP routes")
        return {
            "id": model_name,
            "root": policy_artifact.get("policy_id"),
            "owned_by": "policy_artifact_receipt",
            "max_model_len": max_model_len,
            "identity_source": "policy_artifact_and_live_openapi",
            "openapi_paths_sha256": canonical_sha256(sorted(paths)),
        }
    return _stable_model_identity(model_health, model_name)


def _validate_verifier_result(
    result: dict[str, Any], *, physical_case_id: str, request_id: str
) -> None:
    expected = {
        "case_id": physical_case_id,
        "request_id": request_id,
        "reward_mode": "full",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"verifier {key} mismatch: {result.get(key)!r} != {value!r}")
    components = [
        float(result["validity"]),
        float(result["trajectory_reward"]),
        float(result["residual_reward"]),
        float(result["full_reward"]),
        float(result["reward"]),
    ]
    if not all(math.isfinite(value) for value in components):
        raise ValueError("the verifier returned a nonfinite reward component")
    expected_full = components[0] * components[1] * components[2]
    if not math.isclose(components[3], expected_full, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("the verifier full reward does not equal the component product")
    if not math.isclose(components[4], components[3], rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the verifier reward does not equal full_reward")


def _model_request(
    *,
    model_url: str,
    model_name: str,
    prompt: str,
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    enable_thinking: bool,
    timeout: float,
    transport_retries: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    request_payload = {
        "model": model_name,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": 0,
        "return_token_ids": True,
        "return_prompt_text": True,
    }
    if top_k is not None:
        request_payload["top_k"] = top_k
    response, retry_count = _with_transport_retries(
        lambda: _post_json(model_url, request_payload, timeout=timeout),
        retries=transport_retries,
    )
    return response, retry_count, request_payload


def _verify_request(
    *,
    verifier_base_url: str,
    physical_case_id: str,
    request_id: str,
    messages: list[dict[str, str]],
    content: str,
    timeout: float,
    transport_retries: int,
) -> tuple[dict[str, Any], int]:
    payload = {
        "request_id": request_id,
        "responses_create_params": {"input": messages},
        "response": {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}],
                }
            ]
        },
    }
    return _with_transport_retries(
        lambda: _post_json(
            f"{verifier_base_url.rstrip('/')}/verify/full/{physical_case_id}",
            payload,
            timeout=timeout,
        ),
        retries=transport_retries,
    )


def probe_response(
    *,
    row: dict[str, Any],
    response_index: int,
    attempt_index: int,
    manifest_sha256: str,
    model_url: str,
    model_name: str,
    verifier_base_url: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
    enable_thinking: bool,
    model_timeout: float,
    verifier_timeout: float,
    transport_retries: int,
    seed_namespace: str = SEED_NAMESPACE,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    case_id = str(row["physical_case_id"])
    seed = response_seed(
        manifest_sha256=manifest_sha256,
        physical_case_id=case_id,
        attempt_index=attempt_index,
        response_index=response_index,
        seed_namespace=seed_namespace,
    )
    started = time.monotonic()
    response, model_retries, request_payload = _model_request(
        model_url=model_url,
        model_name=model_name,
        prompt=str(row["prompt"]),
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        enable_thinking=enable_thinking,
        timeout=model_timeout,
        transport_retries=transport_retries,
        system_prompt=system_prompt,
    )
    content, token_ids, logprobs, prompt_token_ids, finish_reason = _extract_choice(response)
    request_id = f"probe-{attempt_index}-{response_index}-{case_id}"
    verifier, verifier_retries = _verify_request(
        verifier_base_url=verifier_base_url,
        physical_case_id=case_id,
        request_id=request_id,
        messages=request_payload["messages"],
        content=content,
        timeout=verifier_timeout,
        transport_retries=transport_retries,
    )
    _validate_verifier_result(verifier, physical_case_id=case_id, request_id=request_id)
    return {
        "response_index": response_index,
        "seed": seed,
        "request_sha256": canonical_sha256(request_payload),
        "content": content,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "token_ids": token_ids,
        "token_ids_sha256": canonical_sha256(token_ids),
        "behavior_logprobs": logprobs,
        "behavior_logprobs_sha256": canonical_sha256(logprobs),
        "finish_reason": finish_reason,
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": canonical_sha256(prompt_token_ids),
        "model_created": response.get("created"),
        "model_id": response.get("id"),
        "model_retries": model_retries,
        "verifier_retries": verifier_retries,
        "verifier": verifier,
        "wall_seconds": time.monotonic() - started,
    }


def probe_case(
    *,
    row: dict[str, Any],
    group_index: int,
    generations: int,
    attempt_index: int,
    workers: int,
    config_bindings: dict[str, str],
    probe_kwargs: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers, generations)) as executor:
        results = list(
            executor.map(
                lambda response_index: probe_response(
                    row=row,
                    response_index=response_index,
                    attempt_index=attempt_index,
                    **probe_kwargs,
                ),
                range(generations),
            )
        )
    results.sort(key=lambda value: int(value["response_index"]))
    rewards = arm_rewards(result["verifier"] for result in results)
    return {
        "schema": PROBE_SCHEMA,
        "record_type": "group",
        **config_bindings,
        "group_index": group_index,
        "attempt_index": attempt_index,
        "row_id": row["row_id"],
        "family_id": row["family_id"],
        "physical_case_id": row["physical_case_id"],
        "prompt_form": row.get("prompt_form"),
        "prompt_sha256": row["prompt_sha256"],
        "group_size": generations,
        "arms": {name: arm_statistics(values) for name, values in rewards.items()},
        "responses": results,
        "transport_clean": all(
            int(result["model_retries"]) == 0 and int(result["verifier_retries"]) == 0
            for result in results
        ),
        "wall_seconds": time.monotonic() - started,
    }


def summarize_groups(groups: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(groups)
    response_count = sum(len(group["responses"]) for group in values)
    output_token_count = sum(
        len(response["token_ids"]) for group in values for response in group["responses"]
    )
    group_wall_seconds_sum = sum(float(group["wall_seconds"]) for group in values)
    summary: dict[str, Any] = {
        "group_count": len(values),
        "response_count": response_count,
        "output_token_count": output_token_count,
        "group_wall_seconds_sum": group_wall_seconds_sum,
        "saved_output_tokens_per_second": (
            output_token_count / group_wall_seconds_sum if group_wall_seconds_sum > 0.0 else None
        ),
        "transport_clean_groups": sum(bool(group["transport_clean"]) for group in values),
        "arms": {},
        "families": {},
    }
    for arm_name in ARM_NAMES:
        arm_rows = [group["arms"][arm_name] for group in values]
        summary["arms"][arm_name] = {
            "exact_zero_variance_groups_float64": sum(
                bool(row["exact_zero_variance_float64"]) for row in arm_rows
            ),
            "exact_zero_variance_groups_float32": sum(
                bool(row["exact_zero_variance_float32"]) for row in arm_rows
            ),
            "exact_zero_variance_groups_bfloat16": sum(
                bool(row["exact_zero_variance_bfloat16"]) for row in arm_rows
            ),
            "median_population_std_float64": (
                statistics.median(row["population_std_float64"] for row in arm_rows)
                if arm_rows
                else None
            ),
            "median_population_std_bfloat16": (
                statistics.median(row["population_std_bfloat16"] for row in arm_rows)
                if arm_rows
                else None
            ),
        }
    for group in values:
        family = str(group["family_id"])
        family_rows = summary["families"].setdefault(family, {"groups": 0, "arms": {}})
        family_rows["groups"] += 1
        for arm_name in ARM_NAMES:
            arm = family_rows["arms"].setdefault(
                arm_name,
                {
                    "exact_zero_variance_groups_float64": 0,
                    "exact_zero_variance_groups_bfloat16": 0,
                    "group_population_std_float64": [],
                    "group_population_std_bfloat16": [],
                    "rewards_float64": [],
                    "rewards_bfloat16": [],
                },
            )
            stats = group["arms"][arm_name]
            arm["exact_zero_variance_groups_float64"] += int(stats["exact_zero_variance_float64"])
            arm["exact_zero_variance_groups_bfloat16"] += int(
                stats["exact_zero_variance_bfloat16"]
            )
            arm["group_population_std_float64"].append(stats["population_std_float64"])
            arm["group_population_std_bfloat16"].append(
                stats["population_std_bfloat16"]
            )
            arm["rewards_float64"].extend(stats["rewards_float64"])
            arm["rewards_bfloat16"].extend(stats["rewards_bfloat16"])
    for family in summary["families"].values():
        for arm in family["arms"].values():
            rewards64 = np.asarray(arm.pop("rewards_float64"), dtype=np.float64)
            rewards_bfloat16 = np.asarray(
                arm.pop("rewards_bfloat16"), dtype=np.float32
            )
            stds64 = arm.pop("group_population_std_float64")
            stds_bfloat16 = arm.pop("group_population_std_bfloat16")
            arm["response_count"] = int(rewards64.size)
            arm["mean_float64"] = float(rewards64.mean())
            arm["population_std_float64"] = float(rewards64.std(ddof=0))
            arm["mean_bfloat16"] = float(rewards_bfloat16.mean())
            arm["population_std_bfloat16"] = float(
                rewards_bfloat16.std(ddof=0, dtype=np.float32)
            )
            arm["nonzero_advantage_groups_float64"] = sum(value > 0.0 for value in stds64)
            arm["nonzero_advantage_groups_bfloat16"] = sum(
                value > 0.0 for value in stds_bfloat16
            )
    return summary


def _read_groups(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    config_bindings: dict[str, str],
    attempt_index: int,
    generations: int,
    seed_namespace: str = SEED_NAMESPACE,
    seed_manifest_sha256: str | None = None,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    groups = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(groups) > len(rows):
        raise ValueError(f"{path} contains more groups than the locked row set")
    for expected_index, group in enumerate(groups):
        if group.get("schema") != PROBE_SCHEMA or group.get("record_type") != "group":
            raise ValueError(f"unexpected record in {path}")
        expected_row = rows[expected_index]
        checks = {
            **config_bindings,
            "group_index": expected_index,
            "attempt_index": attempt_index,
            "group_size": generations,
            "physical_case_id": expected_row["physical_case_id"],
            "prompt_sha256": expected_row["prompt_sha256"],
        }
        for key, value in checks.items():
            if group.get(key) != value:
                raise ValueError(
                    f"group {expected_index} has a {key} mismatch: {group.get(key)!r} != {value!r}"
                )
        responses = group.get("responses") or []
        if len(responses) != generations:
            raise ValueError(f"group {expected_index} does not contain {generations} responses")
        for response_index, response in enumerate(responses):
            expected_seed = response_seed(
                manifest_sha256=seed_manifest_sha256 or config_bindings["manifest_sha256"],
                physical_case_id=str(expected_row["physical_case_id"]),
                attempt_index=attempt_index,
                response_index=response_index,
                seed_namespace=seed_namespace,
            )
            if response.get("response_index") != response_index:
                raise ValueError(f"group {expected_index} has an out-of-order response")
            if response.get("seed") != expected_seed:
                raise ValueError(f"group {expected_index} has a response seed mismatch")
            if not response.get("prompt_token_ids_sha256"):
                raise ValueError(f"group {expected_index} has no prompt-token hash")
    return groups


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    experiment_path = Path(args.experiment_config)
    experiment_contract = load_experiment_contract(experiment_path)
    sampling = sampling_from_experiment(experiment_contract)
    generations = int(sampling["generations"])
    attempt_indices = [int(value) for value in experiment_contract["attempt_indices"]]
    if args.attempt_index not in attempt_indices:
        raise ValueError("attempt_index is not in the experiment evaluation contract")
    seed_namespace = str(experiment_contract["seed_namespace"])
    if not seed_namespace:
        raise ValueError("experiment seed namespace must not be empty")
    manifest_path = Path(args.manifest)
    all_rows = load_locked_rows(
        manifest_path, expected_role=str(experiment_contract["manifest_role"])
    )
    rows = select_row_shard(
        all_rows,
        shard_index=int(args.shard_index),
        shard_count=int(args.shard_count),
    )
    rows = select_physical_case(rows, args.physical_case_id)
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    manifest_sha256 = file_sha256(manifest_path)
    # Prompt-variant manifests can reuse another manifest's response seeds so arms that
    # differ only in prompt text share seeds case by case.
    seed_manifest_sha256 = experiment_contract["seed_manifest_sha256"] or manifest_sha256
    system_prompt = experiment_contract["system_prompt"] or SYSTEM_PROMPT
    case_concurrency = int(experiment_contract["case_concurrency"])
    policy_artifact = None
    policy_artifact_sha256 = None
    if args.policy_artifact_receipt is not None:
        source_policy_receipt = Path(args.policy_artifact_receipt)
        policy_artifact = json.loads(source_policy_receipt.read_text())
        required_policy_fields = {
            "schema",
            "policy_id",
            "base_model_tree_sha256",
            "tokenizer_json_sha256",
            "checkpoint_tree_sha256",
            "lora_rank",
            "lora_alpha",
            "weight_version",
            "served_model_name",
            "refit_receipt_sha256",
        }
        missing_policy_fields = sorted(required_policy_fields - set(policy_artifact))
        if missing_policy_fields:
            raise ValueError(
                f"the policy artifact receipt is missing {missing_policy_fields}"
            )
        if policy_artifact["served_model_name"] != args.model_name:
            raise ValueError("the served model name does not match the policy receipt")
        if policy_artifact["base_model_tree_sha256"] != args.model_tree_sha256:
            raise ValueError("the base model tree does not match the policy receipt")
        policy_artifact_sha256 = file_sha256(source_policy_receipt)
    model_base_url = args.model_url.rsplit("/v1/", 1)[0]
    stable_model_identity = _resolve_model_identity(
        model_base_url=model_base_url,
        model_name=args.model_name,
        policy_artifact=policy_artifact,
        max_model_len=int(experiment_contract["runtime"]["max_model_len"]),
    )
    verifier_health = _get_json(args.verifier_base_url.rstrip("/") + "/health", timeout=30)
    if int(verifier_health.get("cases", -1)) != args.expected_verifier_cases:
        raise RuntimeError(
            f"verifier has {verifier_health.get('cases')} cases; expected {args.expected_verifier_cases}"
        )
    if verifier_health.get("reward") != experiment_contract["reward"]:
        raise RuntimeError(
            "verifier reward contract does not match the experiment config: "
            f"{verifier_health.get('reward')!r} != {experiment_contract['reward']!r}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=args.resume)
    config_path = out_dir / "probe_config.json"
    groups_path = out_dir / "groups.jsonl"
    summary_path = out_dir / "summary.json"
    seed_table_path = out_dir / "seed_table.jsonl"
    saved_policy_receipt_path = out_dir / "policy-artifact-receipt.json"
    weight_version: int | str = 0
    if args.policy_artifact_receipt is not None:
        source_policy_receipt = Path(args.policy_artifact_receipt)
        assert policy_artifact is not None
        weight_version = policy_artifact["weight_version"]
        if saved_policy_receipt_path.exists():
            if saved_policy_receipt_path.read_bytes() != source_policy_receipt.read_bytes():
                raise RuntimeError("the saved policy artifact receipt does not match")
        else:
            saved_policy_receipt_path.write_bytes(source_policy_receipt.read_bytes())
    seed_table = build_seed_table(
        rows=rows,
        manifest_sha256=seed_manifest_sha256,
        attempt_index=args.attempt_index,
        generations=generations,
        seed_namespace=seed_namespace,
    )
    seed_table_content = jsonl_bytes(seed_table)
    seed_table_sha256 = hashlib.sha256(seed_table_content).hexdigest()
    config = {
        "schema": PROBE_SCHEMA,
        "claim": "no-update policy and verifier variance probe; no replay or optimizer",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_total_rows": len(all_rows),
        "selected_rows": len(rows),
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "physical_case_id_filter": args.physical_case_id,
        "experiment_config": str(experiment_path),
        "experiment_config_sha256": file_sha256(experiment_path),
        "algorithm_contract": {
            key: value
            for key, value in experiment_contract.items()
            if not (key in OPTIONAL_CONTRACT_DEFAULTS and value == OPTIONAL_CONTRACT_DEFAULTS[key])
        },
        "model_url": args.model_url,
        "model_name": args.model_name,
        "stable_model_identity": stable_model_identity,
        "model_tree_sha256": args.model_tree_sha256,
        "verifier_base_url": args.verifier_base_url,
        "verifier_health": verifier_health,
        "case_store_sha256": args.case_store_sha256,
        "solver_image_digest": args.solver_image_digest,
        "attempt_index": args.attempt_index,
        "generations_per_prompt": generations,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "max_tokens": sampling["max_tokens"],
        "enable_thinking": sampling["enable_thinking"],
        "workers": int(experiment_contract["workers"]),
        "model_timeout_seconds": float(experiment_contract["model_timeout_seconds"]),
        "verifier_timeout_seconds": float(experiment_contract["verifier_timeout_seconds"]),
        "transport_retries": int(experiment_contract["transport_retries"]),
        "seed_namespace": seed_namespace,
        "runtime": experiment_contract["runtime"],
        "cross_host_output_equality": experiment_contract["cross_host_output_equality"],
        "minimum_aggregate_saved_output_tokens_per_second": experiment_contract[
            "minimum_aggregate_saved_output_tokens_per_second"
        ],
        "bounded_attempt_indices": attempt_indices,
        "resampling_policy": experiment_contract["resampling_policy"],
        "selection_policy": experiment_contract["selection_policy"],
        "additional_rollout_exposure_control": "same fixed attempts for every reward arm",
        "paper_derived_resampling_rule": False,
        "seed_table_sha256": seed_table_sha256,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "probe_source_sha256": file_sha256(Path(__file__)),
        "policy_artifact": policy_artifact,
        "policy_artifact_sha256": policy_artifact_sha256,
        "weight_version": weight_version,
        "optimizer_enabled": False,
        "replay_enabled": False,
        "refit_enabled": False,
    }
    if experiment_contract["system_prompt"] is not None:
        config["system_prompt"] = system_prompt
    if experiment_contract["seed_manifest_sha256"] is not None:
        config["seed_manifest_sha256"] = seed_manifest_sha256
    if case_concurrency != 1:
        config["case_concurrency"] = case_concurrency
    config["config_sha256"] = canonical_sha256(config)
    config_bindings = {
        "config_sha256": config["config_sha256"],
        "manifest_sha256": manifest_sha256,
        "probe_source_sha256": config["probe_source_sha256"],
        "system_prompt_sha256": config["system_prompt_sha256"],
        "experiment_config_sha256": config["experiment_config_sha256"],
    }
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != config:
            raise RuntimeError("the existing probe configuration does not match this run")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if seed_table_path.exists():
        if seed_table_path.read_bytes() != seed_table_content:
            raise RuntimeError("the existing seed table does not match this run")
    else:
        seed_table_path.write_bytes(seed_table_content)

    groups = _read_groups(
        groups_path,
        rows=rows,
        config_bindings=config_bindings,
        attempt_index=args.attempt_index,
        generations=generations,
        seed_namespace=seed_namespace,
        seed_manifest_sha256=seed_manifest_sha256,
    )
    completed = {str(group["physical_case_id"]) for group in groups}
    probe_kwargs = {
        "manifest_sha256": seed_manifest_sha256,
        "model_url": args.model_url,
        "model_name": args.model_name,
        "verifier_base_url": args.verifier_base_url,
        "max_tokens": sampling["max_tokens"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "enable_thinking": sampling["enable_thinking"],
        "model_timeout": float(experiment_contract["model_timeout_seconds"]),
        "verifier_timeout": float(experiment_contract["verifier_timeout_seconds"]),
        "transport_retries": int(experiment_contract["transport_retries"]),
        "seed_namespace": seed_namespace,
        "system_prompt": system_prompt,
    }
    pending = [
        (group_index, row)
        for group_index, row in enumerate(rows)
        if str(row["physical_case_id"]) not in completed
    ]

    def run_case(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        group_index, row = item
        return probe_case(
            row=row,
            group_index=group_index,
            generations=generations,
            attempt_index=args.attempt_index,
            workers=int(experiment_contract["workers"]),
            config_bindings=config_bindings,
            probe_kwargs=probe_kwargs,
        )

    # Executor.map yields in submission order, so groups are still written in row order.
    with groups_path.open("a") as handle, ThreadPoolExecutor(max_workers=case_concurrency) as cases:
        for group in cases.map(run_case, pending):
            handle.write(json.dumps(group, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            groups.append(group)
            summary = {
                "schema": PROBE_SCHEMA,
                "record_type": "summary",
                "config_sha256": config["config_sha256"],
                "complete": len(groups) == len(rows),
                "expected_group_count": len(rows),
                **summarize_groups(groups),
            }
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "completed_groups": len(groups),
                        "family_id": group["family_id"],
                        "physical_case_id": group["physical_case_id"],
                        "validity_std": group["arms"]["validity"]["population_std_float64"],
                        "trajectory_std": group["arms"]["trajectory"]["population_std_float64"],
                        "full_std": group["arms"]["full"]["population_std_float64"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return json.loads(summary_path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlvp-variance-probe")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--verifier-base-url", required=True)
    parser.add_argument("--expected-verifier-cases", type=int, required=True)
    parser.add_argument("--case-store-sha256", required=True)
    parser.add_argument("--solver-image-digest", required=True)
    parser.add_argument("--model-tree-sha256", required=True)
    parser.add_argument("--policy-artifact-receipt")
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-timeout", type=float, default=900)
    parser.add_argument("--verifier-timeout", type=float, default=900)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--physical-case-id")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_probe(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
