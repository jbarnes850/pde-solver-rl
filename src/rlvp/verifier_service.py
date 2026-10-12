from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from .contracts import RewardMode
from .data import load_case_store
from .executor import DockerSolverExecutor
from .pdes.registry import environment_for_case
from .prompts import extract_python_code
from .rewards import best_trivial_error, evaluate_case

_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,200}\Z")


def _completion_text(payload: dict[str, Any]) -> str:
    response = payload.get("response") or {}
    output = response.get("output") or []
    pieces: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                pieces.append(str(content.get("text", "")))
    return "\n".join(pieces)


def _request_id(payload: dict[str, Any]) -> str:
    """Read an untrusted public request identifier or make a local one."""

    candidates = [payload.get("request_id"), payload.get("rollout_id")]
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend((metadata.get("request_id"), metadata.get("rollout_id")))
    params = payload.get("responses_create_params")
    if isinstance(params, dict) and isinstance(params.get("metadata"), dict):
        nested = params["metadata"]
        candidates.extend((nested.get("request_id"), nested.get("rollout_id")))
    for candidate in candidates:
        if isinstance(candidate, str) and _SAFE_REQUEST_ID.fullmatch(candidate):
            return candidate
    return f"rlvp-{uuid.uuid4().hex}"


def create_app(
    *,
    case_store_path: Path,
    solver_image: str,
    log_path: Path,
    normalization: str = "reference_range",
    trajectory_temperature: float = 0.05,
    residual_temperature: float = 2.0,
    epsilon: float = 1.0e-12,
    max_concurrency: int = 8,
) -> FastAPI:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    cases = load_case_store(case_store_path)
    executor = DockerSolverExecutor(image=solver_image)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(mode=0o600, exist_ok=True)
    os.chmod(log_path, 0o600)
    log_lock = threading.Lock()
    trivial_lock = threading.Lock()
    trivial_errors: dict[str, float] = {}

    def trivial_error(case_id: str, case: object) -> float:
        # Private: needs the trusted reference. Computed once per case and cached.
        with trivial_lock:
            if case_id not in trivial_errors:
                trivial_errors[case_id] = best_trivial_error(environment_for_case(case), case)
            return trivial_errors[case_id]

    execution_slots = threading.BoundedSemaphore(max_concurrency)
    app = FastAPI(title="RLVP deterministic numerical verifier")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "cases": len(cases),
            "solver_image": solver_image,
            "max_concurrency": max_concurrency,
            "reward": {
                "trajectory_normalization": normalization,
                "trajectory_temperature": trajectory_temperature,
                "residual_temperature": residual_temperature,
                "epsilon": epsilon,
            },
        }

    @app.post("/verify/{mode}/{case_id}")
    def verify(mode: RewardMode, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        case = cases.get(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="unknown RLVP verifier route")
        request_id = _request_id(payload)
        completion = _completion_text(payload)
        candidate_code = extract_python_code(completion)
        candidate_sha256 = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
        with execution_slots:
            execution = executor.execute(completion, case)
        components = evaluate_case(
            execution,
            environment_for_case(case),
            case,
            normalization=normalization,
            trajectory_temperature=trajectory_temperature,
            residual_temperature=residual_temperature,
            epsilon=epsilon,
            best_trivial=trivial_error(case_id, case),
        )
        result = {
            "reward": components.scalar(mode),
            "reward_mode": mode.value,
            "validity": components.validity,
            "trajectory_loss": components.trajectory_loss,
            "trajectory_reward": components.trajectory_reward,
            "trajectory_loss_reference_l2": components.trajectory_loss_reference_l2,
            "trajectory_loss_reference_range": components.trajectory_loss_reference_range,
            "residual_loss": components.residual_loss,
            "residual_reward": components.residual_reward,
            "full_reward": components.full_reward,
            "best_trivial_error": components.best_trivial_error,
            "anchored_skill": components.anchored_skill,
            "anchored_reward": components.anchored_reward,
            "candidate_residual": components.candidate_residual,
            "reference_residual": components.reference_residual,
            "case_id": case_id,
            "request_id": request_id,
            "candidate_sha256": candidate_sha256,
            "execution_error": execution.error,
            "execution_seconds": execution.duration_seconds,
        }
        with log_lock, log_path.open("a") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
        return result

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-store",
        default=os.environ.get("RLVP_CASE_STORE", ".artifacts/private/advection1d_cases.json"),
    )
    parser.add_argument(
        "--solver-image", default=os.environ.get("RLVP_SOLVER_IMAGE", "rlvp-solver:py312")
    )
    parser.add_argument(
        "--log", default=os.environ.get("RLVP_VERIFIER_LOG", ".artifacts/logs/verifier.jsonl")
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19240)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--normalization", choices=("reference_l2", "reference_range"), default="reference_range"
    )
    parser.add_argument("--trajectory-temperature", type=float, default=0.05)
    parser.add_argument("--residual-temperature", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1.0e-12)
    args = parser.parse_args()
    uvicorn.run(
        create_app(
            case_store_path=Path(args.case_store),
            solver_image=args.solver_image,
            log_path=Path(args.log),
            normalization=args.normalization,
            trajectory_temperature=args.trajectory_temperature,
            residual_temperature=args.residual_temperature,
            epsilon=args.epsilon,
            max_concurrency=args.max_concurrency,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
