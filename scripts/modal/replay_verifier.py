"""Prepare and replay saved verifier candidates without sampling or training.

Historical responses retain their exact completion and expected reward fields.
Synthetic boundary checks are labeled separately and never enter learner replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verifier_runtime import private_json, request

COMPONENT_FIELDS = (
    "reward",
    "full_reward",
    "trajectory_loss",
    "trajectory_loss_reference_l2",
    "trajectory_loss_reference_range",
    "trajectory_reward",
    "residual_loss",
    "residual_reward",
    "candidate_residual",
    "reference_residual",
)
DISCRETE_FIELDS = ("case_id", "candidate_sha256", "execution_error", "validity", "reward_mode")


def compare(
    expected: dict, actual: dict, *, rtol: float = 1e-8, atol: float = 1e-12
) -> list[dict[str, Any]]:
    differences = []
    for key in DISCRETE_FIELDS + COMPONENT_FIELDS:
        if key not in expected:
            continue
        value = expected[key]
        observed = actual.get(key)
        if key in DISCRETE_FIELDS or value is None or observed is None:
            equal = key in actual and value == observed
        else:
            # Tiny full rewards carry the signal under study: an absolute tolerance
            # must not silently accept their underflow to zero on the new backend.
            effective_atol = 0.0 if key.endswith("reward") else atol
            equal = math.isfinite(float(observed)) and math.isclose(
                float(value), float(observed), rel_tol=rtol, abs_tol=effective_atol
            )
        if not equal:
            differences.append({"field": key, "expected": value, "actual": observed})
    return differences


def prepare_saved(groups: Path, output: Path) -> int:
    """One saved valid candidate per family, plus each observed execution failure."""
    selected = []
    categories = set()
    for line in groups.open():
        group = json.loads(line)
        for response in group["responses"]:
            expected = response["verifier"]
            category = (
                ("valid", group["family_id"])
                if expected["validity"] == 1.0
                else ("invalid", expected["execution_error"])
            )
            if category in categories:
                continue
            categories.add(category)
            selected.append(
                {
                    "fixture_id": f"saved-{len(selected):03}",
                    "evidence_type": "saved-historical-candidate",
                    "category": list(category),
                    "case_id": expected["case_id"],
                    "completion": response["content"],
                    "expected": expected,
                    "source": {
                        "path": str(groups.resolve()),
                        "row_id": group["row_id"],
                        "response_index": response["response_index"],
                    },
                }
            )
    if not selected:
        raise ValueError("no saved candidate fixtures")
    private_json(
        output,
        {
            "schema": "rlvp-verifier-replay-v1",
            "source_sha256": hashlib.sha256(groups.read_bytes()).hexdigest(),
            "fixtures": selected,
        },
    )
    return len(selected)


def add_boundary_checks(case_store: Path, fixture_path: Path) -> int:
    import numpy as np

    from rlvp.contracts import ExecutionResult
    from rlvp.data import load_case_store
    from rlvp.pdes.registry import environment_for_case
    from rlvp.prompts import extract_python_code
    from rlvp.rewards import evaluate_case
    from rlvp.solver_bank.render import render_teacher

    cases = load_case_store(case_store)
    case = next(
        case for case in cases.values() if environment_for_case(case).family_id == "advection1d"
    )
    environment = environment_for_case(case)
    invocation = environment.invocation(case)
    args = ", ".join(invocation.argument_order)
    shape = "(u0_batch.shape[0], len(t_coordinate), u0_batch.shape[1])"
    teacher = render_teacher("advection1d/fourier_exact_shift")
    # This is a trusted, versioned clean-room teacher, never a generated completion.
    namespace: dict[str, Any] = {}
    exec(compile(teacher, "trusted-clean-room-teacher", "exec"), namespace)  # noqa: S102
    inputs = {**invocation.arrays, **invocation.scalars}
    teacher_output = namespace["solver"](*(inputs[name] for name in invocation.argument_order))
    teacher_reward = asdict(
        evaluate_case(
            ExecutionResult(True, True, True, output=np.asarray(teacher_output)), environment, case
        )
    )
    teacher_reward.update(reward=teacher_reward["full_reward"], execution_error=None)
    zero_reward = asdict(evaluate_case(ExecutionResult(False, False, False), environment, case))
    zero_reward["reward"] = 0.0
    shortcut = f"import numpy as np\ndef solver({args}):\n    return np.repeat(u0_batch[:, None, :], len(t_coordinate), axis=1)\n"
    shortcut_output = np.repeat(
        invocation.arrays["u0_batch"][:, None, :], len(invocation.arrays["t_coordinate"]), axis=1
    )
    shortcut_reward = asdict(
        evaluate_case(ExecutionResult(True, True, True, output=shortcut_output), environment, case)
    )
    shortcut_reward.update(reward=shortcut_reward["full_reward"], execution_error=None)
    fixtures = [
        ("teacher", "```python\n" + teacher + "\n```", teacher_reward),
        ("invalid-syntax", "def solver(:\n", {**zero_reward, "execution_error": "solver_exit_1"}),
        (
            "invalid-shape",
            f"import numpy as np\ndef solver({args}):\n    return np.zeros(1)\n",
            {**zero_reward, "execution_error": "invalid_shape_or_nonfinite"},
        ),
        (
            "nonfinite",
            f"import numpy as np\ndef solver({args}):\n    return np.full({shape}, np.nan)\n",
            {**zero_reward, "execution_error": "invalid_shape_or_nonfinite"},
        ),
        (
            "timeout",
            f"import time\ndef solver({args}):\n    time.sleep(60)\n",
            {**zero_reward, "execution_error": "solver_timeout"},
        ),
        ("copy-initial-shortcut", shortcut, shortcut_reward),
        (
            "hidden-reference-isolation",
            "import os\n"
            "assert not os.path.exists('/rlvp-state/cases.json')\n"
            "assert not os.path.exists('/var/run/docker.sock')\n"
            "assert not any(k.startswith(('MODAL_', 'RLVP_VERIFIER_')) for k in os.environ)\n"
            + teacher,
            teacher_reward,
        ),
    ]
    data = json.loads(fixture_path.read_text())
    for name, source, expected in fixtures:
        expected = {
            **expected,
            "case_id": case.case_id,
            "reward_mode": "full",
            "candidate_sha256": hashlib.sha256(extract_python_code(source).encode()).hexdigest(),
        }
        data["fixtures"].append(
            {
                "fixture_id": "boundary-" + name,
                "evidence_type": "synthetic-runtime-contract",
                "category": name,
                "case_id": case.case_id,
                "completion": source,
                "expected": expected,
            }
        )
    private_json(fixture_path, data)
    return len(data["fixtures"])


def replay(
    connection: Path, fixtures: Path, output: Path, *, rtol: float = 1e-8, atol: float = 1e-12
) -> dict[str, Any]:
    credentials = json.loads(connection.read_text())
    data = json.loads(fixtures.read_text())
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for fixture in data["fixtures"]:
        payload = {
            "request_id": "parity-" + fixture["fixture_id"],
            "response": {
                "output": [{"content": [{"type": "output_text", "text": fixture["completion"]}]}]
            },
        }
        actual = request(credentials, "/verify/full/" + fixture["case_id"], payload)
        differences = compare(fixture["expected"], actual, rtol=rtol, atol=atol)
        result = {
            "fixture_id": fixture["fixture_id"],
            "evidence_type": fixture["evidence_type"],
            "passed": not differences,
            "differences": differences,
            "actual": actual,
        }
        private_json(output / (fixture["fixture_id"] + ".json"), result)
        results.append(result)
    summary = {
        "schema": "rlvp-verifier-parity-v1",
        "passed": all(r["passed"] for r in results),
        "total": len(results),
        "failures": sum(not r["passed"] for r in results),
        "rtol": rtol,
        "atol": atol,
        "fixture_sha256": hashlib.sha256(fixtures.read_bytes()).hexdigest(),
        "results": results,
    }
    private_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--groups", type=Path, required=True)
    prep.add_argument("--case-store", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--connection", type=Path, required=True)
    run.add_argument("--fixtures", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_saved(args.groups, args.output)
        print(json.dumps({"fixtures": add_boundary_checks(args.case_store, args.output)}))
    else:
        result = replay(args.connection, args.fixtures, args.output)
        print(json.dumps({key: result[key] for key in ("passed", "total", "failures")}))
        raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
