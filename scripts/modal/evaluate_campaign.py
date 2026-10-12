"""Compare matched saved evaluations; does not sample or inspect training data."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rlvp.evaluation import CaseEvaluation, evaluate_saved_rollouts
from rlvp.manifests import file_sha256
from rlvp.variance_probe import load_locked_rows

MATCHED_CONFIG_FIELDS = (
    "manifest_sha256", "model_tree_sha256", "case_store_sha256", "solver_image_digest",
    "system_prompt_sha256", "generations_per_prompt", "max_tokens", "temperature", "top_p",
    "top_k", "enable_thinking", "seed_namespace", "attempt_index", "workers", "runtime",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired_bootstrap(
    family_case_differences: dict[str, list[float]], *, repeats: int = 10_000, seed: int = 20261009
) -> dict[str, Any]:
    """Resample paired CASES within each family; keep all eight responses together."""
    if repeats < 1000 or not family_case_differences:
        raise ValueError("bootstrap needs families and at least 1000 replicates")
    rng = np.random.default_rng(seed)
    draws = np.zeros(repeats)
    for family in sorted(family_case_differences):
        values = np.asarray(family_case_differences[family], dtype=np.float64)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("family case differences must be nonempty and finite")
        indices = rng.integers(0, len(values), size=(repeats, len(values)))
        draws += values[indices].mean(axis=1) / len(family_case_differences)
    bounds = np.quantile(draws, [0.025, 0.975])
    return {
        "method": "paired percentile case-bootstrap stratified within family; equal family weights",
        "replicates": repeats,
        "seed": seed,
        "confidence": 0.95,
        "lower": float(bounds[0]),
        "upper": float(bounds[1]),
        "unit": "absolute probability difference",
    }


def paired_sign_flip(
    family_case_differences: dict[str, list[float]], *, draws: int = 100_000, seed: int = 20261009
) -> dict[str, Any]:
    """Paired label-swap test; each case contributes its fixed macro-family weight."""
    if not family_case_differences or draws < 1000:
        raise ValueError("sign-flip test needs families and at least 1000 Monte Carlo draws")
    weighted = []
    for family in sorted(family_case_differences):
        values = np.asarray(family_case_differences[family], dtype=np.float64)
        if not values.size or not np.isfinite(values).all():
            raise ValueError("family case differences must be nonempty and finite")
        weighted.extend(values / (len(values) * len(family_case_differences)))
    nonzero = np.asarray([value for value in weighted if value != 0], dtype=np.float64)
    observed = float(nonzero.sum())
    count = len(nonzero)
    exact = count <= 20
    total = 2 ** count if exact else draws
    rng = np.random.default_rng(seed)
    at_least_positive = at_least_absolute = 0
    for start in range(0, total, 4096):
        size = min(4096, total - start)
        if exact:
            codes = np.arange(start, start + size, dtype=np.uint32)
            signs = ((codes[:, None] >> np.arange(count, dtype=np.uint32)) & 1).astype(float) * 2 - 1
        else:
            signs = rng.integers(0, 2, size=(size, count), dtype=np.int8) * 2 - 1
        null = signs @ nonzero
        at_least_positive += int(np.count_nonzero(null >= observed - 1e-14))
        at_least_absolute += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-14))
    correction = 0 if exact else 1
    return {
        "method": "exact paired case sign-flip" if exact else "Monte Carlo paired case sign-flip",
        "null": "baseline/candidate labels exchangeable within case; fixed equal-family weights",
        "alternative": "candidate improves macro pass@1",
        "nonzero_case_count": count, "permutations": total, "seed": None if exact else seed,
        "monte_carlo_add_one_correction": not exact,
        "observed_delta": observed,
        "one_sided_p": (at_least_positive + correction) / (total + correction),
        "two_sided_p": (at_least_absolute + correction) / (total + correction),
    }


def _case(group: dict[str, Any]) -> CaseEvaluation:
    results = [response["verifier"] for response in group["responses"]]
    return CaseEvaluation(
        family_id=str(group["family_id"]),
        valid=tuple(float(result.get("validity", 0)) == 1 for result in results),
        nrmse=tuple(
            float(result["trajectory_loss_reference_range"])
            if result["trajectory_loss_reference_range"] is not None else float("inf")
            for result in results
        ),
    )


def _diagnostics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    responses = [response for group in groups for response in group["responses"]]
    result: dict[str, Any] = {
        "finish_reasons": dict(sorted(Counter(str(r.get("finish_reason")) for r in responses).items())),
        "truncated_response_rate": sum(r.get("finish_reason") == "length" for r in responses)
        / len(responses),
    }
    # Missing optional diagnostic fields remain visibly missing, never zero-filled.
    for key in ("full_reward", "trajectory_reward", "residual_reward"):
        values = [r["verifier"].get(key) for r in responses]
        finite = [float(x) for x in values if x is not None and np.isfinite(float(x))]
        result[key] = {"finite_count": len(finite), "mean": float(np.mean(finite)) if finite else None}
    return result


def compare_evaluations(
    *, baseline_dir: Path, candidate_dir: Path, manifest_path: Path, output_dir: Path,
    role: str = "rl_validation", bootstrap_repeats: int = 10_000,
    bootstrap_seed: int = 20261009, frozen_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    if role not in {"rl_validation", "rl_test"}:
        raise ValueError("comparison accepts validation or a separately frozen test only")
    configs = [json.loads((path / "probe_config.json").read_text())
               for path in (baseline_dir, candidate_dir)]
    if role == "rl_test":
        # Check the selected identity BEFORE opening the test manifest or groups.
        candidate_hash = (configs[1].get("policy_artifact") or {}).get("checkpoint_tree_sha256")
        if not frozen_checkpoint_sha256 or candidate_hash != frozen_checkpoint_sha256:
            raise ValueError("sealed-test comparison requires the preselected checkpoint hash")
    manifest = load_locked_rows(manifest_path, expected_role=role)
    manifest_hash = file_sha256(manifest_path)
    for field in MATCHED_CONFIG_FIELDS:
        if field not in configs[0] or field not in configs[1]:
            raise ValueError(f"missing matching evidence: {field}")
        if configs[0][field] != configs[1][field]:
            raise ValueError(f"evaluation contract mismatch: {field}")
    rewards = [(config.get("verifier_health") or {}).get("reward") for config in configs]
    if not rewards[0] or rewards[0] != rewards[1]:
        raise ValueError("missing or mismatched verifier reward contract")
    expected_sampling = {"max_tokens": 9216, "temperature": .7, "top_p": 1.,
                         "top_k": None, "enable_thinking": False, "workers": 8}
    for key, expected in expected_sampling.items():
        if configs[0][key] != expected:
            raise ValueError(f"evaluation does not use the accepted sampling contract: {key}")
    for key, expected in {"max_model_len": 9728, "quantization": "fp8", "max_num_seqs": 8}.items():
        if configs[0]["runtime"].get(key) != expected:
            raise ValueError(f"evaluation does not use the accepted runtime contract: {key}")
    if configs[0]["manifest_sha256"] != manifest_hash:
        raise ValueError("evaluation manifest hash does not match the supplied manifest")
    if configs[0]["generations_per_prompt"] != 8:
        raise ValueError("campaign evaluation requires exactly eight responses per case")
    all_groups = [_read_jsonl(path / "groups.jsonl") for path in (baseline_dir, candidate_dir)]
    for groups in all_groups:
        if len(groups) != len(manifest):
            raise ValueError("evaluation must cover the complete manifest")
        for row, group in zip(manifest, groups, strict=True):
            for key in ("family_id", "physical_case_id", "prompt_sha256"):
                if row[key] != group.get(key):
                    raise ValueError(f"evaluation case/order mismatch: {key}")
            if group.get("manifest_sha256") != manifest_hash:
                raise ValueError("group manifest hash mismatch")
            if len(group.get("responses", [])) != 8:
                raise ValueError("incomplete response group")
    differences: dict[str, list[float]] = defaultdict(list)
    for before, after in zip(*all_groups, strict=True):
        for index, (left, right) in enumerate(zip(before["responses"], after["responses"], strict=True)):
            for response in (left, right):
                if response.get("response_index") != index or not isinstance(response.get("seed"), int):
                    raise ValueError("missing response index/seed or reordered responses")
                if response.get("model_retries", 0) or response.get("verifier_retries", 0):
                    raise ValueError("retried responses violate the fixed evaluation contract")
            if left["seed"] != right["seed"]:
                raise ValueError("paired response seed mismatch")
        difference = float(np.mean(_case(after).success) - np.mean(_case(before).success))
        differences[str(before["family_id"])].append(difference)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [evaluate_saved_rollouts(
        run_dir=directory, manifest_path=manifest_path, expected_role=role,
        output_path=output_dir / f"{label}-metrics.json",
    ) for directory, label in ((baseline_dir, "baseline"), (candidate_dir, "candidate"))]
    baseline_macro, candidate_macro = [m["macro_equal_family"] for m in metrics]
    delta = {key: candidate_macro[key] - baseline_macro[key] for key in baseline_macro}
    interval = paired_bootstrap(differences, repeats=bootstrap_repeats, seed=bootstrap_seed)
    permutation = paired_sign_flip(differences, seed=bootstrap_seed)
    practical_target = delta["pass@1"] >= 0.02 and delta["valid_rate"] >= -0.02
    significance = permutation["one_sided_p"] <= .05
    result = {
        "role": role,
        "claim_boundary": (
            "Validation used for selection: bootstrap interval is descriptive after selection, "
            "not independent confirmation."
            if role == "rl_validation" else
            "Paired sealed-test comparison of the declared frozen checkpoint; independent "
            "confirmation requires the recorded test-access chronology, not this hash check alone."
        ),
        "frozen_checkpoint_sha256": frozen_checkpoint_sha256,
        "manifest_sha256": manifest_hash,
        "baseline_groups_sha256": file_sha256(baseline_dir / "groups.jsonl"),
        "candidate_groups_sha256": file_sha256(candidate_dir / "groups.jsonl"),
        "baseline_config_sha256": file_sha256(baseline_dir / "probe_config.json"),
        "candidate_config_sha256": file_sha256(candidate_dir / "probe_config.json"),
        "baseline": metrics[0], "candidate": metrics[1], "macro_delta": delta,
        "paired_pass_at_1_interval": interval,
        "paired_pass_at_1_permutation": permutation,
        "target": {"pass_at_1_delta_min": 0.02, "one_sided_permutation_p_max": .05,
                   "validity_delta_min": -0.02, "validity_guard_is_point_estimate": True,
                   "practical_target_met": practical_target,
                   "met": practical_target and significance},
        "statistical_plan": "Before new campaign data: paired sign-flip is the primary "
        "significance test, replacing a positive bootstrap lower-bound gate; percentile bootstrap "
        "is descriptive, especially with few nonzero cases. One selected frozen checkpoint "
        "only may be confirmed on sealed test; validation selection is development evidence.",
        "confirmatory_positive": role == "rl_test" and practical_target and significance,
        "family_deltas": {family: {key: metrics[1]["families"][family][key] - value
                                   for key, value in family_result.items()}
                          for family, family_result in metrics[0]["families"].items()},
        "baseline_diagnostics": _diagnostics(all_groups[0]),
        "candidate_diagnostics": _diagnostics(all_groups[1]),
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=["rl_validation", "rl_test"], default="rl_validation")
    parser.add_argument("--frozen-checkpoint-sha256")
    args = parser.parse_args()
    result = compare_evaluations(
        baseline_dir=args.baseline, candidate_dir=args.candidate, manifest_path=args.manifest,
        output_dir=args.out, role=args.role, frozen_checkpoint_sha256=args.frozen_checkpoint_sha256,
    )
    print(json.dumps({"macro_delta": result["macro_delta"], "target": result["target"],
                      "interval": result["paired_pass_at_1_interval"]}, indent=2))


if __name__ == "__main__":
    main()
