from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def pass_at_k(successes: int, samples: int, k: int) -> float:
    """Unbiased combinatorial pass@k estimator used by the paper."""
    if not 0 <= successes <= samples:
        raise ValueError("successes must be between zero and samples")
    if not 1 <= k <= samples:
        raise ValueError("k must be between one and samples")
    if samples - successes < k:
        return 1.0
    return 1.0 - math.comb(samples - successes, k) / math.comb(samples, k)


@dataclass(frozen=True)
class CaseEvaluation:
    family_id: str
    valid: tuple[bool, ...]
    nrmse: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.valid) != len(self.nrmse) or not self.valid:
            raise ValueError("valid and nrmse must have the same positive length")

    @property
    def success(self) -> tuple[bool, ...]:
        return tuple(
            valid and np.isfinite(error) and error <= 1e-2
            for valid, error in zip(self.valid, self.nrmse, strict=True)
        )

    @property
    def best_error(self) -> float:
        penalties = [
            min(float(error), 1.0) if valid and np.isfinite(error) else 1.0
            for valid, error in zip(self.valid, self.nrmse, strict=True)
        ]
        return min(penalties)


def family_metrics(
    cases: Iterable[CaseEvaluation], k_values: tuple[int, ...] = (1, 4, 8)
) -> dict[str, float]:
    records = tuple(cases)
    if not records:
        raise ValueError("at least one case is required")
    sample_count = len(records[0].valid)
    if any(len(record.valid) != sample_count for record in records):
        raise ValueError("all cases must use the same sample count")
    metrics: dict[str, float] = {
        "valid_rate": float(np.mean([value for record in records for value in record.valid])),
        "median_best_nrmse": float(np.median([record.best_error for record in records])),
    }
    for k in k_values:
        values = [pass_at_k(sum(record.success), sample_count, k) for record in records]
        metrics[f"pass@{k}"] = float(np.mean(values))
    return metrics


def macro_average(family_results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Give every PDE prompt equal weight, as required by the paper."""
    if not family_results:
        raise ValueError("at least one family result is required")
    keys = set(next(iter(family_results.values())))
    if any(set(result) != keys for result in family_results.values()):
        raise ValueError("family metric keys do not match")
    return {
        key: float(np.mean([result[key] for result in family_results.values()]))
        for key in sorted(keys)
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_saved_rollouts(
    *,
    run_dir: Path,
    manifest_path: Path,
    output_path: Path,
    expected_role: str = "rl_validation",
) -> dict[str, object]:
    """Compute the paper metrics from a complete audited rollout group file."""

    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
    ]
    group_path = run_dir / "groups.jsonl"
    groups = [json.loads(line) for line in group_path.read_text().splitlines() if line.strip()]
    if not manifest_rows or len(groups) != len(manifest_rows):
        raise ValueError("rollout groups must cover the complete non-empty manifest")
    records: list[CaseEvaluation] = []
    by_family: dict[str, list[CaseEvaluation]] = {}
    response_count = 0
    success_count = 0
    valid_count = 0
    for index, (row, group) in enumerate(zip(manifest_rows, groups, strict=True)):
        if row.get("role") != expected_role:
            raise ValueError(f"manifest row {index} does not have role {expected_role!r}")
        for key in ("family_id", "physical_case_id", "prompt_sha256"):
            if group.get(key) != row.get(key):
                raise ValueError(f"group {index} has a {key} mismatch")
        responses = group.get("responses") or []
        if not responses:
            raise ValueError(f"group {index} has no responses")
        valid: list[bool] = []
        nrmse: list[float] = []
        for response in responses:
            verifier = response.get("verifier") or {}
            is_valid = float(verifier.get("validity", 0.0)) == 1.0
            if "trajectory_loss_reference_range" not in verifier:
                raise ValueError(
                    f"group {index} has no range-normalized trajectory loss"
                )
            raw_error = verifier["trajectory_loss_reference_range"]
            if raw_error is None:
                error = math.inf
            else:
                error = float(raw_error)
            valid.append(is_valid)
            nrmse.append(error)
            response_count += 1
            valid_count += int(is_valid)
            success_count += int(is_valid and math.isfinite(error) and error <= 1.0e-2)
        record = CaseEvaluation(
            family_id=str(row["family_id"]),
            valid=tuple(valid),
            nrmse=tuple(nrmse),
        )
        records.append(record)
        by_family.setdefault(record.family_id, []).append(record)
    sample_counts = {len(record.valid) for record in records}
    if len(sample_counts) != 1:
        raise ValueError("all evaluation cases must have the same sample count")
    sample_count = sample_counts.pop()
    if sample_count < 8:
        raise ValueError("paper evaluation requires at least eight samples per case")
    family_results = {
        family: family_metrics(family_records)
        for family, family_records in sorted(by_family.items())
    }
    result: dict[str, object] = {
        "schema": "rlvp.paper-evaluation.v1",
        "claim_boundary": (
            "paper metrics on the locked source-derived validation set; not author-case "
            "or author-artifact parity"
        ),
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "groups_sha256": _file_sha256(group_path),
        "case_count": len(records),
        "response_count": response_count,
        "samples_per_case": sample_count,
        "valid_response_count": valid_count,
        "successful_response_count": success_count,
        "success_definition": "valid and finite range-normalized nRMSE <= 0.01",
        "micro": family_metrics(records),
        "macro_equal_family": macro_average(family_results),
        "families": family_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
