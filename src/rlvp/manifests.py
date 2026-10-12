from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

BENCHMARK_ID = "rlvp-source-derived-v1"
SCHEMA_VERSION = "rlvp.manifest.v1"
PAPER_SPLIT_SEED = 1234
PAPER_EVALUATION_SAMPLE_SEED = 2026
SFT_TEACHERS_PER_PROMPT = 4

PROMPT_FORMS = ("generic", "parameter", "parameter_ic")

SEEN_FAMILIES = (
    "advection1d",
    "burgers1d",
    "reaction_diffusion1d",
    "diffusion_sorption1d",
    "reaction_diffusion2d",
    "darcy2d",
    "incompressible_ns2d",
    "shallow_water2d",
)

HELD_OUT_FAMILIES = (
    "advection2d",
    "advection_diffusion1d",
    "advection_diffusion2d",
    "advection_reaction_diffusion1d",
    "allen_cahn2d",
    "cahn_hilliard1d",
    "darcy_reaction2d",
    "gray_scott2d",
    "heat1d",
    "kdv1d",
)

# Extra training-only families added after the paper split. They are never held out, have
# no paper method bank and are not in SEEN_FAMILIES, so every paper manifest count, SFT
# bank rule and leakage check above is unchanged.
EXTRA_TRAIN_FAMILIES = (
    "advection1d_domain",
    "airy1d",
    "hyperdiffusion1d",
)

# Exact active-paper training prompt counts (Appendix, Table "Per-PDE prompt counts").
PAPER_TRAIN_PROMPT_MATRIX: dict[str, dict[str, int]] = {
    "advection1d": {"generic": 30, "parameter": 29, "parameter_ic": 9},
    "burgers1d": {"generic": 35, "parameter": 20, "parameter_ic": 8},
    "reaction_diffusion1d": {"generic": 36, "parameter": 21, "parameter_ic": 6},
    "diffusion_sorption1d": {"generic": 25, "parameter": 20, "parameter_ic": 11},
    "reaction_diffusion2d": {"generic": 34, "parameter": 22, "parameter_ic": 13},
    "darcy2d": {"generic": 27, "parameter": 29, "parameter_ic": 12},
    "incompressible_ns2d": {"generic": 29, "parameter": 23, "parameter_ic": 14},
    "shallow_water2d": {"generic": 21, "parameter": 23, "parameter_ic": 15},
}

# The paper gives only the 64/128 totals. Equal family allocation is a declared local choice.
VALIDATION_CASE_COUNTS = {family_id: 8 for family_id in SEEN_FAMILIES}
TEST_CASE_COUNTS = {family_id: 16 for family_id in SEEN_FAMILIES}

# These allocations satisfy the published total and every rounded per-family result. They are
# the most-even feasible allocations with a canonical-family tie-break. They are not claimed to
# be the unreleased author manifests.
SEEN_EVALUATION_CASE_COUNTS = {
    "advection1d": 16,
    "burgers1d": 16,
    "reaction_diffusion1d": 19,
    "diffusion_sorption1d": 18,
    "reaction_diffusion2d": 18,
    "darcy2d": 18,
    "incompressible_ns2d": 18,
    "shallow_water2d": 20,
}

HELD_OUT_EVALUATION_CASE_COUNTS = {
    "advection2d": 12,
    "advection_diffusion1d": 24,
    "advection_diffusion2d": 12,
    "advection_reaction_diffusion1d": 16,
    "allen_cahn2d": 12,
    "cahn_hilliard1d": 12,
    "darcy_reaction2d": 14,
    "gray_scott2d": 12,
    "heat1d": 14,
    "kdv1d": 9,
}

PAPER_METHOD_COUNTS = {
    "advection1d": 8,
    "burgers1d": 18,
    "reaction_diffusion1d": 6,
    "diffusion_sorption1d": 8,
    "reaction_diffusion2d": 6,
    "darcy2d": 8,
    "incompressible_ns2d": 10,
    "shallow_water2d": 7,
}

ASSUMPTIONS = {
    "A-SEED-001": ("Case seeds use a counter-based SHA-256 derivation from the paper split seed."),
    "A-SPLIT-001": (
        "Validation and test cases use equal allocations of 8 and 16 cases per seen family."
    ),
    "A-SFT-001": (
        "Each of the 512 RL prompt rows receives four distinct compatible teacher methods."
    ),
    "A-EVAL-SEEN-001": (
        "The 143 seen cases use the most-even count vector compatible with rounded tables."
    ),
    "A-EVAL-OOD-001": (
        "The 137 held-out cases use the most-even count vector compatible with rounded tables."
    ),
}


def canonical_json(value: Any) -> str:
    """Return the only JSON encoding used for content hashes and JSONL locks."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(kind: str, identity: Mapping[str, Any]) -> str:
    return f"{kind}:{content_sha256({'kind': kind, **identity})}"


def _stable_seed(*, role: str, family_id: str, ordinal: int) -> int:
    payload = {
        "benchmark_id": BENCHMARK_ID,
        "family_id": family_id,
        "master_seed": PAPER_SPLIT_SEED,
        "ordinal": ordinal,
        "role": role,
    }
    # Stay in the positive signed 63-bit range accepted by NumPy and runtime interfaces.
    return (int(content_sha256(payload)[:16], 16) & ((1 << 63) - 1)) or 1


def _case_id(*, role: str, family_id: str, ordinal: int, case_seed: int) -> str:
    return _stable_id(
        "case",
        {
            "benchmark_id": BENCHMARK_ID,
            "case_seed": case_seed,
            "family_id": family_id,
            "ordinal": ordinal,
            "role": role,
        },
    )


def _prompt_id(*, purpose: str, family_id: str, prompt_form: str, ordinal: int = 0) -> str:
    return _stable_id(
        "prompt",
        {
            "benchmark_id": BENCHMARK_ID,
            "family_id": family_id,
            "ordinal": ordinal,
            "prompt_form": prompt_form,
            "purpose": purpose,
        },
    )


def _seal_row(row: dict[str, Any]) -> dict[str, Any]:
    if "row_id" in row:
        raise ValueError("row must not contain row_id before sealing")
    return {**row, "row_id": _stable_id("row", row)}


def _case_row(
    *,
    role: str,
    split: str,
    family_id: str,
    ordinal: int,
    prompt_id: str,
    prompt_form: str,
    assumption_ids: Sequence[str],
) -> dict[str, Any]:
    case_seed = _stable_seed(role=role, family_id=family_id, ordinal=ordinal)
    return _seal_row(
        {
            "assumption_ids": list(assumption_ids),
            "benchmark_id": BENCHMARK_ID,
            "case_id": _case_id(
                role=role,
                family_id=family_id,
                ordinal=ordinal,
                case_seed=case_seed,
            ),
            "case_seed": case_seed,
            "family_case_ordinal": ordinal,
            "family_id": family_id,
            "prompt_form": prompt_form,
            "prompt_id": prompt_id,
            "role": role,
            "schema_version": SCHEMA_VERSION,
            "split": split,
        }
    )


def _normalized_method_bank(
    method_bank: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, ...]]:
    if method_bank is None:
        from .pdes.registry import get_environment

        loaded: dict[str, Sequence[str]] = {}
        for family_id in SEEN_FAMILIES:
            environment = get_environment(family_id)
            if not environment.seen_in_training:
                raise ValueError(f"seen family is not training-enabled: {family_id}")
            loaded[family_id] = environment.method_ids
        method_bank = loaded

    if set(method_bank) != set(SEEN_FAMILIES):
        missing = sorted(set(SEEN_FAMILIES) - set(method_bank))
        extra = sorted(set(method_bank) - set(SEEN_FAMILIES))
        raise ValueError(f"method bank family mismatch; missing={missing}, extra={extra}")

    normalized: dict[str, tuple[str, ...]] = {}
    for family_id in SEEN_FAMILIES:
        methods = tuple(sorted(str(value) for value in method_bank[family_id]))
        if len(methods) != len(set(methods)):
            raise ValueError(f"duplicate method ID in {family_id}")
        expected = PAPER_METHOD_COUNTS[family_id]
        if len(methods) != expected:
            raise ValueError(
                f"{family_id} must expose {expected} paper-listed methods, got {len(methods)}"
            )
        if len(methods) < SFT_TEACHERS_PER_PROMPT:
            raise ValueError(f"{family_id} has fewer than four compatible teacher methods")
        normalized[family_id] = methods
    if sum(map(len, normalized.values())) != 71:
        raise AssertionError("the source-derived SFT bank must contain 71 family-method entries")
    return normalized


def _build_training_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_id in SEEN_FAMILIES:
        family_ordinal = 0
        for prompt_form in PROMPT_FORMS:
            count = PAPER_TRAIN_PROMPT_MATRIX[family_id][prompt_form]
            for _ in range(count):
                rows.append(
                    _case_row(
                        role="rl_train",
                        split="train",
                        family_id=family_id,
                        ordinal=family_ordinal,
                        prompt_id=_prompt_id(
                            purpose="rl_train",
                            family_id=family_id,
                            prompt_form=prompt_form,
                            ordinal=family_ordinal,
                        ),
                        prompt_form=prompt_form,
                        assumption_ids=("A-SEED-001",),
                    )
                )
                family_ordinal += 1
    return rows


def _build_fixed_case_rows(
    *,
    role: str,
    split: str,
    case_counts: Mapping[str, int],
    assumption_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_id, count in case_counts.items():
        prompt_id = _prompt_id(
            purpose=role,
            family_id=family_id,
            prompt_form="generic",
        )
        for ordinal in range(count):
            rows.append(
                _case_row(
                    role=role,
                    split=split,
                    family_id=family_id,
                    ordinal=ordinal,
                    prompt_id=prompt_id,
                    prompt_form="generic",
                    assumption_ids=("A-SEED-001", *assumption_ids),
                )
            )
    return rows


def _build_sft_rows(
    training_rows: Sequence[Mapping[str, Any]],
    method_bank: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_ordinals: Counter[str] = Counter()
    for prompt in training_rows:
        family_id = str(prompt["family_id"])
        methods = method_bank[family_id]
        rotation = int(content_sha256(prompt["prompt_id"])[:16], 16) % len(methods)
        for teacher_slot in range(SFT_TEACHERS_PER_PROMPT):
            ordinal = family_ordinals[family_id]
            family_ordinals[family_id] += 1
            case_seed = _stable_seed(role="sft", family_id=family_id, ordinal=ordinal)
            row = {
                "assumption_ids": ["A-SEED-001", "A-SFT-001"],
                "benchmark_id": BENCHMARK_ID,
                "case_id": _case_id(
                    role="sft",
                    family_id=family_id,
                    ordinal=ordinal,
                    case_seed=case_seed,
                ),
                "case_seed": case_seed,
                "family_case_ordinal": ordinal,
                "family_id": family_id,
                "prompt_form": prompt["prompt_form"],
                "prompt_id": prompt["prompt_id"],
                "role": "sft",
                "schema_version": SCHEMA_VERSION,
                "source_prompt_id": prompt["prompt_id"],
                "split": "sft_train",
                "teacher_method_id": methods[(rotation + teacher_slot) % len(methods)],
                "teacher_slot": teacher_slot,
            }
            rows.append(_seal_row(row))
    return rows


def build_manifest_bundle(
    *, method_bank: Mapping[str, Sequence[str]] | None = None
) -> dict[str, Any]:
    """Build the locked source-derived data contract without materializing PDE arrays."""
    normalized_bank = _normalized_method_bank(method_bank)
    training_rows = _build_training_rows()
    manifests = {
        "held_out_evaluation": _build_fixed_case_rows(
            role="held_out_evaluation",
            split="held_out_evaluation",
            case_counts=HELD_OUT_EVALUATION_CASE_COUNTS,
            assumption_ids=("A-EVAL-OOD-001",),
        ),
        "rl_test": _build_fixed_case_rows(
            role="rl_test",
            split="test",
            case_counts=TEST_CASE_COUNTS,
            assumption_ids=("A-SPLIT-001",),
        ),
        "rl_train": training_rows,
        "rl_validation": _build_fixed_case_rows(
            role="rl_validation",
            split="validation",
            case_counts=VALIDATION_CASE_COUNTS,
            assumption_ids=("A-SPLIT-001",),
        ),
        "seen_evaluation": _build_fixed_case_rows(
            role="seen_evaluation",
            split="seen_evaluation",
            case_counts=SEEN_EVALUATION_CASE_COUNTS,
            assumption_ids=("A-EVAL-SEEN-001",),
        ),
        "sft": _build_sft_rows(training_rows, normalized_bank),
    }
    payload: dict[str, Any] = {
        "assumptions": dict(sorted(ASSUMPTIONS.items())),
        "benchmark_id": BENCHMARK_ID,
        "claim_boundary": "source-derived; not an author-artifact reproduction",
        "evaluation_protocol": {
            "max_new_tokens": 3072,
            "pass_k": [1, 4, 8],
            "sample_seed": PAPER_EVALUATION_SAMPLE_SEED,
            "samples_per_prompt": 8,
            "temperature": 0.7,
            "top_p": 1.0,
        },
        "manifest_digests": {
            name: content_sha256(rows) for name, rows in sorted(manifests.items())
        },
        "manifests": manifests,
        "method_bank": {key: list(value) for key, value in normalized_bank.items()},
        "paper_split_seed": PAPER_SPLIT_SEED,
        "schema_version": SCHEMA_VERSION,
    }
    payload["bundle_digest"] = content_sha256(payload)
    validate_manifest_bundle(payload)
    return payload


def materialize_physical_manifests(
    bundle: Mapping[str, Any],
    *,
    public_dir: Path,
    private_dir: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Bind logical manifest rows to complete generated PDE cases.

    Public rows contain prompts and stable route IDs. Complete physical parameters stay in
    mode-600 case stores used by the verifier. The private stores are transparent research
    artifacts, but they must never be mounted into the policy runtime.
    """
    validate_manifest_bundle(bundle)
    from .pdes.registry import get_environment

    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, Any] = {}
    for manifest_name, rows in sorted(bundle["manifests"].items()):
        public_path = public_dir / f"{manifest_name}.jsonl"
        private_path = private_dir / f"{manifest_name}.cases.json"
        public_rows: list[dict[str, Any]] = []
        cases: dict[str, dict[str, Any]] = {}
        for logical in rows:
            family_id = str(logical["family_id"])
            environment = get_environment(family_id)
            case = environment.make_case(
                seed=int(logical["case_seed"]), split=str(logical["split"])
            )
            prompt = environment.prompt(case, form=str(logical["prompt_form"]))
            public_rows.append(
                {
                    **logical,
                    "generator_version": str(
                        getattr(case, "generator_version", "advection-source-derived-v1")
                    ),
                    "physical_case_id": str(case.case_id),
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "reference_operator_id": f"{family_id}.primary.source-derived-v1",
                    "residual_operator_id": f"{family_id}.residual.source-derived-v1",
                }
            )
            cases[case.case_id] = {**case.to_dict(), "family_id": family_id}
        with public_path.open("w") as handle:
            for row in public_rows:
                handle.write(canonical_json(row) + "\n")
        private_path.write_text(
            json.dumps(
                {
                    "benchmark_id": BENCHMARK_ID,
                    "cases": dict(sorted(cases.items())),
                    "manifest": manifest_name,
                    "schema": "rlvp.private-case-store.v1",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.chmod(private_path, 0o600)
        receipts[manifest_name] = {
            "case_count": len(cases),
            "private_file": str(private_path),
            "private_sha256": file_sha256(private_path),
            "public_file": str(public_path),
            "public_sha256": file_sha256(public_path),
            "row_count": len(public_rows),
        }
    receipt: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "bundle_digest": bundle["bundle_digest"],
        "claim_boundary": "source-derived locked cases; not the unreleased author cases",
        "manifests": receipts,
        "schema": "rlvp.physical-manifest-receipt.v1",
    }
    receipt["receipt_digest"] = content_sha256(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _expected_row_id(row: Mapping[str, Any]) -> str:
    value = dict(row)
    value.pop("row_id", None)
    return _stable_id("row", value)


def validate_manifest_bundle(bundle: Mapping[str, Any]) -> None:
    """Fail closed if counts, provenance, IDs, compatibility, or locks drift."""
    if bundle.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("wrong benchmark identity")
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("wrong manifest schema")
    if bundle.get("paper_split_seed") != PAPER_SPLIT_SEED:
        raise ValueError("paper split seed drifted")
    expected_protocol = {
        "max_new_tokens": 3072,
        "pass_k": [1, 4, 8],
        "sample_seed": PAPER_EVALUATION_SAMPLE_SEED,
        "samples_per_prompt": 8,
        "temperature": 0.7,
        "top_p": 1.0,
    }
    if bundle.get("evaluation_protocol") != expected_protocol:
        raise ValueError("evaluation protocol drifted")

    normalized_bank = _normalized_method_bank(bundle.get("method_bank"))
    manifests = bundle.get("manifests")
    if not isinstance(manifests, Mapping):
        raise TypeError("manifests must be a mapping")
    expected_sizes = {
        "held_out_evaluation": 137,
        "rl_test": 128,
        "rl_train": 512,
        "rl_validation": 64,
        "seen_evaluation": 143,
        "sft": 2048,
    }
    if set(manifests) != set(expected_sizes):
        raise ValueError("manifest names drifted")
    for name, expected in expected_sizes.items():
        if len(manifests[name]) != expected:
            raise ValueError(f"{name} must contain {expected} rows")

    train_counts = Counter(
        (str(row["family_id"]), str(row["prompt_form"])) for row in manifests["rl_train"]
    )
    expected_train_counts = Counter(
        {
            (family_id, prompt_form): count
            for family_id, forms in PAPER_TRAIN_PROMPT_MATRIX.items()
            for prompt_form, count in forms.items()
        }
    )
    if train_counts != expected_train_counts:
        raise ValueError("the exact paper 512-row prompt matrix drifted")

    count_contracts = {
        "rl_validation": VALIDATION_CASE_COUNTS,
        "rl_test": TEST_CASE_COUNTS,
        "seen_evaluation": SEEN_EVALUATION_CASE_COUNTS,
        "held_out_evaluation": HELD_OUT_EVALUATION_CASE_COUNTS,
    }
    for name, expected in count_contracts.items():
        observed = Counter(str(row["family_id"]) for row in manifests[name])
        if observed != Counter(expected):
            raise ValueError(f"{name} family counts drifted")

    train_prompt_ids = {str(row["prompt_id"]) for row in manifests["rl_train"]}
    if len(train_prompt_ids) != 512:
        raise ValueError("RL training prompt IDs must be unique")
    sft_by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifests["sft"]:
        source_prompt_id = str(row["source_prompt_id"])
        sft_by_prompt[source_prompt_id].append(row)
        family_id = str(row["family_id"])
        if str(row["teacher_method_id"]) not in normalized_bank[family_id]:
            raise ValueError(f"incompatible SFT teacher for {family_id}")
    if set(sft_by_prompt) != train_prompt_ids:
        raise ValueError("SFT must cover every RL training prompt exactly")
    for prompt_id, rows in sft_by_prompt.items():
        if len(rows) != SFT_TEACHERS_PER_PROMPT:
            raise ValueError(f"SFT prompt {prompt_id} does not have four teachers")
        methods = {str(row["teacher_method_id"]) for row in rows}
        if len(methods) != SFT_TEACHERS_PER_PROMPT:
            raise ValueError(f"SFT prompt {prompt_id} has duplicate teachers")

    seen = set(SEEN_FAMILIES)
    held_out = set(HELD_OUT_FAMILIES)
    for name in ("sft", "rl_train", "rl_validation", "rl_test"):
        families = {str(row["family_id"]) for row in manifests[name]}
        if not families <= seen or families & held_out:
            raise ValueError(f"held-out family leaked into {name}")
    if {str(row["family_id"]) for row in manifests["seen_evaluation"]} != seen:
        raise ValueError("seen evaluation family set drifted")
    if {str(row["family_id"]) for row in manifests["held_out_evaluation"]} != held_out:
        raise ValueError("held-out evaluation family set drifted")

    case_owner: dict[str, str] = {}
    row_ids: set[str] = set()
    for name, rows in manifests.items():
        for row in rows:
            if row.get("benchmark_id") != BENCHMARK_ID:
                raise ValueError(f"row benchmark drifted in {name}")
            if row.get("schema_version") != SCHEMA_VERSION:
                raise ValueError(f"row schema drifted in {name}")
            expected_case_id = _case_id(
                role=str(row["role"]),
                family_id=str(row["family_id"]),
                ordinal=int(row["family_case_ordinal"]),
                case_seed=int(row["case_seed"]),
            )
            if row.get("case_id") != expected_case_id:
                raise ValueError(f"case ID hash drifted in {name}")
            case_id = str(row["case_id"])
            if case_id in case_owner:
                raise ValueError(
                    f"case ID overlap between {case_owner[case_id]} and {name}: {case_id}"
                )
            case_owner[case_id] = name
            expected_row_id = _expected_row_id(row)
            if row.get("row_id") != expected_row_id or expected_row_id in row_ids:
                raise ValueError(f"row ID hash or uniqueness failure in {name}")
            row_ids.add(expected_row_id)

    expected_digests = {name: content_sha256(rows) for name, rows in sorted(manifests.items())}
    if bundle.get("manifest_digests") != expected_digests:
        raise ValueError("manifest digest drifted")
    without_digest = dict(bundle)
    observed_bundle_digest = without_digest.pop("bundle_digest", None)
    if observed_bundle_digest != content_sha256(without_digest):
        raise ValueError("bundle digest drifted")


def write_manifest_bundle(bundle: Mapping[str, Any], output_dir: Path) -> Path:
    """Write canonical JSONL files and a file-level lock receipt."""
    validate_manifest_bundle(bundle)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = bundle["manifests"]
    file_receipts: dict[str, dict[str, Any]] = {}
    for name in sorted(manifests):
        path = output_dir / f"{name}.jsonl"
        encoded = "".join(f"{canonical_json(row)}\n" for row in manifests[name]).encode()
        path.write_bytes(encoded)
        file_receipts[name] = {
            "count": len(manifests[name]),
            "file": path.name,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    lock: dict[str, Any] = {
        "benchmark_id": BENCHMARK_ID,
        "bundle_digest": bundle["bundle_digest"],
        "claim_boundary": bundle["claim_boundary"],
        "files": file_receipts,
        "schema_version": SCHEMA_VERSION,
    }
    lock["lock_digest"] = content_sha256(lock)
    lock_path = output_dir / "manifest-lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock_path
