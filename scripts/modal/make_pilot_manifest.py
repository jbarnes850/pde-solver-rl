"""Freeze one matched training order from existing locked rows, without reading test data."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlvp.manifests import SEEN_FAMILIES, content_sha256, file_sha256
from rlvp.variance_probe import jsonl_bytes, load_locked_rows, response_seed


def make_pilot_manifest(
    *, manifest_path: Path, output_dir: Path, seed: int = 20261009, pilot_updates: int = 4,
    expected_cases: int = 512, proof_cases_path: Path | None = None,
) -> dict[str, Any]:
    rows = load_locked_rows(manifest_path, expected_role="rl_train")
    if len(rows) != expected_cases or expected_cases % 8 or not 1 <= pilot_updates <= len(rows) // 8:
        raise ValueError("expected complete training cases and a valid pilot update count")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family_id"])].append(row)
    if set(by_family) != set(SEEN_FAMILIES):
        raise ValueError("training manifest must contain exactly the existing eight families")
    rng = random.Random(seed)
    # Sorting before shuffling makes identity independent of incoming JSONL row order.
    for family in SEEN_FAMILIES:
        by_family[family].sort(key=lambda row: str(row["physical_case_id"]))
        rng.shuffle(by_family[family])
    if proof_cases_path is None:
        proof = [by_family[family].pop() for family in SEEN_FAMILIES]
    else:
        prior_rows = [json.loads(line) for line in proof_cases_path.read_text().splitlines()
                      if line.strip()]
        prior_ids = [row.get("physical_case_id", row.get("id")) for row in prior_rows]
        rows_by_id = {row["physical_case_id"]: row for row in rows}
        if len(prior_ids) != 8 or len(set(prior_ids)) != 8 or any(
            case_id not in rows_by_id for case_id in prior_ids
        ):
            raise ValueError("prior proof must select eight distinct existing training cases")
        proof = [rows_by_id[case_id] for case_id in prior_ids]
        if {row["family_id"] for row in proof} != set(SEEN_FAMILIES):
            raise ValueError("prior proof must select exactly one case from each family")
        for row in proof:
            by_family[row["family_id"]].remove(row)
    remainder = [row for family in SEEN_FAMILIES for row in by_family[family]]
    rng.shuffle(remainder)
    ordered = proof + remainder
    source_hash = file_sha256(manifest_path)
    schedule = []
    for index, row in enumerate(ordered):
        occurrence = 0  # This horizon is one pass, with no repeat or zero-reward replacement.
        group_id = content_sha256({"source_manifest_sha256": source_hash,
                                   "physical_case_id": row["physical_case_id"],
                                   "occurrence": occurrence, "schedule_seed": seed})
        schedule.append({
            "group_id": group_id, "group_index": index, "optimizer_update": index // 8 + 1,
            "slot_in_update": index % 8, "physical_case_id": row["physical_case_id"],
            "family_id": row["family_id"], "row_id": row["row_id"],
            "prompt_sha256": row["prompt_sha256"], "occurrence": occurrence,
            "is_proof": index < 8, "is_pilot": index < pilot_updates * 8,
            "response_seeds": [response_seed(
                manifest_sha256=source_hash, physical_case_id=str(row["physical_case_id"]),
                attempt_index=occurrence, response_index=response_index,
                seed_namespace=f"rlvp-modal-matched-training-{seed}",
            ) for response_index in range(8)],
        })
    # Never modify source row contents: order is the only manifest transformation.
    outputs = {"train-ordered.jsonl": jsonl_bytes(ordered), "proof.jsonl": jsonl_bytes(proof),
               "pilot.jsonl": jsonl_bytes(ordered[:pilot_updates * 8]),
               "schedule.jsonl": jsonl_bytes(schedule)}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in outputs.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"refusing to replace a different frozen sequence: {path}")
    for name, payload in outputs.items():
        (output_dir / name).write_bytes(payload)
    receipt = {
        "source_manifest_sha256": source_hash, "seed": seed, "case_count": len(ordered),
        "proof_case_count": 8, "pilot_updates": pilot_updates,
        "proof_selection": "reuse prior training-development proof cases" if proof_cases_path else
        "deterministic selection without reward access",
        "proof_source_sha256": file_sha256(proof_cases_path) if proof_cases_path else None,
        "pilot_case_count": pilot_updates * 8, "samples_per_case": 8,
        "files": {name: file_sha256(output_dir / name) for name in outputs},
        "trial_contract": "Every algorithm/LR trial uses this same order and response seeds, "
        "starts at the same SFT checkpoint, and records failures without replacing cases. "
        "Continue the selected trial at its saved schedule cursor; never repeat its proof cases.",
    }
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def make_continuation_schedule(
    *, ordered_manifest: Path, parent_schedule: Path, output_dir: Path,
    namespace: str, responses_per_prompt: int = 16,
) -> dict[str, Any]:
    """Preserve every training case and order, replacing only branch sampling IDs."""
    rows = load_locked_rows(ordered_manifest, expected_role="rl_train")
    old = [json.loads(line) for line in parent_schedule.read_text().splitlines() if line.strip()]
    if len(rows) != 512 or len(old) != len(rows) or responses_per_prompt != 16:
        raise ValueError("MaxRL continuation requires the complete512-case order and G16")
    if not namespace or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in namespace):
        raise ValueError("continuation requires an explicit safe branch namespace")
    source_hash = file_sha256(ordered_manifest)
    revised = []
    for index, (row, previous) in enumerate(zip(rows, old, strict=True)):
        case_id = str(row["physical_case_id"])
        if previous["physical_case_id"] != case_id or previous["group_index"] != index:
            raise ValueError("parent schedule and training case order differ")
        occurrence = int(previous["occurrence"])
        identities = [f"{namespace}|{source_hash}|{case_id}|{occurrence}|{slot}"
                      for slot in range(responses_per_prompt)]
        seeds = [response_seed(
            manifest_sha256=source_hash, physical_case_id=case_id,
            attempt_index=occurrence, response_index=slot, seed_namespace=namespace,
        ) for slot in range(responses_per_prompt)]
        if len(set(seeds)) != responses_per_prompt:
            raise ValueError("32-bit seed collision within a group; choose a new frozen namespace")
        revised.append({**previous, "parent_group_id": previous["group_id"],
                        "group_id": content_sha256({"namespace": namespace,
                                                    "response_uids": identities}),
                        "seed_namespace": namespace, "response_uids": identities,
                        "response_seeds": seeds, "ordered_manifest_sha256": source_hash})
    payload = jsonl_bytes(revised)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "schedule.jsonl"
    if target.exists() and target.read_bytes() != payload:
        raise ValueError("refusing to replace a different frozen continuation schedule")
    target.write_bytes(payload)
    receipt = {"kind": "training_only_continuation_schedule", "case_count": len(rows),
               "responses_per_prompt": responses_per_prompt, "namespace": namespace,
               "ordered_manifest_sha256": source_hash,
               "parent_schedule_sha256": file_sha256(parent_schedule),
               "schedule_sha256": file_sha256(target), "case_order_unchanged": True,
               "evaluation_schedule_unchanged": True, "full_response_uids_preserved": True}
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--pilot-updates", type=int, default=4)
    parser.add_argument("--proof-cases", type=Path,
                        help="Existing eight-case training manifest or NeMo-Gym JSONL")
    args = parser.parse_args()
    print(json.dumps(make_pilot_manifest(manifest_path=args.manifest, output_dir=args.out,
                                        seed=args.seed, pilot_updates=args.pilot_updates,
                                        proof_cases_path=args.proof_cases), indent=2))


if __name__ == "__main__":
    main()
