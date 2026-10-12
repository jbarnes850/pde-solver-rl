from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .contracts import RewardMode
from .pdes.advection1d import AdvectionCase, make_case
from .pdes.registry import get_environment
from .prompts import SYSTEM_PROMPT, advection_prompt


def materialize_locked_manifest_dataset(
    *,
    manifest_path: Path,
    output_path: Path,
    verifier_base_url: str,
    family_id: str | None = None,
    family_case_ordinal: int | None = None,
    reward_mode: RewardMode = RewardMode.FULL,
    limit: int | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> Path:
    """Create NeMo Gym rows from public locked-manifest fields only."""
    from .variance_probe import load_locked_rows

    rows = load_locked_rows(manifest_path)
    if family_id is not None:
        rows = [row for row in rows if str(row["family_id"]) == family_id]
    if family_case_ordinal is not None:
        if family_case_ordinal < 0:
            raise ValueError("family_case_ordinal must be nonnegative")
        rows = [
            row
            for row in rows
            if int(row["family_case_ordinal"]) == family_case_ordinal
        ]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if not rows:
        raise ValueError("no locked manifest rows match the requested selection")

    public_rows: list[dict[str, Any]] = []
    for row in rows:
        physical_case_id = str(row["physical_case_id"])
        public_rows.append(
            {
                "answer": "__external_rlvp_verifier__",
                "id": physical_case_id,
                "metadata": {
                    "agentic": {
                        "resource_verify_url": (
                            verifier_base_url.rstrip("/")
                            + f"/verify/{reward_mode.value}/{physical_case_id}"
                        )
                    },
                    "manifest_row_id": str(row["row_id"]),
                    "paper_family": str(row["family_id"]),
                    "prompt_sha256": str(row["prompt_sha256"]),
                    "source_dataset": "rlvp_locked_manifest",
                    "split": str(row.get("split") or "train"),
                },
                "responses_create_params": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "input": [
                        {"content": system_prompt, "role": "system"},
                        {"content": str(row["prompt"]), "role": "user"},
                    ],
                },
                "verifier": "rlvp_continuous",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in public_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return output_path


def generate_advection_dataset(
    *,
    count: int,
    seed: int,
    public_dir: Path,
    private_path: Path,
    verifier_base_url: str,
    split: str = "train",
) -> tuple[list[Path], Path]:
    cases = [make_case(seed=seed + index, split=split) for index in range(count)]
    public_dir.mkdir(parents=True, exist_ok=True)
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_text(
        json.dumps({case.case_id: case.to_dict() for case in cases}, indent=2, sort_keys=True)
        + "\n"
    )
    os.chmod(private_path, 0o600)
    outputs: list[Path] = []
    for mode in (RewardMode.VALIDITY, RewardMode.TRAJECTORY, RewardMode.FULL):
        path = public_dir / f"advection1d_{split}_{mode.value}.jsonl"
        with path.open("w") as handle:
            for index, case in enumerate(cases):
                form = ("generic", "parameter", "parameter_ic")[index % 3]
                row = {
                    "id": case.case_id,
                    "responses_create_params": {
                        "input": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": advection_prompt(case, form=form)},
                        ]
                    },
                    "answer": "__external_rlvp_verifier__",
                    "metadata": {
                        "source_dataset": "rlvp_advection1d",
                        "split": split,
                        "paper_family": "1d_linear_advection",
                        "agentic": {
                            "resource_verify_url": (
                                verifier_base_url.rstrip("/")
                                + f"/verify/{mode.value}/{case.case_id}"
                            )
                        },
                    },
                    "verifier": "rlvp_continuous",
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        outputs.append(path)
    return outputs, private_path


def load_case_store(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    records = data.get("cases", data) if isinstance(data, dict) else data
    loaded: dict[str, object] = {}
    for key, value in records.items():
        if "family_id" in value:
            case = get_environment(str(value["family_id"])).case_from_dict(value)
        else:
            case = AdvectionCase.from_dict(value)
        if str(key) != str(case.case_id):
            raise ValueError(f"case-store key does not match case_id: {key}")
        loaded[str(key)] = case
    return loaded
