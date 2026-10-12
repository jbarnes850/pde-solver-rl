from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


def canonical_case_id(family_id: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    prefix = "".join(part[:4] for part in family_id.split("_"))[:12]
    return f"{prefix}-{hashlib.sha256(canonical.encode()).hexdigest()[:20]}"


@dataclass(frozen=True)
class PDECase:
    """Canonical source-derived hidden case description.

    The record stores generator inputs, not reference arrays. Arrays are regenerated from the
    pinned implementation and seed. This format makes every local choice reviewable.
    """

    case_id: str
    family_id: str
    split: str
    seed: int
    grid_shape: tuple[int, ...]
    time_count: int | None
    final_time: float | None
    input_kind: str
    parameters: dict[str, int | float | str]
    generator_version: str = "source-derived-v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PDECase:
        return cls(
            case_id=str(value["case_id"]),
            family_id=str(value["family_id"]),
            split=str(value["split"]),
            seed=int(value["seed"]),
            grid_shape=tuple(int(item) for item in value["grid_shape"]),
            time_count=(None if value.get("time_count") is None else int(value["time_count"])),
            final_time=(None if value.get("final_time") is None else float(value["final_time"])),
            input_kind=str(value["input_kind"]),
            parameters={str(key): item for key, item in value["parameters"].items()},
            generator_version=str(value.get("generator_version", "source-derived-v1")),
        )


def make_case_record(
    *,
    family_id: str,
    split: str,
    seed: int,
    grid_shape: tuple[int, ...],
    time_count: int | None,
    final_time: float | None,
    input_kind: str,
    parameters: dict[str, int | float | str],
) -> PDECase:
    payload: dict[str, Any] = {
        "family_id": family_id,
        "split": split,
        "seed": seed,
        "grid_shape": grid_shape,
        "time_count": time_count,
        "final_time": final_time,
        "input_kind": input_kind,
        "parameters": parameters,
        "generator_version": "source-derived-v1",
    }
    return PDECase(case_id=canonical_case_id(family_id, payload), **payload)
