from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(file_path)))
    return digest.hexdigest()


def prepare_grpo_bootstrap(*, source_checkpoint: Path, output_path: Path) -> dict[str, Any]:
    """Create a weight-only NeMo GRPO bootstrap from an SFT checkpoint."""

    if output_path.exists():
        raise FileExistsError(f"output path already exists: {output_path}")
    required = {
        "config": source_checkpoint / "config.yaml",
        "training_info": source_checkpoint / "training_info.json",
        "dataloader": source_checkpoint / "train_dataloader.pt",
        "weights": source_checkpoint / "policy" / "weights",
        "tokenizer": source_checkpoint / "policy" / "tokenizer",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"source checkpoint is missing {missing}")

    source_training_info = json.loads(required["training_info"].read_text())
    source_epoch_step = int(source_training_info["step"])
    source_total_steps = int(source_training_info.get("total_steps", source_epoch_step))
    if source_total_steps <= 0:
        raise ValueError("source checkpoint total_steps must be positive")

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        shutil.copy2(required["config"], temporary / "config.yaml")
        shutil.copy2(required["dataloader"], temporary / "train_dataloader.pt")
        policy = temporary / "policy"
        shutil.copytree(required["weights"], policy / "weights")
        shutil.copytree(required["tokenizer"], policy / "tokenizer")
        # NeMo's GRPO loops read the GRPOSaveState keys (current_step, current_epoch).
        # The SFT keys (step, epoch) are not read by GRPO and would raise KeyError in
        # async_grpo_train, so the bootstrap writes the GRPO default save state.
        reset_training_info = {
            "consumed_samples": 0,
            "current_epoch": 0,
            "current_step": 0,
            "total_steps": 0,
            "total_valid_tokens": 0,
            "val_reward": -99999999.0,
        }
        (temporary / "training_info.json").write_text(
            json.dumps(reset_training_info, indent=2, sort_keys=True) + "\n"
        )
        if (policy / "optimizer").exists():
            raise RuntimeError("bootstrap must not contain an optimizer")

        receipt = {
            "schema": "rlvp.nemo-grpo-bootstrap.v1",
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_step": source_total_steps,
            "source_epoch_step": source_epoch_step,
            "source_total_steps": source_total_steps,
            "source_training_info_sha256": _file_sha256(required["training_info"]),
            "source_optimizer_present": (source_checkpoint / "policy" / "optimizer").exists(),
            "optimizer_copied": False,
            "weights_tree_sha256": tree_sha256(policy / "weights"),
            "tokenizer_tree_sha256": tree_sha256(policy / "tokenizer"),
            "bootstrap_payload_tree_sha256": tree_sha256(temporary),
            "training_info": reset_training_info,
        }
        (temporary / "bootstrap-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        temporary.replace(output_path)
        return receipt
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
