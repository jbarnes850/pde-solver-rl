"""Fail-closed numerical and tensor receipts for the bounded Modal RLVP run.

Installed into nemo_rl.utils. CPU-only helpers are deliberately testable without
NeMo. This module neither samples nor modifies rewards or optimizer parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


def _cpu(value):
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        if hasattr(value, "to_local"):
            value = value.to_local()
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(k): _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu(v) for v in value]
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return _cpu(value.tolist() if hasattr(value, "tolist") else value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported receipt value {type(value).__name__}")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save(kind, identity, tensors, metadata):
    root = os.environ.get("NRL_AUDIT_DIR")
    if not root:
        raise RuntimeError("NRL_AUDIT_DIR must be set for the Modal experiment")
    directory = Path(root) / kind
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{identity}.pt"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite an existing receipt: {target}")
    temporary = target.with_suffix(".tmp")
    torch.save(_cpu(tensors), temporary)
    os.replace(temporary, target)
    receipt = {"kind": kind, "path": str(target), "sha256": _sha(target), **metadata}
    manifest = target.with_suffix(".json")
    manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print("[rlvp_receipt] " + json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def group_statistics_fp32(prompt_ids, rewards, leave_one_out=False):
    """Centered sample statistics, FP32 throughout, with scale-safe variance.

    Scaling avoids squaring tiny rewards before normalization. It changes neither
    the reward nor the sample-standard-deviation formula, and does not bypass the
    declared 1e-6 denominator epsilon. Constant/singleton groups have zero std.
    """
    rewards = rewards.float()
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("nonfinite physics reward")
    _, groups = torch.unique(prompt_ids.to(rewards.device), dim=0, return_inverse=True)
    baseline = torch.empty_like(rewards)
    std = torch.zeros_like(rewards)
    for group in range(int(groups.max().item()) + 1):
        indices = groups == group
        values = rewards[indices]
        if leave_one_out and values.numel() > 1:
            raise ValueError("this experiment requires group-mean, not LOO, advantages")
        baseline[indices] = values.mean()
        if values.numel() > 1:
            centered = values - values.mean()
            scale = centered.abs().max()
            if scale > 0:
                std[indices] = (
                    (centered / scale).square().sum() / (values.numel() - 1)
                ).sqrt() * scale
    return baseline, std


def archive_batch(train_data, rewards, prompt_ids, step):
    # Direct worker probes can mix CUDA inputs with CPU logprob outputs.
    # Normalize the archival copy before mask-indexing; never mutate live data.
    tensors = _cpu(dict(train_data))
    tensors["rewards"] = _cpu(rewards).float()
    tensors["trajectory_group_ids"] = _cpu(prompt_ids)
    required = (
        "input_ids",
        "generation_logprobs",
        "prev_logprobs",
        "reference_policy_logprobs",
        "token_mask",
        "sample_mask",
        "advantages",
    )
    missing = [key for key in required if key not in tensors]
    if missing:
        raise ValueError(f"missing training receipt fields: {missing}")
    mask = tensors["token_mask"].bool() & tensors["sample_mask"].bool().unsqueeze(-1)
    for key in ("generation_logprobs", "prev_logprobs", "reference_policy_logprobs", "advantages"):
        if not bool(torch.isfinite(tensors[key][mask]).all()):
            raise ValueError(f"nonfinite masked {key}")
    return _save(
        "batches",
        f"step-{int(step):06d}",
        tensors,
        {
            "step": int(step),
            "rows": int(rewards.numel()),
            "masked_tokens": int(mask.sum().item()),
            "advantage_dtype": str(tensors["advantages"].dtype),
        },
    )


def _trainable(model):
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def trainable_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in _trainable(model):
        value = _cpu(parameter).contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def archive_gradients(model, optimizer, step, dp_size=1, cp_size=1):
    if dp_size != 1 or cp_size != 1:
        raise ValueError("gradient receipt is qualified only for one learner rank")
    gradients = {}
    missing = []
    for name, parameter in _trainable(model):
        if parameter.grad is None:
            missing.append(name)
        else:
            grad = _cpu(parameter.grad).float()
            if not bool(torch.isfinite(grad).all()):
                raise ValueError(f"nonfinite gradient: {name}")
            gradients[name] = grad
    if not gradients:
        raise ValueError("optimizer update has no trainable gradients")
    squared_norm = sum(float(g.double().square().sum()) for g in gradients.values())
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    receipt = _save(
        "gradients",
        f"step-{int(step):06d}-rank-{rank}",
        gradients,
        {
            "step": int(step),
            "phase": "accumulated_pre_clip",
            "grad_l2": squared_norm**0.5,
            "parameter_count": sum(g.numel() for g in gradients.values()),
            "missing_gradient_names": missing,
            "trainable_weights_sha256_before": trainable_sha256(model),
            "learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
        },
    )
    receipt["_weights_before"] = {name: _cpu(parameter) for name, parameter in _trainable(model)}
    return receipt


def finish_gradient_receipt(receipt, model, clip_max_norm, reported_grad_norm):
    # At dp=cp=1 and num_label_tokens=1 the NeMo helper's pre-clip scaling is 1.
    value = float(reported_grad_norm)
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError("nonfinite reported gradient norm")
    before_weights = receipt.pop("_weights_before")
    receipt = dict(receipt)
    update_squared = 0.0
    weight_squared = 0.0
    for name, parameter in _trainable(model):
        before = before_weights.pop(name).double()
        after = _cpu(parameter).double()
        update_squared += float((after - before).square().sum())
        weight_squared += float(before.square().sum())
    receipt["trainable_update_l2"] = update_squared**0.5
    receipt["trainable_relative_update_l2"] = (
        (update_squared / weight_squared) ** 0.5 if weight_squared else None
    )
    post_clip_squared = 0.0
    for _, parameter in _trainable(model):
        if parameter.grad is not None:
            gradient = parameter.grad
            if hasattr(gradient, "to_local"):
                gradient = gradient.to_local()
            post_clip_squared += float(gradient.detach().double().square().sum().item())
    post_clip_norm = post_clip_squared**0.5
    receipt["grad_l2_post_clip"] = post_clip_norm
    receipt["measured_clip_scale"] = (
        post_clip_norm / receipt["grad_l2"] if receipt["grad_l2"] else 1.0
    )
    receipt.update(
        {
            "reported_pre_clip_norm": value,
            "clip_max_norm": float(clip_max_norm),
            "clip_scale": min(1.0, float(clip_max_norm) / (value + 1e-6))
            if clip_max_norm > 0
            else 1.0,
            "trainable_weights_sha256_after": trainable_sha256(model),
        }
    )
    Path(receipt["path"]).with_suffix(".json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        "[rlvp_receipt] "
        + json.dumps({"event": "gradient_and_weight_archive", **receipt}, sort_keys=True),
        flush=True,
    )


def storage_receipt(model):
    totals = {}
    for _, parameter in model.named_parameters():
        key = f"{'trainable' if parameter.requires_grad else 'frozen'}:{parameter.dtype}"
        totals[key] = totals.get(key, 0) + parameter.numel() * parameter.element_size()
    print(
        "[rlvp_receipt] "
        + json.dumps(
            {"event": "parameter_storage_bytes", "by_role_dtype": totals}, sort_keys=True
        ),
        flush=True,
    )
    return totals


def preserve_sft_reference(reference_state, optimizer_path):
    """Restore the original reference on resume instead of resnapshotting policy."""
    raw_path = os.environ.get("NRL_SFT_REFERENCE_PATH")
    if not raw_path:
        raise RuntimeError("NRL_SFT_REFERENCE_PATH is required to preserve the SFT KL reference")
    path = Path(raw_path)
    manifest = path.with_suffix(".json")
    if path.exists():
        receipt = json.loads(manifest.read_text())
        if _sha(path) != receipt["sha256"]:
            raise ValueError("frozen SFT reference hash mismatch")
        original = torch.load(path, map_location="cpu", weights_only=True)
        if set(original) != set(reference_state):
            raise ValueError("frozen SFT reference parameter names changed")
        for key, value in original.items():
            if (
                value.shape != reference_state[key].shape
                or value.dtype != reference_state[key].dtype
            ):
                raise ValueError(f"frozen SFT reference shape/dtype changed: {key}")
        print(
            "[rlvp_receipt] "
            + json.dumps({"event": "sft_reference_restored", **receipt}, sort_keys=True),
            flush=True,
        )
        return original
    if optimizer_path is not None:
        raise RuntimeError(
            "optimizer resume cannot create a new KL reference from resumed weights"
        )
    if not reference_state or any(".lora_" not in key for key in reference_state):
        raise ValueError("Modal reference artifact must contain only frozen SFT LoRA tensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(_cpu(reference_state), temporary)
    os.replace(temporary, path)
    receipt = {
        "path": str(path),
        "sha256": _sha(path),
        "tensor_count": len(reference_state),
        "parameter_count": sum(t.numel() for t in reference_state.values()),
    }
    manifest.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "[rlvp_receipt] "
        + json.dumps({"event": "sft_reference_frozen", **receipt}, sort_keys=True),
        flush=True,
    )
    return reference_state


def full_length_backward_probe(worker):
    """Run the real native worker backward without updating weights or replay.

    Synthetic token repetition proves the maximum tensor shape only. It is not a
    solver trajectory, reward measurement, optimizer proof, or capacity promise
    for more than one learner microbatch.
    """
    if os.environ.get("NRL_FULL_LENGTH_BACKWARD_PROBE") != "1":
        return
    import time

    from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    # Accepted contracts: 9728 context / 9216 generated (proven), 16384 / 15872 (qualification).
    context_tokens = int(worker.cfg["max_total_sequence_length"])
    generated_tokens = {9728: 9216, 16384: 15872}.get(context_tokens)
    if generated_tokens is None:
        raise ValueError("full-length proof requires an accepted context length (9728 or 16384)")
    seed_tokens = worker.tokenizer.encode(
        "import numpy as np\ndef solve(case):\n    values = np.zeros((32, 32))\n    return values\n",
        add_special_tokens=False,
    )
    if not seed_tokens:
        raise ValueError("synthetic shape proof tokenizer produced no tokens")
    token_ids = (seed_tokens * ((context_tokens + len(seed_tokens) - 1) // len(seed_tokens)))[
        :context_tokens
    ]
    token_mask = torch.zeros(1, context_tokens, dtype=torch.float32)
    token_mask[:, context_tokens - generated_tokens :] = 1.0
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "input_lengths": torch.tensor([context_tokens], dtype=torch.long),
            "token_mask": token_mask,
            "sample_mask": torch.ones(1),
            "advantages": torch.ones_like(token_mask),
        }
    )
    original_audit_dir = os.environ["NRL_AUDIT_DIR"]
    os.environ["NRL_AUDIT_DIR"] = str(Path(original_audit_dir) / "full-length-probe")
    before_hash = trainable_sha256(worker.model)
    before_steps = {
        id(p): float(worker.optimizer.state.get(p, {}).get("step", 0))
        for _, p in _trainable(worker.model)
    }
    worker._rlvp_no_optimizer = True
    started = time.monotonic()
    try:
        torch.cuda.reset_peak_memory_stats()
        worker.prepare_for_lp_inference()
        data["prev_logprobs"] = worker.get_logprobs(data)["logprobs"]
        data["generation_logprobs"] = data["prev_logprobs"].clone()
        with worker.use_reference_model():
            data["reference_policy_logprobs"] = worker.get_logprobs(data)["logprobs"]
        archive_batch(data, torch.ones(1), torch.zeros(1, 1, dtype=torch.long), 0)
        worker.prepare_for_training()
        objective = ClippedPGLossFn(
            ClippedPGLossConfig(
                use_cispo=True,
                ratio_clip_min=1.0,
                ratio_clip_max=4.0,
                reference_policy_kl_penalty=0.001,
                use_importance_sampling_correction=True,
            )
        )
        metrics = worker.train(data, objective, gbs=1, mbs=1)
        torch.cuda.synchronize()
        after_hash = trainable_sha256(worker.model)
        after_steps = {
            id(p): float(worker.optimizer.state.get(p, {}).get("step", 0))
            for _, p in _trainable(worker.model)
        }
        if before_hash != after_hash or before_steps != after_steps:
            raise RuntimeError("synthetic backward probe mutated weights or optimizer step")
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        peak_reserved = torch.cuda.max_memory_reserved()
        if peak_reserved >= 0.85 * total_bytes:
            raise RuntimeError("synthetic backward peak exceeds 85% CUDA reservation gate")
        receipt = {
            "event": "full_length_backward_probe",
            "evidence_type": "synthetic_shape_only",
            "context_tokens": context_tokens,
            "loss_tokens": generated_tokens,
            "optimizer_updated": False,
            "replay_inserted": False,
            "trainable_weights_sha256_before": before_hash,
            "trainable_weights_sha256_after": after_hash,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": peak_reserved,
            "cuda_capacity_bytes": total_bytes,
            "elapsed_seconds": time.monotonic() - started,
            "metrics": _cpu(metrics),
        }
        target = Path(os.environ["NRL_AUDIT_DIR"]) / "proof.json"
        # Native metrics may include scalar tensors; preserve them as a tensor
        # archive while the JSON proof contains only JSON-safe summary fields.
        tensor_metrics = receipt.pop("metrics")
        _save("metrics", "backward", tensor_metrics, {"evidence_type": "synthetic_shape_only"})
        target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print("[rlvp_receipt] " + json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        worker.optimizer.zero_grad(set_to_none=True)
        worker._rlvp_no_optimizer = False
        os.environ["NRL_AUDIT_DIR"] = original_audit_dir
        torch.cuda.empty_cache()


def fixed_learner_logprob_probe(worker, optimizer_step):
    """A short fixed-ID, raw-T=1 diagnostic shared with the FP8 refit probe.

    It measures numerical weight-update visibility; it is not a rollout sample,
    behavioral probability at the experiment's T=0.7, or a quality evaluation.
    """
    raw_path = os.environ.get("NRL_FIXED_LOGPROB_PROBE_PATH")
    if not raw_path or getattr(worker, "_rlvp_no_optimizer", False):
        return
    from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    text = "def solve(u0, t_eval):\n    import numpy as np\n    return np.asarray(u0)\n"
    token_ids = worker.tokenizer.encode(text, add_special_tokens=False)
    expected = {"text": text, "token_ids": token_ids, "diagnostic_temperature": 1.0}
    path = Path(raw_path)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("fixed logprob probe tokens/text/temperature changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    if len(token_ids) < 2 or len(token_ids) > 256:
        raise ValueError("fixed learner probe length is outside 2..256 tokens")
    saved_sampling = worker.sampling_params
    saved_training = worker.model.training
    saved_memory_peak = torch.cuda.max_memory_allocated()
    token_mask = torch.ones(1, len(token_ids), dtype=torch.float32)
    token_mask[:, 0] = 0
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "input_lengths": torch.tensor([len(token_ids)], dtype=torch.long),
            "token_mask": token_mask,
            "sample_mask": torch.ones(1),
        }
    )
    # prepare_for_lp_inference wakes the allocator with a random draw. Isolate
    # all diagnostic RNG effects so future training is unchanged by the probe.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        try:
            worker.sampling_params = TrainingSamplingParams(top_k=None, top_p=1.0, temperature=1.0)
            worker.prepare_for_lp_inference()
            logprobs = worker.get_logprobs(data, micro_batch_size=1)["logprobs"].float().cpu()
            if not bool(torch.isfinite(logprobs[:, 1:]).all()):
                raise ValueError("fixed learner probe has nonfinite token probabilities")
            _save(
                "learner-fixed-logprobs",
                f"step-{int(optimizer_step):06d}",
                {"input_ids": torch.tensor(token_ids), "logprobs": logprobs.squeeze(0)},
                {
                    "optimizer_step": int(optimizer_step),
                    "evidence_type": "fixed_token_raw_t1_diagnostic",
                    "temperature": 1.0,
                    "token_count": len(token_ids),
                    "probe_manifest": str(path),
                    "probe_manifest_sha256": _sha(path),
                    "trainable_weights_sha256": trainable_sha256(worker.model),
                    "cuda_peak_before_probe_bytes": saved_memory_peak,
                },
            )
        finally:
            worker.sampling_params = saved_sampling
            worker.prepare_for_training()
            worker.model.train(saved_training)
