"""Optional FP32 mean-normalized full-reward advantage for the existing trainer.

Installed as nemo_rl.utils.rlvp_maxrl_advantage. No policy, reward, mask,
optimizer, or reference-loss mutation occurs here. Unset mode preserves GRPO.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch

NORMALIZER = "mean_plus_epsilon"


def _setting(config, key, env, default=None):
    configured = config.get(key)
    environment = os.environ.get(env)
    if configured is not None and environment is not None:
        if str(configured) != environment:
            # Numeric values may have equivalent decimal/scientific notation.
            try:
                equal = float(configured) == float(environment)
            except (ValueError, TypeError):
                equal = False
            if not equal:
                raise ValueError(f"conflicting {key} and {env}")
    return configured if configured is not None else (
        environment if environment is not None else default
    )


def maxrl_settings(estimator_config, loss_config):
    """Return explicit MaxRL settings, or None for the unchanged legacy branch."""
    mode = _setting(estimator_config, "rlvp_normalizer", "RLVP_ADVANTAGE_NORMALIZER")
    if mode is None or mode == "legacy":
        return None
    if mode != NORMALIZER:
        raise ValueError(f"unknown RLVP advantage normalizer: {mode!r}")
    if estimator_config.get("use_leave_one_out_baseline", False):
        raise ValueError("mean_plus_epsilon requires group-mean, not LOO, baseline")
    if not estimator_config.get("minus_baseline", True):
        raise ValueError("mean_plus_epsilon requires centering")
    kl_in_reward = (loss_config.get("use_kl_in_reward", False)
                    if isinstance(loss_config, dict)
                    else getattr(loss_config, "use_kl_in_reward", False))
    if kl_in_reward:
        raise ValueError("mean_plus_epsilon requires unchanged full reward; keep KL in loss")
    size = _setting(estimator_config, "rlvp_group_size", "RLVP_MAXRL_GROUP_SIZE")
    if size is None or isinstance(size, bool):
        raise ValueError("explicit RLVP_MAXRL_GROUP_SIZE / rlvp_group_size is required")
    try:
        integer_size = int(size)
        exact_size = float(size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("MaxRL group size must be an integer >=2") from exc
    if integer_size < 2 or integer_size != exact_size:
        raise ValueError("MaxRL group size must be an integer >=2")
    epsilon = float(_setting(estimator_config, "rlvp_epsilon", "RLVP_MAXRL_EPSILON", 1e-6))
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("MaxRL epsilon must be finite and positive")
    return {"expected_group_size": integer_size, "epsilon": epsilon}


@torch.no_grad()
def compute_maxrl_advantage(prompt_ids, rewards, mask, *, expected_group_size,
                            epsilon=1e-6, leave_one_out=False):
    """Return detached FP32 masked advantages and a complete scalar-math receipt.

    Every row remains present, including all-zero groups. The caller retains its
    independent token/sample masks and all-row reference KL loss.
    """
    if leave_one_out:
        raise ValueError("mean_plus_epsilon forbids a LOO baseline")
    if isinstance(expected_group_size, bool) or int(expected_group_size) != expected_group_size or expected_group_size < 2:
        raise ValueError("MaxRL requires complete groups of size >=2")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("MaxRL epsilon must be finite and positive")
    if rewards.ndim != 1 or rewards.numel() == 0:
        raise ValueError("full rewards must be a nonempty one-dimensional tensor")
    if not bool(torch.isfinite(rewards).all()) or bool((rewards < 0).any()):
        raise ValueError("full rewards must be finite and nonnegative")
    if prompt_ids.ndim not in (1, 2) or prompt_ids.shape[0] != rewards.numel():
        raise ValueError("prompt/group IDs must identify every reward row")
    if not bool(torch.isfinite(prompt_ids).all()):
        raise ValueError("group IDs must be finite")
    if mask.ndim != 2 or mask.shape[0] != rewards.numel() or mask.shape[1] == 0:
        raise ValueError("token mask must have shape [reward rows, nonempty sequence]")
    if not bool(torch.isfinite(mask).all()) or not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("response token mask must be finite and binary")
    source_dtype = str(rewards.dtype)
    rewards = rewards.detach().float()
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("full rewards overflow FP32")
    epsilon32 = torch.tensor(epsilon, dtype=torch.float32, device=rewards.device)
    if not bool(torch.isfinite(epsilon32)) or float(epsilon32) <= 0:
        raise ValueError("epsilon is not finite and positive in FP32")
    ids, inverse, counts = torch.unique(
        prompt_ids.to(rewards.device), dim=0, return_inverse=True, return_counts=True
    )
    if not bool((counts == expected_group_size).all()):
        raise ValueError(f"incomplete MaxRL groups: expected {expected_group_size}, got {counts.tolist()}")
    means = torch.empty(ids.shape[0], device=rewards.device, dtype=torch.float32)
    minimum = torch.empty_like(means)
    maximum = torch.empty_like(means)
    for group in range(ids.shape[0]):
        values = rewards[inverse == group]
        minimum[group], maximum[group] = values.min(), values.max()
        # An exactly equal group has exactly zero signal, including at tiny scale.
        means[group] = values[0] if bool(minimum[group] == maximum[group]) else values.mean()
    denominator = means + epsilon32
    if not bool(torch.isfinite(denominator).all()):
        raise ValueError("nonfinite FP32 group mean/denominator")
    scalar = (rewards - means[inverse]) / denominator[inverse]
    if not bool(torch.isfinite(scalar).all()):
        raise ValueError("nonfinite MaxRL advantage")
    advantages = scalar.unsqueeze(-1).expand(mask.shape) * mask.to(device=rewards.device, dtype=torch.float32)
    receipt = {
        "event": "maxrl_advantages", "normalizer": NORMALIZER,
        "baseline": "group_mean", "leave_one_out": False,
        "epsilon": epsilon, "effective_epsilon_fp32": float(epsilon32),
        "expected_group_size": int(expected_group_size), "group_count": len(counts),
        "sample_count": rewards.numel(), "source_reward_dtype": source_dtype,
        "reward_dtype": str(rewards.dtype), "advantage_dtype": str(advantages.dtype),
        "group_ids": ids.cpu().tolist(), "group_index": inverse.cpu().tolist(),
        "group_counts": counts.cpu().tolist(), "group_mean_rewards": means.cpu().tolist(),
        "group_denominators": denominator.cpu().tolist(),
        "group_epsilon_attenuation": (means / denominator).cpu().tolist(),
        "group_min_rewards": minimum.cpu().tolist(), "group_max_rewards": maximum.cpu().tolist(),
        "groups_with_reward_contrast": int((minimum != maximum).sum()),
        "rewards_fp32": rewards.cpu().tolist(), "scalar_advantages": scalar.cpu().tolist(),
        "masked_token_count": int(mask.sum()),
        "token_mask_sha256": hashlib.sha256(mask.detach().cpu().to(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    receipt["group_batch_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return advantages, receipt


def record_maxrl_receipt(receipt):
    """Persist one scalar receipt; full per-token tensors use the existing archive."""
    root = os.environ.get("NRL_AUDIT_DIR")
    if not root:
        raise RuntimeError("NRL_AUDIT_DIR is required for mean_plus_epsilon receipts")
    directory = Path(root) / "maxrl-advantages"
    directory.mkdir(parents=True, exist_ok=True)
    # Native group IDs are local to each batch. Identical statistics/masks can
    # legitimately recur on different cases, so content identity is not an
    # occurrence identity. Preserve the content hash and archive every call.
    recorded = {**receipt, "receipt_occurrence_id": uuid.uuid4().hex}
    target = directory / f"groups-{receipt['group_batch_sha256']}-{recorded['receipt_occurrence_id']}.json"
    text = json.dumps(recorded, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x") as handle:
        handle.write(text)
    print("[rlvp_receipt] " + json.dumps({**recorded, "path": str(target)}, sort_keys=True, allow_nan=False), flush=True)
    return target
