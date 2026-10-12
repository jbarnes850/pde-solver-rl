"""CPU-only receipts for the predeclared MaxRL LR and four-update signal gates.

This module neither changes LR nor authorizes a run. Callers preserve the input
tensors and establish that post-update probabilities came from the SAME batch.
Native logprobs and reference use T=.7. KL is the existing clamped sampled k3
statistic, not exact KL; clipped token ESS is not an independent sample count.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping, Sequence

import torch

# Must equal the advantage estimator's epsilon (RLVP_MAXRL_EPSILON); 1e-6 is the original run.
EPSILON = float(os.environ.get("RLVP_MAXRL_EPSILON", "1e-6"))
# First optimizer update is START_STEP + 1 (33 for the native step-32 continuation).
START_STEP = int(os.environ.get("RLVP_MAXRL_START_STEP", "32"))
# Minimum usable groups across the first four updates (the original run required 1).
MIN_USABLE_GROUPS = int(os.environ.get("RLVP_MAXRL_MIN_USABLE_GROUPS", "1"))
THRESHOLDS = {
    "first_update_learning_rate": 1e-6,
    "target_learning_rate": 2e-6,
    "minimum_measured_clip_scale": 0.5,
    "maximum_relative_update_l2": 8e-5,
    "maximum_post_ratio_above5_fraction": 0.005,
    "minimum_post_clipped_token_ess_fraction": 0.9,
    "minimum_four_update_epsilon_attenuation": 0.1,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor(value, name):
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    _require(value.device.type == "cpu", f"{name} must be a CPU snapshot")
    return value.detach()


def _finite(value, name):
    _require(bool(torch.isfinite(value).all()), f"nonfinite {name}")


def _scalar(value, name):
    _require(not isinstance(value, bool), f"{name} must be numeric")
    result = float(value)
    _require(math.isfinite(result), f"nonfinite {name}")
    return result


def _core(batch, group_size, group_count):
    _require(isinstance(batch, Mapping), "batch must be a mapping")
    ids = _tensor(batch["input_ids"], "input_ids")
    _require(ids.ndim == 2 and ids.shape[1] >= 2, "input_ids must have shape [B,L>=2]")
    _require(not ids.is_floating_point() and not ids.is_complex() and ids.dtype != torch.bool,
             "input_ids must be integers")
    _require(bool((ids >= 0).all()), "input_ids must be nonnegative")
    tokens = _tensor(batch["token_mask"], "token_mask")
    samples = _tensor(batch["sample_mask"], "sample_mask")
    _require(tokens.shape == ids.shape, "token_mask must exactly match input_ids")
    _require(samples.shape == (ids.shape[0],), "sample_mask must have shape [B]")
    for value, name in ((tokens, "token_mask"), (samples, "sample_mask")):
        _finite(value, name)
        _require(bool(((value == 0) | (value == 1)).all()), f"{name} must be binary")
    _require(not bool(tokens[:, 0].any()), "token_mask[:,0] must be zero before native shift")
    _require(bool((samples == 1).all()), "all rows/groups must be retained")
    active = tokens[:, 1:].bool() & samples.bool().unsqueeze(-1)
    _require(bool(active.any(dim=1).all()), "every retained row needs an action token")
    rewards = _tensor(batch["rewards"], "rewards")
    _require(rewards.shape == samples.shape and rewards.dtype == torch.float32,
             "rewards must be FP32 [B]")
    _finite(rewards, "rewards")
    _require(bool(((rewards >= 0) & (rewards <= 1)).all()), "full rewards must lie in [0,1]")
    identities = _tensor(batch["trajectory_group_ids"], "trajectory_group_ids")
    _require(identities.ndim in (1, 2) and identities.shape[0] == ids.shape[0]
             and identities.numel() > 0, "group IDs must have shape [B] or [B,K>0]")
    _require(not identities.is_floating_point() and not identities.is_complex()
             and identities.dtype != torch.bool, "group IDs must be integer identities")
    _, inverse, counts = torch.unique(identities, dim=0, return_inverse=True, return_counts=True)
    _require(counts.numel() == group_count and bool((counts == group_size).all()),
             f"expected {group_count} complete groups of {group_size} rows")
    return ids, rewards, active, inverse


def _support(batch, group_size, group_count):
    ids, rewards, active, inverse = _core(batch, group_size, group_count)
    advantages = torch.zeros_like(rewards)
    groups = []
    for group in range(group_count):
        take = inverse == group
        values = rewards[take]
        mean = values.mean()  # FP32, matching the native group estimator.
        advantages[take] = (values - mean) / (mean + EPSILON)
        nonconstant = bool((values != values[0]).any())
        attenuation = float(mean / (mean + EPSILON))
        groups.append({
            "group": group, "rows": int(take.sum()), "mean_full_reward": float(mean),
            "maximum_full_reward": float(values.max()), "nonconstant": nonconstant,
            "epsilon_attenuation": attenuation,
            "usable": nonconstant and attenuation >= 0.1,
            "nonzero_advantage_rows": int((advantages[take] != 0).sum()),
        })
    _finite(advantages, "FP32 mean+epsilon advantages")
    report = {
        "schema": "rlvp.maxrl-full-reward-support.v1", "input_contract_passed": True,
        "rows": ids.shape[0], "groups": group_count, "action_tokens": int(active.sum()),
        "positive_reward_rows": int((rewards > 0).sum()),
        "nonconstant_groups": sum(g["nonconstant"] for g in groups),
        "nonzero_advantage_rows": int((advantages != 0).sum()),
        "usable_groups": sum(g["usable"] for g in groups), "per_group": groups,
        "advantage_formula": f"FP32 (R-group_mean)/(group_mean+{EPSILON:g})",
    }
    return report, advantages, active, ids.shape


def full_reward_support(batch, expected_group_size=16, expected_groups=8):
    """Summarize one complete batch; malformed inputs raise ValueError/KeyError."""
    return _support(batch, expected_group_size, expected_groups)[0]


def _logprobs(value, name, shape, active, layout="full"):
    value = _tensor(value, name)
    _require(value.is_floating_point(), f"{name} must be floating point")
    if layout == "full":
        _require(value.shape == shape, f"{name} must have full shape {tuple(shape)}")
        value = value[:, 1:]
    elif layout == "next_token":
        _require(value.shape == active.shape, f"{name} must have shifted shape {tuple(active.shape)}")
    else:
        raise ValueError("post_logprob_layout must be 'full' or 'next_token'")
    # Index before subtraction/exp: nonfinite padding must not poison active math.
    value = value[active].float()
    _finite(value, f"active FP32 {name}")
    return value


def _ratio(current, behavior):
    difference = current - behavior
    _finite(difference, "FP32 behavior log-ratio")
    ratio = difference.exp()
    _finite(ratio, "FP32 behavior ratio before clipping")
    return ratio


def _kl(current, reference):
    difference = reference - current
    _finite(difference, "FP32 reference log-ratio")
    difference = difference.clamp(-20, 20)
    # Exact convention from pinned NeMo calculate_kl(k3), including both clamps.
    value = (difference.exp() - 1 - difference).clamp(-10, 10)
    _finite(value, "sampled clipped reference k3")
    return float(value.mean())


def evaluate_lr_gate(batch, post_update_logprobs, gradient_receipt, integrity_checks, *,
                     post_logprob_layout, post_refit_passed=False,
                     expected_group_size=16, expected_groups=8):
    """Check update33 at LR1e-6; promotion to2e-6 also requires actual refit.

    Required integrity_checks includes same_batch_post_update=True; the caller
    proves that provenance using native recomputation and preserved tensors.
    Every supplied native integrity check must be a real bool and true. Prior to
    actual successful refit, numerical success returns pending_refit, never pass.
    """
    names = ("input_contract", "all_finite", "native_integrity", "same_batch_post_update",
             "first_update_lr_and_step", "advantages_match", "useful_actor_signal",
             "measured_clip_scale", "relative_update", "post_ratio_cap_fraction",
             "post_clipped_token_ess", "post_reference_k3", "post_refit_integrity")
    report = {
        "schema": "rlvp.maxrl-lr-gate.v1", "status": "fail", "passed": False,
        "decision": "stop_and_diagnose", "promotion_ready": False,
        "numerical_gate_passed": False, "next_learning_rate": None,
        "proposed_learning_rate": THRESHOLDS["target_learning_rate"],
        "thresholds": dict(THRESHOLDS), "checks": dict.fromkeys(names, False),
        "metrics": {}, "reasons": [],
        "scope": "Same-batch CPU numerical/receipt check; no LR mutation or learning claim. "
                 "Nonfinite inactive logprob padding is ignored. Actor mass and token ESS "
                 "are correlated token diagnostics, not parameter gradients or independent samples.",
    }
    checks, metrics = report["checks"], report["metrics"]
    try:
        support, advantage, active, shape = _support(batch, expected_group_size, expected_groups)
        report["full_reward_support"] = support
        pre = _logprobs(batch["prev_logprobs"], "prev_logprobs", shape, active)
        behavior = _logprobs(batch["generation_logprobs"], "generation_logprobs", shape, active)
        reference = _logprobs(batch["reference_policy_logprobs"], "reference_policy_logprobs", shape, active)
        post = _logprobs(post_update_logprobs, "post_update_logprobs", shape, active, post_logprob_layout)
        stored = _tensor(batch["advantages"], "advantages")
        _require(stored.dtype == torch.float32 and stored.shape == shape,
                 "stored advantages must be FP32 full shape")
        stored = stored[:, 1:][active]
        _finite(stored, "active stored advantages")
        expected = advantage.unsqueeze(-1).expand(active.shape)[active]
        checks["advantages_match"] = bool(torch.allclose(stored, expected, rtol=1e-5, atol=1e-7))
        metrics["advantage_max_abs_error"] = float((stored - expected).abs().max())
        pre_ratio, post_ratio = _ratio(pre, behavior), _ratio(post, behavior)
        clipped = post_ratio.clamp(max=5).double()
        squared_sum = float(clipped.square().sum())
        ess = float(clipped.sum()) ** 2 / (clipped.numel() * squared_sum) if squared_sum else 0.
        fraction = int((post_ratio > 5).sum()) / post_ratio.numel()
        pre_kl, post_kl = _kl(pre, reference), _kl(post, reference)
        kl_limit = max(.03, 2 * pre_kl + .005)
        grad = _scalar(gradient_receipt["grad_l2"], "grad_l2")
        update = _scalar(gradient_receipt["trainable_update_l2"], "trainable_update_l2")
        relative = _scalar(gradient_receipt["trainable_relative_update_l2"], "relative update")
        clip = _scalar(gradient_receipt["measured_clip_scale"], "measured clip scale")
        _require(grad >= 0 and update >= 0 and relative >= 0 and 0 <= clip <= 1.00001,
                 "invalid gradient/update/clip receipt range")
        rates = gradient_receipt["learning_rates"]
        _require(isinstance(rates, (list, tuple)) and bool(rates), "missing actual learning rates")
        rates = [_scalar(r, "learning_rate") for r in rates]
        before = gradient_receipt["trainable_weights_sha256_before"]
        after = gradient_receipt["trainable_weights_sha256_after"]
        _require(all(isinstance(s, str) and re.fullmatch(r"[0-9a-f]{64}", s)
                     for s in (before, after)), "missing weight hashes")
        _require(isinstance(integrity_checks, Mapping) and bool(integrity_checks)
                 and all(type(v) is bool for v in integrity_checks.values()),
                 "native integrity checks must be a nonempty boolean mapping")
        _require(type(post_refit_passed) is bool, "post_refit_passed must be bool")
        actor = float((expected.abs() * pre_ratio.clamp(max=5)).mean())
        metrics.update({
            "rows": shape[0], "action_tokens": int(active.sum()), "groups": support["groups"],
            "pre_actor_coefficient_l1": actor, "grad_l2": grad, "trainable_update_l2": update,
            "trainable_relative_update_l2": relative, "measured_clip_scale": clip,
            "post_ratio_max": float(post_ratio.max()), "post_ratio_above5_tokens": int((post_ratio > 5).sum()),
            "post_ratio_above5_fraction": fraction, "post_clipped_token_ess_fraction": ess,
            "pre_reference_k3_mean": pre_kl, "post_reference_k3_mean": post_kl,
            "post_reference_k3_limit": kl_limit,
        })
        _require(all(math.isfinite(v) for v in metrics.values()), "nonfinite gate summary")
        checks.update({
            "input_contract": True, "all_finite": True,
            "native_integrity": all(integrity_checks.values()),
            "same_batch_post_update": integrity_checks.get("same_batch_post_update") is True,
            "first_update_lr_and_step": gradient_receipt.get("step") == START_STEP + 1
                and all(math.isclose(r, 1e-6, rel_tol=1e-9, abs_tol=0) for r in rates),
            "useful_actor_signal": support["nonconstant_groups"] > 0 and actor > 0
                and grad > 0 and update > 0 and relative > 0 and before != after
                and not gradient_receipt.get("missing_gradient_names", []),
            "measured_clip_scale": clip >= .5, "relative_update": relative <= 8e-5,
            "post_ratio_cap_fraction": fraction <= .005,
            "post_clipped_token_ess": ess >= .9,
            "post_reference_k3": post_kl <= kl_limit,
            "post_refit_integrity": post_refit_passed,
        })
        report["native_integrity_checks"] = dict(integrity_checks)
        report["numerical_gate_passed"] = all(v for k, v in checks.items() if k != "post_refit_integrity")
        if report["numerical_gate_passed"]:
            report.update(status="pass" if post_refit_passed else "pending_refit",
                          decision="promote_to_2e-6" if post_refit_passed else "await_successful_refit",
                          passed=post_refit_passed, promotion_ready=post_refit_passed,
                          next_learning_rate=2e-6 if post_refit_passed else None)
    except (ValueError, KeyError, TypeError, RuntimeError, OverflowError) as error:
        report["reasons"].append(f"{type(error).__name__}: {error}")
    report["reasons"].extend(k for k, v in checks.items() if not v)
    return report


def evaluate_support_gate(receipts: Sequence[Mapping]):
    """Check contiguous actual steps33..36; absence of signal before4 is pending."""
    result = {"schema": "rlvp.maxrl-four-update-signal.v1", "status": "fail",
              "decision": "stop_and_diagnose", "updates": len(receipts),
              "usable_groups": 0, "reasons": [], "threshold": .1}
    try:
        steps = [r["optimizer_step"] for r in receipts]
        first = START_STEP + 1
        _require(steps == list(range(first, first + len(receipts))), "need contiguous unique steps from the first update")
        for receipt in receipts[:4]:
            support = receipt["full_reward_support"]
            _require(support["schema"] == "rlvp.maxrl-full-reward-support.v1"
                     and support["input_contract_passed"] is True, "invalid full-reward support")
            groups = support["per_group"]
            _require(len(groups) == support["groups"], "incomplete group support")
            for group in groups:
                attenuation = _scalar(group["epsilon_attenuation"], "epsilon attenuation")
                _require(0 <= attenuation <= 1 and type(group["nonconstant"]) is bool,
                         "invalid support statistics")
                result["usable_groups"] += int(group["nonconstant"] and attenuation >= .1)
        if len(receipts) < 4:
            result.update(status="continue", decision="await_four_updates")
        elif result["usable_groups"] >= MIN_USABLE_GROUPS:
            result.update(status="pass", decision="continue_within_existing_cap")
        else:
            result["reasons"].append(
                f"fewer than {MIN_USABLE_GROUPS} nonconstant groups with mean/(mean+eps)>=.1 in first4 updates")
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        result["reasons"].append(f"{type(error).__name__}: {error}")
    return result
