"""Opt-in qualification hooks for the native MaxRL continuation (start step 32, or 0 from base).

No model loading, optimizer replacement, rollout, or training loop lives here.
Installed into nemo_rl.utils by patch_maxrl_training.py.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import torch


def settings():
    if os.environ.get("RLVP_MAXRL_TRAINING") != "1":
        return None
    start = int(os.environ["RLVP_MAXRL_START_STEP"])
    initial = float(os.environ["RLVP_MAXRL_INITIAL_LR"])
    target = float(os.environ["RLVP_MAXRL_TARGET_LR"])
    warmup = int(os.environ.get("RLVP_MAXRL_WARMUP_UPDATES", "0"))
    if start not in (0, 32, 64, 96, 128, 160, 192, 224, 256) or initial != 1e-6 or target not in (2e-6, 5e-6, 1e-5) or not 0 <= warmup <= 8:
        raise ValueError("MaxRL qualification: start 0 or a multiple of 32 up to 256, LR1e-6 then target 2e-6/5e-6/1e-5, warmup<=8")
    root = Path(os.environ["NRL_AUDIT_DIR"])
    if not root.is_absolute():
        raise ValueError("NRL_AUDIT_DIR must be absolute")
    return {"start_step": start, "initial_lr": initial, "target_lr": target, "warmup_updates": warmup,
            "root": root, "directory": root / "maxrl-training"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)
    print("[rlvp_receipt] " + json.dumps({"event": "maxrl_training", "path": str(path),
          "sha256": _sha(path), "status": value.get("status")}, sort_keys=True), flush=True)


def _optimizer_step(optimizer, allow_fresh=False):
    if allow_fresh and not optimizer.state:
        return 0  # a fresh AdamW creates its per-parameter state lazily on the first step
    steps = []
    for state in optimizer.state.values():
        if "step" not in state:
            raise ValueError("restored Adam state is missing its step counter")
        value = state["step"]
        if hasattr(value, "to_local"):
            value = value.to_local()
        value = float(value.item() if isinstance(value, torch.Tensor) else value)
        if not math.isfinite(value) or value != int(value):
            raise ValueError("invalid Adam step counter")
        steps.append(int(value))
    if not steps or len(set(steps)) != 1:
        raise ValueError("Adam counters must exist and agree")
    return steps[0]


def scheduled_lr(config, update):
    """Update 1 qualifies at the initial LR; later updates warm up linearly to the target."""
    if update <= 1:
        return config["initial_lr"]
    warmup = config["warmup_updates"]
    if warmup <= 0 or update - 1 >= warmup:
        return config["target_lr"]
    return config["initial_lr"] + (config["target_lr"] - config["initial_lr"]) * (update - 1) / warmup


def _constant_scheduler_lr(optimizer, scheduler, value):
    """Retarget the restored constant scheduler without resetting optimizer state."""
    if scheduler.optimizer is not optimizer:
        raise ValueError("scheduler does not own the actual optimizer")
    if not isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
        raise ValueError("qualification requires the native constant LambdaLR")
    for function in scheduler.lr_lambdas:
        if any(float(function(i)) != 1.0 for i in (0, scheduler.last_epoch, scheduler.last_epoch + 1)):
            raise ValueError("qualification rejects a nonconstant LR scheduler")
    for group in optimizer.param_groups:
        group["lr"] = value
        group["initial_lr"] = value
    scheduler.base_lrs = [value] * len(optimizer.param_groups)
    scheduler._last_lr = [value] * len(optimizer.param_groups)


def prepare_worker_train(worker, data, gbs, mbs, eval_mode=False):
    config = settings()
    if config is None or eval_mode:
        return
    if getattr(worker, "_rlvp_no_optimizer", False):
        # The synthetic full-length backward probe takes no optimizer step and verifies the
        # trainable weights are unchanged; MaxRL LR and Adam checks apply to real updates only.
        return
    if (gbs, mbs, data.size, worker.dp_size, worker.cp_size) != (128, 1, 128, 1, 1):
        raise ValueError("MaxRL qualification requires one native 128-row optimizer batch")
    if not isinstance(worker.optimizer, torch.optim.AdamW):
        raise ValueError("qualification requires the restored AdamW optimizer")
    if float(worker.sampling_params.temperature) != 0.7:
        raise ValueError("learner log probabilities must use the sampling temperature 0.7")
    previous = getattr(worker, "_rlvp_maxrl_last_pre_step", None)
    step = _optimizer_step(worker.optimizer, allow_fresh=config["start_step"] == 0 and previous is None)
    expected = config["start_step"] if previous is None else previous + 1
    if step != expected:
        raise ValueError(f"MaxRL Adam continuity mismatch: expected {expected}, got {step}")
    decision_path = config["directory"] / "lr-decision.json"
    decision_sha = None
    if step == config["start_step"]:
        if decision_path.exists():
            raise ValueError("fresh qualification must not inherit an LR decision")
        lr = config["initial_lr"]
    else:
        decision = json.loads(decision_path.read_text())
        if not (decision.get("schema") == "rlvp.maxrl.lr-decision.v1"
                and decision.get("optimizer_step") == config["start_step"] + 1
                and decision.get("initial_lr") == config["initial_lr"]
                and decision.get("target_lr") == config["target_lr"]
                and decision.get("status") == "pass"
                and decision.get("promotion_ready") is True):
            raise RuntimeError("MaxRL LR promotion requires a successful actual update and refit")
        decision_sha = _sha(decision_path)
        lr = scheduled_lr(config, step - config["start_step"] + 1)
    before = [float(group["lr"]) for group in worker.optimizer.param_groups]
    _constant_scheduler_lr(worker.optimizer, worker.scheduler, lr)
    worker._rlvp_maxrl_last_pre_step = step
    path = config["directory"] / f"lr-before-update-{step + 1:06d}.json"
    if path.exists():
        raise FileExistsError(path)
    _write(path, {"schema": "rlvp.maxrl.optimizer-lr.v1", "status": "set",
                 "optimizer_step_before": step, "optimizer_step_after_expected": step + 1,
                 "learning_rates_before": before,
                 "learning_rates": [float(g["lr"]) for g in worker.optimizer.param_groups],
                 "scheduler_base_lrs": list(worker.scheduler.base_lrs),
                 "scheduler_last_lrs": list(worker.scheduler.get_last_lr()),
                 "scheduler_last_epoch": worker.scheduler.last_epoch,
                 "promotion_decision_sha256": decision_sha})


def _load_batch(config, step, live):
    path = config["root"] / "batches" / f"step-{step:06d}.pt"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["step"] != step or receipt["sha256"] != _sha(path):
        raise ValueError("saved batch receipt does not authenticate this native step")
    batch = torch.load(path, map_location="cpu", weights_only=True)
    for name in ("input_ids", "input_lengths", "token_mask", "sample_mask", "advantages",
                 "prev_logprobs", "generation_logprobs", "reference_policy_logprobs"):
        if name in batch:
            if name not in live or not torch.equal(batch[name], live[name].detach().cpu()):
                raise ValueError(f"native training changed the saved same-batch field {name}")
    return batch, receipt


def _gradient_receipt(config, optimizer_step):
    path = config["root"] / "gradients" / f"step-{optimizer_step:06d}-rank-0.json"
    receipt = json.loads(path.read_text())
    tensor_path = path.with_suffix(".pt")
    if receipt["step"] != optimizer_step or receipt["sha256"] != _sha(tensor_path):
        raise ValueError("gradient receipt does not authenticate this optimizer update")
    return receipt, {"path": str(path), "sha256": _sha(path),
                     "tensor_sha256": receipt["sha256"]}


def after_train_step(policy, train_data, timer, step, master_config):
    """Archive real step33 post-update likelihoods; defer failures until checkpoint."""
    config = settings()
    if config is None:
        return None
    from nemo_rl.utils.maxrl_lr_gate import evaluate_lr_gate, evaluate_support_gate, full_reward_support
    from nemo_rl.utils.rlvp_modal_audit import _save

    state = {"config": config, "optimizer_step": int(step) + 1, "failure": None}
    try:
        if not (config["start_step"] <= int(step)):
            raise ValueError("native MaxRL driver must continue from its declared start step")
        if float(master_config.policy["generation"]["temperature"]) != 0.7:
            raise ValueError("post-update logprob temperature must remain 0.7")
        if int(master_config.checkpointing["save_period"]) != 1 or not master_config.checkpointing["enabled"]:
            raise ValueError("qualification requires a native checkpoint after every update")
        batch, batch_receipt = _load_batch(config, int(step), train_data)
        if int(step) < config["start_step"] + 4:
            support_path = config["directory"] / f"support-{int(step) + 1:06d}.json"
            if support_path.exists():
                raise FileExistsError(support_path)
            support = {"optimizer_step": int(step) + 1,
                       "full_reward_support": full_reward_support(batch, expected_group_size=16, expected_groups=8),
                       "batch_sha256": batch_receipt["sha256"]}
            _write(support_path, support)
            receipts = [json.loads((config["directory"] / f"support-{i:06d}.json").read_text())
                        for i in range(config["start_step"] + 1, int(step) + 2)]
            support_decision = evaluate_support_gate(receipts)
            _write(config["directory"] / "support-decision.json", support_decision)
            if support_decision["status"] == "fail":
                state["failure"] = "first-four-update full-reward support gate failed"
        if int(step) != config["start_step"]:
            return state
        policy.prepare_for_lp_inference()
        post = policy.get_logprobs(train_data, timer=timer)["logprobs"].detach().float().cpu()
        # Prove inference left the same archived training values intact.
        _load_batch(config, int(step), train_data)
        post_receipt = _save("maxrl-post-update-logprobs", f"step-{int(step) + 1:06d}",
                             {"input_ids": batch["input_ids"], "token_mask": batch["token_mask"],
                              "sample_mask": batch["sample_mask"], "post_update_logprobs": post},
                             {"optimizer_step": int(step) + 1, "preupdate_batch_step": int(step),
                              "temperature": 0.7, "post_logprob_layout": "full",
                              "batch_sha256": batch_receipt["sha256"]})
        gradient, gradient_identity = _gradient_receipt(config, int(step) + 1)
        integrity = {"same_batch_post_update": True, "saved_batch_hash_verified": True,
                     "gradient_hash_verified": True, "temperature_0_7": True,
                     "post_logprob_tokens_unchanged": True}
        report = evaluate_lr_gate(batch, post, gradient, integrity, post_logprob_layout="full",
                                  post_refit_passed=False, expected_group_size=16, expected_groups=8)
        state.update(batch=batch, post=post, gradient=gradient, integrity=integrity,
                     report=report, batch_receipt=batch_receipt,
                     post_receipt=post_receipt, gradient_identity=gradient_identity)
        _write(config["directory"] / "numerical-gate.json", report)
        if report["status"] == "fail":
            state["failure"] = "first real MaxRL update failed numerical qualification"
    except Exception as error:
        state["failure"] = f"{type(error).__name__}: {error}"
        _write(config["directory"] / f"failure-{int(step) + 1:06d}.json",
               {"status": "fail", "optimizer_step": int(step) + 1, "reason": state["failure"]})
    return state


def after_refit_step(state, post_refit_passed):
    if state is None or state["optimizer_step"] != state["config"]["start_step"] + 1:
        return
    from nemo_rl.utils.maxrl_lr_gate import evaluate_lr_gate

    config = state["config"]
    decision = {"schema": "rlvp.maxrl.lr-decision.v1", "optimizer_step": config["start_step"] + 1,
                "initial_lr": config["initial_lr"], "target_lr": config["target_lr"],
                "status": "fail", "promotion_ready": False,
                "post_refit_passed": bool(post_refit_passed)}
    try:
        if not post_refit_passed:
            raise RuntimeError("native refit/post-refit generation proof failed")
        if state["failure"]:
            raise RuntimeError(state["failure"])
        report = evaluate_lr_gate(state["batch"], state["post"], state["gradient"],
                                  state["integrity"], post_logprob_layout="full",
                                  post_refit_passed=True, expected_group_size=16, expected_groups=8)
        decision.update(status=report["status"], promotion_ready=report["promotion_ready"],
                        gate=report, batch_sha256=state["batch_receipt"]["sha256"],
                        post_logprobs_sha256=state["post_receipt"]["sha256"],
                        gradient_receipt=state["gradient_identity"])
        if not (report["status"] == "pass" and report["promotion_ready"] is True):
            raise RuntimeError("MaxRL promotion gate failed after refit")
    except Exception as error:
        state["failure"] = f"{type(error).__name__}: {error}"
        decision.update(status="fail", promotion_ready=False, reason=state["failure"])
    _write(config["directory"] / "lr-decision.json", decision)
    # Release this same-batch diagnostic after its immutable receipts exist.
    for name in ("batch", "post", "gradient"):
        state.pop(name, None)


def raise_pending_failure(state):
    if state is not None and state["failure"]:
        raise RuntimeError("MaxRL qualification failed; native checkpoint preserved: " + state["failure"])
