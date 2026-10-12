#!/usr/bin/env python3
"""Apply bounded audit/loss fixes after the historical NeMo patches are copied.

Usage: python apply_audit_patches.py --nemo-root /opt/nemo-rl
No source acquisition or launch occurs here. Anchor mismatches fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def replace(source, before, after, count=1):
    if source.count(before) != count:
        raise RuntimeError(
            f"expected {count} occurrences, found {source.count(before)}: {before[:100]!r}"
        )
    return source.replace(before, after)


def patch_loss(source):
    source, tail = source.split("\nclass NLLLossFn", 1)
    source = replace(
        source,
        "    use_cispo: bool = False\n",
        "    use_cispo: bool = False\n    # RLVP: Dr. GRPO uses a fixed response-token denominator.\n    rlvp_fixed_response_tokens: int = 0\n",
    )
    source = replace(
        source,
        "        self.use_cispo = cfg.use_cispo\n",
        '        self.use_cispo = cfg.use_cispo\n        self.rlvp_fixed_response_tokens = cfg.rlvp_fixed_response_tokens\n        if self.rlvp_fixed_response_tokens < 0:\n            raise ValueError("fixed response-token denominator must be nonnegative")\n',
    )
    source = replace(
        source,
        "        curr_logprobs = next_token_logprobs\n",
        "        curr_logprobs = next_token_logprobs.float()\n",
    )
    source = replace(
        source,
        '        advantages = data["advantages"][:, 1:]\n',
        '        advantages = data["advantages"][:, 1:].float()\n',
    )
    source = replace(
        source,
        "        mask = token_mask * sample_mask.unsqueeze(-1)\n",
        "        mask = token_mask * sample_mask.unsqueeze(-1)\n        # RLVP: fixed-max response normalization changes the objective only,\n        # leaving diagnostic means normalized by actual valid tokens.\n        objective_valid_toks = (global_valid_seqs * self.rlvp_fixed_response_tokens\n                                if self.rlvp_fixed_response_tokens else global_valid_toks)\n",
        count=1,
    )
    source = replace(
        source,
        "                    kl, mask, global_normalization_factor=global_valid_toks\n",
        "                    kl, mask, global_normalization_factor=objective_valid_toks\n",
    )
    source = replace(
        source,
        "            log_ratios = curr_logprobs - prev_logprobs\n",
        "            # RLVP CISPO clips the actual current/behavior ratio once.\n            # Clipping current/prev then multiplying prev/behavior is unbounded.\n            ratio_denominator = generation_logprobs if self.use_cispo else prev_logprobs\n            log_ratios = curr_logprobs - ratio_denominator\n",
    )
    source = replace(
        source,
        "        if self.use_importance_sampling_correction:\n            importance_weights_to_use = actor_importance_weights\n",
        "        if self.use_importance_sampling_correction and not self.use_cispo:\n            importance_weights_to_use = actor_importance_weights\n",
    )
    source = replace(
        source,
        "                importance_weights_to_use * clip_loss,\n                mask,\n                global_normalization_factor=global_valid_toks,\n",
        "                importance_weights_to_use * clip_loss,\n                mask,\n                global_normalization_factor=objective_valid_toks,\n",
    )
    source = replace(
        source,
        '                "loss": loss.item(),\n',
        '                "loss": loss.item(),\n                "actor_loss": actor_loss.item(),\n                "kl_loss": kl.item(),\n',
    )
    return source + "\nclass NLLLossFn" + tail


def patch_model_utils(source):
    # Restrict to the custom chunked autograd class, preserving every other path.
    start = source.index("class ChunkedDistributedLogprob(")
    end = source.index("\nclass ", start + 6)
    part = source[start:end]
    part = replace(
        part,
        "            vocab_parallel_logits, dtype=torch.float32\n",
        "            vocab_parallel_logits\n",
    )
    before = """            # 1 if it's the chosen log prob, 0 otherwise
            is_chosen = (~(target_mask[:, chunk_start:chunk_end])).unsqueeze(
                -1
            ) * torch.nn.functional.one_hot(
                masked_target[:, chunk_start:chunk_end],
                num_classes=partition_vocab_size,
            )

            # Inplace index into the preallocated grad_input tensor
            grad_input_chunk = grad_input[:, chunk_start:chunk_end, :]

            grad_input_chunk.copy_(
                is_chosen.float().sub_(softmax_output)
            )  # inplace copy
            grad_input_chunk.mul_(
                grad_output[:, chunk_start:chunk_end].unsqueeze(dim=-1)
            )"""
    after = """            # RLVP: calculate one chunk in FP32, then cast only the completed
            # derivative to the input dtype, as autograd would do at this edge.
            # Avoid a full-sequence FP32 gradient and an int64 one-hot temporary.
            softmax_output.neg_()
            softmax_output.scatter_add_(
                -1,
                masked_target[:, chunk_start:chunk_end].unsqueeze(-1),
                (~target_mask[:, chunk_start:chunk_end]).float().unsqueeze(-1),
            )
            softmax_output.mul_(grad_output[:, chunk_start:chunk_end].unsqueeze(-1))
            grad_input[:, chunk_start:chunk_end, :].copy_(softmax_output)"""
    part = replace(part, before, after)
    part = replace(
        part,
        "            del softmax_output, is_chosen, logits\n",
        "            del softmax_output, logits\n",
    )
    return source[:start] + part + source[end:]


def patch_advantage(source):
    source = replace(
        source,
        "import torch\n",
        "import torch\nfrom nemo_rl.utils.rlvp_modal_audit import group_statistics_fp32\n",
    )
    before = """        baseline, std = calculate_baseline_and_std_per_prompt(
            prompt_ids,
            rewards,
            torch.ones_like(rewards),
            leave_one_out_baseline=self.use_leave_one_out_baseline,
        )"""
    after = """        # Reward/statistic/advantage math stays FP32 regardless of model dtype.
        rewards = rewards.float()
        baseline, std = group_statistics_fp32(
            prompt_ids, rewards, leave_one_out=self.use_leave_one_out_baseline,
        )"""
    return replace(source, before, after)


def patch_grpo(source):
    source = replace(
        source,
        '_RLVP_ACTIVE_ONLY = os.environ.get("NRL_ACTIVE_ONLY") == "1"\n',
        '_RLVP_ACTIVE_ONLY = os.environ.get("NRL_ACTIVE_ONLY") == "1"\nif _RLVP_ACTIVE_ONLY:\n    raise RuntimeError("Modal RLVP forbids active-only masking: all-row KL and logprobs are required")\nfrom nemo_rl.utils.rlvp_modal_audit import archive_batch\n',
    )
    # Both sync and async can produce evaluation-like reward tensors, force dtype.
    source = replace(
        source,
        '                    rewards = repeated_batch["total_reward"]\n',
        '                    rewards = repeated_batch["total_reward"].float()\n',
        count=2,
    )
    # Full tensors are archived before any minibatch reshuffling and optimizer step.
    source = replace(
        source,
        "                    del prompt_ids_for_adv\n",
        "                    rlvp_receipt_group_ids = prompt_ids_for_adv\n                    del prompt_ids_for_adv\n",
        count=2,
    )
    before = """                    train_data["advantages"] = _clip_grpo_advantages(
                        train_data["advantages"], master_config.grpo
                    )"""
    source = replace(
        source,
        before,
        before
        + "\n                    archive_batch(train_data, rewards, rlvp_receipt_group_ids, step)",
        count=2,
    )
    return source


def patch_worker(source):
    source = replace(
        source,
        '_ACTIVE_ONLY = os.getenv("NRL_ACTIVE_ONLY") == "1"\n',
        '_ACTIVE_ONLY = os.getenv("NRL_ACTIVE_ONLY") == "1"\nfrom nemo_rl.utils.rlvp_modal_audit import (archive_gradients, finish_gradient_receipt, storage_receipt, preserve_sft_reference, full_length_backward_probe, fixed_learner_logprob_probe)\nif _ACTIVE_ONLY:\n    raise RuntimeError("Modal RLVP requires all-row training and log probabilities")\n',
    )
    source = replace(
        source,
        "        # Initialize reference model if requested\n",
        "        storage_receipt(self.model)\n\n        # Initialize reference model if requested\n",
    )
    source = replace(
        source,
        "                if not eval_mode:\n                    grad_norm = scale_grads_and_clip_grad_norm(\n",
        "                if not eval_mode:\n                    gradient_archive = archive_gradients(\n                        self.model, self.optimizer, _optimizer_step_value(self.optimizer) + 1,\n                        dp_size=self.dp_size, cp_size=self.cp_size,\n                    )\n                    grad_norm = scale_grads_and_clip_grad_norm(\n",
    )
    source = replace(
        source,
        "                    self.optimizer.step()\n",
        "                    self.optimizer.step()\n                    finish_gradient_receipt(gradient_archive, self.model, self.max_grad_norm, grad_norm)\n",
    )
    source = replace(
        source,
        "        # Set instance attributes from runtime config (tuple unpacking)\n",
        "        if self.reference_model_state_dict is not None:\n            self.reference_model_state_dict = preserve_sft_reference(self.reference_model_state_dict, optimizer_path)\n\n        # Set instance attributes from runtime config (tuple unpacking)\n",
    )
    source = replace(
        source,
        "    def _update_moe_gate_bias_if_supported(self) -> None:\n",
        "        full_length_backward_probe(self)\n        fixed_learner_logprob_probe(self, _optimizer_step_value(self.optimizer))\n\n    def _update_moe_gate_bias_if_supported(self) -> None:\n",
    )
    source = replace(
        source,
        "                    self.optimizer.step()\n",
        '                    if not getattr(self, "_rlvp_no_optimizer", False):\n                        self.optimizer.step()\n',
    )
    source = replace(
        source,
        "                    self._update_moe_gate_bias_if_supported()\n",
        '                    if not getattr(self, "_rlvp_no_optimizer", False):\n                        self._update_moe_gate_bias_if_supported()\n',
    )
    source = replace(
        source,
        '                            "optimizer_update",\n',
        '                            "backward_probe_no_update" if getattr(self, "_rlvp_no_optimizer", False) else "optimizer_update",\n',
    )
    source = replace(
        source,
        "            if not eval_mode:\n                self.scheduler.step()\n",
        '            if not eval_mode and not getattr(self, "_rlvp_no_optimizer", False):\n                self.scheduler.step()\n',
    )
    source = replace(
        source,
        "                    finish_gradient_receipt(gradient_archive, self.model, self.max_grad_norm, grad_norm)\n",
        "                    finish_gradient_receipt(gradient_archive, self.model, self.max_grad_norm, grad_norm)\n                    fixed_learner_logprob_probe(self, _optimizer_step_value(self.optimizer))\n",
    )
    return source


PATCHES = {
    "nemo_rl/algorithms/loss/loss_functions.py": patch_loss,
    "nemo_rl/distributed/model_utils.py": patch_model_utils,
    "nemo_rl/algorithms/advantage_estimator.py": patch_advantage,
    "nemo_rl/algorithms/grpo.py": patch_grpo,
    "nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py": patch_worker,
}


def apply(root):
    root = Path(root)
    pending = {}
    manifest = {}
    for relative, patch in PATCHES.items():
        path = root / relative
        before = path.read_text()
        after = patch(before)
        compile(after, str(path), "exec")
        pending[path] = after
        manifest[relative] = {
            "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        }
    # Validate all anchors/syntax before modifying any file.
    for path, after in pending.items():
        path.write_text(after)
    helper = root / "nemo_rl/utils/rlvp_modal_audit.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("rlvp_modal_audit.py"), helper)
    manifest[str(helper.relative_to(root))] = {
        "after_sha256": hashlib.sha256(helper.read_bytes()).hexdigest()
    }
    (root / "rlvp-modal-audit-patches.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo-root", required=True)
    args = parser.parse_args()
    print(json.dumps(apply(args.nemo_root), indent=2, sort_keys=True))
