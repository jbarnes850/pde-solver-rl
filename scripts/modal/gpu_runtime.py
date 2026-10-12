"""Single-host orchestration around the existing NeMo/Polar training entrypoint.

No model loader or learning loop is defined here. All optimization, replay and
FP8 refit remain in the byte-pinned NeMo/Polar implementation.
"""

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

BRIDGE = Path("/opt/rlvp-bridge")
SCRIPTS = Path("/opt/rlvp-modal")
MODEL = "/inputs/models/Qwen3.5-9B"
NEMO_PYTHON = "/opt/nemo_rl_venv/bin/python"
BOOTSTRAP = Path("/inputs/checkpoints/sft128/step_0")
WEIGHTS_SHA256 = "15c7e44e754956a2ded85159c2cf0c036c6d3fe33d41eb4fe3f35cec87c4950e"
FLASHINFER_CACHE = Path("/root/.cache/flashinfer")
PERSISTENT_FLASHINFER_CACHE = Path("/runs/flashinfer-cache")


def copy_flashinfer_cache(source, destination):
    """Reuse this campaign's pinned kernels at their original absolute paths."""
    import shutil

    if not source.is_dir():
        return False
    shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=shutil.copy2)
    return True


def _validated_replay_cursor(state, current_step, rows, events, responses_per_prompt=8):
    """Accept an empty boundary or one exact, complete, next-target batch."""
    trajectories = state["trajectories"]
    versions = state["trajectory_versions"]
    targets = state["target_weight_versions"]
    if len(trajectories) != len(versions) or len(trajectories) != len(targets):
        raise ValueError("resume replay lengths disagree")
    if not trajectories:
        return {"prefetched_groups": [], "next_group_offset": current_step * 8}
    if len(trajectories) != 8 or any(int(target) != current_step for target in targets):
        raise ValueError("resume requires exactly eight complete groups for the next target")
    if any(not max(0, current_step - 1) <= int(version) <= current_step for version in versions):
        raise ValueError("prefetched replay exceeds the one-version age contract")
    receipts = [event for event in events if event.get("event") == "group_before_replay"]

    def digest(values):
        return hashlib.sha256(json.dumps(values, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

    matched = []
    for trajectory, version, target in zip(trajectories, versions, targets, strict=True):
        batch = trajectory["batch"]
        message_logs = batch["message_log"]
        if len(message_logs) != responses_per_prompt or len(batch["total_reward"]) != responses_per_prompt:
            raise ValueError("prefetched replay has an incomplete response group")
        token_rows, logprob_rows = [], []
        for messages in message_logs:
            tokens, logprobs = [], []
            for message in messages:
                if "generation_logprobs" not in message:
                    continue
                tokens.extend(int(value) for value in message["token_ids"])
                logprobs.extend(float(value) for value in message["generation_logprobs"])
            if not tokens or len(tokens) != len(logprobs):
                raise ValueError("prefetched replay token/logprob coverage differs")
            token_rows.append(tokens)
            logprob_rows.append(logprobs)
        tokens_sha, logprobs_sha = digest(token_rows), digest(logprob_rows)
        matches = [event for event in receipts
                   if event.get("action_token_ids_sha256") == tokens_sha
                   and event.get("behavior_logprobs_sha256") == logprobs_sha]
        if len(matches) != 1:
            raise ValueError("prefetched replay has no unique original token/logprob receipt")
        event = matches[0]
        group_id = int(event["group_id"])
        if not current_step * 8 <= group_id < (current_step + 1) * 8:
            raise ValueError("prefetched replay group is outside the next manifest block")
        if group_id >= len(rows) or event["task_name"] != rows[group_id]["physical_case_id"]:
            raise ValueError("prefetched replay physical case differs from the ordered manifest")
        if int(event["generation_weight_version"]) != int(version) or int(event["target_weight_version"]) != int(target):
            raise ValueError("prefetched replay version differs from the original receipt")
        rewards = [float(value) for value in batch["total_reward"]]
        if rewards != event["rewards"]:
            raise ValueError("prefetched replay rewards differ from the original receipt")
        matched.append({"group_id": group_id, "physical_case_id": event["task_name"],
                        "generation_weight_version": int(version), "target_weight_version": int(target),
                        "action_token_ids_sha256": tokens_sha, "behavior_logprobs_sha256": logprobs_sha})
    if {group["group_id"] for group in matched} != set(range(current_step * 8, (current_step + 1) * 8)):
        raise ValueError("prefetched replay duplicates or omits a manifest group")
    return {"prefetched_groups": matched, "next_group_offset": (current_step + 1) * 8}


def inspect_training_resume(checkpoint, max_steps, manifest_path, source_log_path, responses_per_prompt=8):
    """Read the trusted native checkpoint with NeMo's own torch environment."""
    import io
    import pickle

    import torch

    from rlvp.checkpoint_bootstrap import tree_sha256
    from rlvp.variance_probe import load_locked_rows

    info = json.loads((checkpoint / "training_info.json").read_text())
    current_step = int(info["current_step"])
    if max_steps <= current_step:
        raise ValueError("training continuation requires an absolute horizon beyond the saved optimizer step")
    if int(info["consumed_samples"]) != current_step * 8:
        raise ValueError("checkpoint consumed cases differ from the fixed eight-group step contract")
    metadata_path = checkpoint / "policy/optimizer/optim/.metadata"
    with metadata_path.open("rb") as handle:
        optimizer_metadata = pickle.load(handle)
    adam_steps = []
    for index, storage in optimizer_metadata.storage_data.items():
        if index.fqn.endswith(".step"):
            with (metadata_path.parent / storage.relative_path).open("rb") as handle:
                handle.seek(storage.offset)
                value = torch.load(io.BytesIO(handle.read(storage.length)), map_location="cpu", weights_only=True)
            adam_steps.append(float(value))
    if not adam_steps or set(adam_steps) != {float(current_step)}:
        raise ValueError("checkpoint Adam steps do not match the saved native training step")
    state = torch.load(checkpoint / "replay_buffer.pt", map_location="cpu", weights_only=False)
    events = []
    if state["trajectories"]:
        for line in source_log_path.read_text().splitlines():
            if "[rlvp_receipt] " in line:
                try:
                    event, _ = json.JSONDecoder().raw_decode(line.split("[rlvp_receipt] ", 1)[1])
                    events.append(event)
                except json.JSONDecodeError:
                    continue
    result = _validated_replay_cursor(
        state, current_step, load_locked_rows(manifest_path), events, responses_per_prompt,
    )
    result.update({"current_step": current_step, "consumed_samples": info["consumed_samples"],
                   "optimizer_step_tensor_count": len(adam_steps), "optimizer_step": current_step,
                   "checkpoint_weights_tree_sha256": tree_sha256(checkpoint / "policy/weights"),
                   "replay_sha256": hashlib.sha256((checkpoint / "replay_buffer.pt").read_bytes()).hexdigest(),
                   "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   "source_log": str(source_log_path), "replay_preserved_unchanged": True})
    return result


def validate_training_resume(checkpoint, max_steps, manifest_path, responses_per_prompt=8):
    source_log = checkpoint.parents[2] / "logs/grpo-nemo-polar.log"
    result = subprocess.run(
        [NEMO_PYTHON, "-c",
         ("import json,sys; from pathlib import Path; from gpu_runtime import inspect_training_resume; "
          "print(json.dumps(inspect_training_resume(Path(sys.argv[1]),int(sys.argv[2]),"
          "Path(sys.argv[3]),Path(sys.argv[4]),int(sys.argv[5]))))"),
         str(checkpoint), str(max_steps), str(manifest_path), str(source_log), str(responses_per_prompt)],
        capture_output=True, text=True, check=True, timeout=120,
    )
    return json.loads(result.stdout)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_inputs(manifest_path=Path("/inputs/input-manifest.json"), deadline=None):
    """Rehash every uploaded base/bootstrap byte before starting Ray or models."""
    manifest = json.loads(manifest_path.read_text())
    receipts = []
    for label, root in (("base_model", Path(MODEL)), ("bootstrap", BOOTSTRAP)):
        entries = manifest[label]["files"]
        expected = {entry["path"] for entry in entries}
        actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        if expected != actual:
            raise RuntimeError(f"input tree differs for {label}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        for entry in entries:
            if deadline and time.monotonic() >= deadline:
                raise TimeoutError("input verification exceeded invocation deadline")
            path = root / entry["path"]
            if not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError("manifest path escapes input root")
            if path.stat().st_size != entry["bytes"]:
                raise RuntimeError(f"input size differs: {label}/{entry['path']}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != entry["sha256"]:
                raise RuntimeError(f"input hash differs: {label}/{entry['path']}")
        receipts.append({"tree": label, "files_verified": len(entries),
                         "bytes_verified": sum(row["bytes"] for row in entries)})
    if manifest["bootstrap_weights_tree_sha256"] != WEIGHTS_SHA256:
        raise RuntimeError("wrong SFT step-128 bootstrap")
    return {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "trees": receipts, "all_input_bytes_verified": True}


def apply_historical_patches():
    import shutil

    patches = BRIDGE / "scripts/patch"
    mappings = {
        "nemo_rl_fp8_broadcast/dtensor_policy_worker_v2.py": "models/policy/workers/dtensor_policy_worker_v2.py",
        "nemo_rl_fp8_broadcast/vllm_backend.py": "models/generation/vllm/vllm_backend.py",
        "nemo_rl_fp8_broadcast/fp8_broadcast_quant.py": "utils/fp8_broadcast_quant.py",
        "nemo_rl_fp8_broadcast/packed_tensor.py": "utils/packed_tensor.py",
        "nemo_rl_qwen35_fp8/fp8.py": "models/generation/vllm/quantization/fp8.py",
        "nemo_rl_memory/model_utils.py": "distributed/model_utils.py",
        "nemo_rl_memory/train.py": "models/automodel/train.py",
        "nemo_rl_memory/grpo.py": "algorithms/grpo.py",
        "nemo_rl_memory/advantage_estimator.py": "algorithms/advantage_estimator.py",
    }
    receipts = []
    for source, destination in mappings.items():
        source, destination = patches / source, Path("/opt/nemo-rl/nemo_rl") / destination
        shutil.copy2(source, destination)
        receipts.append({"source": str(source), "destination": str(destination),
                         "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
    return receipts


def maxrl_start_step(config):
    """Native step-32 continuation (default) or a fresh start from the zero-delta base adapter."""
    start = int(config.get("maxrl_start_step", 32))
    if start not in (0, 32, 64, 96, 128, 160, 192, 224, 256):
        raise ValueError("MaxRL start step must be 0 (zero-delta base) or a native resume at a multiple of 32 up to 256")
    return start


def bootstrap_source(config):
    """Weights-only step-0 policy: the SFT bootstrap by default, or an explicit /runs checkpoint."""
    path = config.get("bootstrap_checkpoint")
    if path is None:
        return BOOTSTRAP, WEIGHTS_SHA256
    if not Path(path).is_relative_to("/runs") or not config.get("bootstrap_weights_tree_sha256"):
        raise ValueError("an explicit bootstrap must live under /runs with a frozen weights hash")
    return Path(path), config["bootstrap_weights_tree_sha256"]


def kl_reference_path(config):
    """The frozen SFT reference by default; a base start freezes a new zero-delta reference."""
    path = config.get("kl_reference_path", "/runs/frozen-sft-reference.pt")
    if maxrl_start_step(config) == 0 and config.get("advantage_normalizer") == "mean_plus_epsilon":
        if path == "/runs/frozen-sft-reference.pt" or Path(path).exists():
            raise ValueError("a base start must freeze a new, not-yet-existing KL reference")
    return path


LENGTH_CONTRACTS = {9728: 9216, 16384: 15872}  # context tokens -> generated tokens


def length_contract(config):
    """The proven 9,216/9,728 contract by default; 15,872/16,384 only as an explicit qualified choice."""
    context = int(config.get("max_model_len", 9728))
    if context not in LENGTH_CONTRACTS:
        raise ValueError("context length must be 9728 (proven) or 16384 (qualification)")
    return LENGTH_CONTRACTS[context], context


def gpu_pressure_limit(config):
    """Watchdog threshold matches the learner cap: 0.85 by default, 0.97 under the 16k-only 0.95 cap."""
    return 0.97 if learner_memory_fraction(config) == "0.95" else 0.85


def learner_memory_fraction(config):
    """0.85 is the proven fail-closed cap; 0.95 is allowed only for the 16k contract, whose
    full-vocabulary logits-gradient buffer (~7.6 GiB) does not fit under 0.85 on one H200."""
    value = str(config.get("learner_cuda_memory_fraction", "0.85"))
    if value not in ("0.85", "0.95"):
        raise ValueError("learner CUDA memory fraction must be 0.85 or 0.95")
    if value == "0.95" and length_contract(config)[1] != 16384:
        raise ValueError("the 0.95 learner memory fraction is reserved for the 16k contract")
    return value


def maxrl_target_lr(config):
    """2e-6 was the qualified rate from SFT; LoRA from a zero B adapter over few steps needs more."""
    value = str(config.get("maxrl_target_lr", "2e-6"))
    if value not in ("2e-6", "5e-6", "1e-5"):
        raise ValueError("MaxRL target LR must be 2e-6, 5e-6 or 1e-5")
    return value


def expected_maxrl_lr(config, step):
    from maxrl_training import scheduled_lr

    schedule = {"initial_lr": 1e-6, "target_lr": float(maxrl_target_lr(config)),
                "warmup_updates": int(config.get("maxrl_warmup_updates", 0))}
    return scheduled_lr(schedule, int(step) - maxrl_start_step(config))


def maxrl_epsilon(config):
    """Original runs used 1e-6; 1e-2 is the frozen absolute anchor for the anchored reward."""
    value = str(config.get("maxrl_epsilon", "1e-6"))
    if value not in ("1e-6", "1e-2"):
        raise ValueError("MaxRL epsilon must be the original 1e-6 or the frozen 1e-2")
    if value == "1e-2" and config.get("reward_mode") != "anchored":
        raise ValueError("epsilon 1e-2 is calibrated only for the anchored [0, 1] reward")
    return value


def response_group_size(config):
    """Keep the established G8 contract unless the reviewed MaxRL branch is explicit."""
    size = int(config.get("responses_per_prompt", 8))
    maxrl = config.get("advantage_normalizer") == "mean_plus_epsilon"
    if size not in (8, 16) or (size == 16 and not maxrl):
        raise ValueError("response group size requires the reviewed G8 or MaxRL G16 contract")
    if maxrl:
        if config.get("mode", "train") != "train" or size != 16:
            raise ValueError("MaxRL continuation requires training with G16")
        if int(config.get("concurrency", 128)) != 128:
            raise ValueError("MaxRL continuation retains native concurrency128")
        if float(config.get("learning_rate", 1e-6)) != 1e-6:
            raise ValueError("MaxRL starts with one1e-6 qualification update")
    elif config.get("advantage_normalizer") not in (None, "std_plus_epsilon"):
        raise ValueError("unreviewed advantage normalizer")
    return size


def nemo_overrides(run_dir, config):
    """Native same-host allocation: 2 total GPUs minus 1 inference = 1 learner."""
    steps, concurrency = int(config.get("steps", 1)), int(config.get("concurrency", 8))
    if not 1 <= steps <= 288 or concurrency not in (8, 16, 32, 64, 128):
        raise ValueError("steps/concurrency outside bounded campaign")
    lr = float(config.get("learning_rate", 2e-6))
    if lr not in (1e-6, 2e-6, 5e-6):
        raise ValueError("LR outside approved candidate range")
    group_size = response_group_size(config)
    values = {
        "grpo.async_grpo.enabled": "true",
        "grpo.async_grpo.max_trajectory_age_steps": "1",
        "grpo.async_grpo.in_flight_weight_updates": "true",
        "grpo.async_grpo.recompute_kv_cache_after_weight_updates": "true",
        "grpo.num_prompts_per_step": "8", "grpo.num_generations_per_prompt": str(group_size),
        # 512 cases at 8 groups per update: steps beyond 64 begin a second pass over the manifest.
        "grpo.max_num_steps": str(steps), "grpo.max_num_epochs": str(-(-steps * 8 // 512)), "grpo.seed": "17",
        "grpo.val_at_start": "false", "grpo.val_at_end": "false", "grpo.val_period": "0",
        "grpo.overlong_filtering": "false", "grpo.use_dynamic_sampling": "false",
        "grpo.reward_scaling.enabled": "false", "grpo.reward_shaping.enabled": "false",
        "grpo.normalize_rewards": "true", "grpo.use_leave_one_out_baseline": "false",
        "loss_fn.reference_policy_kl_penalty": "0.001",
        "loss_fn.use_importance_sampling_correction": "true",
        "policy.model_name": MODEL, "policy.tokenizer.name": MODEL,
        "policy.precision": "bfloat16", "policy.train_global_batch_size": str(8 * group_size),
        "policy.train_micro_batch_size": "1", "policy.logprob_batch_size": "1",
        "policy.generation_batch_size": "1", "policy.max_total_sequence_length": str(length_contract(config)[1]),
        "policy.sequence_packing.enabled": "false", "policy.dynamic_batching.enabled": "false",
        "policy.dtensor_cfg.activation_checkpointing": "true",
        "+policy.dtensor_cfg.automodel_kwargs.force_hf": "true",
        "++policy.dtensor_cfg.automodel_kwargs.freeze_config.freeze_vision_tower": "true",
        "++policy.dtensor_cfg.automodel_kwargs.freeze_config.freeze_audio_tower": "true",
        "policy.dtensor_cfg.lora_cfg.enabled": "true", "policy.dtensor_cfg.lora_cfg.dim": "32",
        "policy.dtensor_cfg.lora_cfg.alpha": "32",
        "policy.dtensor_cfg.lora_cfg.target_modules": "[model.language_model.*proj*]",
        "policy.dtensor_cfg.lora_cfg.exclude_modules": "[]",
        "policy.dtensor_cfg.lora_cfg.match_all_linear": "false",
        "policy.dtensor_cfg.lora_cfg.dropout": "0.0",
        "policy.dtensor_cfg.lora_cfg.dropout_position": "post",
        "policy.dtensor_cfg.lora_cfg.lora_A_init": "xavier",
        "policy.dtensor_cfg.lora_cfg.use_triton": "true",
        "policy.optimizer.name": "torch.optim.AdamW", "policy.optimizer.kwargs.lr": str(lr),
        "policy.optimizer.kwargs.weight_decay": "0.01", "policy.optimizer.kwargs.betas": "[0.9,0.999]",
        "policy.optimizer.kwargs.eps": "1e-08", "++policy.logprob_chunk_size": "512",
        "++policy.dtensor_cfg.clear_cache_every_n_steps": "1",
        "++policy.tokenizer.chat_template_kwargs.enable_thinking": "false",
        "policy.generation.backend": "vllm", "policy.generation.max_new_tokens": str(length_contract(config)[0]),
        "policy.generation.temperature": "0.7", "policy.generation.top_p": "1.0",
        "policy.generation.top_k": "null", "policy.generation.port_range_low": "31000",
        "policy.generation.port_range_high": "31001", "policy.generation.vllm_cfg.async_engine": "true",
        "+policy.generation.vllm_cfg.expose_http_server": "true",
        "policy.generation.vllm_cfg.enable_vllm_metrics_logger": "false",
        "policy.generation.vllm_cfg.max_model_len": str(length_contract(config)[1]),
        "policy.generation.vllm_cfg.gpu_memory_utilization": "0.65",
        "policy.generation.vllm_cfg.enforce_eager": "false",
        "policy.generation.vllm_cfg.precision": "fp8",
        "policy.generation.vllm_cfg.kv_cache_dtype": "auto",
        "++policy.generation.vllm_cfg.enable_prefix_caching": "true",
        "++policy.generation.vllm_kwargs.max_num_seqs": str(concurrency),
        "++policy.generation.vllm_kwargs.max_num_batched_tokens": "4096",
        "++policy.generation.vllm_kwargs.enable_chunked_prefill": "true",
        "+policy.generation.vllm_kwargs.generation_config": "vllm",
        "++policy.generation.vllm_kwargs.mamba_cache_mode": "align",
        "++policy.generation.vllm_kwargs.language_model_only": "true",
        "++policy.generation.vllm_cfg.quantization_ignored_layer_kws": "[linear_attn.in_proj_a,linear_attn.in_proj_b]",
        "policy.generation.colocated.enabled": "false",
        "policy.generation.colocated.resources.gpus_per_node": "1",
        "policy.generation.colocated.resources.num_nodes": "1",
        "data.train.dataset_name": "ResponseDataset",
        "+data.train.data_path": str(run_dir / "data/train.jsonl"),
        "+data.train.input_key": "input", "+data.train.output_key": "output",
        "data.train.split_validation_size": "0.0", "data.default.processor": "math_hf_data_processor",
        "data.default.env_name": "math", "env.math.num_workers": "1", "data_plane.enabled": "false",
        "logger.log_dir": str(run_dir / "logs/grpo-polar"), "logger.wandb_enabled": "false",
        "logger.tensorboard_enabled": "false", "logger.mlflow_enabled": "false",
        "logger.swanlab_enabled": "false", "logger.monitor_gpus": "false",
        "checkpointing.enabled": "true", "checkpointing.checkpoint_dir": str(run_dir / "checkpoints/grpo"),
        "checkpointing.save_period": "1", "checkpointing.save_optimizer": "true",
        "checkpointing.metric_name": "null", "checkpointing.keep_top_k": "3",
        "cluster.gpus_per_node": "2", "cluster.num_nodes": "1",
    }
    # Algorithm choice is an audited config overlay, not a second trainer.
    allowed = {"loss_fn.use_cispo", "loss_fn.ratio_clip_min", "loss_fn.ratio_clip_max",
               "loss_fn.use_on_policy_kl_approximation", "loss_fn.token_level_loss",
               "grpo.normalize_rewards", "grpo.use_leave_one_out_baseline",
               "loss_fn.sequence_level_importance_ratios", "loss_fn.use_importance_sampling_correction",
               "loss_fn.ratio_clip_c", "++loss_fn.rlvp_fixed_response_tokens"}
    for key, value in config.get("algorithm_overrides", {}).items():
        if key not in allowed:
            raise ValueError(f"unreviewed algorithm override: {key}")
        values[key] = "null" if value is None else (str(value).lower() if isinstance(value, bool) else str(value))
    # Pinned NeMo sets logprobs_mode="processed_logprobs" explicitly in its
    # constructor; duplicating it in vllm_kwargs raises before engine creation.
    return [f"{key}={value}" for key, value in values.items()] + ["~policy.scheduler"]


def prepare_tasks(run_dir, manifest_path, config, environment, deadline=None):
    import shutil

    from rlvp.contracts import RewardMode
    from rlvp.data import materialize_locked_manifest_dataset
    from rlvp.prompts import SYSTEM_PROMPT

    dataset = run_dir / "data/locked-gym.jsonl"
    reward_mode = RewardMode(config.get("reward_mode", "full"))
    if reward_mode not in (RewardMode.FULL, RewardMode.ANCHORED):
        raise ValueError("training reward must be the paper full reward or the frozen anchored transform")
    system_prompt = config.get("system_prompt", SYSTEM_PROMPT)
    if hashlib.sha256(system_prompt.encode()).hexdigest() != config.get(
            "expected_system_prompt_sha256", hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()):
        raise ValueError("training system prompt differs from the frozen prompt hash")
    materialize_locked_manifest_dataset(
        manifest_path=Path(manifest_path), output_path=dataset,
        verifier_base_url="http://127.0.0.1:18081", reward_mode=reward_mode,
        system_prompt=system_prompt,
    )
    schedule_path = Path(config["schedule_path"])
    if config.get("advantage_normalizer") == "mean_plus_epsilon":
        for path, key in ((schedule_path, "expected_schedule_sha256"),
                          (Path(manifest_path), "expected_train_manifest_sha256")):
            if hashlib.sha256(path.read_bytes()).hexdigest() != config[key]:
                raise ValueError(f"MaxRL frozen input hash differs: {key}")
    schedule = [json.loads(line) for line in schedule_path.read_text().splitlines() if line.strip()]
    records = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    by_case = {entry["physical_case_id"]: entry for entry in schedule}
    if len(records) != 512 or len(by_case) != 512:
        raise ValueError("every phase must use the same complete ordered 512-case training manifest")
    for index, record in enumerate(records):
        entry = by_case[record["id"]]
        if int(entry["group_index"]) != index:
            raise ValueError("training manifest and declared case/seed schedule order differ")
        size = response_group_size(config)
        seeds = entry["response_seeds"]
        if len(seeds) != size or len(set(seeds)) != size:
            raise ValueError("response seed schedule differs from the complete group size")
        if size == 16:
            identities = entry.get("response_uids", [])
            if len(identities) != size or len(set(identities)) != size:
                raise ValueError("MaxRL requires all16 full response identities")
        record["metadata"]["modal_schedule"] = entry
    dataset.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    shutil.copy2(schedule_path, run_dir / "data/schedule.jsonl")
    rows = sum(1 for line in dataset.open() if line.strip())
    prepare = [NEMO_PYTHON, "-m", "nemo_polar_bridge.datasets.prepare_dataset",
        "--dataset-id", "rlvp_locked_manifest", "--local-jsonl", str(dataset),
        "--limit", str(rows), "--scan-rows", str(rows), "--output-dir", str(run_dir),
        "--repo-host", str(BRIDGE), "--model-name", "Qwen/Qwen3.5-9B", "--model-path", MODEL,
        "--head-ip", "127.0.0.1", "--worker-ip", "127.0.0.1", "--head-hostname", "modal",
        "--worker-hostname", "modal", "--runtime-image", "trusted-http-only",
        "--model-max-tokens", str(length_contract(config)[0]), "--model-temperature", "0.7", "--model-top-p", "1.0",
        "--verifier-type", "rlvp_continuous", "--difficulty-band", "mixed-seen",
        "--model-request-timeout-seconds", "1800", "--task-timeout-seconds", "2100"]
    subprocess.run(prepare, check=True, env=environment,
                   timeout=max(1, deadline-time.monotonic()) if deadline else 120)
    template_file = run_dir / "polar/task_template.json"
    template = json.loads(template_file.read_text())
    command = template["agent"]["custom_shell"]["command"]
    seed_anchor = 'base = os.environ["OPENAI_BASE_URL"].rstrip("/")'
    if command.count(seed_anchor) != 1:
        raise RuntimeError("fixed HTTP harness seed anchor changed")
    command = command.replace(seed_anchor,
        'schedule = task["metadata"]["modal_schedule"]\n'
        'payload["seed"] = int(schedule["response_seeds"][int(os.environ["POLAR_RESPONSE_INDEX"])])\n'
        + seed_anchor)
    template["agent"]["custom_shell"]["command"] = command
    template["runtime"]["prepare"] = []
    template["runtime"].setdefault("env", {})["POLAR_RESPONSE_INDEX"] = "{attempt_index}"
    template["runtime"]["import_path"] = "trusted_runtime:TrustedHTTPRuntime"
    template["runtime"]["kwargs"] = {"trusted_command_sha256": hashlib.sha256(command.encode()).hexdigest()}
    write_json(template_file, template)
    topology = f"""rollout:
  host: 127.0.0.1
  port: 19080
  public_url: http://127.0.0.1:19080
  save_dir: {run_dir}/polar/rollout_results
  dispatch_poll_interval_seconds: 0.25
  callback_grace_seconds: 60
gateway:
  heartbeat_interval_seconds: 2
  nodes:
    - id: nemo-polar-gateway
      host: 127.0.0.1
      port: 19100
      public_url: http://127.0.0.1:19100
      max_init_workers: {config.get('concurrency', 8)}
      max_run_workers: {config.get('concurrency', 8)}
      max_postrun_workers: 8
      model_served: {MODEL}
      inference:
        engine: vllm
        base_url: http://127.0.0.1:31000
"""
    (run_dir / "polar/topology.yaml").write_text(topology)
    checkpoint = run_dir / "checkpoints/grpo/step_0"
    resume = config.get("resume_checkpoint")
    if resume:
        source = Path(resume)
        if not source.is_relative_to("/runs"):
            raise ValueError("resume checkpoint must be from this campaign's volume")
        if config.get("mode", "train") == "train":
            resume_cursor = validate_training_resume(
                source, int(config["steps"]), Path(manifest_path), response_group_size(config),
            )
            if config.get("advantage_normalizer") == "mean_plus_epsilon":
                if resume_cursor["current_step"] != maxrl_start_step(config) or resume_cursor["prefetched_groups"]:
                    raise ValueError("MaxRL branch must start from its declared native step with empty replay")
                if resume_cursor["checkpoint_weights_tree_sha256"] != config["expected_parent_weights_sha256"]:
                    raise ValueError("MaxRL branch parent weights differ from the selected checkpoint")
            environment["NEMO_POLAR_GROUP_OFFSET"] = str(resume_cursor["next_group_offset"])
            write_json(run_dir / "receipts/resume-cursor.json", resume_cursor)
        checkpoint = checkpoint.parent / source.name
    else:
        source, expected_tree = bootstrap_source(config)
        if source != BOOTSTRAP:  # the default SFT bootstrap is rehashed by verify_inputs
            from rlvp.checkpoint_bootstrap import tree_sha256

            if tree_sha256(source / "policy/weights") != expected_tree:
                raise ValueError("bootstrap weights differ from the frozen hash")
    shutil.copytree(source, checkpoint)
    return rows


def run_evaluation(run_dir, config, health, trainer, started, wall_seconds, environment):
    """Use the existing locked variance-probe sampler after the native SFT refit."""
    import yaml
    from modal_refit_probe import record_fp8_probe

    from rlvp.checkpoint_bootstrap import tree_sha256
    from rlvp.variance_probe import build_parser

    ready = run_dir / "logs/refit-only-ready.json"
    while not ready.exists():
        if trainer.poll() is not None:
            raise RuntimeError(f"refit-only server exited {trainer.returncode}")
        if time.monotonic() - started >= wall_seconds:
            raise TimeoutError("refit-only readiness exceeded invocation budget")
        time.sleep(1)
    record_fp8_probe("http://127.0.0.1:31000", "evaluation", int(config.get("weight_version", 0)))
    evaluation = yaml.safe_load(Path(config["evaluation_config"]).read_text())
    # Runtime fields describe this same-host Modal invocation.
    evaluation["runtime"].update({
        "image_ref": "modal-neMo-c236061b-from-ngc-v0.7.0",
        "image_digest": "sha256:9392cc2cc07140f6a4cbd30f31ebffd0a54de3ded25c5a3b49505407a779e4f9",
        # Pinned native vLLM uses engine seed0; request seeds remain explicit.
        "seed": 0,
        "gpu_memory_utilization": 0.65, "max_model_len": length_contract(config)[1],
        "max_num_batched_tokens": 4096, "max_num_seqs": int(config.get("concurrency", 8)),
    })
    case_concurrency = int((evaluation.get("execution") or {}).get("case_concurrency", 1))
    if 8 * case_concurrency != int(config.get("concurrency", 8)):
        raise ValueError("evaluation concurrency must equal eight responses times case_concurrency")
    evaluation_path = run_dir / "evaluation-contract.yaml"
    evaluation_path.write_text(yaml.safe_dump(evaluation, sort_keys=False))
    refit_lines = [line.split("[rlvp_receipt] ", 1)[1] for line in
                   (run_dir / "logs/grpo-nemo-polar.log").read_text().splitlines()
                   if "[rlvp_receipt] " in line and '"event": "refit_consumer"' in line]
    successful = [json.loads(line) for line in refit_lines if json.loads(line).get("success")]
    if not successful:
        raise RuntimeError("evaluation requires a successful actual refit receipt")
    checkpoint = Path(config.get("resume_checkpoint") or bootstrap_source(config)[0])
    policy_artifact = {
        "schema": "rlvp.modal-policy.v1", "policy_id": config["run_id"],
        "base_model_tree_sha256": config["model_tree_sha256"],
        "tokenizer_json_sha256": hashlib.sha256(Path(MODEL, "tokenizer.json").read_bytes()).hexdigest(),
        "checkpoint_tree_sha256": tree_sha256(checkpoint / "policy/weights"),
        "lora_rank": 32, "lora_alpha": 32,
        "weight_version": config.get("weight_version", 0), "served_model_name": MODEL,
        "refit_receipt_sha256": hashlib.sha256(json.dumps(successful[-1], sort_keys=True).encode()).hexdigest(),
    }
    artifact_path = run_dir / "receipts/policy-artifact.json"
    write_json(artifact_path, policy_artifact)
    # Check frozen actual refit identity before variance_probe opens any test rows.
    from sealed_eval_gate import assert_frozen_policy

    sealed_receipt = assert_frozen_policy(
        config, evaluation, policy_artifact, json.loads(ready.read_text()), health,
    )
    if sealed_receipt is not None:
        write_json(run_dir / "receipts/sealed-policy-identity.json", sealed_receipt)
    args = build_parser().parse_args([
        "--manifest", config["evaluation_manifest"], "--experiment-config", str(evaluation_path),
        "--out", str(run_dir / "evaluation"), "--model-url", "http://127.0.0.1:31000/v1/chat/completions",
        "--model-name", MODEL, "--verifier-base-url", "http://127.0.0.1:18081",
        "--expected-verifier-cases", str(health["cases"]), "--case-store-sha256", config["case_store_sha256"],
        "--solver-image-digest", config["solver_image_digest"], "--model-tree-sha256", config["model_tree_sha256"],
        "--policy-artifact-receipt", str(artifact_path), "--attempt-index", "0", "--workers", "8",
        "--transport-retries", "0",
    ])
    # Execute in a child so the same wall/cost deadline bounds the whole evaluation.
    args_path = run_dir / "evaluation-args.json"
    write_json(args_path, vars(args))
    command = [NEMO_PYTHON, "-c",
               ("import argparse,json,sys; from rlvp.variance_probe import run_probe; "
               "result=run_probe(argparse.Namespace(**json.load(open(sys.argv[1])))); "
               "print(json.dumps(result))"), str(args_path)]
    with (run_dir / "logs/evaluation.log").open("w") as output:
        subprocess.run(command, check=True, env=environment, stdout=output, stderr=subprocess.STDOUT,
                       timeout=max(1, wall_seconds - (time.monotonic() - started)))
    (run_dir / "stop-refit-only").touch()


def audit_completed_training(run_dir, responses_per_prompt=8, start_step=32, config=None):
    events = []
    for line in (run_dir / "logs/grpo-nemo-polar.log").read_text().splitlines():
        if "[rlvp_receipt] " in line:
            text = line.split("[rlvp_receipt] ", 1)[1]
            try:
                value, _ = json.JSONDecoder().raw_decode(text)
                if isinstance(value, dict):
                    events.append(value)
            except json.JSONDecodeError:
                continue
    generations = [row for row in events if row.get("event") == "post_refit_generation"]
    refits = [row for row in events if row.get("event") == "refit_consumer"]
    gradients = [json.loads(path.read_text()) for path in (run_dir / "receipts/gradients").glob("*.json")]
    batches = [json.loads(path.read_text()) for path in (run_dir / "receipts/batches").glob("*.json")]
    checkpoints = [path for path in (run_dir / "checkpoints/grpo").glob("step_*")
                   if path.name != "step_0" and (path / "training_info.json").is_file()
                   and (path / "policy/optimizer").exists()]
    checks = {
        "post_refit_generation": bool(generations) and all(row.get("success") is True for row in generations),
        "refit_receipts": len(refits) >= 2 and all(row.get("success") is True for row in refits),
        "actual_gradient_archives": bool(gradients),
        "changed_trainable_weights": bool(gradients) and any(
            row.get("trainable_weights_sha256_before") != row.get("trainable_weights_sha256_after")
            and row.get("trainable_update_l2", 0) > 0 for row in gradients),
        "all_rows_archived": bool(batches) and all(row.get("rows") == 8 * responses_per_prompt for row in batches),
        "optimizer_checkpoint_present": bool(checkpoints),
    }
    if responses_per_prompt == 16:
        decision_file = run_dir / "receipts/maxrl-training/lr-decision.json"
        decision = json.loads(decision_file.read_text()) if decision_file.exists() else {}
        checks["maxrl_first_update_and_refit_qualified"] = (
            decision.get("status") == "pass" and decision.get("promotion_ready") is True
            and decision.get("post_refit_passed") is True
        )
        checks["maxrl_actual_learning_rate_schedule"] = bool(gradients) and all(
            row.get("learning_rates") == [expected_maxrl_lr(config or {"maxrl_start_step": start_step}, row["step"])]
            for row in gradients
        )
        drains = [row for row in events if row.get("event") == "group_refit_drain"]
        checks["maxrl_whole_group_refit_drain"] = bool(drains) and all(
            row.get("success") is True and row.get("active_groups") == 0 for row in drains
        )
        completions = [row for row in events if row.get("event") == "group_refit_completed"]
        checks["maxrl_complete_groups_kept_their_version"] = bool(completions) and all(
            row.get("success") is True for row in completions
        )
        checks["maxrl_group_statistics_archived"] = bool(list(
            (run_dir / "receipts/maxrl-advantages").glob("*.json")))
    fp8_files = sorted((run_dir / "receipts/fixed-logprob-probe").glob("fp8-*.json"))
    fp8 = [json.loads(path.read_text()) for path in fp8_files]
    initial = [row for row in fp8 if row["stage"] == "initial"]
    updated = [row for row in fp8 if row["stage"] == "post-refit"]
    checks["fixed_fp8_likelihood_before_after"] = bool(initial and updated)
    checks["learner_fixed_likelihood_before_after"] = len(list(
        (run_dir / "receipts/learner-fixed-logprobs").glob("*.json"))) >= 2
    likelihood_changes = []
    if initial:
        for row in updated:
            if row["token_ids"] != initial[0]["token_ids"]:
                raise RuntimeError("fixed FP8 diagnostic prefix changed")
            differences = [b - a for a, b in zip(initial[0]["next_token_logprobs"], row["next_token_logprobs"], strict=True)]
            likelihood_changes.append({"weight_version": row["weight_version"],
                                       "maximum_absolute_logprob_change": max(abs(x) for x in differences),
                                       "l2_logprob_change": sum(x*x for x in differences) ** .5,
                                       "tokens_with_changed_logprob": sum(x != 0 for x in differences)})
    checks["fp8_fixed_likelihood_changed"] = any(
        row["tokens_with_changed_logprob"] > 0 for row in likelihood_changes)
    receipt = {"checks": checks, "passed": all(checks.values()),
               "fp8_likelihood_changes": likelihood_changes,
               "checkpoints": [str(path) for path in checkpoints]}
    write_json(run_dir / "receipts/pipeline-audit.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("pipeline proof failed: " + ", ".join(key for key, value in checks.items() if not value))
    return receipt


def run(config, verifier, volume):
    from verifier_proxy import start_proxy

    run_id = config["run_id"]
    group_size = response_group_size(config)
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("run_id must be lowercase alphanumeric/hyphen/underscore")
    wall_seconds = min(int(config.get("max_seconds", 7200)), 36000)
    budget = float(config.get("max_estimated_dollars", 40))
    if not 0 < budget <= 190:
        raise ValueError("invocation budget outside approved total ceiling")
    # Includes both GPU Function and CPU verifier; parent owns build/storage costs
    # and cumulative campaign ledger. 20% cushion over the scoped public estimate.
    hourly_rate = max(float(config.get("accounted_hourly_rate", 17.60)), 17.60)
    wall_seconds = min(wall_seconds, int(budget / hourly_rate * 3600))
    gpu_pressure_threshold = gpu_pressure_limit(config)
    run_dir = Path("/runs") / run_id
    run_dir.mkdir(exist_ok=False)
    for part in ("data", "logs", "polar", "receipts"):
        (run_dir / part).mkdir()
    started = time.monotonic()
    deadline = started + wall_seconds
    environment = {**os.environ,
        "PYTHONPATH": f"{BRIDGE}/src:/opt/rlvp-src:{SCRIPTS}:/opt/nemo-rl",
        "PYTHONUNBUFFERED": "1", "HF_HOME": "/runs/hf-cache",
        "VLLM_CACHE_ROOT": "/runs/vllm-cache", "RAY_USAGE_STATS_ENABLED": "0",
        "NRL_TRAIN_RECEIPTS": "1", "NRL_REFIT_RECEIPTS": "1", "NRL_POST_REFIT_PROBE": "1",
        "NRL_QWEN35_FP8_REFIT": "1", "NRL_QWEN35_TEXT_REFIT": "1",
        "NRL_REFIT_SKIP_OPT_OFFLOAD": "1", "NRL_REFIT_PROFILE": "1",
        "NRL_FP8_BROADCAST": "0", "NRL_REFIT_BUFFER_MEMORY_RATIO": "0.02",
        "NRL_ACTIVE_ONLY": "0", "RLVP_ACTIVE_ONLY_TRAINING": "0",
        "RLVP_ADVANTAGE_NORMALIZER": "legacy", "RLVP_MAXRL_TRAINING": "0",
        "NEMO_POLAR_GROUP_REFIT_BARRIER": "0",
        "NRL_LOGPROB_CHUNK_TOKENS": "64", "NRL_CUDA_MEMORY_FRACTION": learner_memory_fraction(config),
        "NRL_INITIAL_POLICY_WEIGHTS_PATH": str(bootstrap_source(config)[0] / "policy/weights"),
        "NRL_INITIAL_POLICY_WEIGHTS_TREE_SHA256": bootstrap_source(config)[1],
        "NRL_VALIDATION_MODE": "learning", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "eth0",
        "NCCL_NET": "Socket", "NCCL_NET_PLUGIN": "none",
        "GLOO_SOCKET_IFNAME": "eth0", "RAY_memory_usage_threshold": "0.88",
        "NEMO_POLAR_ROLLOUT_URL": "http://127.0.0.1:19080",
        "NEMO_POLAR_GATEWAY_URL": "http://127.0.0.1:19100",
        "NEMO_POLAR_TASK_TEMPLATE_PATH": str(run_dir / "polar/task_template.json"),
        "NEMO_POLAR_ATTEMPT_MATRIX_PATH": str(run_dir / "polar/attempt_matrix.json"),
        "NEMO_POLAR_GROUP_WORKERS": str(group_size),
        "NEMO_POLAR_CONCURRENT_GROUPS": str(max(1, int(config.get("concurrency", 8)) // group_size)),
        "NEMO_POLAR_REQUEST_TIMEOUT": "2100", "NEMO_POLAR_GATEWAY_DRAIN_TIMEOUT": "2100",
        # The frozen manifest already contains the declared seeded order, with
        # eight validated mixed-family cases first. Never shuffle it a second time.
        "NEMO_POLAR_EPOCH_SHUFFLE_SEED": "",
        "RLVP_RECEIPT_DIR": str(run_dir / "receipts"),
        "NRL_AUDIT_DIR": str(run_dir / "receipts"),
        "NRL_GROUP_BY_TRAJECTORY": "1",
        "NRL_SFT_REFERENCE_PATH": kl_reference_path(config),
        "NRL_FIXED_LOGPROB_PROBE_PATH": "/runs/fixed-logprob-probe.json",
        "NRL_FULL_LENGTH_BACKWARD_PROBE": "1" if config.get("full_length_backward_probe", False) else "0",
    }
    # Only the checkpoint validator may advance this campaign's collection cursor.
    environment.pop("NEMO_POLAR_GROUP_OFFSET", None)
    if config.get("advantage_normalizer") == "mean_plus_epsilon":
        start = maxrl_start_step(config)
        if not start + 1 <= int(config["steps"]) <= start + 32:
            raise ValueError("MaxRL horizon is at most 32 updates after its start step")
        if bool(config.get("resume_checkpoint")) != (start > 0):
            raise ValueError("a positive start step resumes a native checkpoint; start step 0 uses the zero-delta bootstrap")
        environment.update({
            "RLVP_ADVANTAGE_NORMALIZER": "mean_plus_epsilon",
            "RLVP_MAXRL_GROUP_SIZE": "16", "RLVP_MAXRL_EPSILON": maxrl_epsilon(config),
            "RLVP_MAXRL_TRAINING": "1", "RLVP_MAXRL_START_STEP": str(start),
            "RLVP_MAXRL_MIN_USABLE_GROUPS": str(int(config.get("maxrl_min_usable_groups", 1))),
            "RLVP_MAXRL_INITIAL_LR": "1e-6", "RLVP_MAXRL_TARGET_LR": maxrl_target_lr(config),
            "RLVP_MAXRL_WARMUP_UPDATES": str(int(config.get("maxrl_warmup_updates", 0))),
            "NEMO_POLAR_GROUP_REFIT_BARRIER": "1",
            "NEMO_POLAR_GROUP_REFIT_DRAIN_TIMEOUT": "2100",
        })
    if config.get("mode", "train") == "evaluate":
        # Eight responses per case; 64 means eight cases in flight (execution.case_concurrency 8).
        if int(config.get("concurrency", 8)) not in (8, 64):
            raise ValueError("evaluation uses eight or sixty-four native sequences")
        environment.update({"NRL_REFIT_ONLY_SERVE": "1",
                            "NRL_REFIT_ONLY_READY_FILE": str(run_dir / "logs/refit-only-ready.json"),
                            "NRL_REFIT_ONLY_STOP_FILE": str(run_dir / "stop-refit-only")})
    # Same-host scheduling must not inherit multi-host node pinning variables.
    for key in ("NEMO_POLAR_TRAIN_NODE_IP", "NEMO_POLAR_INFERENCE_NODE_IP", "NEMO_POLAR_COLLECTOR_NODE_IP"):
        environment.pop(key, None)
    os.environ.update(environment)
    sys.path[:0] = environment["PYTHONPATH"].split(":")
    processes, logs, proxy = [], [], None
    status = "failed"
    reason = None
    stop = threading.Event()
    violations = []

    def spawn(command, name, cwd=None):
        output = (run_dir / "logs" / (name + ".log")).open("w")
        logs.append(output)
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   cwd=cwd, env=environment, start_new_session=True)
        processes.append(process)
        return process

    def monitor():
        count = 0
        pressure_checks = 0
        while not stop.wait(10):
            row = {"elapsed_seconds": time.monotonic() - started}
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                                         "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                        check=False, timeout=10)
                if result.returncode or len(result.stdout.splitlines()) != 2:
                    raise RuntimeError("expected resource observations for both H200 GPUs")
                row["gpu"] = result.stdout.strip()
                meminfo = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                           for line in Path("/proc/meminfo").read_text().splitlines()}
                available = int(meminfo["MemAvailable"].split()[0]) * 1024
                row["host_memavailable_bytes"] = available
                pressure = []
                for line in result.stdout.splitlines():
                    index, used, capacity, _utilization = [float(value.strip()) for value in line.split(",")]
                    if used / capacity >= gpu_pressure_threshold:
                        pressure.append(f"GPU {int(index)} exceeded {gpu_pressure_threshold:.0%} capacity: {used}/{capacity} MiB")
                cgroup = Path("/sys/fs/cgroup")
                if (cgroup / "memory.max").is_file():
                    limit = (cgroup / "memory.max").read_text().strip()
                    if limit != "max":
                        used = int((cgroup / "memory.current").read_text())
                        row["host_cgroup_limit_bytes"] = int(limit)
                        row["host_cgroup_current_bytes"] = used
                        row["host_cgroup_available_bytes"] = int(limit) - used
                        available = min(available, int(limit) - used)
                row["effective_host_available_bytes"] = available
                if available < 16 * 1024**3:
                    pressure.append("GPU Function effective host memory headroom fell below 16 GiB")
                pressure_checks = pressure_checks + 1 if pressure else 0
                row["memory_pressure_checks"] = pressure_checks
                if pressure_checks >= 2:
                    violations.extend(pressure)
            except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                row["memory_monitor_error"] = str(error)
                violations.append("resource observations failed: " + str(error))
            if violations:
                row["stop_reasons"] = list(violations)
                for process in reversed(processes):
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
            with (run_dir / "receipts/resources.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            count += 1
            if count % 6 == 0 or violations:
                try:
                    volume.commit()
                except Exception as error:  # noqa: BLE001 - record SDK persistence failure
                    write_json(run_dir / "receipts/volume-commit-error.json", {"error": str(error)})
                    violations.append("periodic persistent-volume commit failed")
                    for process in reversed(processes):
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
            if violations:
                return

    try:
        write_json(run_dir / "run_config.json", config)
        copy_flashinfer_cache(PERSISTENT_FLASHINFER_CACHE, FLASHINFER_CACHE)
        write_json(run_dir / "receipts/verified-inputs.json", verify_inputs(deadline=deadline))
        patches = apply_historical_patches()
        write_json(run_dir / "receipts/historical-patches.json", patches)
        audit_patch = SCRIPTS / "patches/apply_audit_patches.py"
        if not audit_patch.is_file():
            raise RuntimeError("required loss/group/gradient audit patch is missing")
        subprocess.run([NEMO_PYTHON, str(audit_patch), "--nemo-root", "/opt/nemo-rl"], check=True,
                       timeout=max(1, deadline-time.monotonic()))
        from patch_transport import apply as apply_transport_patches

        write_json(run_dir / "receipts/transport-patches.json", apply_transport_patches())
        if config.get("advantage_normalizer") == "mean_plus_epsilon":
            from patch_maxrl_advantage import apply as apply_maxrl_advantage
            from patch_group_refit import apply as apply_group_refit
            from patch_maxrl_training import apply as apply_maxrl_training

            write_json(run_dir / "receipts/maxrl-advantage-patch.json",
                       apply_maxrl_advantage(Path("/opt/nemo-rl")))
            write_json(run_dir / "receipts/maxrl-group-refit-patch.json",
                       apply_group_refit(BRIDGE))
            write_json(run_dir / "receipts/maxrl-training-patch.json",
                       apply_maxrl_training(Path("/opt/nemo-rl")))
        proxy = start_proxy(verifier["url"], verifier["token"])
        with urllib.request.urlopen("http://127.0.0.1:18081/health", timeout=30) as response:
            health = json.load(response)
        write_json(run_dir / "receipts/verifier-health.json", health)
        rows = prepare_tasks(run_dir, config["manifest_path"], config, environment, deadline=deadline)
        if config.get("resume_checkpoint"):
            from rlvp.checkpoint_bootstrap import tree_sha256

            source = Path(config["resume_checkpoint"])
            environment["NRL_INITIAL_POLICY_WEIGHTS_PATH"] = str(source / "policy/weights")
            environment["NRL_INITIAL_POLICY_WEIGHTS_TREE_SHA256"] = tree_sha256(source / "policy/weights")
            write_json(run_dir / "receipts/resume-source.json", {
                "checkpoint": str(source), "training_info": json.loads((source / "training_info.json").read_text()),
                "weights_tree_sha256": environment["NRL_INITIAL_POLICY_WEIGHTS_TREE_SHA256"],
                "optimizer_present": (source / "policy/optimizer").exists(),
            })
            os.environ.update(environment)
        write_json(run_dir / "receipts/placement.json", {"same_host": True, "ray_gpus": 2,
                   "learner_gpus": 1, "inference_gpus": 1, "case_rows": rows})
        overrides = nemo_overrides(run_dir, config)
        write_json(run_dir / "nemo_overrides.json", overrides)
        ray_command = ["ray", "start", "--head", "--num-gpus=2", "--num-cpus=16",
                       "--object-store-memory=2147483648", "--disable-usage-stats", "--port=6379"]
        subprocess.run(ray_command, check=True, env=environment, timeout=min(120, max(1, deadline-time.monotonic())))
        environment["RAY_ADDRESS"] = "127.0.0.1:6379"
        spawn([NEMO_PYTHON, "-m", "polar.cli", "serve_rollout", "-c", str(run_dir / "polar/topology.yaml")], "polar-rollout")
        spawn([NEMO_PYTHON, "-m", "polar.cli", "serve_gateway", "-c", str(run_dir / "polar/topology.yaml"), "--node-id", "nemo-polar-gateway"], "polar-gateway")
        threading.Thread(target=monitor, daemon=True).start()
        trainer = spawn([NEMO_PYTHON, "-m", "nemo_polar_bridge.run_grpo_with_polar", "--config",
                         "examples/configs/grpo_math_1B_sglang.yaml", *overrides],
                        "grpo-nemo-polar", cwd="/opt/nemo-rl")
        if config.get("mode", "train") == "evaluate":
            run_evaluation(run_dir, config, health, trainer, started, wall_seconds, environment)
        trainer.wait(timeout=max(1, wall_seconds - (time.monotonic() - started)))
        if violations:
            raise RuntimeError("; ".join(violations))
        if trainer.returncode:
            raise RuntimeError(f"native trainer exited {trainer.returncode}")
        if config.get("mode", "train") == "train":
            audit_completed_training(run_dir, group_size, maxrl_start_step(config), config)
        status = "completed"
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
        subprocess.run(["ray", "stop", "--force"], env=environment, capture_output=True, timeout=60, check=False)
        if proxy:
            proxy.shutdown()
        for handle in logs:
            handle.close()
        try:
            copy_flashinfer_cache(FLASHINFER_CACHE, PERSISTENT_FLASHINFER_CACHE)
        except OSError as error:
            # Cache reuse is optional; preserve the experiment's original result.
            write_json(run_dir / "receipts/flashinfer-cache-error.json", {"error": str(error)})
        elapsed = time.monotonic() - started
        write_json(run_dir / "receipts/run-exit.json", {"status": status, "reason": reason,
            "elapsed_seconds": elapsed, "combined_estimated_cost": elapsed / 3600 * hourly_rate,
            "hard_modal_billing_cap": False, "same_host_gpu_function_timeout_seconds": wall_seconds})
        volume.commit()
    return {"run_id": run_id, "status": status, "run_dir": str(run_dir)}
