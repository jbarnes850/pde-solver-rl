# Modal RLVP loss and audit patch contract

Apply **after** copying the bridge tree's staged patches into the pinned NeMo tree.
`gpu_runtime.run` does both inside the GPU container:

```bash
/opt/nemo_rl_venv/bin/python /opt/rlvp-modal/patches/apply_audit_patches.py --nemo-root /opt/nemo-rl
```

The patch validates every source anchor and Python syntax before any write. It
emits `rlvp-modal-audit-patches.json` with before/after hashes. The staged bridge
source is never edited; `gpu_runtime.apply_historical_patches` copies its `grpo.py`
unconditionally.

## Required configuration

- `NRL_AUDIT_DIR`: persistent directory unique to the trial/attempt. An existing
  tensor receipt is never overwritten. A failed-step retry gets a new attempt
  directory while retaining its original reference path.
- `NRL_SFT_REFERENCE_PATH`: persistent campaign-specific adapter-only SFT reference
  file shared by trials and restores. Initial creation requires a fresh optimizer;
  optimizer resume without it fails. Every restore verifies its SHA-256.
- `NRL_ACTIVE_ONLY=0`, `NRL_GROUP_BY_TRAJECTORY=1`.
- `NRL_LOGPROB_CHUNK_TOKENS=64` (or a separately measured chunk size).
- `NRL_FULL_LENGTH_BACKWARD_PROBE=1` for initial qualification, `0` for later
  trials/restores. The probe uses 9,728 synthetic tokens, 9,216 loss tokens, real
  learner/reference forwards, and native backward with optimizer/MoE/scheduler
  updates suppressed. It never enters replay. It must retain unchanged weights
  and optimizer steps and remain below 85% CUDA reservation. This does not prove
  NVML/host headroom or real mixed-family learning; the real proof follows it.
- `NRL_FIXED_LOGPROB_PROBE_PATH`: shared fixed token-ID manifest, e.g.
  `/runs/fixed-logprob-probe.json`. A short learner forward at initialization and
  after each real update saves raw-temperature-1 probabilities. The FP8 endpoint
  uses the same IDs before/after refit. This diagnoses whether quantization or
  stale caches obscure small updates; it is not the temperature-0.7 behavior
  probability or a quality evaluation. RNG and training/sampling state are restored.
- The pinned NeMo constructor already sets `logprobs_mode="processed_logprobs"`.
  Do not also add it to `vllm_kwargs`: duplicate keyword arguments fail startup.
  Both learner and reference explicitly use temperature 0.7, so behavior
  probabilities must be taken after temperature processing. Top-p=1 and no top-k
  keep the full support.
- `grpo.use_leave_one_out_baseline=false`, no advantage clipping or message-level
  penalties beyond the accepted contract, no overlong filtering.

For the main CISPO candidate set `use_cispo=true`, `token_level_loss=true`,
`ratio_clip_min=1`, `ratio_clip_max=4`, `ratio_clip_c=null`,
`use_importance_sampling_correction=true`,
`grpo.normalize_rewards=true`, `++loss_fn.rlvp_fixed_response_tokens=0`.
Keep KL 0.001, `use_on_policy_kl_approximation=false`, no extra truncated IS,
`force_on_policy_ratio=false`, and sequence-level ratios disabled.

Its implemented objective is **token-pooled normalized-advantage CISPO with KL**:

`L_actor = -sum(mask * stopgrad(min(exp(logp_current-logp_behavior),5)) * A * logp_current) / sum(mask)`

`A = (reward - group_mean) / (sample_std + 1e-6)`.

The final recipe replaces this advantage. With `advantage_normalizer: mean_plus_epsilon`,
`patch_maxrl_advantage.py` installs `maxrl_advantage.py`, which computes
`A = (reward - group_mean) / (group_mean + epsilon)` in FP32 over complete groups of 16
(epsilon 1e-2 with the anchored reward).

Groups are case occurrences, not identical prompt strings. This deliberately
retains the established global token denominator across 64 sequences. MiniMax's
reported prompt-mean of group token means is a different weighting when groups
have different lengths; do not call this exact author reproduction. KL applies
to every valid row, including zero-advantage groups.

The original NeMo async CISPO clipped `current/previous_learner` and multiplied
an additional `previous_learner/behavior` factor. At a single update, the first
ratio is approximately one; the second could be arbitrarily large. This patch
clips the actual behavior ratio once and disables the second actor multiplier
for CISPO. Actor and KL scalar losses are separately logged.

For a Dr. GRPO diagnostic set `use_cispo=false`,
`grpo.normalize_rewards=false`, `++loss_fn.rlvp_fixed_response_tokens=9216`,
and group mean without LOO. The PPO actor and KL use denominator
`valid_sequence_count * 9216`; diagnostic token means retain their actual-token
denominator. Any async behavior correction is an explicit extension of Dr. GRPO.
The legacy unbounded correction is not recommended merely because this switch
exists; inspect first-batch coefficient diagnostics before allocating a trial.

## Numerical and artifact changes

- Statistics, rewards, and advantages stay FP32. Centered variance uses a scale
  before squaring to avoid avoidable underflow, retaining the same sample std and
  1e-6 epsilon. FP32 reward values already rounded to zero cannot be recovered.
- The existing custom chunked logprob backward now writes its completed FP32
  per-chunk derivative into an input-dtype gradient buffer. The full BF16 logits
  remain; this is not fused linear/CE. At 9,728 tokens and a 248,320 vocabulary,
  it removes about 4.83 GB of full-gradient allocation. CPU comparison is bit
  identical to original FP32 and BF16 gradients on representative tensors.
- Frozen-base storage dtype is reported; compute BF16 is not described as BF16
  parameter storage. Original base storage is preserved pending measured fit.
- Full batch archives contain IDs, masks, behavior/current/reference logprobs,
  actual consumed advantages, FP32 rewards, and trajectory group IDs.
- Every optimizer step preserves the actual accumulated pre-clip trainable
  gradient tensors, mapping and hashes, reported and measured clipping scales,
  before/after adapter hashes, and exact parameter-update L2 and relative L2.
- The original frozen SFT LoRA reference is restored after checkpoint resume;
  it is never silently replaced by the resumed learner's weights.

## Checks

```bash
RLVP_BRIDGE_DIR=/path/to/patched-bridge-tree <torch-python> -m pytest tests/modal/test_audit_patches.py -q
```

These CPU tests execute the actual pinned loss/custom-backward functions via AST
extraction. This is unit evidence, not live NeMo/CUDA qualification. The normal
RLVP virtualenv has no torch and skips this module.
