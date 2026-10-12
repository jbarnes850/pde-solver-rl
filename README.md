# Training language models to write reliable physics simulations with reinforcement learning

Code for training Qwen3.5-9B with reinforcement learning against a physics verifier to write numerical solvers for
partial differential equations (PDEs).

- Research write-up: https://jbarnes850.github.io
- Trained adapter: https://huggingface.co/Jarrodbarnes/qwen3.5-9b-pde-solver

## Overview

The model reads a problem statement, such as "advance this initial condition under Burgers' equation and return
the solution at these times", and writes a Python function. A sandbox runs the function on a hidden case and
compares its output with a reference computed by two independent methods. An answer passes only if it runs,
returns the requested shape, and its root-mean-square error over the trajectory, divided by the range of the
reference, is at most 1%. The model never sees the reference, the error or the reward while it writes, and no
supervised fine-tuning is used.

After 256 updates, pass@1 on ten equation families held out from training rose from 9.0% to 55.7%, above GLM-5.3
(49.1%) and MiMo-V2.6-Pro (45.6%) and below Qwen3.8-Max (69.2%) under the same verifier. Two evaluation sets
generated and sealed before the final checkpoint existed confirmed the gains. With 64 samples per problem, most
of the improvement is increased reliability on problems the base model already solved occasionally, and five
held-out problems are solved only by the trained model.

This repository is an independent, source-derived implementation of the task in *Reinforcement Learning with
Verifiable Physics* (RLVP, arXiv 2607.10474). It does not use the authors' solver bank, verifier, generators,
cases or training code, which were not released. The advantage normalization follows *MaxRL* (arXiv
2602.02710). The Python package is named `rlvp` after the task it implements.

## Results

Equal-family strict pass@1, 8 responses per case, with the same cases and response seeds in every arm.
Differences are step 256 minus step 160 in percentage points, with a family-stratified 95% bootstrap interval
where the comparison script computes one. P-values come from 2,000,000 case-paired sign flips, and "< 1e-6"
means no flip reached the observed difference.

| Set | Base | Step 160 | Step 256 (final) | Step 256 minus step 160 | p |
|---|---:|---:|---:|---:|---:|
| Fresh-v4, six training families (288 cases, sealed) | | 37.3 | 46.7 | +9.4 [7.5, 11.3] | < 1e-6, one-sided |
| Held-out-v2, ten held-out families (136 cases, sealed) | | 41.6 | 55.3 | +13.6 [10.5, 16.8] | < 1e-6, one-sided |
| Held-out, ten held-out families (135 cases) | 9.0 | 39.8 | 55.7 | +15.9 | < 1e-6, two-sided |
| Fresh-v1, six training families (288 cases) | 6.6 | 36.2 | 47.3 | +11.0 [9.1, 12.9] | < 1e-6, one-sided |

The two sealed sets were generated and pre-registered before training passed step 160, and one trivially
solvable held-out-v2 case was excluded by the pre-registered rule. The pre-registered prediction that held-out-v2
at the final checkpoint lands within 3 points of the 135-case held-out set was met (55.3 against 55.7).

Pre-registered secondary analyses. Without Darcy, fresh-v4 rises from 44.4 to 55.9, and without Darcy and
Navier-Stokes from 55.4 to 69.9. Held-out-v2 without Darcy-reaction rises from 46.2 to 61.4.

Further results on the 135-case held-out set at step 256.

| Measure | Base | Step 256 |
|---|---:|---:|
| Pass@8 (8 samples per case) | 47.2 | 88.2 |
| Residual-selected best of 8 | 46.4 | 87.8 |
| Pass@8, estimated from 64 samples per case | 48.7 | 87.6 |
| Pass@64 | 85.2 | 90.0 |
| Problems solved only by this model in 64 samples | | 5 of 135 |
| Problems solved only by the base model in 64 samples | 0 of 135 | |

External models on the same set, scored by the same verifier with up to 32,000 output tokens and reasoning
enabled, reach pass@1 of 69.2 (Qwen3.8-Max), 49.1 (GLM-5.3) and 45.6 (MiMo-V2.6-Pro). Step 256 minus each,
case-paired and two-sided, is -13.5 (p < 1e-6), +6.6 (p = 0.009) and +10.1 (p = 0.005).

## Method

- **Policy.** Qwen3.5-9B with a LoRA adapter (rank 32, alpha 32). The base weights stay frozen. Generation uses
  up to 9,216 new tokens, temperature 0.7, top-p 1.0 and no thinking mode.
- **Task.** Each prompt names a PDE family and a required signature `def solver(...)` that returns a float64
  trajectory array. Training cases come from eight families (advection, Burgers, reaction-diffusion and
  diffusion-sorption in one dimension, and reaction-diffusion, Darcy, incompressible Navier-Stokes and shallow
  water in two). The phases after step 160 dropped Navier-Stokes, shallow water and diffusion-sorption, and the
  last phase added three linear families (`advection1d_domain`, `airy1d`, `hyperdiffusion1d`). Ten further
  families are held out for evaluation and never trained on.
- **Verifier.** A CPU sandbox (Docker, no network, read-only, 2 CPUs, 4 GiB, 30 s) runs each program on a hidden
  case and scores it against the reference. A response is a strict pass when the output is valid (runs, correct
  shape, finite) and its range-normalized RMSE is at most 1%.
- **Reward.** The anchored reward `R = validity * (0.5 * skill + 0.5 * 1[error <= 1%])`, where `skill` is a
  log-scaled improvement over the best trivial predictor, such as returning the initial state at every time.
- **Advantage and loss.** MaxRL advantage `A = (R - mean) / (mean + 0.01)` within each group of 16 responses, 8
  prompts per update, CISPO loss with importance-sampling correction for the FP8 rollout policy, and KL 1e-3 to
  the base model.
- **Learning rate.** One qualification update at 1e-6, then 1e-5 (`maxrl_warmup_updates` was 4 in the first
  phase and 0 afterwards).
- **Infrastructure.** NeMo RL asynchronous GRPO on Modal, with one H200 for the learner and one for vLLM rollout
  (H200:2), plus a CPU VM that hosts the verifier. A patched NeMo RL and Polar bridge connects the collector to
  the verifier (see "What is not included").
- **Schedule.** Eight phases of 32 updates, from the base model to step 256. Updates 1-96 passed through the
  original 512-case training order, and updates 97-128 reused those cases with zero-signal cases replaced. From
  step 128 on, each phase drew new cases, screened them with the phase's starting policy (8 samples each), kept
  cases that were neither solved nor dead, and filled per-family quotas.

## Repository layout

| Path | Contents |
|---|---|
| `src/rlvp/pdes/` | The 21 PDE families, with case generators, trusted reference solvers and prompt specifications, registered in `registry.py`. |
| `src/rlvp/solver_bank/` | Clean-room teacher solvers, used for the solvability ceilings (`pipeline/offline/family_ceiling.py`) and the verifier replay checks. |
| `src/rlvp/` (modules) | `manifests.py` (locked data contract and seeded case IDs), `data.py` (case stores and training rows), `prompts.py` (prompts and code extraction), `rewards.py` (validity, trajectory error, trivial predictors, anchored reward), `reference_gates.py` (two-route reference tolerances), `contracts.py`, `executor.py` and `runner/` (Docker sandbox), `verifier_service.py` (verifier HTTP API), `variance_probe.py` (the evaluation engine that samples, verifies and archives response groups), `evaluation.py` (pass@k and family metrics), `checkpoint_bootstrap.py` (checkpoint tree hashing and weight-only bootstrap). |
| `scripts/modal/` | Modal harness. GPU side, `gpu_app.py`, `gpu_image.py`, `gpu_runtime.py`, `trusted_runtime.py`, `modal_refit_probe.py`. Verifier side, `verifier_runtime.py` (controller), `verifier_vm.py` (VM entrypoint), `verifier_proxy.py` (loopback proxy on the GPU host), `replay_verifier.py` (parity replay). Training patches in `maxrl_*.py`, `patch_*.py` and `patches/` (see `patches/README.md`). Evaluation in `sealed_eval_gate.py` and `evaluate_campaign.py`. `trial.example.json` is a minimal trial. |
| `pipeline/` | Case generation, curriculum, launch, monitoring, evaluation and statistics (`offline/` and the evaluation folders), and the analyses behind the reported figures (`analysis/`). See `pipeline/README.md`. |
| `prompts/` | The system prompt used in training and evaluation, and the per-family grid conventions appended to problem statements. |
| `containers/solver.Dockerfile` | Solver sandbox image (Python 3.12, NumPy 2.5.2, SciPy 1.17.0). |

## Reproduction

Paths below are relative to the repository root. The campaign scripts take their locations from environment
variables, listed in `pipeline/README.md`.

### 1. Environment

```bash
uv sync                                         # library and verifier service (.venv)
uv venv .venv-modal --python 3.12
uv pip install --python .venv-modal/bin/python "modal==1.6.1"   # isolated Modal controller
export MODAL_PROFILE=<your-modal-profile>       # optional; Modal's active profile otherwise
export RLVP_BRIDGE_DIR=/path/to/patched-bridge-tree   # required by scripts/modal/gpu_app.py
```

`pipeline/offline/monitor_training.py` needs an interpreter with `torch` (CPU is enough). Docker is needed to run
the verifier service or replay programs locally. The Modal verifier VM builds its own solver image.

### 2. Cases

The locked source-derived data contract (512 training, 64 validation, 128 test, 143 seen evaluation and 137
held-out cases, plus 2,048 rows from the project's earlier supervised phase, kept so that new cases are checked
against them) is built by `rlvp.manifests.build_manifest_bundle()`. `write_manifest_bundle` writes the public
JSONL manifests, and `materialize_physical_manifests` generates the hidden cases into mode-600 private stores
under `.artifacts/private/`. Private stores must never be mounted into the policy runtime.

New cases for training pools and fresh test sets come from `pipeline/offline/generate_cases_counts.py`. It uses a
new seed role, rejects any case whose public input matches an existing case, and requires the two-route reference
gate.

### 3. Verifier VM

```bash
.venv-modal/bin/python scripts/modal/verifier_runtime.py check
.venv/bin/python scripts/modal/verifier_runtime.py prepare \
  --case-store .artifacts/private/<train>.cases.json --output .artifacts/modal/verifier-bundle
.venv-modal/bin/python scripts/modal/verifier_runtime.py launch \
  --bundle .artifacts/modal/verifier-bundle --state-dir .artifacts/modal/controller \
  --run-id <run-id> --timeout 28800
.venv/bin/python scripts/modal/replay_verifier.py run \
  --connection .artifacts/modal/controller/verifier-connection.json \
  --fixtures <replay-fixtures.json> --output .artifacts/modal/verifier-parity
```

`launch` starts a paid Modal resource. The VM runs `rlvp.verifier_service` with the same Docker executor as a
local run. Only the solver Dockerfile and the execution runner enter the inner Docker build context, and no case
store, reference code, credential or Docker socket enters a solver container, which has networking disabled.
Private cases are uploaded through the Sandbox file API and are never baked into an image or put on a GPU Volume.
Inbound HTTP goes through Modal Connect Tokens with no public port, and `launch` checks that an unauthenticated
`/health` request is rejected. On the GPU host, `verifier_proxy.py` holds the token and exposes only the health
and verify routes on loopback.

The connection file (mode 600) holds the sandbox ID, URL and bearer token and should be kept private. The VM
filesystem is not durable, and the sandbox terminates at its launch timeout, so call `export` periodically and
`stop` at the end, which exports and then terminates. `prepare` refuses sealed test stores, which go through
`scripts/modal/sealed_eval_gate.py` after a freeze. `prepare` also records the verifier source manifest, and
`launch` refuses a bundle whose recorded source differs from the current tree, so re-run `prepare` after any
change under `src/rlvp`.

`replay_verifier.py run` replays saved candidates (one valid program per family plus each observed failure class)
and labeled synthetic checks against a new VM before any sampling. Discrete fields must match exactly. Numerical
fields use rtol 1e-8 and atol 1e-12, with zero absolute tolerance on reward components so that tiny rewards cannot
silently underflow to zero. This checks port parity, not the scientific pass criterion.

### 4. GPU image and inputs Volume

`.venv-modal/bin/modal run scripts/modal/gpu_image.py` builds and checks the pinned NeMo RL image (CPU only). The
GPU app reads two Modal Volumes, named by `RLVP_INPUTS_VOLUME` and `RLVP_RUNS_VOLUME`. `gpu_runtime.verify_inputs`
defines what the inputs Volume must hold, namely the byte-pinned model tree at `/inputs/models/Qwen3.5-9B`, a
bootstrap checkpoint at `/inputs/checkpoints/sft128/step_0`, an `input-manifest.json` with every file hash, and
the public manifests and contracts under `/inputs/...`. Uploading these is not scripted in this repository.

The image starts from the NGC NeMo RL v0.7.0 amd64 image for its compiled dependencies and restores the source,
submodules and Ray actor environments to NeMo RL commit `c236061b`. `gpu_runtime.run` then applies the bridge
tree's staged patches, the audit and loss patches, the transport patches and, for MaxRL trials, the MaxRL
patches, and writes a receipt for each. One H200 runs vLLM and one runs the learner, both under one Ray head in
the same container. Rollouts go through `trusted_runtime.py`, which runs only the SHA256-frozen one-turn HTTP
harness. Generated programs are never executed on the GPU host.

Each invocation is bounded by the trial's `max_seconds` and `max_estimated_dollars` at an accounted rate of at
least $17.60 per hour for the GPU Function and the CPU verifier together. This is an estimate, not a Modal billing
cap. A watchdog samples memory every ten seconds and stops the run after two consecutive readings above the GPU
memory threshold (85% by default) or below 16 GiB of host headroom. A training run passes only if its final audit
finds gradient and batch archives, changed adapter weights, an optimizer checkpoint, successful refits and a
response generated after a refit (`gpu_runtime.audit_completed_training`).

### 5. Training phases

Each phase is 32 updates (16 responses for each of 8 prompts). The first phase starts from a zero-delta LoRA
adapter on the base model (`pipeline/trials/calib-train-base-p2-anchored.json`). Later phases resume the previous
phase's native checkpoint, including its optimizer state. For phase `P` that starts at step `S`, the steps are as
follows.

1. Generate candidates for the phase with `generate_cases_counts.py`, prepare and launch the phase's training
   verifier, and run the screen with
   `build_screen_phase.py P S <resume-checkpoint> <bundle-cases-sha256> <solver-digest>` followed by
   `run_eval_arm.sh`.
2. Build the curriculum with
   `build_curriculum_phase.py P S <screen groups.jsonl> <previous manifest> <previous parent schedule> [spec.json]`
   and upload the manifest and schedule to the inputs Volume.
3. Write the trial with
   `make_phase_trial.py <previous trial> P S <parent weights sha256> <bundle sha256> <solver digest>`.
4. Launch with `launch_training.sh` (or `launch_named.sh`) and apply the stop rules with `monitor_training.py` as
   updates complete.
5. At the end of the phase, measure the checkpoint with `finalize_and_eval.sh` on the gate sets, stop the training
   verifier with a terminal export, and screen candidates for the next phase. `phase_training_summary.py` reports
   per-family training signal and the Darcy gate.

### 6. Evaluation and statistics

Every evaluation fixes the checkpoint hash, the contract and the comparison script in a selection freeze before
the evaluated set is opened. Arms use the same cases and the same response seeds (8 responses per case). The final
comparison is step 160 against step 256 on two sets that were generated and pre-registered before training past
step 160 began, fresh-v4 (288 cases, six training families, 48 each) and held-out-v2 (137 cases from the ten
held-out families, 136 scored after one pre-registered exclusion). Neither set is opened until the final
checkpoint hash is frozen. Freezes and trials come from `pipeline/offline/prepare_final256.py`.

The primary statistic is equal-family strict pass@1 with a one-sided case-paired sign-flip test (exact for up to
24 nonzero cases, otherwise 2,000,000 Monte Carlo flips) and a family-stratified bootstrap interval. The comparison
scripts are in `pipeline/final-eval*/`, `pipeline/heldout-v2-eval/` and `pipeline/transfer-eval/`.
`pipeline/offline/final_report.py` builds the held-out comparison (by equation type, with and without
Darcy-reaction, residual-selected best of 8, case-paired tests), and `pipeline/offline/passk_large.py` the
64-sample pass@k analysis.

### Hardware and cost

One 32-update phase took about 3.2 hours on Modal H200:2. At the campaign's accounted rate that is about $55-60
of GPU time plus about $12-15 for the CPU verifier VM. Builds, storage, screening probes and evaluations are extra.
Each evaluation arm is bounded by its trial's `max_seconds` and `max_estimated_dollars`.

## Provenance

The verifier binds its source by a tree hash over `src/rlvp/**/*.py`, `containers/solver.Dockerfile` and
`scripts/modal/verifier_vm.py` (`verifier_runtime.source_manifest`). Training updates 1-224 were scored by tree
`d6a57690f156d0e6aea976145fda62366b29052cc08bd237d630f585f86890b0`, the tree before the three linear families were
added. Updates 225-256 were scored by tree `8b0a74899dcdf541b17a031e88b01b18e21f99df2995cb0b2856a59d0afdac7c`.
This release removes eleven modules from that tree that served the earlier two-host and supervised phases
(`cli.py`, `concurrent_resume.py`, `launch_plan.py`, `method_validation.py`, `nemo_matrix.py`, `nemo_sft_audit.py`,
`nemo_validation.py`, `sft.py`, `sft_replay.py`, `truncation_rerun.py`, `variance_audit.py`). The verifier service
imports none of them, and the other 53 files in the manifest are byte-identical to tree `8b0a7489...`. The release
tree hash is `6255029a030952e73e7aa667db0cc6f65dab2ad9b889f4649cedd98566cddd44`.

The comparison outputs cited above (`final-eval-v4/compare-calib-freshv4-s256b-vs-calib-freshv4-s160.json`,
`heldout-v2-eval/compare-calib-heldoutv2-s256b-vs-calib-heldoutv2-s160.json`, `final-report/leaderboard-final.json`,
`final-report/passk-large.json` and `final-eval/compare-calib-fresh-ext256-vs-calib-fresh-ext160.json`) live in the
campaign directory, which is not included.

## What is not included

- **Private case stores and sealed evaluation sets.** The hidden cases (`*.cases.json`), the verifier and sealed
  bundles, the fresh and held-out test sets with their freezes and pre-registration records, rollouts and logs are
  not included. Regenerate cases with `src/rlvp` and `pipeline/offline/generate_cases_counts.py`. New seed roles
  produce new cases, so a rerun will not match the campaign's numbers case for case.
- **The patched NeMo RL and Polar bridge.** `scripts/modal/gpu_app.py` packages a patched fork of NVIDIA's
  ProRL-Agent-Server (Polar, Apache-2.0), consisting of `src/nemo_polar_bridge` and the NeMo RL patches in
  `scripts/patch/`. It is not included. Point `RLVP_BRIDGE_DIR` at it.
- **Model weights and checkpoints.** The base model is `Qwen/Qwen3.5-9B` from Hugging Face, pinned by tree hash
  (`model_tree_sha256` in the trial templates). The trained adapter is at
  https://huggingface.co/Jarrodbarnes/qwen3.5-9b-pde-solver. The first phase starts from a zero-delta adapter
  checkpoint and a fresh base KL reference on the runs Volume, and creating the zero-delta checkpoint is not
  scripted here. `gpu_runtime.verify_inputs` also rehashes the project's earlier SFT step-128 tree at
  `/inputs/checkpoints/sft128/step_0`, which the final recipe does not train from. None of these checkpoints is
  included.
- **Earlier infrastructure.** The supervised fine-tuning path and the two-host launch code from the project's
  first phase are not part of the final recipe and are not included.

## Citation

```bibtex
@misc{barnes2026pdesolver,
  title  = {Training language models to write reliable physics simulations with reinforcement learning},
  author = {Barnes, Jarrod},
  year   = {2026},
  url    = {https://jbarnes850.github.io}
}
```

This work builds on Cai, P., Utkarsh, U., Edelman, A., Rackauckas, C. V. and Gómez-Bombarelli, R. Reinforcement
Learning with Verifiable Physics. Post-training LLMs with Continuous Rewards. arXiv 2607.10474, 2026.

## License

The code in this repository is released under the MIT License (see `LICENSE`). The trained adapter on Hugging Face
is released under Apache 2.0, following the base model. The scripts in `scripts/modal/` that patch NVIDIA NeMo RL
and Polar at run time produce modified copies of Apache-2.0 files inside the training container, and those
modified files remain under Apache 2.0.
