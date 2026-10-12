# Campaign pipeline

These are the scripts that ran the October 2026 calibration campaign: case generation, the
per-phase curriculum, training launches and monitoring, sealed evaluations, and the analyses
behind the reported numbers. They sit on top of the library in `src/rlvp` and the Modal harness
in `scripts/modal`.

## Layout and path conventions

The directory mirrors the campaign directory the scripts were written for. Python scripts find
the campaign directory as the parent of their own folder (`pipeline/`), and they write their
outputs there (`train-inputs/`, `trials/`, `probe-results/`, `logs/`, `training-monitor/`,
`external-eval/`). To run a campaign somewhere else, copy `pipeline/` to that location and
set `RLVP_CAMPAIGN_DIR` to it. The shell scripts honor `RLVP_CAMPAIGN_DIR` directly.

| Variable | Meaning | Default |
|---|---|---|
| `RLVP_ROOT` | repository root | two levels above the script |
| `RLVP_CAMPAIGN_DIR` | campaign directory (shell scripts) | `pipeline/` |
| `RLVP_PRIVATE_DIR` | private case stores | `$RLVP_ROOT/.artifacts/private` |
| `RLVP_SRC` | library source for case generation and curricula | `$RLVP_ROOT/src` |
| `RLVP_PYTHON` | project interpreter (numpy, scipy) | `$RLVP_ROOT/.venv/bin/python` |
| `RLVP_MODAL_PYTHON`, `MODAL_BIN` | Modal controller interpreter and CLI | `$RLVP_ROOT/.venv-modal/bin/...` |
| `RLVP_TORCH_PYTHON` | interpreter with torch, for `monitor_training.py` | `python3` |
| `MODAL_PROFILE` | Modal profile; passed through only when set | Modal's active profile |
| `RLVP_INPUTS_VOLUME`, `RLVP_RUNS_VOLUME` | Modal Volume names | `rlvp-modal-20261009-inputs`, `-runs` |
| `RLVP_REPLAY_FIXTURES` | verifier replay fixtures for training-verifier parity | `.artifacts/modal/2026-10-09/verifier-replay-fixtures.json` |
| `OPENROUTER_API_KEY` | key for `external_heldout_eval.py` only | none |

Paths that start with `/inputs/` or `/runs/` are paths inside the Modal Volumes, not local paths.

## Scripts by role

### Cases and curriculum

| File | Role |
|---|---|
| `offline/generate_cases_counts.py` | New cases for chosen families under a new seed role. Rejects any case whose public input matches an existing train, validation, test, evaluation or SFT case, and requires the two-route reference gate. Writes a mode-600 private store and a public manifest. Used for training pools, fresh-v3/v4 and held-out-v2. |
| `offline/build_screen_phase.py` | Screening probe for one phase: the phase's starting policy, 8 samples per candidate, screened families only (advection, Burgers, stiff RD1D, new linear families). Writes the screen manifest, config and trial. |
| `offline/build_curriculum_phase.py` | Builds the 256-row active window for one 32-update phase from the screen results (keeps 1/8 to 5/8 pass rates plus "alive zeros"), the per-family quotas, Darcy strata and prompt forms, and optional re-exposure. Writes the 512-row ordered manifest, parent schedule and seeded schedule. Without a spec file it reproduces phase A exactly. |
| `offline/make_phase_trial.py` | Writes the next phase's training trial from the previous one; only phase-specific fields change. |
| `prompts/user-grid-lines-p2.json`, `prompts/user-grid-lines-p2-extra.json` | Per-family grid sentences appended to training prompts (P2). The extra file covers the three linear families added at step 224. |
| `probe-inputs/configs/promptdev-p2.yaml` | Evaluation contract used as the template for screening probes. |

### Launch, monitoring and phase boundaries

| File | Role |
|---|---|
| `offline/launch_training.sh` | Runs one bounded trial through `scripts/modal/gpu_app.py` and records start and end in `logs/launch-ledger.jsonl`. |
| `offline/launch_named.sh` | Same, under an explicit Modal app name, so parallel runs never share one. |
| `offline/monitor_training.py` | Parent-side per-update monitor. Pulls gradient, advantage and batch receipts from the runs Volume and applies the pre-declared stop rules (nonfinite statistics, importance-ratio incidence, clipped ESS, reference K3, early support, reward collapse). |
| `offline/phase_training_summary.py` | Per-family training signal for a finished phase from the training verifier log, including the Darcy gate G5 that sets the next phase's Darcy quota. |
| `offline/finalize_and_eval.sh` | After a run: hashes the final checkpoint from two downloads, writes a selection freeze before any outcome exists, uploads it, writes the evaluation trial and launches it. |
| `offline/run_eval_arm.sh` | Launches one evaluation trial and exports its arm. |
| `offline/export_probe_arm.sh` | Downloads one evaluation arm from the runs Volume into `probe-results/` and checks its internal hashes. Called by the two scripts above. |
| `final-eval/sealed_parity_probe.py`, `transfer-eval/heldout_parity_probe.py` | Verifier parity on sealed sets using input-only programs (no policy outputs). Run after each verifier launch. |
| `offline/prepare_final256.py` | Writes the freezes and trials for the final comparison (step 160 against step 256) on the sealed fresh-v4 and held-out-v2 sets. |

### Evaluation contracts, templates and statistics

| File | Role |
|---|---|
| `final-eval/fresh-test-p2.yaml` | Contract for the fresh test (P2 prompt, 8 responses, 9,216 new tokens, temperature 0.7). |
| `final-eval-v4/fresh-test-v4-p2.yaml`, `heldout-v2-eval/heldout-v2-p1.yaml` | Contracts for the two sealed final sets. They differ from the fresh and held-out contracts only in id, manifest role and seed namespace. |
| `transfer-eval/heldout-p1.yaml` | Contract for the 10 held-out families (no grid sentence in the user prompt). |
| `transfer-eval/heldout-p1-k64.yaml` | The same contract with 64 responses per case, for the large-k pass@k analysis (`offline/passk_large.py`). |
| `final-eval/selection-freeze.json`, `transfer-eval/selection-freeze.json` | The campaign's freeze records. `finalize_and_eval.sh` and `prepare_final256.py` use them as templates. Their hashes refer to the campaign's private sets, so a new campaign must write its own. |
| `trials/*.json` | Trial templates: the first training phase (`calib-train-base-p2-anchored`, steps 0-32), the final phase (`-ext256`, steps 224-256), the screening probe, and the fresh, fresh-v3 and held-out evaluation templates. They hold the full recipe (reward, MaxRL epsilon, LR schedule, G16, CISPO settings, system prompt). |
| `final-eval/compare_arms.py` | Fresh-test comparison: six-family equal-weight strict pass@1, one-sided case-paired sign-flip test, family-stratified bootstrap. |
| `final-eval-v4/compare_v4.py` | Same statistics for fresh-v4 with explicit equal-family weights. Byte-identical to the pre-registered script (sha256 `60a14ad7a6a368684b4577821f48efc5014ba155597639fbb7422fa4f9d83922`). |
| `heldout-v2-eval/compare_heldout_v2.py` | Held-out-v2 comparison. Byte-identical to the pre-registered script (sha256 `7cc843ad4c9c39da7258392f6d8e8838ec799600376e3b43392b016b4490ebe0`). |
| `transfer-eval/compare_transfer_arms.py` | Comparison on the original held-out families (135 cases after excluding two trivially solved Gray-Scott cases). |

### Analyses

| File | Role |
|---|---|
| `offline/behavior_analysis.py` | Outcome mix (strict, valid but inaccurate, invalid, truncated), error, length and method markers across checkpoints. |
| `offline/family_ceiling.py` | How much of each family the solver-bank methods solve at the 1% bar (reads the private training store). |
| `offline/replay_with_tracebacks.py` | Local replay of saved programs in the solver container with stderr capture. Diagnosis, not scoring. |
| `offline/exit_taxonomy.py` | Buckets `solver_exit_1` failures from the replay into mechanistic classes. |
| `offline/reward_counterfactual.py` | Where the original full reward would have put positive MaxRL mass on the same rollouts, compared with the anchored reward. |
| `offline/external_heldout_eval.py` | Runs an external model through OpenRouter on the held-out cases and scores it with the same verifier. Reads `OPENROUTER_API_KEY`. |

## Inputs that are not included

The scripts read campaign data that is not part of this release:

- private case stores under `$RLVP_PRIVATE_DIR` (`rlvp-source-derived-v1/manifests/*.cases.json`,
  `rlvp-calibration-20261010/*.cases.json`) and the sealed bundles
  (`final-eval/sealed-bundle/`, `transfer-eval/sealed-bundle/`, `verifier-bundle*/`);
- per-phase candidate pools (`train-inputs/new{PHASE}/new-train-candidates.jsonl`) and
  training-verifier logs;
- trivial-predictor error tables (`offline/heldout-trivial-errors.json`,
  `offline/heldout-v2-trivial.json`), computed from the private held-out stores with
  `rlvp.rewards.best_trivial_error`;
- evaluation outputs (`probe-results/`, `external-eval/`), plan files (`scale288-plan.json`,
  `night-plan-20261011.json`) and the pre-registration records for fresh-v4 and held-out-v2;
- the verifier replay fixtures, built by `scripts/modal/replay_verifier.py prepare`;
- the earlier campaign scripts that some docstrings mention (`generate_fresh_cases.py`,
  `build_screen160.py`, `build_curriculum128.py`, `build_curriculum160.py`,
  `transform_analysis.py`). The curricula for updates 97-160 came from those builders;
  `build_curriculum_phase.py` covers the phases from step 160.

Regenerate the case stores with `src/rlvp` and `offline/generate_cases_counts.py`. New seed
roles produce new cases, so numbers from a regenerated campaign will not match the campaign's
numbers exactly.

### Final comparison and figure analyses

| File | Role |
|---|---|
| `offline/final_report.py` | Held-out leaderboard for our checkpoints against the frozen frontier responses: equal-family first-try accuracy, by equation type, with and without Darcy-reaction, best of 8 selected by PDE residual, pass@k, and case-paired sign-flip tests. |
| `offline/passk_large.py` | Pass@k for k up to 64 from 64-sample runs of the untrained and final models, and which problems each solves. |
| `offline/burgers_ceiling.py`, `offline/family_ceiling.py` | Solvability ceilings: independent solvers scored against the reference at the 1% bar. |
| `analysis/` | Behavior and training-curve analyses behind the reported figures (`extract.py` first, then `behavior.py`, `solvable.py`, `training_curve.py`, `content_stats.py`, `screens.py`, `assemble.py`), and `build_figure_data.py`, which assembles every figure's data into one JSON file. `linear_bug_counts.py` (with the static detectors in `bugs.py`) counts recurring spectral-program bugs on the linear held-out families. `rollouts_to_records.py` rebuilds a phase's training attempts from the run's own rollout records when the phase's verifier log is unavailable (used for updates 225-256). |
