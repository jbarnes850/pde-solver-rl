# Materials decision study

Code and saved results for the section *From a solver to a processing decision* of the write-up
([post](https://jbarnes850.github.io/2026/10/12/post-training-language-models-for-scientific-simulation/)).
The study asks whether a verified simulation can choose a field-annealing schedule for a synthetic Fe–Cr–Co
permanent magnet, find which uncertain alloy properties would change that choice, and decide which measurement,
if any, is worth making, and where the post-trained 9B model helps in that loop.

Everything here is a labeled synthetic example. The model below also plays the part of the real alloy, so every
number is synthetic and none is a prediction for a real magnet.

## Model

A two-dimensional stochastic Cahn–Hilliard equation for a pseudo-binary composition field `c` (Cahn and Hilliard
1958, Cook 1970), with a regular-solution free energy, Arrhenius mobility, a gradient energy, the long-range
magnetostatic energy of the alloy, the Zeeman energy of a 1 T applied field and a magnetization that vanishes at a
composition-dependent Curie temperature. A fixed structure–property closure turns the final field into a synthetic
energy product, (BH)max in kJ/m³. The equations and parameters are in `magnet/physics.py`, the reference integrator
in `magnet/reference_etd.py` (exponential time differencing with a real-spectrum projection) and the closure in
`magnet/properties.py`.

| Parameter | Nominal | Basis |
|---|---|---|
| Coherent spinodal at the alloy composition | 933 K | miscibility gap near 650–700 °C; field anneals at 630–675 °C ([Zhang et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6116236/)) |
| Composition | 0.50 | near-critical commercial design |
| Interdiffusion at 913 K | 3e-18 m²/s, Q = 250 kJ/mol | Cr in α-Fe magnitudes |
| Gradient energy | 1.14e-10 J/m | Fe–Cr phase-field literature value divided by molar volume |
| Thermal noise | Cook noise with a 20 nm coarse-graining thickness | fluctuation–dissipation |

The engineer controls four hours in the field, either cooling steadily from 943 to 903 K (`ramp10Kph`) or holding at
a start temperature T for an hour and cooling at 3 K/h for three hours (`sweepT`, T from 929 to 947 K), followed by a
fixed aging sequence. Two properties are uncertain: the Curie-temperature offset (−15, −5, +5 K) and the mobility
multiplier (×0.5, ×1, ×2), nine worlds in all. Each world is simulated on four specimens that differ only in their
random starting fluctuations: two fit specimens (701, 702) that a simulator consults to choose and two production
specimens (801, 802) that score the choice. A fifth run per world (specimen 701) uses the base model's first-attempt
magnetic term, which drops a derivative factor from the field term, as a faulty simulator.

Rules fixed before outcomes were read: choices use fit specimens only; the fast ramp is analyzed with its utility
corrected by 0, +1.5 (the measured step bias, below) and +3, and only conclusions that hold in all three are
reported; generated code never sees reference trajectories or outcomes.

## Layout

```
magnet/    the model and its tools (Python package)
scripts/   one entry point per result, named by stage: simulate_, check_, run_, analyze_, collect_, figures
data/      saved results the analyses read; data/SHA256SUMS has their hashes
out/       written by the analyses (not tracked)
```

| `magnet/` | Contents |
|---|---|
| `physics.py` | equations, parameters and processing schedules |
| `reference.py` | pseudo-spectral reference simulator: grid, coefficients, thermal noise |
| `reference_etd.py` | exponential time-differencing integrator with the real-spectrum projection, used for every reported run |
| `reference_etd_before_fix.py` | the same integrator before that fix, kept byte for byte so `check_fourier_defect.py` can show the failure |
| `properties.py` | microstructure descriptors and the synthetic energy product |
| `chords.py` | directional chord lengths, a geometry measure independent of the spectral descriptors |
| `routes.py` | the fixed aging sequence, plus route families and a parameter prior from earlier runs |
| `magnetic_task.py` | the magnetic-term task: energy, cases and variational check |
| `magnetic_task_matched.py` | the same task posed in the adapter's training format |
| `training_system_prompt.txt` | the system prompt the adapter was trained with |
| `solver_task.py`, `solver_runner.py` | the whole-solver task with its analytic checks, and its generate–check–repair loop |
| `llm.py`, `sandbox.py` | client for a local vLLM server, and the Docker sandbox that runs generated code |

| `data/` | Contents | Written by |
|---|---|---|
| `decision_map/mobility_0.5.json`, `mobility_1.json`, `mobility_2_fit.json`, `mobility_2_production.json` | the nine-world map, one file per mobility (×2 ran in two halves) | `simulate_decision_map.py` |
| `decision_map/mobility_0.5_step_0.25.json` | the ×0.5 map rerun at a uniform step of 0.25, for the step bias | `simulate_decision_map.py` |
| `decision_map/cold_starts_fit.json`, `cold_starts_production.json` | starts at 920 and 925 K | `simulate_decision_map.py` |
| `decision_map/geometry.json` | chord geometry of every final field | `analyze_geometry.py` |
| `decision_map/window_fields.npz` | the two final fields in Figure 3c | taken from the ×1 run |
| `step_size/h_0.25.json`, `h_0.0625.json` | the step-size check at two step caps | `simulate_step_size.py` |
| `fourier_defect.json` | the Fourier bookkeeping controls | `check_fourier_defect.py` |
| `noise_response/seed_41.json`, `seed_42.json` | thermal-noise response | `simulate_noise_response.py` |
| `magnetic_term/summary.json` | the 9B rerun by model, policy and call budget | `analyze_magnetic_term.py` |
| `magnetic_term/attempts.json` | one row per model call: model, chain, turn, outcome, tokens | `analyze_magnetic_term.py` |
| `magnetic_term/selected_programs.json` | the 18 distinct selected programs and which chains chose them | `analyze_magnetic_term.py` |
| `magnetic_term/audit.json`, `audit_specs/` | domain audit of the selected programs, and its fixtures | `audit_magnetic_term.py` |
| `magnetic_term/earlier_format_summary.json` | the earlier cohort, before the format was matched to training | |
| `whole_solver/pilot.json`, `budget.json` | the whole-solver attempts | `run_whole_solver.py`, `run_whole_solver_budget.py` |

## Reproduce the reported numbers from saved results

CPU only, a few minutes (PyTorch is used on the CPU to evaluate the in-field onset). From the repository root:

```bash
uv sync --group dev
M=materials/data/decision_map
R="$M/mobility_0.5.json $M/mobility_1.json $M/mobility_2_fit.json $M/mobility_2_production.json"
uv run --with torch python materials/scripts/analyze_decision.py $R --geometry $M/geometry.json --out materials/out/decision.json
for adj in 1.5 3.0; do uv run --with torch python materials/scripts/analyze_decision.py $R --geometry $M/geometry.json --ramp-adjust $adj --out materials/out/decision_ramp$adj.json; done
uv run python materials/scripts/analyze_step_bias.py --map $M/mobility_0.5.json --uniform $M/mobility_0.5_step_0.25.json
uv run python materials/scripts/analyze_step_size.py
uv run python materials/scripts/collect_numbers.py
uv run --python 3.12 --with matplotlib==3.10.7 python materials/scripts/figures.py
```

`analyze_decision.py` prints the decision table (best treatment per world, production utility of each measurement
policy with bootstrap intervals over production specimens). `analyze_step_bias.py` gives the fast ramp +1.47 (SE
0.74) and the 929 K start −0.09 (SE 0.82). `collect_numbers.py` gathers every number in the section into
`out/numbers.json`, and `figures.py` draws Figures 1 to 3 from it into `out/figures/`; with the pinned matplotlib and
the fonts below they match the published figures pixel for pixel. Put IBM Plex Sans TTF files in `materials/fonts/`
to match the post's type; otherwise DejaVu Sans is used. `cd materials/data && sha256sum -c SHA256SUMS` checks the
saved files.

## Rerun the experiments

The simulations use PyTorch on one GPU; the full nine-world map took about four hours of wall-clock on two A100s.
Add `--with torch --with scipy` to `uv run` for these. Each simulation writes `results.json` into its output folder;
the noise response writes one `seed_*.json` per seed.

| Result | Command | Saved as |
|---|---|---|
| Decision map, one mobility | `scripts/simulate_decision_map.py --out runs/mobility_1 --device cuda --kinetic-multiplier 1` (repeat for 0.5 and 2) | `decision_map/mobility_*.json` |
| Step bias | `scripts/simulate_decision_map.py --out runs/step_0.25 --device cuda --kinetic-multiplier 0.5 --h-max 0.25 --h-aging 0.25 --t-starts 929` | `decision_map/mobility_0.5_step_0.25.json` |
| Colder starts | `scripts/simulate_decision_map.py --out runs/cold_starts --device cuda --kinetic-multiplier 0.5 --t-starts 920 925 --no-ramp` | `decision_map/cold_starts_*.json` |
| Chord geometry | `scripts/analyze_geometry.py runs/mobility_0.5 runs/mobility_1 runs/mobility_2 --out runs/geometry.json` | `decision_map/geometry.json` |
| Animation frames | `scripts/simulate_animation.py --out runs/animation --device cuda` | |
| Step size | `scripts/simulate_step_size.py --out runs/h_0.25 --device cuda --h-max 0.25`, then `--out runs/h_0.0625 --h-max 0.0625` | `step_size/h_*.json` |
| Fourier bookkeeping defect | `scripts/check_fourier_defect.py --out runs/fourier_defect` (CPU) | `fourier_defect.json` |
| Restart invariance | `scripts/check_restart.py runs/restart` | |
| Thermal-noise response | `scripts/simulate_noise_response.py runs/noise_response` | `noise_response/seed_*.json` |

The ×2 map ran as two halves by specimen role (`--fit-seeds 701 702 --omitted-seeds 701 --production-seeds` and
`--production-seeds 801 802 --fit-seeds --omitted-seeds`), each with its own thermal stream.

## The 9B model writing the magnetic term

Serve the base model and the adapter with vLLM (0.32 was used) on one GPU, thinking off. `magnet/llm.py` reads the
server from `VLLM_URL` (default `http://127.0.0.1:8000`).

```bash
vllm serve Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --port 8000 \
  --served-model-name qwen35-9b-base --enable-lora --max-lora-rank 32 \
  --lora-modules pde-solver=<path to Jarrodbarnes/qwen3.5-9b-pde-solver at aa5083ca21cf11715552e40b59af9a1a5ac541f5> \
  --language-model-only --dtype bfloat16 --max-model-len 32768 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
```

The rerun poses the task in the format the adapter was trained on (training system prompt, `def solver(...)`
interface, training extractor, 10,000-token cap). Generated code runs only in the Docker sandbox built from
`containers/solver.Dockerfile` (`docker build -f containers/solver.Dockerfile -t rlvp-solver:py312 .`; `SOLVER_IMAGE`
names another image).

```bash
uv run python materials/scripts/run_magnetic_term.py init --root runs/magnetic_term
for turn in 1 2 3 4 5 6 7 8; do
  uv run python materials/scripts/generate.py --requests runs/magnetic_term/requests-turn$turn.json --out runs/generations-turn$turn.jsonl
  uv run python materials/scripts/run_magnetic_term.py apply --root runs/magnetic_term --generations runs/generations-turn$turn.jsonl --turn $turn
done
uv run python materials/scripts/analyze_magnetic_term.py --root runs/magnetic_term --out runs/tables/summary.json
uv run python materials/scripts/audit_magnetic_term.py --programs runs/tables/selected_programs.json \
  --specs materials/data/magnetic_term/audit_specs/*.json --out runs/audit
```

`analyze_magnetic_term.py` writes the three tables saved in `data/magnetic_term/`. The audit can also be rerun on the
saved programs alone, with `--programs materials/data/magnetic_term/selected_programs.json`.

The whole-solver attempts (336, none passing) come from `scripts/run_whole_solver.py` (two models × three guidance
levels × eight lineages × up to two repairs, 9,216 tokens) and `scripts/run_whole_solver_budget.py` (two models × two
guidance levels × 9,216 and 20,000 tokens). `scripts/analyze_whole_solver.py` summarizes a run's candidates; the
summaries are in `data/whole_solver/`.
