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
energy product, (BH)max in kJ/m³. The equations and parameters are in `rlvpm/physics.py`, the reference integrator
in `rlvpm/reference_etd.py` (exponential time differencing with a real-spectrum projection) and the closure in
`rlvpm/qoi.py`.

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
rlvpm/     simulator, structure-property closure, solver tasks and checks, model client, sandbox
scripts/   experiments, analyses and figures (one entry point per result, listed below)
data/      saved results the analyses read (data/MANIFEST.json has their sha256)
out/       written by the analyses (not tracked)
```

## Reproduce the reported numbers from saved results

CPU only, a few minutes (PyTorch is used on the CPU to evaluate the in-field onset). From the repository root:

```bash
uv sync --group dev
R="materials/data/onset-map/D0.5/results.json materials/data/onset-map/D1/results.json materials/data/onset-map/D2-fit/results.json materials/data/onset-map/D2-production/results.json"
uv run --with torch python materials/scripts/analyze_onset_study.py $R --geometry materials/data/onset-map/geometry.json --out materials/out/onset-analysis.json
for adj in 1.5 3.0; do uv run --with torch python materials/scripts/analyze_onset_study.py $R --geometry materials/data/onset-map/geometry.json --ramp-adjust $adj --out materials/out/onset-analysis-ramp$adj.json; done
uv run python materials/scripts/analyze_step_bias.py --map materials/data/onset-map/D0.5/results.json --uniform materials/data/onset-map/step-check-uniform/results.json
uv run python materials/scripts/analyze_step_qualification.py
uv run python materials/scripts/collect_post_data.py
uv run --python 3.12 --with matplotlib==3.10.7 python materials/scripts/post_figures.py
```

`analyze_onset_study.py` prints the decision table (best treatment per world, production utility of each
measurement policy with bootstrap intervals over production specimens). `analyze_step_bias.py` gives the fast ramp
+1.47 (SE 0.74) and the 929 K start −0.09 (SE 0.82). `collect_post_data.py` gathers every number in the section into
`out/post-data.json`, and `post_figures.py` draws the figures from it; with the pinned matplotlib and the fonts below
they match the published figures pixel for pixel. Put IBM Plex Sans TTF files in
`materials/fonts/` to match the post's type; otherwise DejaVu Sans is used.

## Rerun the experiments

The simulations use PyTorch on one GPU; the full nine-world map took about four hours of wall-clock on two A100s.
Add `--with torch --with scipy` to `uv run` for these.

| Result | Command |
|---|---|
| Decision map, one mobility batch | `scripts/onset_route_map.py --out runs/D1 --device cuda --kinetic-multiplier 1` (repeat for 0.5 and 2) |
| Step-bias check | `scripts/onset_route_map.py --out runs/uniform --device cuda --kinetic-multiplier 0.5 --h-max 0.25 --h-aging 0.25 --t-starts 929` |
| Colder starts (920, 925 K) | `scripts/onset_route_map.py --out runs/cold --device cuda --kinetic-multiplier 0.5 --t-starts 920 925 --no-ramp` |
| Chord-length geometry | `scripts/onset_geometry.py runs/D0.5 runs/D1 runs/D2 --out runs/geometry.json` |
| Animation frames | `scripts/onset_movie.py --out runs/movie --device cuda` |
| Step-size qualification | `scripts/probe_magnetic_decision_routes.py --out runs/qual-coarse --device cuda --h-max 0.25`, then `--h-max 0.0625` |
| Fourier bookkeeping defect | `scripts/verify_etd_reality.py --out runs/etd/cpu-controls` (CPU), plotted by `analyze_etd_reality.py --root runs/etd`; `rlvpm/reference_etd_frozen.py` is the integrator before the fix |
| Restart invariance | `scripts/probe_etd_restart_invariance.py runs/restart` |
| Thermal-noise response | `scripts/probe_stochastic_response.py runs/response` |

The D×2 batch in `data/` ran as two halves by specimen role (`--fit-seeds 701 702 --omitted-seeds 701
--production-seeds` and `--production-seeds 801 802 --fit-seeds --omitted-seeds`), each with its own thermal stream.

## The 9B model writing the magnetic term

Serve the base model and the adapter with vLLM (0.32 was used) on one GPU, thinking off:

```bash
vllm serve Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --port 8000 \
  --served-model-name qwen35-9b-base --enable-lora --max-lora-rank 32 \
  --lora-modules pde-solver=<path to Jarrodbarnes/qwen3.5-9b-pde-solver at aa5083ca21cf11715552e40b59af9a1a5ac541f5> \
  --language-model-only --dtype bfloat16 --max-model-len 32768 \
  --default-chat-template-kwargs '{"enable_thinking": false}'
export RLVPM_VLLM=http://127.0.0.1:8000
```

The matched test poses the task in the format the adapter was trained on (training system prompt, `def solver(...)`
interface, training extractor, 10,000-token cap). Generated code runs only in the Docker sandbox from
`containers/solver.Dockerfile` (image `rlvp-solver:py312`).

```bash
uv run python materials/scripts/magnetic_matched_ttc_batch.py init --root runs/ttc
for turn in 1 2 3 4 5 6 7 8; do
  uv run python materials/scripts/generate_batch.py --requests runs/ttc/requests-turn$turn.json --out runs/gen-turn$turn.jsonl
  uv run python materials/scripts/magnetic_matched_ttc_batch.py apply --root runs/ttc --generations runs/gen-turn$turn.jsonl --turn $turn
done
uv run python materials/scripts/analyze_matched_ttc.py --root runs/ttc --out runs/matched-ttc-compute.json
uv run python materials/scripts/audit_matched_magnetic_domain.py --root runs/ttc --specs materials/data/component-9b/audit-specs/*.json --out runs/audit
```

`data/component-9b/` holds the saved run: every prompt, completion, check result and selection
(`ttc/state-*.json`, `ttc/generations-turn*.jsonl`), the compute summary and the 42-case audit of the selected
programs. `mismatched-format-ttc-compute.json` is the earlier cohort that changed the prompt, interface and
extractor and capped completions at 4,096 tokens.

The complete-solver attempts (336, none passing) come from `scripts/gen_pilot.py` (two models × three guidance
levels × eight lineages × up to two repairs, 9,216 tokens) and `scripts/budget_factorial.py` (two models × two
guidance levels × 9,216 and 20,000 tokens); their summaries are in `data/whole-solver/`.
