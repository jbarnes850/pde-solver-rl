"""Common-noise finite-difference response near the nominal spinodal.

This diagnoses whether noise changes weak-mode amplification. It does not estimate
a calibrated material spinodal or tune the potential to achieve the morphology target.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnet.physics import MaterialParams, Schedule, fsecond, isothermal
from magnet.reference import Coefficients, Grid, band_limited_noise, simulate, simulate_physical

torch.cuda.set_per_process_memory_fraction(.05)
torch.set_num_threads(1)
out = Path(sys.argv[1])  # output folder
base = MaterialParams(c0=.5, T_s=933, B0=4e6)

# Contract preflight: common noise makes identical members exactly identical;
# independent groups differ; both conserve their initialized spatial means.
small = Grid(32, 32, 64, 64)
route = Schedule((isothermal(943, .001, True),), "common-noise-preflight")
initial = np.full((3, 32, 32), .5)
test = simulate_physical(initial, small, [(base, route)] * 3, dtype=torch.float64,
                         device="cuda", noise_groups=[0, 0, 1], noise_seed=20261023)
a = test["final"]
receipt = {"identical_group_maxdiff": float(np.max(abs(a[0] - a[1]))),
           "independent_group_maxdiff": float(np.max(abs(a[0] - a[2]))),
           "mass_drift": float(np.max(abs(a.mean(axis=(1, 2)) - .5)))}
assert receipt["identical_group_maxdiff"] == 0
assert receipt["independent_group_maxdiff"] > 0
assert receipt["mass_drift"] < 1e-12
(out / "common-noise-preflight.json").write_text(json.dumps(receipt, indent=2) + "\n")

# Independent analytic control for the +/- response estimator itself.
cg = Grid(64, 64, 96, 96)
xx = np.arange(64)[:, None]
yy = np.arange(64)[None, :]
delta_control = 1e-7
mode_control = 2
shape = np.cos(2*np.pi*mode_control*xx/64) + np.cos(2*np.pi*mode_control*yy/64)
co = Coefficients(np.array([500.]), np.array([.98]), np.array([.98]),
                  np.array([0.]), np.array([0.]), np.array([.003]))
rr = simulate(np.stack([.5+delta_control*shape, .5-delta_control*shape]), cg,
              [co, co], device="cuda", dtype=torch.float64, h_max=1,
              noise_groups=[0, 0])
spec = np.fft.fft2(.5*(rr["final"][0]-rr["final"][1]))/(64*64)
k_control = 2*np.pi*mode_control/96
control = {}
for axis, observed, penalty in (("x", spec[2, 0], .003), ("y", spec[0, 2], 0)):
    predicted = -k_control**2*(fsecond(.5, .98)+k_control**2+penalty)*500
    measured = float(np.log(2*observed.real/delta_control))
    control[axis] = {"predicted_log_gain": float(predicted), "measured_log_gain": measured}
    assert abs(predicted-measured) < .005
(out / "response-estimator-control.json").write_text(json.dumps(control, indent=2) + "\n")

grid = Grid(384, 384, 768, 768)
mode, delta = 7, 1e-4
x = np.arange(grid.nx)[:, None]
y = np.arange(grid.ny)[None, :]
perturbation = delta * (np.cos(2 * np.pi * mode * x / grid.nx)
                        + np.cos(2 * np.pi * mode * y / grid.ny))
times = [.05, .1, .25, .5, 1, 1.5, 2]
settings = [(depth, dz) for depth in (0, 2, 5, 8, 12) for dz in (5, 20, 50)]
plan = {"scope": "development response diagnostic; not physical validation",
        "undercooling_K_and_thickness_nm": settings, "initial_seeds": [41, 42],
        "mode": mode, "perturbation_amplitude": delta, "times_hours": times,
        "grid": vars(grid), "params": base.to_dict(), "h_max": 4,
        "common_noise": "same Brownian draws for +/- perturbations, independent across parameter pairs",
        "limits": "finite-difference tangent only while perturbed paths stay close; two replicates"}
(out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
for seed in plan["initial_seeds"]:
    initial = base.c0 + band_limited_noise(grid, base.noise_rms, seed=seed, k_cut=.8)
    members, inits, groups = [], [], []
    for group, (depth, dz) in enumerate(settings):
        p = base.with_(d_z_nm=dz)
        route = Schedule((isothermal(933 - depth, 2, True),), f"depth{depth}-dz{dz}")
        for sign in (1, -1):
            members.append((p, route))
            inits.append(initial + sign * perturbation)
            groups.append(group)
    started = time.monotonic()

    def progress(d):
        row = {**d, "wall_s": time.monotonic() - started}
        (out / f"progress-{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
        print(seed, json.dumps(row), flush=True)

    r = simulate_physical(np.stack(inits), grid, members, out_hours=times,
                          dtype=torch.float64, device="cuda", h_max=4, dc_tol=.01,
                          noise_seed=20261023 + seed, noise_groups=groups,
                          progress_callback=progress, progress_every=10000)
    rows = []
    k = 2 * np.pi * mode / grid.Lx
    for hour, fields in r["snapshots"].items():
        for group, (depth, dz) in enumerate(settings):
            cp, cm = fields[2 * group:2 * group + 2]
            response = .5 * (cp - cm)
            spec = np.fft.fft2(response) / (grid.nx * grid.ny)
            p = members[2 * group][0]
            temperature = 933 - depth
            curvature = float(fsecond(.5, p.theta(temperature)))
            rate = p.m(temperature) / p.tau0
            row = {"seed": seed, "hours": hour, "undercooling_K": depth,
                   "d_z_nm": dz, "response_rms": float(np.sqrt(np.mean(response**2))),
                   "mean_field_variance": float((.5 * (cp + cm)).var()),
                   "mean_local_curvature": float(fsecond(.5 * (cp + cm), p.theta(temperature)).mean())}
            for axis, coefficient, penalty in (("x", spec[mode, 0], p.b(temperature, True)),
                                                ("y", spec[0, mode], 0)):
                row[f"gain_{axis}"] = float(2 * coefficient.real / delta)
                row[f"gain_imag_{axis}"] = float(2 * coefficient.imag / delta)
                row[f"bare_log_gain_{axis}"] = float(-rate * k*k * (curvature + k*k + penalty) * hour * 3600)
            rows.append(row)
    (out / f"seed_{seed}.json").write_text(json.dumps({"rows": rows, "steps": r["steps"],
        "wall_s": time.monotonic() - started, "rejected": r["rejected"]}, indent=2) + "\n")
    print("Completed response", seed, flush=True)
