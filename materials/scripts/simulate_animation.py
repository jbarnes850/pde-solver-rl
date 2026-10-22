"""Frequent snapshots of two onset-study routes for the post's animation.

One specimen of the nominal world (Curie offset -5 K, mobility x1), the same initial
field as production specimen 801, routes sweep929 and sweep941, full magnetic law,
the onset study's two-stage step cap. Fields are saved downsampled to 192 x 192 in
float16 every 300 s during the field stage and every 1200 s during aging. The thermal
noise differs from the production run, so the final fields are a fresh realization.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_decision_map import KINETICS, OFFSETS, routes, world_params  # noqa: E402
from magnet.reference import Grid, band_limited_noise, schedule_coefficients  # noqa: E402
from magnet.reference_etd import simulate_etd  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--device', default='cuda')
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
if args.device == 'cuda':
    torch.cuda.set_per_process_memory_fraction(.2)
grid = Grid(384, 384, 768., 768.)
offset, kinetic, seed = -5.0, 1.0, 801
p = world_params(offset, kinetic)
picked = [r for r in routes() if r.label in ('sweep929', 'sweep941')]
world_id = OFFSETS.index(offset)*3+KINETICS.index(kinetic)
specimen_seed = 2026220000+1000*world_id+seed
field = p.c0+band_limited_noise(grid, p.noise_rms, seed=specimen_seed, k_cut=.8)
coeffs = [schedule_coefficients(p, r) for r in picked]
end = max(c.end for c in coeffs)
times = sorted(set(np.arange(300., 14400.+1, 300.).tolist()+np.arange(15600., end, 1200.).tolist()))
frames, stamps = [], []


def snapshot(t, fields):
    frames.append(fields.reshape(len(picked), 192, 2, 192, 2).mean(axis=(2, 4)).astype(np.float16))
    stamps.append(t)


started = time.monotonic()
result = simulate_etd(np.stack([field, field]), grid, coeffs, out_times=times, h_max=1., h_max_after=.25,
                      h_switch_time=14400., device=args.device, dtype=torch.float64, noise_seed=2026250000,
                      noise_groups=[specimen_seed, specimen_seed], max_steps=8_000_000, cache_linear=True,
                      enforce_real_spectrum=True, snapshot_callback=snapshot)
snapshot(end, result['final'])
first = field.reshape(192, 2, 192, 2).mean(axis=(1, 3)).astype(np.float16)
np.savez_compressed(args.out/'movie.npz', frames=np.stack(frames), t=np.array(stamps), initial=first)
(args.out/'movie.json').write_text(json.dumps({'scope': __doc__, 'routes': [r.to_dict() for r in picked],
    'world': {'dTc_K': offset, 'kinetic_multiplier': kinetic}, 'specimen_seed': specimen_seed,
    'steps': result['steps'], 'wall_s': time.monotonic()-started, 'frames': len(frames)}, indent=1))
print('done', len(frames), result['steps'])
