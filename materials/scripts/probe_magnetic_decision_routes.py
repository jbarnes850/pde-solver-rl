"""Prospective full-route magnetic decision qualification (the step-size check runs).

Trusted supplied solver; generated-component execution is a separate sandbox
bridge. Every requested world, law, route and specimen is retained. The primary
mathematical problem is the original finite grid, not a continuum limit claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvpm.physics import MaterialParams, Schedule, Segment, isothermal
from rlvpm.reference import Grid, band_limited_noise, schedule_coefficients
from rlvpm.reference_etd import simulate_etd
from rlvpm.qoi import microstructure, magnet_property
from rlvpm.routes import AGING


def routes():
    result = []
    for first, label in [([Segment(4., 943., 903., True)], 'ramp10Kph'),
                         ([isothermal(932.5, 1., True), Segment(3., 932.5, 923.5, True)],
                          'warm-hold-ramp3Kph')]:
        segments = list(first)
        for temperature, hours in AGING:
            previous = segments[-1].T_end
            if previous != temperature:
                segments.append(Segment(abs(previous-temperature)/600., previous, temperature, False))
            segments.append(isothermal(temperature, hours, False))
        route = Schedule(tuple(segments), label)
        assert route.hours <= 24 and sum(s.hours for s in route.segments if s.field) <= 4
        assert all(a.T_end == b.T_start for a, b in zip(route.segments, route.segments[1:]))
        for s in route.segments:
            assert 773 <= min(s.T_start, s.T_end) <= max(s.T_start, s.T_end) <= 973
            if s.T_start != s.T_end:
                assert .05-1e-12 <= abs(s.T_end-s.T_start)/(60*s.hours) <= 10+1e-12
        result.append(route)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    ap.add_argument('--h-max', type=float, required=True)
    ap.add_argument('--offsets', type=float, nargs='+', default=[-5., 5.])
    ap.add_argument('--seeds', type=int, nargs='+', default=[301, 302])
    ap.add_argument('--kinetic-multiplier', type=float, default=1.)
    ap.add_argument('--laws', choices=['full', 'omitted_local'], nargs='+', default=['full', 'omitted_local'])
    ap.add_argument('--role', choices=['qualification', 'fit', 'pilot', 'production'], default='qualification')
    args = ap.parse_args()
    assert args.h_max in [1., .25, .0625, .015625]
    assert set(args.offsets) <= {-15., -5., 5.}
    assert args.kinetic_multiplier in [.5, 1., 2.]
    namespace = {'qualification': (301, 308), 'fit': (401, 404),
                 'pilot': (501, 504), 'production': (601, 608)}[args.role]
    assert len(set(args.seeds)) == len(args.seeds) and all(namespace[0] <= s <= namespace[1] for s in args.seeds)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.05)
    grid = Grid(384, 384, 768., 768.)
    processing = routes()
    labels, parameters, coefficients, initial, groups = [], [], [], [], []
    for offset in args.offsets:
        p = MaterialParams(c0=.5, T_s=933., mag_model='curie', Tc_a=1250.,
                           Tc_b=2*(933.+offset)-1250., D_ref=3e-18*args.kinetic_multiplier,
                           applied_field_T=1.)
        for seed in args.seeds:
            # Independent specimens across worlds/roles; paired routes and laws
            # only for qualification/fitting/production variance reduction.
            world_id = [-15., -5., 5.].index(offset)*3+[.5, 1., 2.].index(args.kinetic_multiplier)
            for law in args.laws:
                for route_index, route in enumerate(processing):
                    specimen_seed = 2026120000+1000*world_id+seed
                    if args.role == 'pilot':
                        specimen_seed += 100000*route_index
                    field = p.c0+band_limited_noise(grid, p.noise_rms, seed=specimen_seed, k_cut=.8)
                    param = p if law == 'full' else p.with_(applied_field_T=0.)
                    labels.append({'role': args.role, 'world_id': world_id, 'dTc_K': offset,
                                   'kinetic_multiplier': args.kinetic_multiplier, 'seed': seed,
                                   'initial_seed': specimen_seed, 'law': law, 'route': route.label})
                    parameters.append(param)
                    coefficients.append(schedule_coefficients(param, route))
                    initial.append(field)
                    groups.append(specimen_seed)
    initial = np.stack(initial)
    noise_seed = 2026130000
    noise_seed += {'qualification': 0, 'fit': 10000, 'pilot': 20000, 'production': 30000}[args.role]
    noise_seed += 100*[1., .25, .0625, .015625].index(args.h_max)+10*[.5, 1., 2.].index(args.kinetic_multiplier)
    noise_seed += min(args.seeds)
    hashes = {name: hashlib.sha256((Path(__file__).resolve().parents[1]/'rlvpm'/name).read_bytes()).hexdigest()
              for name in ['reference_etd.py', 'reference.py', 'physics.py', 'qoi.py']}
    plan = {'scope': __doc__, 'labels': labels, 'grid': vars(grid), 'routes': [r.to_dict() for r in processing],
            'params': [p.to_dict() for p in parameters], 'h_max': args.h_max, 'device': args.device,
            'noise_seed': noise_seed, 'noise_groups': groups, 'initial_array_sha256': hashlib.sha256(initial.tobytes()).hexdigest(),
            'source_sha256': hashes, 'enforce_real_spectrum': True,
            'role': args.role, 'target': {'spectral_aspect_min': 3., 'width_proxy_nm': [10., 30.]},
            'limits': 'Specified finite-grid synthetic model. Independent streams across caps. Omitted-local reference uses its own mean linearization; it is equation-equivalent, not automatically finite-step-equivalent, to erroneous generated code inside the full-law split.'}
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2))
    np.savez_compressed(args.out/'initial.npz', fields=initial)
    started = time.monotonic()
    rows = []

    def snapshot(t, fields, final=False):
        for label, field, p, co in zip(labels, fields, parameters, coefficients):
            ms = microstructure(field, grid.dx, grid.dy, p)
            rows.append({**label, 'hours': (co.end if final else t)/3600, 'final': final,
                         'h_max': args.h_max, **ms.to_dict(), **magnet_property(ms, p).to_dict()})
        np.savez_compressed(args.out/f'fields-{t}.npz', fields=fields)
        temp = args.out/'results.tmp'
        temp.write_text(json.dumps(rows, indent=2))
        temp.replace(args.out/'results.json')

    def progress(d):
        row = d | {'wall_s': time.monotonic()-started}
        (args.out/'progress.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)

    def failure(d, before, after):
        np.savez_compressed(args.out/'failure-fields.npz', before=before, after=after)
        (args.out/'failure-detail.json').write_text(json.dumps(d | {
            'invalid_labels': [labels[i] for i in d['invalid_members']]}, indent=2))

    try:
        result = simulate_etd(initial, grid, coefficients, out_times=[3600., 14400., 28800., 43200.],
                              h_max=args.h_max, device=args.device, dtype=torch.float64,
                              noise_seed=noise_seed, noise_groups=groups, max_steps=5_000_000,
                              cache_linear=True, enforce_real_spectrum=True, audit_spectrum=True,
                              snapshot_callback=snapshot, progress_callback=progress,
                              progress_every=2000, failure_callback=failure)
        snapshot(max(co.end for co in coefficients), result['final'], True)
    except Exception as error:
        (args.out/'failed.json').write_text(json.dumps({'error': repr(error), 'wall_s': time.monotonic()-started}, indent=2))
        raise
    (args.out/'done.json').write_text(json.dumps({k: result[k] for k in ['steps', 'rejected', 'linear_cache_hits', 'linear_cache_misses']} |
                                               {'wall_s': time.monotonic()-started}, indent=2))


if __name__ == '__main__':
    main()
