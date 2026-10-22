"""Onset-referenced field-treatment map across the nine declared worlds.

Question: where should the slow field sweep start relative to the in-field
decomposition onset, and does that choice depend on the uncertain Curie offset
(via the applied-field Zeeman term) and kinetics? One process runs one kinetic
multiplier (all three Curie offsets), so every member shares the step mesh.

Routes: the fast 10 K/h ramp (943 -> 903 K, field on 4 h) and a family of
1 h holds at T_start followed by a 3 K/h ramp for 3 h (field on), T_start on a
fixed grid; all then take the declared aging sequence. 'sweep932.5' is the
warm-hold-ramp3Kph route of the step-size runs. Specimen roles are fixed before any
outcome: fit seeds choose routes, production seeds only score them, and the
omitted-local-field law (the 9B first-attempt error) runs on fit seeds only.
Routes and laws share initial fields and thermal increments within a specimen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from magnet.physics import MaterialParams, Schedule, Segment, isothermal
from magnet.reference import Grid, band_limited_noise, schedule_coefficients
from magnet.reference_etd import simulate_etd
from magnet.properties import microstructure, magnet_property
from magnet.routes import AGING

T_STARTS = (929.0, 932.5, 935.0, 938.0, 941.0, 944.0, 947.0)
OFFSETS = (-15.0, -5.0, 5.0)
KINETICS = (.5, 1., 2.)


def with_aging(first, label):
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
    return route


def routes(t_starts=T_STARTS, ramp=True):
    result = [with_aging([Segment(4., 943., 903., True)], 'ramp10Kph')] if ramp else []
    for T in t_starts:
        result.append(with_aging([isothermal(T, 1., True), Segment(3., T, T-9., True)], f'sweep{T:g}'))
    return result


def world_params(offset, kinetic):
    return MaterialParams(c0=.5, T_s=933., mag_model='curie', Tc_a=1250.,
                          Tc_b=2*(933.+offset)-1250., D_ref=3e-18*kinetic, applied_field_T=1.)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    ap.add_argument('--h-max', type=float, default=1.)
    ap.add_argument('--h-aging', type=float, default=.25)
    ap.add_argument('--kinetic-multiplier', type=float, required=True)
    ap.add_argument('--fit-seeds', type=int, nargs='*', default=[701, 702])
    ap.add_argument('--production-seeds', type=int, nargs='*', default=[801, 802])
    ap.add_argument('--omitted-seeds', type=int, nargs='*', default=[701])
    ap.add_argument('--memory-fraction', type=float, default=.45)
    ap.add_argument('--smoke', action='store_true', help='tiny CPU grid, coarse cap: plumbing only')
    ap.add_argument('--t-starts', type=float, nargs='+', default=list(T_STARTS))
    ap.add_argument('--no-ramp', action='store_true')
    ap.add_argument('--noise-offset', type=int, default=0)
    args = ap.parse_args()
    assert args.kinetic_multiplier in KINETICS
    assert set(args.omitted_seeds) <= set(args.fit_seeds)
    assert not set(args.fit_seeds) & set(args.production_seeds)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    grid = Grid(48, 48, 96., 96.) if args.smoke else Grid(384, 384, 768., 768.)
    h_max = 4. if args.smoke else args.h_max
    h_aging = args.h_aging
    processing = routes(tuple(args.t_starts), not args.no_ramp)
    labels, parameters, coefficients, initial, groups = [], [], [], [], []
    plan_rows = [(s, 'fit', 'full') for s in args.fit_seeds] + \
                [(s, 'production', 'full') for s in args.production_seeds] + \
                [(s, 'fit', 'omitted_local') for s in args.omitted_seeds]
    for offset in (OFFSETS[:1] if args.smoke else OFFSETS):
        p = world_params(offset, args.kinetic_multiplier)
        world_id = OFFSETS.index(offset)*3+KINETICS.index(args.kinetic_multiplier)
        for seed, role, law in (plan_rows[:1] if args.smoke else plan_rows):
            specimen_seed = 2026220000+1000*world_id+seed
            field = p.c0+band_limited_noise(grid, p.noise_rms, seed=specimen_seed, k_cut=.8)
            param = p if law == 'full' else p.with_(applied_field_T=0.)
            for route in processing:
                labels.append({'world_id': world_id, 'dTc_K': offset, 'kinetic_multiplier': args.kinetic_multiplier,
                               'seed': seed, 'role': role, 'law': law, 'initial_seed': specimen_seed,
                               'route': route.label})
                parameters.append(param)
                coefficients.append(schedule_coefficients(param, route))
                initial.append(field)
                groups.append(specimen_seed)
    initial = np.stack(initial)
    noise_seed = 2026240000+10*KINETICS.index(args.kinetic_multiplier)+100*[1., .25, .0625, 4.].index(h_max)
    if set(args.production_seeds) != {801, 802} or set(args.fit_seeds) != {701, 702}:
        # Split batches must not reuse another batch's thermal stream (row k of the white noise).
        noise_seed += 1000*min(args.fit_seeds+args.production_seeds)
    noise_seed += args.noise_offset
    src = Path(__file__).resolve().parent
    root = src.parent
    hashes = {name: hashlib.sha256((root/'magnet'/name).read_bytes()).hexdigest()
              for name in ['reference_etd.py', 'reference.py', 'physics.py', 'properties.py', 'routes.py']}
    plan = {'scope': __doc__, 'labels': labels, 'grid': vars(grid), 'routes': [r.to_dict() for r in processing],
            'params': [p.to_dict() for p in parameters], 'h_max': h_max, 'h_aging': h_aging, 'h_switch_time': 14400., 'device': args.device,
            'noise_seed': noise_seed, 'noise_groups': groups, 'smoke': args.smoke,
            'initial_array_sha256': hashlib.sha256(initial.tobytes()).hexdigest(),
            'source_sha256': hashes, 'enforce_real_spectrum': True,
            'target': {'spectral_aspect_min': 3., 'width_proxy_nm': [10., 30.]}}
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2))
    started = time.monotonic()
    rows = []
    out_times = [3600., 14400., 28800., 43200.]

    def snapshot(t, fields, final=False):
        for label, field, p, co in zip(labels, fields, parameters, coefficients):
            ms = microstructure(field, grid.dx, grid.dy, p)
            rows.append({**label, 'hours': (co.end if final else t)/3600, 'final': final,
                         'h_max': h_max, 'h_aging': h_aging, **ms.to_dict(), **magnet_property(ms, p).to_dict()})
        if final or t == 14400.:
            np.savez_compressed(args.out/f'fields-{"final" if final else int(t)}.npz', fields=fields.astype(np.float32))
        temp = args.out/'results.tmp'
        temp.write_text(json.dumps(rows, indent=1))
        temp.replace(args.out/'results.json')

    def progress(d):
        row = d | {'wall_s': time.monotonic()-started, 'members': len(labels)}
        (args.out/'progress.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)

    def failure(d, before, after):
        np.savez_compressed(args.out/'failure-fields.npz', before=before, after=after)
        (args.out/'failure-detail.json').write_text(json.dumps(d | {
            'invalid_labels': [labels[i] for i in d['invalid_members']]}, indent=2))

    try:
        result = simulate_etd(initial, grid, coefficients, out_times=out_times,
                              h_max=h_max, device=args.device, dtype=torch.float64,
                              noise_seed=noise_seed, noise_groups=groups, max_steps=8_000_000,
                              cache_linear=True, enforce_real_spectrum=True, audit_spectrum=True,
                              h_max_after=h_aging, h_switch_time=14400.,
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
