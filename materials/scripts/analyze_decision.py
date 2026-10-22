"""Decision analysis for the onset-referenced route map (materials/README.md).

Inputs: results.json files from simulate_decision_map.py. Choices use fit specimens
(correct simulator: full law, fit seeds; erroneous simulator: omitted local
field, fit seed 701). Scoring uses production specimens only.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from magnet.physics import MaterialParams  # noqa: E402

OFFSETS = (-15.0, -5.0, 5.0)
KINETICS = (.5, 1., 2.)
FALLBACK = 'ramp10Kph'
PILOT = ('ramp10Kph', 'sweep932.5')
COST = {'none': 0., 'kinetics': .25, 'curie': .25, 'onset': .5, 'onset+kinetics': .75, 'pilot': 1.}


def linear_onset(offset, zeeman=True):
    """Highest temperature at which k_x = 0 modes are linearly unstable (c0 = 0.5)."""
    import torch
    from magnet.reference_etd import curie_second
    p = MaterialParams(c0=.5, T_s=933., mag_model='curie', Tc_a=1250., Tc_b=2*(933.+offset)-1250., applied_field_T=1.)

    def a0(T):
        value = 4*(p.theta(T)-1)
        if zeeman:
            g2 = float(curie_second(torch.tensor(.5, dtype=torch.float64), torch.tensor(T, dtype=torch.float64),
                                    p.Tc_a, p.Tc_b, p.curie_width))
            value -= p.beta_M*(p.applied_field_T/p.Js_a)*g2
        return value
    lo, hi = 900., 980.
    assert a0(lo) < 0 < a0(hi)
    for _ in range(60):
        mid = .5*(lo+hi)
        lo, hi = (mid, hi) if a0(mid) < 0 else (lo, mid)
    return .5*(lo+hi)


def load(paths):
    rows, snaps = [], []
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            (rows if r['final'] else snaps).append(r)
    return rows, snaps


def tables(rows, routes):
    worlds = sorted({(r['dTc_K'], r['kinetic_multiplier']) for r in rows})
    def mean(role, law, seeds=None):
        out = {}
        for w in worlds:
            for rt in routes:
                v = [r['BHmax_kJ_m3'] for r in rows if (r['dTc_K'], r['kinetic_multiplier']) == w and r['route'] == rt
                     and r['role'] == role and r['law'] == law and (seeds is None or r['seed'] in seeds)]
                out[w, rt] = (float(np.mean(v)) if v else math.nan, len(v))
        return out
    return worlds, mean


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', nargs='+')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--geometry', type=Path)
    ap.add_argument('--ramp-adjust', type=float, default=0., help='sensitivity: added to every ramp10Kph utility')
    ap.add_argument('--magnetic-term', type=Path, default=Path(__file__).resolve().parents[1]/'data/magnetic_term/summary.json')
    args = ap.parse_args()
    rows, snaps = load(args.results)
    if args.ramp_adjust:
        for r in rows:
            if r['route'] == 'ramp10Kph':
                r['BHmax_kJ_m3'] += args.ramp_adjust
    routes = ['ramp10Kph'] + sorted({r['route'] for r in rows if r['route'].startswith('sweep')}, key=lambda s: float(s[5:]))
    worlds, mean = tables(rows, routes)
    fit = mean('fit', 'full')
    err = mean('fit', 'omitted_local', {701})
    prod = mean('production', 'full')
    W, R = len(worlds), len(routes)
    F = np.array([[fit[w, r][0] for r in routes] for w in worlds])
    E = np.array([[err[w, r][0] for r in routes] for w in worlds])
    P = np.array([[prod[w, r][0] for r in routes] for w in worlds])
    prod_seeds = sorted({r['seed'] for r in rows if r['role'] == 'production'})
    Pspec = np.array([[[next((x['BHmax_kJ_m3'] for x in rows if (x['dTc_K'], x['kinetic_multiplier']) == w and x['route'] == r
                              and x['role'] == 'production' and x['law'] == 'full' and x['seed'] == sd), math.nan)
                        for sd in prod_seeds] for r in routes] for w in worlds])
    coupons = {(w, r): [x['BHmax_kJ_m3'] for x in rows if (x['dTc_K'], x['kinetic_multiplier']) == w and x['route'] == r
                        and x['role'] == 'fit' and x['law'] == 'full'] for w in worlds for r in PILOT}
    onset = {o: linear_onset(o) for o in OFFSETS}
    onset_err = linear_onset(OFFSETS[0], zeeman=False)
    sweep_T = {r: float(r[5:]) for r in routes if r.startswith('sweep')}

    rng = np.random.default_rng(2026221301)
    n_draw = 2048
    best_prod = P.max(1)

    def choose_bayes(S, post):
        return int(np.argmax(post @ S))

    def snap(T):
        return min(sweep_T, key=lambda r: abs(sweep_T[r]-T))

    results, shares = {}, {}
    for sim_name, S in [('correct', F), ('erroneous', E)]:
        for obs in COST:
            utilities, chosen = np.zeros((W, n_draw)), np.zeros((W, R))
            for i, (dtc, kin) in enumerate(worlds):
                for d in range(n_draw):
                    logpost = np.zeros(W)
                    if obs in ('kinetics', 'onset+kinetics'):
                        y = math.log(kin)+rng.normal(0, .25)
                        logpost += np.array([-.5*((y-math.log(k))/.25)**2 for _, k in worlds])
                    if obs == 'curie':
                        y = dtc+rng.normal(0, 2.)
                        logpost += np.array([-.5*((y-o)/2.)**2 for o, _ in worlds])
                    if obs == 'pilot':
                        means = {r: np.mean(np.array(coupons[(dtc, kin), r])+rng.normal(0, 1., len(coupons[(dtc, kin), r]))) for r in PILOT}
                        j = routes.index(max(means, key=means.get))
                    elif obs in ('onset', 'onset+kinetics'):
                        y = onset[dtc]+rng.normal(0, 1.5)
                        if sim_name == 'correct':
                            logpost += np.array([-.5*((y-onset[o])/1.5)**2 for o, _ in worlds])
                            post = np.exp(logpost-logpost.max()); post /= post.sum()
                            j = choose_bayes(S, post)
                        else:
                            post = np.exp(logpost-logpost.max()); post /= post.sum()
                            avg = post @ S
                            best = routes[int(np.argmax(avg))]
                            if best.startswith('sweep'):
                                j = routes.index(snap(y+sweep_T[best]-onset_err))
                            else:
                                j = routes.index(best)
                    else:
                        post = np.exp(logpost-logpost.max()); post /= post.sum()
                        j = choose_bayes(S, post)
                    utilities[i, d] = P[i, j]
                    chosen[i, j] += 1
            u = utilities.mean(1)
            shares[sim_name, obs] = chosen/n_draw
            results[sim_name, obs] = {
                'production_utility_mean': float(u.mean()),
                'per_world_utility': {f'dTc{w[0]:+g}_D{w[1]:g}': float(x) for w, x in zip(worlds, u)},
                'regret_vs_production_best_mean': float((best_prod-u).mean()),
                'cost': COST[obs], 'net_utility': float(u.mean()-COST[obs]),
                'choice_shares': {f'dTc{w[0]:+g}_D{w[1]:g}': {routes[j]: float(c/n_draw) for j, c in enumerate(chosen[i]) if c}
                                  for i, w in enumerate(worlds)}}
    # Bootstrap over production specimens (resampled within each world and route), paired across policies.
    brng = np.random.default_rng(2026221302)
    n_boot = 2000
    S_ = len(prod_seeds)
    boot = {k: np.zeros(n_boot) for k in shares}
    for b in range(n_boot):
        pick = brng.integers(0, S_, size=(W, R, S_))
        Pb = np.take_along_axis(Pspec, pick, axis=2).mean(2)
        for k, sh in shares.items():
            boot[k][b] = float((sh*Pb).sum(1).mean())
    for (sim_name, obs), v in boot.items():
        ref = boot['correct', 'none']
        results[sim_name, obs]['production_utility_ci95'] = [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
        d = v-COST[obs]-ref
        results[sim_name, obs]['net_gain_vs_verified_none_ci95'] = [float(np.quantile(d, .025)), float(np.quantile(d, .975))]
    # exact-information controls with the correct simulator
    controls = {}
    for name, key in [('know_D', lambda w: w[1]), ('know_dTc', lambda w: w[0]), ('know_world', lambda w: w)]:
        u = []
        for i, w in enumerate(worlds):
            mask = np.array([key(v) == key(w) for v in worlds], float)
            j = int(np.argmax((mask/mask.sum()) @ F))
            u.append(P[i, j])
        controls[name] = float(np.mean(u))
    report = {'scope': __doc__, 'worlds': [list(w) for w in worlds], 'routes': routes,
              'linear_onset_K': {str(o): onset[o] for o in OFFSETS}, 'onset_omitted_K': onset_err,
              'fit_mean': F.tolist(), 'erroneous_fit': E.tolist(), 'production_mean': P.tolist(),
              'counts': {'fit': [fit[w, r][1] for w in worlds for r in routes][:1],
                         'production': [prod[w, r][1] for w in worlds for r in routes][:1]},
              'fit_best': {f'dTc{w[0]:+g}_D{w[1]:g}': routes[int(np.argmax(F[i]))] for i, w in enumerate(worlds)},
              'erroneous_best': {f'dTc{w[0]:+g}_D{w[1]:g}': routes[int(np.argmax(E[i]))] for i, w in enumerate(worlds)},
              'production_best': {f'dTc{w[0]:+g}_D{w[1]:g}': routes[int(np.argmax(P[i]))] for i, w in enumerate(worlds)},
              'policies': {f'{s}|{o}': v for (s, o), v in results.items()},
              'exact_information_correct_simulator': controls,
              'production_best_mean_noisy_comparator': float(best_prod.mean())}
    if args.magnetic_term.exists():
        ttc = json.loads(args.magnetic_term.read_text())
        fallback = results['correct', 'none']['production_utility_mean']
        j = routes.index(FALLBACK)
        u_fallback = float(P[:, j].mean())
        curve = []
        for row in ttc['rows']:
            p_pass = row['passed_components']/row['lineages']
            for obs in COST:
                u_c = results['correct', obs]['production_utility_mean']
                u_e = results['erroneous', obs]['production_utility_mean']
                curve.append({'arm': row['arm'], 'policy': row['policy'], 'budget': row['budget'], 'p_pass': p_pass,
                              'tokens': row['prompt_tokens']+row['completion_tokens'], 'observation': obs,
                              'gated_utility': p_pass*u_c+(1-p_pass)*(u_fallback if obs != 'pilot' else u_c),
                              'ungated_utility': p_pass*u_c+(1-p_pass)*u_e})
        report['ttc_curve'] = curve
        report['fallback_utility'] = u_fallback
        del fallback
    stage = {}
    for hours in (4.0,):
        for q in ('aspect', 'contrast'):
            stage[f'{q}_at_{hours:g}h'] = [[float(np.mean([x[q] for x in snaps if abs(x['hours']-hours) < 1e-6
                                                           and (x['dTc_K'], x['kinetic_multiplier']) == w and x['route'] == r
                                                           and x['role'] == 'fit' and x['law'] == 'full'] or [math.nan]))
                                            for r in routes] for w in worlds]
    final_q = {q: [[float(np.mean([x[q] for x in rows if (x['dTc_K'], x['kinetic_multiplier']) == w and x['route'] == r
                                   and x['law'] == 'full'] or [math.nan])) for r in routes] for w in worlds]
               for q in ('aspect', 'L_perp_nm', 'd_nm', 'contrast')}
    report['field_stage_end'] = stage
    report['final_descriptors_all_full_law'] = final_q
    report['target_hits'] = sum(1 for x in rows if x['aspect'] >= 3 and 10 <= x['d_nm'] <= 30)
    report['max_final_aspect'] = max(x['aspect'] for x in rows)
    if args.geometry and args.geometry.exists():
        geo = json.loads(args.geometry.read_text())['rows']
        report['chord_ratio_full_law'] = [[float(np.nanmean([g['finite_chord_ratio'] for g in geo
                                                             if (g['dTc_K'], g['kinetic_multiplier']) == w and g['route'] == r
                                                             and g['law'] == 'full' and g['finite_chord_ratio']] or [math.nan]))
                                           for r in routes] for w in worlds]
        report['chord_across_nm_full_law'] = [[float(np.nanmean([g['chord_across_nm'] for g in geo
                                                                 if (g['dTc_K'], g['kinetic_multiplier']) == w and g['route'] == r
                                                                 and g['law'] == 'full' and g['chord_across_nm']] or [math.nan]))
                                               for r in routes] for w in worlds]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ['linear_onset_K', 'fit_best', 'erroneous_best', 'production_best',
                                              'exact_information_correct_simulator', 'production_best_mean_noisy_comparator']}, indent=1))
    for k, v in report['policies'].items():
        print(f"{k:28s} utility {v['production_utility_mean']:6.2f} [{v['production_utility_ci95'][0]:5.2f},{v['production_utility_ci95'][1]:5.2f}]  net {v['net_utility']:6.2f}  gain vs none [{v['net_gain_vs_verified_none_ci95'][0]:5.2f},{v['net_gain_vs_verified_none_ci95'][1]:5.2f}]")


if __name__ == '__main__':
    main()
