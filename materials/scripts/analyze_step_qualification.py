"""Step-size check from the trusted two-route qualification runs (probe_magnetic_decision_routes.py).

Same 16 members (Curie offset ±5 K, specimens 301/302, two routes, two magnetic
laws), same initial fields, independent thermal noise across step caps. Compares
h = 0.25 and h = 0.0625 at every saved stage, member by member. Differences mix
numerical error with noise-realization differences, so the between-member spread
at h = 0.25 is reported alongside as the scale of realization noise.
"""
import json
from pathlib import Path

import numpy as np

MATERIALS = Path(__file__).resolve().parents[1]
R2 = MATERIALS/'data/step-qualification'
OUT = MATERIALS/'out/step-qualification.json'


def main():
    coarse = json.loads((R2/'coarse/results.json').read_text())
    fine = json.loads((R2/'fine/results.json').read_text())
    key = lambda r: (r['dTc_K'], r['seed'], r['law'], r['route'])  # noqa: E731
    report = {'scope': __doc__, 'stages': {}}
    stages = sorted({('final' if r['final'] else r['hours']) for r in fine}, key=str)
    for stage in stages:
        pick = (lambda r: r['final']) if stage == 'final' else (lambda r, s=stage: not r['final'] and abs(r['hours']-s) < 1e-6)
        A = {key(r): r for r in coarse if pick(r)}
        B = {key(r): r for r in fine if pick(r)}
        ks = sorted(set(A) & set(B))
        report['stages'][str(stage)] = {q: {'mean_diff_fine_minus_coarse': float(np.mean([B[k][q]-A[k][q] for k in ks])),
                                            'sd_diff': float(np.std([B[k][q]-A[k][q] for k in ks], ddof=1)),
                                            'coarse_member_sd': float(np.std([A[k][q] for k in ks], ddof=1)), 'n': len(ks)}
                                        for q in ('aspect', 'contrast', 'BHmax_kJ_m3', 'L_perp_nm')}
        # route contrast (warm hold minus fast ramp) per law and offset, both caps
        contr = {}
        for law in ('full', 'omitted_local'):
            for off in (-5.0, 5.0):
                for name, D in (('coarse', A), ('fine', B)):
                    v = [D[(off, s, law, 'warm-hold-ramp3Kph')]['BHmax_kJ_m3']-D[(off, s, law, 'ramp10Kph')]['BHmax_kJ_m3']
                         for s in (301, 302) if (off, s, law, 'warm-hold-ramp3Kph') in D and (off, s, law, 'ramp10Kph') in D]
                    contr[f'{law}|{off:+g}|{name}'] = v
        report['stages'][str(stage)]['warm_minus_ramp_utility'] = contr
    done = (R2/'fine/done.json').exists()
    report['fine_complete'] = done
    OUT.write_text(json.dumps(report, indent=1))
    for stage, v in report['stages'].items():
        print(stage, {q: (round(v[q]['mean_diff_fine_minus_coarse'], 3), round(v[q]['sd_diff'], 3), round(v[q]['coarse_member_sd'], 3))
                      for q in ('aspect', 'contrast', 'BHmax_kJ_m3')})


if __name__ == '__main__':
    main()
