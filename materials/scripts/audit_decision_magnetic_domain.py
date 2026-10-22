"""Independent domain audit after magnetic candidates have been selected.

Use the unchanged variational verifier with the separately declared physical
fixture. No additional result is fed back or used to select a replacement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'materials'), str(ROOT/'materials/scripts')]
from budget_factorial import GuardedSandbox
from rlvpm import magnetic_task
from rlvpm.physics import MaterialParams


def fixture(spec):
    rng = np.random.default_rng(spec['direction_seed'])
    nx, ny = spec['grid']
    x = np.arange(nx)[:, None]/nx
    y = np.arange(ny)[None, :]/ny
    fields = np.stack([.5+.3*np.cos(2*np.pi*x+phase)+.1*np.sin(4*np.pi*y+.7*phase)
                       for phase in [.2, .9]])
    assert fields.min() > 0 and fields.max() < 1
    rows = []
    params = MaterialParams(c0=.5, T_s=933.)
    for offset in spec['dTc_K']:
        for temperature in spec['field_on_T_K']+[spec['field_off_T_K']]:
            on = temperature != spec['field_off_T_K']
            scalars = {'T': temperature, 'beta': params.beta_M if on else 0.,
                       'h': params.beta_M/2 if on else 0., 'Tc_a': spec['Tc_a'],
                       'Tc_b': 2*(933+offset)-spec['Tc_a'], 'width': spec['curie_width'],
                       'dx': spec['dx'], 'dy': spec['dy']}
            directions = []
            for i in range(5):
                v = rng.normal(size=fields.shape)
                directions.append(v/v.std()+(.3 if i == 0 else 0))
            rows.append({'name': f'dTc{offset}_T{temperature}_'+('field' if on else 'off'),
                         'c': fields.copy(), 'scalars': scalars, 'directions': directions})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--spec', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    spec = json.loads(args.spec.read_text())
    # Retain the existing three additional mathematical cases as well as the
    # twelve decision-domain cases. This preserves their batch/grid coverage.
    cases = magnetic_task.cases(True)+fixture(spec)
    sources = {}
    for path in sorted(args.root.glob('state-*.json')):
        state = json.loads(path.read_text())
        if state['policy'] not in ['fresh', 'feedback'] or not state['selected']:
            continue
        selected = next(a for a in state['attempts'] if a['passed'])
        sha = hashlib.sha256(selected['code'].encode()).hexdigest()
        assert sha == state['selected']['code_sha256']
        row = sources.setdefault(sha, {'code': selected['code'], 'attribution': []})
        row['attribution'].append({'arm': state['arm'], 'policy': state['policy'],
                                  'lineage_seed': state['lineage_seed'], 'turn': selected['turn']})
    (args.out/'frozen-selections.json').write_text(json.dumps(sources, indent=2))
    (args.out/'plan.json').write_text(json.dumps({'scope': __doc__, 'spec_sha256': hashlib.sha256(args.spec.read_bytes()).hexdigest(),
        'verifier_sha256': hashlib.sha256(Path(magnetic_task.__file__).read_bytes()).hexdigest(),
        'unique_selected_sources': len(sources), 'cases': [c['name'] for c in cases],
        'case_sources': 'Existing additional mathematical fixture plus frozen decision-domain fixture',
        'selection': 'Already frozen first original-verifier pass. No replacement after additional failure.',
        'limits': 'Finite manufactured operator probes across processing parameters; not full stochastic route verification or proof for every possible input.'}, indent=2))
    sb = GuardedSandbox(timeout_seconds=30, cpus='1')
    sb.image = 'rlvp-solver:py312'
    # Fixture injection is scoped to this independent evaluator process. The
    # original checker/energy/thresholds and generation-visible task stay intact.
    original = magnetic_task.cases
    magnetic_task.cases = lambda additional=False: cases
    reports = []
    try:
        control = magnetic_task.check(magnetic_task.CONTROL, sb)
        (args.out/'control.json').write_text(json.dumps(control, indent=2))
        assert control['passed'], 'Fixture does not resolve the supplied derivative; do not score candidates'
        assert max(c['finite_difference_halving_change'] for c in control['cases']) < 1e-4
        for sha, row in sources.items():
            report = magnetic_task.check(row['code'], sb)
            reports.append({'code_sha256': sha, 'attribution': row['attribution'], 'check': report})
            (args.out/'results.json').write_text(json.dumps(reports, indent=2))
            print(sha, report['n_passed'], '/', len(cases), flush=True)
    finally:
        magnetic_task.cases = original
    (args.out/'done.json').write_text(json.dumps({'unique_sources': len(sources)}, indent=2))


if __name__ == '__main__':
    main()
