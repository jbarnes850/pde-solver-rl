"""Independent domain audit of the selected magnetic-term programs.

Reruns the unchanged variational check on every program the rerun selected, over the original
task's three additional cases plus the physical fixtures declared in the spec files (the original
domain and endpoint specs and the decision-map temperature range). Selections stay frozen and no
result is fed back. Writes audit.json (the summary) and checks.json (every case of every program)
to --out. Runs in the Docker solver sandbox (containers/solver.Dockerfile).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnet import magnetic_task, magnetic_task_matched  # noqa: E402
from magnet.physics import MaterialParams  # noqa: E402
from magnet.sandbox import GuardedSandbox  # noqa: E402


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


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--programs', type=Path, required=True, help='selected_programs.json from analyze_magnetic_term.py')
ap.add_argument('--specs', type=Path, nargs='+', required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
cases = magnetic_task.cases(True)
for spec in args.specs:
    cases += fixture(json.loads(spec.read_text()))
sources = json.loads(args.programs.read_text())
assert all(hashlib.sha256(row['code'].encode()).hexdigest() == sha for sha, row in sources.items())
sb = GuardedSandbox(timeout_seconds=30, cpus='1')
magnetic_task_matched.cases = lambda additional=False: cases
control = magnetic_task_matched.check(magnetic_task_matched.CONTROL, sb)
assert control['passed'], 'Fixture does not resolve the supplied derivative'
missing = magnetic_task_matched.check(magnetic_task_matched.CONTROL.replace('beta*gp*Ag-h*gp', 'beta*gp*Ag-h'), sb)
assert not missing['passed'], 'Fixture cannot detect the first-attempt error'
reports = []
for sha, row in sources.items():
    report = magnetic_task_matched.check(row['code'], sb)
    reports.append({'code_sha256': sha, 'attribution': row['attribution'], 'check': report})
    (args.out/'checks.json').write_text(json.dumps(reports, indent=1))
    print(sha[:12], [a['arm'] for a in row['attribution']], report['n_passed'], '/', len(cases), flush=True)
(args.out/'audit.json').write_text(json.dumps({'unique_sources': len(sources), 'cases': len(cases),
    'passed_sources': sum(r['check']['passed'] for r in reports), 'missing_g_prime_control_cases_passed': missing['n_passed']}, indent=1))
