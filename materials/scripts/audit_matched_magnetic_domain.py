"""Independent domain audit of the matched-rerun selections (solver interface).

Same fixtures and verifier as audit_decision_magnetic_domain.py, applied to the
training-format task: the original domain and endpoint specs plus the onset-study
temperature range. Selections stay frozen; no result is fed back.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'materials'), str(ROOT/'materials/scripts')]
from audit_decision_magnetic_domain import fixture  # noqa: E402
from budget_factorial import GuardedSandbox  # noqa: E402
from rlvpm import magnetic_task, magnetic_task_rlvp  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--specs', type=Path, nargs='+', required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
cases = magnetic_task.cases(True)
for spec in args.specs:
    cases += fixture(json.loads(spec.read_text()))
sources = {}
for path in sorted(args.root.glob('state-*.json')):
    state = json.loads(path.read_text())
    if state['policy'] not in ('fresh', 'feedback') or not state['selected']:
        continue
    selected = next(a for a in state['attempts'] if a['passed'])
    sha = hashlib.sha256(selected['code'].encode()).hexdigest()
    assert sha == state['selected']['code_sha256']
    sources.setdefault(sha, {'code': selected['code'], 'attribution': []})['attribution'].append(
        {'arm': state['arm'], 'policy': state['policy'], 'lineage_seed': state['lineage_seed'], 'turn': selected['turn']})
(args.out/'plan.json').write_text(json.dumps({'scope': __doc__, 'cases': [c['name'] for c in cases],
    'spec_sha256': {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in args.specs},
    'unique_selected_sources': len(sources)}, indent=1))
sb = GuardedSandbox(timeout_seconds=30, cpus='1')
sb.image = 'rlvp-solver:py312'
magnetic_task_rlvp.cases = lambda additional=False: cases
control = magnetic_task_rlvp.check(magnetic_task_rlvp.CONTROL, sb)
assert control['passed'], 'Fixture does not resolve the supplied derivative'
missing = magnetic_task_rlvp.check(magnetic_task_rlvp.CONTROL.replace('beta*gp*Ag-h*gp', 'beta*gp*Ag-h'), sb)
assert not missing['passed'], 'Fixture cannot detect the first-attempt error'
reports = []
for sha, row in sources.items():
    report = magnetic_task_rlvp.check(row['code'], sb)
    reports.append({'code_sha256': sha, 'attribution': row['attribution'], 'check': report})
    (args.out/'results.json').write_text(json.dumps(reports, indent=1))
    print(sha[:12], [a['arm'] for a in row['attribution']], report['n_passed'], '/', len(cases), flush=True)
(args.out/'done.json').write_text(json.dumps({'unique_sources': len(sources), 'cases': len(cases),
    'passed_sources': sum(r['check']['passed'] for r in reports), 'missing_g_prime_control_cases_passed': missing['n_passed']}, indent=1))
