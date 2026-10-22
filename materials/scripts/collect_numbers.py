"""Collect every number the materials section of the post reports into one JSON.

Reads the saved results in materials/data and the decision analysis that analyze_decision.py
writes to materials/out. Figures and prose take values from the output; nothing here is typed in
by hand except paths.
"""
import json
from pathlib import Path

MATERIALS = Path(__file__).resolve().parents[1]
DATA = MATERIALS/'data'
OUT = MATERIALS/'out'
MAGNETIC = DATA/'magnetic_term'


def load(path):
    return json.loads(Path(path).read_text())


def whole_solver():
    pilot = load(DATA/'whole_solver/pilot.json')
    budget = load(DATA/'whole_solver/budget.json')
    pilot_attempts = sum(c['generations'] for c in pilot['cells'].values())
    pilot_pass = sum(int(c['lineage_pass_within_rounds'][-1]*c['lineages']) for c in pilot['cells'].values())
    budget_pass = sum(c['passed_attempts'] for c in budget['cells'])
    return {'attempts': pilot_attempts+budget['total_attempts'], 'passes': pilot_pass+budget_pass,
            'studies': {'pilot_9216_tokens': {'attempts': pilot_attempts, 'passes': pilot_pass},
                        'factorial_9216_20000_tokens': {'attempts': budget['total_attempts'], 'passes': budget_pass,
                                                        'completion_tokens': budget['total_completion_tokens']}}}


def magnetic_matched():
    """The rerun in the training format; the earlier-format cohort is reported separately."""
    d = load(MAGNETIC/'summary.json')
    attempts = load(MAGNETIC/'attempts.json')
    by_arm = {arm: {k: sum(1 for a in attempts if a['arm'] == arm and a['outcome'] == k)
                    for k in ('pass', 'wrong physics', 'crash')} for arm in ('base', 'adapter')}
    return {'rows': d['rows'], 'unique_calls': d['unique_calls'],
            'unique_tokens': d['unique_prompt_tokens']+d['unique_completion_tokens'],
            'length_limited_unique_calls': d['length_limited_unique_calls'],
            'completion_tokens_by_arm': d['completion_tokens_by_arm'], 'outcomes_by_arm': by_arm,
            'audit': load(MAGNETIC/'audit.json'), 'cap': d['plan']['completion_cap']}


def magnetic_mismatched():
    """Earlier cohort with a different prompt, interface, extractor and a 4,096-token cap."""
    d = load(MAGNETIC/'earlier_format_summary.json')
    rows = [{k: r[k] for k in ['arm', 'policy', 'budget', 'lineages', 'passed_components', 'observed_calls',
                               'prompt_tokens', 'completion_tokens', 'length_limited_calls']} for r in d['rows']]
    return {'rows': rows, 'unique_calls': d['unique_actual_generation_calls'],
            'unique_tokens': d['unique_actual_prompt_tokens']+d['unique_actual_completion_tokens']}


def fourier_defect():
    rows = load(DATA/'fourier_defect.json')['long']
    out = []
    for r in rows:
        trace = [{k: t[k] for k in ('time', 'hermitian_error', 'min_composition', 'max_composition')} for t in r['trace']]
        fail = r.get('failure') or []
        out.append({'noise': r['noise'], 'corrected': r['corrected'], 'trace': trace,
                    'failure': {k: fail[0][k] for k in ('time', 'after_hermitian_error', 'after_min', 'after_max')} if fail else None,
                    'final': {'time': 6000., 'min': r.get('min'), 'max': r.get('max')} if not fail and r.get('steps') else None})
    return out


def noise_response():
    rows = []
    for f in sorted((DATA/'noise_response').glob('seed_*.json')):
        rows += [{k: r[k] for k in ('seed', 'd_z_nm', 'undercooling_K', 'gain_x', 'gain_y', 'bare_log_gain_x', 'bare_log_gain_y')}
                 for r in load(f)['rows'] if r['hours'] == 2]
    return rows


def main():
    data = {'whole_solver': whole_solver(), 'magnetic_matched': magnetic_matched(),
            'magnetic_mismatched_format': magnetic_mismatched(), 'fourier_defect': fourier_defect(),
            'fluctuation_response': noise_response()}
    decision = OUT/'decision.json'
    if decision.exists():
        data['onset'] = load(decision)
        # Pre-registered step-bias sensitivity: conclusions must hold with the fast ramp corrected by 0, 1.5 and 3.
        data['onset_ramp_sensitivity'] = {adj: {k: load(OUT/f'decision_ramp{adj}.json')[k]
                                                for k in ('policies', 'fit_best')} for adj in ('1.5', '3.0')}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'numbers.json').write_text(json.dumps(data, indent=1))
    print('wrote', OUT/'numbers.json')


if __name__ == '__main__':
    main()
