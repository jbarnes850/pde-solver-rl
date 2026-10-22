"""Summarize the matched 9B rerun by model, policy and call budget.

Reads the durable state files written by magnetic_matched_ttc_batch.py. A chain
counts as passing at budget b if its first passing call is at turn <= b. Fresh and
feedback chains share their first call; tokens count each policy's calls, with the
shared first call counted in both policies, as in the earlier summary.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--root', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
args = ap.parse_args()
states = [json.loads(p.read_text()) for p in sorted(args.root.glob('state-*.json'))]
states = [s for s in states if s['policy'] in ('fresh', 'feedback')]
plan = json.loads((args.root/'plan.json').read_text())
rows = []
for arm in ('base', 'adapter'):
    for policy in ('fresh', 'feedback'):
        chains = [s for s in states if s['arm'] == arm and s['policy'] == policy]
        for budget in (1, 2, 4, 8):
            calls = [a for s in chains for a in s['attempts'] if a['turn'] <= budget]
            failures = Counter('pass' if a['passed'] else 'execution' if any(c.get('error') for c in a.get('check', {}).get('cases', []))
                               else 'numerical_check' for a in calls)
            rows.append({'arm': arm, 'policy': policy, 'budget': budget, 'lineages': len(chains),
                         'passed_components': sum(1 for s in chains if s['selected'] and s['selected']['turn'] <= budget),
                         'observed_calls': len(calls),
                         'prompt_tokens': sum(a['prompt_tokens'] for a in calls),
                         'completion_tokens': sum(a['completion_tokens'] for a in calls),
                         'length_limited_calls': sum(a['finish_reason'] == 'length' for a in calls),
                         'attempt_outcomes': dict(failures)})
unique = {}
for s in states:
    for a in s['attempts']:
        unique[(a['arm'], a['policy'] if a['turn'] > 1 else 'shared', a['lineage_seed'], a['turn'])] = a
selected = {}
for s in states:
    if s['selected']:
        a = next(a for a in s['attempts'] if a.get('code_sha256') == s['selected']['code_sha256'])
        selected.setdefault(s['selected']['code_sha256'], {'code': a['code'], 'attribution': []})['attribution'].append(
            {'arm': s['arm'], 'policy': s['policy'], 'lineage_seed': s['lineage_seed'], 'turn': s['selected']['turn']})
report = {'scope': __doc__, 'plan': {k: plan[k] for k in ('completion_cap', 'system_prompt_sha256', 'changes_from_original')},
          'rows': rows, 'unique_calls': len(unique),
          'unique_prompt_tokens': sum(a['prompt_tokens'] for a in unique.values()),
          'unique_completion_tokens': sum(a['completion_tokens'] for a in unique.values()),
          'length_limited_unique_calls': sum(a['finish_reason'] == 'length' for a in unique.values()),
          'completion_tokens_by_arm': {arm: sorted(a['completion_tokens'] for k, a in unique.items() if k[0] == arm)
                                       for arm in ('base', 'adapter')},
          'selected_sources': len(selected)}
args.out.write_text(json.dumps(report, indent=1))
(args.out.parent/'matched-selected-sources.json').write_text(json.dumps(selected, indent=1))
for r in rows:
    print(f"{r['arm']:8s} {r['policy']:9s} budget {r['budget']}: {r['passed_components']}/{r['lineages']}  calls {r['observed_calls']:3d}  "
          f"tokens {r['prompt_tokens']+r['completion_tokens']:7d}  length-limited {r['length_limited_calls']}  {r['attempt_outcomes']}")
print('unique calls', report['unique_calls'], 'tokens', report['unique_prompt_tokens']+report['unique_completion_tokens'],
      'length-limited', report['length_limited_unique_calls'], 'selected sources', report['selected_sources'])
