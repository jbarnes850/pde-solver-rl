"""Matched rerun: fresh sampling versus verifier revision, in the RLVP training format.

Changes from the earlier-format run, all to match how the adapter was trained:
the training system prompt byte for byte, a def solver(...) contract, the training code
extractor, a 10,000-token completion cap and no instruction against drafts. New lineage
seeds. Physics, cases, check threshold, budgets and selection rule are unchanged.

Init only prepares text. Apply executes unchanged checks in CPU sandboxes.
Generation runs separately with generate.py against a local vLLM server.
No route utility, physical reference trajectory or production result is read.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'materials'), str(ROOT/'materials/scripts')]
from magnet.magnetic_task_matched import SYSTEM, SYSTEM_SHA256, STATEMENT, check, extract


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def prompt():
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': STATEMENT}]


def prepare(root, turn):
    jobs = []
    for path in sorted(root.glob('state-*.json')):
        state = json.loads(path.read_text())
        if state['selected'] is not None:
            continue
        assert len(state['attempts']) == turn-1
        policy = state['policy']
        messages = prompt()
        if policy == 'feedback' and state['attempts']:
            parent = state['attempts'][-1]
            messages += [{'role': 'assistant', 'content': parent['content']},
                         {'role': 'user', 'content': 'The verifier ran your solver. Fix it using these check results and return the complete module.\n'+parent['feedback']}]
        seed = state['lineage_seed']+(turn-1)*7919+(1_000_000 if policy == 'fresh' and turn > 1 else 0)
        jobs.append({'episode': path.stem, 'arm': state['arm'], 'lineage_seed': state['lineage_seed'],
                     'policy': policy, 'turn': turn, 'seed': seed, 'variant': 'matched_magnetic',
                     'cap': 10000, 'messages': messages})
    write(root/f'requests-turn{turn}.json', {'scope': __doc__, 'messages': [], 'jobs': jobs})
    print('Prepared turn', turn, 'calls', len(jobs), flush=True)


def init(root):
    root.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/'materials/magnet/magnetic_task.py', ROOT/'materials/magnet/magnetic_task_matched.py',
             ROOT/'materials/magnet/llm.py']
    write(root/'plan.json', {'scope': __doc__, 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'n_lineages_per_arm': 8, 'arms': ['base', 'adapter'], 'policies': ['fresh', 'feedback'],
        'call_budgets': [1, 2, 4, 8], 'max_actual_calls': 240, 'completion_cap': 10000, 'system_prompt_sha256': SYSTEM_SHA256,
        'temperature': .7, 'top_p': 1., 'enable_thinking': False,
        'base_revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        'adapter_revision': 'aa5083ca21cf11715552e40b59af9a1a5ac541f5',
        'selection': 'Earliest complete original magnetic variational pass; stop that branch. Shared first calls and checks. Otherwise abstain; faster-cooling operational fallback is separately labeled.',
        'supplied_guidance': 'Same energy and constitutive law as the original task, restated in the training format; unchanged variational-check feedback only in the feedback branch.',
        'changes_from_original': 'training system prompt; def solver contract; training extractor; 10,000-token cap; drafts allowed; new lineage seeds',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    for arm in ['base', 'adapter']:
        for i in range(8):
            write(root/f'state-{arm}-{i:02d}-shared.json', {'arm': arm, 'lineage_seed': 2026222400+1000*i,
                 'policy': 'shared', 'attempts': [], 'selected': None})
    prepare(root, 1)


def apply(root, generations, turn):
    from magnet.sandbox import GuardedSandbox
    completed = root/f'checked-turn{turn}.json'
    if completed.exists():
        receipt = json.loads(completed.read_text())
        assert receipt['generation_source_sha256'] == hashlib.sha256(generations.read_bytes()).hexdigest()
        print('Turn already checked; preserving its frozen results', turn)
        return
    expected = json.loads((root/f'requests-turn{turn}.json').read_text())['jobs']
    source = [json.loads(s) for s in generations.read_text().splitlines()]
    lookup = {g['episode']: g for g in source}
    assert len(source) == len(lookup) == len(expected)
    assert set(lookup) == {r['episode'] for r in expected}
    # Preserve complete raw output before any execution. No silent partial round.
    raw = root/f'generations-turn{turn}.jsonl'
    if raw.exists():
        assert raw.read_bytes() == generations.read_bytes()
    else:
        raw.write_bytes(generations.read_bytes())
    assert not any(g['error'] for g in source), 'Generation infrastructure failure: preserve and diagnose before advancing'
    sb = GuardedSandbox(timeout_seconds=30, cpus='1')
    sb.image = 'rlvp-solver:py312'
    for request in expected:
        g = lookup[request['episode']]
        for key in ['arm', 'seed', 'turn', 'policy', 'lineage_seed', 'cap']:
            assert g[key] == request[key], key
        assert g['messages_sha256'] == hashlib.sha256(json.dumps(request['messages'], sort_keys=True).encode()).hexdigest()
        path = root/(request['episode']+'.json')
        if turn == 1 and not path.exists():
            path = root/'shared-first'/path.name
        state = json.loads(path.read_text())
        if len(state['attempts']) == turn:
            continue  # Durable already-checked receipt, never regenerated.
        assert len(state['attempts']) == turn-1 and state['selected'] is None
        row = {k: g[k] for k in ['arm', 'seed', 'turn', 'policy', 'lineage_seed', 'prompt_tokens', 'completion_tokens',
                                  'latency_s', 'finish_reason', 'messages_sha256', 'content']}
        try:
            code = extract(g['content'])
        except (ValueError, SyntaxError) as error:
            row.update(extraction_error=str(error), feedback=str(error), passed=False, verifier_executions=0, verifier_seconds=0.)
        else:
            report = check(code, sb)
            assert not any(c.get('error') == 'exit_125' for c in report['cases']), 'Sandbox infrastructure failure'
            row.update(code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest(), check=report,
                       feedback=report['feedback'], passed=report['passed'], verifier_executions=len(report['cases']),
                       verifier_seconds=sum(c['seconds'] for c in report['cases']))
        state['attempts'].append(row)
        if row['passed']:
            state['selected'] = {'turn': turn, 'code_sha256': row['code_sha256']}
        write(path, state)
        print(request['episode'], 'pass', row['passed'], flush=True)
    if turn == 1:
        archive = root/'shared-first'
        archive.mkdir(exist_ok=True)
        for path in sorted(root.glob('state-*-shared.json')):
            state = json.loads(path.read_text())
            for policy in ['fresh', 'feedback']:
                write(root/(path.stem.replace('-shared', '-'+policy)+'.json'), state | {'policy': policy})
            path.replace(archive/path.name)
    write(root/f'checked-turn{turn}.json', {'calls': len(expected), 'generation_source_sha256': hashlib.sha256(generations.read_bytes()).hexdigest()})
    if turn < 8:
        prepare(root, turn+1)
    else:
        write(root/'terminal-selections.json', [{'state': p.name, 'selection': json.loads(p.read_text())['selected']}
                                               for p in sorted(root.glob('state-*.json'))])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['init', 'apply'])
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--generations', type=Path)
    ap.add_argument('--turn', type=int, choices=range(1, 9))
    args = ap.parse_args()
    if args.action == 'init':
        init(args.root)
    else:
        assert args.generations and args.turn
        apply(args.root, args.generations, args.turn)
