"""Reference-free deterministic checkpoint/restart test for a real-valued PDE.

Identical fixed timesteps and constant coefficients. The concentration field is
the complete declared physical state; checkpointing it must not materially alter
the final solution. This exposes hidden numerical state without giving a model
the reference trajectory or the diagnosed implementation fix.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from magnet.reference import Coefficients,Grid
from magnet.reference_etd import simulate_etd

torch.set_num_threads(1)
out=Path(sys.argv[1])  # output folder;out.mkdir(exist_ok=False)
grid=Grid(48,48,96.,96.)
initial=.5+np.random.default_rng(2026113205).normal(0,.015,(1,48,48))
initial+=.5-initial.mean()
plan={'scope':__doc__,'grid':vars(grid),'duration':4500.,'checkpoint':2250.,
    'unstable_theta':873/933,'stable_control_theta':1.02,'h_caps':[.25,.0625],
    'threshold_max_absolute_difference':1e-8,
    'rubric':'Report all raw/corrected cases. Stable-regime positive control should pass. The newly discovered failure is exploratory and not a held-out evaluation of earlier model policies. No noise, no stochastic-state omission, no changed time mesh or reference answer.',
    'limits':'One initial field and one box. A passing restart test does not establish PDE accuracy, model validity or decision benefit.'}
(out/'plan.json').write_text(json.dumps(plan,indent=2));rows=[]
def member(theta,end):
    return Coefficients(np.array([end]),np.array([theta]),np.array([theta]),np.zeros(1),np.zeros(1),np.zeros(1))
for theta,h in [(1.02,.25),(873/933,.25),(873/933,.0625)]:
    for corrected in [False,True]:
        start=time.monotonic();kw=dict(device='cpu',h_max=h,cache_linear=True,enforce_real_spectrum=corrected)
        row=dict(theta=theta,h_max=h,corrected=corrected)
        try:
            uninterrupted=simulate_etd(initial,grid,[member(theta,4500.)],out_times=[2250.],**kw)
            restarted=simulate_etd(uninterrupted['snapshots'][2250.],grid,[member(theta,2250.)],**kw)
        except FloatingPointError as error:row.update(error=repr(error),passed=False)
        else:
            delta=restarted['final']-uninterrupted['final']
            row.update(max_difference=float(abs(delta).max()),RMS_difference=float(np.sqrt(np.mean(delta**2))),
                passed=bool(abs(delta).max()<1e-8),uninterrupted_steps=uninterrupted['steps'],
                continuation_steps=restarted['steps'])
            np.savez_compressed(out/f'theta{theta}-h{h}-corrected{corrected}.npz',
                uninterrupted=uninterrupted['final'],restarted=restarted['final'])
        row['wall_s']=time.monotonic()-start;rows.append(row)
        (out/'results.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
assert all(r['passed'] for r in rows if r['corrected'] or r['theta']==1.02)
(out/'done.json').write_text('{}\n')
