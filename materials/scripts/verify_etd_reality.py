"""Verify an opt-in real-spectrum correction before any processing use.

Projection geometry, frozen-default preservation, short noisy M1/M2 composition,
and a long deterministic/noisy counterexample. No stochastic route convergence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlvpm.physics import MaterialParams
from rlvpm.reference import Coefficients, Grid
from rlvpm.reference_etd import project_rfft_reality, simulate_etd
from rlvpm.reference_etd_frozen import simulate_etd as frozen

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
ap.add_argument('--out',type=Path,required=True)
ap.add_argument('--skip-long',action='store_true')
args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1)
if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(.05)
rng=np.random.default_rng(2026113206)
receipt={'scope':__doc__,'device':args.device,'skip_long':args.skip_long,
    'hashes':{n:hashlib.sha256(Path(__file__).resolve().parents[1].joinpath('rlvpm',n).read_bytes()).hexdigest()
        for n in ['reference_etd.py','reference_etd_frozen.py']},
    'rubric':'Projection preserves irfft real fields and zero mode to 1e-12 for odd/even shapes, is idempotent, and matches rfft(irfft) to 1e-12. Default matches frozen source bitwise. Corrected short M1/M2 noisy trajectories differ by <1e-10, mass drift <1e-12. All long branches retained, including failures; no route claim.',
    'geometry':[],'short':[],'long':[]}
def save():
    (args.out/'results.json').write_text(json.dumps(receipt,indent=2))
save()
for nx,ny in [(8,8),(9,7),(8,9),(9,8),(48,48)]:
    raw=torch.as_tensor(rng.normal(size=(3,nx,ny//2+1))+1j*rng.normal(size=(3,nx,ny//2+1)),device=args.device)
    projected=project_rfft_reality(raw,ny)
    physical=torch.fft.irfft2(raw,s=(nx,ny))
    errors=dict(field=float(abs(torch.fft.irfft2(projected,s=(nx,ny))-physical).max()),
        roundtrip=float(abs(torch.fft.rfft2(physical)-projected).max()),
        idempotence=float(abs(project_rfft_reality(projected,ny)-projected).max()),
        zero_mode=float(abs(projected[:,0,0].real-raw[:,0,0].real).max()))
    assert max(errors.values())<1e-12,errors
    receipt['geometry'].append(dict(shape=[nx,ny],**errors));save()

grid=Grid(32,24,64.,48.)
initial=.5+rng.normal(0,.005,(4,32,24));initial+=.5-initial.mean(axis=(-2,-1),keepdims=True)
for case in range(5):
    curie=case>=3;varied=case in (2,4)
    members=[]
    for i in range(4):
        ends=np.array([40.,80.,120.]) if varied else np.array([120.])
        theta0=np.array([.99,1.02,.98]) if varied else np.array([.99])
        theta1=np.array([.99,.98,.98]) if varied else theta0.copy()
        b=(np.array([.003,0.,.002]) if varied else np.array([.003]))*(i%2)
        members.append(Coefficients(ends,theta0,theta1,np.zeros(len(ends)),np.zeros(len(ends)),b,
            eps_th=.002 if case else 0.,curie=(933.,1250.,620.,.01) if curie else None,
            zeeman_ratio=.5 if curie else 0.))
    kw=dict(out_times=[17.1,40.,95.],h_max=.25,device=args.device,
        dtype=torch.float64,noise_seed=2026113207,noise_groups=[0,0,1,1],cache_linear=True)
    old=frozen(initial,grid,members,**kw)
    default=simulate_etd(initial,grid,members,**kw)
    corrected=simulate_etd(initial,grid,members,enforce_real_spectrum=True,**kw)
    assert np.array_equal(old['final'],default['final'])
    for t in old['snapshots']:assert np.array_equal(old['snapshots'][t],default['snapshots'][t])
    errors=[float(abs(corrected['final']-default['final']).max())]+[
        float(abs(corrected['snapshots'][t]-default['snapshots'][t]).max()) for t in old['snapshots']]
    mass=float(abs(corrected['final'].mean(axis=(-2,-1))-.5).max())
    assert max(errors)<1e-10 and mass<1e-12
    receipt['short'].append(dict(case=case,default_bitwise=True,max_corrected_difference=max(errors),mass_drift=mass));save()

if not args.skip_long:
    grid=Grid(48,48,96.,96.)
    initial=.5+np.random.default_rng(2026113205).normal(0,.015,(1,48,48));initial+=.5-initial.mean()
    for noise in [False,True]:
        co=Coefficients(np.array([6000.]),np.array([873/933]),np.array([873/933]),
            np.zeros(1),np.zeros(1),np.zeros(1),eps_th=MaterialParams(c0=.5).eps_coef if noise else 0.)
        for corrected in [False,True]:
            start=time.monotonic();trace=[];failure=[]
            def capture(meta,before,after):failure.append(meta|{'before_min':float(before.min()),'before_max':float(before.max()),'after_min':float(after.min()),'after_max':float(after.max())})
            row=dict(noise=noise,corrected=corrected,eps_th=co.eps_th,h_max=.25)
            try:
                result=simulate_etd(initial,grid,[co],device=args.device,h_max=.25,
                    cache_linear=True,enforce_real_spectrum=corrected,audit_spectrum=True,
                    noise_seed=2026113208,progress_callback=trace.append,progress_every=2000,
                    failure_callback=capture)
            except FloatingPointError as error:row.update(failed=repr(error),failure=failure)
            else:
                row.update(steps=result['steps'],min=float(result['final'].min()),max=float(result['final'].max()),
                    mass_drift=float(abs(result['final'].mean()-.5)))
                np.save(args.out/f'long-noise{noise}-corrected{corrected}.npy',result['final'])
            row.update(wall_s=time.monotonic()-start,trace=trace)
            receipt['long'].append(row);save();print({k:v for k,v in row.items() if k!='trace'},flush=True)
receipt['controls_passed']=True
save();(args.out/'done.json').write_text('{}\n')
