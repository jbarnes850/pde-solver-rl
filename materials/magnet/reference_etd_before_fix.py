"""Experimental independent finite-grid stochastic phase-field reference.

Exponential Euler drift with exact frozen-linear additive-noise convolution.
This is a diagnostic control, not a validated replacement for reference.py.
See docs/materials/STOCHASTIC-REFERENCE-AUDIT.md for its verification obligations.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import torch

from .reference import Coefficients, Grid, _VecCoeffs, curie_g, wavenumbers


def curie_second(c, T, Tc_a, Tc_b, width):
    """Second composition derivative of the specified regularized Curie law."""
    d=Tc_b-Tc_a
    tc=Tc_a+d*c
    q=1-T/tc
    q1=T*d/(tc*tc)
    q2=-2*T*d*d/(tc*tc*tc)
    root=torch.sqrt(q*q+width*width)
    s=(q+root)/2
    sq=(1+q/root)/2
    sqq=width*width/(2*root**3)
    u=torch.sqrt(s)
    u1=sq*q1/(2*u)
    u2=(sqq*q1*q1+sq*q2)/(2*u)-(sq*q1)**2/(4*u**3)
    return -2*u1+(1-c)*u2


def phi1(z):
    """(exp(z)-1)/z with a continuous, cancellation-safe zero limit."""
    small = z.abs() < 1e-5
    safe = torch.where(small, torch.ones_like(z), z)
    series = 1 + z/2 + z*z/6 + z*z*z/24
    return torch.where(small, series, torch.expm1(z)/safe)


def simulate_etd(c_init: np.ndarray, grid: Grid, members: Sequence[Coefficients], *,
                 out_times=(), h_max=.25, device='cpu', dtype=torch.float64,
                 noise_seed=0, noise_groups=None, max_steps=5_000_000,
                 progress_callback=None, progress_every=1000,
                 snapshot_callback=None, cache_linear=False,
                 linear_step=None, magnetic_mu=None, failure_callback=None):
    """Fixed rate-scaled cap; schedule-only mesh; no rejection/resampling.

    Units follow Coefficients. Linearized chemical curvature is evaluated at the
    conserved mean, while the full nonlinear residual remains explicit. Output
    times are step boundaries. The default path uses no generated components.
    An optional snapshot callback observes an independent NumPy copy after each
    requested output; it cannot mutate the solver state or retained snapshot.
    Optional linear-factor caching reuses identical coefficient/step evaluations;
    it changes no integration formula, noise draw, or per-step validity check.
    Optional NumPy component callables are a CPU-sandbox composition interface.
    They replace only the specified coefficients or magnetic chemical potential;
    the numerical driver, noise, linearization and physical units stay supplied.
    A failure callback can preserve the last valid and rejected fields for an
    audit. It does not authorize continuation, clipping, or a new noise draw.
    """
    mem = list(members)
    B = len(mem)
    if not 0 < h_max or not mem:
        raise ValueError('Positive step cap and at least one member required')
    if any(m.e_el != 0 for m in mem):
        raise ValueError('ETD diagnostic does not yet support elasticity')
    if any(m.zeeman_ratio and m.curie is None for m in mem):
        raise ValueError('Zeeman term requires the specified nonlinear Curie law')
    if (linear_step is not None or magnetic_mu is not None) and str(device)!='cpu':
        raise ValueError('Generated-component interface is CPU-sandbox only')
    if magnetic_mu is not None and any(m.curie is None for m in mem):
        raise ValueError('Magnetic component composition requires every member to use the specified Curie law')
    if c_init.shape != (B,grid.nx,grid.ny):
        raise ValueError('Initial field shape does not match grid and batch')
    c = torch.as_tensor(c_init,device=device,dtype=dtype)
    if not bool(torch.isfinite(c).all()) or bool((c <= 0).any()) or bool((c >= 1).any()):
        raise ValueError('Initial composition must be finite and strictly between zero and one')
    means = c.mean(dim=(-2,-1),keepdim=True)
    any_curie=any(m.curie is not None for m in mem)
    if any_curie:
        cur=[m.curie if m.curie is not None else (1.,1.,1.,1.) for m in mem]
        Ttop,Tca,Tcb,width=[torch.tensor([v[j] for v in cur],device=device,dtype=dtype).view(B,1,1) for j in range(4)]
        cmask=torch.tensor([m.curie is not None for m in mem],device=device,dtype=dtype).view(B,1,1)
        zratio=torch.tensor([m.zeeman_ratio for m in mem],device=device,dtype=dtype).view(B,1,1)
    chat = torch.fft.rfft2(c)
    k2,K,_,_ = wavenumbers(grid,device,dtype)
    eps = torch.tensor([m.eps_th for m in mem],device=device,dtype=dtype).view(B,1,1)
    if bool((eps < 0).any()):raise ValueError('Negative thermal noise coefficient')
    any_noise = bool((eps > 0).any())
    noise_index,noise_count = None,B
    if noise_groups is not None:
        if len(noise_groups)!=B:raise ValueError('One noise-group label per member required')
        _,inverse=np.unique(np.asarray(noise_groups),return_inverse=True)
        noise_count=int(inverse.max())+1
        noise_index=torch.as_tensor(inverse,device=device)
    gen=torch.Generator(device=device)
    gen.manual_seed(int(noise_seed))
    vec=_VecCoeffs(mem)
    end=max(m.end for m in mem)
    outs=sorted(float(x) for x in out_times)
    if any(x <= 0 or x > end for x in outs):raise ValueError('Output times outside simulation interval')
    breaks=sorted({float(x) for m in mem for x in m.seg_end}|set(outs))
    snapshots,final={},[None]*B
    t,steps=0.,0
    cached_key,cached_factors=None,None
    cache_hits,cache_misses=0,0
    while t < end-1e-12*max(1.,abs(t)):
        if steps>=max_steps:raise RuntimeError(f'max_steps reached at t={t}')
        tiny=1e-12*max(1.,abs(t))
        nxt=next(x for x in breaks if x > t+tiny)
        # log-mobility is monotone on each subsegment; endpoint maxima bound it.
        bound=max(float(vec.at(t)[0].max()),float(vec.at(nxt,left=True)[0].max()))
        step=min(nxt-t,h_max/bound) if bound>0 else nxt-t
        rate,theta,b=vec.at(t+.5*step)
        coefficient_array=np.stack([rate,theta,b])
        key=(step,coefficient_array.tobytes()) if cache_linear else None
        if cache_linear and key==cached_key:
            rate,theta,b,a0,temperature,gp0,exp_z,step_phi,noise_gain=cached_factors
            cache_hits+=1
        else:
            coeff=torch.as_tensor(coefficient_array,device=device,dtype=dtype)
            rate,theta,b=(coeff[j].view(B,1,1) for j in range(3))
            a0=theta/(means*(1-means))-4
            magnetic_linear=b
            temperature,gp0=None,None
            if any_curie:
                temperature=theta*Ttop
                _,gp0=curie_g(means,temperature,Tca,Tcb,width)
                gpp0=curie_second(means,temperature,Tca,Tcb,width)
                a0=a0-b*cmask*zratio*gpp0
                magnetic_linear=b*((1-cmask)+cmask*gp0*gp0)
            lam=-rate*k2*(a0+k2+magnetic_linear*K)
            z=lam*step
            if float(z.max())>50:raise FloatingPointError('Excessive linear amplification; reduce the step cap')
            if cache_linear or linear_step is not None:
                if linear_step is None:
                    exp_z=torch.exp(z)
                    step_phi=step*phi1(z)
                    noise_gain=None
                    if any_noise:
                        amp=torch.sqrt(2*rate*eps*theta*step/(grid.dx*grid.dy))
                        noise_gain=amp*torch.sqrt(k2*phi1(2*z))
                else:
                    q=2*rate*eps*theta*k2/(grid.dx*grid.dy)
                    factors=linear_step(lam.detach().numpy().copy(),q.detach().numpy().copy(),step)
                    if len(factors)!=3:raise ValueError('Linear component must return E,A,B')
                    factors=[np.asarray(v) for v in factors]
                    if any(v.shape!=tuple(lam.shape) or not np.isrealobj(v) or not np.isfinite(v).all() for v in factors):
                        raise ValueError('Invalid linear component coefficients')
                    if (factors[2]<0).any():raise ValueError('Negative noise amplitude')
                    exp_z,step_phi,noise_gain=[torch.as_tensor(v.copy(),dtype=dtype) for v in factors]
            if cache_linear:
                cached_key=key
                cached_factors=(rate,theta,b,a0,temperature,gp0,exp_z,step_phi,noise_gain)
                cache_misses+=1
        fprime=theta*torch.log(c/(1-c))+2*(1-2*c)
        residual=-rate*k2*torch.fft.rfft2(fprime-a0*c)
        if any_curie:
            if magnetic_mu is None:
                g,gp=curie_g(c,temperature,Tca,Tcb,width)
                Ag=torch.fft.irfft2(K*torch.fft.rfft2(g),s=(grid.nx,grid.ny))
                mu=b*cmask*(gp*Ag-zratio*gp)
            else:
                parameters=torch.cat([temperature,b,b*zratio,Tca,Tcb,width],dim=1).squeeze(-1).numpy()
                c_numpy=c.detach().numpy().copy()
                values=np.empty_like(c_numpy)
                groups={}
                for i,row in enumerate(parameters):groups.setdefault(tuple(row),[]).append(i)
                for magnetic_parameters,indices in groups.items():
                    value=np.asarray(magnetic_mu(c_numpy[indices].copy(),*magnetic_parameters,grid.dx,grid.dy))
                    if value.shape!=c_numpy[indices].shape or not np.isrealobj(value) or not np.isfinite(value).all():
                        raise ValueError('Invalid magnetic component output')
                    values[indices]=value
                mu=torch.as_tensor(values,dtype=dtype)
            residual=residual-rate*k2*(torch.fft.rfft2(mu)-b*cmask*gp0*gp0*K*chat)
        newhat=(exp_z*chat+step_phi*residual if cache_linear or linear_step is not None else
                torch.exp(z)*chat+step*phi1(z)*residual)
        if any_noise:
            white=torch.randn((noise_count,grid.nx,grid.ny),generator=gen,device=device,dtype=dtype)
            if noise_index is not None:white=white[noise_index]
            if cache_linear or linear_step is not None:
                newhat=newhat+noise_gain*torch.fft.rfft2(white)
            else:
                amp=torch.sqrt(2*rate*eps*theta*step/(grid.dx*grid.dy))
                newhat=newhat+amp*torch.sqrt(k2*phi1(2*z))*torch.fft.rfft2(white)
        newc=torch.fft.irfft2(newhat,s=(grid.nx,grid.ny))
        if not bool(torch.isfinite(newc).all()) or bool((newc <= 0).any()) or bool((newc >= 1).any()):
            if failure_callback is not None:
                invalid=(~torch.isfinite(newc)|(newc<=0)|(newc>=1)).any(dim=(-2,-1))
                failure_callback({'time':t+step,'step':step,'steps':steps+1,
                    'invalid_members':torch.nonzero(invalid).reshape(-1).cpu().tolist()},
                    c.detach().cpu().numpy().copy(),newc.detach().cpu().numpy().copy())
            raise FloatingPointError(f'Invalid composition at t={t+step}; no noisy step was resampled')
        c,chat=newc,newhat
        t+=step
        if abs(t-nxt)<=1e-12*max(1.,abs(nxt)):t=nxt
        steps+=1
        for i,m in enumerate(mem):
            if final[i] is None and t>=m.end-1e-12*max(1.,m.end):
                final[i]=c[i].detach().cpu().numpy().copy()
        for o in outs:
            if o not in snapshots and abs(t-o)<=1e-12*max(1.,abs(o)):
                snapshots[o]=c.detach().cpu().numpy().copy()
                if snapshot_callback is not None:
                    snapshot_callback(o,snapshots[o].copy())
        if progress_callback is not None and steps%progress_every==0:
            progress_callback({'steps':steps,'time':t,'end_time':end,'step':step,'rejected':0,
                               'min_composition':float(c.min()),'max_composition':float(c.max())})
    result={'final':np.stack(final),'snapshots':snapshots,'steps':steps,'rejected':0}
    if cache_linear:
        result.update(linear_cache_hits=cache_hits,linear_cache_misses=cache_misses)
    return result
