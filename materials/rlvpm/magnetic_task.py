"""Specified nonlinear magnetic free energy: component synthesis and variational checks.

This is a mathematical adaptation task. It does not establish that the constitutive
law, applied-field ensemble, or coefficients describe a particular material.
"""
from __future__ import annotations

import numpy as np

SYSTEM = '''Write one complete NumPy Python module for the requested physical term.
Return only code, with no Markdown, drafts, tests, files, network, or subprocesses.
Use FFTs on the last two axes only. Preserve the requested function signature.
Derive the chemical potential from the specified energy; no time integrator is needed.
'''
STATEMENT = '''We are composing a phase-field solver with a verified integration driver.
Implement the magnetic contribution to chemical potential for this exact energy.
For each periodic 2D composition field c(x,y), the magnetic free energy per grid cell is

    F_mag[c] = beta/2 * mean(g(c,T) * A[g(c,T)]) - h * mean(g(c,T)).

The linear, self-adjoint operator A has Fourier multiplier kx**2/(kx**2+ky**2),
set to zero at the zero wavevector. Use kx=2*pi*fftfreq(Nx,dx),
ky=2*pi*fftfreq(Ny,dy), and the full FFT on spatial axes only.

The specified regularized magnetization law is
    Tc(c) = Tc_a + (Tc_b - Tc_a)*c
    q(c,T) = 1 - T/Tc(c)
    s(q) = (q + sqrt(q*q + width*width))/2
    g(c,T) = (1-c)*sqrt(s(q(c,T))).

T, beta, h, Tc_a, Tc_b and width are scalar constants during a call. beta >= 0,
h >= 0, width > 0, and Tc(c) > 0. They do not depend on c. The term with h is
the specified local applied-field energy. Do not add chemical, gradient, elastic,
or stochastic terms; those are owned by other components of the solver.

Write:
    def magnetic_mu(c, T, beta, h, Tc_a, Tc_b, width, dx, dy):
        ...

c is a finite float64 NumPy array of shape [B,Nx,Ny], with 0<c<1. B, Nx, Ny vary
and Nx can differ from Ny. Return a finite real float64 array of exactly that shape.
Define mu by delta F_mag = mean(mu * delta c), including the complete chain rule.
For the batch, each member is a separate field, and the same function applies.
Do not replace g by a linearized magnetization or omit either energy contribution.
'''

CONTROL = '''import numpy as np
def magnetic_mu(c, T, beta, h, Tc_a, Tc_b, width, dx, dy):
    tc = Tc_a+(Tc_b-Tc_a)*c
    q = 1-T/tc
    root = np.sqrt(q*q+width*width)
    s = .5*(q+root)
    sh = np.sqrt(s)
    g = (1-c)*sh
    dq = T*(Tc_b-Tc_a)/(tc*tc)
    gp = -sh+(1-c)*(.5*(1+q/root))*dq/(2*sh)
    nx,ny = c.shape[-2:]
    kx = 2*np.pi*np.fft.fftfreq(nx,d=dx)[:,None]
    ky = 2*np.pi*np.fft.fftfreq(ny,d=dy)[None,:]
    k2 = kx*kx+ky*ky
    K = np.divide(kx*kx,k2,out=np.zeros_like(k2),where=k2>0)
    Ag = np.fft.ifft2(K*np.fft.fft2(g,axes=(-2,-1)),axes=(-2,-1)).real
    return beta*gp*Ag-h*gp
'''
WRAPPER = '''
def solver(c, T, beta, h, Tc_a, Tc_b, width, dx, dy):
    return magnetic_mu(c, T, beta, h, Tc_a, Tc_b, width, dx, dy)
'''
ARGS = ('c','T','beta','h','Tc_a','Tc_b','width','dx','dy')


def cases(additional=False):
    rng=np.random.default_rng(2026102701 if additional else 2026102601)
    specs = ([('additional_below',3,33,29,905.,.006,.0007,.008),
              ('additional_above',2,40,52,956.,.002,.002,.025),
              ('additional_mixed',3,39,31,929.,.004,.001,.015)] if additional else
             [('mixed',2,32,24,930.,.003,.0015,.01),
              ('self_energy_only',2,36,28,910.,.007,0.,.01),
              ('applied_field_only',2,34,26,935.,0.,.002,.01),
              ('uniform',2,30,22,925.,.005,.001,.01),
              ('zero_couplings',2,28,26,930.,0.,0.,.01)])
    result=[]
    for name,B,nx,ny,T,beta,h,width in specs:
        x=np.arange(nx)[:,None]/nx
        y=np.arange(ny)[None,:]/ny
        c=np.stack([.46+.035*j+.08*np.cos(2*np.pi*x+rng.uniform(0,6))
                    +.045*np.sin(4*np.pi*y+rng.uniform(0,6)) for j in range(B)])
        if name=='uniform':c[:]=.49
        scalars={'T':T,'beta':beta,'h':h,'Tc_a':1250.,'Tc_b':606.,
                 'width':width,'dx':1.3,'dy':.85}
        directions=[]
        for j in range(5):
            v=rng.normal(size=c.shape)
            # Include a mean-changing direction to test the stated derivative,
            # not just its mass-constrained equivalence class modulo a constant.
            v=v/(np.std(v) or 1)+(0.3 if j==0 else 0)
            directions.append(v)
        result.append({'name':name,'c':c,'scalars':scalars,'directions':directions})
    return result


def energy(c, p):
    tc=p['Tc_a']+(p['Tc_b']-p['Tc_a'])*c
    q=1-p['T']/tc
    g=(1-c)*np.sqrt(.5*(q+np.sqrt(q*q+p['width']**2)))
    nx,ny=c.shape[-2:]
    kx=2*np.pi*np.fft.fftfreq(nx,d=p['dx'])[:,None]
    ky=2*np.pi*np.fft.fftfreq(ny,d=p['dy'])[None,:]
    denom=kx*kx+ky*ky
    symbol=np.divide(kx*kx,denom,out=np.zeros_like(denom),where=denom>0)
    # Independently compute the self-energy by Parseval, without constructing mu.
    power=abs(np.fft.fft2(g,axes=(-2,-1)))**2/(nx*ny)**2
    return float(.5*p['beta']*(power*symbol).sum(axis=(-2,-1)).mean()-p['h']*g.mean())


def check(code, sandbox, additional=False):
    from .sandbox import Invocation

    rows=[]
    for case in cases(additional):
        c,p=case['c'],case['scalars']
        r=sandbox.run(code+WRAPPER,Invocation(ARGS,{'c':c},p,c.shape))
        row={'case':case['name'],'ok':r.ok,'error':r.error,'stderr':r.stderr_tail,'seconds':r.seconds}
        if r.ok:
            errors=[]
            consistency=[]
            for v in case['directions']:
                gradients=[]
                for eps in (1e-5,5e-6):
                    gradients.append((energy(c+eps*v,p)-energy(c-eps*v,p))/(2*eps))
                observed=float(np.mean(r.output*v))
                errors.append(abs(observed-gradients[-1])/max(1e-7,abs(gradients[-1])))
                consistency.append(abs(gradients[1]-gradients[0])/max(1e-7,abs(gradients[1])))
            row.update({'max_relative_directional_error':float(max(errors)),
                        'finite_difference_halving_change':float(max(consistency)),
                        'passed':bool(max(errors)<1e-4)})
            if case['name']=='zero_couplings':row['passed'] &= bool(abs(r.output).max()<1e-12)
            if case['name']=='uniform':row['passed'] &= bool(r.output.std(axis=(-2,-1)).max()<1e-12)
        else:row['passed']=False
        rows.append(row)
    return {'passed':all(r['passed'] for r in rows),'n_passed':sum(r['passed'] for r in rows),
            'cases':rows,'feedback':'\n'.join(str(r) for r in rows if not r['passed']) or 'All checks passed.'}
